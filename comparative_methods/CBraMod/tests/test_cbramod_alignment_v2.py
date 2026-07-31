from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from comparative_methods.BIOT.alignment_data import load_config as load_biot_config
from comparative_methods.BIOT.audit_alignment_v2 import comparison_fields as biot_fields
from comparative_methods.CBraMod.alignment_data import (
    CBraModPublicView,
    SUPPORTED_TASKS,
    load_config,
)
from comparative_methods.CBraMod.audit_alignment_v2 import (
    DEFAULT_CONFIG,
    comparison_fields,
    load_alignment_contract,
    unsupported_refed_cell,
    write_json,
)
from comparative_methods.audit_adapter_alignment import validate_cell


METHOD_ROOT = Path(__file__).resolve().parents[1]
BIOT_CONFIG = METHOD_ROOT.parent / "BIOT/configs/alignment_v2.yaml"


def test_alignment_config_is_public_only_and_support_matched_to_biot() -> None:
    config, path = load_config(DEFAULT_CONFIG)
    biot, _ = load_biot_config(BIOT_CONFIG)
    assert path == (METHOD_ROOT / "configs/alignment_v2.yaml").resolve()
    assert config["method_id"] == "cbramod"
    assert config["mode"] == "public_audit_only"
    assert config["protected_test_default"] == "locked"
    assert config["adapter"]["pooling"] == "official_avgpooling_patch_reps"
    assert config["adapter"]["deterministic_source_declared_sample_transform"] == (
        "none_after_canonical_200hz_coordinate"
    )
    for task in SUPPORTED_TASKS:
        assert config["tasks"][task]["panel"] == biot["tasks"][task]["panel"]
        assert config["tasks"][task]["duration_s"] == biot["tasks"][task]["duration_s"]
    assert config["tasks"]["refed_regression"]["supported"] is False


def test_method_neutral_comparison_fields_match_biot_exactly() -> None:
    inventory = SimpleNamespace(
        sample_inventory_sha256="a" * 64,
        split_fingerprint="b" * 64,
        panel=("F3", "F4"),
        indices=(1, 2, 3),
        duration_s=8.0,
    )
    contract = load_alignment_contract()
    fingerprints = {"branch": "c" * 64}
    actual = comparison_fields(
        task="motor_imagery",
        inventory=inventory,
        alignment_contract=contract,
        branch_fingerprints=fingerprints,
    )
    expected = biot_fields(
        task="motor_imagery",
        inventory=inventory,
        alignment_contract=contract,
        branch_fingerprints=fingerprints,
    )
    assert actual == expected


def test_refed_unsupported_cell_satisfies_alignment_schema() -> None:
    config, _ = load_config(DEFAULT_CONFIG)
    contract = load_alignment_contract()
    cell = unsupported_refed_cell(config=config, alignment_contract=contract)
    report = validate_cell(cell, contract, source="synthetic_refed_cell")
    assert report["method_id"] == "cbramod"
    assert report["task_id"] == "refed_regression"
    assert report["cell_status"] == "unsupported"


def test_evidence_writer_refuses_protected_path(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="protected evidence path"):
        write_json(tmp_path / "protected" / "cell.json", {"status": "forbidden"})


class _FakeBase:
    def __init__(self, sample: dict[str, Any]) -> None:
        self.sample = sample
        self._record_cache: dict[str, Any] = {}

    def __getitem__(self, index: int) -> dict[str, Any]:
        assert index == 0
        return self.sample


class _FakeDataset:
    def __init__(self, sample: dict[str, Any]) -> None:
        self.base = _FakeBase(sample)
        self.indices = [0]

    def __len__(self) -> int:
        return 1

    def lightweight_metadata(self, index: int) -> dict[str, Any]:
        assert index == 0
        return {
            "join_key": "record-1",
            "event_index": 7,
            "window_offset_s": 0.0,
        }


@dataclass(frozen=True)
class _FakeInventory:
    dataset: _FakeDataset
    panel: tuple[str, ...]
    duration_s: float = 2.0


def _sample(panel: tuple[str, ...]) -> dict[str, Any]:
    return {
        "join_key": "record-1",
        "eeg": np.ones((16, 400), dtype=np.float32),
        "channel_names": {"eeg": list(panel)},
        "sample_rate_hz": {"eeg": 200.0},
        "valid_mask": {"eeg": np.ones(400, dtype=bool)},
        "analysis_valid_mask": {"eeg": np.ones(400, dtype=bool)},
        "bad_channel_mask": {"eeg": np.zeros(16, dtype=bool)},
        "channel_geometry": {
            "eeg": [
                {"channel_name": name, "position_available": True} for name in panel
            ]
        },
    }


def test_public_view_preserves_canonical_amplitude_without_divide_by_100() -> None:
    panel = tuple(f"E{index}" for index in range(16))
    inventory = _FakeInventory(dataset=_FakeDataset(_sample(panel)), panel=panel)
    view = CBraModPublicView(inventory, sample_rate_hz=200.0)  # type: ignore[arg-type]
    item = view[0]
    np.testing.assert_array_equal(item["eeg"].numpy(), np.ones((16, 400), dtype=np.float32))
    assert int(item["recorded_support_count"]) == 400


def test_public_view_rejects_padding_and_bad_measured_channels() -> None:
    panel = tuple(f"E{index}" for index in range(16))
    padded = _sample(panel)
    padded["valid_mask"]["eeg"][-1] = False
    inventory = _FakeInventory(dataset=_FakeDataset(padded), panel=panel)
    with pytest.raises(ValueError, match="unrecorded/padded support"):
        CBraModPublicView(inventory, sample_rate_hz=200.0)[0]  # type: ignore[arg-type]

    bad = _sample(panel)
    bad["bad_channel_mask"]["eeg"][3] = True
    inventory = _FakeInventory(dataset=_FakeDataset(bad), panel=panel)
    with pytest.raises(ValueError, match="bad measured channels"):
        CBraModPublicView(inventory, sample_rate_hz=200.0)[0]  # type: ignore[arg-type]
