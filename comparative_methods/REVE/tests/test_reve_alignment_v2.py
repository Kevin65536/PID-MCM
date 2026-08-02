from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from comparative_methods.BIOT.alignment_data import load_config as load_biot_config
from comparative_methods.BIOT.audit_alignment_v2 import comparison_fields as biot_fields
from comparative_methods.REVE.alignment_data import (
    SUPPORTED_TASKS,
    REVEPublicView,
    load_config,
)
from comparative_methods.REVE.audit_alignment_v2 import (
    DEFAULT_CONFIG,
    comparison_fields,
    load_alignment_contract,
    parse_tasks,
    unsupported_refed_cell,
    write_json,
)
from comparative_methods.audit_adapter_alignment import validate_cell


METHOD_ROOT = Path(__file__).resolve().parents[1]
BIOT_CONFIG = METHOD_ROOT.parent / "BIOT/configs/alignment_v2.yaml"


def test_alignment_config_is_public_only_support_matched_and_position_covered() -> None:
    config, path = load_config(DEFAULT_CONFIG)
    biot, _ = load_biot_config(BIOT_CONFIG)
    position_config = json.loads(
        (METHOD_ROOT / "checkpoints/reve-positions/config.json").read_text(encoding="utf-8")
    )
    official_names = set(position_config["position_names"])
    assert path == (METHOD_ROOT / "configs/alignment_v2.yaml").resolve()
    assert config["method_id"] == "reve"
    assert config["mode"] == "public_audit_only"
    assert config["protected_test_default"] == "locked"
    assert config["adapter"]["pooling"] == "frozen_pretrained_cls_query_attention_pooling"
    assert config["adapter"]["deterministic_source_declared_sample_transform"] == (
        "none_after_canonical_200hz_coordinate"
    )
    for task in SUPPORTED_TASKS:
        assert config["tasks"][task]["panel"] == biot["tasks"][task]["panel"]
        assert config["tasks"][task]["duration_s"] == biot["tasks"][task]["duration_s"]
        assert set(config["tasks"][task]["panel"]) <= official_names
    assert config["tasks"]["refed_regression"]["supported"] is False


def test_single_trial_tasks_are_confined_to_known_pretraining_overlap_track() -> None:
    config, _ = load_config(DEFAULT_CONFIG)
    for task in ("motor_imagery", "mental_arithmetic"):
        assert config["tasks"][task]["track"] == (
            "open_world_pretrained_with_target_corpus_overlap"
        )
    for task in ("wg", "nback", "dsr", "visual"):
        assert config["tasks"][task]["track"] == (
            "single_modal_eeg_official_pretrained_linear_probe"
        )


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
    assert report["method_id"] == "reve"
    assert report["task_id"] == "refed_regression"
    assert report["cell_status"] == "unsupported"
    assert cell["unsupported_reason_code"] == "REVE_NO_PARTIAL_TIME_MASK_CONTRACT"


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
        return {"join_key": "record-1", "event_index": 7, "window_offset_s": 0.0}


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


def test_public_view_preserves_canonical_amplitude_without_task_scale_factor() -> None:
    panel = tuple(f"E{index}" for index in range(16))
    inventory = _FakeInventory(dataset=_FakeDataset(_sample(panel)), panel=panel)
    view = REVEPublicView(inventory, sample_rate_hz=200.0)  # type: ignore[arg-type]
    item = view[0]
    np.testing.assert_array_equal(item["eeg"].numpy(), np.ones((16, 400), dtype=np.float32))
    assert int(item["recorded_support_count"]) == 400


def test_public_view_rejects_padding_and_bad_measured_channels() -> None:
    panel = tuple(f"E{index}" for index in range(16))
    padded = _sample(panel)
    padded["valid_mask"]["eeg"][-1] = False
    inventory = _FakeInventory(dataset=_FakeDataset(padded), panel=panel)
    with pytest.raises(ValueError, match="unrecorded/padded support"):
        REVEPublicView(inventory, sample_rate_hz=200.0)[0]  # type: ignore[arg-type]

    bad = _sample(panel)
    bad["bad_channel_mask"]["eeg"][3] = True
    inventory = _FakeInventory(dataset=_FakeDataset(bad), panel=panel)
    with pytest.raises(ValueError, match="bad measured channels"):
        REVEPublicView(inventory, sample_rate_hz=200.0)[0]  # type: ignore[arg-type]


def test_task_parser_is_serial_scope_safe() -> None:
    assert parse_tasks([]) == SUPPORTED_TASKS
    assert parse_tasks(["motor_imagery"]) == ("motor_imagery",)
    with pytest.raises(ValueError, match="unknown or unsupported"):
        parse_tasks(["refed_regression"])
    with pytest.raises(ValueError, match="must be unique"):
        parse_tasks(["wg", "wg"])


def test_retained_full_public_evidence_is_terminal_through_a7() -> None:
    root = METHOD_ROOT / "evidence/alignment_v2"
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "implementation_review_complete_A0_A7_pass_A8_pending"
    assert summary["protected_test_opened"] is False
    assert len(summary["schema_audit"]["direct_group_reports"]) == 7
    assert summary["schema_audit"]["status"] == "pass"

    total = 0
    for task in SUPPORTED_TASKS:
        cell = json.loads((root / f"{task}.json").read_text(encoding="utf-8"))
        assert cell["evidence_scope"] == "public_complete"
        assert cell["cell_status"] == "pending"
        assert [cell["gate_status"][f"A{index}"] for index in range(8)] == ["pass"] * 8
        assert cell["gate_status"]["A8"] == "pending"
        assert cell["public_audit"]["all_unique_public_samples_audited"] is True
        assert cell["public_audit"]["deterministic_replay_exact"] is True
        assert cell["public_audit"]["feature_shape"][1] == 512
        assert cell["public_audit"]["nonconstant_coordinate_count"] == 512
        assert cell["public_audit"]["protected_test_opened"] is False
        total += int(cell["public_audit"]["unique_sample_count"])
    assert total == 22_442

    refed = json.loads((root / "refed_regression.json").read_text(encoding="utf-8"))
    assert refed["cell_status"] == "unsupported"
    assert refed["unsupported_reason_code"] == "REVE_NO_PARTIAL_TIME_MASK_CONTRACT"
    assert refed["protected_test_opened"] is False
