from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from comparative_methods.BIOT.alignment_data import (
    BIOTPublicView,
    SUPPORTED_TASKS,
    load_config,
)
from comparative_methods.BIOT.audit_alignment_v2 import (
    DEFAULT_CONFIG,
    load_alignment_contract,
    unsupported_refed_cell,
)
from comparative_methods.audit_adapter_alignment import validate_cell


METHOD_ROOT = Path(__file__).resolve().parents[1]


def test_alignment_config_freezes_biot_only_and_truthful_refed_disposition() -> None:
    config, path = load_config(DEFAULT_CONFIG)
    assert path == (METHOD_ROOT / "configs/alignment_v2.yaml").resolve()
    assert config["method_id"] == "biot"
    assert config["mode"] == "public_audit_only"
    assert config["protected_test_default"] == "locked"
    assert tuple(task for task in config["tasks"] if config["tasks"][task]["supported"]) == (
        *SUPPORTED_TASKS,
    )
    assert config["tasks"]["refed_regression"]["supported"] is False
    assert config["tasks"]["refed_regression"]["unsupported_reason_code"]


def test_refed_unsupported_cell_satisfies_alignment_schema() -> None:
    config, _ = load_config(DEFAULT_CONFIG)
    contract = load_alignment_contract()
    cell = unsupported_refed_cell(config=config, alignment_contract=contract)
    report = validate_cell(cell, contract, source="synthetic_refed_cell")
    assert report["method_id"] == "biot"
    assert report["task_id"] == "refed_regression"
    assert report["cell_status"] == "unsupported"


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


def test_public_view_fails_closed_on_recorded_support_not_analysis_alias() -> None:
    panel = tuple(f"E{index}" for index in range(16))
    sample = _sample(panel)
    sample["valid_mask"]["eeg"][-1] = False
    inventory = _FakeInventory(dataset=_FakeDataset(sample), panel=panel)
    view = BIOTPublicView(inventory, sample_rate_hz=200.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unrecorded/padded support"):
        view[0]


def test_public_view_rejects_bad_measured_channel_without_copy_or_padding() -> None:
    panel = tuple(f"E{index}" for index in range(16))
    sample = _sample(panel)
    sample["bad_channel_mask"]["eeg"][3] = True
    inventory = _FakeInventory(dataset=_FakeDataset(sample), panel=panel)
    view = BIOTPublicView(inventory, sample_rate_hz=200.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bad measured channels"):
        view[0]
