from __future__ import annotations

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


METHOD_ROOT = Path(__file__).resolve().parents[1]
for import_path in (METHOD_ROOT, METHOD_ROOT / "adapters"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from alignment_data import (
    BrainFusionPublicView,
    PublicInventory,
    SUPPORTED_TASKS,
    load_config,
    load_public_inventory,
)
from brainfusion_gpu.nvc import NVCConfig


CONFIG = METHOD_ROOT / "configs/alignment_v2.yaml"


def test_config_freezes_support_matched_observation_budget_and_real_channels() -> None:
    config, path = load_config(CONFIG)
    assert path == CONFIG.resolve()
    assert config["method_id"] == "brainfusion_nvc_csp_stacking_reimplementation"
    assert config["protected_test_default"] == "locked"
    assert config["observation_budget"]["alignment_profile"] == "support_matched_direct"
    assert config["observation_budget"]["extra_pre_anchor_context_s"] == 0.0
    assert config["observation_budget"]["extra_post_interval_context_s"] == 0.0
    assert config["data"]["required_modalities"] == ["eeg", "fnirs_hbo", "fnirs_hbr"]
    assert tuple(task for task in config["tasks"] if config["tasks"][task]["supported"]) == (
        *SUPPORTED_TASKS,
    )
    assert config["tasks"]["dsr"]["unsupported_reason_code"] == (
        "BRAINFUSION_NVC_TWO_SECOND_CONTEXT_UNSUPPORTED"
    )
    assert config["tasks"]["refed_regression"]["supported"] is False
    assert NVCConfig().hrf_oversampling == 1


def test_motor_imagery_public_inventory_uses_all_public_identities_without_protected_data() -> None:
    config, _ = load_config(CONFIG)
    inventory = load_public_inventory(config, task="motor_imagery")
    assert len(inventory.indices) == 1740
    assert len(inventory.sample_ids) == len(set(inventory.sample_ids))
    assert len(inventory.split_rows) == 5
    assert len(inventory.eeg_channels) == 30
    assert len(inventory.fnirs_locations) == 36
    assert all("protected" not in " ".join(row).lower() for row in inventory.sample_ids)


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


def _sample() -> dict[str, Any]:
    return {
        "join_key": "record-1",
        "eeg": np.arange(800, dtype=np.float32).reshape(2, 400),
        "fnirs": np.arange(80, dtype=np.float32).reshape(4, 20),
        "channel_names": {
            "eeg": ["E1", "E2"],
            "fnirs": ["L1_HbO", "L1_HbR", "L2_HbO", "L2_HbR"],
        },
        "component_roles": {"fnirs": ["HbO", "HbR", "HbO", "HbR"]},
        "sample_rate_hz": {"eeg": 200.0, "fnirs": 10.0},
        "valid_mask": {
            "eeg": np.ones(400, dtype=bool),
            "fnirs": np.ones(20, dtype=bool),
        },
        "analysis_valid_mask": {
            "eeg": np.ones(400, dtype=bool),
            "fnirs": np.ones(20, dtype=bool),
        },
        "bad_channel_mask": {
            "eeg": np.zeros(2, dtype=bool),
            "fnirs": np.zeros(4, dtype=bool),
        },
    }


def _inventory(sample: dict[str, Any]) -> PublicInventory:
    dataset = _FakeDataset(sample)
    return PublicInventory(
        task="synthetic",
        dataset=dataset,  # type: ignore[arg-type]
        indices=(0,),
        sample_ids=("record-1|event=7|offset_ms=0",),
        split_rows=(),
        duration_s=2.0,
        eeg_channels=("E2", "E1"),
        fnirs_locations=("L2", "L1"),
    )


def test_public_view_reorders_only_real_synchronized_measurements() -> None:
    sample = _sample()
    item = BrainFusionPublicView(_inventory(sample))[0]
    np.testing.assert_array_equal(item["eeg"].numpy(), sample["eeg"][[1, 0]])
    np.testing.assert_array_equal(item["hbo"].numpy(), sample["fnirs"][[2, 0]])
    np.testing.assert_array_equal(item["hbr"].numpy(), sample["fnirs"][[3, 1]])
    assert item["recorded_support_count"] == {
        "eeg": 400,
        "fnirs_hbo": 20,
        "fnirs_hbr": 20,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda sample: sample["valid_mask"]["fnirs"].__setitem__(-1, False), "padded"),
        (
            lambda sample: sample["analysis_valid_mask"]["eeg"].__setitem__(-1, False),
            "analysis-invalid",
        ),
        (lambda sample: sample["bad_channel_mask"]["fnirs"].__setitem__(2, True), "bad"),
        (lambda sample: sample["component_roles"]["fnirs"].__setitem__(3, "other"), "absent"),
    ),
)
def test_public_view_rejects_padding_invalid_support_and_missing_measurements(
    mutation: Any, message: str
) -> None:
    sample = copy.deepcopy(_sample())
    mutation(sample)
    with pytest.raises(ValueError, match=message):
        BrainFusionPublicView(_inventory(sample))[0]
