from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import comparative_methods.NormWear.audit_alignment_v2 as alignment_audit_module
import comparative_methods.NormWear.audit_data_boundary_v2 as data_audit_module
from comparative_methods.NormWear.alignment_data import (
    NormWearPublicView,
    PublicInventory,
    SUPPORTED_TASKS,
    load_config,
    load_public_inventory,
)
from comparative_methods.NormWear.audit_data_boundary_v2 import (
    load_alignment_contract,
    parse_tasks,
    unsupported_refed_cell,
    write_json,
)
from comparative_methods.NormWear.audit_alignment_v2 import (
    feature_cache_identity,
    parse_tasks as parse_model_tasks,
    write_json as write_model_json,
)
from comparative_methods.audit_adapter_alignment import validate_cell


METHOD_ROOT = Path(__file__).resolve().parents[1]
CONFIG = METHOD_ROOT / "configs/alignment_v2.yaml"
BRAINFUSION_CONFIG = METHOD_ROOT.parent / "BrainFusion-NVC-CSP-Stacking/configs/alignment_v2.yaml"
ALIGNMENT_CONTRACT = METHOD_ROOT.parent / "adapter_alignment_gate_contract_v2.yaml"


@pytest.fixture(autouse=True)
def _review_historical_normwear_under_its_frozen_active_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = yaml.safe_load(ALIGNMENT_CONTRACT.read_text(encoding="utf-8"))
    historical = copy.deepcopy(contract)
    historical["execution_policy"]["active_delivery_method"] = (
        "normwear_eeg_fnirs_adapted"
    )
    path = tmp_path / "normwear_active_contract.yaml"
    path.write_text(yaml.safe_dump(historical, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(data_audit_module, "ALIGNMENT_CONTRACT", path)
    monkeypatch.setattr(alignment_audit_module, "ALIGNMENT_CONTRACT", path)


def test_config_freezes_public_support_matched_multimodal_boundary() -> None:
    config, path = load_config(CONFIG)
    assert path == CONFIG.resolve()
    assert config["method_id"] == "normwear_eeg_fnirs_adapted"
    assert config["protected_test_default"] == "locked"
    assert config["data"]["required_modalities"] == ["eeg", "fnirs_hbo", "fnirs_hbr"]
    assert config["data"]["delivered_order"] == "eeg_then_fnirs_hbo_then_fnirs_hbr"
    assert config["observation_budget"]["extra_pre_anchor_context_s"] == 0.0
    assert config["observation_budget"]["extra_post_interval_context_s"] == 0.0
    assert tuple(task for task, cell in config["tasks"].items() if cell["supported"]) == (
        *SUPPORTED_TASKS,
    )
    assert config["tasks"]["refed_regression"]["supported"] is False


def test_channel_inventories_match_existing_multimodal_direct_profile() -> None:
    normwear = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    brainfusion = yaml.safe_load(BRAINFUSION_CONFIG.read_text(encoding="utf-8"))
    assert normwear["channel_inventories"] == brainfusion["channel_inventories"]


def test_motor_imagery_public_inventory_uses_all_public_identities() -> None:
    config, _ = load_config(CONFIG)
    inventory = load_public_inventory(config, task="motor_imagery")
    assert len(inventory.indices) == 1740
    assert len(inventory.sample_ids) == len(set(inventory.sample_ids))
    assert len(inventory.split_rows) == 5
    assert len(inventory.eeg_channels) == 30
    assert len(inventory.fnirs_locations) == 36
    assert len(inventory.delivered_channel_names) == 102
    assert all("protected" not in sample.lower() for sample in inventory.sample_ids)


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
    return PublicInventory(
        task="synthetic",
        dataset=_FakeDataset(sample),  # type: ignore[arg-type]
        indices=(0,),
        sample_ids=("record-1|event=7|offset_ms=0",),
        split_rows=(),
        duration_s=2.0,
        eeg_channels=("E2", "E1"),
        fnirs_locations=("L2", "L1"),
    )


def test_public_view_reorders_only_real_synchronized_measurements() -> None:
    sample = _sample()
    item = NormWearPublicView(_inventory(sample))[0]
    np.testing.assert_array_equal(item["eeg"].numpy(), sample["eeg"][[1, 0]])
    np.testing.assert_array_equal(item["hbo"].numpy(), sample["fnirs"][[2, 0]])
    np.testing.assert_array_equal(item["hbr"].numpy(), sample["fnirs"][[3, 1]])
    assert item["delivered_channel_names"] == (
        "eeg:E2",
        "eeg:E1",
        "fnirs_hbo:L2",
        "fnirs_hbo:L1",
        "fnirs_hbr:L2",
        "fnirs_hbr:L1",
    )
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
        NormWearPublicView(_inventory(sample))[0]


def test_refed_unsupported_cell_satisfies_alignment_schema() -> None:
    config, _ = load_config(CONFIG)
    contract = load_alignment_contract()
    cell = unsupported_refed_cell(config=config, alignment_contract=contract)
    report = validate_cell(cell, contract, source="synthetic_refed_cell")
    assert report["method_id"] == "normwear_eeg_fnirs_adapted"
    assert report["task_id"] == "refed_regression"
    assert report["cell_status"] == "unsupported"


def test_task_parser_is_serial_scope_safe() -> None:
    assert parse_tasks([]) == SUPPORTED_TASKS
    assert parse_tasks(["dsr"]) == ("dsr",)
    with pytest.raises(ValueError, match="unknown or unsupported"):
        parse_tasks(["refed_regression"])
    with pytest.raises(ValueError, match="must be unique"):
        parse_tasks(["wg", "wg"])


def test_evidence_writer_refuses_protected_path(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="protected NormWear evidence"):
        write_json(tmp_path / "protected" / "cell.json", {"status": "forbidden"})


def test_retained_full_public_data_boundary_evidence_is_complete() -> None:
    import json

    root = METHOD_ROOT / "evidence/alignment_v2"
    summary = json.loads((root / "data_boundary_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "A0_A4_pass_A5_A8_pending_protected_locked"
    assert summary["audited_unique_public_sample_count"] == 22_442
    assert summary["supported_cell_count"] == 6
    assert summary["unsupported_cell_count"] == 1
    assert summary["protected_test_opened"] is False
    for task in SUPPORTED_TASKS:
        cell = json.loads((root / f"{task}.json").read_text(encoding="utf-8"))
        assert cell["evidence_scope"] == "public_complete"
        assert cell["cell_status"] == "pass"
        assert [cell["gate_status"][f"A{index}"] for index in range(5)] == ["pass"] * 5
        assert cell["gate_status"]["A6"] == "pass"
        if "public_adapter_audit" in cell:
            assert cell["gate_status"]["A5"] == "pass"
            assert cell["gate_status"]["A7"] == "pass"
        else:
            assert cell["gate_status"]["A5"] == "pending"
            assert cell["gate_status"]["A7"] == "pending"
        assert cell["gate_status"]["A8"] == "pass"
        assert cell["public_data_audit"]["all_unique_public_samples_audited"] is True
        assert cell["public_data_audit"]["protected_test_opened"] is False


def test_retained_common_multimodal_fields_match_brainfusion_exactly() -> None:
    import json

    contract = load_alignment_contract()
    fields = contract["alignment_profiles"]["support_matched_direct"]["exact_equal_fields"]
    normwear_root = METHOD_ROOT / "evidence/alignment_v2"
    brainfusion_root = METHOD_ROOT.parent / "BrainFusion-NVC-CSP-Stacking/evidence/alignment_v2"
    for task in ("motor_imagery", "mental_arithmetic", "wg", "nback", "visual"):
        normwear = json.loads((normwear_root / f"{task}.json").read_text(encoding="utf-8"))
        brainfusion = json.loads(
            (brainfusion_root / f"{task}.json").read_text(encoding="utf-8")
        )
        for field in fields:
            assert normwear["comparison_fields"][field] == brainfusion["comparison_fields"][field]


def test_model_task_parser_and_protected_writer_are_serial_safe(tmp_path: Path) -> None:
    assert parse_model_tasks([]) == SUPPORTED_TASKS
    assert parse_model_tasks(["nback"]) == ("nback",)
    with pytest.raises(ValueError, match="unknown or unsupported"):
        parse_model_tasks(["refed_regression"])
    with pytest.raises(ValueError, match="must be unique"):
        parse_model_tasks(["wg", "wg"])
    with pytest.raises(PermissionError, match="protected NormWear evidence"):
        write_model_json(tmp_path / "protected" / "cell.json", {"status": "bad"})


def test_feature_cache_identity_covers_semantic_inputs() -> None:
    config, _ = load_config(CONFIG)
    inventory = _inventory(_sample())
    method = {"checkpoint_sha256": "a" * 64, "adapter_sha256": "b" * 64}
    identity = feature_cache_identity(inventory=inventory, method=method, config=config)
    assert identity["task"] == "synthetic"
    assert identity["sample_inventory_sha256"] == inventory.sample_inventory_sha256
    assert identity["delivered_channel_order"] == list(inventory.delivered_channel_names)
    assert identity["feature_extraction"]["feature_dimension"] == 6 * 768
    assert identity["feature_extraction"]["dtype"] == "float32"
    assert identity["feature_extraction"]["feature_batch_size"] == 2
    assert len(identity["feature_cache_key"]) == 64
    changed = feature_cache_identity(
        inventory=inventory,
        method={**method, "checkpoint_sha256": "c" * 64},
        config=config,
    )
    assert changed["feature_cache_key"] != identity["feature_cache_key"]


def test_retained_nback_production_pilot_passes_a7() -> None:
    import json

    cell = json.loads(
        (METHOD_ROOT / "evidence/alignment_v2/nback.json").read_text(encoding="utf-8")
    )
    assert [cell["gate_status"][f"A{index}"] for index in range(8)] == ["pass"] * 8
    assert cell["gate_status"]["A8"] == "pass"
    audit = cell["public_adapter_audit"]
    assert audit["unique_sample_count"] == 702
    assert audit["all_unique_public_samples_executed"] is True
    assert audit["feature_shape"] == [702, 76_800]
    assert audit["feature_dtype"] == "float32"
    assert audit["nonconstant_coordinate_count"] == 76_800
    assert audit["cache_replay_exact"] is True
    assert audit["cache_replay_batch_size"] == 2
    assert audit["protected_test_opened"] is False


def test_retained_full_public_production_replay_passes_a7() -> None:
    import json

    root = METHOD_ROOT / "evidence/alignment_v2"
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "A0_A7_pass_A8_pending_protected_locked"
    assert summary["all_supported_tasks_complete"] is True
    assert summary["schema_audit"]["status"] == "pass"
    assert len(summary["schema_audit"]["cell_reports"]) == 7
    assert summary["protected_test_opened"] is False

    expected = {
        "motor_imagery": (1740, 78_336),
        "mental_arithmetic": (1740, 78_336),
        "wg": (1560, 76_800),
        "nback": (702, 76_800),
        "dsr": (8980, 76_800),
        "visual": (7720, 59_904),
    }
    total = 0
    for task, (count, width) in expected.items():
        cell = json.loads((root / f"{task}.json").read_text(encoding="utf-8"))
        assert [cell["gate_status"][f"A{index}"] for index in range(8)] == ["pass"] * 8
        assert cell["gate_status"]["A8"] == "pass"
        audit = cell["public_adapter_audit"]
        assert audit["feature_shape"] == [count, width]
        assert audit["nonconstant_coordinate_count"] == width
        assert audit["cache_replay_exact"] is True
        assert audit["protected_test_opened"] is False
        total += count
    assert total == 22_442
    refed = json.loads((root / "refed_regression.json").read_text(encoding="utf-8"))
    assert refed["cell_status"] == "unsupported"
    assert refed["protected_test_opened"] is False
