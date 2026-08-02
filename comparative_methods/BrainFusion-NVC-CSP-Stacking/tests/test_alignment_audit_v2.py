from __future__ import annotations

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from alignment_data import load_config
from audit_alignment_v2 import (
    DEFAULT_CONFIG,
    adapter_identity,
    comparison_fields,
    load_alignment_contract,
    parse_tasks,
    unsupported_cell,
)
from comparative_methods.audit_adapter_alignment import validate_cell


def _inventory() -> SimpleNamespace:
    return SimpleNamespace(
        task="motor_imagery",
        duration_s=8.0,
        eeg_channels=("E1", "E2"),
        fnirs_locations=("L1", "L2", "L3"),
        indices=(1, 2, 3),
        sample_inventory_sha256="a" * 64,
        split_fingerprint="b" * 64,
    )


def test_comparison_fields_freeze_three_modalities_and_equal_intervals() -> None:
    contract = load_alignment_contract()
    fields = comparison_fields(
        task="motor_imagery",
        inventory=_inventory(),  # type: ignore[arg-type]
        alignment_contract=contract,
        branch_fingerprints={"branch": "c" * 64},
    )
    assert fields["modality_identity"] == ["eeg", "fnirs_hbo", "fnirs_hbr"]
    assert fields["modality_intervals_s"] == {
        "eeg": [0.0, 8.0],
        "fnirs_hbo": [0.0, 8.0],
        "fnirs_hbr": [0.0, 8.0],
    }
    assert fields["measured_channel_identity_set"] == {
        "eeg": ["E1", "E2"],
        "fnirs_hbo": ["L1", "L2", "L3"],
        "fnirs_hbr": ["L1", "L2", "L3"],
    }


def test_adapter_identity_names_every_source_deviation_and_fold_local_fit() -> None:
    config, config_path = load_config(DEFAULT_CONFIG)
    identity = adapter_identity(
        task="motor_imagery",
        inventory=_inventory(),  # type: ignore[arg-type]
        config=config,
        config_path=config_path,
    )
    assert identity["original_numeric_reproduction_claim_allowed"] is False
    assert identity["patch_and_token_grid"]["nvc_unselected_pair_count"] == 12
    assert identity["patch_and_token_grid"]["nvc_selected_pair_count"] == 12
    assert identity["trainable_parameter_boundary"] == "all_fitted_state_outer_training_only"
    assert len(identity["source_deviation"]) == 4
    assert all(len(value) == 64 for value in identity["source_file_sha256"].values())


@pytest.mark.parametrize("task", ("dsr", "refed_regression"))
def test_unsupported_cells_are_preregistered_and_schema_valid(task: str) -> None:
    config, _ = load_config(DEFAULT_CONFIG)
    contract = load_alignment_contract()
    cell = unsupported_cell(task=task, config=config, alignment_contract=contract)
    report = validate_cell(cell, contract, source=f"synthetic_{task}")
    assert report["cell_status"] == "unsupported"
    assert cell["protected_test_opened"] is False
    assert cell["gate_status"]["A7"] == "unsupported"


def test_task_parser_excludes_unsupported_cells_from_execution() -> None:
    assert parse_tasks([]) == (
        "motor_imagery",
        "mental_arithmetic",
        "wg",
        "nback",
        "visual",
    )
    assert parse_tasks(["visual"]) == ("visual",)
    with pytest.raises(ValueError, match="unknown or unsupported"):
        parse_tasks(["dsr"])
    with pytest.raises(ValueError, match="must be unique"):
        parse_tasks(["wg", "wg"])


def test_retained_full_public_evidence_is_terminal_through_a7() -> None:
    root = METHOD_ROOT / "evidence/alignment_v2"
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "implementation_review_complete_A0_A7_pass_A8_pending"
    assert summary["supported_unique_public_sample_count"] == 13_462
    assert summary["schema_audit"]["status"] == "pass"
    assert len(summary["schema_audit"]["direct_group_reports"]) == 7
    assert summary["protected_test_opened"] is False

    expected = {
        "motor_imagery": 1740,
        "mental_arithmetic": 1740,
        "wg": 1560,
        "nback": 702,
        "visual": 7720,
    }
    for task, count in expected.items():
        cell = json.loads((root / f"{task}.json").read_text(encoding="utf-8"))
        assert cell["evidence_scope"] == "public_complete"
        assert cell["cell_status"] == "pending"
        assert [cell["gate_status"][f"A{index}"] for index in range(8)] == ["pass"] * 8
        assert cell["gate_status"]["A8"] == "pending"
        assert cell["public_audit"]["unique_sample_count"] == count
        assert cell["public_audit"]["all_unique_public_samples_audited"] is True
        assert cell["public_audit"]["all_model_inputs_finite_and_nonconstant"] is True
        assert cell["public_audit"]["deterministic_nvc_replay_exact"] is True
        assert cell["public_audit"]["protected_test_opened"] is False

    for task in ("dsr", "refed_regression"):
        cell = json.loads((root / f"{task}.json").read_text(encoding="utf-8"))
        assert cell["cell_status"] == "unsupported"
        assert cell["protected_test_opened"] is False
