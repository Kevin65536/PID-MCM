from __future__ import annotations

from pathlib import Path
import subprocess

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    REPO_ROOT / "comparative_methods/adapter_alignment_gate_contract_v2.yaml"
)
EXPECTED_TASKS = {
    "motor_imagery",
    "mental_arithmetic",
    "wg",
    "nback",
    "dsr",
    "visual",
    "refed_regression",
}
EXPECTED_METHODS = {
    "biot",
    "cbramod",
    "reve",
    "normwear_eeg_fnirs_adapted",
    "efrm_sync_200_10_variable_channel_v1",
    "brainfusion_nvc_csp_stacking_reimplementation",
    "sta_net_eeg_fnirs_supervised",
}


def _contract() -> dict:
    value = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_alignment_contract_has_complete_versioned_surface() -> None:
    contract = _contract()
    assert contract["schema"] == "adapter_alignment_gate_contract_v2"
    assert contract["contract_version"] == 2
    assert contract["authority"]["legacy_b1_promotes_to_v2"] is False
    assert contract["authority"]["protected_test_default"] == "locked"
    assert set(contract["task_contracts"]) == EXPECTED_TASKS
    assert set(contract["current_repository_assessment"]["method_readiness"]) == EXPECTED_METHODS
    assert contract["current_repository_assessment"]["new_public_matrix_launch_hold"] is True
    policy = contract["execution_policy"]
    assert policy["mode"] == "strict_serial_new_method_delivery_queue"
    assert policy["active_delivery_method"] == "biot"
    assert policy["max_active_new_method_implementations"] == 1
    assert policy["max_active_new_method_experiment_queues"] == 1
    assert policy["grandfathered_background_execution"][0] == {
        "method_id": "efrm_sync_200_10_variable_channel_v1",
        "scope": "already_running_frozen_lodo_v2",
        "blocks_new_method_delivery_queue": False,
        "may_be_modified_by_new_method_work": False,
    }
    assert [row["method_id"] for row in policy["ordered_queue"]] == [
        "biot",
        "cbramod",
        "reve",
        "brainfusion_nvc_csp_stacking_reimplementation",
        "normwear_eeg_fnirs_adapted",
    ]


def test_direct_profile_aligns_information_not_internal_tensorization() -> None:
    contract = _contract()
    profile = contract["alignment_profiles"]["support_matched_direct"]
    exact = set(profile["exact_equal_fields"])
    assert profile["direct_ranking_allowed"] is True
    assert {
        "sample_inventory_sha256",
        "split_fingerprint",
        "target_valid_mask",
        "observation_anchor",
        "modality_intervals_s",
        "modality_identity",
        "measured_channel_identity_set",
        "recorded_support_mask",
        "canonical_signal_branch",
    } <= exact
    assert {
        "delivered_channel_order",
        "patch_and_token_grid",
        "geometry_encoding",
        "pooling",
        "architecture",
    } <= set(profile["method_native_fields"])
    assert contract["alignment_profiles"]["native_capacity_secondary"]["direct_ranking_allowed"] is False
    assert contract["alignment_profiles"]["method_native_context_reference"]["direct_ranking_allowed"] is False


def test_primary_task_observation_budgets_are_explicit_and_synchronized() -> None:
    contract = _contract()
    for task, spec in contract["task_contracts"].items():
        observation = spec["primary_observation"]
        assert observation["anchor"] == "canonical_registry_window_start", task
        eeg_interval = observation["eeg_interval_s"]
        fnirs_interval = observation["fnirs_interval_s"]
        assert eeg_interval == fnirs_interval, task
        assert eeg_interval[0] == 0.0
        assert eeg_interval[1] > eeg_interval[0]
    assert contract["task_contracts"]["dsr"]["primary_observation"]["fnirs_interpretation"] == (
        "context_only_no_event_level_hemodynamic_response_claim"
    )
    assert contract["task_contracts"]["refed_regression"]["primary_observation"][
        "partial_support_policy"
    ].startswith("retain_masks")


def test_gate_order_and_evidence_scopes_prevent_mini_audit_promotion() -> None:
    contract = _contract()
    assert [gate["id"] for gate in contract["gates"]] == [
        "A0",
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "A7",
        "A8",
    ]
    gates = {gate["id"]: gate for gate in contract["gates"]}
    assert gates["A4"]["minimum_evidence_scope"] == "public_complete"
    assert gates["A7"]["minimum_evidence_scope"] == "public_complete"
    assert contract["evidence_scopes"]["public_mini"]["permits_full_coverage_claim"] is False
    partial = contract["current_repository_assessment"]["partial_preflight"]
    assert partial["reported_status"] == "partial_started"
    assert partial["promotes_A4_or_A7"] is False


def test_known_observation_mismatch_cannot_enter_direct_profile_silently() -> None:
    contract = _contract()
    migrations = contract["audited_migration_findings"]
    sta = migrations["sta_net_default_classification_observation"]
    efrm = migrations["efrm_default_classification_observation"]
    assert sta["eeg_interval_s"] != efrm["eeg_interval_s"]
    assert sta["fnirs_interval_s"] != efrm["fnirs_interval_s"]
    assert sta["disposition"] == "method_native_context_reference"
    assert migrations["sta_net_dsr_observation"]["fnirs_interval_s"] != (
        migrations["efrm_dsr_observation"]["fnirs_interval_s"]
    )


def test_alignment_auditor_validates_the_contract_without_touching_data() -> None:
    result = subprocess.run(
        [
            str(REPO_ROOT / ".venv/bin/python"),
            str(REPO_ROOT / "comparative_methods/audit_adapter_alignment.py"),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = yaml.safe_load(result.stdout)
    assert report["schema"] == "adapter_alignment_audit_report_v2"
    assert report["status"] == "pass"
    assert report["contract"]["gate_ids"] == [f"A{index}" for index in range(9)]
    assert report["cell_reports"] == []
    assert report["protected_test_opened"] is False
