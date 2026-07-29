from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.scripts.build_r1p_train_only_perturbation_bundles import (
    DEFAULT_CONFIG,
    DEFAULT_PREVALIDATION_SEAL,
    DEFAULT_REGISTRY,
    assert_requested_subjects,
    build_one_perturbation,
    definition_source,
    fit_perturbation_normalization,
    load_perturbation_registry,
    load_prevalidation_seal,
    reverify_builder_seal_before_validation,
    registry_audit,
    select_definition,
    validate_prevalidation_seal_state,
    validate_fit_trials,
    verify_builder_sealed_inputs,
    write_perturbation_artifacts,
)


@dataclass(frozen=True)
class TrialStub:
    subject: str
    subject_key: str
    development_role: str
    record_id: str
    condition: str
    event_index: int

    @property
    def sample_key(self) -> str:
        return (
            f"eeg_fnirs_single_trial|{self.subject}|{self.record_id}|"
            f"{self.event_index}"
        )


def _fit_trials(definition):
    trials = []
    event = 0
    for subject in definition["retained_fit_subjects"]:
        for session in ("session_01", "session_03", "session_05"):
            for condition in ("BL", "MA"):
                for _ in range(10):
                    trials.append(
                        TrialStub(
                            subject=subject,
                            subject_key=f"eeg_fnirs_single_trial|{subject}",
                            development_role="train_fit",
                            record_id=session,
                            condition=condition,
                            event_index=event,
                        )
                    )
                    event += 1
    return trials


def test_runtime_registry_contract_has_three_exact_15_subject_definitions():
    registry, registry_sha = load_perturbation_registry(DEFAULT_REGISTRY)
    assert len(registry["perturbations"]) == 3
    assert len(registry_sha) == 64
    for definition in registry["perturbations"]:
        assert len(definition["retained_fit_subjects"]) == 15
        assert len(definition["excluded_fit_subjects"]) == 3
        assert definition["expected_fit_windows"] == 900
        audit = validate_fit_trials(_fit_trials(definition), definition)
        assert audit["fit_sample_count"] == 900
        assert audit["anchor_fit_sample_count"] == 150


def test_fit_contract_rejects_cohort_and_anchor_drift():
    registry, _ = load_perturbation_registry(DEFAULT_REGISTRY)
    definition = select_definition(
        registry, "P1_TRAIN_STRESS_DROP01_03_ANCHOR_S01"
    )
    trials = _fit_trials(definition)
    invalid = list(trials)
    invalid[0] = TrialStub(
        subject="subject_24",
        subject_key="eeg_fnirs_single_trial|subject_24",
        development_role="train_fit",
        record_id="session_01",
        condition="MA",
        event_index=0,
    )
    with pytest.raises(ValueError, match="cohort"):
        validate_fit_trials(invalid, definition)

    wrong_anchor = dict(definition)
    wrong_anchor["anchor_rows"] = {
        "condition": "MA",
        "sessions": ["session_99"],
        "expected_windows": 150,
    }
    with pytest.raises(ValueError, match="anchor"):
        validate_fit_trials(trials, wrong_anchor)


def test_own_scalar_gauge_uses_only_registered_900_joint_trajectories():
    registry, _ = load_perturbation_registry(DEFAULT_REGISTRY)
    definition = registry["perturbations"][1]
    trials = _fit_trials(definition)
    results = [
        SimpleNamespace(joint=np.linspace(index, index + 1, 200))
        for index in range(900)
    ]
    normalization = fit_perturbation_normalization(
        trials, results, definition
    )
    assert normalization["fit_sample_count"] == 900
    assert normalization["fit_point_count"] == 180000
    assert len(normalization["fit_subject_keys"]) == 15
    assert normalization["validation_subjects_used"] is False
    assert normalization["protected_subjects_used"] is False
    assert normalization["applied_identically_to"] == [
        "target_shared_driver",
        "target_eeg_only_driver",
    ]
    assert normalization["scale"] > 0
    assert len(normalization["sha256"]) == 64


def test_protected_subjects_fail_before_loader_access():
    with pytest.raises(PermissionError, match="before measured-array access"):
        assert_requested_subjects(["subject_24"], role="train_fit_universe")


def test_artifact_writer_rejects_missing_pure_apply_rows_before_serialization(
    tmp_path,
):
    registry, _ = load_perturbation_registry(DEFAULT_REGISTRY)
    definition = registry["perturbations"][0]
    trials = _fit_trials(definition)
    with pytest.raises(ValueError, match="validation row count is not 300"):
        write_perturbation_artifacts(
            tmp_path,
            trials,
            [object()] * len(trials),
            None,
            source={},
            definition=definition,
            train_audit={},
            validation_audit={},
        )


def test_source_binds_exact_runtime_registry_and_definition_hashes():
    registry, registry_sha = load_perturbation_registry(DEFAULT_REGISTRY)
    definition = registry["perturbations"][2]
    _, seal_sha = load_prevalidation_seal(DEFAULT_PREVALIDATION_SEAL)
    source = definition_source(
        config_path=DEFAULT_CONFIG,
        registry_path=DEFAULT_REGISTRY,
        registry_sha256=registry_sha,
        definition=definition,
        prevalidation_seal_path=DEFAULT_PREVALIDATION_SEAL,
        prevalidation_seal_sha256=seal_sha,
    )
    assert source["perturbation_registry_sha256"] == registry_sha
    assert source["perturbation_id"] == definition["perturbation_id"]
    assert source["anchor_fit_sessions"] == definition["anchor_rows"]["sessions"]
    assert len(source["perturbation_definition_sha256"]) == 64
    assert source["prevalidation_seal_sha256"] == seal_sha
    assert source["input_hashes"]["prevalidation_seal"] == seal_sha
    assert "builder" in source["input_hashes"]
    assert "adaptive_solver" in source["input_hashes"]
    assert "clean_cache_manifest" in source["input_hashes"]
    assert "event_manifest" in source["input_hashes"]
    assert source["raw_source_provenance"]["eeg_signal_branch"] == (
        "single_trial_eeg_artifact_clean_v4"
    )


def test_existing_output_is_rejected_before_any_data_loader(monkeypatch, tmp_path):
    registry, _ = load_perturbation_registry(DEFAULT_REGISTRY)
    definition = registry["perturbations"][0]
    (tmp_path / definition["output_name"]).mkdir()
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("data loader must not run")

    monkeypatch.setattr(
        "experiments.scripts.build_r1p_train_only_perturbation_bundles."
        "load_perturbation_trials",
        forbidden,
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        build_one_perturbation(
            config_path=DEFAULT_CONFIG,
            registry_path=DEFAULT_REGISTRY,
            perturbation_id=definition["perturbation_id"],
            output_parent=tmp_path,
        )
    assert called is False


def test_registry_audit_does_not_build_or_open_validation():
    audit = registry_audit(DEFAULT_CONFIG, DEFAULT_REGISTRY)
    assert audit["protected_open"] is False
    assert len(audit["perturbations"]) == 3
    assert all(row["fit_sample_count"] == 900 for row in audit["perturbations"])
    assert audit["validation_subjects"] == [
        "subject_19",
        "subject_20",
        "subject_21",
        "subject_22",
        "subject_23",
    ]


def test_builder_rejects_alternate_seal_path(tmp_path):
    alternate = tmp_path / "seal.json"
    alternate.write_text(
        DEFAULT_PREVALIDATION_SEAL.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="tracked default"):
        load_prevalidation_seal(alternate)


def test_builder_rejects_mechanical_amendment_scope_drift():
    seal, _ = load_prevalidation_seal(DEFAULT_PREVALIDATION_SEAL)
    changed = copy.deepcopy(seal)
    changed["mechanical_amendment"]["gate_changed"] = True
    with pytest.raises(RuntimeError, match="mechanical amendment scope"):
        validate_prevalidation_seal_state(changed)


def test_builder_rejects_cli_config_not_bound_by_seal(tmp_path):
    seal, _ = load_prevalidation_seal(DEFAULT_PREVALIDATION_SEAL)
    changed = tmp_path / "changed.yaml"
    changed.write_text(
        DEFAULT_CONFIG.read_text(encoding="utf-8") + "\n# changed\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="teacher_config"):
        verify_builder_sealed_inputs(
            seal,
            config_path=changed,
            registry_path=DEFAULT_REGISTRY,
        )


def test_builder_toctou_guard_fails_before_validation_on_seal_hash_mismatch():
    seal, seal_sha = load_prevalidation_seal(DEFAULT_PREVALIDATION_SEAL)
    checks = verify_builder_sealed_inputs(
        seal,
        config_path=DEFAULT_CONFIG,
        registry_path=DEFAULT_REGISTRY,
    )
    with pytest.raises(RuntimeError, match="changed before validation load"):
        reverify_builder_seal_before_validation(
            prevalidation_seal_path=DEFAULT_PREVALIDATION_SEAL,
            expected_prevalidation_seal_sha256="0" * len(seal_sha),
            expected_input_checks=checks,
            config_path=DEFAULT_CONFIG,
            registry_path=DEFAULT_REGISTRY,
        )
