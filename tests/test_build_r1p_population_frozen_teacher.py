from dataclasses import replace
from pathlib import Path
import copy
import json

import numpy as np
import pytest
import yaml

from experiments.scripts.build_r1p_population_frozen_teacher import (
    PopulationFrozenBundle,
    PopulationTrial,
    _json_sha256,
    _write_artifacts,
    apply_paired_driver,
    build_population_frozen_teacher,
    fit_population_bundle,
    fit_shared_driver_normalization,
    load_population_bundle,
    save_population_bundle,
    validate_population_config,
)
from experiments.evaluate_adaptive_shared_neural_ssm import EEGAdapter
from src.data.shared_driver_targets import (
    PhysiologyRawViewRegistry,
    SharedDriverTrajectorySidecar,
)
from src.inference.adaptive_neurovascular_ssm import (
    AdaptiveSSMFit,
    HemodynamicParameters,
    build_state_transition,
)


CONFIG_PATH = Path(
    "experiments/configs/physiology_semantic_tokenizer/"
    "r1p_population_frozen_teacher.yaml"
)


def _config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _trial(subject, role, event_index, seed):
    rng = np.random.default_rng(seed)
    eeg_names = tuple(f"E{index}" for index in range(1, 7))
    fnirs_names = ("A_HbO", "A_HbR")
    time = np.arange(200, dtype=np.float64) / 10.0
    fnirs = np.column_stack(
        (
            np.sin(time * 0.3) + rng.normal(scale=0.01, size=200),
            -0.5 * np.sin(time * 0.3) + rng.normal(scale=0.01, size=200),
        )
    )
    return PopulationTrial(
        condition_id="single_trial_session_01_ma",
        condition="MA",
        dataset_id="eeg_fnirs_single_trial",
        subject=subject,
        subject_key=f"eeg_fnirs_single_trial|{subject}",
        development_role=role,
        record_id="session_01",
        event_index=event_index,
        eeg=rng.normal(size=(4000, 6)),
        fnirs=fnirs,
        eeg_channel_names=eeg_names,
        fnirs_channel_names=fnirs_names,
        fnirs_roles=("HbO", "HbR"),
        eeg_positions=np.column_stack(
            (np.arange(6, dtype=np.float64), np.zeros((6, 2)))
        ),
        fnirs_positions=np.zeros((2, 3), dtype=np.float64),
        eeg_bad_channel_mask=np.zeros(6, dtype=bool),
        fnirs_bad_channel_mask=np.zeros(2, dtype=bool),
    )


def _bundle():
    params = HemodynamicParameters()
    transition = build_state_transition(params, phi=0.85, fs_hz=10.0)
    process_cov = np.eye(5, dtype=np.float64) * 1e-6
    process_cov[4, 4] = 0.2
    observation = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    fit = AdaptiveSSMFit(
        params=params,
        transition=transition,
        process_cov=process_cov,
        observation=observation,
        observation_cov=np.diag([0.5, 0.2, 0.2]),
        initial_cov=np.eye(5, dtype=np.float64),
        hbo_mean=0.0,
        hbo_std=1.0,
        hbr_mean=0.0,
        hbr_std=1.0,
        baseline_samples=0,
        phi=0.85,
        q_driver=0.2,
        q_scale=1.0,
        fnirs_noise_scale=1.0,
        hbo_gain=1.0,
        hbr_gain=1.0,
        eeg_noise=0.5,
        hbo_noise_base=0.2,
        hbr_noise_base=0.2,
        training_score=0.1,
        optimizer_success=True,
        optimizer_objective=0.1,
    )
    adapter = EEGAdapter(
        indices=np.arange(6, dtype=int),
        channel_names=tuple(f"E{index}" for index in range(1, 7)),
        feature_mean=np.zeros(6, dtype=np.float64),
        feature_std=np.ones(6, dtype=np.float64),
        pca_mean=np.zeros(6, dtype=np.float64),
        loading=np.ones(6, dtype=np.float64) / np.sqrt(6.0),
        pc_scale=1.0,
    )
    normalization = {
        "policy": "scalar_joint_train_subject_points_v1",
        "coordinate": "adaptive_state_index_4_shared_driver",
        "fit_subject_keys": [
            f"eeg_fnirs_single_trial|subject_{index:02d}"
            for index in range(1, 19)
        ],
        "fit_sample_count": 1080,
        "fit_point_count": 216000,
        "mean": 0.0,
        "scale": 1.0,
        "applied_identically_to": [
            "target_shared_driver",
            "target_eeg_only_driver",
        ],
        "validation_subjects_used": False,
        "protected_subjects_used": False,
    }
    normalization["sha256"] = _json_sha256(normalization)
    return PopulationFrozenBundle(
        adapter=adapter,
        fit=fit,
        selected_hbo_indices=np.asarray([0]),
        selected_hbr_indices=np.asarray([1]),
        selected_fnirs_channels=("A_HbO", "A_HbR"),
        anchor_id="A",
        normalization=normalization,
        fit_subject_keys=tuple(
            f"eeg_fnirs_single_trial|subject_{index:02d}"
            for index in range(1, 19)
        ),
        fit_sample_order_sha256="fit-order",
    )


def test_config_rejects_protected_subject_before_data_access():
    config = _config()
    registry = validate_population_config(config)
    assert registry["expected_train_count"] == 1080
    assert registry["expected_validation_count"] == 300

    invalid = copy.deepcopy(config)
    invalid["data"]["conditions"][0]["subjects"].append("subject_24")
    with pytest.raises(ValueError, match="exactly subjects 01-23"):
        validate_population_config(invalid)


def test_formal_builder_refuses_existing_output_before_loading_data(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        build_population_frozen_teacher(CONFIG_PATH, output)


def test_fit_and_normalization_reject_validation_rows_without_computation():
    validation = _trial("subject_19", "validation_pure_apply", 1, 19)
    with pytest.raises(ValueError, match="train_fit"):
        fit_population_bundle([validation], _config())
    result = apply_paired_driver(validation, _bundle())
    with pytest.raises(ValueError, match="train_fit"):
        fit_shared_driver_normalization([validation], [result])


def test_parameter_bundle_roundtrip_and_paired_apply_are_frozen(tmp_path):
    original = _bundle()
    bundle_root = tmp_path / "parameter_bundle"
    save_population_bundle(bundle_root, original, source={"test": True})
    reloaded = load_population_bundle(bundle_root)
    trial = _trial("subject_19", "validation_pure_apply", 1, 19)

    first = apply_paired_driver(trial, reloaded)
    perturbed = apply_paired_driver(
        replace(trial, fnirs=trial.fnirs + np.asarray([2.0, -1.0])),
        reloaded,
    )

    assert reloaded.bundle_sha256
    np.testing.assert_array_equal(first.eeg_only, perturbed.eeg_only)
    assert not np.array_equal(first.joint, perturbed.joint)
    assert first.joint.shape == first.eeg_only.shape == (200,)


def test_written_artifacts_are_readable_and_marked_non_promotable(tmp_path):
    bundle_root = tmp_path / "parameter_bundle"
    save_population_bundle(bundle_root, _bundle(), source={"test": True})
    bundle = load_population_bundle(bundle_root)
    train = _trial("subject_01", "train_fit", 1, 1)
    validation = _trial("subject_19", "validation_pure_apply", 2, 19)
    results = [
        apply_paired_driver(train, bundle),
        apply_paired_driver(validation, bundle),
    ]

    manifest = _write_artifacts(
        tmp_path,
        [train, validation],
        results,
        bundle,
        config_path=CONFIG_PATH.resolve(),
        train_audit={"sample_count": 1},
        validation_audit={"sample_count": 1},
    )

    teacher = SharedDriverTrajectorySidecar(
        tmp_path / "trajectory_targets",
        expected_scope="population_frozen",
        expected_family="adaptive_joint_full_trajectory",
    )
    view = PhysiologyRawViewRegistry(tmp_path / "raw_view_registry")
    leakage = json.loads((tmp_path / "leakage_audit.json").read_text())

    assert len(teacher) == len(view) == 2
    assert manifest["promotion_eligible"] is False
    assert manifest["promotion_blocker"] == "population_frozen_teacher_panel_not_run"
    assert leakage["validation_fit_calls"] == 0
    assert leakage["protected_array_dereference_count"] == 0
    assert teacher.lookup(validation.sample_key)["target_shared_driver"].shape == (
        10,
        20,
    )
