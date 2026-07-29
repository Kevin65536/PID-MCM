import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.scripts.qualify_r1p_population_frozen_teacher import (
    DEFAULT_PERTURBATION_REGISTRY,
    DEFAULT_PREVALIDATION_SEAL,
    DEFAULT_REGISTRY,
    PanelRecord,
    freeze_calibration,
    _heldout_sse,
    _json_sha256,
    _write_json,
    load_perturbation_registry,
    load_prevalidation_seal,
    load_registry,
    load_and_verify_frozen_calibration,
    perturbation_stability_gate,
    physical_subject_metrics,
    reverify_qualification_seal_before_validation,
    _condition_templates,
    _sha256,
    shifted_solver_null_draws,
    _balanced_subject_indices,
    _fit_ridge,
    _optimized_loso_target_permutation_null,
    _stack,
    _template_cell,
    sidecar_recompute_deltas,
    save_frozen_calibration_arrays,
    validate_prevalidation_seal_state,
    verify_qualification_sealed_inputs,
)


def _records(subject_numbers, split, seed=1):
    rng = np.random.default_rng(seed)
    records = []
    time = np.linspace(-1.0, 1.0, 200)
    for subject_number in subject_numbers:
        subject = f"subject_{subject_number:02d}"
        subject_shift = rng.normal(scale=0.05)
        for condition_index, condition in enumerate(("BL", "MA")):
            for event_index in range(2):
                eeg = rng.normal(size=4)
                fnirs = rng.normal(size=3)
                phase = (0.25 + 0.15 * condition_index) * np.sin(np.pi * time)
                target = (
                    phase
                    + eeg[0] * (0.2 + 0.1 * time)
                    + eeg[1] * (0.1 - 0.05 * time)
                    + fnirs[0] * (0.15 + 0.03 * time)
                    + subject_shift
                )
                correction = 0.12 * target + 0.03 * np.cos(2 * np.pi * time)
                eeg_only_target = target - correction
                base_hbo = 0.4 * np.sin(np.pi * time)
                base_hbr = -0.2 * np.sin(np.pi * time)
                hbo_observed = base_hbo + 0.3 * correction
                hbr_observed = base_hbr - 0.2 * correction
                records.append(
                    PanelRecord(
                        sample_key=f"{subject}|{condition}|{event_index}",
                        subject=subject,
                        subject_key=f"eeg_fnirs_single_trial|{subject}",
                        split=split,
                        session="session_01",
                        condition=condition,
                        event_index=event_index,
                        rj=target,
                        rj_masked=target,
                        re=eeg_only_target,
                        hbo_observed=hbo_observed,
                        hbr_observed=hbr_observed,
                        hbo_joint=hbo_observed + rng.normal(scale=0.002, size=200),
                        hbr_joint=hbr_observed + rng.normal(scale=0.002, size=200),
                        hbo_eeg_only=base_hbo,
                        hbr_eeg_only=base_hbr,
                        eeg_features=eeg,
                        fnirs_features=fnirs,
                    )
                )
    return records


def _small_registry():
    registry = copy.deepcopy(load_registry(DEFAULT_REGISTRY))
    registry["threshold_policy"]["null_replicates"] = 8
    registry["uncertainty"]["replicates"] = 50
    return registry


def _shift_cache(records):
    rng = np.random.default_rng(123)
    eeg_level = np.full((len(records), 2), 100.0)
    eeg_difference = np.full((len(records), 2), 80.0)
    return {
        "joint_level_sse": rng.uniform(70.0, 110.0, size=(len(records), 2, 9)),
        "joint_first_difference_sse": rng.uniform(
            60.0, 90.0, size=(len(records), 2, 9)
        ),
        "eeg_level_sse": eeg_level,
        "eeg_first_difference_sse": eeg_difference,
        "shift_patches": np.arange(1, 10),
    }


def test_registries_are_frozen_and_protected_closed():
    qualification = load_registry(DEFAULT_REGISTRY)
    perturbations = load_perturbation_registry(DEFAULT_PERTURBATION_REGISTRY)

    assert qualification["threshold_policy"]["validation_may_set_thresholds"] is False
    assert qualification["input_contract"]["protected_open"] is False
    assert len(perturbations["perturbations"]) == 3
    assert all(
        len(item["retained_fit_subjects"]) == 15
        and len(item["excluded_fit_subjects"]) == 3
        for item in perturbations["perturbations"]
    )


def test_prevalidation_seal_matches_all_frozen_sources():
    seal = load_prevalidation_seal(DEFAULT_PREVALIDATION_SEAL)
    assert seal["validation_metric_disclosure"] == {
        "computed_in_memory": True,
        "serialized": False,
        "inspected_by_operator": False,
        "failure_point": "final_panel_summary_json_serialization",
    }
    assert seal["base_bundle_identity"]["parameter_bundle_sha256"]


def test_json_serialization_normalizes_nested_numpy_without_value_change(tmp_path):
    payload = {
        "gate": {
            "passed": np.bool_(True),
            "counts": np.asarray([1, 2], dtype=np.int64),
            "nested": [np.float64(0.25), {"failed": np.bool_(False)}],
        }
    }
    output = tmp_path / "nested.json"
    _write_json(output, payload)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == {
        "gate": {
            "passed": True,
            "counts": [1, 2],
            "nested": [0.25, {"failed": False}],
        }
    }
    assert _json_sha256(payload) == _json_sha256(loaded)


def test_mechanical_amendment_seal_state_fails_closed_on_disclosure_drift():
    payload = json.loads(
        DEFAULT_PREVALIDATION_SEAL.read_text(encoding="utf-8")
    )
    payload["validation_metric_disclosure"]["inspected_by_operator"] = True
    with pytest.raises(RuntimeError, match="disclosure"):
        validate_prevalidation_seal_state(payload)


def test_train_calibration_is_finite_and_declares_no_validation_use():
    train = _records(range(1, 7), "train", seed=10)
    calibration, models, diagnostics = freeze_calibration(
        train,
        _small_registry(),
        _shift_cache(train),
        strict_contract=False,
    )

    assert calibration["validation_subjects_used"] is False
    assert calibration["protected_subjects_used"] is False
    assert set(models) == {"eeg", "fnirs"}
    assert len(diagnostics["train_physical"]) == 6
    assert all(
        np.isfinite(value)
        for value in calibration["thresholds"].values()
    )
    assert (
        calibration["thresholds"]["correction_rms_ratio_lower"]
        < calibration["thresholds"]["correction_rms_ratio_upper"]
    )
    assert all(
        calibration["observability"][modality]["null_crossfit_refit"]
        and calibration["observability"][modality][
            "null_held_subject_targets_excluded_from_each_fit"
        ]
        for modality in ("eeg", "fnirs")
    )


def test_physical_train_calibration_uses_subject_excluded_templates():
    train = _records(range(1, 7), "train", seed=11)
    calibration, _, diagnostics = freeze_calibration(
        train,
        _small_registry(),
        _shift_cache(train),
        strict_contract=False,
    )
    del calibration
    subject = "subject_01"
    selected = [record for record in train if record.subject == subject]
    expected = physical_subject_metrics(
        selected,
        {
            "hbo_observed": _condition_templates(
                train, "hbo_observed", excluded_subject=subject
            ),
            "hbr_observed": _condition_templates(
                train, "hbr_observed", excluded_subject=subject
            ),
        },
    )[0]
    observed = {
        row["subject"]: row for row in diagnostics["train_physical"]
    }[subject]
    assert observed["hbo_physical_gain"] == expected["hbo_physical_gain"]
    assert observed["hbr_physical_gain"] == expected["hbr_physical_gain"]


def test_validation_values_cannot_change_frozen_thresholds():
    train = _records(range(1, 7), "train", seed=12)
    first, _, _ = freeze_calibration(
        train,
        _small_registry(),
        _shift_cache(train),
        strict_contract=False,
    )
    extreme_validation = _records(range(19, 24), "validation", seed=99)
    extreme_validation = [
        PanelRecord(
            **{
                **record.__dict__,
                "rj": record.rj * 1000.0,
                "hbo_observed": record.hbo_observed * -500.0,
            }
        )
        for record in extreme_validation
    ]
    del extreme_validation
    second, _, _ = freeze_calibration(
        train,
        _small_registry(),
        _shift_cache(train),
        strict_contract=False,
    )

    assert first["thresholds"] == second["thresholds"]
    assert first["observability"] == second["observability"]


def test_missing_perturbation_bundles_fail_closed():
    registry = load_perturbation_registry(DEFAULT_PERTURBATION_REGISTRY)
    seal = load_prevalidation_seal(DEFAULT_PREVALIDATION_SEAL)
    result = perturbation_stability_gate(
        Path("/unused/base"),
        [],
        registry,
        registry_sha256=_sha256(DEFAULT_PERTURBATION_REGISTRY),
        prevalidation_seal=seal,
        prevalidation_seal_sha256=_sha256(DEFAULT_PREVALIDATION_SEAL),
    )
    assert result["status"] == "not_evaluated"
    assert result["pass"] is False
    assert result["provided_bundle_count"] == 0


def test_shift_null_uses_nonzero_whole_patch_offsets():
    train = _records(range(1, 5), "train", seed=14)
    null = shifted_solver_null_draws(
        train,
        _shift_cache(train),
        replicates=6,
        seed=2,
    )

    assert null["shift_draws"].min() >= 0
    assert null["shift_draws"].max() <= 8
    assert len(null["level_gain_draws"]) == 6
    assert np.isfinite(null["level_gain_draws"]).all()
    assert np.isfinite(null["first_difference_gain_draws"]).all()


def test_first_difference_is_computed_within_each_heldout_patch():
    observed = np.zeros(200)
    prediction = np.zeros(200)
    # Large discontinuities exactly at patch boundaries must not contribute.
    for patch in range(10):
        prediction[patch * 20 : (patch + 1) * 20] = patch * 100.0
        observed[patch * 20 : (patch + 1) * 20] = patch * 100.0
    assert (
        _heldout_sse(
            observed,
            prediction,
            scale=1.0,
            parity=0,
            first_difference=True,
        )
        == 0.0
    )


def test_strict_calibration_rejects_incomplete_train_registry():
    train = _records(range(1, 7), "train", seed=31)
    with pytest.raises(ValueError, match="1080"):
        freeze_calibration(
            train,
            _small_registry(),
            _shift_cache(train),
            strict_contract=True,
        )


def test_subject_block_alignment_uses_canonical_within_cell_rank_not_raw_event_id():
    train = _records(range(1, 4), "train", seed=32)
    changed = [
        replace(
            record,
            event_index=record.event_index + 100,
        )
        if record.subject == "subject_02"
        else record
        for record in train
    ]
    changed = list(reversed(changed))
    indices = _balanced_subject_indices(changed)
    for subject, subject_indices in indices.items():
        observed = [
            (
                changed[index].session,
                changed[index].condition,
                changed[index].event_index,
            )
            for index in subject_indices
        ]
        assert observed == sorted(observed)
        assert len(observed) == 4


def test_subject_block_alignment_rejects_cell_count_and_time_order_drift():
    train = _records(range(1, 4), "train", seed=32)
    changed = list(train)
    index = next(
        index
        for index, record in enumerate(changed)
        if record.subject == "subject_02" and record.condition == "BL"
    )
    changed[index] = replace(changed[index], condition="MA", event_index=99)
    with pytest.raises(RuntimeError, match="within-cell-rank"):
        _balanced_subject_indices(changed)

    changed = list(train)
    index = next(
        index
        for index, record in enumerate(changed)
        if record.subject == "subject_02"
    )
    changed[index] = replace(changed[index], rj=changed[index].rj[:-1])
    with pytest.raises(RuntimeError, match="within-cell-rank"):
        _balanced_subject_indices(changed)


def test_subject_block_alignment_rejects_duplicate_event_within_cell():
    train = _records(range(1, 4), "train", seed=32)
    changed = list(train)
    indices = [
        index
        for index, record in enumerate(changed)
        if record.subject == "subject_02" and record.condition == "BL"
    ]
    changed[indices[1]] = replace(
        changed[indices[1]],
        event_index=changed[indices[0]].event_index,
    )
    with pytest.raises(RuntimeError, match="unique raw event IDs"):
        _balanced_subject_indices(changed)


def test_cached_permutation_null_matches_explicit_refits():
    records = [
        replace(record, event_index=record.event_index + 100 * subject_number)
        for subject_number in range(1, 5)
        for record in _records([subject_number], "train", seed=32 + subject_number)
    ]
    records = list(reversed(records))
    features = _stack(records, "eeg_features")
    target = _stack(records, "rj")
    alphas = (0.1, 1.0)
    replicates = 3
    seed = 4
    observed = _optimized_loso_target_permutation_null(
        features,
        target,
        records,
        alphas=alphas,
        replicates=replicates,
        seed=seed,
        batch_size=2,
    )

    subjects = sorted({record.subject for record in records})
    indices = _balanced_subject_indices(records)
    rng = np.random.default_rng(seed)
    score_cube = np.empty((replicates, len(subjects), len(alphas)))
    for held_position, held_subject in enumerate(subjects):
        training_subjects = [
            subject for subject in subjects if subject != held_subject
        ]
        permutations = np.stack(
            [rng.permutation(len(training_subjects)) for _ in range(replicates)]
        )
        training_indices = np.concatenate(
            [indices[subject] for subject in training_subjects]
        )
        held_indices = indices[held_subject]
        templates = {
            cell: np.mean(
                np.stack(
                    [
                        record.rj
                        for record in records
                        if record.subject != held_subject
                        and _template_cell(record) == cell
                    ]
                ),
                axis=0,
            )
            for cell in sorted({_template_cell(record) for record in records})
        }
        baseline = np.stack(
            [templates[_template_cell(records[index])] for index in held_indices]
        )
        denominator = np.sum(np.square(target[held_indices] - baseline))
        for alpha_position, alpha in enumerate(alphas):
            for replicate in range(replicates):
                permuted_target = np.concatenate(
                    [
                        target[indices[training_subjects[source]]]
                        for source in permutations[replicate]
                    ]
                )
                model = _fit_ridge(
                    features[training_indices],
                    permuted_target,
                    alpha,
                )
                prediction = model.predict(features[held_indices])
                score_cube[replicate, held_position, alpha_position] = (
                    1.0
                    - np.sum(np.square(target[held_indices] - prediction))
                    / denominator
                )
    expected = np.max(np.mean(score_cube, axis=1), axis=1)
    np.testing.assert_allclose(observed, expected, atol=1e-10, rtol=0)


def test_sidecar_recompute_alignment_is_strict():
    records = _records([1], "train", seed=34)
    keys = [record.sample_key for record in records]
    rj = np.stack([record.rj for record in records])
    re = np.stack([record.re for record in records])
    assert sidecar_recompute_deltas(records, keys, rj, re) == (0.0, 0.0)
    changed = rj.copy()
    changed[0, 0] += 1e-8
    max_rj, max_re = sidecar_recompute_deltas(records, keys, changed, re)
    assert max_rj > 1e-10
    assert max_re == 0.0
    with pytest.raises(RuntimeError, match="not unique"):
        sidecar_recompute_deltas(
            records[:1],
            [records[0].sample_key, records[0].sample_key],
            np.stack([records[0].rj, records[0].rj]),
            np.stack([records[0].re, records[0].re]),
        )
    with pytest.raises(RuntimeError, match="not exactly equal"):
        sidecar_recompute_deltas(
            records,
            keys + ["extra"],
            np.concatenate((rj, rj[:1])),
            np.concatenate((re, re[:1])),
        )


def test_qualification_rejects_alternate_seal_path(tmp_path):
    alternate = tmp_path / "seal.json"
    alternate.write_text(
        DEFAULT_PREVALIDATION_SEAL.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="tracked default"):
        load_prevalidation_seal(alternate)


def test_qualification_toctou_guard_fails_before_validation_on_seal_hash_mismatch():
    with pytest.raises(RuntimeError, match="changed before validation load"):
        reverify_qualification_seal_before_validation(
            prevalidation_seal_path=DEFAULT_PREVALIDATION_SEAL,
            expected_prevalidation_seal_sha256="0" * 64,
            expected_input_checks={},
            config_path=Path(
                "experiments/configs/physiology_semantic_tokenizer/"
                "r1p_population_frozen_teacher.yaml"
            ),
            bundle_root=Path(
                "data/cache/shared_driver_r1_v1/r1_p_development_v1"
            ),
            registry_path=DEFAULT_REGISTRY,
            perturbation_registry_path=DEFAULT_PERTURBATION_REGISTRY,
        )


def test_qualification_rejects_unsealed_cli_config(tmp_path):
    seal = load_prevalidation_seal(DEFAULT_PREVALIDATION_SEAL)
    changed = tmp_path / "changed.yaml"
    source = Path(
        "experiments/configs/physiology_semantic_tokenizer/"
        "r1p_population_frozen_teacher.yaml"
    )
    changed.write_text(
        source.read_text(encoding="utf-8") + "\n# changed\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="teacher_config"):
        verify_qualification_sealed_inputs(
            seal,
            config_path=changed,
            bundle_root=Path(
                "data/cache/shared_driver_r1_v1/r1_p_development_v1"
            ),
            registry_path=DEFAULT_REGISTRY,
            perturbation_registry_path=DEFAULT_PERTURBATION_REGISTRY,
        )


def test_qualification_rejects_unsealed_base_parameter_manifest(tmp_path):
    seal = load_prevalidation_seal(DEFAULT_PREVALIDATION_SEAL)
    bundle = tmp_path / "bundle" / "parameter_bundle"
    bundle.mkdir(parents=True)
    source = Path(
        "data/cache/shared_driver_r1_v1/r1_p_development_v1/"
        "parameter_bundle/manifest.json"
    )
    (bundle / "manifest.json").write_text(
        source.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="base_parameter_manifest"):
        verify_qualification_sealed_inputs(
            seal,
            config_path=Path(
                "experiments/configs/physiology_semantic_tokenizer/"
                "r1p_population_frozen_teacher.yaml"
            ),
            bundle_root=tmp_path / "bundle",
            registry_path=DEFAULT_REGISTRY,
            perturbation_registry_path=DEFAULT_PERTURBATION_REGISTRY,
        )


def test_complete_frozen_calibration_roundtrip_and_hash_fail_closed(tmp_path):
    train = _records(range(1, 19), "train", seed=35)
    calibration, models, diagnostics = freeze_calibration(
        train,
        _small_registry(),
        _shift_cache(train),
        strict_contract=False,
    )
    arrays = tmp_path / "frozen_calibration_arrays.npz"
    save_frozen_calibration_arrays(arrays, models, diagnostics)
    train_table = tmp_path / "train_diagnostics.tsv"
    train_table.write_text("subject\n", encoding="utf-8")
    calibration["calibration_arrays_file"] = arrays.name
    calibration["calibration_arrays_sha256"] = _sha256(arrays)
    calibration["train_diagnostics_file"] = train_table.name
    calibration["train_diagnostics_sha256"] = _sha256(train_table)
    threshold = tmp_path / "threshold_manifest.json"
    threshold.write_text(
        json.dumps(calibration, sort_keys=True),
        encoding="utf-8",
    )
    loaded, loaded_models = load_and_verify_frozen_calibration(threshold)
    assert loaded["thresholds"] == calibration["thresholds"]
    assert set(loaded_models) == {"eeg", "fnirs"}

    with arrays.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(RuntimeError, match="array hash"):
        load_and_verify_frozen_calibration(threshold)
