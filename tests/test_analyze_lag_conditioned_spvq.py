import numpy as np
import pytest

from experiments.analyze_lag_conditioned_spvq import (
    LAGS,
    PROBE_TARGET_LABEL_SMOOTHING,
    _coupling,
    _paired_m1_n1_rows,
    _role,
    _smooth_probe_train_targets,
    _validate_paired_variant_exports,
)


def _one_hot(values: np.ndarray, classes: int = 16) -> np.ndarray:
    return np.eye(classes, dtype=np.float64)[values]


def _archive() -> dict[str, np.ndarray]:
    eeg_ids = np.asarray(
        [[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0], [1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]
    )
    fnirs_ids = np.asarray(
        [[1, 0, 1, 2, 3, 4], [4, 5, 4, 3, 2, 1], [7, 1, 2, 3, 4, 5], [5, 6, 5, 4, 3, 2]]
    )
    prefix = "fit_parameter__"
    return {
        prefix + "eeg_posterior": _one_hot(eeg_ids),
        prefix + "fnirs_posterior": _one_hot(fnirs_ids),
        prefix + "eeg_token_valid_mask": np.ones(eeg_ids.shape, dtype=bool),
        prefix + "fnirs_token_valid_mask": np.ones(fnirs_ids.shape, dtype=bool),
        prefix + "subject": np.asarray(["s1", "s1", "s2", "s2"]),
        prefix + "condition": np.asarray(["a", "a", "b", "b"]),
        prefix + "record_id": np.asarray(["r", "r", "r", "r"]),
        prefix + "eeg_event_time_ms": np.arange(4, dtype=np.float64) * 30_000.0,
        prefix + "fnirs_event_time_ms": np.arange(4, dtype=np.float64) * 30_000.0,
        prefix + "donor_index": np.asarray([1, 0, 3, 2]),
        prefix + "sample_id": np.asarray(["x0", "x1", "x2", "x3"]),
        prefix + "target": np.asarray([0, 0, 1, 1]),
    }


def test_role_enforces_independent_k16_and_derangement_contract():
    role = _role(_archive(), "fit_parameter")
    assert role["eeg_posterior"].shape == (4, 6, 16)
    bad = _archive()
    bad["fit_parameter__donor_index"] = np.arange(4)
    with pytest.raises(ValueError, match="identity"):
        _role(bad, "fit_parameter")
    duplicate = _archive()
    duplicate["fit_parameter__sample_id"][1] = "x0"
    with pytest.raises(ValueError, match="not unique"):
        _role(duplicate, "fit_parameter")
    fractional = _archive()
    fractional["fit_parameter__donor_index"] = np.asarray([1.9, 0.1, 3.0, 2.0])
    with pytest.raises(ValueError, match="integer vector"):
        _role(fractional, "fit_parameter")
    reused = _archive()
    reused["fit_parameter__donor_index"] = np.asarray([1, 0, 1, 2])
    with pytest.raises(ValueError, match="not a permutation"):
        _role(reused, "fit_parameter")
    overlap = _archive()
    overlap["fit_parameter__fnirs_event_time_ms"][1] = 10_000.0
    with pytest.raises(ValueError, match="overlap"):
        _role(overlap, "fit_parameter")


def test_paired_variant_analysis_rejects_cross_export_identity_drift():
    base = _archive()
    roles = {}
    for variant in ("M1", "N1"):
        archive = {
            "schema": np.asarray("lc_spvq_token_exports_v2"),
            "task_id": np.asarray("motor_imagery"),
            "seed": np.asarray(1),
            "variant": np.asarray(variant),
            "development_is_new_independent_holdout": np.asarray(False),
            "derangement_nonoverlap_verified": np.asarray(True),
            "registered_hard_negative_policy": np.asarray(
                "same_subject_condition_nonidentity_same_token_time"
            ),
        }
        variant_roles = {}
        for role_name in ("fit_parameter", "fit_selection", "development_apply"):
            for key, value in base.items():
                suffix = key.split("__", 1)[1]
                archive[f"{role_name}__{suffix}"] = value.copy()
            variant_roles[role_name] = _role(archive, role_name)
        roles[variant] = variant_roles
        if variant == "M1":
            m1 = archive
        else:
            n1 = archive
    _validate_paired_variant_exports({"M1": m1, "N1": n1}, roles)
    roles["N1"]["development_apply"]["donor_index"] = np.asarray([1, 0, 2, 3])
    with pytest.raises(ValueError, match="donor_index"):
        _validate_paired_variant_exports({"M1": m1, "N1": n1}, roles)


def test_coupling_outputs_all_registered_lags_without_crossing_windows():
    role = _role(_archive(), "fit_parameter")
    matched, deranged, residual, subjects = _coupling(role)
    assert matched.shape == (len(LAGS), 16, 16)
    assert deranged.shape == matched.shape
    assert residual.shape == matched.shape
    assert subjects.shape == (2, len(LAGS), 16, 16)
    expected_support = np.asarray([24, 20, 16, 12, 8, 4])
    np.testing.assert_allclose(matched.sum(axis=(1, 2)), expected_support)
    assert np.isfinite(residual).all()


def test_probe_train_target_smoothing_is_fixed_and_evaluation_independent():
    target = np.zeros((2, 16), dtype=float)
    target[:, 3] = 1.0
    smoothed = _smooth_probe_train_targets(target)
    np.testing.assert_allclose(smoothed.sum(axis=1), 1.0)
    assert smoothed[0, 3] == pytest.approx(
        1.0 - PROBE_TARGET_LABEL_SMOOTHING + PROBE_TARGET_LABEL_SMOOTHING / 16
    )
    assert smoothed[0, 0] == pytest.approx(PROBE_TARGET_LABEL_SMOOTHING / 16)
    np.testing.assert_array_equal(target[:, 0], 0.0)


def test_m1_n1_proper_scores_are_paired_by_subject_and_fixed_lag():
    rows = []
    for variant, gain, brier in (("M1", 0.2, 0.1), ("N1", 0.05, 0.02)):
        for role in ("fit_selection", "development_apply"):
            for lag in LAGS:
                rows.append(
                    {
                        "variant": variant,
                        "role": role,
                        "lag_tokens": lag,
                        "subject": "s1",
                        "log_loss_gain_nats": gain + 0.01 * lag,
                        "brier_gain": brier,
                    }
                )
    paired = _paired_m1_n1_rows(
        rows, bootstrap_iterations=10, bootstrap_seed=11
    )
    assert len(paired) == 2 * len(LAGS)
    assert all(
        row["m1_minus_n1_log_loss_gain_nats"] == pytest.approx(0.15)
        for row in paired
    )
    assert all(row["bootstrap_subject_count"] == 1 for row in paired)
