import numpy as np
import torch

from experiments.evaluate_physical_teacher_e0 import (
    _fit_ridge,
    _permutation_null,
    _synthetic_recovery,
    signal_features,
    subject_bootstrap_ci,
    subject_signflip_null,
)


def test_signal_features_are_patch_local_and_finite():
    patch = torch.randn(2, 4, 3, 20)
    features = signal_features(patch, spectral_bins=5)
    assert features.shape == (2, 4, 3 * 4 + 3 * 5)
    assert torch.isfinite(features).all()
    changed = patch.clone()
    changed[:, 3] += 100.0
    changed_features = signal_features(changed, spectral_bins=5)
    assert torch.equal(features[:, :3], changed_features[:, :3])


def test_subject_bootstrap_aggregates_by_subject():
    differences = np.asarray([1.0, 1.0, 3.0, 3.0])
    subjects = np.asarray([1, 1, 2, 2])
    observed, lower, upper = subject_bootstrap_ci(
        differences, subjects, iterations=200, rng=np.random.default_rng(1)
    )
    assert observed == 2.0
    assert lower <= observed <= upper
    threshold = subject_signflip_null(
        differences, subjects, iterations=200, quantile=0.95, rng=np.random.default_rng(2)
    )
    assert threshold <= observed


def test_ridge_gain_exceeds_permutation_reference_on_observable_state():
    rng = np.random.default_rng(2)
    train_x = rng.normal(size=(400, 8))
    val_x = rng.normal(size=(160, 8))
    weights = rng.normal(size=8)
    train_y = train_x @ weights + rng.normal(scale=0.1, size=400)
    val_y = val_x @ weights + rng.normal(scale=0.1, size=160)
    _, _, alpha, gain = _fit_ridge(train_x, train_y, val_x, val_y, [0.1, 1.0, 10.0])
    null = _permutation_null(
        train_x, train_y, val_x, val_y, alpha, 32, np.random.default_rng(3)
    )
    assert gain > np.quantile(null, 0.95)


def test_synthetic_recovery_control_has_positive_gain():
    result = _synthetic_recovery(4)
    assert result["gain"] > 0
    assert result["mse"] < result["baseline_mse"]
