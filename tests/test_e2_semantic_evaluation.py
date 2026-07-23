import numpy as np

from experiments.build_adaptive_teacher_sidecar import patch_targets
from src.analysis.semantic_token_evaluation import (
    fit_grouped_ridge_probe,
    match_prototype_signatures,
    prototype_signatures,
    r2_per_coordinate,
)


def test_adaptive_sidecar_patch_targets_use_registered_coordinate_order():
    time = np.linspace(0.0, 1.0, 200)
    states = np.column_stack((2 * time, 3 * time, 4 * time, 5 * time, 6 * time))
    std = np.full_like(states, 0.1)

    eeg, eeg_uncertainty, fnirs, fnirs_uncertainty = patch_targets(states, std)

    assert eeg.shape == (10, 6)
    assert fnirs.shape == (10, 9)
    assert np.allclose(eeg[:, 0], states[:, 4].reshape(10, 20).mean(axis=1))
    assert np.allclose(eeg[:, 3], states[:, 0].reshape(10, 20).mean(axis=1))
    assert np.allclose(fnirs[:, 0], states[:, 1].reshape(10, 20).mean(axis=1))
    assert np.all(eeg_uncertainty > 0)
    assert np.all(fnirs_uncertainty > 0)


def test_grouped_frozen_probe_recovers_heldout_linear_target():
    rng = np.random.default_rng(5)
    features = rng.normal(size=(120, 4))
    target = np.column_stack((features[:, 0] - features[:, 1], 2 * features[:, 2]))
    groups = np.repeat([f"s{index}" for index in range(6)], 20)

    probe, selection = fit_grouped_ridge_probe(
        features, target, groups, alphas=(0.0, 0.1, 1.0)
    )
    score = r2_per_coordinate(target, probe.predict(features))

    assert np.all(score > 0.99)
    assert selection["group_count"] == 6


def test_prototype_signature_matching_is_invariant_to_code_permutation():
    hard = np.asarray([[0, 0, 1, 1]])
    target = np.asarray([[[1.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 2.0]]])
    valid = np.ones_like(hard, dtype=bool)
    left, counts = prototype_signatures(hard, target, valid, codebook_size=2)
    right = left[[1, 0]]
    match = match_prototype_signatures(left, right, counts, counts[[1, 0]])

    assert match["matched_count"] == 2
    assert match["mean_cosine"] > 0.99
