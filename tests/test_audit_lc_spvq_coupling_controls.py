import numpy as np

from experiments.audit_lc_spvq_coupling_controls import (
    _cross_entropy,
    _fit_private_calibrators,
    _within_group_permutation,
)


def test_private_calibrators_are_fit_only_proper_score_improvements():
    logits = np.asarray(
        [[3.0, -1.0], [2.5, -0.5], [2.0, 0.0], [1.5, 0.5]],
        dtype=np.float64,
    )
    target = np.asarray([0, 1, 0, 1], dtype=np.int64)
    calibration = _fit_private_calibrators(logits, target)
    bias_logits = logits + calibration["bias"][None, :]
    temperature_logits = (
        logits / calibration["temperature"]
        + calibration["temperature_intercept"][None, :]
    )
    assert np.isclose(np.sum(calibration["bias"]), 0.0)
    assert np.isclose(np.sum(calibration["temperature_intercept"]), 0.0)
    assert calibration["temperature"] > 0.0
    assert _cross_entropy(bias_logits, target) <= _cross_entropy(logits, target) + 1e-9
    assert _cross_entropy(temperature_logits, target) <= _cross_entropy(logits, target) + 1e-9


def test_shuffle_is_a_within_subject_condition_derangement():
    subjects = np.repeat(np.asarray(["s1", "s2"]), 6)
    conditions = np.tile(np.repeat(np.asarray(["a", "b"]), 3), 2)
    permutation = _within_group_permutation(subjects, conditions, seed=7)
    np.testing.assert_array_equal(subjects[permutation], subjects)
    np.testing.assert_array_equal(conditions[permutation], conditions)
    assert np.all(permutation != np.arange(len(permutation)))
    np.testing.assert_array_equal(np.sort(permutation), np.arange(len(permutation)))


def test_cross_entropy_is_invariant_to_rowwise_logit_offsets():
    logits = np.asarray([[1.0, -1.0], [-0.2, 0.4], [0.5, 0.5]])
    target = np.asarray([0, 1, 1])
    offsets = np.asarray([[100.0], [-33.0], [0.7]])
    assert np.isclose(
        _cross_entropy(logits, target),
        _cross_entropy(logits + offsets, target),
    )
