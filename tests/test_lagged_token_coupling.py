"""Contract tests for held-out soft lagged-token coupling statistics."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.lagged_token_coupling import (
    apply_token_order,
    build_lagged_categorical_rows,
    categorical_proper_scores,
    conditional_fnirs_given_eeg,
    conditional_log_lift,
    evaluate_q0_q1,
    evaluate_q0_q1_by_subject,
    fit_q0_q1,
    fit_token_order,
    fnirs_marginal,
    matched_minus_deranged_expected_residual_log_lift,
    pair_concentration,
    positive_vs_negative_lag_specificity,
    soft_cooccurrence_tensor,
    subject_block_bootstrap,
    summarize_lagged_coupling,
    top_pair_jaccard,
    top_pair_stability,
)


def _one_hot(values: np.ndarray, n_codes: int) -> np.ndarray:
    return np.eye(n_codes, dtype=np.float64)[np.asarray(values, dtype=np.int64)]


def _known_lag_fixture(
    *,
    seed: int = 7,
    batch: int = 700,
    time: int = 12,
    lag: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Many-to-many E -> F fixture with a known positive lag."""

    rng = np.random.default_rng(seed)
    eeg_ids = rng.integers(0, 3, size=(batch, time))
    fnirs_ids = rng.integers(0, 3, size=(batch, time))
    # Each EEG state has two likely fNIRS outcomes, so this is not a one-to-one
    # lookup-table fixture.
    conditional = np.asarray(
        [[0.78, 0.22, 0.0], [0.0, 0.78, 0.22], [0.22, 0.0, 0.78]],
        dtype=np.float64,
    )
    for row in range(batch):
        for position in range(time - lag):
            source = int(eeg_ids[row, position])
            fnirs_ids[row, position + lag] = rng.choice(3, p=conditional[source])
    return _one_hot(eeg_ids, 3), _one_hot(fnirs_ids, 3)


def test_soft_cooccurrence_is_mask_aware_and_respects_positive_lag():
    eeg = _one_hot(np.asarray([[0, 1, 0]]), 2)
    fnirs = _one_hot(np.asarray([[1, 0, 1]]), 2)
    eeg_mask = np.asarray([[True, True, False]])
    fnirs_mask = np.asarray([[True, True, True]])

    counts = soft_cooccurrence_tensor(
        eeg,
        fnirs,
        lags=[1, -1],
        eeg_valid_mask=eeg_mask,
        fnirs_valid_mask=fnirs_mask,
    )
    # +1: E[0]=0 with F[1]=0 and E[1]=1 with F[2]=1.
    expected_positive = np.zeros((2, 2))
    expected_positive[0, 0] = 1
    expected_positive[1, 1] = 1
    np.testing.assert_allclose(counts[0], expected_positive)
    # -1 only uses E[1] with F[0]; E[2] is masked.
    expected_negative = np.zeros((2, 2))
    expected_negative[1, 1] = 1
    np.testing.assert_allclose(counts[1], expected_negative)
    np.testing.assert_allclose(counts.sum(axis=(1, 2)), [2, 1])


def test_known_lag_recovers_peak_and_residual_log_lift_against_derangements():
    eeg, fnirs = _known_lag_fixture()
    lags = np.arange(-3, 5)
    matched = soft_cooccurrence_tensor(eeg, fnirs, lags=lags)

    deranged = []
    rng = np.random.default_rng(12)
    for _ in range(30):
        shifted = np.empty_like(fnirs)
        for row in range(fnirs.shape[0]):
            shifted[row] = np.roll(
                fnirs[row], int(rng.integers(1, fnirs.shape[1])), axis=0
            )
        deranged.append(soft_cooccurrence_tensor(eeg, shifted, lags=lags))
    residual = matched_minus_deranged_expected_residual_log_lift(
        matched, np.asarray(deranged), alpha=0.5
    )
    peak_lag = int(lags[np.argmax(np.max(residual, axis=(1, 2)))])
    assert peak_lag == 2
    assert float(np.max(residual[lags == 2])) > 0.4


def test_independent_fixture_is_close_to_deranged_null():
    rng = np.random.default_rng(44)
    batch, time = 1000, 12
    eeg = _one_hot(rng.integers(0, 3, size=(batch, time)), 3)
    fnirs = _one_hot(rng.integers(0, 3, size=(batch, time)), 3)
    lags = np.arange(-2, 3)
    matched = soft_cooccurrence_tensor(eeg, fnirs, lags=lags)
    deranged = []
    for _ in range(24):
        shifted = np.empty_like(fnirs)
        for row in range(batch):
            shifted[row] = np.roll(
                fnirs[row], int(rng.integers(1, time)), axis=0
            )
        deranged.append(soft_cooccurrence_tensor(eeg, shifted, lags=lags))
    residual = matched_minus_deranged_expected_residual_log_lift(
        matched, np.asarray(deranged), alpha=0.5
    )
    assert float(np.mean(np.abs(residual))) < 0.08
    assert float(np.max(np.abs(residual))) < 0.35


def test_dirichlet_alpha_is_a_per_cell_pseudocount():
    counts = np.asarray([[[2.0, 0.0], [0.0, 0.0]]])
    conditional = conditional_fnirs_given_eeg(counts, alpha=0.5)
    np.testing.assert_allclose(conditional[0, 0], [2.5 / 3.0, 0.5 / 3.0])
    np.testing.assert_allclose(conditional[0, 1], [0.5, 0.5])


def test_probability_transforms_and_summaries():
    counts = np.asarray(
        [
            [[5.0, 5.0], [5.0, 5.0]],
            [[9.0, 1.0], [1.0, 9.0]],
        ]
    )
    conditional = conditional_fnirs_given_eeg(counts, alpha=1.0)
    marginal = fnirs_marginal(counts)
    np.testing.assert_allclose(conditional.sum(axis=-1), 1.0)
    np.testing.assert_allclose(marginal.sum(axis=-1), 1.0)
    lift = conditional_log_lift(conditional, marginal)
    assert lift.shape == counts.shape
    assert pair_concentration(counts, top_k=1).shape == (2,)
    specificity = positive_vs_negative_lag_specificity(
        lift, lags=[-1, 1], reduction="max"
    )
    assert specificity["positive_minus_negative"] > 0
    summary = summarize_lagged_coupling(
        counts, lags=[-1, 1], residual_log_lift=lift, top_k=1
    )
    assert summary["schema_version"]
    assert len(summary["top_pairs_per_lag"]) == 2


def test_train_only_token_order_and_top_pair_stability():
    train = np.asarray(
        [
            [9.0, 1.0, 0.0],
            [0.0, 8.0, 2.0],
            [1.0, 0.0, 7.0],
        ]
    )
    held_out = train[:, ::-1]
    order = fit_token_order(train, method="svd")
    assert sorted(order.row_order) == [0, 1, 2]
    assert sorted(order.column_order) == [0, 1, 2]
    reordered = apply_token_order(held_out, order)
    assert reordered.shape == held_out.shape

    same = np.stack([train, train + 0.01], axis=0)
    changed = np.stack([train, train[::-1]], axis=0)
    assert top_pair_jaccard(train, train, top_k=1) == 1.0
    stability = top_pair_stability(same, top_k=1)
    assert stability["mean_pairwise_jaccard"] == 1.0
    assert top_pair_stability(changed, top_k=1)["mean_pairwise_jaccard"] < 1.0


@pytest.mark.parametrize(
    "bad_eeg",
    [
        np.asarray([[[0.5, 0.4]]]),
        np.asarray([[[0.5, -0.5]]]),
        np.asarray([[[np.nan, 1.0]]]),
    ],
)
def test_probability_and_mask_validation(bad_eeg):
    fnirs = np.asarray([[[1.0, 0.0]]])
    with pytest.raises(ValueError):
        soft_cooccurrence_tensor(bad_eeg, fnirs, lags=[0])
    good = np.asarray([[[1.0, 0.0], [0.0, 1.0]]])
    with pytest.raises(ValueError):
        soft_cooccurrence_tensor(good, good, lags=[2])
    with pytest.raises(ValueError):
        soft_cooccurrence_tensor(
            good, good, lags=[0], eeg_valid_mask=np.asarray([[1, 2]])
        )


def test_boundary_safe_q0_q1_rows_and_subject_equal_evaluation():
    eeg_ids = np.asarray([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]])
    fnirs_ids = np.asarray([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]])
    rows = build_lagged_categorical_rows(
        _one_hot(eeg_ids, 2),
        _one_hot(fnirs_ids, 2),
        lag=1,
        subject_ids=("s1", "s2"),
        condition_ids=np.asarray([0, 1]),
        condition_count=2,
        fnirs_history_steps=1,
    )

    assert len(rows.subject) == 8
    assert rows.q0_design.shape == (8, 2 + 5 + 2)
    assert np.all(rows.target_time == rows.source_time + 1)
    assert set(rows.window_index.tolist()) == {0, 1}
    models = fit_q0_q1(
        rows.fnirs_target,
        q0_design_train=rows.q0_design,
        eeg_posterior_train=rows.eeg_posterior,
        n_classes=2,
        max_iter=100,
    )
    summary = evaluate_q0_q1_by_subject(models, rows)
    assert summary["subject_count"] == 2
    assert {row["subject"] for row in summary["subject_rows"]} == {"s1", "s2"}
    assert all(row["row_count"] == 4 for row in summary["subject_rows"])


def test_boundary_safe_rows_drop_invalid_source_target_and_history():
    eeg = _one_hot(np.asarray([[0, 1, 0, 1]]), 2)
    fnirs = _one_hot(np.asarray([[1, 0, 1, 0]]), 2)
    eeg_mask = np.asarray([[True, True, False, True]])
    fnirs_mask = np.asarray([[True, False, True, True]])
    rows = build_lagged_categorical_rows(
        eeg,
        fnirs,
        lag=0,
        subject_ids=("s",),
        condition_ids=np.asarray([0]),
        condition_count=1,
        eeg_valid_mask=eeg_mask,
        fnirs_valid_mask=fnirs_mask,
        fnirs_history_steps=1,
    )
    # t=3 is the only row with valid EEG target, fNIRS target, and fNIRS t-1.
    np.testing.assert_array_equal(rows.source_time, [3])
    np.testing.assert_array_equal(rows.target_time, [3])


def test_q0_q1_fit_eval_separation_and_proper_score_increment():
    rng = np.random.default_rng(101)
    n_train, n_eval = 500, 250
    controls_train = rng.normal(size=(n_train, 2))
    controls_eval = rng.normal(size=(n_eval, 2))
    eeg_train_ids = rng.integers(0, 3, size=n_train)
    eeg_eval_ids = rng.integers(0, 3, size=n_eval)
    eeg_train = _one_hot(eeg_train_ids, 3)
    eeg_eval = _one_hot(eeg_eval_ids, 3)
    # The target is carried by EEG; controls contain only nuisance variation.
    train_targets = eeg_train_ids.copy()
    eval_targets = eeg_eval_ids.copy()
    models = fit_q0_q1(
        train_targets,
        q0_design_train=controls_train,
        eeg_posterior_train=eeg_train,
        n_classes=3,
        max_iter=600,
    )
    scores = evaluate_q0_q1(
        models,
        eval_targets,
        q0_design_eval=controls_eval,
        eeg_posterior_eval=eeg_eval,
    )
    assert scores["q1"]["n_samples"] == n_eval
    assert scores["log_loss_gain_nats"] > 0.2
    assert scores["brier_gain"] > 0.1
    direct = categorical_proper_scores(
        np.full((2, 3), 1 / 3), np.asarray([0, 1])
    )
    assert direct["log_loss_nats"] == pytest.approx(np.log(3.0))
    assert direct["brier_score"] == pytest.approx(2 / 3)


def test_subject_block_bootstrap_is_deterministic_and_subject_equal():
    values = np.asarray([-1.0, 1.0, 3.0])
    first = subject_block_bootstrap(values, iterations=2000, seed=21)
    second = subject_block_bootstrap(values, iterations=2000, seed=21)
    assert first == second
    assert first["subject_count"] == 3
    assert first["mean"] == pytest.approx(1.0)
    assert first["ci_lower"] <= first["mean"] <= first["ci_upper"]
    assert 0.0 <= first["bootstrap_probability_positive"] <= 1.0


def test_q0_q1_rejects_bad_eval_shapes():
    targets = np.asarray([0, 1, 0, 1])
    controls = np.zeros((4, 1))
    eeg = _one_hot(targets, 2)
    models = fit_q0_q1(
        targets,
        q0_design_train=controls,
        eeg_posterior_train=eeg,
        n_classes=2,
        max_iter=100,
    )
    with pytest.raises(ValueError):
        evaluate_q0_q1(
            models,
            targets,
            q0_design_eval=np.zeros((4, 2)),
            eeg_posterior_eval=eeg,
        )
