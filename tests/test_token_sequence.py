import numpy as np

from src.analysis.token_sequence import (
    analyze_cross_modal_lags,
    circular_shift_coupling_null,
    coupling_metrics_from_counts,
    cross_modal_lag_counts,
    markov_log_loss,
    summarize_sequences,
    transition_counts,
)


def test_transition_counts_do_not_cross_windows_or_invalid_gaps():
    tokens = np.asarray([[0, 0, 1, 1], [1, 0, 0, 1]])
    mask = np.asarray([[True, True, False, True], [True, True, True, True]])
    counts = transition_counts(tokens, mask, codebook_size=2)
    assert counts.tolist() == [[2, 1], [1, 0]]


def test_sequence_summary_reports_runs_and_self_transitions():
    tokens = np.asarray([[0, 0, 1, 1], [1, 1, 1, 0]])
    summary = summarize_sequences(tokens, codebook_size=2)
    assert summary.token_count == 8
    assert summary.transition_count == 6
    assert summary.self_transition_fraction == 4 / 6
    rows = {row["token_id"]: row for row in summary.token_rows}
    assert rows[0]["run_length_max_tokens"] == 2
    assert rows[1]["run_length_max_tokens"] == 3


def test_sequence_summary_handles_empty_and_single_position_windows():
    empty = summarize_sequences(np.empty((3, 0), dtype=np.int64), codebook_size=2)
    assert empty.token_count == 0
    assert empty.transition_count == 0
    assert empty.self_transition_fraction is None
    assert empty.occupancy_entropy_nats is None

    singleton = summarize_sequences(
        np.asarray([[0], [1], [0]]),
        np.asarray([[True], [False], [True]]),
        codebook_size=2,
    )
    assert singleton.token_count == 2
    assert singleton.transition_count == 0
    rows = {row["token_id"]: row for row in singleton.token_rows}
    assert rows[0]["transition_out_count"] == 0
    assert rows[0]["transition_in_count"] == 0
    assert rows[0]["transition_out_entropy_nats"] is None


def test_cross_modal_null_is_deterministic_and_preserves_shape():
    eeg = np.asarray([[0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
    fnirs = np.asarray([[1, 1, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
    first = analyze_cross_modal_lags(
        eeg,
        fnirs,
        lags=(-1, 0, 1),
        eeg_codebook_size=2,
        fnirs_codebook_size=2,
        permutations=8,
        seed=7,
    )
    second = analyze_cross_modal_lags(
        eeg,
        fnirs,
        lags=(-1, 0, 1),
        eeg_codebook_size=2,
        fnirs_codebook_size=2,
        permutations=8,
        seed=7,
    )
    rows, matrices, null = first
    assert matrices.shape == (3, 2, 2)
    assert null.shape == (8, 3)
    assert np.array_equal(null, second[2])
    assert rows[1]["null_policy"] == "within-window whole-sequence circular shift"


def test_single_position_circular_null_is_explicitly_not_estimable():
    rows, matrices, null = analyze_cross_modal_lags(
        np.asarray([[0], [1]]),
        np.asarray([[1], [0]]),
        lags=(0,),
        eeg_codebook_size=2,
        fnirs_codebook_size=2,
        permutations=4,
        seed=3,
    )
    assert matrices.shape == (1, 2, 2)
    assert null.shape == (4, 1)
    assert np.isnan(null).all()
    assert rows[0]["null_nmi_mean"] is None
    assert rows[0]["nmi_empirical_p"] is None
    assert rows[0]["null_degenerate_reason"] is not None


def test_circular_null_rolls_each_whole_window_with_its_mask():
    eeg = np.asarray([[0, 1, 0, 1], [1, 1, 0, 0]])
    fnirs = np.asarray([[1, 0, 1, 0], [0, 1, 1, 0]])
    eeg_mask = np.asarray(
        [[True, False, True, True], [True, True, False, True]]
    )
    fnirs_mask = np.asarray(
        [[True, True, True, False], [True, False, True, True]]
    )
    seed = 19
    null = circular_shift_coupling_null(
        eeg,
        fnirs,
        eeg_valid_mask=eeg_mask,
        fnirs_valid_mask=fnirs_mask,
        lags=(0,),
        eeg_codebook_size=2,
        fnirs_codebook_size=2,
        permutations=1,
        seed=seed,
    )
    shifts = np.random.default_rng(seed).integers(1, eeg.shape[1], size=eeg.shape[0])
    shifted_eeg = np.stack(
        [np.roll(row, int(shift)) for row, shift in zip(eeg, shifts)]
    )
    shifted_mask = np.stack(
        [np.roll(row, int(shift)) for row, shift in zip(eeg_mask, shifts)]
    )
    _, expected_counts = cross_modal_lag_counts(
        shifted_eeg,
        fnirs,
        eeg_valid_mask=shifted_mask,
        fnirs_valid_mask=fnirs_mask,
        lags=(0,),
        eeg_codebook_size=2,
        fnirs_codebook_size=2,
    )
    expected = coupling_metrics_from_counts(expected_counts[0])[
        "normalized_mutual_information"
    ]
    assert null[0, 0] == expected


def test_lag_validation_is_strict_and_documents_boundary():
    eeg = np.zeros((2, 3), dtype=np.int64)
    fnirs = np.zeros((2, 3), dtype=np.int64)
    with np.testing.assert_raises_regex(ValueError, "lags must contain integers"):
        cross_modal_lag_counts(
            eeg,
            fnirs,
            lags=(0.5,),
            eeg_codebook_size=2,
            fnirs_codebook_size=2,
        )
    with np.testing.assert_raises_regex(ValueError, r"abs\(lag\) < positions"):
        cross_modal_lag_counts(
            eeg,
            fnirs,
            lags=(3,),
            eeg_codebook_size=2,
            fnirs_codebook_size=2,
        )
    lags, counts = cross_modal_lag_counts(
        eeg,
        fnirs,
        lags=(-2, 0, 2),
        eeg_codebook_size=2,
        fnirs_codebook_size=2,
    )
    assert lags.tolist() == [-2, 0, 2]
    assert counts[:, 0, 0].tolist() == [2, 6, 2]


def test_coupling_metrics_report_sparse_support_in_both_directions():
    metrics = coupling_metrics_from_counts(np.asarray([[3, 0, 1], [0, 0, 0]]))
    assert metrics["possible_cells"] == 6
    assert metrics["nonzero_cells"] == 2
    assert metrics["nonzero_cell_fraction"] == 2 / 6
    assert metrics["active_eeg_tokens"] == 1
    assert metrics["active_fnirs_tokens"] == 2
    assert metrics["conditional_entropy_eeg_given_fnirs_nats"] == 0.0


def test_markov_log_loss_favors_predictable_transition():
    train = np.tile(np.asarray([[0, 1, 0, 1, 0, 1]]), (20, 1))
    validation = np.tile(np.asarray([[0, 1, 0, 1, 0, 1]]), (5, 1))
    result = markov_log_loss(
        train,
        validation,
        codebook_size=2,
        alpha=0.5,
    )
    assert result["order1_gain_nats"] > 0.4


def test_markov_log_loss_handles_single_position_training_windows():
    result = markov_log_loss(
        np.asarray([[0], [1], [0]]),
        np.asarray([[0], [1]]),
        codebook_size=2,
    )
    assert result["validation_transition_count"] == 0
    assert np.isnan(result["order0_log_loss_nats"])
