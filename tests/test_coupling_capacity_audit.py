import numpy as np

from experiments.scripts.analyze_coupling_identifiability import (
    counts_by_lag,
    equal_support_counts_by_lag,
    lag_count_from_max,
)


def test_max_lag_tokens_maps_to_inclusive_lag_count():
    assert lag_count_from_max(tokens_per_window=10, max_lag_tokens=5) == 6
    assert lag_count_from_max(tokens_per_window=10, max_lag_tokens=99) == 10
    assert lag_count_from_max(tokens_per_window=10, max_lag_tokens=None) == 10


def test_equal_support_counts_use_same_number_of_pairs_per_lag():
    eeg = np.tile(np.arange(10), (3, 1))
    fnirs = np.tile(np.arange(10), (3, 1))

    ordinary = counts_by_lag(eeg, fnirs, k_eeg=10, k_fnirs=10, max_lag_tokens=5)
    equal_support = equal_support_counts_by_lag(eeg, fnirs, k_eeg=10, k_fnirs=10, max_lag_tokens=5)

    assert ordinary.shape[0] == 6
    assert ordinary.sum(axis=(1, 2)).tolist() == [30, 27, 24, 21, 18, 15]
    assert equal_support.shape[0] == 6
    assert equal_support.sum(axis=(1, 2)).tolist() == [15, 15, 15, 15, 15, 15]
