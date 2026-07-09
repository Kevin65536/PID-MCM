import numpy as np

from experiments.build_clean_eeg_fnirs_cache import _pair_single_trial_wavelengths


def test_single_trial_homer2_pair_labels_drop_wavelength_suffixes():
    values = np.arange(20, dtype=np.float64).reshape(5, 4)
    labels = ["AF7Fp1lowWL", "AF3Fp1lowWL", "AF7Fp1highWL", "AF3Fp1highWL"]

    paired, pair_labels = _pair_single_trial_wavelengths(values, labels)

    assert paired.shape == (5, 2, 2)
    assert pair_labels == ("AF7Fp1", "AF3Fp1")
