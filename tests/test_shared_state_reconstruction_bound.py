import numpy as np
from sklearn.decomposition import PCA

from experiments.evaluate_shared_state_reconstruction_bound import (
    _descriptor,
    _patchify,
    _pca_reconstruct,
    _safe_corr,
    _weighted,
)


def test_patchify_preserves_channel_major_waveform_order():
    signal = np.arange(24, dtype=np.float64).reshape(12, 2)
    patches = _patchify(signal, 4)

    assert patches.shape == (3, 2, 4)
    np.testing.assert_array_equal(patches[0, 0], [0, 2, 4, 6])
    np.testing.assert_array_equal(patches[0, 1], [1, 3, 5, 7])


def test_descriptor_is_finite_and_keeps_one_row_per_patch():
    rng = np.random.default_rng(0)
    patches = rng.normal(size=(7, 3, 20))
    features = _descriptor(patches, spectral_bins=5)

    assert features.shape == (7, 3 * 9)
    assert np.isfinite(features).all()


def test_validation_oracle_rank_curve_is_monotone():
    rng = np.random.default_rng(1)
    eeg = rng.normal(size=(80, 12))
    fnirs = rng.normal(size=(80, 4))
    joint = _weighted(eeg, fnirs)
    model = PCA(n_components=8, random_state=0).fit(joint)
    errors = [np.mean((_pca_reconstruct(model, joint, rank) - joint) ** 2) for rank in (1, 2, 4, 8)]

    assert all(right <= left + 1e-12 for left, right in zip(errors, errors[1:]))


def test_safe_corr_preserves_sign_and_rejects_constant_input():
    values = np.arange(10, dtype=np.float64)

    assert np.isclose(_safe_corr(values, values), 1.0)
    assert np.isclose(_safe_corr(values, -values), -1.0)
    assert np.isnan(_safe_corr(values, np.ones_like(values)))
