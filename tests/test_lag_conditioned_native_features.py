import numpy as np
import pytest

from src.analysis.lag_conditioned_native_features import (
    EEG_NATIVE_FEATURES,
    FNIRS_COMPONENT_ROLES,
    FNIRS_NATIVE_FEATURES,
    NativeFeatureTargets,
    apply_masked_standardizer,
    extract_eeg_native_targets,
    extract_fnirs_native_targets,
    fit_masked_standardizer,
)


def test_eeg_native_target_names_shapes_and_mask():
    rng = np.random.default_rng(1)
    eeg = rng.normal(size=(2, 3, 800)).astype(np.float32)
    mask = np.asarray([[True, True], [True, False]])
    targets = extract_eeg_native_targets(
        eeg,
        mask,
        channel_names=("a", "b", "c"),
        sample_rate_hz=200.0,
        patch_size=400,
    )

    assert targets.values.shape == (2, 2, len(EEG_NATIVE_FEATURES))
    assert targets.valid_mask.shape == targets.values.shape
    assert targets.feature_names == EEG_NATIVE_FEATURES
    assert targets.valid_mask[0].all()
    assert not targets.valid_mask[1, 1].any()
    assert np.equal(targets.values[1, 1], 0.0).all()


def test_eeg_channel_mask_excludes_invalid_channels_from_aggregation():
    rng = np.random.default_rng(12)
    eeg = rng.normal(size=(1, 2, 400)).astype(np.float32)
    token_mask = np.ones((1, 1), dtype=bool)

    expected = extract_eeg_native_targets(
        eeg[:, :1],
        token_mask,
        channel_names=("valid",),
    )
    masked = extract_eeg_native_targets(
        eeg,
        token_mask,
        channel_valid_mask=np.asarray([[True, False]]),
        channel_names=("valid", "invalid"),
    )

    np.testing.assert_allclose(masked.values, expected.values)
    np.testing.assert_array_equal(masked.valid_mask, expected.valid_mask)


def test_fnirs_channel_mask_excludes_invalid_channels_within_each_component():
    time = np.arange(20, dtype=np.float32) / 10.0
    fnirs = np.stack(
        [
            1.0 + 2.0 * time,
            100.0 + 7.0 * time,
            -1.0 - 1.0 * time,
            -100.0 - 9.0 * time,
        ],
        axis=0,
    )[None, ...]
    token_mask = np.ones((1, 1), dtype=bool)

    expected = extract_fnirs_native_targets(
        fnirs[:, [0, 2]],
        token_mask,
        component_roles=("HbO", "HbR"),
        channel_names=("hbo_valid", "hbr_valid"),
    )
    masked = extract_fnirs_native_targets(
        fnirs,
        token_mask,
        component_roles=("HbO", "HbO", "HbR", "HbR"),
        channel_valid_mask=np.asarray([[True, False, True, False]]),
        channel_names=("hbo_valid", "hbo_invalid", "hbr_valid", "hbr_invalid"),
    )

    np.testing.assert_allclose(masked.values, expected.values)
    np.testing.assert_array_equal(masked.valid_mask, expected.valid_mask)


def test_channel_masks_require_batch_channel_shape_for_eeg_and_fnirs():
    eeg = np.zeros((2, 2, 400), dtype=np.float32)
    with pytest.raises(ValueError, match="channel_valid_mask"):
        extract_eeg_native_targets(
            eeg,
            np.ones((2, 1), dtype=bool),
            channel_valid_mask=np.ones((2, 1), dtype=bool),
        )

    fnirs = np.zeros((2, 2, 20), dtype=np.float32)
    with pytest.raises(ValueError, match="channel_valid_mask"):
        extract_fnirs_native_targets(
            fnirs,
            np.ones((2, 1), dtype=bool),
            component_roles=("HbO", "HbR"),
            channel_valid_mask=np.ones((1, 2), dtype=bool),
        )


def test_fnirs_targets_are_component_resolved_with_known_linear_slopes():
    time = np.arange(40, dtype=np.float32) / 10.0
    # Two locations, alternating HbO/HbR.  Channel means within a component
    # preserve the component's common slope.
    fnirs = np.stack(
        [
            1.0 + 2.0 * time,
            -1.0 - 1.0 * time,
            3.0 + 2.0 * time,
            -3.0 - 1.0 * time,
        ],
        axis=0,
    )[None, ...]
    targets = extract_fnirs_native_targets(
        fnirs,
        np.ones((1, 2), dtype=bool),
        component_roles=("HbO", "HbR", "HbO", "HbR"),
        channel_names=("a_o", "a_r", "b_o", "b_r"),
        sample_rate_hz=10.0,
        patch_size=20,
    )

    assert targets.values.shape == (
        1,
        2,
        len(FNIRS_COMPONENT_ROLES) * len(FNIRS_NATIVE_FEATURES),
    )
    assert targets.feature_names == tuple(
        f"{role}/{feature}"
        for role in FNIRS_COMPONENT_ROLES
        for feature in FNIRS_NATIVE_FEATURES
    )
    hbo_slope = targets.feature_names.index("HbO/slope")
    hbr_slope = targets.feature_names.index("HbR/slope")
    np.testing.assert_allclose(targets.values[..., hbo_slope], 2.0, atol=1e-5)
    np.testing.assert_allclose(targets.values[..., hbr_slope], -1.0, atol=1e-5)


def test_fnirs_roles_are_required_and_shape_checked():
    fnirs = np.zeros((1, 2, 20), dtype=np.float32)
    mask = np.ones((1, 1), dtype=bool)
    with pytest.raises(ValueError, match="component roles"):
        extract_fnirs_native_targets(fnirs, mask, component_roles=("HbO",))
    with pytest.raises(ValueError, match="lacks required"):
        extract_fnirs_native_targets(
            fnirs, mask, component_roles=("HbO", "HbO")
        )


def test_train_only_standardizer_uses_supported_entries_and_round_trips_mask():
    values = np.asarray(
        [
            [[1.0, 10.0], [3.0, 20.0]],
            [[5.0, 30.0], [1000.0, 40.0]],
        ],
        dtype=np.float32,
    )
    mask = np.asarray(
        [
            [[True, True], [True, True]],
            [[True, True], [False, True]],
        ]
    )
    targets = NativeFeatureTargets(values, mask, ("a", "b"))
    stats = fit_masked_standardizer(targets)
    transformed = apply_masked_standardizer(targets, stats)

    np.testing.assert_allclose(stats.mean, [3.0, 25.0])
    assert stats.count.tolist() == [3, 4]
    np.testing.assert_array_equal(transformed.valid_mask, mask)
    assert transformed.values[1, 1, 0] == 0.0
    for coordinate in range(2):
        admitted = transformed.values[..., coordinate][mask[..., coordinate]]
        np.testing.assert_allclose(admitted.mean(), 0.0, atol=1e-6)
        np.testing.assert_allclose(admitted.std(), 1.0, atol=1e-6)


def test_standardizer_fails_when_a_coordinate_has_no_fit_support():
    targets = NativeFeatureTargets(
        values=np.zeros((1, 2, 2), dtype=np.float32),
        valid_mask=np.asarray([[[True, False], [True, False]]]),
        feature_names=("a", "b"),
    )
    with pytest.raises(ValueError, match="lack fit support"):
        fit_masked_standardizer(targets)
