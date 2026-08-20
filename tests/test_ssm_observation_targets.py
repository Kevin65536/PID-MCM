import numpy as np

from src.data.ssm_observation_targets import (
    apply_ssm_observation_teachers,
    extract_eeg_spatial_band_trajectory,
    extract_fnirs_patch_trajectory,
    fit_observation_channel_selection,
    fit_ssm_observation_teachers,
)


def _features(sample_count: int = 10):
    rng = np.random.default_rng(14)
    eeg = rng.normal(size=(sample_count, 2, 400)).astype(np.float32)
    fnirs = rng.normal(size=(sample_count, 2, 20)).astype(np.float32)
    token = np.ones((sample_count, 10), dtype=bool)
    eeg_point = np.ones((sample_count, 400), dtype=bool)
    fnirs_point = np.ones((sample_count, 20), dtype=bool)
    channel = np.ones((sample_count, 2), dtype=bool)
    eeg_features = extract_eeg_spatial_band_trajectory(
        eeg,
        token_valid_mask=token,
        point_valid_mask=eeg_point,
        channel_valid_mask=channel,
        channel_names=("C3", "C4"),
        sampling_rate_hz=200.0,
    )
    fnirs_features = extract_fnirs_patch_trajectory(
        fnirs,
        token_valid_mask=token,
        point_valid_mask=fnirs_point,
        channel_valid_mask=channel,
        channel_names=("HbO", "HbR"),
    )
    return eeg_features, fnirs_features


def test_channel_selection_is_fit_only_spatial_and_chromophore_paired():
    rng = np.random.default_rng(9)
    eeg = rng.normal(size=(8, 8, 40))
    fnirs = rng.normal(size=(8, 6, 20))
    selection = fit_observation_channel_selection(
        eeg,
        fnirs,
        eeg_channel_valid_mask=np.ones((8, 8), dtype=bool),
        fnirs_channel_valid_mask=np.ones((8, 6), dtype=bool),
        eeg_channel_names=("Fp1", "C4", "Pz", "C3", "F4", "Cz", "O1", "F3"),
        fnirs_channel_names=(
            "S1_D1_HbO",
            "S2_D2_HbO",
            "S3_D3_HbO",
            "S1_D1_HbR",
            "S2_D2_HbR",
            "S3_D3_HbR",
        ),
        fnirs_component_roles=("HbO", "HbO", "HbO", "HbR", "HbR", "HbR"),
        eeg_channel_count=6,
    )
    assert len(selection.eeg_indices) == 6
    assert "C3" in selection.eeg_channel_names
    assert "C4" in selection.eeg_channel_names
    assert len(selection.fnirs_indices) == 2
    hbo_stem = selection.fnirs_channel_names[0].replace("HbO", "")
    hbr_stem = selection.fnirs_channel_names[1].replace("HbR", "")
    assert hbo_stem == hbr_stem


def test_observation_targets_keep_eeg_space_and_full_fnirs_patch():
    eeg, fnirs = _features()
    assert eeg.values.shape == (10, 10, 6)
    assert eeg.feature_names == (
        "C3:alpha",
        "C3:beta",
        "C3:low_gamma",
        "C4:alpha",
        "C4:beta",
        "C4:low_gamma",
    )
    assert fnirs.values.shape == (10, 10, 4)
    assert fnirs.feature_names == (
        "HbO:sample_00",
        "HbO:sample_01",
        "HbR:sample_00",
        "HbR:sample_01",
    )


def test_teacher_fits_pool_all_sequences_without_label_input():
    eeg, fnirs = _features()
    fits = fit_ssm_observation_teachers(eeg, fnirs)
    assert fits.labels_used is False
    assert len(fits.provenance_id) == 64
    assert fits.eeg_self.training_sequence_count == 10
    assert fits.fnirs_self.training_sequence_count == 10
    teacher = apply_ssm_observation_teachers(eeg, fnirs, fits)
    self_targets = teacher.targets("SSM-SELF")
    joint_targets = teacher.targets("SSM-JOINT")
    native_targets = teacher.targets("NATIVE")
    assert self_targets["eeg_clean_target"].shape == eeg.values.shape
    assert self_targets["fnirs_clean_target"].shape == fnirs.values.shape
    np.testing.assert_allclose(
        self_targets["eeg_clean_target"] + self_targets["eeg_residual_target"],
        teacher.eeg_observation,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        joint_targets["fnirs_clean_target"] + joint_targets["fnirs_residual_target"],
        teacher.fnirs_observation,
        atol=1e-6,
    )
    assert np.all(native_targets["eeg_residual_target"] == 0.0)
    assert not np.allclose(
        self_targets["fnirs_clean_target"],
        joint_targets["fnirs_clean_target"],
    )
