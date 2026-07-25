from pathlib import Path

import numpy as np
import pytest

from src.data.eeg_artifact_preprocessing import (
    EEGArtifactCleaningConfig,
    SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA,
    SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V2,
    SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V3,
    SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V4,
    clean_single_trial_eeg,
)
from src.data.unified_physiology import (
    SIMULTANEOUS_EEG_EOG_CLEAN_SCHEMA_V1,
    NativeEEGRecord,
    preprocess_eeg_record_with_quality,
)


def _synthetic_record(duration_s: float = 60.0, sample_rate_hz: float = 200.0):
    rng = np.random.default_rng(20260714)
    time = np.arange(int(duration_s * sample_rate_hz)) / sample_rate_hz
    vertical = np.zeros_like(time)
    for center in np.arange(3.0, duration_s - 1.0, 6.0):
        vertical += 8.0 * np.exp(-0.5 * ((time - center) / 0.12) ** 2)
    horizontal = np.sin(2 * np.pi * 0.2 * time)
    eog = np.column_stack(
        (
            vertical + 0.05 * rng.normal(size=len(time)),
            horizontal + 0.05 * rng.normal(size=len(time)),
        )
    )
    brain = np.column_stack(
        [
            np.sin(2 * np.pi * (8.0 + index * 0.4) * time)
            + 0.15 * rng.normal(size=len(time))
            for index in range(8)
        ]
    )
    contamination = np.linspace(0.15, 0.8, brain.shape[1])
    eeg = brain + vertical[:, None] * contamination[None, :]
    burst_start = int(min(30.0, max(1.0, duration_s * 0.5)) * sample_rate_hz)
    burst_stop = burst_start + int(sample_rate_hz)
    burst_time = time[: burst_stop - burst_start]
    eeg[burst_start:burst_stop] += (
        2.0
        * np.sin(2 * np.pi * 38.0 * burst_time)[:, None]
        * rng.normal(size=(burst_stop - burst_start, brain.shape[1]))
    )
    return eeg, eog, brain, sample_rate_hz


def test_cleaner_reduces_eog_correlation_and_preserves_shape():
    eeg, eog, _, sample_rate = _synthetic_record()
    result = clean_single_trial_eeg(
        eeg,
        eog,
        sample_rate_hz=sample_rate,
        channel_names=[f"C{index}" for index in range(eeg.shape[1])],
        eog_channel_names=["VEOG", "HEOG"],
    )
    assert result.cleaned_values.shape == eeg.shape
    assert result.artifact_mask.shape == (len(eeg),)
    assert result.bad_channel_mask.shape == (eeg.shape[1],)
    assert not np.any(result.bad_channel_mask)
    assert result.state["schema"] == SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA
    assert result.state["median_eog_correlation_after"] < 0.25 * result.state["median_eog_correlation_before"]
    assert 0.01 < result.state["artifact_fraction"] < 0.4


def test_high_frequency_burst_is_masked_without_mass_channel_rejection():
    eeg, eog, _, sample_rate = _synthetic_record()
    result = clean_single_trial_eeg(eeg, eog, sample_rate_hz=sample_rate)
    burst = result.high_frequency_mask[int(29.5 * sample_rate) : int(31.5 * sample_rate)]
    assert np.mean(burst) > 0.4
    assert not np.any(result.bad_channel_mask)
    assert result.state["schema"] == SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V4
    assert result.state["muscle_action"] == "mask_gated_high_frequency_attenuation_v1"
    assert result.state["muscle_correction"]["high_frequency_energy_reduction_in_mask"] > 0.0


def test_cleaning_is_deterministic():
    eeg, eog, _, sample_rate = _synthetic_record(duration_s=20.0)
    config = EEGArtifactCleaningConfig(max_regression_samples=5_000)
    first = clean_single_trial_eeg(eeg, eog, sample_rate_hz=sample_rate, config=config)
    second = clean_single_trial_eeg(eeg, eog, sample_rate_hz=sample_rate, config=config)
    np.testing.assert_array_equal(first.cleaned_values, second.cleaned_values)
    np.testing.assert_array_equal(first.artifact_mask, second.artifact_mask)
    assert first.state == second.state


def test_unified_preprocessing_clean_branch_does_not_expose_artifact_masks():
    eeg, eog, _, sample_rate = _synthetic_record(duration_s=20.0)
    record = NativeEEGRecord(
        values=eeg,
        sample_rate_hz=sample_rate,
        channel_names=tuple(f"C{index}" for index in range(eeg.shape[1])),
        native_unit="uV",
        source_path=Path("synthetic.mat"),
        auxiliary_values=eog,
        auxiliary_channel_names=("VEOG", "HEOG"),
    )
    canonical, state, quality = preprocess_eeg_record_with_quality(
        record,
        signal_branch=SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA,
    )
    assert canonical.shape == eeg.shape
    assert quality["artifact_mask"].shape == (len(eeg),)
    assert not np.any(quality["artifact_mask"])
    assert quality["bad_channel_mask"].shape == (eeg.shape[1],)
    assert state["signal_branch"] == SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA
    assert state["artifact_cleaning"]["schema"] == SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA
    assert state["artifact_cleaning"]["artifact_fraction"] > 0.0
    assert state["artifact_mask_policy"] == "disabled_all_false_no_invalid_authority_v1"


def test_clean_branch_requires_eog_auxiliary_channels():
    eeg, _, _, sample_rate = _synthetic_record(duration_s=10.0)
    record = NativeEEGRecord(
        values=eeg,
        sample_rate_hz=sample_rate,
        channel_names=tuple(f"C{index}" for index in range(eeg.shape[1])),
        native_unit="uV",
        source_path=Path("synthetic.mat"),
    )
    with pytest.raises(ValueError, match="retained EOG"):
        preprocess_eeg_record_with_quality(record, signal_branch=SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA)


def test_v2_remains_mask_only_and_v3_attenuates_detected_high_frequency_bursts():
    eeg, eog, _, sample_rate = _synthetic_record(duration_s=40.0)
    record = NativeEEGRecord(
        values=eeg,
        sample_rate_hz=sample_rate,
        channel_names=tuple(f"C{index}" for index in range(eeg.shape[1])),
        native_unit="uV",
        source_path=Path("synthetic.mat"),
        auxiliary_values=eog,
        auxiliary_channel_names=("VEOG", "HEOG"),
    )
    v2, v2_state, v2_quality = preprocess_eeg_record_with_quality(
        record, signal_branch=SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V2
    )
    v3, v3_state, v3_quality = preprocess_eeg_record_with_quality(
        record, signal_branch=SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V3
    )
    np.testing.assert_array_equal(v2_quality["artifact_mask"], v3_quality["artifact_mask"])
    assert v2_state["artifact_cleaning"]["muscle_correction"]["method"] == "mask_only"
    assert v3_state["artifact_cleaning"]["muscle_correction"]["method"] == (
        "mask_gated_high_frequency_attenuation_v1"
    )
    assert not np.array_equal(v2, v3)


def test_simultaneous_eog_branch_repairs_ocular_signal_without_other_interventions():
    eeg, eog, _, sample_rate = _synthetic_record(duration_s=40.0)
    record = NativeEEGRecord(
        values=eeg,
        sample_rate_hz=sample_rate,
        channel_names=tuple(f"C{index}" for index in range(eeg.shape[1])),
        native_unit="uV",
        source_path=Path("synthetic.mat"),
        auxiliary_values=eog,
        auxiliary_channel_names=("VEOG", "HEOG"),
    )
    _, state, quality = preprocess_eeg_record_with_quality(
        record, signal_branch=SIMULTANEOUS_EEG_EOG_CLEAN_SCHEMA_V1
    )
    cleaning = state["artifact_cleaning"]
    assert cleaning["schema"] == SIMULTANEOUS_EEG_EOG_CLEAN_SCHEMA_V1
    assert cleaning["median_eog_correlation_after"] < cleaning["median_eog_correlation_before"]
    assert cleaning["muscle_correction"]["method"] == "mask_only"
    assert not np.any(quality["bad_channel_mask"])


def test_flat_channel_does_not_emit_dataset_specific_bad_mask_or_interpolation():
    eeg, eog, _, sample_rate = _synthetic_record(duration_s=20.0)
    eeg[:, 0] = 0.0
    angles = np.linspace(0.0, 2.0 * np.pi, eeg.shape[1], endpoint=False)
    positions = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
    result = clean_single_trial_eeg(
        eeg,
        eog,
        sample_rate_hz=sample_rate,
        channel_positions=positions,
    )
    assert not np.any(result.bad_channel_mask)
    assert result.state["interpolation"]["method"] == "disabled_for_cross_dataset_uniformity"
    assert result.state["bad_channel_policy"]["mask_emitted"] is False
    assert result.state["bad_channel_policy"]["interpolation_applied"] is False


def test_v4_notch_removes_50_hz_without_bad_channel_mask():
    eeg, eog, _, sample_rate = _synthetic_record(duration_s=20.0)
    time = np.arange(len(eeg)) / sample_rate
    eeg[:, 0] += 50.0 * np.sin(2.0 * np.pi * 50.0 * time)
    result = clean_single_trial_eeg(eeg, eog, sample_rate_hz=sample_rate)
    line = result.state["line_noise"]
    assert line["method"] == "zero_phase_iirnotch"
    assert line["frequency_hz"] == 50.0
    assert line["ratio_after_by_channel"][0] < 0.01 * line["ratio_before_by_channel"][0]
    assert not np.any(result.bad_channel_mask)
