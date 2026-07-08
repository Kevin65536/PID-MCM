import numpy as np

from src.data.homer2_preprocessing import (
    HOMER2_ALIGNMENT_SCHEMA,
    apply_homer2_aligned_contract,
    get_homer2_dataset_compatibility,
    homer2_compatibility_manifest,
    intensity_to_optical_density,
    modified_beer_lambert,
)


def test_compatibility_manifest_marks_full_and_partial_datasets():
    single = get_homer2_dataset_compatibility("eeg_fnirs_single_trial")
    simultaneous = get_homer2_dataset_compatibility("simultaneous_eeg_nirs")
    manifest = homer2_compatibility_manifest()

    assert manifest["schema"] == HOMER2_ALIGNMENT_SCHEMA
    assert single.entry_stage == "raw_intensity"
    assert "intensity_to_optical_density" in single.possible_steps
    assert simultaneous.completeness == "partial_post_conversion"
    assert "raw_wl1_wl2_intensity" in simultaneous.missing_inputs


def test_intensity_to_optical_density_is_finite_and_baselined():
    time = np.arange(400) / 10.0
    intensity = np.stack(
        (
            1.2 + 0.01 * np.sin(time),
            0.9 + 0.02 * np.cos(time),
        ),
        axis=1,
    )
    od, quality = intensity_to_optical_density(intensity)

    assert od.shape == intensity.shape
    assert np.isfinite(od).all()
    assert abs(float(np.median(od))) < 1e-3
    assert quality["clamped_nonpositive_fraction"] == 0.0


def test_raw_intensity_branch_applies_od_filter_and_mbll():
    time = np.arange(1000) / 10.0
    base = np.ones((time.size, 3, 2), dtype=np.float64)
    base[:, :, 0] += 0.02 * np.sin(2 * np.pi * 0.05 * time)[:, None]
    base[:, :, 1] += 0.01 * np.cos(2 * np.pi * 0.04 * time)[:, None]

    result = apply_homer2_aligned_contract(
        base,
        dataset_id="eeg_fnirs_single_trial",
        sample_rate_hz=10.0,
        entry_stage="raw_intensity",
        wavelengths_nm=(760.0, 850.0),
    )

    assert result.values.shape == (time.size, 6)
    assert np.isfinite(result.values).all()
    assert "intensity_to_optical_density" in result.state.applied_steps
    assert "modified_beer_lambert" in result.state.applied_steps
    assert result.state.schema == HOMER2_ALIGNMENT_SCHEMA


def test_chromophore_branch_records_missing_raw_homer2_inputs():
    time = np.arange(1000) / 10.0
    values = np.column_stack(
        (
            np.sin(2 * np.pi * 0.04 * time),
            -0.5 * np.cos(2 * np.pi * 0.04 * time),
        )
    )

    result = apply_homer2_aligned_contract(
        values,
        dataset_id="simultaneous_eeg_nirs",
        sample_rate_hz=10.0,
        entry_stage="chromophore",
        wavelengths_nm=(760.0, 850.0),
    )

    assert result.values.shape == values.shape
    assert "bandpass" in result.state.applied_steps
    assert "intensity_to_optical_density" in result.state.skipped_steps
    assert "raw_light_intensity" in result.state.missing_inputs
    assert "modified_beer_lambert" in result.state.skipped_steps


def test_modified_beer_lambert_returns_hbo_hbr_axis():
    od = np.zeros((50, 4, 2), dtype=np.float64)
    od[:, :, 0] = 0.01
    od[:, :, 1] = -0.005
    concentration, quality = modified_beer_lambert(od, wavelengths_nm=(760.0, 850.0))

    assert concentration.shape == (50, 4, 2)
    assert np.isfinite(concentration).all()
    assert quality["wavelengths_nm"] == [760.0, 850.0]
