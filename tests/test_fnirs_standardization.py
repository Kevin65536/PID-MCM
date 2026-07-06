import numpy as np

from src.data.eeg_fnirs_dataset import apply_temporal_filter
from src.data.fnirs_standardization import (
    FNIRS_STANDARDIZATION_SCHEMA,
    get_fnirs_measurement_contract,
    restore_fnirs_record,
    standardize_fnirs_record,
)


def test_contracts_preserve_distinct_native_measurement_families():
    single = get_fnirs_measurement_contract("eeg_fnirs_single_trial", "wavelength_pair")
    simultaneous = get_fnirs_measurement_contract("simultaneous_eeg_nirs", "oxy_deoxy")
    visual = get_fnirs_measurement_contract("visual_cognitive_motivation", "oxy_deoxy")

    assert single.native_unit == "V"
    assert single.measurement_family == "optical_intensity"
    assert simultaneous.native_unit == "mmol/L"
    assert visual.native_unit.startswith("unreported_")
    assert single.schema == FNIRS_STANDARDIZATION_SCHEMA


def test_full_record_standardization_removes_offset_linear_drift_and_scale():
    sample_rate = 10.0
    time = np.arange(1200) / sample_rate
    physiological = np.column_stack((np.sin(2 * np.pi * 0.08 * time), 0.4 * np.cos(2 * np.pi * 0.05 * time)))
    raw = np.column_stack((20.0 + 0.03 * time, -5.0 - 0.01 * time)) + physiological * np.asarray([4.0, 0.2])
    result = standardize_fnirs_record(
        raw,
        sample_rate_hz=sample_rate,
        contract=get_fnirs_measurement_contract("simultaneous_eeg_nirs", "oxy_deoxy"),
    )

    assert result.values.shape == raw.shape
    assert np.max(np.abs(np.median(result.values, axis=0))) < 0.1
    assert result.quality["residual_drift_sd_per_min_median"] < 0.1
    np.testing.assert_allclose(restore_fnirs_record(result.values, result.state), raw, rtol=1e-6, atol=1e-5)


def test_crop_values_are_invariant_when_full_record_is_standardized_first():
    time = np.arange(1000) / 10.0
    raw = np.column_stack((5 + 0.02 * time + np.sin(time), 10 - 0.01 * time + np.cos(time)))
    result = standardize_fnirs_record(
        raw,
        sample_rate_hz=10.0,
        contract=get_fnirs_measurement_contract("eeg_fnirs_single_trial", "wavelength_pair"),
    )
    start, end = 200, 350
    restored_crop = restore_fnirs_record(result.values[start:end], result.state, start_sample=start)
    np.testing.assert_allclose(restored_crop, raw[start:end], rtol=1e-6, atol=1e-5)


def test_nonfinite_samples_are_interpolated_and_recorded():
    raw = np.column_stack((np.arange(20.0), np.arange(20.0) ** 2))
    raw[4:7, 0] = np.nan
    result = standardize_fnirs_record(
        raw,
        sample_rate_hz=10.0,
        contract=get_fnirs_measurement_contract("refed", "hbo_hbr"),
    )
    assert np.isfinite(result.values).all()
    assert result.state.repaired_nonfinite == (3, 0)


def test_loader_preprocessing_can_enable_versioned_measurement_standardization():
    time = np.arange(600) / 10.0
    raw = np.column_stack((100 + 0.1 * time + np.sin(time), 20 - 0.2 * time + np.cos(time)))
    processed = apply_temporal_filter(
        raw,
        sample_rate=10.0,
        modality="fnirs",
        preprocessing={
            "measurement_standardization": {
                "enabled": True,
                "dataset_id": "eeg_fnirs_single_trial",
                "signal_key": "wavelength_pair",
            }
        },
    )
    assert processed.dtype == np.float32
    assert np.max(np.abs(np.median(processed, axis=0))) < 0.1
