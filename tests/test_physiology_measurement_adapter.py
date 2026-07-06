import numpy as np

from src.data.physiology_measurement_adapter import (
    ADAPTER_SCHEMA,
    MeasurementAdapterSpec,
    PhysiologyMeasurementAdapter,
)


def _fit(records, transform="center"):
    return PhysiologyMeasurementAdapter.fit(
        records,
        dataset="example",
        modality="fnirs",
        original_semantics="relative HbO/HbR",
        original_unit="mmol/L",
        canonical_semantics="baseline-relative paired optical measurement",
        transform=transform,
        channel_names=("primary", "secondary"),
        fit_subjects=("01", "02"),
    )


def test_adapter_round_trip_and_schema():
    record = np.asarray([[10.0, 20.0], [11.0, 18.0], [9.0, 22.0], [12.0, 17.0]])
    adapter = _fit([record])
    baseline = adapter.record_baseline(record)
    canonical = adapter.transform(record, baseline=baseline)

    np.testing.assert_allclose(adapter.inverse_transform(canonical, baseline=baseline), record)
    assert adapter.spec.schema == ADAPTER_SCHEMA
    restored = MeasurementAdapterSpec.from_dict(adapter.spec.to_dict())
    assert restored == adapter.spec


def test_adapter_is_crop_position_invariant_when_record_baseline_is_reused():
    record = np.column_stack((np.linspace(2.0, 4.0, 100), np.linspace(5.0, 1.0, 100)))
    adapter = _fit([record])
    baseline = adapter.record_baseline(record)
    full = adapter.transform(record, baseline=baseline)

    np.testing.assert_allclose(
        adapter.transform(record[20:40], baseline=baseline),
        full[20:40],
    )


def test_shared_pair_scale_preserves_relative_amplitude_ratio():
    time = np.linspace(0.0, 2.0 * np.pi, 200)
    record = np.column_stack((2.0 * np.sin(time), 0.5 * np.sin(time)))
    adapter = _fit([record])
    canonical = adapter.transform(record)

    raw_ratio = np.ptp(record[:, 0]) / np.ptp(record[:, 1])
    canonical_ratio = np.ptp(canonical[:, 0]) / np.ptp(canonical[:, 1])
    np.testing.assert_allclose(canonical_ratio, raw_ratio)


def test_relative_change_handles_zero_baseline_without_nonfinite_values():
    record = np.asarray([[0.0, 10.0], [1.0, 11.0], [-1.0, 9.0], [0.5, 10.5]])
    adapter = _fit([record], transform="relative_change")
    canonical = adapter.transform(record)

    assert np.all(np.isfinite(canonical))
