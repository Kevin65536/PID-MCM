import numpy as np
import pytest

from src.metrics.trajectory_reliability import trajectory_reliability_metrics


def test_nrmse_is_scale_invariant_and_reports_temporal_spread():
    observed = np.asarray([0.0, 1.0, 2.0, 3.0])
    reconstructed = np.asarray([0.0, 0.5, 2.5, 3.0])
    base = trajectory_reliability_metrics(observed, reconstructed)
    scaled = trajectory_reliability_metrics(20.0 * observed, 20.0 * reconstructed)

    assert scaled["trajectory_deviation_nrmse"] == pytest.approx(
        base["trajectory_deviation_nrmse"]
    )
    assert scaled["temporal_sd_ratio"] == pytest.approx(base["temporal_sd_ratio"])
    assert scaled["observed_temporal_sd"] == pytest.approx(
        20.0 * base["observed_temporal_sd"]
    )
    assert base["valid_point_count"] == 4
    assert not base["low_observed_variance"]


def test_metric_uses_exact_mask_and_keeps_low_variance_undefined():
    result = trajectory_reliability_metrics(
        np.asarray([2.0, 2.0, 999.0]),
        np.asarray([1.0, 3.0, -999.0]),
        valid_mask=np.asarray([True, True, False]),
    )
    assert result["valid_point_count"] == 2
    assert result["low_observed_variance"]
    assert np.isnan(result["trajectory_deviation_nrmse"])
    assert np.isnan(result["temporal_sd_ratio"])


def test_predictive_diagnostics_use_only_positive_finite_standard_deviation():
    observed = np.asarray([0.0, 1.0, 2.0, 3.0])
    reconstructed = np.asarray([0.0, 0.0, 4.0, 3.0])
    predictive_std = np.asarray([1.0, 1.0, 0.0, np.nan])
    result = trajectory_reliability_metrics(
        observed,
        reconstructed,
        predictive_std=predictive_std,
    )

    assert result["predictive_valid_point_count"] == 2
    assert result["posterior_predictive_sd_mean"] == pytest.approx(1.0)
    assert result["standardized_residual_rms"] == pytest.approx(np.sqrt(0.5))
    assert result["predictive_95_coverage"] == pytest.approx(1.0)


def test_underdispersed_predictive_standard_deviation_is_detected():
    observed = np.linspace(-1.0, 1.0, 21)
    reconstructed = observed + 0.2
    calibrated = trajectory_reliability_metrics(
        observed,
        reconstructed,
        predictive_std=np.full_like(observed, 0.2),
    )
    underdispersed = trajectory_reliability_metrics(
        observed,
        reconstructed,
        predictive_std=np.full_like(observed, 0.01),
    )

    assert calibrated["predictive_95_coverage"] == pytest.approx(1.0)
    assert calibrated["standardized_residual_rms"] == pytest.approx(1.0)
    assert underdispersed["predictive_95_coverage"] == pytest.approx(0.0)
    assert underdispersed["standardized_residual_rms"] == pytest.approx(20.0)


def test_metric_rejects_shape_drift_or_empty_support():
    with pytest.raises(ValueError, match="must match"):
        trajectory_reliability_metrics(np.ones(3), np.ones(2))
    with pytest.raises(ValueError, match="no finite valid points"):
        trajectory_reliability_metrics(
            np.ones(3),
            np.ones(3),
            valid_mask=np.zeros(3, dtype=bool),
        )
    with pytest.raises(ValueError, match="predictive_std"):
        trajectory_reliability_metrics(
            np.ones(3),
            np.ones(3),
            predictive_std=np.ones(2),
        )
