import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/scripts/analyze_r2d_continuous_observability.py"
SPEC = importlib.util.spec_from_file_location("r2d_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_patch_timing_derives_eight_second_center_from_frozen_window():
    config = {
        "data": {"window": {"offset_s": -5.0, "duration_s": 20.0}},
        "model": {"num_tokens": 10},
    }
    starts, centers, ends = analysis._patch_timing(config, 10)
    assert starts[6] == pytest.approx(7.0)
    assert centers[6] == pytest.approx(8.0)
    assert ends[6] == pytest.approx(9.0)


def test_subject_patch_delta_r2_aggregates_points_before_ratio():
    observed = np.asarray([[1.0, 1.0], [3.0, 3.0]])
    prediction = np.asarray([[0.5, 0.5], [2.0, 2.0]])
    baseline = np.zeros_like(observed)
    mask = np.ones_like(observed, dtype=bool)
    expected = 1.0 - (2.5 / 20.0)
    assert analysis._delta_r2(
        observed, prediction, baseline, mask
    ) == pytest.approx(expected)


def test_simultaneous_band_covers_full_modality_patch_family():
    values = {
        "eeg": np.asarray([[0.0, 0.2], [0.1, 0.3], [0.2, 0.4]]),
        "fnirs": np.asarray([[-0.1, 0.1], [0.0, 0.2], [0.1, 0.3]]),
    }
    rng = np.random.default_rng(7)
    indices = rng.integers(0, 3, size=(2000, 3))
    bands = analysis._simultaneous_bootstrap_bands(values, indices)
    for modality in analysis.MODALITIES:
        means = values[modality].mean(axis=0)
        lower = np.asarray(bands[modality]["simultaneous_ci95_lower"])
        upper = np.asarray(bands[modality]["simultaneous_ci95_upper"])
        assert lower.shape == (2,)
        assert upper.shape == (2,)
        assert np.all(lower <= means)
        assert np.all(upper >= means)


def test_primary_consistency_fails_closed_on_summary_drift():
    values = np.asarray([0.1, 0.2, -0.1, 0.0, 0.3])
    expected = analysis._subject_bootstrap_summary(
        values, iterations=1000, confidence_level=0.95, seed=3
    )
    source = {
        "eeg": dict(expected),
        "fnirs": dict(expected),
        "equal_modalities": dict(expected),
    }
    recomputed = {
        "eeg": dict(expected),
        "fnirs": dict(expected),
        "equal_modalities": dict(expected),
    }
    analysis._assert_primary_consistency(source, recomputed)
    source["fnirs"]["subject_equal_mean_delta_r2"] += 0.01
    with pytest.raises(ValueError, match="Primary summary mismatch"):
        analysis._assert_primary_consistency(source, recomputed)
