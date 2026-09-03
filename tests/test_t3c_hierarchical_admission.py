from __future__ import annotations

import json
from math import log, sqrt

import numpy as np
import pytest

import experiments.evaluate_t3c_hierarchical_composite_admission as t3c


def _raw_parameters() -> dict[str, float]:
    """A strictly interior raw parameter point for the composite map."""

    return {
        "beta": 1.5,
        "kappa": 0.8,
        "gamma": 0.4,
        "tau": 1.7,
        "alpha": 0.32,
    }


def test_composite_roundtrip_is_explicit_and_requires_alpha_gauge() -> None:
    raw = _raw_parameters()
    phi = t3c.composite_from_raw(raw)

    assert set(phi) == {"log_gain", "log_time", "logit_zeta", "log_tv"}
    assert phi["log_gain"] == pytest.approx(log(raw["beta"] / raw["gamma"]))
    assert phi["log_time"] == pytest.approx(log(1.0 / sqrt(raw["gamma"])))
    zeta = raw["kappa"] / (2.0 * sqrt(raw["gamma"]))
    assert phi["logit_zeta"] == pytest.approx(log(zeta / (1.0 - zeta)))
    assert phi["log_tv"] == pytest.approx(log(raw["tau"] * raw["alpha"]))

    restored = t3c.raw_from_composite(phi, alpha_gauge=raw["alpha"])
    for name, value in raw.items():
        assert restored[name] == pytest.approx(value)

    # T_v = tau * alpha is underdetermined without an explicit alpha gauge.
    with pytest.raises((TypeError, ValueError)):
        t3c.raw_from_composite(phi)  # type: ignore[call-arg]


def test_composite_inverse_rejects_induced_raw_bound_violations_without_clipping() -> None:
    raw = _raw_parameters()
    phi = t3c.composite_from_raw(raw)
    bounds = {
        "beta": (0.25, 4.0),
        "gamma": (0.1, 1.0),
        "kappa": (0.2, 1.5),
        "tau": (0.5, 5.0),
        "alpha": (0.1, 0.8),
    }

    valid = t3c.raw_from_composite(
        phi,
        alpha_gauge=raw["alpha"],
        raw_bounds=bounds,
    )
    for name, (lower, upper) in bounds.items():
        assert lower <= valid[name] <= upper

    out_of_bounds = dict(phi)
    out_of_bounds["log_gain"] += log(10.0)
    with pytest.raises(ValueError):
        t3c.raw_from_composite(
            out_of_bounds,
            alpha_gauge=raw["alpha"],
            raw_bounds=bounds,
        )


def test_normal_normal_partial_pool_has_correct_shrinkage_limits() -> None:
    local = np.asarray([-1.0, 1.5], dtype=np.float64)
    variance = np.asarray([0.25, 4.0], dtype=np.float64)
    population_mean = 0.2
    population_variance = 1.0

    result = t3c.normal_normal_partial_pool(
        local,
        variance,
        population_mean=population_mean,
        population_variance=population_variance,
    )
    weight = population_variance / (population_variance + variance)
    expected_mean = population_mean + weight * (local - population_mean)
    expected_variance = population_variance * variance / (population_variance + variance)
    np.testing.assert_allclose(result["shrinkage_weight"], weight)
    np.testing.assert_allclose(result["posterior_mean"], expected_mean)
    np.testing.assert_allclose(result["posterior_variance"], expected_variance)
    assert np.all(np.abs(result["posterior_mean"] - population_mean) < np.abs(local - population_mean))

    collapsed = t3c.normal_normal_partial_pool(
        local,
        variance,
        population_mean=population_mean,
        population_variance=0.0,
    )
    np.testing.assert_allclose(collapsed["posterior_mean"], population_mean)
    np.testing.assert_allclose(collapsed["posterior_variance"], 0.0)
    np.testing.assert_allclose(collapsed["shrinkage_weight"], 0.0)

    unpooled = t3c.normal_normal_partial_pool(
        local,
        variance,
        population_mean=population_mean,
        population_variance=1.0e12,
    )
    np.testing.assert_allclose(unpooled["posterior_mean"], local, rtol=1.0e-6, atol=1.0e-9)
    np.testing.assert_allclose(unpooled["posterior_variance"], variance, rtol=1.0e-6, atol=1.0e-9)
    np.testing.assert_allclose(unpooled["shrinkage_weight"], 1.0, rtol=1.0e-6, atol=1.0e-9)


def test_admission_fails_closed_before_measured_array_access() -> None:
    config = t3c.load_config(t3c.DEFAULT_CONFIG_PATH)
    result = t3c.evaluate_admission(config, {}, {}, {})

    assert result["decision"] == "BLOCKED_PREREQUISITE"
    assert result["required_met"] is False
    assert result["measured_array_access_count"] == 0


def test_frozen_step2_step3_evidence_blocks_only_the_six_scientific_prerequisites() -> None:
    config = t3c.load_config(t3c.DEFAULT_CONFIG_PATH)
    evidence = {
        name: json.loads((t3c.REPO_ROOT / item["path"]).read_text(encoding="utf-8"))
        for name, item in config["sources"].items()
    }
    result = t3c.evaluate_admission(
        config,
        evidence["step2_summary"],
        evidence["step3_summary"],
        evidence["step3_fold_calibration"],
        step2_manifest=evidence["step2_manifest"],
        step3_manifest=evidence["step3_manifest"],
    )

    assert result["decision"] == "BLOCKED_PREREQUISITE"
    assert result["blockers"] == [
        "t_p2_identifiability_supported",
        "cross_subject_stability_failure_in_common_gauge",
        "common_observation_and_driver_gauge_frozen",
        "fixed_local_fnirs_endpoint_frozen",
        "composite_synthetic_sbc_profile_multistart_passed",
        "practical_margin_frozen_before_measured_scoring",
    ]
    assert sum(row["met"] for row in result["checks"]) == 2
    assert result["measured_metadata_access_count"] == result["measured_array_access_count"] == 0
