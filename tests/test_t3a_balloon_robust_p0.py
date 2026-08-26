"""Executable contract checks for the synthetic-only T3a P0 suite."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from experiments.evaluate_t3a_balloon_robust_p0 import (
    _apply_smoke_overrides,
    _gate_results,
    _make_cases,
    _prior_predictive,
    load_config,
    run_suite,
    validate_config,
)


def _small_config() -> dict:
    config = load_config()
    config["simulation"].update({
        "duration_s": 6.0,
        "replicates": 2,
        "prior_predictive_draws": 2,
        "severity_levels": [0.0, 1.0],
    })
    config["inference"].update({
        "multistarts": 1,
        "starts": [config["inference"]["starts"][0]],
        "max_iterations": 1,
        "irls_iterations": 1,
        "profile_points": 1,
    })
    return config


def test_config_fails_closed_before_measured_or_invalid_physiology() -> None:
    config = load_config()
    config["experiment"]["measured_data_enabled"] = True
    with pytest.raises(ValueError, match="measured data"):
        validate_config(config)

    config = load_config()
    config["physiology"]["fixed"]["gamma"] = float("nan")
    with pytest.raises(ValueError, match="gamma/p0/q0"):
        validate_config(config)


def test_cases_use_prior_drawn_truth_and_paired_corruptions() -> None:
    bases, cases = _make_cases(_small_config())
    assert len(bases) == 2
    assert bases[0].true_parameters["kappa_per_s"] != bases[1].true_parameters["kappa_per_s"]

    spike = next(case for case in cases if case.replicate_id == 0 and case.stress_case == "spike")
    np.testing.assert_allclose(spike.clean, bases[0].clean)
    np.testing.assert_allclose(spike.truth_states, bases[0].truth_states)
    np.testing.assert_allclose(spike.observations - bases[0].observations, spike.artifact, equal_nan=True)

    missing = next(case for case in cases if case.replicate_id == 0 and case.stress_case == "missing_fnirs")
    assert missing.observation_mask[:, 0].all()
    assert not missing.observation_mask[:, 1:].any()


def test_prior_predictive_records_hbo_response_lag() -> None:
    config = _small_config()
    rows = _prior_predictive(config, 2, 77)
    response = [row for row in rows if row["parameter_name"] == "response_amplitude"]
    assert len(response) == 2
    assert all(row["response_target"] == "HbO" for row in response)
    assert all(0.0 <= row["delay_s"] <= 20.0 for row in response)
    assert all(row["max"] > row["min"] for row in response)


def test_missing_gate_evidence_and_smoke_cannot_qualify() -> None:
    config = _apply_smoke_overrides(load_config())
    gates = _gate_results(
        config,
        [],
        [],
        [],
        [],
        [],
        available_models={"T3a-balloon-robust": True},
    )
    assert gates["T-P0"]["status"] == "INCONCLUSIVE"
    assert all(gates[name]["status"] != "PASS" for name in ("T-P1", "T-P2", "T-P3", "synthetic-T-G4"))


def test_truncated_prior_predictive_evidence_cannot_pass() -> None:
    config = load_config()
    rows = _prior_predictive(config, 1, 91)
    gates = _gate_results(
        config,
        rows,
        [],
        [],
        [],
        [],
        available_models={"T3a-balloon-robust": True},
    )
    assert gates["T-P0"]["checks"]["prior_draw_support"] is False
    assert gates["T-P1"]["checks"]["response_support"] is False
    assert gates["T-P0"]["status"] != "PASS"
    assert gates["T-P1"]["status"] != "PASS"


def test_small_suite_publishes_complete_contract(tmp_path) -> None:
    output = run_suite(_small_config(), tmp_path / "run")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["completion_status"] == "complete"
    assert summary["models"]["T3a-balloon-robust"] is True
    assert summary["models"]["T2b-adaptive-legacy"] is True
    for name in (
        "trajectories.csv",
        "states.csv",
        "uncertainty.csv",
        "parameter_recovery.csv",
        "calibration.csv",
        "null_metrics.csv",
        "physical_checks.csv",
        "prior_predictive.csv",
        "profile_likelihood.csv",
        "gates.json",
        "manifest.json",
        "resolved_config.yaml",
    ):
        assert (output / name).is_file()
    with (output / "trajectories.csv").open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        assert any(
            row["model_id"] == "T2b-adaptive-legacy"
            and row["scenario_id"] == "clean"
            and np.isfinite(float(row["posterior_mean"]))
            for row in rows
        )
