from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from experiments.evaluate_t3_measured_reconstruction_null import (
    PreparedTrial,
    STAGE_RECOMMENDATION_CONTRACT,
    Trial,
    _finite_hessian_bounded,
    _fit_subject_stage,
    _from_optimizer_coordinate,
    _matched_null_metrics,
    _masked_metrics,
    _null_inputs,
    _parameter_values,
    _physical_check_status,
    _replace_parameter_values,
    _shift_non_circular,
    _to_optimizer_coordinate,
    _write_json,
    load_config,
    run,
    validate_config,
)
from experiments.scripts.render_t3_measured_parameter_effects import (
    _curve_rows,
    _require_source,
    _stage_vectors,
)
from src.inference.t3a_balloon_robust_ssm import BalloonConfig, BalloonObservationSpec, BalloonParameters


CONFIG = Path("experiments/configs/physiology_semantic_tokenizer/t3_measured_reconstruction_null_v1.yaml")


def test_measured_boundary_and_non_circular_null(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    stages = config["ssm"]["t3a"]["parameter_fit"]["stages"]
    assert tuple((stage["id"], stage["recommendation_eligible"]) for stage in stages) == STAGE_RECOMMENDATION_CONTRACT

    invalid_stage = deepcopy(config)
    invalid_stage["ssm"]["t3a"]["parameter_fit"]["stages"][4]["recommendation_eligible"] = True
    with pytest.raises(ValueError, match="only M0"):
        run(invalid_stage, tmp_path)

    invalid = deepcopy(config)
    invalid["data"]["conditions"][0]["subjects"].append("subject_24")
    with pytest.raises(ValueError, match="loader subjects"):
        validate_config(invalid)

    shifted = _shift_non_circular(np.arange(5.0), 2)
    assert np.isnan(shifted[:2]).all()
    np.testing.assert_array_equal(shifted[2:], np.arange(3.0))


def test_null_pair_and_donor_use_the_same_finite_support() -> None:
    paired, donor = _matched_null_metrics(
        np.asarray([0.0, 1.0, 2.0, 3.0]),
        np.asarray([np.nan, np.nan, 0.0, 1.0]),
        np.zeros(4),
        np.ones(4),
    )

    assert paired["n"] == donor["n"] == 2
    assert paired["rmse"] == pytest.approx(np.sqrt(6.5))
    assert donor["rmse"] == pytest.approx(np.sqrt(0.5))


def test_metrics_use_the_evaluation_mask_and_flag_low_variance() -> None:
    metrics = _masked_metrics(
        np.asarray([0.0, 100.0, 10.0, 12.0]),
        np.asarray([0.0, 100.0, 9.0, 11.0]),
        np.asarray([False, False, True, True]),
        np.ones(4),
    )
    assert metrics["nrmse"] == pytest.approx(1.0)
    assert metrics["observed_temporal_sd"] == pytest.approx(1.0)
    assert not metrics["low_observed_variance"]

    constant = _masked_metrics(
        np.asarray([2.0, 2.0]),
        np.asarray([1.0, 1.0]),
        np.ones(2, dtype=bool),
    )
    assert constant["low_observed_variance"]
    assert np.isnan(constant["nrmse"])


def test_null_donor_identities_are_explicit() -> None:
    def prepared(subject: str, event_index: int) -> PreparedTrial:
        trial = Trial(
            condition_id="single_trial_ma_session_01",
            dataset_id="eeg_fnirs_single_trial",
            subject=subject,
            record_id="session_01",
            event_index=event_index,
            eeg=np.zeros((5, 1)),
            fnirs=np.zeros((5, 2)),
            fnirs_channel_names=("x_HbO", "x_HbR"),
            fnirs_roles=("HbO", "HbR"),
            eeg_artifact_fraction=0.0,
        )
        values = np.arange(5.0) + event_index
        return PreparedTrial(trial, values, values + 10.0, values - 10.0)

    trials = [
        prepared("subject_19", 1),
        prepared("subject_19", 2),
        prepared("subject_20", 1),
        prepared("subject_20", 2),
    ]
    pairing = _null_inputs(trials, "pairing", 2)[0]
    independent = _null_inputs(trials, "independent", 2)[0]
    shifted = _null_inputs(trials, "time_shift", 2)[0]
    assert (pairing.donor_subject, pairing.donor_event_index) == ("subject_19", 2)
    assert (independent.donor_subject, independent.donor_event_index) == ("subject_20", 1)
    assert (shifted.donor_subject, shifted.donor_event_index) == ("subject_19", 1)
    assert np.isnan(shifted.donor_hbo[:2]).all()


def test_manifest_json_publication_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    _write_json(path, {"completion_status": "complete"})
    assert '"completion_status": "complete"' in path.read_text()
    assert not (tmp_path / ".manifest.json.tmp").exists()


def test_physical_check_status_distinguishes_gates_from_ranges() -> None:
    assert _physical_check_status(True) == "pass"
    assert _physical_check_status(False) == "fail"
    assert _physical_check_status(0.42) == "diagnostic"


def test_parameter_fit_coordinates_and_local_hessian() -> None:
    parameters = _replace_parameter_values(
        BalloonParameters(),
        {"beta": 1.5, "kappa": 0.7, "tau": 2.5, "gamma": 0.4, "alpha": 0.3, "E0": 0.35},
    )
    assert parameters.fixed.neurovascular_gain == pytest.approx(1.5)
    assert parameters.free.kappa == pytest.approx(0.7)
    for name, value in (("beta", 1.5), ("E0", 0.35)):
        assert _from_optimizer_coordinate(name, _to_optimizer_coordinate(name, value)) == pytest.approx(value)
    optimum = np.asarray([0.0, 0.0])
    hessian = _finite_hessian_bounded(
        optimum,
        lambda vector: float(vector[0] ** 2 + 2.0 * vector[1] ** 2),
        ((-1.0, 1.0), (-1.0, 1.0)),
        0.02,
    )
    np.testing.assert_allclose(hessian, np.diag([2.0, 4.0]), atol=1e-8)


def test_boundary_fit_reports_status_without_covariance_index_error() -> None:
    config = load_config(CONFIG)["ssm"]["t3a"]["parameter_fit"]
    fit_config = dict(config)
    fit_config.update(optimizer_starts=1, optimizer_max_iterations=1)
    specs = deepcopy(config["parameters"])
    specs["beta"] = {**specs["beta"], "prior_mean": 1.0, "bounds": [1.0, 1.1]}
    result = _fit_subject_stage({
        "subject": "subject_test",
        "split": "fit",
        "fit_scope": "unit_test",
        "stage": {"id": "boundary", "label_zh": "边界", "free": ["beta"]},
        "parameter_specs": specs,
        "base_parameters": BalloonParameters(),
        "observation_spec": BalloonObservationSpec().resolved(BalloonParameters().fixed),
        "balloon_config": BalloonConfig(),
        "fit_config": fit_config,
        "train_trials": (),
        "heldout_trials": (),
        "center_mask": np.zeros(5, dtype=bool),
        "initial_values": {"beta": 1.0},
    })
    beta = next(row for row in result["parameter_rows"] if row["parameter"] == "beta")
    assert beta["boundary_status"] == "BOUNDARY"
    assert np.isnan(beta["posterior_sd_laplace"])


def test_parameter_effect_renderer_keeps_stage_and_curve_identity() -> None:
    base = BalloonParameters()
    base_values = _parameter_values(base)
    rows = []
    for subject, kappa in (("subject_01", 0.4), ("subject_02", 0.8)):
        for name, value in base_values.items():
            rows.append({
                "subject": subject,
                "split": "fit",
                "fit_scope": "eight_trials_fit_two_trials_internal_holdout",
                "stage": "M1_kappa",
                "parameter": name,
                "estimate": kappa if name == "kappa" else value,
                "is_free": name == "kappa",
            })
    population, subjects = _stage_vectors(base, rows, "M1_kappa")
    assert population["kappa"] == pytest.approx(0.6)
    assert subjects["subject_01"]["kappa"] == pytest.approx(0.4)
    assert population["tau"] == pytest.approx(base_values["tau"])

    trial = Trial(
        condition_id="single_trial_ma_session_01",
        dataset_id="eeg_fnirs_single_trial",
        subject="subject_19",
        record_id="session_01",
        event_index=1,
        eeg=np.zeros((5, 1)),
        fnirs=np.zeros((5, 2)),
        fnirs_channel_names=("x_HbO", "x_HbR"),
        fnirs_roles=("HbO", "HbR"),
        eeg_artifact_fraction=0.0,
    )
    item = PreparedTrial(trial, np.arange(5.0), np.arange(5.0) + 1, np.arange(5.0) - 1)
    input_values = np.column_stack((item.eeg_driver, item.hbo, item.hbr))
    input_values[2, 1] = np.nan
    curve = _curve_rows(
        item,
        np.arange(5.0),
        np.asarray([False, False, True, False, False]),
        "test",
        "center_masked_fnirs",
        "HbO",
        "test curve",
        {"input": input_values, "estimate": np.zeros((5, 3)), "predictive_std": np.ones((5, 3))},
        population,
    )
    assert len(curve) == 5
    assert {row["target"] for row in curve} == {"HbO"}
    assert curve[2]["input_standardized"] is None


def test_parameter_effect_renderer_rejects_protected_source_before_loading(tmp_path: Path) -> None:
    _write_json(tmp_path / "manifest.json", {
        "schema": "t3_measured_reconstruction_null_v1",
        "analysis_kind": "staged_subject_parameter_fit",
        "completion_status": "complete",
        "protected_data_opened": True,
        "protected_data_enabled": False,
    })
    with pytest.raises(ValueError, match="protected data access"):
        _require_source(tmp_path)
