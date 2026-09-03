from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

import experiments.evaluate_t3_multisession_loso as step3
from experiments.evaluate_shared_neural_driver_unified import Trial
from experiments.evaluate_t3_measured_reconstruction_null import PreparedTrial
from src.inference.t3a_balloon_robust_ssm import BalloonConfig, BalloonObservationSpec, BalloonParameters


def test_config_and_loader_view_reject_closed_subjects_before_work(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config = step3.load_config(step3.DEFAULT_CONFIG_PATH)
    view = step3._loader_config(config)
    assert [item["record_id"] for item in view["data"]["conditions"]] == list(step3.SESSION_IDS)
    assert all(item["subjects"] == [f"subject_{index:02d}" for index in range(1, 19)] for item in view["data"]["conditions"])
    assert not any(set(item["subjects"]) & {f"subject_{index:02d}" for index in range(19, 30)} for item in view["data"]["conditions"])

    invalid = deepcopy(config)
    invalid["experiment"]["protected_data_enabled"] = True
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("experiment work must not start")

    monkeypatch.setattr(step3, "_synthetic_preflight", forbidden)
    with pytest.raises(ValueError, match="boundary"):
        step3.run(invalid, tmp_path / "run")
    assert not called


def test_metadata_gate_is_array_free_and_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    config = step3.load_config(step3.DEFAULT_CONFIG_PATH)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("metadata validation must not load an array")

    monkeypatch.setattr(step3.CleanPhysiologyCacheIndex, "load_record_arrays", forbidden)
    summary, rows, _ = step3._validate_metadata(config)
    assert summary["metadata_only"]
    assert summary["selected_record_count"] == 54
    assert summary["selected_all_event_count"] == 1080
    assert summary["selected_target_trial_count"] == len(rows) == 540
    assert summary["minimum_next_event_margin_s"]["fnirs"] >= 0.5
    assert summary["minimum_next_event_margin_s"]["eeg"] >= 0.5
    assert {(row["subject"], row["record_id"]) for row in rows} == {
        (f"subject_{subject:02d}", session)
        for subject in range(1, 19) for session in step3.SESSION_IDS
    }


def test_loso_split_and_inventory_have_no_session_overlap() -> None:
    folds = step3._folds()
    assert len(folds) == 3
    for fold in folds:
        assert len(fold["train_sessions"]) == 2
        assert fold["heldout_session"] not in fold["train_sessions"]
        assert set(fold["train_sessions"]) | {fold["heldout_session"]} == set(step3.SESSION_IDS)
    metadata = [
        {"sample_id": session, "subject": "subject_01", "record_id": session, "event_index": 1}
        for session in step3.SESSION_IDS
    ]
    inventory = step3._trial_inventory(metadata)
    assert len(inventory) == 9
    assert sum(row["role"] == "fit" for row in inventory) == 6
    assert sum(row["role"] == "heldout" for row in inventory) == 3
    assert all(not row["used_for_parameter_fit"] for row in inventory if row["role"] == "heldout")


def test_target_masks_cover_response_but_score_recovery_only() -> None:
    config = step3.load_config(step3.DEFAULT_CONFIG_PATH)
    masks = step3._time_masks(config, 300)
    assert int(masks["heldout_input"].sum()) == 250
    assert int(masks["recovery"].sum()) == 150
    assert int(masks["response"].sum()) == 250
    assert np.all(~masks["recovery"] | masks["heldout_input"])
    assert not masks["heldout_input"][:50].any()
    assert masks["heldout_input"][50:].all()


def test_scalar_session_decomposition_is_zero_sum_and_never_sees_heldout(monkeypatch: pytest.MonkeyPatch) -> None:
    targets = {"session_01": math_log(0.50), "session_03": math_log(1.00)}
    seen = []

    def fake_smoother(observations, *, parameters, **_kwargs):
        target = float(np.asarray(observations)[0, 0])
        seen.append(target)
        log_kappa = math_log(parameters.free.kappa)
        return SimpleNamespace(
            predictive_log_likelihood=-100.0 * (log_kappa - target) ** 2,
            physical_checks={"finite": True, "positive": True},
        )

    monkeypatch.setattr(step3, "smooth_balloon", fake_smoother)
    result = step3._fit_subject_models({
        "subject": "subject_01",
        "fold_id": "holdout_session_05",
        "heldout_session": "session_05",
        "train_sessions": ("session_01", "session_03"),
        "train_by_session": {
            session: (np.asarray([[target, 0.0, 0.0]]),) for session, target in targets.items()
        },
        "heldout_trials": tuple(range(10)),
        "base_parameters": BalloonParameters(),
        "observation_spec": BalloonObservationSpec().resolved(BalloonParameters().fixed),
        "balloon_config": BalloonConfig(),
        "fit_config": {
            "bounds": (0.2, 1.5),
            "prior_mean": 0.64,
            "prior_sd": 0.2,
            "grid_points": 9,
            "max_iterations": 50,
            "xatol": 1e-5,
            "boundary_fraction": 0.01,
        },
    })
    row = result["parameter_row"]
    assert row["session_a_log_deviation"] == pytest.approx(-row["session_b_log_deviation"])
    assert row["session_center_kappa"] == pytest.approx(np.sqrt(row["session_a_kappa"] * row["session_b_kappa"]))
    assert row["heldout_fit_calls"] == 0
    assert set(seen) == set(targets.values())


def math_log(value: float) -> float:
    return float(np.log(value))


def test_heldout_scoring_masks_fnirs_and_only_applies_frozen_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    config = step3.load_config(step3.DEFAULT_CONFIG_PATH)
    base = BalloonParameters()
    parameter_row = {
        "fold_id": "holdout_session_05",
        "heldout_session": "session_05",
        "subject": "subject_01",
        "all_optimizer_success": True,
    }
    monkeypatch.setattr(step3, "_fit_subject_models", lambda _task: {
        "parameter_row": parameter_row,
        "optimizer_rows": [],
        "shared_parameters": base,
        "session_center_parameters": base,
    })
    observed_inputs = []

    def fake_smoother(observations, **_kwargs):
        values = np.asarray(observations, dtype=np.float64)
        observed_inputs.append(values.copy())
        return SimpleNamespace(
            state_mean=np.column_stack((np.linspace(0.0, 1.0, 300), np.zeros((300, 5)))),
            observation_mean=np.nan_to_num(values),
            total_variance=np.ones((300, 3)),
            physical_checks={"finite": True},
        )

    monkeypatch.setattr(step3, "smooth_balloon", fake_smoother)
    trial = Trial(
        condition_id="single_trial_ma_session_05",
        dataset_id="eeg_fnirs_single_trial",
        subject="subject_01",
        record_id="session_05",
        event_index=1,
        eeg=np.zeros((6000, 1)),
        fnirs=np.zeros((300, 2)),
        fnirs_channel_names=("x_HbO", "x_HbR"),
        fnirs_roles=("HbO", "HbR"),
        eeg_artifact_fraction=0.0,
    )
    heldout = PreparedTrial(trial, np.ones(300), np.ones(300), -np.ones(300))
    result = step3._fit_and_score_subject({
        "subject": "subject_01",
        "fold_id": "holdout_session_05",
        "heldout_session": "session_05",
        "train_sessions": ("session_01", "session_03"),
        "train_by_session": {"session_01": (), "session_03": ()},
        "heldout_trials": (heldout,),
        "base_parameters": base,
        "observation_spec": BalloonObservationSpec().resolved(base.fixed),
        "balloon_config": BalloonConfig(),
        "fit_config": {},
        "config": config,
    })
    assert len(observed_inputs) == 3
    assert all(np.isfinite(values[:50]).all() for values in observed_inputs)
    assert all(np.isnan(values[50:, 1:]).all() for values in observed_inputs)
    assert all(np.isfinite(values[:, 0]).all() for values in observed_inputs)
    assert all(row["target_observed_by_smoother"] is False for row in result["metric_rows"])
    assert np.isfinite(heldout.hbo).all() and np.isfinite(heldout.hbr).all()


def test_subject_block_bootstrap_is_reproducible() -> None:
    values = {f"subject_{index:02d}": value for index, value in enumerate(np.linspace(-1.0, 1.0, 18), start=1)}
    first = step3._bootstrap_mean_ci(values.values(), seed=7, replicates=1000, confidence=0.95)
    second = step3._bootstrap_mean_ci(values.values(), seed=7, replicates=1000, confidence=0.95)
    assert first == second
    assert first["lower"] < first["mean"] < first["upper"]
