import numpy as np
import pytest

from experiments.evaluate_adaptive_shared_neural_ssm import (
    AdaptivePrediction,
    _aggregate_metrics,
    _downsample_valid_mask,
    _local_eeg_indices,
    _prediction_metrics,
    _trial_valid_masks,
)
from experiments.evaluate_shared_neural_driver_unified import Trial
from src.inference.adaptive_neurovascular_ssm import (
    HemodynamicParameters,
    apply_adaptive_ssm,
    continuous_hemodynamic_matrices,
    fit_adaptive_ssm,
    measurement_aligned_state_gauge,
    rts_smoother,
    simulate_hemodynamics,
)
from src.inference.neurovascular_smc import NeurovascularSMCFilter


def test_linearized_hemodynamics_are_stable_and_delayed():
    drift, _ = continuous_hemodynamic_matrices(HemodynamicParameters())
    assert np.all(np.real(np.linalg.eigvals(drift)) < 0.0)
    impulse = np.zeros(200)
    impulse[0] = 1.0
    states = simulate_hemodynamics(impulse, HemodynamicParameters())
    hbo_peak = int(np.argmax(np.abs(states[:, 2])))
    assert hbo_peak > 5
    assert np.max(np.abs(states[:, 2])) > 0.0


def test_joint_smoother_recovers_delayed_signal_without_erasing_eeg():
    time = np.arange(200) / 10.0
    driver = np.sin(2.0 * np.pi * 0.08 * time) + 0.25 * np.sin(2.0 * np.pi * 0.16 * time)
    states = simulate_hemodynamics(driver, HemodynamicParameters())
    hbo = 20.0 * states[:, 2]
    hbr = -10.0 * states[:, 3]
    fit = fit_adaptive_ssm(
        [driver, 0.9 * driver],
        [hbo, 0.9 * hbo],
        [hbr, 0.9 * hbr],
        max_iterations=8,
        q_scale_candidates=(1.0,),
        fnirs_noise_scale_candidates=(0.5, 1.0),
    )
    result = apply_adaptive_ssm(driver, fit, hbo_observation=hbo, hbr_observation=hbr)
    assert np.corrcoef(hbo, result.hbo_reconstructed)[0, 1] > 0.95
    assert np.var(result.hbo_reconstructed) / np.var(hbo) > 0.6
    assert np.corrcoef(driver, result.eeg_reconstructed)[0, 1] > 0.9
    assert np.min(1.0 + result.states[:, 1]) > 0.0
    assert result.observation_predictive_std.shape == (len(driver), 3)
    assert np.all(np.isfinite(result.observation_predictive_std))
    assert np.all(result.observation_predictive_std > 0.0)
    assert np.isfinite(result.predictive_log_likelihood)

    gauge = measurement_aligned_state_gauge(result, fit)
    np.testing.assert_allclose(gauge.states[:, 2], result.hbo_reconstructed, atol=1e-10)
    np.testing.assert_allclose(gauge.states[:, 3], result.hbr_reconstructed, atol=1e-10)
    np.testing.assert_allclose(
        gauge.state_std[:, 2],
        result.state_std[:, 2] * abs(fit.hbo_gain * fit.hbo_std),
    )
    assert gauge.reconstruction_max_abs_delta < 1e-10
    # The synthetic HbR observation intentionally uses a negative gain.  The
    # observation-aligned target must absorb that sign without changing output.
    assert gauge.scales[3] < 0.0

    normalized_hbo_std = result.observation_predictive_std[:, 1] / fit.hbo_std
    normalized_hbr_std = result.observation_predictive_std[:, 2] / fit.hbr_std
    assert np.all(normalized_hbo_std > 0.0)
    assert np.all(normalized_hbr_std > 0.0)

    fnirs_only = apply_adaptive_ssm(
        None,
        fit,
        hbo_observation=hbo,
        hbr_observation=hbr,
        observation_mode="fnirs_only",
    )
    fnirs_only_with_length_reference = apply_adaptive_ssm(
        driver,
        fit,
        hbo_observation=hbo,
        hbr_observation=hbr,
        observation_mode="fnirs_only",
    )
    eeg_only = apply_adaptive_ssm(driver, fit, observation_mode="eeg_only")
    assert fnirs_only.hbo_reconstructed.shape == hbo.shape
    assert fnirs_only.observation_mode == "fnirs_only"
    assert result.observation_mode == "joint"
    assert np.all(np.isfinite(fnirs_only.hbo_reconstructed))
    assert np.all(np.isfinite(fnirs_only.observation_predictive_std))
    np.testing.assert_allclose(
        fnirs_only.states,
        fnirs_only_with_length_reference.states,
    )
    partial_hbo = hbo.copy()
    partial_hbr = hbr.copy()
    partial_hbo[4:7] = np.nan
    partial_hbr[9:11] = np.nan
    partial = apply_adaptive_ssm(
        None,
        fit,
        hbo_observation=partial_hbo,
        hbr_observation=partial_hbr,
        observation_mode="fnirs_only",
    )
    assert np.all(np.isfinite(partial.states))
    assert np.all(np.isfinite(partial.observation_predictive_std))
    # The modality-specific smoothers use genuinely different observation
    # updates; they are not aliases of joint smoothing.
    assert not np.allclose(fnirs_only.states, result.states)
    assert not np.allclose(eeg_only.states, result.states)
    with pytest.raises(ValueError, match="requires both"):
        apply_adaptive_ssm(driver, fit, observation_mode="fnirs_only")
    with pytest.raises(ValueError, match="only for fnirs_only"):
        apply_adaptive_ssm(None, fit, observation_mode="eeg_only")

    normalized_observations = np.column_stack(
        (
            driver,
            (hbo - fit.hbo_mean) / fit.hbo_std,
            (hbr - fit.hbr_mean) / fit.hbr_std,
        )
    )
    _, expected_state_std, smoothed_cov, _ = rts_smoother(
        normalized_observations,
        fit.transition,
        fit.process_cov,
        fit.observation,
        fit.observation_cov,
        fit.initial_cov,
    )
    expected_predictive_std = np.sqrt(
        np.maximum(
            np.einsum(
                "oi,tij,oj->to",
                fit.observation,
                smoothed_cov,
                fit.observation,
                optimize=True,
            )
            + np.diag(fit.observation_cov)[None, :],
            0.0,
        )
    )
    expected_predictive_std[:, 1] *= abs(fit.hbo_std)
    expected_predictive_std[:, 2] *= abs(fit.hbr_std)
    np.testing.assert_allclose(result.state_std, expected_state_std)
    np.testing.assert_allclose(
        result.observation_predictive_std,
        expected_predictive_std,
    )


def test_local_protocol_selects_nearest_six_eeg_channels():
    eeg_positions = np.column_stack((np.arange(8, dtype=float), np.zeros(8), np.zeros(8)))
    fnirs_positions = np.asarray([[0.1, 0.0, 0.0], [6.9, 0.0, 0.0]])
    trial = Trial(
        condition_id="test",
        dataset_id="test",
        subject="s",
        record_id="r",
        event_index=0,
        eeg=np.zeros((20, 8)),
        fnirs=np.zeros((10, 2)),
        fnirs_channel_names=("a_HbO", "b_HbO"),
        fnirs_roles=("HbO", "HbO"),
        eeg_artifact_fraction=0.0,
        eeg_channel_names=tuple(f"e{index}" for index in range(8)),
        eeg_positions=eeg_positions,
        fnirs_positions=fnirs_positions,
    )
    selected = _local_eeg_indices(trial, np.asarray([0, 1]), count=6)
    assert len(selected) == 6
    assert {0, 1, 6, 7}.issubset(set(selected.tolist()))


def test_current_ssm_runner_fails_closed_on_partial_window_support():
    mask = np.ones(40, dtype=bool)
    mask[5] = False
    trial = Trial(
        condition_id="test",
        dataset_id="test",
        subject="s",
        record_id="r",
        event_index=0,
        eeg=np.zeros((40, 2)),
        fnirs=np.zeros((2, 2)),
        fnirs_channel_names=("a_HbO", "a_HbR"),
        fnirs_roles=("HbO", "HbR"),
        eeg_artifact_fraction=0.0,
        eeg_valid_mask=mask,
        fnirs_valid_mask=np.ones(2, dtype=bool),
    )

    np.testing.assert_array_equal(
        _downsample_valid_mask(mask, 2),
        np.asarray([False, True]),
    )
    with pytest.raises(RuntimeError, match="fully supported"):
        _trial_valid_masks(trial)


def test_runner_exposes_separate_deviation_and_predictive_diagnostics():
    points = 20
    time = np.linspace(0.0, 2.0 * np.pi, points)
    truth_hbo = np.sin(time)
    truth_hbr = -0.5 * np.sin(time)
    eeg = np.cos(time)
    states = np.zeros((points, 5), dtype=np.float64)
    states[:, 4] = eeg
    prediction = AdaptivePrediction(
        condition_id="test",
        dataset_id="test",
        subject="s1",
        heldout_trial=0,
        model="adaptive_eeg_only",
        spatial_mode="local",
        truth_hbo=truth_hbo,
        estimate_hbo=truth_hbo + 0.1,
        truth_hbr=truth_hbr,
        estimate_hbr=truth_hbr - 0.05,
        eeg_observation=eeg,
        eeg_reconstruction=eeg + 0.2,
        observation_predictive_std=np.full((points, 3), 0.25),
        eeg_valid_mask=np.ones(points, dtype=bool),
        fnirs_valid_mask=np.ones(points, dtype=bool),
        states=states,
        state_std=np.full_like(states, 0.1),
        target_states=states.copy(),
        target_state_std=np.full_like(states, 0.1),
        gauge_scales=np.ones(5),
        gauge_offsets=np.zeros(5),
        gauge_reconstruction_max_abs_delta=0.0,
        selected_fnirs_channels=("a_HbO", "a_HbR"),
        selected_eeg_channels=("e1",),
    )

    metrics = _prediction_metrics(prediction, baseline_n=5)
    assert metrics["trajectory_deviation_nrmse"] > 0.0
    assert metrics["posterior_predictive_sd_mean"] == pytest.approx(0.25)
    assert metrics["predictive_valid_point_count"] == points
    assert metrics["hbr_predictive_valid_point_count"] == points
    assert metrics["eeg_predictive_valid_point_count"] == points


def test_subject_equal_aggregation_and_bootstrap_are_deterministic():
    def row(subject: str, heldout: int, value: float) -> dict[str, object]:
        return {
            "condition_id": "condition",
            "dataset_id": "dataset",
            "subject": subject,
            "validation": "leave_one_trial",
            "heldout_trial": heldout,
            "model": "adaptive_eeg_only",
            "spatial_mode": "local",
            "selected_fnirs_channels": "a_HbO|a_HbR",
            "selected_eeg_channels": "e1",
            "trajectory_deviation_nrmse": value,
        }

    folds = [row("s1", 0, 1.0), row("s1", 1, 3.0), row("s2", 0, 9.0)]
    subjects, summary = _aggregate_metrics(
        folds,
        bootstrap_iterations=200,
        seed=17,
    )
    repeated_subjects, repeated_summary = _aggregate_metrics(
        folds,
        bootstrap_iterations=200,
        seed=17,
    )

    assert [item["trajectory_deviation_nrmse"] for item in subjects] == [2.0, 9.0]
    assert summary[0]["trajectory_deviation_nrmse"] == pytest.approx(5.5)
    assert subjects == repeated_subjects
    assert summary == repeated_summary


def test_particle_filter_carries_weights_without_resampling():
    model = NeurovascularSMCFilter(
        hrf_kernel=np.ones(1),
        state_transition_matrix=np.asarray([[0.99]]),
        process_noise_cov=np.asarray([[0.02]]),
        eeg_forward=np.ones((1, 1)),
        fnirs_forward=np.ones((1, 1)),
        eeg_noise_cov=np.asarray([[0.20]]),
        fnirs_noise_cov=np.asarray([[1e9]]),
        n_particles=12_000,
        resample_threshold=0.0,
        seed=1,
    )
    result = model.filter(
        np.asarray([[2.0], [0.0]]),
        np.zeros((2, 1)),
        return_particles=False,
    )
    # The second posterior must retain evidence from the first observation.
    # Resetting importance weights at every step makes this value near zero.
    assert result.state_mean[1, 0] > 0.45
