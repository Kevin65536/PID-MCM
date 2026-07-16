import numpy as np

from experiments.evaluate_adaptive_shared_neural_ssm import _local_eeg_indices
from experiments.evaluate_shared_neural_driver_unified import Trial
from src.inference.adaptive_neurovascular_ssm import (
    HemodynamicParameters,
    apply_adaptive_ssm,
    continuous_hemodynamic_matrices,
    fit_adaptive_ssm,
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
