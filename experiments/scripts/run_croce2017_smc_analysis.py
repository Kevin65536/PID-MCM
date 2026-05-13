"""
Run Croce et al. 2017 SMC filter on continuous EEG-fNIRS data.

Produces:
    1. Filtered state trajectories (joint vs single-modality)
    2. Original vs reconstructed signal overlays
    3. Particle diagnostic plots (ESS, weights)
    4. Lag-correlation comparison (joint vs EEG-only vs fNIRS-only)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.data.registry import load_experiment_config, resolve_modality_preprocessing
from src.data.eeg_fnirs_dataset import MultiModalEEGfNIRSDataset
from src.inference.neurovascular_smc import (
    NeurovascularSMCFilter,
    build_model_from_data,
    double_gamma_hrf,
    normalize_observations,
    denormalize_observations,
    reduce_eeg_to_pc1,
    reconstruct_eeg_from_pc1,
)


OUT_DIR = Path('experiments/results/croce2017_smc_analysis_20260513')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_continuous_session(dataset, subject_id, session_idx):
    """Load continuous preprocessed EEG and fNIRS from one session."""
    eeg_raw_list, _, eeg_info = dataset._get_eeg_data(subject_id, processed=True)
    fnirs_raw_list, _, fnirs_info = dataset._get_nirs_data(subject_id, processed=True)

    eeg = np.asarray(eeg_raw_list[session_idx], dtype=np.float32)  # [T_eeg, E]
    fnirs = np.asarray(fnirs_raw_list[session_idx], dtype=np.float32)  # [T_fnirs, F]

    eeg_fs = float(eeg_info['fs'])
    fnirs_fs = float(fnirs_info['fs'])

    return eeg, fnirs, eeg_fs, fnirs_fs


def downsample_eeg_power(eeg: np.ndarray, eeg_fs: float, target_fs: float) -> np.ndarray:
    """Downsample EEG broadband power to target rate via avg_pool."""
    factor = int(round(eeg_fs / target_fs))
    if factor < 1:
        raise ValueError(f'Target fs {target_fs} > EEG fs {eeg_fs}')
    power = eeg ** 2  # [T, E]
    usable = (power.shape[0] // factor) * factor
    power = power[:usable]
    # [T, E] -> [T//factor, E]
    reshaped = power.reshape(usable // factor, factor, power.shape[1])
    return reshaped.mean(axis=1)  # [T_ds, E]


def align_lengths(eeg_power_ds, fnirs):
    """Trim to common length."""
    L = min(len(eeg_power_ds), len(fnirs))
    return eeg_power_ds[:L], fnirs[:L]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = load_experiment_config('source_observation/phase1/default.yaml')
    data_cfg = config['data']

    eeg_preproc = resolve_modality_preprocessing(data_cfg, 'eeg')
    fnirs_preproc = resolve_modality_preprocessing(data_cfg, 'fnirs')

    dataset = MultiModalEEGfNIRSDataset(
        data_root=data_cfg['data_root'],
        subject_ids=list(range(1, 30)),
        task=data_cfg.get('task', 'motor_imagery'),
        window_duration_s=float(data_cfg['window']['duration_s']),
        window_offset_ms=float(data_cfg['window'].get('offset_ms', 0.0)),
        normalize=False,
        normalization_mode='none',
        eeg_preprocessing=eeg_preproc,
        fnirs_preprocessing=fnirs_preproc,
        use_artifact_data=bool(data_cfg.get('use_artifact_data', True)),
        exclude_eog=bool(data_cfg.get('exclude_eog', True)),
        hbo_only=bool(data_cfg.get('hbo_only', True)),
        hbr_only=bool(data_cfg.get('hbr_only', False)),
    )

    # Pick the first session with decent length
    subject_id = 1
    eeg, fnirs, eeg_fs, fnirs_fs = load_continuous_session(dataset, subject_id, 0)

    print(f'Session: subject {subject_id}, session 0')
    print(f'  EEG:   {eeg.shape} @ {eeg_fs} Hz  ({eeg.shape[0] / eeg_fs:.1f} s)')
    print(f'  fNIRS: {fnirs.shape} @ {fnirs_fs} Hz  ({fnirs.shape[0] / fnirs_fs:.1f} s)')

    # Downsample EEG power to fNIRS rate
    eeg_power_ds = downsample_eeg_power(eeg, eeg_fs, fnirs_fs)
    eeg_power_ds, fnirs = align_lengths(eeg_power_ds, fnirs)
    T, E = eeg_power_ds.shape
    F = fnirs.shape[1]
    print(f'  Aligned at {fnirs_fs} Hz: EEG_power {eeg_power_ds.shape}, fNIRS {fnirs.shape}')

    # Build HRF kernel at fNIRS rate
    hrf_kernel = double_gamma_hrf(fnirs_fs, duration_s=24.0)
    print(f'  HRF kernel: {len(hrf_kernel)} samples ({len(hrf_kernel)/fnirs_fs:.1f} s)')

    # Normalise observations to put EEG power and fNIRS on comparable scales
    eeg_norm, fnirs_norm, norm_params = normalize_observations(eeg_power_ds, fnirs)
    print(f'  After normalisation: EEG power std={float(eeg_norm.std()):.2f}, '
          f'fNIRS std={float(fnirs_norm.std()):.2f}')

    # Reduce EEG to first PC, split into raw (observation) + slow (state proxy)
    eeg_pc1_raw, eeg_pc1_slow, eeg_loading, eeg_pc1_std = reduce_eeg_to_pc1(
        eeg_norm, fs_hz=fnirs_fs, lowpass_cutoff_hz=0.2,
    )
    # Variance retained in slow component
    slow_var = float(np.var(eeg_pc1_slow))
    fast_var = float(np.var(eeg_pc1_raw[:, 0] - eeg_pc1_slow[:, 0]))
    print(f'  EEG PC1: raw_var=1.00, slow_var={slow_var:.3f}, fast_var(R_eeg)={fast_var:.3f}')
    print(f'  PC loading std={float(eeg_loading.std()):.3f}')

    # ----- Build model & filter -----
    # Use raw PC1 as observation, slow PC1 for state dynamics estimation
    smc = build_model_from_data(eeg_pc1_raw, eeg_pc1_slow, fnirs_norm, hrf_kernel,
                                n_particles=1000, seed=42)

    print(f'\nModel parameters:')
    print(f'  State dim K = {smc.K_dim}')
    print(f'  AR(1) coef α = {float(smc.A[0,0]):.3f}')
    print(f'  Process noise Q = {float(smc.Q[0,0]):.6f}')
    print(f'  EEG noise = {float(smc.R_eeg[0,0]):.4f}')
    print(f'  fNIRS noise trace = {np.trace(smc.R_fnirs):.3f}')

    result = smc.filter(eeg_pc1_raw, fnirs_norm, return_particles=True)
    print(f'  Joint log-likelihood: {result.log_likelihood:.1f}')
    print(f'  Mean ESS: {np.mean(result.ess_history):.0f} / {smc.N}')

    # Reconstruct full-channel EEG from state estimate using PC1 loading
    eeg_recon_raw = reconstruct_eeg_from_pc1(
        result.state_mean, eeg_loading, eeg_pc1_std,
        norm_params.eeg_mean, norm_params.eeg_std,
    )
    # Denormalise fNIRS
    fnirs_recon_raw = (
        result.fnirs_reconstructed * norm_params.fnirs_std[np.newaxis, :]
        + norm_params.fnirs_mean[np.newaxis, :]
    )

    # Also reconstruct the 1-dim EEG for metrics
    eeg_pc1_recon = result.eeg_reconstructed  # [T, 1]

    # ----- Compute metrics -----
    _compute_and_save_metrics(result, eeg_pc1_raw, fnirs_norm, smc, fnirs_fs)

    # ----- Plots (use raw-scale signals for visualisation) -----
    _plot_state_comparison(result, fnirs_fs, OUT_DIR)
    _plot_reconstructed_signals(eeg_recon_raw, fnirs_recon_raw, eeg_power_ds, fnirs, fnirs_fs, OUT_DIR)
    _plot_particle_diagnostics(result, smc, fnirs_fs, OUT_DIR)
    _plot_hrf_validation(result, eeg_pc1_raw, fnirs_norm, smc, fnirs_fs, OUT_DIR)

    print(f'\nResults saved to {OUT_DIR}')


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_and_save_metrics(result, eeg_obs, fnirs_obs, smc, fs_hz):
    """Compute SNR improvement and cross-correlation metrics on normalised signals.

    eeg_obs is [T, 1] (PC1-reduced)
    fnirs_obs is [T, F]
    """
    T = eeg_obs.shape[0]
    F = fnirs_obs.shape[1]

    def snr(clean, noisy):
        power_clean = np.mean(clean ** 2)
        power_noise = np.mean((noisy - clean) ** 2)
        return 10 * np.log10(power_clean / max(power_noise, 1e-16))

    eeg_pred = result.eeg_reconstructed  # [T, 1]
    fnirs_pred = result.fnirs_reconstructed  # [T, F]

    eeg_snr = float(snr(eeg_pred[:, 0], eeg_obs[:, 0]))
    fnirs_snr = float(snr(fnirs_pred, fnirs_obs))

    eeg_corr = np.corrcoef(eeg_obs[:, 0], eeg_pred[:, 0])[0, 1]
    fnirs_corr = np.corrcoef(fnirs_obs.ravel(), fnirs_pred.ravel())[0, 1]

    fnirs_per_ch_corr = [np.corrcoef(fnirs_obs[:, i], fnirs_pred[:, i])[0, 1] for i in range(F)]

    metrics = {
        'subject_id': 1, 'session_idx': 0,
        'duration_s': T / fs_hz,
        'state_dim': smc.K_dim,
        'n_particles': smc.N,
        'ar_coef': float(smc.A[0, 0]),
        'process_noise': float(smc.Q[0, 0]),
        'joint_log_likelihood': float(result.log_likelihood),
        'mean_ess': float(np.mean(result.ess_history)),
        'eeg_snr_db': eeg_snr,
        'fnirs_snr_db': fnirs_snr,
        'eeg_corr_reconstructed_vs_original': float(eeg_corr),
        'fnirs_corr_reconstructed_vs_original': float(fnirs_corr),
        'fnirs_per_channel_corr_mean': float(np.mean(fnirs_per_ch_corr)),
        'fnirs_per_channel_corr_median': float(np.median(fnirs_per_ch_corr)),
    }
    (OUT_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(f'  EEG PC1 reconstructed corr: {eeg_corr:.4f}')
    print(f'  fNIRS reconstructed corr: {fnirs_corr:.4f} (per-ch median {np.median(fnirs_per_ch_corr):.4f})')


# ---------------------------------------------------------------------------
# Plot 1: State comparison (joint vs single-modality)
# ---------------------------------------------------------------------------

def _plot_state_comparison(result, fs_hz, out_dir):
    T = len(result.state_mean)
    t = np.arange(T, dtype=np.float64) / fs_hz

    fig, ax = plt.subplots(figsize=(14, 5))

    # Show all three estimates overlaid
    ax.plot(t, result.state_mean[:, 0], 'k-', linewidth=2, label='Joint (EEG+fNIRS)', alpha=0.9)
    ax.fill_between(
        t,
        result.state_mean[:, 0] - 2 * result.state_std[:, 0],
        result.state_mean[:, 0] + 2 * result.state_std[:, 0],
        color='black', alpha=0.10,
    )
    if result.eeg_only_state_mean is not None:
        ax.plot(t, result.eeg_only_state_mean[:, 0], 'b-', linewidth=1, alpha=0.6, label='EEG-only')
    if result.fnirs_only_state_mean is not None:
        ax.plot(t, result.fnirs_only_state_mean[:, 0], 'r-', linewidth=1, alpha=0.6, label='fNIRS-only')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neural state (a.u.)')
    ax.set_title('Croce et al. 2017 — Joint vs single-modality neural state estimate')
    ax.legend(loc='best')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / 'state_comparison.png', dpi=200, bbox_inches='tight')


# ---------------------------------------------------------------------------
# Plot 2: Reconstructed signals
# ---------------------------------------------------------------------------

def _plot_reconstructed_signals(eeg_recon, fnirs_recon, eeg_orig, fnirs_orig, fs_hz, out_dir):
    T = len(eeg_recon)
    t = np.arange(T, dtype=np.float64) / fs_hz

    # --- Grand-average across channels (raw scale) ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    eeg_grand_orig = eeg_orig.mean(axis=1)
    eeg_grand_recon = eeg_recon.mean(axis=1)
    axes[0].plot(t, eeg_grand_orig, 'gray', linewidth=1, alpha=0.7, label='Original EEG power')
    axes[0].plot(t, eeg_grand_recon, 'blue', linewidth=2, label='Reconstructed (SMC)')
    axes[0].set_ylabel('EEG power (μV²) channel-avg')
    axes[0].set_title('EEG: Original vs SMC-reconstructed (grand-average across channels)')
    axes[0].legend(loc='best')
    axes[0].grid(alpha=0.25)

    fnirs_grand_orig = fnirs_orig.mean(axis=1)
    fnirs_grand_recon = fnirs_recon.mean(axis=1)
    axes[1].plot(t, fnirs_grand_orig, 'gray', linewidth=1, alpha=0.7, label='Original fNIRS (HbO)')
    axes[1].plot(t, fnirs_grand_recon, 'red', linewidth=2, label='Reconstructed (SMC)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('HbO concentration')
    axes[1].set_title('fNIRS: Original vs SMC-reconstructed (grand-average across channels)')
    axes[1].legend(loc='best')
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / 'reconstructed_grand_average.png', dpi=200, bbox_inches='tight')

    # --- Per-channel heatmap ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 8),
                             gridspec_kw={'width_ratios': [20, 1], 'wspace': 0.05})

    vmin_eeg = min(eeg_orig.min(), eeg_recon.min())
    vmax_eeg = max(eeg_orig.max(), eeg_recon.max())

    im0 = axes[0, 0].imshow(eeg_orig.T, aspect='auto', origin='lower', cmap='RdBu_r',
                              vmin=vmin_eeg, vmax=vmax_eeg)
    axes[0, 0].set_title('EEG power — Original')
    axes[0, 0].set_ylabel('Channel')
    plt.colorbar(im0, cax=axes[0, 1])

    im1 = axes[1, 0].imshow(eeg_recon.T, aspect='auto', origin='lower', cmap='RdBu_r',
                              vmin=vmin_eeg, vmax=vmax_eeg)
    axes[1, 0].set_title('EEG power — SMC Reconstructed')
    axes[1, 0].set_xlabel('Time (samples @ 10 Hz)')
    axes[1, 0].set_ylabel('Channel')
    plt.colorbar(im1, cax=axes[1, 1])

    fig.tight_layout()
    fig.savefig(out_dir / 'reconstructed_eeg_heatmap.png', dpi=200, bbox_inches='tight')

    fig, axes = plt.subplots(2, 2, figsize=(16, 8),
                             gridspec_kw={'width_ratios': [20, 1], 'wspace': 0.05})

    vmin_fnirs = min(fnirs_orig.min(), fnirs_recon.min())
    vmax_fnirs = max(fnirs_orig.max(), fnirs_recon.max())

    im2 = axes[0, 0].imshow(fnirs_orig.T, aspect='auto', origin='lower', cmap='RdBu_r',
                              vmin=vmin_fnirs, vmax=vmax_fnirs)
    axes[0, 0].set_title('fNIRS — Original')
    axes[0, 0].set_ylabel('Channel')
    plt.colorbar(im2, cax=axes[0, 1])

    im3 = axes[1, 0].imshow(fnirs_recon.T, aspect='auto', origin='lower', cmap='RdBu_r',
                              vmin=vmin_fnirs, vmax=vmax_fnirs)
    axes[1, 0].set_title('fNIRS — SMC Reconstructed')
    axes[1, 0].set_xlabel('Time (samples @ 10 Hz)')
    axes[1, 0].set_ylabel('Channel')
    plt.colorbar(im3, cax=axes[1, 1])

    fig.tight_layout()
    fig.savefig(out_dir / 'reconstructed_fnirs_heatmap.png', dpi=200, bbox_inches='tight')


# ---------------------------------------------------------------------------
# Plot 3: Particle diagnostics
# ---------------------------------------------------------------------------

def _plot_particle_diagnostics(result, smc, fs_hz, out_dir):
    T = len(result.state_mean)
    t = np.arange(T, dtype=np.float64) / fs_hz

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Particle cloud (subsample for clarity)
    step = max(1, smc.N // 80)
    for i in range(0, smc.N, step):
        alpha_vals = result.weights_history[:, i]
        alpha_norm = alpha_vals / (alpha_vals.max() + 1e-8)
        axes[0].scatter(
            t, result.state_particles[:, i, 0],
            c=f'C{i//step}', s=alpha_norm * 5 + 0.5, alpha=0.3, linewidth=0,
        )
    axes[0].plot(t, result.state_mean[:, 0], 'k-', linewidth=2, label='Posterior mean')
    axes[0].set_ylabel('State value')
    axes[0].set_title('Particle trajectories (point size ∝ weight)')
    axes[0].legend(loc='best')
    axes[0].grid(alpha=0.25)

    # Effective sample size
    axes[1].plot(t, result.ess_history, 'b-', linewidth=1)
    axes[1].axhline(smc.N * smc.resample_threshold, color='red', linestyle='--',
                    label=f'Resample threshold ({smc.N * smc.resample_threshold:.0f})')
    axes[1].set_ylabel('Effective Sample Size')
    axes[1].set_title('Particle filter ESS')
    axes[1].legend(loc='best')
    axes[1].grid(alpha=0.25)
    axes[1].set_ylim(0, smc.N * 1.1)

    # Weight entropy
    entropy = -np.sum(result.weights_history * np.log(result.weights_history + 1e-16), axis=1)
    max_entropy = np.log(smc.N)
    axes[2].plot(t, entropy / max_entropy, 'g-', linewidth=1)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Normalised entropy')
    axes[2].set_title('Particle weight diversity (1 = uniform)')
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / 'particle_diagnostics.png', dpi=200, bbox_inches='tight')


# ---------------------------------------------------------------------------
# Plot 4: HRF validation — lag correlation before/after filtering
# ---------------------------------------------------------------------------

def _plot_hrf_validation(result, eeg_obs, fnirs_obs, smc, fs_hz, out_dir):
    """Show that the filtered state → fNIRS mapping validates the HRF model."""
    T = len(result.state_mean)
    max_lag = int(min(8.0 * fs_hz, T // 3))

    def lag_corr(driver, target):
        vals = []
        for lag in range(max_lag + 1):
            if lag == 0:
                a, b = driver, target
            else:
                a, b = driver[:-lag], target[lag:]
            a = a - a.mean()
            b = b - b.mean()
            da = np.sqrt(np.mean(a * a))
            db = np.sqrt(np.mean(b * b))
            vals.append(float(np.mean(a * b) / max(da * db, 1e-8)))
        return np.array(vals)

    # Use PC1 for EEG (1-dim), grand-average for fNIRS
    eeg_grand = eeg_obs[:, 0] if eeg_obs.ndim == 2 else eeg_obs
    fnirs_grand = fnirs_obs.mean(axis=1)
    state_raw = result.state_mean[:, 0]
    fnirs_recon_grand = result.fnirs_reconstructed.mean(axis=1)

    lags_s = np.arange(max_lag + 1) / fs_hz

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: before vs after — driver→fNIRS lag correlation
    axes[0].plot(lags_s, lag_corr(eeg_grand, fnirs_grand), 'gray', linewidth=2,
                 label='Raw EEG power → fNIRS')
    axes[0].plot(lags_s, lag_corr(state_raw, fnirs_grand), 'blue', linewidth=2,
                 label='Filtered state → fNIRS')
    axes[0].plot(lags_s, lag_corr(fnirs_recon_grand, fnirs_grand), 'red', linewidth=2,
                 label='SMC reconstruction → fNIRS')
    axes[0].axhline(0, color='black', alpha=0.3)
    axes[0].set_xlabel('Lag (s), driver leads fNIRS')
    axes[0].set_ylabel('Pearson r')
    axes[0].set_title('Lag correlation: driver → fNIRS')
    axes[0].legend(loc='best')
    axes[0].grid(alpha=0.25)

    # Right: EEG-only filter vs joint filter state comparison
    if result.eeg_only_state_mean is not None and result.fnirs_only_state_mean is not None:
        axes[1].plot(lags_s, lag_corr(result.eeg_only_state_mean[:, 0], fnirs_grand),
                     'b-', linewidth=2, label='EEG-only filter state → fNIRS')
        axes[1].plot(lags_s, lag_corr(result.fnirs_only_state_mean[:, 0], fnirs_grand),
                     'r-', linewidth=2, label='fNIRS-only filter state → fNIRS')
        axes[1].plot(lags_s, lag_corr(result.state_mean[:, 0], fnirs_grand),
                     'k-', linewidth=2, label='Joint filter state → fNIRS')
        axes[1].axhline(0, color='black', alpha=0.3)
        axes[1].set_xlabel('Lag (s), state leads fNIRS')
        axes[1].set_ylabel('Pearson r')
        axes[1].set_title('Single-modality vs Joint state quality')
        axes[1].legend(loc='best')
        axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / 'hrf_validation.png', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()
