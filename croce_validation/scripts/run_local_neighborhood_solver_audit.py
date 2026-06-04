"""Local-neighborhood Croce 2017 solver audit.

This runner keeps the Croce five-state dynamics but constrains inference to one
local anchor at a time. It never reconstructs r(t) from all channels.

Two modes are supported:

1. synthetic
   Generate one local source, local EEG/fNIRS neighborhoods, and audit recovery
   under controlled noise.
2. npz
   Load one pre-extracted local real-data bundle from an .npz file and audit
   solver stability, timing nulls, and spatial nulls.
3. dataset
    Load one local neighborhood directly from the real EEG+NIRS Single-Trial
    dataset using the existing channel-adjacency metadata.

The real-data path uses deviation coordinates around baseline:

    x(t) = [s(t), delta_f(t), delta_HbO(t), delta_Hb(t), r(t)]

with

    f(t) = 1 + delta_f(t)
    HbO(t) = 1 + delta_HbO(t)
    Hb(t) = 1 + delta_Hb(t)

This preserves the Croce dynamics while allowing zero-centered observation
handling for real data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
from scipy.linalg import expm
from scipy.signal import butter, sosfiltfilt, welch

try:
    import torch
except ImportError:  # pragma: no cover - keep a safe fallback for environments without torch.
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / 'croce_validation' / 'results'
STATE_NAMES = ('s', 'delta_f', 'delta_hbo', 'delta_hb', 'r')

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.channel_adjacency import (  # noqa: E402
    build_channel_adjacency,
    canonicalize_channel_label,
    strip_fnirs_chromophore_suffix,
)
from src.data.eeg_fnirs_dataset import MultiModalEEGfNIRSDataset  # noqa: E402


@dataclass(frozen=True)
class ModelParams:
    epsilon: float = 1.0
    kas: float = 0.41
    kaf: float = 0.65
    tau0: float = 2.0
    alpha: float = 0.32
    e0: float = 0.34


@dataclass(frozen=True)
class SpatialConfig:
    eeg_neighbors: int
    fnirs_neighbors: int
    eeg_radius_mm: float
    fnirs_radius_mm: float
    eeg_sigma_mm: float
    fnirs_sigma_mm: float
    eeg_sign_mode: str


@dataclass(frozen=True)
class FilterConfig:
    integration_dt_s: float
    observation_fs_hz: float
    num_particles: int
    resample_fraction: float
    prior_std: np.ndarray
    state_noise_std: np.ndarray
    sigma_prop: float
    sigma_nirs: float
    seed_list: Tuple[int, ...]
    time_shift_null_s: float
    run_spatial_null: bool
    solver_backend: str = 'torch_exact'
    torch_device: str = 'cpu'


@dataclass(frozen=True)
class ObservationBundle:
    mode: str
    pair_mode: str
    time_s: np.ndarray
    eeg_time_s: np.ndarray
    eeg_obs: np.ndarray
    fnirs_primary_obs: np.ndarray
    fnirs_secondary_obs: np.ndarray
    eeg_fs_hz: float
    fnirs_fs_hz: float
    eeg_substeps_per_fnirs: int
    eeg_positions_mm: np.ndarray
    fnirs_positions_mm: np.ndarray
    anchor_position_mm: np.ndarray
    eeg_indices: np.ndarray
    fnirs_indices: np.ndarray
    lead_field: np.ndarray
    jac_primary: np.ndarray
    jac_secondary: np.ndarray
    r_eeg_projection: np.ndarray
    normalization: Mapping[str, Any]
    units: Mapping[str, str]
    pair_labels: Tuple[str, str]
    eeg_obs_raw: Optional[np.ndarray] = None
    fnirs_primary_obs_raw: Optional[np.ndarray] = None
    fnirs_secondary_obs_raw: Optional[np.ndarray] = None
    eeg_channel_names: Tuple[str, ...] = ()
    fnirs_primary_channel_names: Tuple[str, ...] = ()
    fnirs_secondary_channel_names: Tuple[str, ...] = ()
    metadata: Optional[Mapping[str, Any]] = None
    true_states: Optional[np.ndarray] = None
    true_r_eeg: Optional[np.ndarray] = None
    true_lead_field: Optional[np.ndarray] = None
    true_jac_primary: Optional[np.ndarray] = None
    true_jac_secondary: Optional[np.ndarray] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Audit a local-neighborhood Croce 2017 state-space model on synthetic, npz, or direct dataset-backed real data.',
    )
    parser.add_argument('--mode', choices=('synthetic', 'npz', 'dataset'), default='synthetic')
    parser.add_argument('--input-npz', default='', help='Required in npz mode. Pre-extracted local bundle.')
    parser.add_argument('--output-dir', default='', help='Optional explicit output directory')

    parser.add_argument('--data-root', default='data/EEG+NIRS Single-Trial', help='Dataset root used in dataset mode')
    parser.add_argument('--subject-id', type=int, default=1, help='Subject id used in dataset mode')
    parser.add_argument('--session-idx', type=int, default=0, help='Continuous session index used in dataset mode')
    parser.add_argument('--anchor-fnirs-channel', default='', help='Optional fNIRS base name or channel label used as the local anchor in dataset mode')
    parser.add_argument('--segment-mode', choices=('continuous', 'event_windows'), default='continuous', help='How dataset mode chooses time support for the local bundle')
    parser.add_argument('--segment-start-s', type=float, default=60.0, help='Continuous segment start in seconds for dataset mode')
    parser.add_argument('--segment-duration-s', type=float, default=120.0, help='Continuous segment duration in seconds for dataset mode')
    parser.add_argument('--event-window-pre-s', type=float, default=10.0, help='Seconds kept before the event when --segment-mode=event_windows')
    parser.add_argument('--event-window-post-s', type=float, default=40.0, help='Seconds kept after the event when --segment-mode=event_windows')
    parser.add_argument('--event-idx', type=int, default=-1, help='Event index used when --segment-mode=event_windows')
    parser.add_argument('--use-artifact-eeg', action='store_true', help='Prefer the EEG recordings stored under the ocular-artifact folder in dataset mode')

    parser.add_argument('--duration-s', type=float, default=60.0, help='Synthetic duration in seconds')
    parser.add_argument('--observation-fs', type=float, default=10.0, help='Synthetic fNIRS sampling rate in Hz')
    parser.add_argument('--eeg-fs', type=float, default=200.0, help='Synthetic EEG sampling rate in Hz')
    parser.add_argument('--integration-dt', type=float, default=0.005, help='Transition integration step in seconds')
    parser.add_argument('--snr-db', type=float, default=0.0, help='Synthetic SNR in dB')
    parser.add_argument('--synthetic-eeg-channels', type=int, default=12, help='Synthetic local EEG channel count')
    parser.add_argument('--synthetic-fnirs-channels', type=int, default=8, help='Synthetic local fNIRS channel count per observation family')

    parser.add_argument('--eeg-neighbors', type=int, default=6, help='Maximum local EEG channels used for one anchor')
    parser.add_argument('--fnirs-neighbors', type=int, default=4, help='Maximum local fNIRS channels used for one anchor')
    parser.add_argument('--eeg-radius-mm', type=float, default=60.0, help='Maximum EEG neighborhood radius in mm')
    parser.add_argument('--fnirs-radius-mm', type=float, default=45.0, help='Maximum fNIRS neighborhood radius in mm')
    parser.add_argument('--eeg-sigma-mm', type=float, default=30.0, help='Gaussian decay scale for local EEG weights')
    parser.add_argument('--fnirs-sigma-mm', type=float, default=22.0, help='Gaussian decay scale for local fNIRS weights')
    parser.add_argument(
        '--eeg-sign-mode',
        choices=('covariance', 'geometric_x'),
        default='covariance',
        help='How to assign signs to local EEG forward weights',
    )

    parser.add_argument('--num-particles', type=int, default=400, help='Particle count')
    parser.add_argument('--resample-fraction', type=float, default=0.5, help='Resample when ESS < fraction * N')
    parser.add_argument(
        '--prior-std',
        default='0.05,0.05,0.05,0.05,0.0',
        help='Comma-separated prior std for s,delta_f,delta_hbo,delta_hb,r',
    )
    parser.add_argument(
        '--state-noise-std',
        default='0.02,0.015,0.015,0.015,0.0',
        help='Comma-separated transition-noise std for s,delta_f,delta_hbo,delta_hb,r',
    )
    parser.add_argument('--sigma-prop', type=float, default=0.35, help='Proposal std for r(t) around EEG pseudoinverse projection')
    parser.add_argument('--sigma-nirs', type=float, default=1.0, help='fNIRS likelihood noise scale in observation units')
    parser.add_argument(
        '--solver-backend',
        choices=('auto', 'python_exact', 'torch_exact'),
        default='torch_exact',
        help='Particle-filter backend. Default uses the optimized exact torch implementation.',
    )
    parser.add_argument(
        '--torch-device',
        choices=('cpu', 'cuda', 'auto'),
        default='cpu',
        help='Torch device used by the optimized backend. Current benchmark summaries favor cpu in this workspace.',
    )
    parser.add_argument('--seed-list', default='11,23,37,47,59', help='Comma-separated random seeds for reproducibility audit')
    parser.add_argument('--time-shift-null-s', type=float, default=8.0, help='Shift used for the timing null in seconds')
    parser.add_argument('--run-spatial-null', action='store_true', help='Also run a local channel-order permutation null')

    parser.add_argument('--eeg-unit', default='uV', help='Raw EEG unit label written to the manifest')
    parser.add_argument('--fnirs-primary-unit', default='a.u.', help='Raw unit label for the first fNIRS observation family')
    parser.add_argument('--fnirs-secondary-unit', default='a.u.', help='Raw unit label for the second fNIRS observation family')
    parser.add_argument('--torch-threads', type=int, default=2,
                        help='Number of threads for torch intra-op parallelism. '
                             'Set to 1-2 for small matrices (6x6) to avoid oversubscription '
                             'when using multiprocessing. The old default of 52 threads causes '
                             'severe slowdown (30x+) with fork-based parallel workers.')
    return parser.parse_args()


def parse_vector(spec: str, *, name: str) -> np.ndarray:
    values = [float(item.strip()) for item in spec.split(',') if item.strip()]
    if len(values) != 5:
        raise ValueError(f'{name} must contain 5 comma-separated values, got {spec!r}')
    return np.asarray(values, dtype=np.float64)


def parse_seed_list(spec: str) -> Tuple[int, ...]:
    values = tuple(int(item.strip()) for item in spec.split(',') if item.strip())
    if not values:
        raise ValueError('seed-list must contain at least one integer')
    return values


def resolve_output_dir(spec: str, mode: str) -> Path:
    if spec:
        output_dir = Path(spec)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = RESULTS_ROOT / f'local_solver_audit_{mode}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def ensure_2d_positions(positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)
    if positions.ndim != 2:
        raise ValueError(f'Expected 2D positions, got shape {positions.shape}')
    return positions


def safe_extraction_fraction(flow: np.ndarray, e0: float) -> np.ndarray:
    flow = np.clip(flow, 1e-4, None)
    return 1.0 - np.power(1.0 - e0, 1.0 / flow)


def state_drift(x: np.ndarray, params: ModelParams) -> np.ndarray:
    s, delta_f, delta_hbo, delta_hb, r = x
    f = max(1.0 + float(delta_f), 1e-4)
    hbo = max(1.0 + float(delta_hbo), 1e-4)
    hb = max(1.0 + float(delta_hb), 1e-4)
    alpha = float(params.alpha)
    extraction = safe_extraction_fraction(np.asarray([f]), params.e0)[0] / max(float(params.e0), 1e-8)

    ds = float(params.epsilon) * float(r) - float(params.kas) * float(s) - float(params.kaf) * (f - 1.0)
    d_delta_f = float(s)
    d_delta_hbo = (f - np.power(hbo, 1.0 / alpha)) / float(params.tau0)
    d_delta_hb = (f * extraction - hb * np.power(hbo, (1.0 / alpha) - 1.0)) / float(params.tau0)
    dr = 0.0
    return np.asarray([ds, d_delta_f, d_delta_hbo, d_delta_hb, dr], dtype=np.float64)


def state_jacobian(x: np.ndarray, params: ModelParams) -> np.ndarray:
    _, delta_f, delta_hbo, delta_hb, _ = x
    f = max(1.0 + float(delta_f), 1e-4)
    hbo = max(1.0 + float(delta_hbo), 1e-4)
    hb = max(1.0 + float(delta_hb), 1e-4)
    alpha = float(params.alpha)
    tau0 = float(params.tau0)
    e0 = float(params.e0)
    one_minus_e0 = max(1.0 - e0, 1e-8)
    power_term = np.power(one_minus_e0, 1.0 / f)
    d_extraction_df = power_term * np.log(one_minus_e0) / (f * f)
    d_flow_extraction_df = (
        safe_extraction_fraction(np.asarray([f]), e0)[0] + f * d_extraction_df
    ) / max(e0, 1e-8)

    jac = np.zeros((5, 5), dtype=np.float64)
    jac[0, 0] = -float(params.kas)
    jac[0, 1] = -float(params.kaf)
    jac[0, 4] = float(params.epsilon)
    jac[1, 0] = 1.0
    jac[2, 1] = 1.0 / tau0
    jac[2, 2] = -(1.0 / alpha) * np.power(hbo, (1.0 / alpha) - 1.0) / tau0
    jac[3, 1] = d_flow_extraction_df / tau0
    jac[3, 2] = -hb * ((1.0 / alpha) - 1.0) * np.power(hbo, (1.0 / alpha) - 2.0) / tau0
    jac[3, 3] = -np.power(hbo, (1.0 / alpha) - 1.0) / tau0
    return jac


def local_linearized_step(x: np.ndarray, dt: float, params: ModelParams) -> np.ndarray:
    drift = state_drift(x, params)
    jac = state_jacobian(x, params)
    n = x.shape[0]
    augmented = np.zeros((n + 1, n + 1), dtype=np.float64)
    augmented[:n, :n] = jac
    augmented[:n, n] = drift
    exp_aug = expm(augmented * dt)
    delta = exp_aug[:n, n]
    next_state = np.asarray(x, dtype=np.float64) + delta
    next_state[1:4] = np.clip(next_state[1:4], -0.95, None)
    return next_state


def safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float('nan')
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(diff))))


def lag_peak_seconds(driver: np.ndarray, target: np.ndarray, fs_hz: float, max_lag_s: float = 10.0) -> Tuple[float, float]:
    driver = np.asarray(driver, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64)
    target_mean = target.mean(axis=1) if target.ndim == 2 else target.ravel()
    max_lag = min(int(round(max_lag_s * fs_hz)), max(len(driver) - 2, 0))
    best_lag = 0
    best_corr = float('nan')
    best_abs = -1.0
    for lag in range(max_lag + 1):
        if lag == 0:
            x, y = driver, target_mean
        else:
            x, y = driver[:-lag], target_mean[lag:]
        corr = safe_correlation(x, y)
        if np.isnan(corr):
            continue
        if abs(corr) > best_abs:
            best_abs = abs(corr)
            best_lag = lag
            best_corr = corr
    return best_lag / fs_hz, best_corr


def lowpass_signal(signal: np.ndarray, fs_hz: float, cutoff_hz: float) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.shape[0] < 16:
        return signal.copy()
    max_cutoff = 0.5 * fs_hz * 0.99
    cutoff_hz = min(max(cutoff_hz, 1e-4), max_cutoff)
    sos = butter(4, cutoff_hz / (0.5 * fs_hz), btype='lowpass', output='sos')
    return sosfiltfilt(sos, signal, axis=0)


def bandpass_signal(signal: np.ndarray, fs_hz: float, low_hz: float, high_hz: float) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.shape[0] < 16:
        return signal.copy()
    nyquist = 0.5 * fs_hz
    low_hz = max(float(low_hz), 1e-4)
    high_hz = min(float(high_hz), 0.99 * nyquist)
    if low_hz >= high_hz:
        return signal.copy()
    sos = butter(4, [low_hz / nyquist, high_hz / nyquist], btype='bandpass', output='sos')
    return sosfiltfilt(sos, signal, axis=0)


def project_scalar_source(observations: np.ndarray, lead_field: np.ndarray) -> np.ndarray:
    observations = np.asarray(observations, dtype=np.float64)
    lead = np.asarray(lead_field, dtype=np.float64).ravel()
    denominator = float(np.dot(lead, lead))
    if denominator < 1e-12:
        return np.zeros(observations.shape[0], dtype=np.float64)
    return observations @ lead / denominator


def block_average_signal(signal: np.ndarray, block_size: int, *, block_count: Optional[int] = None) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    if block_size <= 1:
        if block_count is None:
            return values.copy()
        return values[:block_count].copy()
    if block_count is None:
        block_count = values.shape[0] // block_size
    usable = max(int(block_count), 0) * int(block_size)
    trimmed = values[:usable]
    if trimmed.size == 0:
        return np.zeros(0, dtype=np.float64)
    if values.ndim == 1:
        return trimmed.reshape(block_count, block_size).mean(axis=1)
    return trimmed.reshape(block_count, block_size, values.shape[1]).mean(axis=1)


def relative_l2_difference(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denominator = float(np.linalg.norm(b))
    if denominator < 1e-12:
        return 0.0
    return float(np.linalg.norm(a - b) / denominator)


def spectral_band_power(signal: np.ndarray, fs_hz: float, low_hz: float, high_hz: float) -> float:
    values = np.asarray(signal, dtype=np.float64).ravel()
    if values.size < 8 or float(np.std(values)) < 1e-8:
        return 0.0
    nperseg = min(256, values.size)
    freqs, power = welch(values, fs=fs_hz, nperseg=nperseg, scaling='density')
    band_mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(band_mask):
        return 0.0
    if int(np.count_nonzero(band_mask)) == 1:
        return float(power[band_mask][0])
    return float(np.trapezoid(power[band_mask], freqs[band_mask]))


def compute_power_spectrum(signal: np.ndarray, fs_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(signal, dtype=np.float64).ravel()
    if values.size < 8 or float(np.std(values)) < 1e-8:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    nperseg = min(256, values.size)
    freqs, power = welch(values, fs=fs_hz, nperseg=nperseg, scaling='density')
    return np.asarray(freqs, dtype=np.float64), np.asarray(power, dtype=np.float64)


def downsample_signed_channels(channels: np.ndarray, source_fs_hz: float, target_fs_hz: float) -> np.ndarray:
    channels = np.asarray(channels, dtype=np.float64)
    if np.isclose(source_fs_hz, target_fs_hz):
        return channels.copy()
    filtered = lowpass_signal(channels, fs_hz=source_fs_hz, cutoff_hz=min(0.45 * target_fs_hz, 20.0))
    source_time = np.arange(channels.shape[0], dtype=np.float64) / float(source_fs_hz)
    duration_s = source_time[-1] if channels.shape[0] > 1 else 0.0
    target_length = int(np.floor(duration_s * target_fs_hz)) + 1
    target_time = np.arange(target_length, dtype=np.float64) / float(target_fs_hz)
    target_time = target_time[target_time <= source_time[-1] + 1e-9]
    resampled = np.zeros((target_time.shape[0], channels.shape[1]), dtype=np.float64)
    for idx in range(channels.shape[1]):
        resampled[:, idx] = np.interp(target_time, source_time, filtered[:, idx])
    return resampled


def select_local_indices(
    positions_mm: np.ndarray,
    anchor_position_mm: np.ndarray,
    max_channels: int,
    radius_mm: float,
) -> np.ndarray:
    distances = np.linalg.norm(positions_mm - anchor_position_mm.reshape(1, -1), axis=1)
    order = np.argsort(distances)
    within_radius = order[distances[order] <= radius_mm]
    if within_radius.size == 0:
        within_radius = order[:1]
    selected = within_radius[: max(max_channels, 1)]
    return np.asarray(selected, dtype=np.int64)


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(weights))
    if norm < 1e-12:
        return np.full_like(weights, 1.0 / np.sqrt(max(weights.size, 1)))
    return weights / norm


def build_signed_eeg_weights(
    eeg_obs: np.ndarray,
    eeg_positions_mm: np.ndarray,
    anchor_position_mm: np.ndarray,
    sigma_mm: float,
    sign_mode: str,
) -> np.ndarray:
    distances = np.linalg.norm(eeg_positions_mm - anchor_position_mm.reshape(1, -1), axis=1)
    magnitude = np.exp(-0.5 * np.square(distances / max(float(sigma_mm), 1e-6)))

    if sign_mode == 'geometric_x' and eeg_positions_mm.shape[1] >= 1:
        centered = eeg_positions_mm[:, 0] - float(anchor_position_mm[0])
        signs = np.where(centered < 0.0, -1.0, 1.0)
    else:
        anchor_idx = int(np.argmin(distances))
        reference = eeg_obs[:, anchor_idx]
        signs = np.ones(eeg_obs.shape[1], dtype=np.float64)
        for idx in range(eeg_obs.shape[1]):
            corr = safe_correlation(reference, eeg_obs[:, idx])
            if np.isnan(corr) or abs(corr) < 1e-6:
                signs[idx] = 1.0
            else:
                signs[idx] = 1.0 if corr >= 0.0 else -1.0
        signs[anchor_idx] = 1.0

    weights = magnitude * signs
    return normalize_weights(weights.astype(np.float64))


def build_positive_weights(positions_mm: np.ndarray, anchor_position_mm: np.ndarray, sigma_mm: float) -> np.ndarray:
    distances = np.linalg.norm(positions_mm - anchor_position_mm.reshape(1, -1), axis=1)
    weights = np.exp(-0.5 * np.square(distances / max(float(sigma_mm), 1e-6)))
    return normalize_weights(weights.astype(np.float64))


def add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=np.float64)
    power = np.var(signal, axis=0, ddof=0)
    target_ratio = np.power(10.0, snr_db / 10.0)
    noise_power = power / max(target_ratio, 1e-12)
    noise_std = np.sqrt(np.clip(noise_power, 1e-12, None))
    noise = rng.normal(loc=0.0, scale=noise_std.reshape(1, -1), size=signal.shape)
    return signal + noise, noise_std


def standardize_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    matrix = np.asarray(matrix, dtype=np.float64)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=0)
    std = np.where(std < 1e-6, 1.0, std)
    normalized = (matrix - mean.reshape(1, -1)) / std.reshape(1, -1)
    return normalized, {
        'mean': mean.tolist(),
        'std': std.tolist(),
    }


def destandardize_matrix(matrix: np.ndarray, stats: Mapping[str, Any]) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    mean = np.asarray(stats['mean'], dtype=np.float64).reshape(1, -1)
    std = np.asarray(stats['std'], dtype=np.float64).reshape(1, -1)
    return matrix * std + mean


def zscore_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    std = float(np.std(values, ddof=0))
    if std < 1e-8:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std


def is_primary_fnirs_channel(name: str) -> bool:
    return 'highWL' in str(name) or str(name).endswith('_O')


def is_secondary_fnirs_channel(name: str) -> bool:
    return 'lowWL' in str(name) or str(name).endswith('_R')


def build_fnirs_pair_maps(fnirs_channel_names: Sequence[str]) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
    ordered_bases: List[str] = []
    seen = set()
    primary_by_base: Dict[str, int] = {}
    secondary_by_base: Dict[str, int] = {}
    for index, channel_name in enumerate(fnirs_channel_names):
        base_name = strip_fnirs_chromophore_suffix(channel_name)
        base_key = canonicalize_channel_label(base_name)
        if base_key not in seen:
            seen.add(base_key)
            ordered_bases.append(base_name)
        if is_primary_fnirs_channel(channel_name):
            primary_by_base[base_key] = index
        elif is_secondary_fnirs_channel(channel_name):
            secondary_by_base[base_key] = index
    paired_bases = [base_name for base_name in ordered_bases if canonicalize_channel_label(base_name) in primary_by_base and canonicalize_channel_label(base_name) in secondary_by_base]
    return paired_bases, primary_by_base, secondary_by_base


def resolve_anchor_base_name(requested_name: str, base_names: Sequence[str]) -> str:
    if not requested_name:
        return str(base_names[0])
    requested_key = canonicalize_channel_label(strip_fnirs_chromophore_suffix(requested_name))
    for base_name in base_names:
        if canonicalize_channel_label(base_name) == requested_key:
            return str(base_name)
    raise ValueError(f'Could not resolve anchor-fnirs-channel {requested_name!r}')


def select_weighted_eeg_indices(
    combined_weights: np.ndarray,
    eeg_positions_3d: np.ndarray,
    anchor_position_3d: np.ndarray,
    max_channels: int,
) -> np.ndarray:
    selected: List[int] = []
    descending = np.argsort(combined_weights)[::-1]
    for index in descending:
        if combined_weights[index] <= 0.0:
            break
        selected.append(int(index))
        if len(selected) >= max_channels:
            return np.asarray(selected[:max_channels], dtype=np.int64)

    distances = np.linalg.norm(eeg_positions_3d - anchor_position_3d.reshape(1, -1), axis=1)
    for index in np.argsort(distances):
        if int(index) not in selected:
            selected.append(int(index))
        if len(selected) >= max_channels:
            break
    return np.asarray(selected[:max_channels], dtype=np.int64)


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = weights.shape[0]
    positions = (rng.random() + np.arange(n, dtype=np.float64)) / float(n)
    cumulative = np.cumsum(weights)
    indices = np.zeros(n, dtype=np.int64)
    i = 0
    j = 0
    while i < n:
        if positions[i] <= cumulative[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    return indices


def torch_systematic_resample(weights: Any, rng: np.random.Generator) -> Any:
    if torch is None:
        raise RuntimeError('PyTorch systematic resampling requires torch to be available')

    n = int(weights.shape[0])
    positions = (torch.arange(n, device=weights.device, dtype=weights.dtype) + float(rng.random())) / float(n)
    cumulative = torch.cumsum(weights, dim=0)
    cumulative[-1] = 1.0
    return torch.searchsorted(cumulative, positions, right=False)


def pair_mode_uses_concentration_space(pair_mode: str) -> bool:
    return str(pair_mode).strip().lower() in {'concentration', 'chromophore'}


def predict_observations(
    particles: np.ndarray,
    lead_field: np.ndarray,
    jac_primary: np.ndarray,
    jac_secondary: np.ndarray,
    pair_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_eeg = particles[:, 4:5] * lead_field.reshape(1, -1)
    if pair_mode_uses_concentration_space(pair_mode):
        pred_primary = particles[:, 2:3] * jac_primary.reshape(1, -1)
        pred_secondary = particles[:, 3:4] * jac_secondary.reshape(1, -1)
    else:
        pred_primary = (1.00 * particles[:, 2:3] + 0.25 * particles[:, 3:4]) * jac_primary.reshape(1, -1)
        pred_secondary = (0.35 * particles[:, 2:3] + 1.00 * particles[:, 3:4]) * jac_secondary.reshape(1, -1)
    return pred_eeg, pred_primary, pred_secondary


def select_torch_device(requested: str) -> Optional[str]:
    if torch is None:
        return None
    if requested == 'cpu':
        return 'cpu'
    if requested == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but torch.cuda.is_available() is false')
        return 'cuda'
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def torch_safe_extraction_fraction(flow: torch.Tensor, e0: float) -> torch.Tensor:
    base = torch.tensor(1.0 - e0, dtype=flow.dtype, device=flow.device)
    return 1.0 - torch.pow(base, 1.0 / torch.clamp(flow, min=1e-4))


def torch_local_linearized_step_batch(particles: torch.Tensor, dt: float, params: ModelParams) -> torch.Tensor:
    s = particles[:, 0]
    delta_f = particles[:, 1]
    delta_hbo = particles[:, 2]
    delta_hb = particles[:, 3]
    r = particles[:, 4]

    f = torch.clamp(1.0 + delta_f, min=1e-4)
    hbo = torch.clamp(1.0 + delta_hbo, min=1e-4)
    hb = torch.clamp(1.0 + delta_hb, min=1e-4)

    alpha = float(params.alpha)
    tau0 = float(params.tau0)
    e0 = float(params.e0)
    epsilon = float(params.epsilon)
    kas = float(params.kas)
    kaf = float(params.kaf)

    extraction = torch_safe_extraction_fraction(f, e0) / max(e0, 1e-8)
    drift = torch.stack(
        [
            epsilon * r - kas * s - kaf * (f - 1.0),
            s,
            (f - torch.pow(hbo, 1.0 / alpha)) / tau0,
            (f * extraction - hb * torch.pow(hbo, (1.0 / alpha) - 1.0)) / tau0,
            torch.zeros_like(r),
        ],
        dim=1,
    )

    one_minus_e0 = max(1.0 - e0, 1e-8)
    power_term = torch.pow(torch.tensor(one_minus_e0, dtype=particles.dtype, device=particles.device), 1.0 / f)
    d_extraction_df = power_term * np.log(one_minus_e0) / (f * f)
    d_flow_extraction_df = (torch_safe_extraction_fraction(f, e0) + f * d_extraction_df) / max(e0, 1e-8)

    count = particles.shape[0]
    jac = torch.zeros((count, 5, 5), dtype=particles.dtype, device=particles.device)
    jac[:, 0, 0] = -kas
    jac[:, 0, 1] = -kaf
    jac[:, 0, 4] = epsilon
    jac[:, 1, 0] = 1.0
    jac[:, 2, 1] = 1.0 / tau0
    jac[:, 2, 2] = -(1.0 / alpha) * torch.pow(hbo, (1.0 / alpha) - 1.0) / tau0
    jac[:, 3, 1] = d_flow_extraction_df / tau0
    jac[:, 3, 2] = -hb * ((1.0 / alpha) - 1.0) * torch.pow(hbo, (1.0 / alpha) - 2.0) / tau0
    jac[:, 3, 3] = -torch.pow(hbo, (1.0 / alpha) - 1.0) / tau0

    augmented = torch.zeros((count, 6, 6), dtype=particles.dtype, device=particles.device)
    augmented[:, :5, :5] = jac
    augmented[:, :5, 5] = drift
    delta = torch.linalg.matrix_exp(augmented * dt)[:, :5, 5]
    next_state = particles + delta
    next_state[:, 1:4] = torch.clamp(next_state[:, 1:4], min=-0.95)
    return next_state


def torch_predict_observations(
    particles: torch.Tensor,
    lead_field: torch.Tensor,
    jac_primary: torch.Tensor,
    jac_secondary: torch.Tensor,
    pair_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_eeg = particles[:, 4:5] * lead_field.reshape(1, -1)
    if pair_mode_uses_concentration_space(pair_mode):
        pred_primary = particles[:, 2:3] * jac_primary.reshape(1, -1)
        pred_secondary = particles[:, 3:4] * jac_secondary.reshape(1, -1)
    else:
        pred_primary = (1.00 * particles[:, 2:3] + 0.25 * particles[:, 3:4]) * jac_primary.reshape(1, -1)
        pred_secondary = (0.35 * particles[:, 2:3] + 1.00 * particles[:, 3:4]) * jac_secondary.reshape(1, -1)
    return pred_eeg, pred_primary, pred_secondary


def run_particle_filter_python_exact(
    bundle: ObservationBundle,
    filter_config: FilterConfig,
    params: ModelParams,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    num_particles = int(filter_config.num_particles)

    particles = rng.normal(
        loc=np.zeros(5, dtype=np.float64).reshape(1, 5),
        scale=filter_config.prior_std.reshape(1, 5),
        size=(num_particles, 5),
    )
    particles[:, 1:4] = np.clip(particles[:, 1:4], -0.95, None)
    particles[:, 4] = 0.0
    weights = np.full(num_particles, 1.0 / float(num_particles), dtype=np.float64)

    num_fnirs_steps = int(bundle.time_s.shape[0])
    num_eeg_steps = int(bundle.eeg_time_s.shape[0])
    if num_fnirs_steps * int(bundle.eeg_substeps_per_fnirs) != num_eeg_steps:
        raise ValueError('EEG / fNIRS alignment mismatch: eeg_time_s must equal fnirs_time_s * eeg_substeps_per_fnirs')

    estimates = np.zeros((num_fnirs_steps, 5), dtype=np.float64)
    state_std = np.zeros((num_fnirs_steps, 5), dtype=np.float64)
    ess_trace = np.zeros(num_fnirs_steps, dtype=np.float64)
    r_estimates_eeg = np.zeros(num_eeg_steps, dtype=np.float64)
    r_std_eeg = np.zeros(num_eeg_steps, dtype=np.float64)
    log_likelihood_total = 0.0

    integration_dt = float(filter_config.integration_dt_s)
    hemo_noise_std = np.asarray(filter_config.state_noise_std[:4], dtype=np.float64)
    sigma_nirs_sq = max(float(filter_config.sigma_nirs), 1e-8) ** 2

    for step in range(num_fnirs_steps):
        eeg_start = step * int(bundle.eeg_substeps_per_fnirs)
        eeg_stop = eeg_start + int(bundle.eeg_substeps_per_fnirs)

        for eeg_idx in range(eeg_start, eeg_stop):
            proposal_center = float(bundle.r_eeg_projection[eeg_idx])
            particles[:, 4] = proposal_center + float(filter_config.sigma_prop) * rng.normal(size=num_particles)
            for idx in range(num_particles):
                particles[idx] = local_linearized_step(particles[idx], integration_dt, params)
            particles[:, 0:4] += rng.normal(
                loc=0.0,
                scale=hemo_noise_std.reshape(1, 4) * np.sqrt(integration_dt),
                size=(num_particles, 4),
            )
            particles[:, 1:4] = np.clip(particles[:, 1:4], -0.95, None)

            r_estimates_eeg[eeg_idx] = float(np.sum(particles[:, 4] * weights))
            centered_r = particles[:, 4] - r_estimates_eeg[eeg_idx]
            r_std_eeg[eeg_idx] = float(np.sqrt(np.sum(np.square(centered_r) * weights)))

        _, pred_primary, pred_secondary = predict_observations(
            particles,
            bundle.lead_field,
            bundle.jac_primary,
            bundle.jac_secondary,
            bundle.pair_mode,
        )
        log_weights = np.log(np.clip(weights, 1e-300, None))
        log_weights += -0.5 * np.sum(np.square(bundle.fnirs_primary_obs[step].reshape(1, -1) - pred_primary), axis=1) / sigma_nirs_sq
        log_weights += -0.5 * np.sum(np.square(bundle.fnirs_secondary_obs[step].reshape(1, -1) - pred_secondary), axis=1) / sigma_nirs_sq

        max_log_weight = np.max(log_weights)
        stable = np.exp(log_weights - max_log_weight)
        weights = stable / np.clip(stable.sum(), 1e-12, None)
        log_likelihood_total += max_log_weight + np.log(np.clip(stable.sum(), 1e-12, None))

        estimates[step] = np.sum(particles * weights.reshape(-1, 1), axis=0)
        centered = particles - estimates[step].reshape(1, -1)
        state_std[step] = np.sqrt(np.sum(np.square(centered) * weights.reshape(-1, 1), axis=0))
        r_estimates_eeg[eeg_stop - 1] = estimates[step, 4]
        r_std_eeg[eeg_stop - 1] = state_std[step, 4]
        ess = 1.0 / np.sum(np.square(weights))
        ess_trace[step] = ess
        if ess < filter_config.resample_fraction * num_particles:
            indices = systematic_resample(weights, rng)
            particles = particles[indices]
            weights.fill(1.0 / float(num_particles))

    pred_eeg, pred_primary, pred_secondary = predict_observations(
        np.column_stack([
            np.zeros_like(r_estimates_eeg),
            np.zeros_like(r_estimates_eeg),
            np.zeros_like(r_estimates_eeg),
            np.zeros_like(r_estimates_eeg),
            r_estimates_eeg,
        ]),
        bundle.lead_field,
        bundle.jac_primary,
        bundle.jac_secondary,
        bundle.pair_mode,
    )
    _, pred_primary, pred_secondary = predict_observations(
        estimates,
        bundle.lead_field,
        bundle.jac_primary,
        bundle.jac_secondary,
        bundle.pair_mode,
    )
    return {
        'seed': seed,
        'solver_backend': 'python_exact',
        'state_estimates': estimates,
        'state_std': state_std,
        'ess_trace': ess_trace,
        'log_likelihood': float(log_likelihood_total),
        'pred_eeg': pred_eeg,
        'pred_primary': pred_primary,
        'pred_secondary': pred_secondary,
        'r_estimates_eeg': r_estimates_eeg,
        'r_std_eeg': r_std_eeg,
    }


def run_particle_filter_torch_exact(
    bundle: ObservationBundle,
    filter_config: FilterConfig,
    params: ModelParams,
    seed: int,
    device: str,
) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError('PyTorch backend was requested but torch is not available')

    rng = np.random.default_rng(seed)
    num_particles = int(filter_config.num_particles)
    particles_np = rng.normal(
        loc=np.zeros(5, dtype=np.float64).reshape(1, 5),
        scale=np.asarray(filter_config.prior_std, dtype=np.float64).reshape(1, 5),
        size=(num_particles, 5),
    )
    particles_np[:, 1:4] = np.clip(particles_np[:, 1:4], -0.95, None)
    particles_np[:, 4] = 0.0

    particles = torch.from_numpy(particles_np).to(device=device, dtype=torch.float64)
    weights = torch.full((num_particles,), 1.0 / float(num_particles), dtype=torch.float64, device=device)

    num_fnirs_steps = int(bundle.time_s.shape[0])
    num_eeg_steps = int(bundle.eeg_time_s.shape[0])
    if num_fnirs_steps * int(bundle.eeg_substeps_per_fnirs) != num_eeg_steps:
        raise ValueError('EEG / fNIRS alignment mismatch: eeg_time_s must equal fnirs_time_s * eeg_substeps_per_fnirs')

    estimates = np.zeros((num_fnirs_steps, 5), dtype=np.float64)
    state_std = np.zeros((num_fnirs_steps, 5), dtype=np.float64)
    ess_trace = np.zeros(num_fnirs_steps, dtype=np.float64)
    r_estimates_eeg = np.zeros(num_eeg_steps, dtype=np.float64)
    r_std_eeg = np.zeros(num_eeg_steps, dtype=np.float64)
    log_likelihood_total = 0.0

    dt = float(filter_config.integration_dt_s)
    hemo_scale = np.asarray(filter_config.state_noise_std[:4], dtype=np.float64) * np.sqrt(dt)
    sigma_nirs_sq = max(float(filter_config.sigma_nirs), 1e-8) ** 2
    eeg_substeps_per_fnirs = int(bundle.eeg_substeps_per_fnirs)
    proposal_scale = float(filter_config.sigma_prop)

    lead_field_t = torch.from_numpy(np.asarray(bundle.lead_field, dtype=np.float64)).to(device=device, dtype=torch.float64)
    jac_primary_t = torch.from_numpy(np.asarray(bundle.jac_primary, dtype=np.float64)).to(device=device, dtype=torch.float64)
    jac_secondary_t = torch.from_numpy(np.asarray(bundle.jac_secondary, dtype=np.float64)).to(device=device, dtype=torch.float64)
    fnirs_primary_t = torch.from_numpy(np.asarray(bundle.fnirs_primary_obs, dtype=np.float64)).to(device=device, dtype=torch.float64)
    fnirs_secondary_t = torch.from_numpy(np.asarray(bundle.fnirs_secondary_obs, dtype=np.float64)).to(device=device, dtype=torch.float64)
    hemo_scale_row_t = torch.from_numpy(hemo_scale.reshape(1, 4)).to(device=device, dtype=torch.float64)

    if device == 'cuda':
        torch.cuda.synchronize()
    for step in range(num_fnirs_steps):
        eeg_start = step * eeg_substeps_per_fnirs
        eeg_stop = eeg_start + eeg_substeps_per_fnirs
        # Keep the NumPy RNG stream identical to the original implementation while
        # avoiding per-substep NumPy -> Torch tensor wrapping.
        step_standard_noise_t = torch.from_numpy(
            rng.normal(size=(eeg_substeps_per_fnirs, num_particles * 5))
        ).to(device=device, dtype=torch.float64)

        for local_idx, eeg_idx in enumerate(range(eeg_start, eeg_stop)):
            proposal_center = float(bundle.r_eeg_projection[eeg_idx])
            step_noise_t = step_standard_noise_t[local_idx]
            particles[:, 4] = proposal_center + proposal_scale * step_noise_t[:num_particles]
            particles = torch_local_linearized_step_batch(particles, dt, params)
            particles[:, 0:4] += step_noise_t[num_particles:].reshape(num_particles, 4) * hemo_scale_row_t
            particles[:, 1:4] = torch.clamp(particles[:, 1:4], min=-0.95)

            r_mean = torch.sum(particles[:, 4] * weights)
            centered_r = particles[:, 4] - r_mean
            r_estimates_eeg[eeg_idx] = float(r_mean.detach().cpu().item())
            r_std_eeg[eeg_idx] = float(torch.sqrt(torch.sum(torch.square(centered_r) * weights)).detach().cpu().item())

        _, pred_primary_t, pred_secondary_t = torch_predict_observations(
            particles,
            lead_field_t,
            jac_primary_t,
            jac_secondary_t,
            bundle.pair_mode,
        )

        log_weights = torch.log(torch.clamp(weights, min=1e-300))
        log_weights = log_weights + (
            -0.5 * torch.sum(torch.square(fnirs_primary_t[step].reshape(1, -1) - pred_primary_t), dim=1) / sigma_nirs_sq
        )
        log_weights = log_weights + (
            -0.5 * torch.sum(torch.square(fnirs_secondary_t[step].reshape(1, -1) - pred_secondary_t), dim=1) / sigma_nirs_sq
        )

        max_log_weight = torch.max(log_weights)
        stable = torch.exp(log_weights - max_log_weight)
        stable_sum = torch.clamp(torch.sum(stable), min=1e-12)
        weights = stable / stable_sum
        log_likelihood_total += float((max_log_weight + torch.log(stable_sum)).detach().cpu().item())

        estimate_t = torch.sum(particles * weights.reshape(-1, 1), dim=0)
        centered = particles - estimate_t.reshape(1, -1)
        std_t = torch.sqrt(torch.sum(torch.square(centered) * weights.reshape(-1, 1), dim=0))

        estimates[step] = estimate_t.detach().cpu().numpy()
        state_std[step] = std_t.detach().cpu().numpy()
        r_estimates_eeg[eeg_stop - 1] = estimates[step, 4]
        r_std_eeg[eeg_stop - 1] = state_std[step, 4]

        ess = float((1.0 / torch.sum(torch.square(weights))).detach().cpu().item())
        ess_trace[step] = ess
        if ess < filter_config.resample_fraction * num_particles:
            particles = particles[torch_systematic_resample(weights, rng)]
            weights.fill_(1.0 / float(num_particles))

    if device == 'cuda':
        torch.cuda.synchronize()

    pred_eeg, _, _ = predict_observations(
        np.column_stack([
            np.zeros_like(r_estimates_eeg),
            np.zeros_like(r_estimates_eeg),
            np.zeros_like(r_estimates_eeg),
            np.zeros_like(r_estimates_eeg),
            r_estimates_eeg,
        ]),
        bundle.lead_field,
        bundle.jac_primary,
        bundle.jac_secondary,
        bundle.pair_mode,
    )
    _, pred_primary, pred_secondary = predict_observations(
        estimates,
        bundle.lead_field,
        bundle.jac_primary,
        bundle.jac_secondary,
        bundle.pair_mode,
    )
    return {
        'seed': seed,
        'solver_backend': f'torch_exact:{device}',
        'state_estimates': estimates,
        'state_std': state_std,
        'ess_trace': ess_trace,
        'log_likelihood': float(log_likelihood_total),
        'pred_eeg': pred_eeg,
        'pred_primary': pred_primary,
        'pred_secondary': pred_secondary,
        'r_estimates_eeg': r_estimates_eeg,
        'r_std_eeg': r_std_eeg,
    }


def run_particle_filter(
    bundle: ObservationBundle,
    filter_config: FilterConfig,
    params: ModelParams,
    seed: int,
) -> Dict[str, Any]:
    backend = str(getattr(filter_config, 'solver_backend', 'python_exact'))
    if backend == 'auto':
        backend = 'torch_exact' if torch is not None else 'python_exact'
    if backend == 'torch_exact':
        device = select_torch_device(str(getattr(filter_config, 'torch_device', 'cpu')))
        if device is None:
            return run_particle_filter_python_exact(bundle, filter_config, params, seed)
        return run_particle_filter_torch_exact(bundle, filter_config, params, seed, device)
    return run_particle_filter_python_exact(bundle, filter_config, params, seed)


def compute_fit_metrics(
    bundle: ObservationBundle,
    run_result: Mapping[str, Any],
    filter_config: FilterConfig,
) -> Dict[str, Any]:
    per_eeg_corr = [safe_correlation(bundle.eeg_obs[:, idx], run_result['pred_eeg'][:, idx]) for idx in range(bundle.eeg_obs.shape[1])]
    per_primary_corr = [safe_correlation(bundle.fnirs_primary_obs[:, idx], run_result['pred_primary'][:, idx]) for idx in range(bundle.fnirs_primary_obs.shape[1])]
    per_secondary_corr = [safe_correlation(bundle.fnirs_secondary_obs[:, idx], run_result['pred_secondary'][:, idx]) for idx in range(bundle.fnirs_secondary_obs.shape[1])]

    r_estimates_eeg = np.asarray(run_result['r_estimates_eeg'], dtype=np.float64)
    r_projection = np.asarray(bundle.r_eeg_projection, dtype=np.float64)
    r_low = lowpass_signal(r_estimates_eeg, fs_hz=bundle.eeg_fs_hz, cutoff_hz=0.3)
    r_projection_low = lowpass_signal(r_projection, fs_hz=bundle.eeg_fs_hz, cutoff_hz=0.3)
    r_low_fnirs = block_average_signal(r_low, int(bundle.eeg_substeps_per_fnirs), block_count=bundle.time_s.shape[0])
    combined_fnirs = 0.5 * (bundle.fnirs_primary_obs.mean(axis=1) + bundle.fnirs_secondary_obs.mean(axis=1))

    lag_s, lag_corr = lag_peak_seconds(
        r_low_fnirs,
        combined_fnirs,
        fs_hz=bundle.fnirs_fs_hz,
    )
    alpha_power = spectral_band_power(r_estimates_eeg, bundle.eeg_fs_hz, 8.0, 13.0)
    delta_power = spectral_band_power(r_estimates_eeg, bundle.eeg_fs_hz, 0.5, 4.0)
    metrics: Dict[str, Any] = {
        'eeg_corr_mean': float(np.nanmean(per_eeg_corr)),
        'eeg_rmse': rmse(bundle.eeg_obs, run_result['pred_eeg']),
        'fnirs_primary_corr_mean': float(np.nanmean(per_primary_corr)),
        'fnirs_primary_rmse': rmse(bundle.fnirs_primary_obs, run_result['pred_primary']),
        'fnirs_secondary_corr_mean': float(np.nanmean(per_secondary_corr)),
        'fnirs_secondary_rmse': rmse(bundle.fnirs_secondary_obs, run_result['pred_secondary']),
        'fnirs_corr_mean': float(np.nanmean([np.nanmean(per_primary_corr), np.nanmean(per_secondary_corr)])),
        'hemo_to_fnirs_peak_lag_s': lag_s,
        'hemo_to_fnirs_peak_corr': lag_corr,
        'r_alpha_delta_power_ratio': float(alpha_power / max(delta_power, 1e-8)),
        'r_low_modification_ratio': relative_l2_difference(r_low, r_projection_low),
        'r_total_modification_ratio': relative_l2_difference(r_estimates_eeg, r_projection),
        'r_projection_corr': safe_correlation(r_estimates_eeg, r_projection),
        'ess_ratio_mean': float(np.mean(run_result['ess_trace']) / max(filter_config.num_particles, 1)),
        'state_r_std_mean': float(np.mean(run_result['state_std'][:, 4])),
        'log_likelihood': float(run_result['log_likelihood']),
    }
    if bundle.true_r_eeg is not None:
        true_r_eeg = np.asarray(bundle.true_r_eeg, dtype=np.float64)[: r_estimates_eeg.shape[0]]
        metrics.update(
            {
                'r_norm_rmse': rmse(zscore_vector(true_r_eeg), zscore_vector(r_estimates_eeg)),
                'r_high_corr': safe_correlation(
                    bandpass_signal(true_r_eeg, bundle.eeg_fs_hz, 1.0, 30.0),
                    bandpass_signal(r_estimates_eeg, bundle.eeg_fs_hz, 1.0, 30.0),
                ),
                'r_low_corr': safe_correlation(
                    lowpass_signal(true_r_eeg, bundle.eeg_fs_hz, 0.3),
                    lowpass_signal(r_estimates_eeg, bundle.eeg_fs_hz, 0.3),
                ),
            }
        )
    return metrics


def summarise_seed_reproducibility(run_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    pairwise_corr: List[float] = []
    for idx in range(len(run_results)):
        for jdx in range(idx + 1, len(run_results)):
            corr = safe_correlation(run_results[idx]['r_estimates_eeg'], run_results[jdx]['r_estimates_eeg'])
            if np.isfinite(corr):
                pairwise_corr.append(corr)
    return {
        'seed_pairwise_r_corr_mean': float(np.mean(pairwise_corr)) if pairwise_corr else float('nan'),
        'seed_pairwise_r_corr_median': float(np.median(pairwise_corr)) if pairwise_corr else float('nan'),
        'seed_pairwise_count': int(len(pairwise_corr)),
    }


def permute_channels(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[1] < 2:
        return matrix.copy()
    order = np.arange(matrix.shape[1], dtype=np.int64)[::-1]
    return matrix[:, order]


def build_null_bundle(bundle: ObservationBundle, *, time_shift_s: float = 0.0, spatial_permutation: bool = False) -> ObservationBundle:
    shift_samples = int(round(time_shift_s * bundle.eeg_fs_hz)) if bundle.eeg_time_s.shape[0] > 1 else 0
    eeg_obs = np.asarray(bundle.eeg_obs, dtype=np.float64)
    fnirs_primary_obs = np.asarray(bundle.fnirs_primary_obs, dtype=np.float64)
    fnirs_secondary_obs = np.asarray(bundle.fnirs_secondary_obs, dtype=np.float64)
    eeg_obs_raw = None if bundle.eeg_obs_raw is None else np.asarray(bundle.eeg_obs_raw, dtype=np.float64)
    fnirs_primary_obs_raw = None if bundle.fnirs_primary_obs_raw is None else np.asarray(bundle.fnirs_primary_obs_raw, dtype=np.float64)
    fnirs_secondary_obs_raw = None if bundle.fnirs_secondary_obs_raw is None else np.asarray(bundle.fnirs_secondary_obs_raw, dtype=np.float64)
    if shift_samples:
        eeg_obs = np.roll(eeg_obs, shift_samples, axis=0)
        if eeg_obs_raw is not None:
            eeg_obs_raw = np.roll(eeg_obs_raw, shift_samples, axis=0)
    if spatial_permutation:
        eeg_obs = permute_channels(eeg_obs)
        fnirs_primary_obs = permute_channels(fnirs_primary_obs)
        fnirs_secondary_obs = permute_channels(fnirs_secondary_obs)
        if eeg_obs_raw is not None:
            eeg_obs_raw = permute_channels(eeg_obs_raw)
        if fnirs_primary_obs_raw is not None:
            fnirs_primary_obs_raw = permute_channels(fnirs_primary_obs_raw)
        if fnirs_secondary_obs_raw is not None:
            fnirs_secondary_obs_raw = permute_channels(fnirs_secondary_obs_raw)
    return ObservationBundle(
        mode=bundle.mode,
        pair_mode=bundle.pair_mode,
        time_s=bundle.time_s,
        eeg_time_s=bundle.eeg_time_s,
        eeg_obs=eeg_obs,
        fnirs_primary_obs=fnirs_primary_obs,
        fnirs_secondary_obs=fnirs_secondary_obs,
        eeg_fs_hz=bundle.eeg_fs_hz,
        fnirs_fs_hz=bundle.fnirs_fs_hz,
        eeg_substeps_per_fnirs=bundle.eeg_substeps_per_fnirs,
        eeg_positions_mm=bundle.eeg_positions_mm,
        fnirs_positions_mm=bundle.fnirs_positions_mm,
        anchor_position_mm=bundle.anchor_position_mm,
        eeg_indices=bundle.eeg_indices,
        fnirs_indices=bundle.fnirs_indices,
        lead_field=bundle.lead_field,
        jac_primary=bundle.jac_primary,
        jac_secondary=bundle.jac_secondary,
        r_eeg_projection=project_scalar_source(eeg_obs, bundle.lead_field),
        normalization=bundle.normalization,
        units=bundle.units,
        pair_labels=bundle.pair_labels,
        eeg_obs_raw=eeg_obs_raw,
        fnirs_primary_obs_raw=fnirs_primary_obs_raw,
        fnirs_secondary_obs_raw=fnirs_secondary_obs_raw,
        eeg_channel_names=bundle.eeg_channel_names,
        fnirs_primary_channel_names=bundle.fnirs_primary_channel_names,
        fnirs_secondary_channel_names=bundle.fnirs_secondary_channel_names,
        metadata=bundle.metadata,
        true_states=bundle.true_states,
        true_r_eeg=bundle.true_r_eeg,
        true_lead_field=bundle.true_lead_field,
        true_jac_primary=bundle.true_jac_primary,
        true_jac_secondary=bundle.true_jac_secondary,
    )


def simulate_synthetic_bundle(
    args: argparse.Namespace,
    spatial_config: SpatialConfig,
) -> ObservationBundle:
    rng = np.random.default_rng(20260521)
    eeg_fs = float(args.eeg_fs)
    fnirs_fs = float(args.observation_fs)
    substeps = int(round(eeg_fs / fnirs_fs))
    if substeps < 1 or not np.isclose(substeps * fnirs_fs, eeg_fs, atol=1e-9):
        raise ValueError('Synthetic eeg-fs must be an integer multiple of observation-fs')
    integration_dt = 1.0 / eeg_fs

    n_eeg = int(args.synthetic_eeg_channels)
    n_fnirs = int(args.synthetic_fnirs_channels)
    eeg_x = np.linspace(-45.0, 45.0, n_eeg, dtype=np.float64)
    fnirs_x = np.linspace(-24.0, 24.0, n_fnirs, dtype=np.float64)
    eeg_positions = np.stack([eeg_x, 10.0 * np.sin(np.linspace(0.0, np.pi, n_eeg))], axis=1)
    fnirs_positions = np.stack([fnirs_x, 5.0 * np.cos(np.linspace(0.0, np.pi, n_fnirs))], axis=1)
    anchor = np.asarray([0.0, 0.0], dtype=np.float64)

    eeg_indices = select_local_indices(eeg_positions, anchor, spatial_config.eeg_neighbors, spatial_config.eeg_radius_mm)
    fnirs_indices = select_local_indices(fnirs_positions, anchor, spatial_config.fnirs_neighbors, spatial_config.fnirs_radius_mm)
    local_eeg_positions = eeg_positions[eeg_indices]
    local_fnirs_positions = fnirs_positions[fnirs_indices]

    true_lead = build_signed_eeg_weights(
        eeg_obs=np.tile(np.linspace(-1.0, 1.0, n_eeg).reshape(1, -1), (32, 1))[:, eeg_indices],
        eeg_positions_mm=local_eeg_positions,
        anchor_position_mm=anchor,
        sigma_mm=spatial_config.eeg_sigma_mm,
        sign_mode='geometric_x',
    )
    true_jac_primary = build_positive_weights(local_fnirs_positions, anchor, spatial_config.fnirs_sigma_mm)
    true_jac_secondary = build_positive_weights(local_fnirs_positions, anchor, spatial_config.fnirs_sigma_mm * 1.1)

    params = ModelParams()
    num_eeg_steps = int(round(float(args.duration_s) * eeg_fs))
    if num_eeg_steps < substeps:
        raise ValueError('Synthetic duration is too short for at least one fNIRS step')
    num_fnirs_steps = num_eeg_steps // substeps
    num_eeg_steps = num_fnirs_steps * substeps
    time_eeg = np.arange(num_eeg_steps, dtype=np.float64) / eeg_fs
    time_obs = np.arange(num_fnirs_steps, dtype=np.float64) / fnirs_fs

    r_true = (
        0.90 * np.sin(2.0 * np.pi * 10.0 * time_eeg)
        + 0.35 * np.sin(2.0 * np.pi * 0.22 * time_eeg + 0.3)
        + 0.20 * np.sin(2.0 * np.pi * 0.05 * time_eeg + 1.1)
    )
    states = np.zeros((num_eeg_steps, 5), dtype=np.float64)
    states[0, 4] = r_true[0]
    for step in range(num_eeg_steps - 1):
        current_state = states[step].copy()
        current_state[4] = r_true[step]
        next_state = local_linearized_step(current_state, integration_dt, params)
        next_state[1:4] += np.asarray([0.003, 0.002, 0.002], dtype=np.float64) * np.sqrt(integration_dt) * rng.normal(size=3)
        next_state[1:4] = np.clip(next_state[1:4], -0.95, None)
        next_state[4] = r_true[step + 1]
        states[step + 1] = next_state
    states[:, 4] = r_true

    observation_indices = (np.arange(num_fnirs_steps, dtype=np.int64) + 1) * substeps - 1
    states_obs = states[observation_indices]

    eeg_clean = states[:, 4:5] * true_lead.reshape(1, -1)
    fnirs_primary_clean = (1.00 * states_obs[:, 2:3] + 0.25 * states_obs[:, 3:4]) * true_jac_primary.reshape(1, -1)
    fnirs_secondary_clean = (0.35 * states_obs[:, 2:3] + 1.00 * states_obs[:, 3:4]) * true_jac_secondary.reshape(1, -1)

    eeg_noisy, _ = add_awgn(eeg_clean, float(args.snr_db), rng)
    fnirs_primary_noisy, _ = add_awgn(fnirs_primary_clean, float(args.snr_db), rng)
    fnirs_secondary_noisy, _ = add_awgn(fnirs_secondary_clean, float(args.snr_db), rng)

    eeg_norm, eeg_stats = standardize_matrix(eeg_noisy)
    fnirs_primary_norm, fnirs_primary_stats = standardize_matrix(fnirs_primary_noisy)
    fnirs_secondary_norm, fnirs_secondary_stats = standardize_matrix(fnirs_secondary_noisy)

    approx_lead = build_signed_eeg_weights(
        eeg_obs=eeg_norm,
        eeg_positions_mm=local_eeg_positions,
        anchor_position_mm=anchor,
        sigma_mm=spatial_config.eeg_sigma_mm,
        sign_mode=spatial_config.eeg_sign_mode,
    )
    approx_jac_primary = build_positive_weights(local_fnirs_positions, anchor, spatial_config.fnirs_sigma_mm)
    approx_jac_secondary = build_positive_weights(local_fnirs_positions, anchor, spatial_config.fnirs_sigma_mm)

    return ObservationBundle(
        mode='synthetic',
        pair_mode='wavelength',
        time_s=time_obs,
        eeg_time_s=time_eeg,
        eeg_obs=eeg_norm,
        fnirs_primary_obs=fnirs_primary_norm,
        fnirs_secondary_obs=fnirs_secondary_norm,
        eeg_fs_hz=eeg_fs,
        fnirs_fs_hz=fnirs_fs,
        eeg_substeps_per_fnirs=substeps,
        eeg_positions_mm=local_eeg_positions,
        fnirs_positions_mm=local_fnirs_positions,
        anchor_position_mm=anchor,
        eeg_indices=eeg_indices,
        fnirs_indices=fnirs_indices,
        lead_field=approx_lead,
        jac_primary=approx_jac_primary,
        jac_secondary=approx_jac_secondary,
        r_eeg_projection=project_scalar_source(eeg_norm, approx_lead),
        normalization={
            'mode': 'per_channel_zscore_after_local_selection',
            'eeg': eeg_stats,
            'fnirs_primary': fnirs_primary_stats,
            'fnirs_secondary': fnirs_secondary_stats,
        },
        units={
            'eeg': args.eeg_unit,
            'fnirs_primary': args.fnirs_primary_unit,
            'fnirs_secondary': args.fnirs_secondary_unit,
        },
        pair_labels=('highWL_like', 'lowWL_like'),
        eeg_obs_raw=eeg_noisy,
        fnirs_primary_obs_raw=fnirs_primary_noisy,
        fnirs_secondary_obs_raw=fnirs_secondary_noisy,
        eeg_channel_names=tuple(f'EEG_{index}' for index in eeg_indices.tolist()),
        fnirs_primary_channel_names=tuple(f'highWL_like_{index}' for index in fnirs_indices.tolist()),
        fnirs_secondary_channel_names=tuple(f'lowWL_like_{index}' for index in fnirs_indices.tolist()),
        metadata={'signal_source': 'synthetic_local_source'},
        true_states=states_obs,
        true_r_eeg=r_true,
        true_lead_field=true_lead,
        true_jac_primary=true_jac_primary,
        true_jac_secondary=true_jac_secondary,
    )


def load_real_bundle(args: argparse.Namespace, spatial_config: SpatialConfig) -> ObservationBundle:
    if not args.input_npz:
        raise ValueError('--input-npz is required in npz mode')

    with np.load(args.input_npz, allow_pickle=False) as bundle_npz:
        eeg = np.asarray(bundle_npz['eeg'], dtype=np.float64)
        eeg_fs_hz = float(np.asarray(bundle_npz['eeg_fs_hz']).item())
        eeg_positions = ensure_2d_positions(bundle_npz['eeg_positions_mm'])
        fnirs_positions = ensure_2d_positions(bundle_npz['fnirs_positions_mm'])
        fnirs_fs_hz = float(np.asarray(bundle_npz['fnirs_fs_hz']).item())
        eeg_channel_names = tuple(
            str(name) for name in np.asarray(bundle_npz['eeg_channel_names']).reshape(-1).tolist()
        ) if 'eeg_channel_names' in bundle_npz else ()
        fnirs_channel_bases = tuple(
            str(name) for name in np.asarray(bundle_npz['fnirs_channel_names']).reshape(-1).tolist()
        ) if 'fnirs_channel_names' in bundle_npz else ()

        requested_anchor = str(getattr(args, 'anchor_fnirs_channel', '')).strip()
        anchor_base_name = ''
        bundle_task = str(np.asarray(bundle_npz['task']).item()) if 'task' in bundle_npz else ''
        bundle_segment_kind = str(np.asarray(bundle_npz['segment_kind']).item()) if 'segment_kind' in bundle_npz else ''
        bundle_segment_index = int(np.asarray(bundle_npz['segment_index']).item()) if 'segment_index' in bundle_npz else None
        bundle_segment_label = str(np.asarray(bundle_npz['label_name']).item()) if 'label_name' in bundle_npz else ''
        optical_projection_kind = str(np.asarray(bundle_npz['optical_projection_kind']).item()) if 'optical_projection_kind' in bundle_npz else ''
        bundle_segment_start_s = float(np.asarray(bundle_npz['eeg_start_ms']).item()) / 1000.0 if 'eeg_start_ms' in bundle_npz else None
        bundle_segment_duration_s = (
            float(np.asarray(bundle_npz['eeg_end_ms']).item()) - float(np.asarray(bundle_npz['eeg_start_ms']).item())
        ) / 1000.0 if 'eeg_start_ms' in bundle_npz and 'eeg_end_ms' in bundle_npz else None
        bundle_event_idx = int(np.asarray(bundle_npz['event_idx']).item()) if 'event_idx' in bundle_npz else None
        bundle_event_window_pre_s = float(np.asarray(bundle_npz['event_window_pre_s']).item()) if 'event_window_pre_s' in bundle_npz else None
        bundle_event_window_post_s = float(np.asarray(bundle_npz['event_window_post_s']).item()) if 'event_window_post_s' in bundle_npz else None
        bundle_aligned_window_start_s = float(np.asarray(bundle_npz['aligned_window_start_s']).item()) if 'aligned_window_start_s' in bundle_npz else None
        bundle_aligned_window_end_s = float(np.asarray(bundle_npz['aligned_window_end_s']).item()) if 'aligned_window_end_s' in bundle_npz else None
        bundle_eeg_event_onset_s = float(np.asarray(bundle_npz['eeg_event_onset_ms']).item()) / 1000.0 if 'eeg_event_onset_ms' in bundle_npz else None
        bundle_fnirs_event_onset_s = float(np.asarray(bundle_npz['fnirs_event_onset_ms']).item()) / 1000.0 if 'fnirs_event_onset_ms' in bundle_npz else None

        if requested_anchor and fnirs_channel_bases:
            anchor_base_name = resolve_anchor_base_name(requested_anchor, fnirs_channel_bases)
            anchor_key = canonicalize_channel_label(anchor_base_name)
            anchor_index = next(
                index for index, base_name in enumerate(fnirs_channel_bases)
                if canonicalize_channel_label(base_name) == anchor_key
            )
            anchor = fnirs_positions[anchor_index]
        elif 'anchor_position_mm' in bundle_npz:
            anchor = np.asarray(bundle_npz['anchor_position_mm'], dtype=np.float64).ravel()
        elif 'eeg_anchor_index' in bundle_npz:
            anchor = eeg_positions[int(np.asarray(bundle_npz['eeg_anchor_index']).item())]
        elif 'fnirs_anchor_index' in bundle_npz:
            anchor = fnirs_positions[int(np.asarray(bundle_npz['fnirs_anchor_index']).item())]
        else:
            anchor = np.mean(np.concatenate([eeg_positions, fnirs_positions], axis=0), axis=0)

        if 'fnirs_850' in bundle_npz and 'fnirs_760' in bundle_npz:
            fnirs_primary = np.asarray(bundle_npz['fnirs_850'], dtype=np.float64)
            fnirs_secondary = np.asarray(bundle_npz['fnirs_760'], dtype=np.float64)
            pair_mode = 'wavelength'
            pair_labels = ('highWL', 'lowWL')
        elif 'fnirs_690' in bundle_npz and 'fnirs_830' in bundle_npz:
            fnirs_primary = np.asarray(bundle_npz['fnirs_690'], dtype=np.float64)
            fnirs_secondary = np.asarray(bundle_npz['fnirs_830'], dtype=np.float64)
            pair_mode = 'wavelength'
            pair_labels = ('690', '830')
        elif 'fnirs_hbo' in bundle_npz and 'fnirs_hb' in bundle_npz:
            fnirs_primary = np.asarray(bundle_npz['fnirs_hbo'], dtype=np.float64)
            fnirs_secondary = np.asarray(bundle_npz['fnirs_hb'], dtype=np.float64)
            pair_mode = 'concentration'
            pair_labels = ('HbO', 'Hb')
        elif 'fnirs_primary' in bundle_npz and 'fnirs_secondary' in bundle_npz:
            fnirs_primary = np.asarray(bundle_npz['fnirs_primary'], dtype=np.float64)
            fnirs_secondary = np.asarray(bundle_npz['fnirs_secondary'], dtype=np.float64)
            pair_mode = 'wavelength'
            pair_labels = ('optical_0', 'optical_1')
        else:
            raise ValueError('NPZ bundle must provide one of {fnirs_690, fnirs_830}, {fnirs_hbo, fnirs_hb}, or {fnirs_primary, fnirs_secondary}')

    eeg_indices = select_local_indices(eeg_positions, anchor, spatial_config.eeg_neighbors, spatial_config.eeg_radius_mm)
    fnirs_indices = select_local_indices(fnirs_positions, anchor, spatial_config.fnirs_neighbors, spatial_config.fnirs_radius_mm)
    local_eeg_positions = eeg_positions[eeg_indices]
    local_fnirs_positions = fnirs_positions[fnirs_indices]

    eeg_local = np.asarray(eeg[:, eeg_indices], dtype=np.float64)
    fnirs_primary_local = np.asarray(fnirs_primary[:, fnirs_indices], dtype=np.float64)
    fnirs_secondary_local = np.asarray(fnirs_secondary[:, fnirs_indices], dtype=np.float64)

    eeg_substeps_per_fnirs = int(round(eeg_fs_hz / fnirs_fs_hz))
    if eeg_substeps_per_fnirs < 1 or not np.isclose(eeg_substeps_per_fnirs * fnirs_fs_hz, eeg_fs_hz, atol=1e-9):
        raise ValueError('NPZ bundle requires EEG fs to be an integer multiple of fNIRS fs')
    length = min(eeg_local.shape[0] // eeg_substeps_per_fnirs, fnirs_primary_local.shape[0], fnirs_secondary_local.shape[0])
    eeg_local = eeg_local[: length * eeg_substeps_per_fnirs]
    fnirs_primary_local = fnirs_primary_local[:length]
    fnirs_secondary_local = fnirs_secondary_local[:length]
    time_s = np.arange(length, dtype=np.float64) / fnirs_fs_hz
    eeg_time_s = np.arange(eeg_local.shape[0], dtype=np.float64) / eeg_fs_hz

    lead_field = build_signed_eeg_weights(
        eeg_obs=eeg_local,
        eeg_positions_mm=local_eeg_positions,
        anchor_position_mm=anchor,
        sigma_mm=spatial_config.eeg_sigma_mm,
        sign_mode=spatial_config.eeg_sign_mode,
    )
    jac_primary = build_positive_weights(local_fnirs_positions, anchor, spatial_config.fnirs_sigma_mm)
    jac_secondary = build_positive_weights(local_fnirs_positions, anchor, spatial_config.fnirs_sigma_mm)

    eeg_norm, eeg_stats = standardize_matrix(eeg_local)
    fnirs_primary_norm, fnirs_primary_stats = standardize_matrix(fnirs_primary_local)
    fnirs_secondary_norm, fnirs_secondary_stats = standardize_matrix(fnirs_secondary_local)

    if pair_mode_uses_concentration_space(pair_mode):
        primary_suffix = '_O'
        secondary_suffix = '_R'
    else:
        primary_suffix = 'highWL'
        secondary_suffix = 'lowWL'

    if eeg_channel_names:
        eeg_channel_names_local = tuple(eeg_channel_names[index] for index in eeg_indices.tolist())
    else:
        eeg_channel_names_local = tuple(f'EEG_{index}' for index in eeg_indices.tolist())

    if fnirs_channel_bases:
        fnirs_primary_channel_names = tuple(
            f'{fnirs_channel_bases[index]}{primary_suffix}' for index in fnirs_indices.tolist()
        )
        fnirs_secondary_channel_names = tuple(
            f'{fnirs_channel_bases[index]}{secondary_suffix}' for index in fnirs_indices.tolist()
        )
    else:
        fnirs_primary_channel_names = tuple(f'{pair_labels[0]}_{index}' for index in fnirs_indices.tolist())
        fnirs_secondary_channel_names = tuple(f'{pair_labels[1]}_{index}' for index in fnirs_indices.tolist())

    metadata: Dict[str, Any] = {'signal_source': 'npz_bundle', 'segment_mode': 'npz_bundle'}
    if requested_anchor and anchor_base_name:
        metadata['anchor_fnirs_base'] = anchor_base_name
    if fnirs_channel_bases:
        metadata['fnirs_channel_bases'] = list(fnirs_channel_bases)
    if bundle_task:
        metadata['task'] = bundle_task
    if bundle_segment_kind:
        metadata['bundle_segment_kind'] = bundle_segment_kind
    if bundle_segment_index is not None:
        metadata['segment_index'] = bundle_segment_index
    if bundle_event_idx is not None:
        metadata['event_idx'] = bundle_event_idx
    if bundle_segment_label:
        metadata['segment_label_name'] = bundle_segment_label
    if optical_projection_kind:
        metadata['optical_projection_kind'] = optical_projection_kind
    if bundle_segment_start_s is not None and bundle_segment_duration_s is not None:
        metadata['segment_start_s'] = bundle_segment_start_s
        metadata['segment_duration_s'] = bundle_segment_duration_s
    if bundle_event_window_pre_s is not None:
        metadata['event_window_pre_s'] = bundle_event_window_pre_s
    if bundle_event_window_post_s is not None:
        metadata['event_window_post_s'] = bundle_event_window_post_s
    if bundle_aligned_window_start_s is not None:
        metadata['aligned_window_start_s'] = bundle_aligned_window_start_s
    if bundle_aligned_window_end_s is not None:
        metadata['aligned_window_end_s'] = bundle_aligned_window_end_s
    if bundle_eeg_event_onset_s is not None:
        metadata['eeg_event_onset_s'] = bundle_eeg_event_onset_s
    if bundle_fnirs_event_onset_s is not None:
        metadata['fnirs_event_onset_s'] = bundle_fnirs_event_onset_s

    return ObservationBundle(
        mode='npz',
        pair_mode=pair_mode,
        time_s=time_s,
        eeg_time_s=eeg_time_s,
        eeg_obs=eeg_norm,
        fnirs_primary_obs=fnirs_primary_norm,
        fnirs_secondary_obs=fnirs_secondary_norm,
        eeg_fs_hz=eeg_fs_hz,
        fnirs_fs_hz=fnirs_fs_hz,
        eeg_substeps_per_fnirs=eeg_substeps_per_fnirs,
        eeg_positions_mm=local_eeg_positions,
        fnirs_positions_mm=local_fnirs_positions,
        anchor_position_mm=anchor,
        eeg_indices=eeg_indices,
        fnirs_indices=fnirs_indices,
        lead_field=lead_field,
        jac_primary=jac_primary,
        jac_secondary=jac_secondary,
        r_eeg_projection=project_scalar_source(eeg_norm, lead_field),
        normalization={
            'mode': 'per_channel_zscore_after_local_selection',
            'eeg': eeg_stats,
            'fnirs_primary': fnirs_primary_stats,
            'fnirs_secondary': fnirs_secondary_stats,
            'coordinate_system': 'deviation_coordinates_about_baseline',
        },
        units={
            'eeg': args.eeg_unit,
            'fnirs_primary': args.fnirs_primary_unit,
            'fnirs_secondary': args.fnirs_secondary_unit,
        },
        pair_labels=pair_labels,
        eeg_obs_raw=eeg_local,
        fnirs_primary_obs_raw=fnirs_primary_local,
        fnirs_secondary_obs_raw=fnirs_secondary_local,
        eeg_channel_names=eeg_channel_names_local,
        fnirs_primary_channel_names=fnirs_primary_channel_names,
        fnirs_secondary_channel_names=fnirs_secondary_channel_names,
        metadata=metadata,
    )


def extract_marker_event_info(markers: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    event_times_s = np.asarray(markers['time'], dtype=np.float64) / 1000.0
    labels = np.argmax(np.asarray(markers['y']), axis=0).astype(int)
    class_names_raw = markers.get('className')
    class_names = [str(name) for name in class_names_raw] if class_names_raw is not None else []
    return event_times_s, labels, class_names


def label_name_for_index(label: int, class_names: Sequence[str]) -> str:
    if 0 <= int(label) < len(class_names):
        return str(class_names[int(label)])
    return str(int(label))


def resolve_task_session_index(args: argparse.Namespace, total_sessions: int) -> Tuple[int, int, str]:
    """Map task-relative session index to the raw Single-Trial session list."""
    task = str(getattr(args, 'task', 'motor_imagery')).strip().lower()
    task_session_idx = int(getattr(args, 'session_idx', 0))
    task_to_raw_sessions = {
        'motor_imagery': (0, 2, 4),
        'mental_arithmetic': (1, 3, 5),
    }
    raw_candidates = task_to_raw_sessions.get(task)
    if raw_candidates is None:
        if task_session_idx < 0 or task_session_idx >= total_sessions:
            raise ValueError(f'session-idx {task_session_idx} is out of range')
        return task_session_idx, task_session_idx, task

    if task_session_idx < 0 or task_session_idx >= len(raw_candidates):
        raise ValueError(
            f'session-idx {task_session_idx} is out of range for task {task!r}; '
            f'valid task-relative indices are 0..{len(raw_candidates) - 1}'
        )
    raw_session_idx = int(raw_candidates[task_session_idx])
    if raw_session_idx >= total_sessions:
        raise ValueError(
            f'raw session index {raw_session_idx} for task {task!r} is out of range '
            f'for {total_sessions} loaded sessions'
        )
    return task_session_idx, raw_session_idx, task


def resolve_dataset_event_windows(
    args: argparse.Namespace,
    dataset: Optional[MultiModalEEGfNIRSDataset] = None,
) -> List[Dict[str, Any]]:
    dataset = dataset or MultiModalEEGfNIRSDataset(
        data_root=args.data_root,
        subject_ids=[int(args.subject_id)],
        task=getattr(args, 'task', 'motor_imagery'),
        window_duration_s=2.5,
        normalize=False,
        normalization_mode='none',
        eeg_preprocessing={'bandpass': [0.5, 45.0]},
        fnirs_preprocessing={'lowpass': 0.2},
        use_artifact_data=bool(args.use_artifact_eeg),
        exclude_eog=True,
        hbo_only=False,
        hbr_only=False,
    )

    eeg_session_list, eeg_marker_list, eeg_info = dataset._get_eeg_data(int(args.subject_id), processed=True)
    fnirs_session_list, fnirs_marker_list, fnirs_info = dataset._get_nirs_data(int(args.subject_id), processed=True)
    task_session_idx, raw_session_idx, task = resolve_task_session_index(
        args,
        min(len(eeg_session_list), len(fnirs_session_list)),
    )

    eeg_duration_s = float(np.asarray(eeg_session_list[raw_session_idx]).shape[0]) / float(eeg_info['fs'])
    fnirs_duration_s = float(np.asarray(fnirs_session_list[raw_session_idx]).shape[0]) / float(fnirs_info['fs'])
    eeg_event_times_s, eeg_labels, eeg_class_names = extract_marker_event_info(eeg_marker_list[raw_session_idx])
    fnirs_event_times_s, fnirs_labels, fnirs_class_names = extract_marker_event_info(fnirs_marker_list[raw_session_idx])

    common_events = int(min(len(eeg_event_times_s), len(fnirs_event_times_s)))
    pre_s = float(getattr(args, 'event_window_pre_s', 10.0))
    post_s = float(getattr(args, 'event_window_post_s', 40.0))
    descriptors: List[Dict[str, Any]] = []
    for event_idx in range(common_events):
        eeg_onset_s = float(eeg_event_times_s[event_idx])
        fnirs_onset_s = float(fnirs_event_times_s[event_idx])
        eeg_start_s = eeg_onset_s - pre_s
        eeg_end_s = eeg_onset_s + post_s
        fnirs_start_s = fnirs_onset_s - pre_s
        fnirs_end_s = fnirs_onset_s + post_s
        invalid_reasons: List[str] = []
        if eeg_start_s < 0.0:
            invalid_reasons.append('eeg_pre_exceeds_record_start')
        if fnirs_start_s < 0.0:
            invalid_reasons.append('fnirs_pre_exceeds_record_start')
        if eeg_end_s > eeg_duration_s + 1e-9:
            invalid_reasons.append('eeg_post_exceeds_record_end')
        if fnirs_end_s > fnirs_duration_s + 1e-9:
            invalid_reasons.append('fnirs_post_exceeds_record_end')

        eeg_label = int(eeg_labels[event_idx])
        fnirs_label = int(fnirs_labels[event_idx])
        descriptors.append({
            'event_idx': event_idx,
            'task': task,
            'task_session_idx': task_session_idx,
            'raw_session_idx': raw_session_idx,
            'eeg_onset_s': eeg_onset_s,
            'fnirs_onset_s': fnirs_onset_s,
            'eeg_start_s': eeg_start_s,
            'eeg_end_s': eeg_end_s,
            'fnirs_start_s': fnirs_start_s,
            'fnirs_end_s': fnirs_end_s,
            'aligned_window_start_s': -pre_s,
            'aligned_window_end_s': post_s,
            'event_window_pre_s': pre_s,
            'event_window_post_s': post_s,
            'eeg_label': eeg_label,
            'fnirs_label': fnirs_label,
            'eeg_label_name': label_name_for_index(eeg_label, eeg_class_names),
            'fnirs_label_name': label_name_for_index(fnirs_label, fnirs_class_names),
            'label_index_match': bool(eeg_label == fnirs_label),
            'raw_event_offset_s': fnirs_onset_s - eeg_onset_s,
            'is_valid': len(invalid_reasons) == 0,
            'invalid_reasons': invalid_reasons,
        })
    return descriptors


def load_dataset_bundle(args: argparse.Namespace, spatial_config: SpatialConfig) -> ObservationBundle:
    dataset = MultiModalEEGfNIRSDataset(
        data_root=args.data_root,
        subject_ids=[int(args.subject_id)],
        task=getattr(args, 'task', 'motor_imagery'),
        window_duration_s=2.5,
        normalize=False,
        normalization_mode='none',
        eeg_preprocessing={'bandpass': [0.5, 45.0]},
        fnirs_preprocessing={'lowpass': 0.2},
        use_artifact_data=bool(args.use_artifact_eeg),
        exclude_eog=True,
        hbo_only=False,
        hbr_only=False,
    )

    eeg_session_list, _, eeg_info = dataset._get_eeg_data(int(args.subject_id), processed=True)
    fnirs_session_list, _, fnirs_info = dataset._get_nirs_data(int(args.subject_id), processed=True)
    task_session_idx, raw_session_idx, task = resolve_task_session_index(
        args,
        min(len(eeg_session_list), len(fnirs_session_list)),
    )

    adjacency = build_channel_adjacency(
        'eeg_fnirs_single_trial',
        args.data_root,
        dataset.get_eeg_channel_names(),
        dataset.get_fnirs_channel_names(),
        reference_subject_id=int(args.subject_id),
        use_artifact_data=bool(args.use_artifact_eeg),
    )
    paired_bases, primary_by_base, secondary_by_base = build_fnirs_pair_maps(adjacency.fnirs_channel_names)
    if not paired_bases:
        raise ValueError('No paired fNIRS observation families were found in dataset mode')

    anchor_base_name = resolve_anchor_base_name(args.anchor_fnirs_channel, paired_bases)
    ordered_primary_indices = np.asarray([primary_by_base[canonicalize_channel_label(base_name)] for base_name in paired_bases], dtype=np.int64)
    primary_positions_3d = np.asarray(adjacency.fnirs_channel_positions_3d[ordered_primary_indices], dtype=np.float64)
    anchor_index = next(index for index, base_name in enumerate(paired_bases) if canonicalize_channel_label(base_name) == canonicalize_channel_label(anchor_base_name))
    anchor_position_3d = primary_positions_3d[anchor_index]
    distances = np.linalg.norm(primary_positions_3d - anchor_position_3d.reshape(1, -1), axis=1)
    local_base_order = np.argsort(distances)[: max(int(spatial_config.fnirs_neighbors), 1)]
    local_base_names = [paired_bases[index] for index in local_base_order.tolist()]
    primary_indices = np.asarray([primary_by_base[canonicalize_channel_label(base_name)] for base_name in local_base_names], dtype=np.int64)
    secondary_indices = np.asarray([secondary_by_base[canonicalize_channel_label(base_name)] for base_name in local_base_names], dtype=np.int64)

    combined_weights = np.asarray(adjacency.adjacency_matrix[primary_indices], dtype=np.float64).mean(axis=0)
    eeg_indices = select_weighted_eeg_indices(
        combined_weights,
        np.asarray(adjacency.eeg_positions_3d, dtype=np.float64),
        anchor_position_3d,
        max(int(spatial_config.eeg_neighbors), 1),
    )

    eeg_full = np.asarray(eeg_session_list[raw_session_idx], dtype=np.float64)
    fnirs_full = np.asarray(fnirs_session_list[raw_session_idx], dtype=np.float64)
    eeg_fs_hz = float(eeg_info['fs'])
    fnirs_fs_hz = float(fnirs_info['fs'])
    eeg_substeps_per_fnirs = int(round(eeg_fs_hz / fnirs_fs_hz))
    if eeg_substeps_per_fnirs < 1 or not np.isclose(eeg_substeps_per_fnirs * fnirs_fs_hz, eeg_fs_hz, atol=1e-9):
        raise ValueError('Dataset mode requires EEG fs to be an integer multiple of fNIRS fs')
    segment_mode = str(getattr(args, 'segment_mode', 'continuous')).strip().lower()
    event_window: Optional[Dict[str, Any]] = None
    nominal_event_duration_s: Optional[float] = None
    if segment_mode == 'event_windows':
        event_idx = int(getattr(args, 'event_idx', -1))
        if event_idx < 0:
            raise ValueError('event_idx must be set when segment_mode=event_windows')
        if all(hasattr(args, attr) for attr in (
            'eeg_segment_start_s_raw',
            'eeg_segment_end_s_raw',
            'fnirs_segment_start_s_raw',
            'fnirs_segment_end_s_raw',
            'eeg_event_onset_s',
            'fnirs_event_onset_s',
        )):
            event_window = {
                'event_idx': event_idx,
                'eeg_start_s': float(args.eeg_segment_start_s_raw),
                'eeg_end_s': float(args.eeg_segment_end_s_raw),
                'fnirs_start_s': float(args.fnirs_segment_start_s_raw),
                'fnirs_end_s': float(args.fnirs_segment_end_s_raw),
                'aligned_window_start_s': float(getattr(args, 'aligned_window_start_s', -float(getattr(args, 'event_window_pre_s', 10.0)))),
                'aligned_window_end_s': float(getattr(args, 'aligned_window_end_s', float(getattr(args, 'event_window_post_s', 40.0)))),
                'event_window_pre_s': float(getattr(args, 'event_window_pre_s', 10.0)),
                'event_window_post_s': float(getattr(args, 'event_window_post_s', 40.0)),
                'eeg_onset_s': float(args.eeg_event_onset_s),
                'fnirs_onset_s': float(args.fnirs_event_onset_s),
                'eeg_label': int(getattr(args, 'event_label_index', -1)),
                'fnirs_label': int(getattr(args, 'event_label_index', -1)),
                'eeg_label_name': str(getattr(args, 'event_label_name_eeg', getattr(args, 'event_label_name', event_idx))),
                'fnirs_label_name': str(getattr(args, 'event_label_name_fnirs', getattr(args, 'event_label_name', event_idx))),
                'label_index_match': bool(getattr(args, 'event_label_index_match', True)),
                'raw_event_offset_s': float(args.fnirs_event_onset_s) - float(args.eeg_event_onset_s),
                'is_valid': True,
                'invalid_reasons': [],
            }
        else:
            event_windows = resolve_dataset_event_windows(args, dataset=dataset)
            if event_idx >= len(event_windows):
                raise ValueError(f'event_idx {event_idx} is out of range for task session {task_session_idx}')
            event_window = event_windows[event_idx]
        if not bool(event_window['is_valid']):
            raise ValueError(
                f"event_idx {event_idx} does not have a full cross-modal window: {event_window['invalid_reasons']}"
            )

        segment_start_eeg = max(int(round(float(event_window['eeg_start_s']) * eeg_fs_hz)), 0)
        segment_end_eeg = min(int(round(float(event_window['eeg_end_s']) * eeg_fs_hz)), eeg_full.shape[0])
        segment_start_fnirs = max(int(round(float(event_window['fnirs_start_s']) * fnirs_fs_hz)), 0)
        segment_end_fnirs = min(int(round(float(event_window['fnirs_end_s']) * fnirs_fs_hz)), fnirs_full.shape[0])
        segment_start_s = float(event_window['aligned_window_start_s'])
        nominal_event_duration_s = float(event_window['event_window_pre_s']) + float(event_window['event_window_post_s'])
    elif float(args.segment_duration_s) <= 0.0:
        segment_start_eeg = 0
        segment_start_fnirs = 0
        max_fnirs_steps = min(fnirs_full.shape[0], eeg_full.shape[0] // eeg_substeps_per_fnirs)
        segment_end_fnirs = max_fnirs_steps
        segment_end_eeg = max_fnirs_steps * eeg_substeps_per_fnirs
        segment_start_s = 0.0
    else:
        segment_start_eeg = max(int(round(float(args.segment_start_s) * eeg_fs_hz)), 0)
        segment_start_fnirs = max(int(round(float(args.segment_start_s) * fnirs_fs_hz)), 0)
        segment_end_fnirs = min(int(round((float(args.segment_start_s) + float(args.segment_duration_s)) * fnirs_fs_hz)), fnirs_full.shape[0])
        segment_end_eeg = min(segment_start_eeg + (segment_end_fnirs - segment_start_fnirs) * eeg_substeps_per_fnirs, eeg_full.shape[0])
        segment_start_s = float(args.segment_start_s)
    if segment_end_eeg - segment_start_eeg < 32 or segment_end_fnirs - segment_start_fnirs < 16:
        raise ValueError('Selected dataset segment is too short for local solver audit')

    eeg_local_raw = eeg_full[segment_start_eeg:segment_end_eeg, :][:, eeg_indices]
    fnirs_primary_local = fnirs_full[segment_start_fnirs:segment_end_fnirs, :][:, primary_indices]
    fnirs_secondary_local = fnirs_full[segment_start_fnirs:segment_end_fnirs, :][:, secondary_indices]
    length = min(eeg_local_raw.shape[0] // eeg_substeps_per_fnirs, fnirs_primary_local.shape[0], fnirs_secondary_local.shape[0])
    eeg_local_raw = eeg_local_raw[: length * eeg_substeps_per_fnirs]
    fnirs_primary_local = fnirs_primary_local[:length]
    fnirs_secondary_local = fnirs_secondary_local[:length]
    effective_segment_duration_s = float(length) / fnirs_fs_hz
    if segment_mode == 'event_windows' and nominal_event_duration_s is not None:
        expected_fnirs_steps = int(round(nominal_event_duration_s * fnirs_fs_hz))
        expected_eeg_steps = expected_fnirs_steps * eeg_substeps_per_fnirs
        if length != expected_fnirs_steps or eeg_local_raw.shape[0] != expected_eeg_steps:
            raise ValueError(
                'Event-window slicing produced an unexpected shape: '
                f'expected fnirs={expected_fnirs_steps}, eeg={expected_eeg_steps}, '
                f'got fnirs={length}, eeg={eeg_local_raw.shape[0]}'
            )
    time_s = segment_start_s + np.arange(length, dtype=np.float64) / fnirs_fs_hz
    eeg_time_s = segment_start_s + np.arange(eeg_local_raw.shape[0], dtype=np.float64) / eeg_fs_hz

    local_eeg_positions = np.asarray(adjacency.eeg_positions_2d[eeg_indices], dtype=np.float64)
    local_fnirs_positions = np.asarray(adjacency.fnirs_channel_positions_2d[primary_indices], dtype=np.float64)
    anchor_position_2d = np.asarray(adjacency.fnirs_channel_positions_2d[primary_indices[0]], dtype=np.float64)
    lead_field = build_signed_eeg_weights(
        eeg_obs=eeg_local_raw,
        eeg_positions_mm=local_eeg_positions,
        anchor_position_mm=anchor_position_2d,
        sigma_mm=spatial_config.eeg_sigma_mm,
        sign_mode=spatial_config.eeg_sign_mode,
    )
    jac_primary = build_positive_weights(local_fnirs_positions, anchor_position_2d, spatial_config.fnirs_sigma_mm)
    jac_secondary = build_positive_weights(local_fnirs_positions, anchor_position_2d, spatial_config.fnirs_sigma_mm)

    eeg_norm, eeg_stats = standardize_matrix(eeg_local_raw)
    fnirs_primary_norm, fnirs_primary_stats = standardize_matrix(fnirs_primary_local)
    fnirs_secondary_norm, fnirs_secondary_stats = standardize_matrix(fnirs_secondary_local)

    metadata = {
        'signal_source': 'real_optical_continuous_signal',
        'fnirs_signal_semantics': 'paired_optical_wavelength_channels',
        'fnirs_cache_requirement': 'keep_fNIRS_targets_in_optical_measurement_space_before_cross_dataset_caching',
        'data_root': str(args.data_root),
        'task': task,
        'subject_id': int(args.subject_id),
        'session_idx': task_session_idx,
        'raw_session_idx': raw_session_idx,
        'segment_mode': segment_mode,
        'segment_start_s': segment_start_s,
        'segment_duration_s': effective_segment_duration_s,
        'full_session_used': bool(segment_mode == 'continuous' and float(args.segment_duration_s) <= 0.0),
        'anchor_fnirs_base': str(anchor_base_name),
        'local_fnirs_bases': list(local_base_names),
    }
    if event_window is not None:
        metadata.update({
            'event_idx': int(event_window['event_idx']),
            'event_label_index': int(event_window['eeg_label']),
            'event_label_name_eeg': str(event_window['eeg_label_name']),
            'event_label_name_fnirs': str(event_window['fnirs_label_name']),
            'event_label_index_match': bool(event_window['label_index_match']),
            'event_window_pre_s': float(event_window['event_window_pre_s']),
            'event_window_post_s': float(event_window['event_window_post_s']),
            'event_window_duration_s': nominal_event_duration_s,
            'aligned_window_start_s': float(event_window['aligned_window_start_s']),
            'aligned_window_end_s': float(event_window['aligned_window_end_s']),
            'eeg_event_onset_s': float(event_window['eeg_onset_s']),
            'fnirs_event_onset_s': float(event_window['fnirs_onset_s']),
            'event_alignment_offset_s': float(event_window['raw_event_offset_s']),
            'eeg_segment_start_s_raw': float(event_window['eeg_start_s']),
            'eeg_segment_end_s_raw': float(event_window['eeg_end_s']),
            'fnirs_segment_start_s_raw': float(event_window['fnirs_start_s']),
            'fnirs_segment_end_s_raw': float(event_window['fnirs_end_s']),
            'valid_common_event_window': True,
        })

    # The EEG+NIRS Single-Trial recordings store paired raw optical channels
    # (`highWL` / `lowWL`) rather than HbO / HbR concentration traces.
    return ObservationBundle(
        mode='dataset',
        pair_mode='wavelength',
        time_s=time_s,
        eeg_time_s=eeg_time_s,
        eeg_obs=eeg_norm,
        fnirs_primary_obs=fnirs_primary_norm,
        fnirs_secondary_obs=fnirs_secondary_norm,
        eeg_fs_hz=eeg_fs_hz,
        fnirs_fs_hz=fnirs_fs_hz,
        eeg_substeps_per_fnirs=eeg_substeps_per_fnirs,
        eeg_positions_mm=local_eeg_positions,
        fnirs_positions_mm=local_fnirs_positions,
        anchor_position_mm=anchor_position_2d,
        eeg_indices=eeg_indices,
        fnirs_indices=primary_indices,
        lead_field=lead_field,
        jac_primary=jac_primary,
        jac_secondary=jac_secondary,
        r_eeg_projection=project_scalar_source(eeg_norm, lead_field),
        normalization={
            'mode': 'per_channel_zscore_after_local_selection',
            'eeg': eeg_stats,
            'fnirs_primary': fnirs_primary_stats,
            'fnirs_secondary': fnirs_secondary_stats,
            'coordinate_system': 'deviation_coordinates_about_baseline',
        },
        units={
            'eeg': args.eeg_unit,
            'fnirs_primary': args.fnirs_primary_unit,
            'fnirs_secondary': args.fnirs_secondary_unit,
        },
        pair_labels=('highWL', 'lowWL'),
        eeg_obs_raw=eeg_local_raw,
        fnirs_primary_obs_raw=fnirs_primary_local,
        fnirs_secondary_obs_raw=fnirs_secondary_local,
        eeg_channel_names=tuple(str(adjacency.eeg_channel_names[index]) for index in eeg_indices.tolist()),
        fnirs_primary_channel_names=tuple(str(adjacency.fnirs_channel_names[index]) for index in primary_indices.tolist()),
        fnirs_secondary_channel_names=tuple(str(adjacency.fnirs_channel_names[index]) for index in secondary_indices.tolist()),
        metadata=metadata,
    )


def to_serializable_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    serializable: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            serializable[key] = value.item()
        else:
            serializable[key] = value
    return serializable


def add_time_colored_trajectory(
    axis: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    time_s: np.ndarray,
    *,
    cmap: str = 'viridis',
    linewidth: float = 2.0,
) -> Optional[LineCollection]:
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    time_s = np.asarray(time_s, dtype=np.float64)
    if x_values.shape[0] < 2:
        axis.plot(x_values, y_values, color='#111111', linewidth=linewidth)
        return None
    points = np.column_stack([x_values, y_values]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    line = LineCollection(segments, cmap=cmap, norm=Normalize(vmin=float(time_s[0]), vmax=float(time_s[-1])))
    line.set_array(0.5 * (time_s[:-1] + time_s[1:]))
    line.set_linewidth(linewidth)
    axis.add_collection(line)
    axis.scatter([x_values[0]], [y_values[0]], color='#111111', marker='o', s=24, zorder=3)
    axis.scatter([x_values[-1]], [y_values[-1]], color='#111111', marker='x', s=36, zorder=3)
    axis.autoscale_view()
    return line


def plot_state_timecourses(path: Path, bundle: ObservationBundle, run_result: Mapping[str, Any]) -> None:
    fig, axes = plt.subplots(len(STATE_NAMES), 1, figsize=(13, 14), sharex=True)
    state_mean = np.asarray(run_result['state_estimates'], dtype=np.float64)
    state_std = np.asarray(run_result['state_std'], dtype=np.float64)
    for idx, axis in enumerate(axes):
        axis.plot(bundle.time_s, state_mean[:, idx], color='#111111', linewidth=1.35)
        axis.fill_between(
            bundle.time_s,
            state_mean[:, idx] - state_std[:, idx],
            state_mean[:, idx] + state_std[:, idx],
            color='#1f77b4',
            alpha=0.16,
            linewidth=0.0,
        )
        axis.axhline(0.0, color='#BDBDBD', linewidth=0.8, linestyle='--')
        axis.set_ylabel(STATE_NAMES[idx])
        axis.grid(alpha=0.25)
    axes[0].set_title('Local Croce state evolution')
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_time_colored_state_space(path: Path, bundle: ObservationBundle, run_result: Mapping[str, Any]) -> None:
    state_mean = np.asarray(run_result['state_estimates'], dtype=np.float64)
    panels = [
        ('r vs s', 4, 0),
        ('s vs delta_f', 0, 1),
        ('delta_hbo vs delta_hb', 2, 3),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 4.8))
    colorbar_handle: Optional[LineCollection] = None
    for axis, (title, x_idx, y_idx) in zip(axes, panels):
        colorbar_handle = add_time_colored_trajectory(axis, state_mean[:, x_idx], state_mean[:, y_idx], bundle.time_s)
        axis.set_title(title)
        axis.set_xlabel(STATE_NAMES[x_idx])
        axis.set_ylabel(STATE_NAMES[y_idx])
        axis.grid(alpha=0.25)
    if colorbar_handle is not None:
        colorbar = fig.colorbar(colorbar_handle, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        colorbar.set_label('Time (s)')
    fig.suptitle('Time-colored state-space trajectories', y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_observation_reconstruction(path: Path, bundle: ObservationBundle, run_result: Mapping[str, Any]) -> None:
    if bundle.eeg_obs_raw is None or bundle.fnirs_primary_obs_raw is None or bundle.fnirs_secondary_obs_raw is None:
        return
    if bundle.normalization.get('mode') == 'per_channel_zscore_after_local_selection':
        pred_eeg_raw = destandardize_matrix(run_result['pred_eeg'], bundle.normalization['eeg'])
        pred_primary_raw = destandardize_matrix(run_result['pred_primary'], bundle.normalization['fnirs_primary'])
        pred_secondary_raw = destandardize_matrix(run_result['pred_secondary'], bundle.normalization['fnirs_secondary'])
    else:
        pred_eeg_raw = np.asarray(run_result['pred_eeg'], dtype=np.float64)
        pred_primary_raw = np.asarray(run_result['pred_primary'], dtype=np.float64)
        pred_secondary_raw = np.asarray(run_result['pred_secondary'], dtype=np.float64)

    eeg_obs_raw = np.asarray(bundle.eeg_obs_raw, dtype=np.float64)
    fnirs_primary_raw = np.asarray(bundle.fnirs_primary_obs_raw, dtype=np.float64)
    fnirs_secondary_raw = np.asarray(bundle.fnirs_secondary_obs_raw, dtype=np.float64)
    representative_eeg = int(np.argmax(np.abs(bundle.lead_field))) if bundle.lead_field.size else 0
    representative_eeg_name = bundle.eeg_channel_names[representative_eeg] if bundle.eeg_channel_names else f'EEG_{representative_eeg}'

    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    axes[0].plot(bundle.eeg_time_s, eeg_obs_raw.mean(axis=1), color='#111111', linewidth=1.3, label='Observed EEG neighborhood mean')
    axes[0].plot(bundle.eeg_time_s, pred_eeg_raw.mean(axis=1), color='#D62728', linewidth=1.1, label='Reconstructed EEG neighborhood mean')
    axes[0].plot(bundle.eeg_time_s, eeg_obs_raw[:, representative_eeg], color='#7F7F7F', linewidth=0.8, alpha=0.85, label=f'Observed {representative_eeg_name}')
    axes[0].plot(bundle.eeg_time_s, pred_eeg_raw[:, representative_eeg], color='#FF9896', linewidth=0.8, alpha=0.95, label=f'Reconstructed {representative_eeg_name}')
    axes[0].set_ylabel(bundle.units['eeg'])
    axes[0].set_title('EEG source target reconstruction')
    axes[0].legend(loc='upper right', ncol=2)
    axes[0].grid(alpha=0.25)

    axes[1].plot(bundle.time_s, fnirs_primary_raw.mean(axis=1), color='#1F77B4', linewidth=1.3, label=f'Observed {bundle.pair_labels[0]} mean')
    axes[1].plot(bundle.time_s, pred_primary_raw.mean(axis=1), color='#17BECF', linewidth=1.1, label=f'Reconstructed {bundle.pair_labels[0]} mean')
    axes[1].set_ylabel(bundle.units['fnirs_primary'])
    axes[1].set_title(f'{bundle.pair_labels[0]} source target reconstruction')
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.25)

    axes[2].plot(bundle.time_s, fnirs_secondary_raw.mean(axis=1), color='#D62728', linewidth=1.3, label=f'Observed {bundle.pair_labels[1]} mean')
    axes[2].plot(bundle.time_s, pred_secondary_raw.mean(axis=1), color='#FF9896', linewidth=1.1, label=f'Reconstructed {bundle.pair_labels[1]} mean')
    axes[2].set_ylabel(bundle.units['fnirs_secondary'])
    axes[2].set_title(f'{bundle.pair_labels[1]} source target reconstruction')
    axes[2].legend(loc='upper right')
    axes[2].grid(alpha=0.25)
    axes[2].set_xlabel('Time (s)')

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_target_time_frequency_diagnostics(path: Path, bundle: ObservationBundle, run_result: Mapping[str, Any]) -> None:
    if bundle.eeg_obs_raw is None or bundle.fnirs_primary_obs_raw is None or bundle.fnirs_secondary_obs_raw is None:
        return
    if bundle.normalization.get('mode') == 'per_channel_zscore_after_local_selection':
        pred_eeg_raw = destandardize_matrix(run_result['pred_eeg'], bundle.normalization['eeg'])
        pred_primary_raw = destandardize_matrix(run_result['pred_primary'], bundle.normalization['fnirs_primary'])
        pred_secondary_raw = destandardize_matrix(run_result['pred_secondary'], bundle.normalization['fnirs_secondary'])
    else:
        pred_eeg_raw = np.asarray(run_result['pred_eeg'], dtype=np.float64)
        pred_primary_raw = np.asarray(run_result['pred_primary'], dtype=np.float64)
        pred_secondary_raw = np.asarray(run_result['pred_secondary'], dtype=np.float64)

    eeg_obs_raw = np.asarray(bundle.eeg_obs_raw, dtype=np.float64)
    fnirs_primary_raw = np.asarray(bundle.fnirs_primary_obs_raw, dtype=np.float64)
    fnirs_secondary_raw = np.asarray(bundle.fnirs_secondary_obs_raw, dtype=np.float64)
    representative_eeg = int(np.argmax(np.abs(bundle.lead_field))) if bundle.lead_field.size else 0
    representative_eeg_name = bundle.eeg_channel_names[representative_eeg] if bundle.eeg_channel_names else f'EEG_{representative_eeg}'

    traces = [
        {
            'label': f'EEG {representative_eeg_name}',
            'time_s': bundle.eeg_time_s,
            'raw': eeg_obs_raw[:, representative_eeg],
            'source_target': pred_eeg_raw[:, representative_eeg],
            'fs_hz': bundle.eeg_fs_hz,
            'unit': bundle.units['eeg'],
            'freq_max_hz': min(40.0, 0.5 * bundle.eeg_fs_hz),
            'source_color': '#D62728',
            'obs_color': '#1F77B4',
        },
        {
            'label': str(bundle.pair_labels[0]),
            'time_s': bundle.time_s,
            'raw': fnirs_primary_raw.mean(axis=1),
            'source_target': pred_primary_raw.mean(axis=1),
            'fs_hz': bundle.fnirs_fs_hz,
            'unit': bundle.units['fnirs_primary'],
            'freq_max_hz': min(0.5, 0.5 * bundle.fnirs_fs_hz),
            'source_color': '#17BECF',
            'obs_color': '#1F77B4',
        },
        {
            'label': str(bundle.pair_labels[1]),
            'time_s': bundle.time_s,
            'raw': fnirs_secondary_raw.mean(axis=1),
            'source_target': pred_secondary_raw.mean(axis=1),
            'fs_hz': bundle.fnirs_fs_hz,
            'unit': bundle.units['fnirs_secondary'],
            'freq_max_hz': min(0.5, 0.5 * bundle.fnirs_fs_hz),
            'source_color': '#FF9896',
            'obs_color': '#9467BD',
        },
    ]

    # EEG and fNIRS use different frequency ranges, so sharing x by column
    # would collapse the EEG PSD panels onto the fNIRS 0-0.5 Hz range.
    fig, axes = plt.subplots(len(traces), 4, figsize=(22, 11), sharex=False)
    for row_idx, trace in enumerate(traces):
        source_target = np.asarray(trace['source_target'], dtype=np.float64)
        obs_target = np.asarray(trace['raw'], dtype=np.float64) - source_target
        source_freqs, source_power = compute_power_spectrum(source_target, float(trace['fs_hz']))
        obs_freqs, obs_power = compute_power_spectrum(obs_target, float(trace['fs_hz']))

        raw_label = 'raw representative channel' if row_idx == 0 else 'raw mean'
        axes[row_idx, 0].plot(trace['time_s'], trace['raw'], color='#BDBDBD', linewidth=1.0, alpha=0.9, label=raw_label)
        axes[row_idx, 0].plot(trace['time_s'], source_target, color=trace['source_color'], linewidth=1.2, label='source target')
        axes[row_idx, 0].set_title(f'{trace["label"]} source target: time')
        axes[row_idx, 0].set_ylabel(trace['unit'])
        axes[row_idx, 0].grid(alpha=0.25)
        axes[row_idx, 0].legend(loc='upper right')

        if source_freqs.size:
            axes[row_idx, 1].plot(source_freqs, source_power, color=trace['source_color'], linewidth=1.2)
            axes[row_idx, 1].set_xlim(0.0, trace['freq_max_hz'])
        axes[row_idx, 1].set_title(f'{trace["label"]} source target: PSD')
        axes[row_idx, 1].set_ylabel('PSD')
        axes[row_idx, 1].grid(alpha=0.25)

        axes[row_idx, 2].plot(trace['time_s'], obs_target, color=trace['obs_color'], linewidth=1.2)
        axes[row_idx, 2].axhline(0.0, color='#BDBDBD', linewidth=0.8, linestyle='--')
        axes[row_idx, 2].set_title(f'{trace["label"]} observation target: time')
        axes[row_idx, 2].set_ylabel(trace['unit'])
        axes[row_idx, 2].grid(alpha=0.25)

        if obs_freqs.size:
            axes[row_idx, 3].plot(obs_freqs, obs_power, color=trace['obs_color'], linewidth=1.2)
            axes[row_idx, 3].set_xlim(0.0, trace['freq_max_hz'])
        axes[row_idx, 3].set_title(f'{trace["label"]} observation target: PSD')
        axes[row_idx, 3].set_ylabel('PSD')
        axes[row_idx, 3].grid(alpha=0.25)

    for col_idx in range(4):
        axes[-1, col_idx].set_xlabel('Time (s)' if col_idx in (0, 2) else 'Frequency (Hz)')
    fig.suptitle('Tokenizer target diagnostics: clean source targets and linear observation residuals', y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_target_psd_comparison(path: Path, bundle: ObservationBundle, run_result: Mapping[str, Any]) -> None:
    if bundle.eeg_obs_raw is None or bundle.fnirs_primary_obs_raw is None or bundle.fnirs_secondary_obs_raw is None:
        return
    if bundle.normalization.get('mode') == 'per_channel_zscore_after_local_selection':
        pred_eeg_raw = destandardize_matrix(run_result['pred_eeg'], bundle.normalization['eeg'])
        pred_primary_raw = destandardize_matrix(run_result['pred_primary'], bundle.normalization['fnirs_primary'])
        pred_secondary_raw = destandardize_matrix(run_result['pred_secondary'], bundle.normalization['fnirs_secondary'])
    else:
        pred_eeg_raw = np.asarray(run_result['pred_eeg'], dtype=np.float64)
        pred_primary_raw = np.asarray(run_result['pred_primary'], dtype=np.float64)
        pred_secondary_raw = np.asarray(run_result['pred_secondary'], dtype=np.float64)

    eeg_obs_raw = np.asarray(bundle.eeg_obs_raw, dtype=np.float64)
    fnirs_primary_raw = np.asarray(bundle.fnirs_primary_obs_raw, dtype=np.float64)
    fnirs_secondary_raw = np.asarray(bundle.fnirs_secondary_obs_raw, dtype=np.float64)
    representative_eeg = int(np.argmax(np.abs(bundle.lead_field))) if bundle.lead_field.size else 0
    representative_eeg_name = bundle.eeg_channel_names[representative_eeg] if bundle.eeg_channel_names else f'EEG_{representative_eeg}'

    traces = [
        {
            'label': f'EEG {representative_eeg_name}',
            'raw': eeg_obs_raw[:, representative_eeg],
            'source': pred_eeg_raw[:, representative_eeg],
            'fs_hz': bundle.eeg_fs_hz,
            'freq_max_hz': min(40.0, 0.5 * bundle.eeg_fs_hz),
            'raw_color': '#7F7F7F',
            'source_color': '#D62728',
            'obs_color': '#1F77B4',
        },
        {
            'label': str(bundle.pair_labels[0]),
            'raw': fnirs_primary_raw.mean(axis=1),
            'source': pred_primary_raw.mean(axis=1),
            'fs_hz': bundle.fnirs_fs_hz,
            'freq_max_hz': min(0.5, 0.5 * bundle.fnirs_fs_hz),
            'raw_color': '#7F7F7F',
            'source_color': '#17BECF',
            'obs_color': '#1F77B4',
        },
        {
            'label': str(bundle.pair_labels[1]),
            'raw': fnirs_secondary_raw.mean(axis=1),
            'source': pred_secondary_raw.mean(axis=1),
            'fs_hz': bundle.fnirs_fs_hz,
            'freq_max_hz': min(0.5, 0.5 * bundle.fnirs_fs_hz),
            'raw_color': '#7F7F7F',
            'source_color': '#FF9896',
            'obs_color': '#9467BD',
        },
    ]

    fig, axes = plt.subplots(len(traces), 1, figsize=(13, 11), sharex=False)
    if len(traces) == 1:
        axes = [axes]

    epsilon = 1e-18
    for axis, trace in zip(axes, traces):
        source = np.asarray(trace['source'], dtype=np.float64)
        raw = np.asarray(trace['raw'], dtype=np.float64)
        obs = raw - source

        raw_freqs, raw_power = compute_power_spectrum(raw, float(trace['fs_hz']))
        source_freqs, source_power = compute_power_spectrum(source, float(trace['fs_hz']))
        obs_freqs, obs_power = compute_power_spectrum(obs, float(trace['fs_hz']))

        if raw_freqs.size:
            axis.semilogy(raw_freqs, np.maximum(raw_power, epsilon), color=trace['raw_color'], linewidth=1.3, label='raw')
        if source_freqs.size:
            axis.semilogy(source_freqs, np.maximum(source_power, epsilon), color=trace['source_color'], linewidth=1.2, label='source target')
        if obs_freqs.size:
            axis.semilogy(obs_freqs, np.maximum(obs_power, epsilon), color=trace['obs_color'], linewidth=1.1, label='observation target')
        axis.set_xlim(0.0, trace['freq_max_hz'])
        axis.set_title(f'{trace["label"]}: raw vs source target vs observation target PSD')
        axis.set_ylabel('PSD')
        axis.grid(alpha=0.25)
        axis.legend(loc='upper right', ncol=3)

    axes[-1].set_xlabel('Frequency (Hz)')
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_r_vs_neighboring_eeg(path: Path, bundle: ObservationBundle, run_result: Mapping[str, Any]) -> None:
    if bundle.eeg_obs_raw is None or not bundle.eeg_channel_names:
        return
    eeg_obs_raw = np.asarray(bundle.eeg_obs_raw, dtype=np.float64)
    r_signal = zscore_vector(np.asarray(run_result['r_estimates_eeg'], dtype=np.float64))
    r_projection = zscore_vector(np.asarray(bundle.r_eeg_projection, dtype=np.float64))
    order = np.argsort(np.abs(bundle.lead_field))[::-1]
    spacing = 3.0
    offsets = spacing * np.arange(len(order) + 2, 0, -1, dtype=np.float64)
    fig, axis = plt.subplots(figsize=(13, 8))
    axis.plot(bundle.eeg_time_s, r_signal + offsets[0], color='#111111', linewidth=1.6)
    axis.plot(bundle.eeg_time_s, r_projection + offsets[1], color='#1f77b4', linewidth=1.1)
    yticks = [offsets[0]]
    ylabels = ['r_hat(t)']
    yticks.append(offsets[1])
    ylabels.append('L^+ y_eeg(t)')
    palette = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(order), 2)))
    for plot_idx, channel_index in enumerate(order.tolist(), start=2):
        channel_trace = zscore_vector(eeg_obs_raw[:, channel_index])
        axis.plot(bundle.eeg_time_s, channel_trace + offsets[plot_idx], color=palette[(plot_idx - 2) % len(palette)], linewidth=1.0)
        yticks.append(offsets[plot_idx])
        ylabels.append(f'{bundle.eeg_channel_names[channel_index]} ({float(bundle.lead_field[channel_index]):+.2f})')
    axis.set_yticks(yticks)
    axis.set_yticklabels(ylabels)
    axis.set_title('Inferred r(t), EEG pseudoinverse proposal, and neighboring EEG channels\n(z-scored and vertically offset for shape comparison)')
    axis.set_xlabel('Time (s)')
    axis.grid(alpha=0.25, axis='x')
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def write_summary(
    path: Path,
    bundle: ObservationBundle,
    spatial_config: SpatialConfig,
    filter_config: FilterConfig,
    reproducibility: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    null_metrics: Mapping[str, Any],
) -> None:
    lines = [
        '# Local Neighborhood Croce Solver Audit',
        '',
        '## Non-Negotiable Modeling Choices',
        '',
        '- r(t) is treated as a signed local neural driver, not as a global whole-head variable.',
        '- Only spatially neighboring EEG and fNIRS channels enter one local anchor audit.',
        '- r(t) has zero endogenous drift; each EEG sample proposes a fresh local r(t).',
        '- fNIRS likelihood is the only particle weighting term.',
        '- Real-data mode uses deviation coordinates around baseline so signs remain explicit and zero-centering is valid.',
        '',
        '## Observation Setup',
        '',
        f'- Mode: {bundle.mode}',
        f'- Pair mode: {bundle.pair_mode}',
        f'- Pair labels: {bundle.pair_labels[0]} / {bundle.pair_labels[1]}',
        f'- EEG fs (Hz): {bundle.eeg_fs_hz:.3f}',
        f'- fNIRS fs (Hz): {bundle.fnirs_fs_hz:.3f}',
        f'- EEG substeps per fNIRS step: {bundle.eeg_substeps_per_fnirs}',
        f'- EEG channels used: {bundle.eeg_indices.tolist()}',
        f'- fNIRS channels used: {bundle.fnirs_indices.tolist()}',
        f'- EEG sign mode: {spatial_config.eeg_sign_mode}',
        f'- EEG raw unit: {bundle.units["eeg"]}',
        f'- fNIRS primary raw unit: {bundle.units["fnirs_primary"]}',
        f'- fNIRS secondary raw unit: {bundle.units["fnirs_secondary"]}',
        f'- Normalization: {bundle.normalization.get("mode", "raw")}',
    ]
    if bundle.metadata:
        lines.extend(
            [
                '',
                '## Source Metadata',
                '',
            ]
        )
        for key, value in bundle.metadata.items():
            lines.append(f'- {key}: {value}')
    lines.extend(
        [
            '',
            '## Baseline Audit',
            '',
            f'- Mean EEG source-target correlation: {baseline_metrics["eeg_corr_mean"]:.4f}',
            f'- Mean {bundle.pair_labels[0]} source-target correlation: {baseline_metrics["fnirs_primary_corr_mean"]:.4f}',
            f'- Mean {bundle.pair_labels[1]} source-target correlation: {baseline_metrics["fnirs_secondary_corr_mean"]:.4f}',
            f'- Mean fNIRS source-target correlation: {baseline_metrics["fnirs_corr_mean"]:.4f}',
            f'- r(t) alpha/delta power ratio: {baseline_metrics["r_alpha_delta_power_ratio"]:.4f}',
            f'- Low-frequency modification ratio ||r_low - r_eeg_low|| / ||r_eeg_low||: {baseline_metrics["r_low_modification_ratio"]:.4f}',
            f'- Full-band modification ratio ||r - r_eeg|| / ||r_eeg||: {baseline_metrics["r_total_modification_ratio"]:.4f}',
            f'- Correlation between r_hat and EEG pseudoinverse proposal: {baseline_metrics["r_projection_corr"]:.4f}',
            f'- Hemodynamic lag to fNIRS (s): {baseline_metrics["hemo_to_fnirs_peak_lag_s"]:.4f}',
            f'- Mean ESS ratio: {baseline_metrics["ess_ratio_mean"]:.4f}',
            f'- Mean r-state posterior std: {baseline_metrics["state_r_std_mean"]:.4f}',
            f'- Log-likelihood: {baseline_metrics["log_likelihood"]:.4f}',
            '',
            '## Reproducibility Audit',
            '',
            f'- Median pairwise r-correlation across seeds: {reproducibility["seed_pairwise_r_corr_median"]:.4f}',
            f'- Mean pairwise r-correlation across seeds: {reproducibility["seed_pairwise_r_corr_mean"]:.4f}',
            f'- Pairwise comparisons: {reproducibility["seed_pairwise_count"]}',
            '',
            '## Null Audit',
            '',
            f'- Time-shift null log-likelihood delta (baseline - null): {null_metrics["time_shift_log_likelihood_delta"]:.4f}',
            f'- Time-shift null mean fNIRS correlation delta (baseline - null): {null_metrics["time_shift_fnirs_corr_delta"]:.4f}',
        ]
    )
    if 'spatial_null_log_likelihood_delta' in null_metrics:
        lines.extend(
            [
                f'- Spatial-null log-likelihood delta (baseline - null): {null_metrics["spatial_null_log_likelihood_delta"]:.4f}',
                f'- Spatial-null mean fNIRS correlation delta (baseline - null): {null_metrics["spatial_null_fnirs_corr_delta"]:.4f}',
            ]
        )
    if bundle.true_r_eeg is not None:
        lines.extend(
            [
                '',
                '## Synthetic Recovery',
                '',
                f'- Normalized RMSE r(t): {baseline_metrics["r_norm_rmse"]:.6f}',
                f'- High-frequency correlation r_hat vs r_true: {baseline_metrics["r_high_corr"]:.4f}',
                f'- Low-frequency correlation r_hat vs r_true: {baseline_metrics["r_low_corr"]:.4f}',
            ]
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def configure_torch_threads(n_threads: int) -> None:
    """Set torch/OpenMP thread counts to avoid oversubscription with fork workers.

    Must be called before any torch parallel operation. With fork-based
    multiprocessing, child processes inherit these settings.
    """
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
    if torch is not None:
        torch.set_num_threads(n_threads)
        try:
            torch.set_num_interop_threads(n_threads)
        except RuntimeError:
            pass


def main() -> None:
    args = parse_args()
    configure_torch_threads(int(args.torch_threads))
    spatial_config = SpatialConfig(
        eeg_neighbors=int(args.eeg_neighbors),
        fnirs_neighbors=int(args.fnirs_neighbors),
        eeg_radius_mm=float(args.eeg_radius_mm),
        fnirs_radius_mm=float(args.fnirs_radius_mm),
        eeg_sigma_mm=float(args.eeg_sigma_mm),
        fnirs_sigma_mm=float(args.fnirs_sigma_mm),
        eeg_sign_mode=str(args.eeg_sign_mode),
    )
    filter_config = FilterConfig(
        integration_dt_s=float(args.integration_dt),
        observation_fs_hz=float(args.observation_fs),
        num_particles=int(args.num_particles),
        resample_fraction=float(args.resample_fraction),
        prior_std=parse_vector(args.prior_std, name='prior-std'),
        state_noise_std=parse_vector(args.state_noise_std, name='state-noise-std'),
        sigma_prop=float(args.sigma_prop),
        sigma_nirs=float(args.sigma_nirs),
        seed_list=parse_seed_list(args.seed_list),
        time_shift_null_s=float(args.time_shift_null_s),
        run_spatial_null=bool(args.run_spatial_null),
        solver_backend=str(args.solver_backend),
        torch_device=str(args.torch_device),
    )
    filter_config.prior_std[4] = 0.0
    filter_config.state_noise_std[4] = 0.0
    output_dir = resolve_output_dir(args.output_dir, args.mode)

    if args.mode == 'synthetic':
        bundle = simulate_synthetic_bundle(args, spatial_config)
    elif args.mode == 'dataset':
        bundle = load_dataset_bundle(args, spatial_config)
    else:
        bundle = load_real_bundle(args, spatial_config)

    filter_config = FilterConfig(
        integration_dt_s=float(1.0 / bundle.eeg_fs_hz),
        observation_fs_hz=bundle.fnirs_fs_hz,
        num_particles=filter_config.num_particles,
        resample_fraction=filter_config.resample_fraction,
        prior_std=filter_config.prior_std,
        state_noise_std=filter_config.state_noise_std,
        sigma_prop=filter_config.sigma_prop,
        sigma_nirs=filter_config.sigma_nirs,
        seed_list=filter_config.seed_list,
        time_shift_null_s=filter_config.time_shift_null_s,
        run_spatial_null=filter_config.run_spatial_null,
        solver_backend=filter_config.solver_backend,
        torch_device=filter_config.torch_device,
    )

    params = ModelParams()
    run_results = [run_particle_filter(bundle, filter_config, params, seed=seed) for seed in filter_config.seed_list]
    baseline_result = run_results[0]
    reproducibility = summarise_seed_reproducibility(run_results)
    baseline_metrics = compute_fit_metrics(bundle, baseline_result, filter_config)

    time_shift_bundle = build_null_bundle(bundle, time_shift_s=filter_config.time_shift_null_s, spatial_permutation=False)
    time_shift_result = run_particle_filter(time_shift_bundle, filter_config, params, seed=filter_config.seed_list[0])
    time_shift_metrics = compute_fit_metrics(time_shift_bundle, time_shift_result, filter_config)
    null_metrics: Dict[str, Any] = {
        'time_shift_log_likelihood_delta': float(baseline_metrics['log_likelihood'] - time_shift_metrics['log_likelihood']),
        'time_shift_fnirs_corr_delta': float(baseline_metrics['fnirs_corr_mean'] - time_shift_metrics['fnirs_corr_mean']),
    }

    if filter_config.run_spatial_null:
        spatial_bundle = build_null_bundle(bundle, time_shift_s=0.0, spatial_permutation=True)
        spatial_result = run_particle_filter(spatial_bundle, filter_config, params, seed=filter_config.seed_list[0])
        spatial_metrics = compute_fit_metrics(spatial_bundle, spatial_result, filter_config)
        null_metrics.update(
            {
                'spatial_null_log_likelihood_delta': float(baseline_metrics['log_likelihood'] - spatial_metrics['log_likelihood']),
                'spatial_null_fnirs_corr_delta': float(baseline_metrics['fnirs_corr_mean'] - spatial_metrics['fnirs_corr_mean']),
            }
        )

    manifest = {
        'mode': args.mode,
        'output_dir': str(output_dir.relative_to(PROJECT_ROOT)),
        'spatial_config': asdict(spatial_config),
        'filter_config': {
            'integration_dt_s': filter_config.integration_dt_s,
            'observation_fs_hz': filter_config.observation_fs_hz,
            'num_particles': filter_config.num_particles,
            'resample_fraction': filter_config.resample_fraction,
            'prior_std': filter_config.prior_std.tolist(),
            'state_noise_std': filter_config.state_noise_std.tolist(),
            'sigma_prop': filter_config.sigma_prop,
            'sigma_nirs': filter_config.sigma_nirs,
            'solver_backend': filter_config.solver_backend,
            'torch_device': filter_config.torch_device,
            'seed_list': list(filter_config.seed_list),
            'time_shift_null_s': filter_config.time_shift_null_s,
            'run_spatial_null': filter_config.run_spatial_null,
        },
        'model_params': asdict(params),
        'units': bundle.units,
        'pair_labels': list(bundle.pair_labels),
        'normalization': bundle.normalization,
        'metadata': bundle.metadata,
        'selected_channels': {
            'eeg_indices': bundle.eeg_indices.tolist(),
            'fnirs_indices': bundle.fnirs_indices.tolist(),
        },
        'sampling': {
            'eeg_fs_hz': bundle.eeg_fs_hz,
            'fnirs_fs_hz': bundle.fnirs_fs_hz,
            'eeg_substeps_per_fnirs': bundle.eeg_substeps_per_fnirs,
        },
        'channel_names': {
            'eeg': list(bundle.eeg_channel_names),
            'fnirs_primary': list(bundle.fnirs_primary_channel_names),
            'fnirs_secondary': list(bundle.fnirs_secondary_channel_names),
        },
        'anchor_position_mm': bundle.anchor_position_mm.tolist(),
        'lead_field': bundle.lead_field.tolist(),
        'jac_primary': bundle.jac_primary.tolist(),
        'jac_secondary': bundle.jac_secondary.tolist(),
    }
    metrics = {
        'baseline': to_serializable_metrics(baseline_metrics),
        'reproducibility': to_serializable_metrics(reproducibility),
        'nulls': to_serializable_metrics(null_metrics),
    }
    write_json(output_dir / 'run_manifest.json', manifest)
    write_json(output_dir / 'metrics.json', metrics)
    write_summary(
        output_dir / 'design_summary.md',
        bundle=bundle,
        spatial_config=spatial_config,
        filter_config=filter_config,
        reproducibility=reproducibility,
        baseline_metrics=baseline_metrics,
        null_metrics=null_metrics,
    )
    plot_state_timecourses(output_dir / 'state_timecourses.png', bundle, baseline_result)
    plot_time_colored_state_space(output_dir / 'state_space_time_colored.png', bundle, baseline_result)
    plot_observation_reconstruction(output_dir / 'observation_reconstruction.png', bundle, baseline_result)
    plot_target_time_frequency_diagnostics(output_dir / 'target_time_frequency_diagnostics.png', bundle, baseline_result)
    plot_target_psd_comparison(output_dir / 'target_psd_comparison.png', bundle, baseline_result)
    plot_r_vs_neighboring_eeg(output_dir / 'r_vs_neighboring_eeg.png', bundle, baseline_result)

    print(f'Results saved to {output_dir}')
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
