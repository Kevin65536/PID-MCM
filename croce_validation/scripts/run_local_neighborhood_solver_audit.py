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
from scipy.signal import butter, sosfiltfilt


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
    lambda_r: float = 0.10


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
    seed_list: Tuple[int, ...]
    time_shift_null_s: float
    run_spatial_null: bool


@dataclass(frozen=True)
class ObservationBundle:
    mode: str
    pair_mode: str
    time_s: np.ndarray
    eeg_obs: np.ndarray
    fnirs_primary_obs: np.ndarray
    fnirs_secondary_obs: np.ndarray
    eeg_positions_mm: np.ndarray
    fnirs_positions_mm: np.ndarray
    anchor_position_mm: np.ndarray
    eeg_indices: np.ndarray
    fnirs_indices: np.ndarray
    lead_field: np.ndarray
    jac_primary: np.ndarray
    jac_secondary: np.ndarray
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
    parser.add_argument('--segment-start-s', type=float, default=60.0, help='Continuous segment start in seconds for dataset mode')
    parser.add_argument('--segment-duration-s', type=float, default=120.0, help='Continuous segment duration in seconds for dataset mode')
    parser.add_argument('--use-artifact-eeg', action='store_true', help='Prefer the EEG recordings stored under the ocular-artifact folder in dataset mode')

    parser.add_argument('--duration-s', type=float, default=60.0, help='Synthetic duration in seconds')
    parser.add_argument('--observation-fs', type=float, default=10.0, help='Synthetic observation sampling rate in Hz')
    parser.add_argument('--integration-dt', type=float, default=0.05, help='Transition integration step in seconds')
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
        default='0.05,0.05,0.05,0.05,0.05',
        help='Comma-separated prior std for s,delta_f,delta_hbo,delta_hb,r',
    )
    parser.add_argument(
        '--state-noise-std',
        default='0.02,0.015,0.015,0.015,0.03',
        help='Comma-separated transition-noise std for s,delta_f,delta_hbo,delta_hb,r',
    )
    parser.add_argument('--seed-list', default='11,23,37,47,59', help='Comma-separated random seeds for reproducibility audit')
    parser.add_argument('--time-shift-null-s', type=float, default=8.0, help='Shift used for the timing null in seconds')
    parser.add_argument('--run-spatial-null', action='store_true', help='Also run a local channel-order permutation null')

    parser.add_argument('--eeg-unit', default='uV', help='Raw EEG unit label written to the manifest')
    parser.add_argument('--fnirs-primary-unit', default='a.u.', help='Raw unit label for the first fNIRS observation family')
    parser.add_argument('--fnirs-secondary-unit', default='a.u.', help='Raw unit label for the second fNIRS observation family')
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
    dr = -float(params.lambda_r) * float(r)
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
    jac[4, 4] = -float(params.lambda_r)
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


def predict_observations(
    particles: np.ndarray,
    lead_field: np.ndarray,
    jac_primary: np.ndarray,
    jac_secondary: np.ndarray,
    pair_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_eeg = particles[:, 4:5] * lead_field.reshape(1, -1)
    if pair_mode == 'chromophore':
        pred_primary = particles[:, 2:3] * jac_primary.reshape(1, -1)
        pred_secondary = particles[:, 3:4] * jac_secondary.reshape(1, -1)
    else:
        pred_primary = (0.35 * particles[:, 2:3] + 1.00 * particles[:, 3:4]) * jac_primary.reshape(1, -1)
        pred_secondary = (1.00 * particles[:, 2:3] + 0.25 * particles[:, 3:4]) * jac_secondary.reshape(1, -1)
    return pred_eeg, pred_primary, pred_secondary


def run_particle_filter(
    bundle: ObservationBundle,
    filter_config: FilterConfig,
    params: ModelParams,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    num_particles = int(filter_config.num_particles)
    obs_dt = 1.0 / float(filter_config.observation_fs_hz)

    particles = rng.normal(
        loc=np.zeros(5, dtype=np.float64).reshape(1, 5),
        scale=filter_config.prior_std.reshape(1, 5),
        size=(num_particles, 5),
    )
    particles[:, 1:4] = np.clip(particles[:, 1:4], -0.95, None)
    weights = np.full(num_particles, 1.0 / float(num_particles), dtype=np.float64)

    estimates = np.zeros((bundle.time_s.shape[0], 5), dtype=np.float64)
    state_std = np.zeros((bundle.time_s.shape[0], 5), dtype=np.float64)
    ess_trace = np.zeros(bundle.time_s.shape[0], dtype=np.float64)
    log_likelihood_total = 0.0

    integration_dt = float(filter_config.integration_dt_s)
    stride = int(round(obs_dt / integration_dt)) if integration_dt > 0.0 else 1
    use_stride = stride >= 1 and np.isclose(stride * integration_dt, obs_dt, atol=1e-9)

    for step in range(bundle.time_s.shape[0]):
        if use_stride:
            for _ in range(stride):
                for idx in range(num_particles):
                    particles[idx] = local_linearized_step(particles[idx], integration_dt, params)
                particles += rng.normal(
                    loc=0.0,
                    scale=filter_config.state_noise_std.reshape(1, 5) * np.sqrt(integration_dt),
                    size=particles.shape,
                )
                particles[:, 1:4] = np.clip(particles[:, 1:4], -0.95, None)
        else:
            for idx in range(num_particles):
                particles[idx] = local_linearized_step(particles[idx], obs_dt, params)
            particles += rng.normal(
                loc=0.0,
                scale=filter_config.state_noise_std.reshape(1, 5) * np.sqrt(obs_dt),
                size=particles.shape,
            )
            particles[:, 1:4] = np.clip(particles[:, 1:4], -0.95, None)

        pred_eeg, pred_primary, pred_secondary = predict_observations(
            particles,
            bundle.lead_field,
            bundle.jac_primary,
            bundle.jac_secondary,
            bundle.pair_mode,
        )
        log_weights = np.log(np.clip(weights, 1e-300, None))
        log_weights += -0.5 * np.sum(np.square(bundle.eeg_obs[step].reshape(1, -1) - pred_eeg), axis=1)
        log_weights += -0.5 * np.sum(np.square(bundle.fnirs_primary_obs[step].reshape(1, -1) - pred_primary), axis=1)
        log_weights += -0.5 * np.sum(np.square(bundle.fnirs_secondary_obs[step].reshape(1, -1) - pred_secondary), axis=1)

        max_log_weight = np.max(log_weights)
        stable = np.exp(log_weights - max_log_weight)
        weights = stable / np.clip(stable.sum(), 1e-12, None)
        log_likelihood_total += max_log_weight + np.log(np.clip(stable.sum(), 1e-12, None))

        estimates[step] = np.sum(particles * weights.reshape(-1, 1), axis=0)
        centered = particles - estimates[step].reshape(1, -1)
        state_std[step] = np.sqrt(np.sum(np.square(centered) * weights.reshape(-1, 1), axis=0))
        ess = 1.0 / np.sum(np.square(weights))
        ess_trace[step] = ess
        if ess < filter_config.resample_fraction * num_particles:
            indices = systematic_resample(weights, rng)
            particles = particles[indices]
            weights.fill(1.0 / float(num_particles))

    pred_eeg, pred_primary, pred_secondary = predict_observations(
        estimates,
        bundle.lead_field,
        bundle.jac_primary,
        bundle.jac_secondary,
        bundle.pair_mode,
    )
    return {
        'seed': seed,
        'state_estimates': estimates,
        'state_std': state_std,
        'ess_trace': ess_trace,
        'log_likelihood': float(log_likelihood_total),
        'pred_eeg': pred_eeg,
        'pred_primary': pred_primary,
        'pred_secondary': pred_secondary,
    }


def compute_fit_metrics(
    bundle: ObservationBundle,
    run_result: Mapping[str, Any],
    filter_config: FilterConfig,
) -> Dict[str, Any]:
    per_eeg_corr = [safe_correlation(bundle.eeg_obs[:, idx], run_result['pred_eeg'][:, idx]) for idx in range(bundle.eeg_obs.shape[1])]
    per_primary_corr = [safe_correlation(bundle.fnirs_primary_obs[:, idx], run_result['pred_primary'][:, idx]) for idx in range(bundle.fnirs_primary_obs.shape[1])]
    per_secondary_corr = [safe_correlation(bundle.fnirs_secondary_obs[:, idx], run_result['pred_secondary'][:, idx]) for idx in range(bundle.fnirs_secondary_obs.shape[1])]

    lag_s, lag_corr = lag_peak_seconds(
        run_result['state_estimates'][:, 4],
        0.5 * (bundle.fnirs_primary_obs.mean(axis=1) + bundle.fnirs_secondary_obs.mean(axis=1)),
        fs_hz=filter_config.observation_fs_hz,
    )
    metrics: Dict[str, Any] = {
        'eeg_corr_mean': float(np.nanmean(per_eeg_corr)),
        'eeg_rmse': rmse(bundle.eeg_obs, run_result['pred_eeg']),
        'fnirs_primary_corr_mean': float(np.nanmean(per_primary_corr)),
        'fnirs_primary_rmse': rmse(bundle.fnirs_primary_obs, run_result['pred_primary']),
        'fnirs_secondary_corr_mean': float(np.nanmean(per_secondary_corr)),
        'fnirs_secondary_rmse': rmse(bundle.fnirs_secondary_obs, run_result['pred_secondary']),
        'state_to_fnirs_peak_lag_s': lag_s,
        'state_to_fnirs_peak_corr': lag_corr,
        'ess_ratio_mean': float(np.mean(run_result['ess_trace']) / max(filter_config.num_particles, 1)),
        'state_r_std_mean': float(np.mean(run_result['state_std'][:, 4])),
        'log_likelihood': float(run_result['log_likelihood']),
    }
    if bundle.true_states is not None:
        metrics.update(
            {
                'rmse_r': rmse(bundle.true_states[:, 4], run_result['state_estimates'][:, 4]),
                'rmse_delta_hbo': rmse(bundle.true_states[:, 2], run_result['state_estimates'][:, 2]),
                'rmse_delta_hb': rmse(bundle.true_states[:, 3], run_result['state_estimates'][:, 3]),
            }
        )
    return metrics


def summarise_seed_reproducibility(run_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    pairwise_corr: List[float] = []
    for idx in range(len(run_results)):
        for jdx in range(idx + 1, len(run_results)):
            corr = safe_correlation(run_results[idx]['state_estimates'][:, 4], run_results[jdx]['state_estimates'][:, 4])
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
    shift_samples = int(round(time_shift_s * (1.0 / (bundle.time_s[1] - bundle.time_s[0])))) if bundle.time_s.shape[0] > 1 else 0
    eeg_obs = np.asarray(bundle.eeg_obs, dtype=np.float64)
    fnirs_primary_obs = np.asarray(bundle.fnirs_primary_obs, dtype=np.float64)
    fnirs_secondary_obs = np.asarray(bundle.fnirs_secondary_obs, dtype=np.float64)
    if shift_samples:
        eeg_obs = np.roll(eeg_obs, shift_samples, axis=0)
    if spatial_permutation:
        eeg_obs = permute_channels(eeg_obs)
        fnirs_primary_obs = permute_channels(fnirs_primary_obs)
        fnirs_secondary_obs = permute_channels(fnirs_secondary_obs)
    return ObservationBundle(
        mode=bundle.mode,
        pair_mode=bundle.pair_mode,
        time_s=bundle.time_s,
        eeg_obs=eeg_obs,
        fnirs_primary_obs=fnirs_primary_obs,
        fnirs_secondary_obs=fnirs_secondary_obs,
        eeg_positions_mm=bundle.eeg_positions_mm,
        fnirs_positions_mm=bundle.fnirs_positions_mm,
        anchor_position_mm=bundle.anchor_position_mm,
        eeg_indices=bundle.eeg_indices,
        fnirs_indices=bundle.fnirs_indices,
        lead_field=bundle.lead_field,
        jac_primary=bundle.jac_primary,
        jac_secondary=bundle.jac_secondary,
        normalization=bundle.normalization,
        units=bundle.units,
        pair_labels=bundle.pair_labels,
        eeg_obs_raw=bundle.eeg_obs_raw,
        fnirs_primary_obs_raw=bundle.fnirs_primary_obs_raw,
        fnirs_secondary_obs_raw=bundle.fnirs_secondary_obs_raw,
        eeg_channel_names=bundle.eeg_channel_names,
        fnirs_primary_channel_names=bundle.fnirs_primary_channel_names,
        fnirs_secondary_channel_names=bundle.fnirs_secondary_channel_names,
        metadata=bundle.metadata,
        true_states=bundle.true_states,
        true_lead_field=bundle.true_lead_field,
        true_jac_primary=bundle.true_jac_primary,
        true_jac_secondary=bundle.true_jac_secondary,
    )


def simulate_synthetic_bundle(
    args: argparse.Namespace,
    spatial_config: SpatialConfig,
) -> ObservationBundle:
    rng = np.random.default_rng(20260521)
    observation_fs = float(args.observation_fs)
    observation_dt = 1.0 / observation_fs
    integration_dt = float(args.integration_dt)
    stride = int(round(observation_dt / integration_dt))
    if stride < 1 or not np.isclose(stride * integration_dt, observation_dt, atol=1e-9):
        raise ValueError('Synthetic observation-fs and integration-dt must define an integer stride')

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
    total_steps = int(round(float(args.duration_s) / integration_dt))
    states = np.zeros((total_steps + 1, 5), dtype=np.float64)
    for step in range(total_steps):
        next_state = local_linearized_step(states[step], integration_dt, params)
        next_state[4] += 0.03 * np.sqrt(integration_dt) * rng.normal()
        next_state[1:4] += np.asarray([0.004, 0.004, 0.004], dtype=np.float64) * np.sqrt(integration_dt) * rng.normal(size=3)
        next_state[1:4] = np.clip(next_state[1:4], -0.95, None)
        states[step + 1] = next_state

    observation_indices = np.arange(0, total_steps + 1, stride, dtype=np.int64)
    time_obs = observation_indices.astype(np.float64) * integration_dt
    states_obs = states[observation_indices]

    eeg_clean = states_obs[:, 4:5] * true_lead.reshape(1, -1)
    fnirs_primary_clean = (0.35 * states_obs[:, 2:3] + 1.00 * states_obs[:, 3:4]) * true_jac_primary.reshape(1, -1)
    fnirs_secondary_clean = (1.00 * states_obs[:, 2:3] + 0.25 * states_obs[:, 3:4]) * true_jac_secondary.reshape(1, -1)

    eeg_noisy, _ = add_awgn(eeg_clean, float(args.snr_db), rng)
    fnirs_primary_noisy, _ = add_awgn(fnirs_primary_clean, float(args.snr_db), rng)
    fnirs_secondary_noisy, _ = add_awgn(fnirs_secondary_clean, float(args.snr_db), rng)

    approx_lead = build_signed_eeg_weights(
        eeg_obs=eeg_noisy,
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
        eeg_obs=eeg_noisy,
        fnirs_primary_obs=fnirs_primary_noisy,
        fnirs_secondary_obs=fnirs_secondary_noisy,
        eeg_positions_mm=local_eeg_positions,
        fnirs_positions_mm=local_fnirs_positions,
        anchor_position_mm=anchor,
        eeg_indices=eeg_indices,
        fnirs_indices=fnirs_indices,
        lead_field=approx_lead,
        jac_primary=approx_jac_primary,
        jac_secondary=approx_jac_secondary,
        normalization={'mode': 'raw_observation_units'},
        units={
            'eeg': args.eeg_unit,
            'fnirs_primary': args.fnirs_primary_unit,
            'fnirs_secondary': args.fnirs_secondary_unit,
        },
        pair_labels=('690_like', '830_like'),
        eeg_obs_raw=eeg_noisy,
        fnirs_primary_obs_raw=fnirs_primary_noisy,
        fnirs_secondary_obs_raw=fnirs_secondary_noisy,
        eeg_channel_names=tuple(f'EEG_{index}' for index in eeg_indices.tolist()),
        fnirs_primary_channel_names=tuple(f'690_like_{index}' for index in fnirs_indices.tolist()),
        fnirs_secondary_channel_names=tuple(f'830_like_{index}' for index in fnirs_indices.tolist()),
        metadata={'signal_source': 'synthetic_local_source'},
        true_states=states_obs,
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

        if 'anchor_position_mm' in bundle_npz:
            anchor = np.asarray(bundle_npz['anchor_position_mm'], dtype=np.float64).ravel()
        elif 'eeg_anchor_index' in bundle_npz:
            anchor = eeg_positions[int(np.asarray(bundle_npz['eeg_anchor_index']).item())]
        elif 'fnirs_anchor_index' in bundle_npz:
            anchor = fnirs_positions[int(np.asarray(bundle_npz['fnirs_anchor_index']).item())]
        else:
            anchor = np.mean(np.concatenate([eeg_positions, fnirs_positions], axis=0), axis=0)

        if 'fnirs_690' in bundle_npz and 'fnirs_830' in bundle_npz:
            fnirs_primary = np.asarray(bundle_npz['fnirs_690'], dtype=np.float64)
            fnirs_secondary = np.asarray(bundle_npz['fnirs_830'], dtype=np.float64)
            pair_mode = 'wavelength'
            pair_labels = ('690', '830')
        elif 'fnirs_hbo' in bundle_npz and 'fnirs_hb' in bundle_npz:
            fnirs_primary = np.asarray(bundle_npz['fnirs_hbo'], dtype=np.float64)
            fnirs_secondary = np.asarray(bundle_npz['fnirs_hb'], dtype=np.float64)
            pair_mode = 'chromophore'
            pair_labels = ('HbO', 'Hb')
        elif 'fnirs_primary' in bundle_npz and 'fnirs_secondary' in bundle_npz:
            fnirs_primary = np.asarray(bundle_npz['fnirs_primary'], dtype=np.float64)
            fnirs_secondary = np.asarray(bundle_npz['fnirs_secondary'], dtype=np.float64)
            pair_mode = 'wavelength'
            pair_labels = ('primary', 'secondary')
        else:
            raise ValueError('NPZ bundle must provide one of {fnirs_690, fnirs_830}, {fnirs_hbo, fnirs_hb}, or {fnirs_primary, fnirs_secondary}')

    eeg_indices = select_local_indices(eeg_positions, anchor, spatial_config.eeg_neighbors, spatial_config.eeg_radius_mm)
    fnirs_indices = select_local_indices(fnirs_positions, anchor, spatial_config.fnirs_neighbors, spatial_config.fnirs_radius_mm)
    local_eeg_positions = eeg_positions[eeg_indices]
    local_fnirs_positions = fnirs_positions[fnirs_indices]

    eeg_local = np.asarray(eeg[:, eeg_indices], dtype=np.float64)
    eeg_local_ds = downsample_signed_channels(eeg_local, source_fs_hz=eeg_fs_hz, target_fs_hz=fnirs_fs_hz)
    fnirs_primary_local = np.asarray(fnirs_primary[:, fnirs_indices], dtype=np.float64)
    fnirs_secondary_local = np.asarray(fnirs_secondary[:, fnirs_indices], dtype=np.float64)

    length = min(eeg_local_ds.shape[0], fnirs_primary_local.shape[0], fnirs_secondary_local.shape[0])
    eeg_local_ds = eeg_local_ds[:length]
    fnirs_primary_local = fnirs_primary_local[:length]
    fnirs_secondary_local = fnirs_secondary_local[:length]
    time_s = np.arange(length, dtype=np.float64) / fnirs_fs_hz

    lead_field = build_signed_eeg_weights(
        eeg_obs=eeg_local_ds,
        eeg_positions_mm=local_eeg_positions,
        anchor_position_mm=anchor,
        sigma_mm=spatial_config.eeg_sigma_mm,
        sign_mode=spatial_config.eeg_sign_mode,
    )
    jac_primary = build_positive_weights(local_fnirs_positions, anchor, spatial_config.fnirs_sigma_mm)
    jac_secondary = build_positive_weights(local_fnirs_positions, anchor, spatial_config.fnirs_sigma_mm)

    eeg_norm, eeg_stats = standardize_matrix(eeg_local_ds)
    fnirs_primary_norm, fnirs_primary_stats = standardize_matrix(fnirs_primary_local)
    fnirs_secondary_norm, fnirs_secondary_stats = standardize_matrix(fnirs_secondary_local)

    return ObservationBundle(
        mode='npz',
        pair_mode=pair_mode,
        time_s=time_s,
        eeg_obs=eeg_norm,
        fnirs_primary_obs=fnirs_primary_norm,
        fnirs_secondary_obs=fnirs_secondary_norm,
        eeg_positions_mm=local_eeg_positions,
        fnirs_positions_mm=local_fnirs_positions,
        anchor_position_mm=anchor,
        eeg_indices=eeg_indices,
        fnirs_indices=fnirs_indices,
        lead_field=lead_field,
        jac_primary=jac_primary,
        jac_secondary=jac_secondary,
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
        eeg_obs_raw=eeg_local_ds,
        fnirs_primary_obs_raw=fnirs_primary_local,
        fnirs_secondary_obs_raw=fnirs_secondary_local,
        metadata={'signal_source': 'npz_bundle'},
    )


def load_dataset_bundle(args: argparse.Namespace, spatial_config: SpatialConfig) -> ObservationBundle:
    dataset = MultiModalEEGfNIRSDataset(
        data_root=args.data_root,
        subject_ids=[int(args.subject_id)],
        task='motor_imagery',
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
    session_idx = int(args.session_idx)
    if session_idx < 0 or session_idx >= min(len(eeg_session_list), len(fnirs_session_list)):
        raise ValueError(f'session-idx {session_idx} is out of range')

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

    eeg_full = np.asarray(eeg_session_list[session_idx], dtype=np.float64)
    fnirs_full = np.asarray(fnirs_session_list[session_idx], dtype=np.float64)
    eeg_fs_hz = float(eeg_info['fs'])
    fnirs_fs_hz = float(fnirs_info['fs'])
    segment_start_eeg = max(int(round(float(args.segment_start_s) * eeg_fs_hz)), 0)
    segment_end_eeg = min(int(round((float(args.segment_start_s) + float(args.segment_duration_s)) * eeg_fs_hz)), eeg_full.shape[0])
    segment_start_fnirs = max(int(round(float(args.segment_start_s) * fnirs_fs_hz)), 0)
    segment_end_fnirs = min(int(round((float(args.segment_start_s) + float(args.segment_duration_s)) * fnirs_fs_hz)), fnirs_full.shape[0])
    if segment_end_eeg - segment_start_eeg < 32 or segment_end_fnirs - segment_start_fnirs < 16:
        raise ValueError('Selected dataset segment is too short for local solver audit')

    eeg_local_raw = eeg_full[segment_start_eeg:segment_end_eeg, :][:, eeg_indices]
    eeg_local_ds = downsample_signed_channels(eeg_local_raw, source_fs_hz=eeg_fs_hz, target_fs_hz=fnirs_fs_hz)
    fnirs_primary_local = fnirs_full[segment_start_fnirs:segment_end_fnirs, :][:, primary_indices]
    fnirs_secondary_local = fnirs_full[segment_start_fnirs:segment_end_fnirs, :][:, secondary_indices]
    length = min(eeg_local_ds.shape[0], fnirs_primary_local.shape[0], fnirs_secondary_local.shape[0])
    eeg_local_ds = eeg_local_ds[:length]
    fnirs_primary_local = fnirs_primary_local[:length]
    fnirs_secondary_local = fnirs_secondary_local[:length]
    time_s = float(args.segment_start_s) + np.arange(length, dtype=np.float64) / fnirs_fs_hz

    local_eeg_positions = np.asarray(adjacency.eeg_positions_2d[eeg_indices], dtype=np.float64)
    local_fnirs_positions = np.asarray(adjacency.fnirs_channel_positions_2d[primary_indices], dtype=np.float64)
    anchor_position_2d = np.asarray(adjacency.fnirs_channel_positions_2d[primary_indices[0]], dtype=np.float64)
    lead_field = build_signed_eeg_weights(
        eeg_obs=eeg_local_ds,
        eeg_positions_mm=local_eeg_positions,
        anchor_position_mm=anchor_position_2d,
        sigma_mm=spatial_config.eeg_sigma_mm,
        sign_mode=spatial_config.eeg_sign_mode,
    )
    jac_primary = build_positive_weights(local_fnirs_positions, anchor_position_2d, spatial_config.fnirs_sigma_mm)
    jac_secondary = build_positive_weights(local_fnirs_positions, anchor_position_2d, spatial_config.fnirs_sigma_mm)

    eeg_norm, eeg_stats = standardize_matrix(eeg_local_ds)
    fnirs_primary_norm, fnirs_primary_stats = standardize_matrix(fnirs_primary_local)
    fnirs_secondary_norm, fnirs_secondary_stats = standardize_matrix(fnirs_secondary_local)

    return ObservationBundle(
        mode='dataset',
        pair_mode='wavelength',
        time_s=time_s,
        eeg_obs=eeg_norm,
        fnirs_primary_obs=fnirs_primary_norm,
        fnirs_secondary_obs=fnirs_secondary_norm,
        eeg_positions_mm=local_eeg_positions,
        fnirs_positions_mm=local_fnirs_positions,
        anchor_position_mm=anchor_position_2d,
        eeg_indices=eeg_indices,
        fnirs_indices=primary_indices,
        lead_field=lead_field,
        jac_primary=jac_primary,
        jac_secondary=jac_secondary,
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
        eeg_obs_raw=eeg_local_ds,
        fnirs_primary_obs_raw=fnirs_primary_local,
        fnirs_secondary_obs_raw=fnirs_secondary_local,
        eeg_channel_names=tuple(str(adjacency.eeg_channel_names[index]) for index in eeg_indices.tolist()),
        fnirs_primary_channel_names=tuple(str(adjacency.fnirs_channel_names[index]) for index in primary_indices.tolist()),
        fnirs_secondary_channel_names=tuple(str(adjacency.fnirs_channel_names[index]) for index in secondary_indices.tolist()),
        metadata={
            'signal_source': 'real_physiological_continuous_signal',
            'data_root': str(args.data_root),
            'subject_id': int(args.subject_id),
            'session_idx': session_idx,
            'segment_start_s': float(args.segment_start_s),
            'segment_duration_s': float(args.segment_duration_s),
            'anchor_fnirs_base': str(anchor_base_name),
            'local_fnirs_bases': list(local_base_names),
        },
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

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(bundle.time_s, eeg_obs_raw.mean(axis=1), color='#111111', linewidth=1.3, label='Observed EEG neighborhood mean')
    axes[0].plot(bundle.time_s, pred_eeg_raw.mean(axis=1), color='#D62728', linewidth=1.1, label='Reconstructed EEG neighborhood mean')
    axes[0].plot(bundle.time_s, eeg_obs_raw[:, representative_eeg], color='#7F7F7F', linewidth=0.8, alpha=0.85, label=f'Observed {bundle.eeg_channel_names[representative_eeg]}')
    axes[0].plot(bundle.time_s, pred_eeg_raw[:, representative_eeg], color='#FF9896', linewidth=0.8, alpha=0.95, label=f'Reconstructed {bundle.eeg_channel_names[representative_eeg]}')
    axes[0].set_ylabel(bundle.units['eeg'])
    axes[0].set_title('Real local EEG neighborhood: actual vs reconstructed')
    axes[0].legend(loc='upper right', ncol=2)
    axes[0].grid(alpha=0.25)

    axes[1].plot(bundle.time_s, fnirs_primary_raw.mean(axis=1), color='#1F77B4', linewidth=1.3, label=f'Observed {bundle.pair_labels[0]} mean')
    axes[1].plot(bundle.time_s, pred_primary_raw.mean(axis=1), color='#17BECF', linewidth=1.1, label=f'Reconstructed {bundle.pair_labels[0]} mean')
    axes[1].set_ylabel(bundle.units['fnirs_primary'])
    axes[1].set_title(f'Local {bundle.pair_labels[0]} neighborhood: actual vs reconstructed')
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.25)

    axes[2].plot(bundle.time_s, fnirs_secondary_raw.mean(axis=1), color='#D62728', linewidth=1.3, label=f'Observed {bundle.pair_labels[1]} mean')
    axes[2].plot(bundle.time_s, pred_secondary_raw.mean(axis=1), color='#FF9896', linewidth=1.1, label=f'Reconstructed {bundle.pair_labels[1]} mean')
    axes[2].set_ylabel(bundle.units['fnirs_secondary'])
    axes[2].set_title(f'Local {bundle.pair_labels[1]} neighborhood: actual vs reconstructed')
    axes[2].legend(loc='upper right')
    axes[2].grid(alpha=0.25)
    axes[2].set_xlabel('Time (s)')

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_r_vs_neighboring_eeg(path: Path, bundle: ObservationBundle, run_result: Mapping[str, Any]) -> None:
    if bundle.eeg_obs_raw is None or not bundle.eeg_channel_names:
        return
    eeg_obs_raw = np.asarray(bundle.eeg_obs_raw, dtype=np.float64)
    r_signal = zscore_vector(np.asarray(run_result['state_estimates'][:, 4], dtype=np.float64))
    order = np.argsort(np.abs(bundle.lead_field))[::-1]
    spacing = 3.0
    offsets = spacing * np.arange(len(order) + 1, 0, -1, dtype=np.float64)
    fig, axis = plt.subplots(figsize=(13, 8))
    axis.plot(bundle.time_s, r_signal + offsets[0], color='#111111', linewidth=1.6)
    yticks = [offsets[0]]
    ylabels = ['r(t)']
    palette = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(order), 2)))
    for plot_idx, channel_index in enumerate(order.tolist(), start=1):
        channel_trace = zscore_vector(eeg_obs_raw[:, channel_index])
        axis.plot(bundle.time_s, channel_trace + offsets[plot_idx], color=palette[(plot_idx - 1) % len(palette)], linewidth=1.0)
        yticks.append(offsets[plot_idx])
        ylabels.append(f'{bundle.eeg_channel_names[channel_index]} ({float(bundle.lead_field[channel_index]):+.2f})')
    axis.set_yticks(yticks)
    axis.set_yticklabels(ylabels)
    axis.set_title('Inferred local r(t) vs neighboring EEG channels\n(z-scored and vertically offset for shape comparison)')
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
        '- The dynamic core remains Croce\'s five-state model.',
        '- Real-data mode uses deviation coordinates around baseline so signs remain explicit and zero-centering is valid.',
        '',
        '## Observation Setup',
        '',
        f'- Mode: {bundle.mode}',
        f'- Pair mode: {bundle.pair_mode}',
        f'- Pair labels: {bundle.pair_labels[0]} / {bundle.pair_labels[1]}',
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
            f'- Mean EEG correlation: {baseline_metrics["eeg_corr_mean"]:.4f}',
            f'- Mean {bundle.pair_labels[0]} correlation: {baseline_metrics["fnirs_primary_corr_mean"]:.4f}',
            f'- Mean {bundle.pair_labels[1]} correlation: {baseline_metrics["fnirs_secondary_corr_mean"]:.4f}',
            f'- State to fNIRS peak lag (s): {baseline_metrics["state_to_fnirs_peak_lag_s"]:.4f}',
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
            f'- Time-shift null r-correlation delta (baseline - null): {null_metrics["time_shift_r_corr_delta"]:.4f}',
        ]
    )
    if 'spatial_null_log_likelihood_delta' in null_metrics:
        lines.extend(
            [
                f'- Spatial-null log-likelihood delta (baseline - null): {null_metrics["spatial_null_log_likelihood_delta"]:.4f}',
                f'- Spatial-null r-correlation delta (baseline - null): {null_metrics["spatial_null_r_corr_delta"]:.4f}',
            ]
        )
    if bundle.true_states is not None:
        lines.extend(
            [
                '',
                '## Synthetic Recovery',
                '',
                f'- RMSE r(t): {baseline_metrics["rmse_r"]:.6f}',
                f'- RMSE delta_HbO(t): {baseline_metrics["rmse_delta_hbo"]:.6f}',
                f'- RMSE delta_Hb(t): {baseline_metrics["rmse_delta_hb"]:.6f}',
            ]
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    args = parse_args()
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
        seed_list=parse_seed_list(args.seed_list),
        time_shift_null_s=float(args.time_shift_null_s),
        run_spatial_null=bool(args.run_spatial_null),
    )
    output_dir = resolve_output_dir(args.output_dir, args.mode)

    if args.mode == 'synthetic':
        bundle = simulate_synthetic_bundle(args, spatial_config)
    elif args.mode == 'dataset':
        bundle = load_dataset_bundle(args, spatial_config)
        filter_config = FilterConfig(
            integration_dt_s=filter_config.integration_dt_s,
            observation_fs_hz=float(1.0 / (bundle.time_s[1] - bundle.time_s[0])) if bundle.time_s.shape[0] > 1 else float(args.observation_fs),
            num_particles=filter_config.num_particles,
            resample_fraction=filter_config.resample_fraction,
            prior_std=filter_config.prior_std,
            state_noise_std=filter_config.state_noise_std,
            seed_list=filter_config.seed_list,
            time_shift_null_s=filter_config.time_shift_null_s,
            run_spatial_null=filter_config.run_spatial_null,
        )
    else:
        bundle = load_real_bundle(args, spatial_config)
        filter_config = FilterConfig(
            integration_dt_s=filter_config.integration_dt_s,
            observation_fs_hz=float(1.0 / (bundle.time_s[1] - bundle.time_s[0])) if bundle.time_s.shape[0] > 1 else float(args.observation_fs),
            num_particles=filter_config.num_particles,
            resample_fraction=filter_config.resample_fraction,
            prior_std=filter_config.prior_std,
            state_noise_std=filter_config.state_noise_std,
            seed_list=filter_config.seed_list,
            time_shift_null_s=filter_config.time_shift_null_s,
            run_spatial_null=filter_config.run_spatial_null,
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
        'time_shift_r_corr_delta': float(baseline_metrics['state_to_fnirs_peak_corr'] - time_shift_metrics['state_to_fnirs_peak_corr']),
    }

    if filter_config.run_spatial_null:
        spatial_bundle = build_null_bundle(bundle, time_shift_s=0.0, spatial_permutation=True)
        spatial_result = run_particle_filter(spatial_bundle, filter_config, params, seed=filter_config.seed_list[0])
        spatial_metrics = compute_fit_metrics(spatial_bundle, spatial_result, filter_config)
        null_metrics.update(
            {
                'spatial_null_log_likelihood_delta': float(baseline_metrics['log_likelihood'] - spatial_metrics['log_likelihood']),
                'spatial_null_r_corr_delta': float(baseline_metrics['state_to_fnirs_peak_corr'] - spatial_metrics['state_to_fnirs_peak_corr']),
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
    plot_r_vs_neighboring_eeg(output_dir / 'r_vs_neighboring_eeg.png', bundle, baseline_result)

    print(f'Results saved to {output_dir}')
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()