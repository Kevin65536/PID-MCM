"""Visualize lag-aware EEG-fNIRS coupling tensors with simulation and real-run support."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizers import create_tokenizer


def _softmax_numpy(values: np.ndarray, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(values)
    denom = np.sum(exp_values, axis=axis, keepdims=True)
    return exp_values / np.clip(denom, 1e-12, None)


def _compute_token_order(codebook_weight: Optional[np.ndarray], n_tokens: int) -> np.ndarray:
    if codebook_weight is None:
        return np.arange(n_tokens, dtype=np.int64)
    weight = np.asarray(codebook_weight, dtype=np.float64)
    if weight.ndim != 2 or weight.shape[0] != n_tokens:
        return np.arange(n_tokens, dtype=np.int64)
    norms = np.linalg.norm(weight, axis=1, keepdims=True)
    weight = weight / np.clip(norms, 1e-12, None)
    centered = weight - weight.mean(axis=0, keepdims=True)
    if np.allclose(centered, 0.0):
        return np.arange(n_tokens, dtype=np.int64)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ vh[0]
    return np.argsort(scores, kind='stable')


def _prepare_coupling_views(
    coupling_logits: np.ndarray,
    eeg_codebook: Optional[np.ndarray] = None,
    fnirs_codebook: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    logits = np.asarray(coupling_logits, dtype=np.float64)
    if logits.ndim != 3:
        raise ValueError(f'Expected coupling_logits with shape [n_lags, n_eeg, n_fnirs], got {logits.shape}')

    n_lags, n_eeg, n_fnirs = logits.shape
    eeg_order = _compute_token_order(eeg_codebook, n_eeg)
    fnirs_order = _compute_token_order(fnirs_codebook, n_fnirs)
    ordered_logits = logits[:, eeg_order][:, :, fnirs_order]

    slice_probs = _softmax_numpy(ordered_logits, axis=-1)
    joint_logits = np.transpose(ordered_logits, (1, 0, 2))
    joint_probs = _softmax_numpy(joint_logits.reshape(n_eeg, n_lags * n_fnirs), axis=-1).reshape(n_eeg, n_lags, n_fnirs)
    fnirs_marginal = joint_probs.sum(axis=1)
    lag_marginal = joint_probs.sum(axis=2)

    fnirs_positions = np.arange(n_fnirs, dtype=np.float64)
    lag_positions = np.arange(n_lags, dtype=np.float64)
    expected_fnirs = fnirs_marginal @ fnirs_positions
    expected_lag = lag_marginal @ lag_positions

    lag_mass = np.clip(lag_marginal[..., None], 1e-12, None)
    conditional_joint_given_lag = joint_probs / lag_mass
    expected_fnirs_per_lag = (conditional_joint_given_lag * fnirs_positions[None, None, :]).sum(axis=-1)
    row_entropy = -(slice_probs * np.log(slice_probs + 1e-12)).sum(axis=-1)
    lag_entropy = -(lag_marginal * np.log(lag_marginal + 1e-12)).sum(axis=-1)
    joint_entropy = -(joint_probs * np.log(joint_probs + 1e-12)).sum(axis=(1, 2))

    fnirs_roughness = float(np.mean(np.abs(np.diff(expected_fnirs)))) / max(float(n_fnirs - 1), 1.0)
    lag_roughness = float(np.mean(np.abs(np.diff(expected_lag)))) / max(float(n_lags - 1), 1.0)

    return {
        'ordered_logits': ordered_logits,
        'slice_probs': slice_probs,
        'joint_probs': joint_probs,
        'fnirs_marginal': fnirs_marginal,
        'lag_marginal': lag_marginal,
        'expected_fnirs': expected_fnirs,
        'expected_lag': expected_lag,
        'expected_fnirs_per_lag': expected_fnirs_per_lag,
        'row_entropy': row_entropy,
        'lag_entropy': lag_entropy,
        'joint_entropy': joint_entropy,
        'eeg_order': eeg_order,
        'fnirs_order': fnirs_order,
        'summary': {
            'n_lags': int(n_lags),
            'n_eeg_tokens': int(n_eeg),
            'n_fnirs_tokens': int(n_fnirs),
            'row_entropy_ratio_to_logk': float(np.mean(row_entropy) / max(math.log(max(n_fnirs, 2)), 1e-12)),
            'lag_entropy_ratio_to_logl': float(np.mean(lag_entropy) / max(math.log(max(n_lags, 2)), 1e-12)),
            'joint_entropy_ratio': float(np.mean(joint_entropy) / max(math.log(max(n_lags * n_fnirs, 2)), 1e-12)),
            'slice_peak_mean': float(np.mean(slice_probs.max(axis=-1))),
            'lag_focus_mean': float(np.mean(lag_marginal.max(axis=-1))),
            'fnirs_roughness': fnirs_roughness,
            'lag_roughness': lag_roughness,
        },
    }


def prepare_coupling_tensor_views(
    coupling_logits: np.ndarray,
    eeg_codebook: Optional[np.ndarray] = None,
    fnirs_codebook: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Prepare standard all-lag coupling tensor views for analysis integrations."""
    return _prepare_coupling_views(
        coupling_logits=coupling_logits,
        eeg_codebook=eeg_codebook,
        fnirs_codebook=fnirs_codebook,
    )


def _plot_case(views: Dict[str, Any], output_path: Path, title: str, subtitle: str) -> None:
    slice_probs = np.asarray(views['slice_probs'], dtype=np.float64)
    fnirs_marginal = np.asarray(views['fnirs_marginal'], dtype=np.float64)
    lag_marginal = np.asarray(views['lag_marginal'], dtype=np.float64)
    expected_fnirs = np.asarray(views['expected_fnirs'], dtype=np.float64)
    expected_lag = np.asarray(views['expected_lag'], dtype=np.float64)
    expected_fnirs_per_lag = np.asarray(views['expected_fnirs_per_lag'], dtype=np.float64)
    summary = dict(views['summary'])
    n_lags = int(summary['n_lags'])
    n_slice_cols = min(4, n_lags)
    n_slice_rows = int(math.ceil(n_lags / n_slice_cols))

    figure = plt.figure(figsize=(18, 7 + 2.7 * n_slice_rows))
    outer = figure.add_gridspec(2, 1, height_ratios=[1.15, max(1.0, 0.65 * n_slice_rows)], hspace=0.32)
    top = outer[0].subgridspec(1, 3, width_ratios=[1.15, 0.7, 0.95], wspace=0.28)
    axes_top = [figure.add_subplot(top[0, idx]) for idx in range(3)]

    image0 = axes_top[0].imshow(fnirs_marginal, aspect='auto', cmap='viridis')
    axes_top[0].set_title('EEG x fNIRS marginal')
    axes_top[0].set_xlabel('Ordered fNIRS token')
    axes_top[0].set_ylabel('Ordered EEG token')
    figure.colorbar(image0, ax=axes_top[0], fraction=0.046, pad=0.04)

    image1 = axes_top[1].imshow(lag_marginal, aspect='auto', cmap='magma')
    axes_top[1].set_title('EEG x lag marginal')
    axes_top[1].set_xlabel('Lag index')
    axes_top[1].set_ylabel('Ordered EEG token')
    figure.colorbar(image1, ax=axes_top[1], fraction=0.046, pad=0.04)

    image2 = axes_top[2].imshow(expected_fnirs_per_lag, aspect='auto', cmap='cividis', vmin=0.0, vmax=max(fnirs_marginal.shape[1] - 1, 1))
    axes_top[2].set_title('Expected fNIRS index given EEG and lag')
    axes_top[2].set_xlabel('Lag index')
    axes_top[2].set_ylabel('Ordered EEG token')
    figure.colorbar(image2, ax=axes_top[2], fraction=0.046, pad=0.04)

    slice_grid = outer[1].subgridspec(n_slice_rows, n_slice_cols, wspace=0.18, hspace=0.28)
    uniform_prob = 1.0 / max(slice_probs.shape[2], 1)
    slice_delta = slice_probs - uniform_prob
    vmax = max(float(np.abs(slice_delta).max()), 1e-6)
    for lag in range(n_lags):
        axis = figure.add_subplot(slice_grid[lag // n_slice_cols, lag % n_slice_cols])
        axis.imshow(slice_delta[lag], aspect='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax)
        axis.set_title(f'Lag {lag}')
        axis.set_xlabel('fNIRS token')
        axis.set_ylabel('EEG token')

    for empty_idx in range(n_lags, n_slice_rows * n_slice_cols):
        axis = figure.add_subplot(slice_grid[empty_idx // n_slice_cols, empty_idx % n_slice_cols])
        axis.axis('off')

    figure.suptitle(title, fontsize=16, fontweight='bold', y=0.985)
    metrics_text = (
        f"{subtitle}\n"
        f"lag slices show P(fNIRS|EEG, lag) minus uniform baseline\n"
        f"row_entropy/logK={summary['row_entropy_ratio_to_logk']:.3f}\n"
        f"lag_entropy/logL={summary['lag_entropy_ratio_to_logl']:.3f}\n"
        f"joint_entropy/log(LK)={summary['joint_entropy_ratio']:.3f}\n"
        f"slice_peak_mean={summary['slice_peak_mean']:.3f}\n"
        f"lag_focus_mean={summary['lag_focus_mean']:.3f}\n"
        f"fnirs_roughness={summary['fnirs_roughness']:.3f}\n"
        f"lag_roughness={summary['lag_roughness']:.3f}"
    )
    figure.text(0.012, 0.012, metrics_text, fontsize=10, va='bottom', ha='left', bbox={'facecolor': 'white', 'alpha': 0.92, 'edgecolor': '#D0D7DE'})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(figure)


def plot_coupling_tensor_overview(
    *,
    coupling_logits: np.ndarray,
    output_path: Path,
    title: str,
    subtitle: str,
    eeg_codebook: Optional[np.ndarray] = None,
    fnirs_codebook: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Write the standard EEG x fNIRS x lag coupling overview and return summary metrics."""
    views = _prepare_coupling_views(
        coupling_logits=coupling_logits,
        eeg_codebook=eeg_codebook,
        fnirs_codebook=fnirs_codebook,
    )
    _plot_case(views=views, output_path=output_path, title=title, subtitle=subtitle)
    return {
        'figure_path': str(output_path),
        'summary': views['summary'],
        'eeg_order': np.asarray(views['eeg_order'], dtype=np.int64).tolist(),
        'fnirs_order': np.asarray(views['fnirs_order'], dtype=np.int64).tolist(),
    }


def _synthetic_codebook(n_tokens: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n_tokens)
    features = np.stack([
        x,
        np.sin(np.pi * x),
        np.cos(np.pi * x),
        np.sin(2.0 * np.pi * x),
    ], axis=1)
    features += 0.02 * rng.standard_normal(features.shape)
    return features


def _smooth_grid(values: np.ndarray, passes: int = 3) -> np.ndarray:
    smoothed = np.asarray(values, dtype=np.float64)
    for _ in range(max(int(passes), 0)):
        smoothed = (
            4.0 * smoothed
            + np.roll(smoothed, 1, axis=0)
            + np.roll(smoothed, -1, axis=0)
            + np.roll(smoothed, 1, axis=1)
            + np.roll(smoothed, -1, axis=1)
        ) / 8.0
    return smoothed


def _simulate_smooth_case(n_lags: int, n_eeg: int, n_fnirs: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    logits = np.zeros((n_lags, n_eeg, n_fnirs), dtype=np.float64)
    fnirs_axis = np.arange(n_fnirs, dtype=np.float64)
    eeg_axis = np.linspace(0.0, 1.0, n_eeg)
    for eeg_idx, eeg_pos in enumerate(eeg_axis):
        lag_center = 1.4 + 0.55 * (n_lags - 1) * (0.5 + 0.5 * np.sin(1.4 * np.pi * eeg_pos))
        lag_width = 1.0 + 0.18 * np.cos(2.0 * np.pi * eeg_pos)
        for lag_idx in range(n_lags):
            lag_gain = 3.0 * np.exp(-0.5 * ((lag_idx - lag_center) / max(lag_width, 0.6)) ** 2) + 0.35
            fnirs_center = 1.2 + eeg_pos * (n_fnirs - 3.4) + 0.7 * np.sin(0.65 * lag_idx + 2.0 * np.pi * eeg_pos)
            width = 1.2 + 0.18 * np.cos(0.4 * lag_idx + np.pi * eeg_pos)
            primary = np.exp(-0.5 * ((fnirs_axis - fnirs_center) / max(width, 0.8)) ** 2)
            secondary = 0.65 * np.exp(-0.5 * ((fnirs_axis - (fnirs_center + 1.7)) / max(width * 1.35, 1.0)) ** 2)
            logits[lag_idx, eeg_idx] = lag_gain * (primary + secondary) + 0.04 * rng.standard_normal(n_fnirs)
    return {
        'title': 'Synthetic Smooth Coupling',
        'subtitle': 'Desired case: smooth EEG neighborhoods, smooth lag drift, multi-token fNIRS support',
        'coupling_logits': logits,
        'eeg_codebook': _synthetic_codebook(n_eeg, seed + 1),
        'fnirs_codebook': _synthetic_codebook(n_fnirs, seed + 2),
    }


def _simulate_reference_case(n_lags: int, n_eeg: int, n_fnirs: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    logits = np.zeros((n_lags, n_eeg, n_fnirs), dtype=np.float64)
    fnirs_axis = np.arange(n_fnirs, dtype=np.float64)
    eeg_axis = np.linspace(0.0, 1.0, n_eeg)
    center_drift = 0.85 * _smooth_grid(rng.standard_normal((n_lags, n_eeg)), passes=4)
    amplitude_drift = 0.22 * _smooth_grid(rng.standard_normal((n_lags, n_eeg)), passes=3)
    for eeg_idx, eeg_pos in enumerate(eeg_axis):
        lag_center = 1.7 + 0.52 * (n_lags - 1) * (0.5 + 0.5 * np.sin(1.15 * np.pi * eeg_pos + 0.25))
        lag_width = 1.45 + 0.22 * np.cos(1.7 * np.pi * eeg_pos)
        for lag_idx in range(n_lags):
            lag_gain = 2.15 * np.exp(-0.5 * ((lag_idx - lag_center) / max(lag_width, 0.85)) ** 2) + 0.55
            lag_gain += amplitude_drift[lag_idx, eeg_idx]
            fnirs_center = 1.4 + eeg_pos * (n_fnirs - 3.8)
            fnirs_center += 1.0 * np.sin(0.45 * lag_idx + 1.75 * np.pi * eeg_pos)
            fnirs_center += center_drift[lag_idx, eeg_idx]
            width = 1.55 + 0.24 * np.cos(0.33 * lag_idx + 0.8 * np.pi * eeg_pos)
            primary = np.exp(-0.5 * ((fnirs_axis - fnirs_center) / max(width, 1.05)) ** 2)
            secondary = 0.56 * np.exp(-0.5 * ((fnirs_axis - (fnirs_center + 2.1)) / max(width * 1.55, 1.35)) ** 2)
            tertiary = 0.24 * np.exp(-0.5 * ((fnirs_axis - (fnirs_center - 2.4)) / max(width * 1.8, 1.55)) ** 2)
            background = 0.16 * (1.0 + np.cos(0.38 * fnirs_axis - 0.28 * lag_idx + 1.35 * eeg_pos))
            logits[lag_idx, eeg_idx] = lag_gain * (primary + secondary + tertiary) + background
            logits[lag_idx, eeg_idx] += 0.05 * rng.standard_normal(n_fnirs)
    return {
        'title': 'Synthetic Reference Coupling',
        'subtitle': 'Reference case: slightly more diffuse ridge with coherent EEG neighborhoods and lag drift',
        'coupling_logits': logits,
        'eeg_codebook': _synthetic_codebook(n_eeg, seed + 21),
        'fnirs_codebook': _synthetic_codebook(n_fnirs, seed + 22),
    }


def _simulate_fragmented_case(n_lags: int, n_eeg: int, n_fnirs: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    logits = np.zeros((n_lags, n_eeg, n_fnirs), dtype=np.float64)
    fnirs_axis = np.arange(n_fnirs, dtype=np.float64)
    for eeg_idx in range(n_eeg):
        for lag_idx in range(n_lags):
            center_a = rng.uniform(0.0, n_fnirs - 1)
            center_b = rng.uniform(0.0, n_fnirs - 1)
            width_a = rng.uniform(0.45, 1.35)
            width_b = rng.uniform(0.45, 1.75)
            gain_a = rng.uniform(0.6, 3.6)
            gain_b = rng.uniform(0.2, 2.2)
            profile = gain_a * np.exp(-0.5 * ((fnirs_axis - center_a) / width_a) ** 2)
            profile += gain_b * np.exp(-0.5 * ((fnirs_axis - center_b) / width_b) ** 2)
            profile += 0.45 * rng.standard_normal(n_fnirs)
            logits[lag_idx, eeg_idx] = profile
    return {
        'title': 'Synthetic Fragmented Coupling',
        'subtitle': 'Undesired case: irregular lag preference and broken neighborhood continuity',
        'coupling_logits': logits,
        'eeg_codebook': _synthetic_codebook(n_eeg, seed + 11),
        'fnirs_codebook': _synthetic_codebook(n_fnirs, seed + 12),
    }


def _load_real_case(checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config')
    if config is None:
        raise ValueError(f'Checkpoint {checkpoint_path} does not contain an embedded config')
    model = create_tokenizer(config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    coupling_logits = getattr(model, 'coupling_logits', None)
    if coupling_logits is None:
        raise ValueError(f'Model restored from {checkpoint_path} has no coupling_logits parameter')

    eeg_quantizer = getattr(model, 'eeg_source_quantizer', None)
    fnirs_quantizer = getattr(model, 'fnirs_source_quantizer', None)
    eeg_codebook = None if eeg_quantizer is None else eeg_quantizer.get_codebook_weight().detach().float().cpu().numpy()
    fnirs_codebook = None if fnirs_quantizer is None else fnirs_quantizer.get_codebook_weight().detach().float().cpu().numpy()

    run_dir = checkpoint_path.resolve().parents[1]
    return {
        'title': f'Real Run Coupling: {run_dir.name}',
        'subtitle': f'Checkpoint source: {_project_relative_path(checkpoint_path)}',
        'coupling_logits': coupling_logits.detach().float().cpu().numpy(),
        'eeg_codebook': eeg_codebook,
        'fnirs_codebook': fnirs_codebook,
        'run_dir': str(run_dir),
    }


def _write_summary(summary_path: Path, payload: Dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def _project_relative_path(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize 3D EEG-fNIRS coupling tensors with lag-aware diagnostics.')
    parser.add_argument('--output-dir', required=True, help='Directory to write figures and summaries into')
    parser.add_argument('--run-dir', default=None, help='Optional experiment run directory containing checkpoints/best_model.pt')
    parser.add_argument('--checkpoint', default=None, help='Optional explicit checkpoint path; overrides --run-dir when provided')
    parser.add_argument('--seed', type=int, default=7, help='Random seed for synthetic demonstrations')
    parser.add_argument('--n-lags', type=int, default=8, help='Number of lags for synthetic demonstrations')
    parser.add_argument('--n-eeg-tokens', type=int, default=24, help='Number of EEG tokens for synthetic demonstrations')
    parser.add_argument('--n-fnirs-tokens', type=int, default=24, help='Number of fNIRS tokens for synthetic demonstrations')
    parser.add_argument('--device', default='cuda', help='Torch device to use when loading a real checkpoint')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    cases = {
        'synthetic_good': _simulate_smooth_case(args.n_lags, args.n_eeg_tokens, args.n_fnirs_tokens, args.seed),
        'synthetic_reference': _simulate_reference_case(args.n_lags, args.n_eeg_tokens, args.n_fnirs_tokens, args.seed + 41),
        'synthetic_bad': _simulate_fragmented_case(args.n_lags, args.n_eeg_tokens, args.n_fnirs_tokens, args.seed + 101),
    }

    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    elif args.run_dir:
        checkpoint_path = Path(args.run_dir) / 'checkpoints' / 'best_model.pt'

    if checkpoint_path is not None:
        cases['real_run'] = _load_real_case(checkpoint_path=checkpoint_path, device=device)

    summary_payload: Dict[str, Any] = {'cases': {}}
    for case_name, case_payload in cases.items():
        views = _prepare_coupling_views(
            coupling_logits=case_payload['coupling_logits'],
            eeg_codebook=case_payload.get('eeg_codebook'),
            fnirs_codebook=case_payload.get('fnirs_codebook'),
        )
        figure_path = output_dir / f'{case_name}_coupling_tensor_overview.png'
        _plot_case(views=views, output_path=figure_path, title=case_payload['title'], subtitle=case_payload['subtitle'])
        summary_payload['cases'][case_name] = {
            'title': case_payload['title'],
            'subtitle': case_payload['subtitle'],
            'figure_path': _project_relative_path(figure_path),
            'summary': views['summary'],
        }
        if 'run_dir' in case_payload:
            summary_payload['cases'][case_name]['run_dir'] = case_payload['run_dir']

    _write_summary(output_dir / 'summary.json', summary_payload)
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
