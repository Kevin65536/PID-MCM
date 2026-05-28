"""End-to-end PyTorch prototype benchmark for the local Croce solver.

This script leaves the production runner untouched. It reuses the current audit
module for data loading and baseline execution, then runs an independent batched
PyTorch prototype that preserves the same stochastic process and weighting
equations.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / 'croce_validation' / 'results'
AUDIT_SCRIPT = PROJECT_ROOT / 'croce_validation' / 'scripts' / 'run_local_neighborhood_solver_audit.py'
AUDIT_MODULE_NAME = 'croce_torch_end_to_end_audit'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Benchmark an independent end-to-end PyTorch prototype against the current Croce audit solver.')
    parser.add_argument('--data-root', default='data/EEG+NIRS Single-Trial')
    parser.add_argument('--subject-id', type=int, default=1)
    parser.add_argument('--session-idx', type=int, default=0)
    parser.add_argument('--anchor-fnirs-channel', default='AF7Fp1')
    parser.add_argument('--segment-start-s', type=float, default=60.0)
    parser.add_argument('--segment-duration-s', type=float, default=30.0)
    parser.add_argument('--use-artifact-eeg', action='store_true')

    parser.add_argument('--eeg-neighbors', type=int, default=6)
    parser.add_argument('--fnirs-neighbors', type=int, default=4)
    parser.add_argument('--eeg-radius-mm', type=float, default=60.0)
    parser.add_argument('--fnirs-radius-mm', type=float, default=45.0)
    parser.add_argument('--eeg-sigma-mm', type=float, default=30.0)
    parser.add_argument('--fnirs-sigma-mm', type=float, default=22.0)
    parser.add_argument('--eeg-sign-mode', choices=('covariance', 'geometric_x'), default='covariance')

    parser.add_argument('--num-particles', type=int, default=224)
    parser.add_argument('--resample-fraction', type=float, default=0.5)
    parser.add_argument('--prior-std', default='0.05,0.05,0.05,0.05,0.0')
    parser.add_argument('--state-noise-std', default='0.02,0.015,0.015,0.015,0.0')
    parser.add_argument('--sigma-prop', type=float, default=6.0)
    parser.add_argument('--sigma-nirs', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--time-shift-null-s', type=float, default=8.0)
    parser.add_argument('--torch-device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--output-dir', default='')
    return parser.parse_args()


def load_audit_module() -> Any:
    spec = importlib.util.spec_from_file_location(AUDIT_MODULE_NAME, AUDIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load audit module from {AUDIT_SCRIPT}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[AUDIT_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def resolve_output_dir(spec: str) -> Path:
    if spec:
        output_dir = Path(spec)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = RESULTS_ROOT / f'benchmark_torch_end_to_end_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def select_torch_device(requested: str) -> str:
    if requested == 'cpu':
        return 'cpu'
    if requested == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but torch.cuda.is_available() is false')
        return 'cuda'
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def build_spatial_config(audit: Any, args: argparse.Namespace) -> Any:
    return audit.SpatialConfig(
        eeg_neighbors=int(args.eeg_neighbors),
        fnirs_neighbors=int(args.fnirs_neighbors),
        eeg_radius_mm=float(args.eeg_radius_mm),
        fnirs_radius_mm=float(args.fnirs_radius_mm),
        eeg_sigma_mm=float(args.eeg_sigma_mm),
        fnirs_sigma_mm=float(args.fnirs_sigma_mm),
        eeg_sign_mode=str(args.eeg_sign_mode),
    )


def build_dataset_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        data_root=str(args.data_root),
        subject_id=int(args.subject_id),
        session_idx=int(args.session_idx),
        segment_start_s=float(args.segment_start_s),
        segment_duration_s=float(args.segment_duration_s),
        anchor_fnirs_channel=str(args.anchor_fnirs_channel),
        use_artifact_eeg=bool(args.use_artifact_eeg),
        eeg_unit='uV',
        fnirs_primary_unit='a.u.',
        fnirs_secondary_unit='a.u.',
    )


def build_filter_config(audit: Any, bundle: Any, args: argparse.Namespace) -> Any:
    filter_config = audit.FilterConfig(
        integration_dt_s=float(1.0 / bundle.eeg_fs_hz),
        observation_fs_hz=float(bundle.fnirs_fs_hz),
        num_particles=int(args.num_particles),
        resample_fraction=float(args.resample_fraction),
        prior_std=audit.parse_vector(str(args.prior_std), name='prior-std'),
        state_noise_std=audit.parse_vector(str(args.state_noise_std), name='state-noise-std'),
        sigma_prop=float(args.sigma_prop),
        sigma_nirs=float(args.sigma_nirs),
        seed_list=(int(args.seed),),
        time_shift_null_s=float(args.time_shift_null_s),
        run_spatial_null=False,
        solver_backend='python_exact',
        torch_device='cpu',
    )
    filter_config.prior_std[4] = 0.0
    filter_config.state_noise_std[4] = 0.0
    return filter_config


def build_context(audit: Any, args: argparse.Namespace) -> Tuple[Any, Any, Any, Any]:
    spatial_config = build_spatial_config(audit, args)
    bundle = audit.load_dataset_bundle(build_dataset_args(args), spatial_config)
    filter_config = build_filter_config(audit, bundle, args)
    params = audit.ModelParams()
    time_shift_bundle = audit.build_null_bundle(bundle, time_shift_s=filter_config.time_shift_null_s, spatial_permutation=False)
    return bundle, time_shift_bundle, filter_config, params


def torch_safe_extraction_fraction(flow: torch.Tensor, e0: float) -> torch.Tensor:
    base = torch.tensor(1.0 - e0, dtype=flow.dtype, device=flow.device)
    return 1.0 - torch.pow(base, 1.0 / torch.clamp(flow, min=1e-4))


def torch_local_linearized_step_batch(particles: torch.Tensor, dt: float, params: Any) -> torch.Tensor:
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
    if pair_mode == 'chromophore':
        pred_primary = particles[:, 2:3] * jac_primary.reshape(1, -1)
        pred_secondary = particles[:, 3:4] * jac_secondary.reshape(1, -1)
    else:
        pred_primary = (1.00 * particles[:, 2:3] + 0.25 * particles[:, 3:4]) * jac_primary.reshape(1, -1)
        pred_secondary = (0.35 * particles[:, 2:3] + 1.00 * particles[:, 3:4]) * jac_secondary.reshape(1, -1)
    return pred_eeg, pred_primary, pred_secondary


def run_particle_filter_torch_exact(
    audit: Any,
    bundle: Any,
    filter_config: Any,
    params: Any,
    seed: int,
    device: str,
) -> Dict[str, Any]:
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
            particles = particles[audit.torch_systematic_resample(weights, rng)]
            weights.fill_(1.0 / float(num_particles))

    if device == 'cuda':
        torch.cuda.synchronize()

    pred_eeg, _, _ = audit.predict_observations(
        np.column_stack(
            [
                np.zeros_like(r_estimates_eeg),
                np.zeros_like(r_estimates_eeg),
                np.zeros_like(r_estimates_eeg),
                np.zeros_like(r_estimates_eeg),
                r_estimates_eeg,
            ]
        ),
        bundle.lead_field,
        bundle.jac_primary,
        bundle.jac_secondary,
        bundle.pair_mode,
    )
    _, pred_primary, pred_secondary = audit.predict_observations(
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
        'r_estimates_eeg': r_estimates_eeg,
        'r_std_eeg': r_std_eeg,
    }


def compare_arrays(candidate: np.ndarray, baseline: np.ndarray) -> Dict[str, float]:
    diff = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    denominator = max(float(np.linalg.norm(baseline)), 1e-12)
    return {
        'max_abs': float(np.max(np.abs(diff))),
        'rmse': float(np.sqrt(np.mean(np.square(diff)))),
        'relative_l2': float(np.linalg.norm(diff) / denominator),
    }


def compare_run_results(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        'log_likelihood_abs_delta': float(abs(float(candidate['log_likelihood']) - float(baseline['log_likelihood']))),
        'state_estimates': compare_arrays(candidate['state_estimates'], baseline['state_estimates']),
        'state_std': compare_arrays(candidate['state_std'], baseline['state_std']),
        'ess_trace': compare_arrays(candidate['ess_trace'], baseline['ess_trace']),
        'pred_eeg': compare_arrays(candidate['pred_eeg'], baseline['pred_eeg']),
        'pred_primary': compare_arrays(candidate['pred_primary'], baseline['pred_primary']),
        'pred_secondary': compare_arrays(candidate['pred_secondary'], baseline['pred_secondary']),
        'r_estimates_eeg': compare_arrays(candidate['r_estimates_eeg'], baseline['r_estimates_eeg']),
        'r_std_eeg': compare_arrays(candidate['r_std_eeg'], baseline['r_std_eeg']),
    }


def compare_metrics(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> Dict[str, float]:
    keys = sorted(set(candidate.keys()) & set(baseline.keys()))
    return {key: float(abs(float(candidate[key]) - float(baseline[key]))) for key in keys if np.isfinite(candidate[key]) and np.isfinite(baseline[key])}


def time_call(fn: Any, *args: Any, **kwargs: Any) -> Tuple[float, Any]:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return time.perf_counter() - start, result


def evaluate_scenario(
    audit: Any,
    scenario_name: str,
    bundle: Any,
    filter_config: Any,
    params: Any,
    seed: int,
    device: str,
) -> Dict[str, Any]:
    baseline_wall_s, baseline_result = time_call(audit.run_particle_filter, bundle, filter_config, params, seed)
    torch_wall_s, torch_result = time_call(run_particle_filter_torch_exact, audit, bundle, filter_config, params, seed, device)
    baseline_metrics = audit.compute_fit_metrics(bundle, baseline_result, filter_config)
    torch_metrics = audit.compute_fit_metrics(bundle, torch_result, filter_config)
    return {
        'scenario': scenario_name,
        'seed': int(seed),
        'baseline_wall_s': float(baseline_wall_s),
        'torch_wall_s': float(torch_wall_s),
        'speedup': float(baseline_wall_s / max(torch_wall_s, 1e-12)),
        'run_result_deltas': compare_run_results(torch_result, baseline_result),
        'metric_abs_deltas': compare_metrics(torch_metrics, baseline_metrics),
        'baseline_metrics': baseline_metrics,
        'torch_metrics': torch_metrics,
    }


def build_summary_text(args: argparse.Namespace, device: str, results: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        '# End-to-End PyTorch Prototype Benchmark',
        '',
        '## Scope',
        '',
        '- This benchmark forces the explicit python exact backend for baseline comparison against the optimized torch exact path.',
        f'- Dataset segment: subject {int(args.subject_id)}, session {int(args.session_idx)}, start {float(args.segment_start_s):.1f}s, duration {float(args.segment_duration_s):.1f}s.',
        f'- Anchor: {args.anchor_fnirs_channel}',
        f'- Torch device: {device}',
        f'- Seed: {int(args.seed)}',
        '',
        '## Scenario Results',
        '',
    ]
    for record in results:
        lines.extend(
            [
                f'### {record["scenario"]}',
                '',
                f'- Baseline wall time: {record["baseline_wall_s"]:.4f} s',
                f'- PyTorch wall time: {record["torch_wall_s"]:.4f} s',
                f'- Speedup: {record["speedup"]:.4f}x',
                f'- Log-likelihood abs delta: {record["run_result_deltas"]["log_likelihood_abs_delta"]:.8e}',
                f'- Max state_estimates abs delta: {record["run_result_deltas"]["state_estimates"]["max_abs"]:.8e}',
                f'- Max pred_primary abs delta: {record["run_result_deltas"]["pred_primary"]["max_abs"]:.8e}',
                f'- Max pred_secondary abs delta: {record["run_result_deltas"]["pred_secondary"]["max_abs"]:.8e}',
                f'- Max r_estimates abs delta: {record["run_result_deltas"]["r_estimates_eeg"]["max_abs"]:.8e}',
                '',
            ]
        )
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    audit = load_audit_module()
    output_dir = resolve_output_dir(str(args.output_dir))
    device = select_torch_device(str(args.torch_device))
    bundle, time_shift_bundle, filter_config, params = build_context(audit, args)

    results = [
        evaluate_scenario(audit, 'baseline', bundle, filter_config, params, int(args.seed), device),
        evaluate_scenario(audit, 'time_shift_null', time_shift_bundle, filter_config, params, int(args.seed), device),
    ]

    manifest = {
        'audit_script': str(AUDIT_SCRIPT.relative_to(PROJECT_ROOT)),
        'dataset': {
            'data_root': str(args.data_root),
            'subject_id': int(args.subject_id),
            'session_idx': int(args.session_idx),
            'anchor_fnirs_channel': str(args.anchor_fnirs_channel),
            'segment_start_s': float(args.segment_start_s),
            'segment_duration_s': float(args.segment_duration_s),
            'use_artifact_eeg': bool(args.use_artifact_eeg),
        },
        'filter': {
            'num_particles': int(filter_config.num_particles),
            'sigma_prop': float(filter_config.sigma_prop),
            'sigma_nirs': float(filter_config.sigma_nirs),
            'resample_fraction': float(filter_config.resample_fraction),
            'integration_dt_s': float(filter_config.integration_dt_s),
            'time_shift_null_s': float(filter_config.time_shift_null_s),
            'seed': int(args.seed),
        },
        'torch': {
            'device': device,
            'version': torch.__version__,
            'cuda_available': bool(torch.cuda.is_available()),
            'device_name': str(torch.cuda.get_device_name(0)) if device == 'cuda' else 'cpu',
        },
    }

    (output_dir / 'benchmark_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (output_dir / 'benchmark_results.json').write_text(json.dumps({'scenarios': results}, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (output_dir / 'benchmark_summary.md').write_text(build_summary_text(args, device, results), encoding='utf-8')

    print(json.dumps({'output_dir': str(output_dir), 'scenarios': results}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()