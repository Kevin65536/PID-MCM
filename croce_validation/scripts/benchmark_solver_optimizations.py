"""Benchmark optimization candidates for the local Croce solver.

This script does not modify the production audit implementation. It imports the
existing runner as a read-only baseline and measures several acceleration
candidates against the same local-neighborhood configuration.

Benchmarks are split into two groups:

1. Exact audit-level parallelization across independent runs (baseline seeds
   and timing null).
2. Exact or near-exact state-propagation kernel candidates that target the
   inner particle loop.

Outputs are written to croce_validation/results/benchmark_<timestamp>/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import expm

try:
    import torch
except Exception:  # pragma: no cover - optional dependency at runtime
    torch = None

try:
    import numba as nb
except Exception:  # pragma: no cover - optional dependency at runtime
    nb = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / 'croce_validation' / 'results'
AUDIT_SCRIPT = PROJECT_ROOT / 'croce_validation' / 'scripts' / 'run_local_neighborhood_solver_audit.py'
AUDIT_MODULE_NAME = 'croce_solver_audit_benchmark_target'

_PARALLEL_CONTEXT: Dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Benchmark optimization candidates for the Croce local solver without changing production code.')
    parser.add_argument('--data-root', default='data/EEG+NIRS Single-Trial')
    parser.add_argument('--subject-id', type=int, default=1)
    parser.add_argument('--session-idx', type=int, default=0)
    parser.add_argument('--anchor-fnirs-channel', default='AF7Fp1')
    parser.add_argument('--segment-start-s', type=float, default=60.0)
    parser.add_argument('--segment-duration-s', type=float, default=10.0)
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
    parser.add_argument('--seed-list', default='11,23')
    parser.add_argument('--time-shift-null-s', type=float, default=8.0)

    parser.add_argument('--parallel-workers', type=int, default=3)
    parser.add_argument('--kernel-seed', type=int, default=20260527)
    parser.add_argument('--kernel-eeg-steps', type=int, default=0, help='0 means use the whole segment')
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
        output_dir = RESULTS_ROOT / f'benchmark_solver_optimizations_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def parse_versions() -> Dict[str, str]:
    versions = {
        'python': sys.version.replace('\n', ' '),
        'numpy': np.__version__,
        'scipy': __import__('scipy').__version__,
        'torch': 'unavailable',
        'numba': 'unavailable',
    }
    if torch is not None:
        versions['torch'] = getattr(torch, '__version__', 'unknown')
    if nb is not None:
        versions['numba'] = getattr(nb, '__version__', 'unknown')
    return versions


def select_torch_device(requested: str) -> Optional[str]:
    if torch is None:
        return None
    if requested == 'cpu':
        return 'cpu'
    if requested == 'cuda':
        return 'cuda' if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


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
        data_root=args.data_root,
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
        seed_list=audit.parse_seed_list(str(args.seed_list)),
        time_shift_null_s=float(args.time_shift_null_s),
        run_spatial_null=False,
        solver_backend='python_exact',
        torch_device='cpu',
    )
    filter_config.prior_std[4] = 0.0
    filter_config.state_noise_std[4] = 0.0
    return filter_config


def build_benchmark_context(audit: Any, args: argparse.Namespace) -> Tuple[Any, Any, Any, Any]:
    spatial_config = build_spatial_config(audit, args)
    bundle = audit.load_dataset_bundle(build_dataset_args(args), spatial_config)
    filter_config = build_filter_config(audit, bundle, args)
    params = audit.ModelParams()
    null_bundle = audit.build_null_bundle(bundle, time_shift_s=filter_config.time_shift_null_s, spatial_permutation=False)
    return bundle, filter_config, params, null_bundle


def as_float_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def make_kernel_fixture(bundle: Any, filter_config: Any, seed: int, num_eeg_steps: int) -> Dict[str, np.ndarray]:
    steps = int(bundle.eeg_time_s.shape[0] if num_eeg_steps <= 0 else min(num_eeg_steps, bundle.eeg_time_s.shape[0]))
    num_particles = int(filter_config.num_particles)
    rng = np.random.default_rng(seed)
    particles0 = rng.normal(
        loc=np.zeros(5, dtype=np.float64).reshape(1, 5),
        scale=as_float_array(filter_config.prior_std).reshape(1, 5),
        size=(num_particles, 5),
    )
    particles0[:, 1:4] = np.clip(particles0[:, 1:4], -0.95, None)
    particles0[:, 4] = 0.0
    proposal_noise = rng.normal(size=(steps, num_particles)).astype(np.float64)
    process_noise = rng.normal(size=(steps, num_particles, 4)).astype(np.float64)
    return {
        'particles0': particles0,
        'proposal_centers': as_float_array(bundle.r_eeg_projection[:steps]),
        'proposal_noise': proposal_noise,
        'process_noise': process_noise,
    }


def kernel_python_baseline(audit: Any, filter_config: Any, params: Any, fixture: Mapping[str, np.ndarray]) -> np.ndarray:
    particles = np.asarray(fixture['particles0'], dtype=np.float64).copy()
    proposal_centers = np.asarray(fixture['proposal_centers'], dtype=np.float64)
    proposal_noise = np.asarray(fixture['proposal_noise'], dtype=np.float64)
    process_noise = np.asarray(fixture['process_noise'], dtype=np.float64)
    hemo_scale = np.asarray(filter_config.state_noise_std[:4], dtype=np.float64) * np.sqrt(float(filter_config.integration_dt_s))
    for step in range(proposal_centers.shape[0]):
        particles[:, 4] = proposal_centers[step] + float(filter_config.sigma_prop) * proposal_noise[step]
        for idx in range(particles.shape[0]):
            particles[idx] = audit.local_linearized_step(particles[idx], float(filter_config.integration_dt_s), params)
        particles[:, 0:4] += process_noise[step] * hemo_scale.reshape(1, 4)
        particles[:, 1:4] = np.clip(particles[:, 1:4], -0.95, None)
    return particles


if nb is not None:
    @nb.njit(cache=True)
    def _safe_extraction_fraction_numba(flow: float, e0: float) -> float:
        clipped = flow if flow >= 1e-4 else 1e-4
        return 1.0 - np.power(1.0 - e0, 1.0 / clipped)


    @nb.njit(cache=True)
    def _drift_and_jacobian_numba(x: np.ndarray, epsilon: float, kas: float, kaf: float, tau0: float, alpha: float, e0: float) -> Tuple[np.ndarray, np.ndarray]:
        s = x[0]
        delta_f = x[1]
        delta_hbo = x[2]
        delta_hb = x[3]
        r = x[4]

        f = 1.0 + delta_f
        if f < 1e-4:
            f = 1e-4
        hbo = 1.0 + delta_hbo
        if hbo < 1e-4:
            hbo = 1e-4
        hb = 1.0 + delta_hb
        if hb < 1e-4:
            hb = 1e-4

        extraction = _safe_extraction_fraction_numba(f, e0) / max(e0, 1e-8)
        drift = np.empty(5, dtype=np.float64)
        drift[0] = epsilon * r - kas * s - kaf * (f - 1.0)
        drift[1] = s
        drift[2] = (f - np.power(hbo, 1.0 / alpha)) / tau0
        drift[3] = (f * extraction - hb * np.power(hbo, (1.0 / alpha) - 1.0)) / tau0
        drift[4] = 0.0

        one_minus_e0 = max(1.0 - e0, 1e-8)
        power_term = np.power(one_minus_e0, 1.0 / f)
        d_extraction_df = power_term * np.log(one_minus_e0) / (f * f)
        d_flow_extraction_df = (_safe_extraction_fraction_numba(f, e0) + f * d_extraction_df) / max(e0, 1e-8)

        jac = np.zeros((5, 5), dtype=np.float64)
        jac[0, 0] = -kas
        jac[0, 1] = -kaf
        jac[0, 4] = epsilon
        jac[1, 0] = 1.0
        jac[2, 1] = 1.0 / tau0
        jac[2, 2] = -(1.0 / alpha) * np.power(hbo, (1.0 / alpha) - 1.0) / tau0
        jac[3, 1] = d_flow_extraction_df / tau0
        jac[3, 2] = -hb * ((1.0 / alpha) - 1.0) * np.power(hbo, (1.0 / alpha) - 2.0) / tau0
        jac[3, 3] = -np.power(hbo, (1.0 / alpha) - 1.0) / tau0
        return drift, jac


def kernel_numba_scalar_exact(filter_config: Any, params: Any, fixture: Mapping[str, np.ndarray]) -> np.ndarray:
    if nb is None:
        raise RuntimeError('Numba is not available in the current environment')
    particles = np.asarray(fixture['particles0'], dtype=np.float64).copy()
    proposal_centers = np.asarray(fixture['proposal_centers'], dtype=np.float64)
    proposal_noise = np.asarray(fixture['proposal_noise'], dtype=np.float64)
    process_noise = np.asarray(fixture['process_noise'], dtype=np.float64)
    hemo_scale = np.asarray(filter_config.state_noise_std[:4], dtype=np.float64) * np.sqrt(float(filter_config.integration_dt_s))
    dt = float(filter_config.integration_dt_s)
    for step in range(proposal_centers.shape[0]):
        particles[:, 4] = proposal_centers[step] + float(filter_config.sigma_prop) * proposal_noise[step]
        for idx in range(particles.shape[0]):
            drift, jac = _drift_and_jacobian_numba(
                particles[idx],
                float(params.epsilon),
                float(params.kas),
                float(params.kaf),
                float(params.tau0),
                float(params.alpha),
                float(params.e0),
            )
            augmented = np.zeros((6, 6), dtype=np.float64)
            augmented[:5, :5] = jac
            augmented[:5, 5] = drift
            delta = expm(augmented * dt)[:5, 5]
            particles[idx] = particles[idx] + delta
            particles[idx, 1:4] = np.clip(particles[idx, 1:4], -0.95, None)
        particles[:, 0:4] += process_noise[step] * hemo_scale.reshape(1, 4)
        particles[:, 1:4] = np.clip(particles[:, 1:4], -0.95, None)
    return particles


def _torch_safe_extraction_fraction(flow: torch.Tensor, e0: float) -> torch.Tensor:
    clipped = torch.clamp(flow, min=1e-4)
    return 1.0 - torch.pow(torch.tensor(1.0 - e0, dtype=flow.dtype, device=flow.device), 1.0 / clipped)


def _torch_local_linearized_step_batch(particles: torch.Tensor, dt: float, params: Any) -> torch.Tensor:
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

    extraction = _torch_safe_extraction_fraction(f, e0) / max(e0, 1e-8)
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
    d_flow_extraction_df = (_torch_safe_extraction_fraction(f, e0) + f * d_extraction_df) / max(e0, 1e-8)

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


def kernel_torch_exact(filter_config: Any, params: Any, fixture: Mapping[str, np.ndarray], device: str) -> np.ndarray:
    if torch is None:
        raise RuntimeError('PyTorch is not available in the current environment')
    particles = torch.from_numpy(np.asarray(fixture['particles0'], dtype=np.float64)).to(device=device, dtype=torch.float64)
    proposal_centers = torch.from_numpy(np.asarray(fixture['proposal_centers'], dtype=np.float64)).to(device=device, dtype=torch.float64)
    proposal_noise = torch.from_numpy(np.asarray(fixture['proposal_noise'], dtype=np.float64)).to(device=device, dtype=torch.float64)
    process_noise = torch.from_numpy(np.asarray(fixture['process_noise'], dtype=np.float64)).to(device=device, dtype=torch.float64)
    hemo_scale = torch.from_numpy(np.asarray(filter_config.state_noise_std[:4], dtype=np.float64) * np.sqrt(float(filter_config.integration_dt_s))).to(device=device, dtype=torch.float64)
    dt = float(filter_config.integration_dt_s)
    if device == 'cuda':
        torch.cuda.synchronize()
    for step in range(int(proposal_centers.shape[0])):
        particles[:, 4] = proposal_centers[step] + float(filter_config.sigma_prop) * proposal_noise[step]
        particles = _torch_local_linearized_step_batch(particles, dt, params)
        particles[:, 0:4] += process_noise[step] * hemo_scale.reshape(1, 4)
        particles[:, 1:4] = torch.clamp(particles[:, 1:4], min=-0.95)
    if device == 'cuda':
        torch.cuda.synchronize()
    return particles.detach().cpu().numpy()


def summarize_difference(candidate: np.ndarray, baseline: np.ndarray) -> Dict[str, float]:
    diff = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    denominator = max(float(np.linalg.norm(baseline)), 1e-12)
    return {
        'max_abs': float(np.max(np.abs(diff))),
        'rmse': float(np.sqrt(np.mean(np.square(diff)))),
        'relative_l2': float(np.linalg.norm(diff) / denominator),
    }


def time_call(fn: Any, *args: Any, **kwargs: Any) -> Tuple[float, Any]:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed, result


def _parallel_worker(task: Tuple[str, int]) -> Dict[str, float]:
    task_name, seed = task
    audit = _PARALLEL_CONTEXT['audit']
    params = _PARALLEL_CONTEXT['params']
    filter_config = _PARALLEL_CONTEXT['filter_config']
    if task_name == 'baseline':
        bundle = _PARALLEL_CONTEXT['bundle']
    elif task_name == 'time_shift_null':
        bundle = _PARALLEL_CONTEXT['null_bundle']
    else:
        raise ValueError(f'Unsupported task {task_name!r}')
    result = audit.run_particle_filter(bundle, filter_config, params, seed=seed)
    return {
        'task': task_name,
        'seed': float(seed),
        'log_likelihood': float(result['log_likelihood']),
        'ess_mean': float(np.mean(result['ess_trace'])),
    }


def benchmark_exact_parallelization(audit: Any, bundle: Any, null_bundle: Any, filter_config: Any, params: Any, workers: int) -> Dict[str, Any]:
    tasks: List[Tuple[str, int]] = [('baseline', int(seed)) for seed in filter_config.seed_list]
    tasks.append(('time_shift_null', int(filter_config.seed_list[0])))

    global _PARALLEL_CONTEXT
    _PARALLEL_CONTEXT = {
        'audit': audit,
        'bundle': bundle,
        'null_bundle': null_bundle,
        'filter_config': filter_config,
        'params': params,
    }

    sequential_start = time.perf_counter()
    sequential_results = [_parallel_worker(task) for task in tasks]
    sequential_wall_s = time.perf_counter() - sequential_start

    parallel_results = sequential_results
    parallel_wall_s = sequential_wall_s
    if len(tasks) > 1 and workers > 1:
        ctx = mp.get_context('fork')
        parallel_start = time.perf_counter()
        with ctx.Pool(processes=min(workers, len(tasks))) as pool:
            parallel_results = pool.map(_parallel_worker, tasks)
        parallel_wall_s = time.perf_counter() - parallel_start

    sequential_keyed = {(item['task'], int(item['seed'])): item for item in sequential_results}
    parallel_keyed = {(item['task'], int(item['seed'])): item for item in parallel_results}
    max_log_like_delta = max(
        abs(sequential_keyed[key]['log_likelihood'] - parallel_keyed[key]['log_likelihood'])
        for key in sequential_keyed
    )
    return {
        'task_count': len(tasks),
        'workers': int(min(workers, len(tasks))),
        'sequential_wall_s': float(sequential_wall_s),
        'parallel_wall_s': float(parallel_wall_s),
        'speedup': float(sequential_wall_s / max(parallel_wall_s, 1e-12)),
        'max_log_likelihood_delta': float(max_log_like_delta),
        'sequential_results': sequential_results,
        'parallel_results': parallel_results,
    }


def benchmark_kernel_candidates(audit: Any, filter_config: Any, params: Any, fixture: Mapping[str, np.ndarray], torch_device: Optional[str]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    baseline_wall_s, baseline_particles = time_call(kernel_python_baseline, audit, filter_config, params, fixture)
    results['python_baseline'] = {
        'wall_s': float(baseline_wall_s),
        'speedup_vs_python': 1.0,
    }

    if nb is not None:
        _ = _drift_and_jacobian_numba(
            np.zeros(5, dtype=np.float64),
            float(params.epsilon),
            float(params.kas),
            float(params.kaf),
            float(params.tau0),
            float(params.alpha),
            float(params.e0),
        )
        numba_wall_s, numba_particles = time_call(kernel_numba_scalar_exact, filter_config, params, fixture)
        results['numba_scalar_exact'] = {
            'wall_s': float(numba_wall_s),
            'speedup_vs_python': float(baseline_wall_s / max(numba_wall_s, 1e-12)),
            'difference_vs_python': summarize_difference(numba_particles, baseline_particles),
        }
    else:
        results['numba_scalar_exact'] = {'available': False, 'reason': 'numba import failed'}

    if torch_device is not None:
        torch_wall_s, torch_particles = time_call(kernel_torch_exact, filter_config, params, fixture, 'cpu')
        results['torch_cpu_exact'] = {
            'wall_s': float(torch_wall_s),
            'speedup_vs_python': float(baseline_wall_s / max(torch_wall_s, 1e-12)),
            'difference_vs_python': summarize_difference(torch_particles, baseline_particles),
        }
        if torch_device == 'cuda':
            torch_cuda_wall_s, torch_cuda_particles = time_call(kernel_torch_exact, filter_config, params, fixture, 'cuda')
            results['torch_cuda_exact'] = {
                'wall_s': float(torch_cuda_wall_s),
                'speedup_vs_python': float(baseline_wall_s / max(torch_cuda_wall_s, 1e-12)),
                'difference_vs_python': summarize_difference(torch_cuda_particles, baseline_particles),
                'device_name': str(torch.cuda.get_device_name(0)),
            }
    else:
        results['torch_cpu_exact'] = {'available': False, 'reason': 'torch import failed'}

    return results


def build_summary_text(args: argparse.Namespace, bundle: Any, filter_config: Any, parallel_results: Mapping[str, Any], kernel_results: Mapping[str, Any], versions: Mapping[str, str], torch_device: Optional[str]) -> str:
    lines = [
        '# Solver Optimization Benchmark',
        '',
        '## Benchmark Scope',
        '',
        '- This benchmark forces the explicit python exact backend as the baseline and compares exact optimized kernels against it.',
        f'- Dataset segment: subject {int(args.subject_id)}, session {int(args.session_idx)}, start {float(args.segment_start_s):.1f}s, duration {float(args.segment_duration_s):.1f}s.',
        f'- Anchor: {args.anchor_fnirs_channel}',
        f'- EEG / fNIRS rates: {float(bundle.eeg_fs_hz):.1f} Hz / {float(bundle.fnirs_fs_hz):.1f} Hz',
        f'- Particles: {int(filter_config.num_particles)}',
        f'- Seeds benchmarked: {list(filter_config.seed_list)}',
        '',
        '## Environment',
        '',
        f'- Python: {versions["python"]}',
        f'- NumPy: {versions["numpy"]}',
        f'- SciPy: {versions["scipy"]}',
        f'- PyTorch: {versions["torch"]}',
        f'- Numba: {versions["numba"]}',
        f'- Selected torch device: {torch_device or "unavailable"}',
        '',
        '## Exact Parallelization',
        '',
        f'- Sequential wall time: {parallel_results["sequential_wall_s"]:.4f} s',
        f'- Parallel wall time: {parallel_results["parallel_wall_s"]:.4f} s',
        f'- Speedup: {parallel_results["speedup"]:.4f}x',
        f'- Task count: {parallel_results["task_count"]}',
        f'- Max log-likelihood delta between sequential and parallel: {parallel_results["max_log_likelihood_delta"]:.8f}',
        '',
        '## Kernel Candidates',
        '',
    ]
    for name, record in kernel_results.items():
        lines.append(f'- {name}:')
        if not record.get('available', True):
            lines.append(f'  unavailable ({record.get("reason", "unknown")})')
            continue
        lines.append(f'  wall_s={record["wall_s"]:.4f}, speedup_vs_python={record["speedup_vs_python"]:.4f}x')
        if 'difference_vs_python' in record:
            diff = record['difference_vs_python']
            lines.append(
                '  difference_vs_python='
                f'max_abs {diff["max_abs"]:.8e}, rmse {diff["rmse"]:.8e}, relative_l2 {diff["relative_l2"]:.8e}'
            )
        if 'device_name' in record:
            lines.append(f'  device_name={record["device_name"]}')
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    audit = load_audit_module()
    output_dir = resolve_output_dir(str(args.output_dir))
    bundle, filter_config, params, null_bundle = build_benchmark_context(audit, args)
    fixture = make_kernel_fixture(bundle, filter_config, int(args.kernel_seed), int(args.kernel_eeg_steps))
    versions = parse_versions()
    torch_device = select_torch_device(str(args.torch_device))

    parallel_results = benchmark_exact_parallelization(
        audit,
        bundle,
        null_bundle,
        filter_config,
        params,
        workers=int(args.parallel_workers),
    )
    kernel_results = benchmark_kernel_candidates(audit, filter_config, params, fixture, torch_device)

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
            'seed_list': list(filter_config.seed_list),
            'integration_dt_s': float(filter_config.integration_dt_s),
            'time_shift_null_s': float(filter_config.time_shift_null_s),
        },
        'kernel_fixture': {
            'eeg_steps': int(fixture['proposal_centers'].shape[0]),
            'num_particles': int(fixture['particles0'].shape[0]),
            'kernel_seed': int(args.kernel_seed),
        },
        'environment': {
            'versions': versions,
            'torch_device': torch_device,
            'cpu_count': os.cpu_count(),
        },
    }
    results = {
        'exact_parallelization': parallel_results,
        'kernel_candidates': kernel_results,
    }

    (output_dir / 'benchmark_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (output_dir / 'benchmark_results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (output_dir / 'benchmark_summary.md').write_text(
        build_summary_text(args, bundle, filter_config, parallel_results, kernel_results, versions, torch_device),
        encoding='utf-8',
    )

    print(json.dumps({'output_dir': str(output_dir), 'parallel_speedup': parallel_results['speedup'], 'kernel_results': kernel_results}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()