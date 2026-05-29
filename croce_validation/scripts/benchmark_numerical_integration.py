"""Benchmark numerical integration methods vs matrix exponential for Croce 2017 ODE.

Compares explicit numerical integrators (Euler, Heun/RK2, RK4) against the
current augmented-matrix-exponential baseline across four experimental phases:

  Phase 1 — Single-step accuracy: 5000 random states, dt=0.005s
  Phase 2 — Trajectory error accumulation: deterministic multi-step, no noise
  Phase 3 — Full particle filter: identical RNG seeds, compare PF outputs
  Phase 4 — End-to-end target cache: compare source/observation estimates

Outputs are written to croce_validation/results/numint_benchmark_<timestamp>/.

Design notes
------------
The current state evolution solves dx/dt = J*x + f (linearized around x0)
exactly via the augmented matrix exponential.  Numerical integrators solve the
full nonlinear ODE dx/dt = f(x) directly.  For dt = 0.005 s (200 Hz EEG) the
linearisation error and the integrator truncation error are both O(dt^2) or
better, so we expect sub-percent agreement per step.

Phase 3 forces solver_backend='torch_exact' for all kernels so that the only
difference is the state-propagation kernel — the RNG stream, resampling, and
likelihood computation are identical.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_SCRIPT = PROJECT_ROOT / "croce_validation" / "scripts" / "run_local_neighborhood_solver_audit.py"
AUDIT_MODULE_NAME = "numint_bench_audit"


# ---------------------------------------------------------------------------
# Module loader (used once; Phase 3/4 use save/restore, not re-import)
# ---------------------------------------------------------------------------

def _load_audit_module() -> Any:
    spec = importlib.util.spec_from_file_location(AUDIT_MODULE_NAME, str(AUDIT_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load audit module from {AUDIT_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[AUDIT_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark numerical integration vs matrix exponential for Croce 2017 ODE."
    )
    p.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    p.add_argument("--subject-id", type=int, default=1)
    p.add_argument("--session-idx", type=int, default=0)
    p.add_argument("--anchor-fnirs-channel", default="S1_D1 760")
    p.add_argument("--segment-start-s", type=float, default=30.0)
    p.add_argument("--segment-duration-s", type=float, default=60.0)
    p.add_argument("--output-dir", default=str(PROJECT_ROOT / "croce_validation" / "results"))
    p.add_argument("--num-particles", type=int, default=224)
    p.add_argument("--sigma-prop", type=float, default=5.0)
    p.add_argument("--sigma-nirs", type=float, default=0.15)
    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--trajectory-steps", type=int, default=2000)
    p.add_argument("--torch-device", default="cpu")
    p.add_argument("--phase", default="all",
                   choices=["1", "2", "3", "4", "all"])
    # Additional args forwarded to build_benchmark_context
    p.add_argument("--eeg-neighbors", type=int, default=8)
    p.add_argument("--fnirs-neighbors", type=int, default=4)
    p.add_argument("--eeg-radius-mm", type=float, default=40.0)
    p.add_argument("--fnirs-radius-mm", type=float, default=20.0)
    p.add_argument("--eeg-sigma-mm", type=float, default=15.0)
    p.add_argument("--fnirs-sigma-mm", type=float, default=10.0)
    p.add_argument("--eeg-sign-mode", default="covariance")
    p.add_argument("--resample-fraction", type=float, default=0.5)
    p.add_argument("--prior-std", default="0.1,0.05,0.03,0.02,1.0")
    p.add_argument("--state-noise-std", default="0.01,0.005,0.003,0.002,0.0")
    p.add_argument("--seed-list", default="1000,1001,1002")
    p.add_argument("--time-shift-null-s", type=float, default=2.0)
    p.add_argument("--use-artifact-eeg", action="store_true", default=False)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Numerical integration kernels (NumPy scalar) — for Phase 1 & 2
# ---------------------------------------------------------------------------

def _euler_step(x: np.ndarray, dt: float, drift_fn, params: Any) -> np.ndarray:
    f0 = np.asarray(drift_fn(x, params), dtype=np.float64)
    nx = np.asarray(x, dtype=np.float64) + dt * f0
    nx[1:4] = np.clip(nx[1:4], -0.95, None)
    return nx


def _heun_step(x: np.ndarray, dt: float, drift_fn, params: Any) -> np.ndarray:
    f0 = np.asarray(drift_fn(x, params), dtype=np.float64)
    x1 = np.asarray(x, dtype=np.float64) + dt * f0
    x1[1:4] = np.clip(x1[1:4], -0.95, None)
    f1 = np.asarray(drift_fn(x1, params), dtype=np.float64)
    nx = np.asarray(x, dtype=np.float64) + (dt / 2.0) * (f0 + f1)
    nx[1:4] = np.clip(nx[1:4], -0.95, None)
    return nx


def _rk4_step(x: np.ndarray, dt: float, drift_fn, params: Any) -> np.ndarray:
    k1 = np.asarray(drift_fn(x, params), dtype=np.float64)
    x2 = np.asarray(x, dtype=np.float64) + 0.5 * dt * k1
    x2[1:4] = np.clip(x2[1:4], -0.95, None)
    k2 = np.asarray(drift_fn(x2, params), dtype=np.float64)
    x3 = np.asarray(x, dtype=np.float64) + 0.5 * dt * k2
    x3[1:4] = np.clip(x3[1:4], -0.95, None)
    k3 = np.asarray(drift_fn(x3, params), dtype=np.float64)
    x4 = np.asarray(x, dtype=np.float64) + dt * k3
    x4[1:4] = np.clip(x4[1:4], -0.95, None)
    k4 = np.asarray(drift_fn(x4, params), dtype=np.float64)
    nx = np.asarray(x, dtype=np.float64) + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    nx[1:4] = np.clip(nx[1:4], -0.95, None)
    return nx


NUMERICAL_KERNELS = {"euler": _euler_step, "heun": _heun_step, "rk4": _rk4_step}


# ---------------------------------------------------------------------------
# PyTorch batched numerical integration kernels — for Phase 3 & 4
#
# These have the SAME signature as audit.torch_local_linearized_step_batch:
#     fn(particles: Tensor[N,5], dt: float, params: ModelParams) -> Tensor[N,5]
# so they are drop-in replacements via monkey-patching.
# ---------------------------------------------------------------------------

def _torch_safe_extraction_fraction(flow: torch.Tensor, e0: float) -> torch.Tensor:
    base = torch.tensor(1.0 - e0, dtype=flow.dtype, device=flow.device)
    return 1.0 - torch.pow(base, 1.0 / torch.clamp(flow, min=1e-4))


def _torch_state_drift_batch(particles: torch.Tensor, params: Any) -> torch.Tensor:
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

    extraction = _torch_safe_extraction_fraction(f, e0) / max(e0, 1e-8)

    return torch.stack(
        [
            float(params.epsilon) * r - float(params.kas) * s - float(params.kaf) * (f - 1.0),
            s,
            (f - torch.pow(hbo, 1.0 / alpha)) / tau0,
            (f * extraction - hb * torch.pow(hbo, (1.0 / alpha) - 1.0)) / tau0,
            torch.zeros_like(r),
        ],
        dim=1,
    )


def _torch_euler_step_batch(particles: torch.Tensor, dt: float, params: Any) -> torch.Tensor:
    drift = _torch_state_drift_batch(particles, params)
    nx = particles + dt * drift
    nx[:, 1:4] = torch.clamp(nx[:, 1:4], min=-0.95)
    return nx


def _torch_heun_step_batch(particles: torch.Tensor, dt: float, params: Any) -> torch.Tensor:
    f0 = _torch_state_drift_batch(particles, params)
    x1 = particles + dt * f0
    x1[:, 1:4] = torch.clamp(x1[:, 1:4], min=-0.95)
    f1 = _torch_state_drift_batch(x1, params)
    nx = particles + (dt / 2.0) * (f0 + f1)
    nx[:, 1:4] = torch.clamp(nx[:, 1:4], min=-0.95)
    return nx


def _torch_rk4_step_batch(particles: torch.Tensor, dt: float, params: Any) -> torch.Tensor:
    k1 = _torch_state_drift_batch(particles, params)
    x2 = particles + 0.5 * dt * k1
    x2[:, 1:4] = torch.clamp(x2[:, 1:4], min=-0.95)
    k2 = _torch_state_drift_batch(x2, params)
    x3 = particles + 0.5 * dt * k2
    x3[:, 1:4] = torch.clamp(x3[:, 1:4], min=-0.95)
    k3 = _torch_state_drift_batch(x3, params)
    x4 = particles + dt * k3
    x4[:, 1:4] = torch.clamp(x4[:, 1:4], min=-0.95)
    k4 = _torch_state_drift_batch(x4, params)
    nx = particles + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    nx[:, 1:4] = torch.clamp(nx[:, 1:4], min=-0.95)
    return nx


TORCH_NUMERICAL_KERNELS = {
    "euler": _torch_euler_step_batch,
    "heun": _torch_heun_step_batch,
    "rk4": _torch_rk4_step_batch,
}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _summarize_difference(candidate: np.ndarray, baseline: np.ndarray) -> Dict[str, float]:
    diff = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    denom = max(float(np.linalg.norm(baseline)), 1e-12)
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "relative_l2": float(np.linalg.norm(diff) / denom),
    }


def _time_call(fn, *args, **kwargs) -> Tuple[float, Any]:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed, result


def _per_dim_rmse(candidate: np.ndarray, baseline: np.ndarray) -> List[float]:
    return [float(np.sqrt(np.mean((candidate[:, d] - baseline[:, d]) ** 2))) for d in range(candidate.shape[1])]


# ---------------------------------------------------------------------------
# Phase 1: Single-step accuracy
# ---------------------------------------------------------------------------

def _generate_random_states(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.column_stack([
        rng.uniform(-0.3, 0.3, size=n),     # s
        rng.uniform(-0.15, 0.25, size=n),    # delta_f
        rng.uniform(-0.08, 0.12, size=n),    # delta_hbo
        rng.uniform(-0.06, 0.08, size=n),    # delta_hb
        rng.uniform(-8.0, 8.0, size=n),      # r
    ]).astype(np.float64)


def run_phase1(audit: Any, params: Any, dt: float) -> Dict[str, Any]:
    """Single-step accuracy: numerical integrators vs expm on 5000 random states."""
    rng = np.random.default_rng(42)
    test_states = _generate_random_states(5000, rng)
    drift_fn = audit.state_drift

    # expm baseline (scipy, scalar loop)
    expm_results = np.zeros_like(test_states)
    for i in range(test_states.shape[0]):
        expm_results[i] = audit.local_linearized_step(test_states[i], dt, params)

    results: Dict[str, Any] = {"n_states": int(test_states.shape[0]), "dt_s": float(dt)}

    # NumPy scalar numerical integrators
    for name, kernel in NUMERICAL_KERNELS.items():
        numint_results = np.zeros_like(test_states)
        for i in range(test_states.shape[0]):
            numint_results[i] = kernel(test_states[i], dt, drift_fn, params)
        diff = _summarize_difference(numint_results, expm_results)
        diff["per_dim_max_abs"] = [
            float(v) for v in np.max(np.abs(numint_results - expm_results), axis=0)
        ]
        results[name] = diff

    # Torch batched versions (for cross-validation)
    if torch is not None:
        for name, kernel in TORCH_NUMERICAL_KERNELS.items():
            pt = torch.from_numpy(test_states).to(dtype=torch.float64)
            numint_t = kernel(pt, dt, params).cpu().numpy()
            diff = _summarize_difference(numint_t, expm_results)
            diff["per_dim_max_abs"] = [
                float(v) for v in np.max(np.abs(numint_t - expm_results), axis=0)
            ]
            results[f"torch_{name}"] = diff

        # Also confirm torch matrix_exp matches scipy expm (sanity check)
        torch_exact_fn = getattr(audit, "torch_local_linearized_step_batch", None)
        if torch_exact_fn is not None:
            pt = torch.from_numpy(test_states).to(dtype=torch.float64)
            torch_exact = torch_exact_fn(pt, dt, params).cpu().numpy()
            diff = _summarize_difference(torch_exact, expm_results)
            results["torch_exact_vs_scipy"] = diff

    return results


# ---------------------------------------------------------------------------
# Phase 2: Trajectory-level error accumulation
# ---------------------------------------------------------------------------

def _evolve_trajectory_numerical(
    x0: np.ndarray, dt: float, n_steps: int, drift_fn, step_fn, params: Any,
) -> np.ndarray:
    traj = np.zeros((n_steps + 1, 5), dtype=np.float64)
    traj[0] = x0
    x = x0.copy()
    for i in range(n_steps):
        x = step_fn(x, dt, drift_fn, params)
        traj[i + 1] = x
    return traj


def _evolve_trajectory_exact(
    x0: np.ndarray, dt: float, n_steps: int, audit: Any, params: Any,
) -> np.ndarray:
    traj = np.zeros((n_steps + 1, 5), dtype=np.float64)
    traj[0] = x0
    x = x0.copy()
    for i in range(n_steps):
        x = audit.local_linearized_step(x, dt, params)
        traj[i + 1] = x
    return traj


def run_phase2(audit: Any, params: Any, dt: float, n_steps: int) -> Dict[str, Any]:
    """Trajectory error accumulation: deterministic multi-step, no process noise.

    Tests three initial conditions with different r values to cover different
    dynamical regimes (weak, medium, and strong neural drive).
    """
    drift_fn = audit.state_drift

    # Three initial conditions varying only r
    x0_weak = np.array([0.05, 0.02, 0.01, 0.005, 1.0], dtype=np.float64)
    x0_med = np.array([0.05, 0.02, 0.01, 0.005, 4.0], dtype=np.float64)
    x0_strong = np.array([0.05, 0.02, 0.01, 0.005, 8.0], dtype=np.float64)
    initial_conditions = {"weak_r1.0": x0_weak, "medium_r4.0": x0_med, "strong_r8.0": x0_strong}

    results: Dict[str, Any] = {
        "n_steps": n_steps,
        "dt_s": float(dt),
        "total_time_s": float(n_steps * dt),
    }

    for ic_name, x0 in initial_conditions.items():
        ic_results: Dict[str, Any] = {"initial_state": [float(v) for v in x0]}

        traj_exact = _evolve_trajectory_exact(x0, dt, n_steps, audit, params)

        for name, kernel in NUMERICAL_KERNELS.items():
            wall_s, traj_num = _time_call(
                _evolve_trajectory_numerical, x0, dt, n_steps, drift_fn, kernel, params,
            )
            diff = _summarize_difference(traj_num, traj_exact)
            diff["wall_s"] = float(wall_s)
            diff["per_dim_rmse"] = _per_dim_rmse(traj_num, traj_exact)
            # Error growth: RMSE at 1%, 10%, 50%, 100% of trajectory
            checkpoints = [max(1, int(n_steps * p)) for p in [0.01, 0.1, 0.5, 1.0]]
            diff["error_growth"] = {
                f"step_{c}": float(np.sqrt(np.mean((traj_num[:c] - traj_exact[:c]) ** 2)))
                for c in checkpoints
            }
            ic_results[name] = diff

        # expm baseline timing
        wall_exact, _ = _time_call(_evolve_trajectory_exact, x0, dt, n_steps, audit, params)
        ic_results["expm_wall_s"] = float(wall_exact)
        results[ic_name] = ic_results

    return results


# ---------------------------------------------------------------------------
# Phase 3: Full particle filter — save/restore patch
# ---------------------------------------------------------------------------

def _make_torch_config(filter_config: Any, torch_device: str) -> Any:
    """Create a new FilterConfig with solver_backend='torch_exact' (frozen dataclass)."""
    from dataclasses import replace  # type: ignore[attr-defined]
    return replace(filter_config, solver_backend="torch_exact", torch_device=torch_device)


def _patch_and_run_pf(
    audit: Any,
    bundle: Any,
    filter_config: Any,
    params: Any,
    seed: int,
    kernel_name: str,
    kernel_fn,
    torch_device: str,
) -> Dict[str, Any]:
    """Run the PF with a patched state-propagation kernel.

    Save the original torch_local_linearized_step_batch, install the numerical
    kernel, use a torch_exact config, run, then restore.
    """
    torch_config = _make_torch_config(filter_config, torch_device)

    if kernel_name == "expm":
        return audit.run_particle_filter(bundle, torch_config, params, seed)

    original_fn = audit.torch_local_linearized_step_batch

    def _wrapper(particles: torch.Tensor, _dt: float, _params: Any) -> torch.Tensor:
        return kernel_fn(particles, float(filter_config.integration_dt_s), params)

    audit.torch_local_linearized_step_batch = _wrapper

    try:
        result = audit.run_particle_filter(bundle, torch_config, params, seed)
    finally:
        audit.torch_local_linearized_step_batch = original_fn

    return result


def run_phase3(
    audit: Any,
    bundle: Any,
    filter_config: Any,
    params: Any,
    seeds: List[int],
    torch_device: str,
) -> Dict[str, Any]:
    """Full particle filter: identical RNG, compare log-likelihood / ESS across kernels."""
    results: Dict[str, Any] = {
        "seeds": seeds,
        "num_particles": int(filter_config.num_particles),
        "sigma_prop": float(filter_config.sigma_prop),
        "torch_device": torch_device,
    }

    all_kernels = [("expm", None)] + [(k, v) for k, v in TORCH_NUMERICAL_KERNELS.items()]

    for kname, kfn in all_kernels:
        total_wall_s = 0.0
        log_likes = []
        ess_means = []

        for seed in seeds:
            wall_s, pf_result = _time_call(
                _patch_and_run_pf, audit, bundle, filter_config, params, seed, kname, kfn, torch_device,
            )
            total_wall_s += wall_s
            log_likes.append(float(pf_result["log_likelihood"]))
            ess_means.append(float(np.mean(pf_result["ess_trace"])))

        entry: Dict[str, Any] = {
            "total_wall_s": float(total_wall_s),
            "mean_wall_s": float(total_wall_s / len(seeds)),
            "log_likelihoods": log_likes,
            "mean_log_likelihood": float(np.mean(log_likes)),
            "std_log_likelihood": float(np.std(log_likes)),
            "mean_ess": float(np.mean(ess_means)),
        }

        if kname != "expm" and "expm" in results:
            expm_lls = results["expm"]["log_likelihoods"]
            ll_deltas = [abs(a - b) for a, b in zip(log_likes, expm_lls)]
            entry["max_log_likelihood_delta"] = float(max(ll_deltas))
            entry["mean_log_likelihood_delta"] = float(np.mean(ll_deltas))

        results[kname] = entry

    return results


# ---------------------------------------------------------------------------
# Phase 4: End-to-end target cache consistency (deep comparison)
# ---------------------------------------------------------------------------

def run_phase4(
    audit: Any,
    bundle: Any,
    filter_config: Any,
    params: Any,
    seeds: List[int],
    torch_device: str,
) -> Dict[str, Any]:
    """Deep comparison of PF outputs: state estimates, ESS trace, observation predictions."""
    seed = seeds[0]
    results: Dict[str, Any] = {"seed": seed}

    # expm baseline
    _, pf_expm = _time_call(
        _patch_and_run_pf, audit, bundle, filter_config, params, seed, "expm", None, torch_device,
    )

    for kname, kfn in TORCH_NUMERICAL_KERNELS.items():
        _, pf_num = _time_call(
            _patch_and_run_pf, audit, bundle, filter_config, params, seed, kname, kfn, torch_device,
        )

        comparisons: Dict[str, Any] = {}

        # State estimates (fnirs-rate, [T, 5])
        comparisons["state_estimates"] = _summarize_difference(
            pf_num["state_estimates"], pf_expm["state_estimates"],
        )
        comparisons["state_per_dim_rmse"] = _per_dim_rmse(
            pf_num["state_estimates"], pf_expm["state_estimates"],
        )

        # ESS trace
        comparisons["ess_trace"] = _summarize_difference(
            np.asarray(pf_num["ess_trace"]), np.asarray(pf_expm["ess_trace"]),
        )

        # Log likelihood
        comparisons["log_likelihood_delta"] = float(
            abs(pf_num["log_likelihood"] - pf_expm["log_likelihood"])
        )

        # Observation predictions (correlation between methods' predictions)
        for obs_key in ["pred_eeg", "pred_primary", "pred_secondary"]:
            if obs_key in pf_expm and obs_key in pf_num:
                comparisons[f"{obs_key}_diff"] = _summarize_difference(
                    pf_num[obs_key], pf_expm[obs_key],
                )

        # r_estimates at EEG rate
        if "r_estimates_eeg" in pf_expm and "r_estimates_eeg" in pf_num:
            comparisons["r_estimates_eeg_diff"] = _summarize_difference(
                pf_num["r_estimates_eeg"].reshape(-1, 1),
                pf_expm["r_estimates_eeg"].reshape(-1, 1),
            )

        results[kname] = comparisons

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _format_phase1_report(data: Dict[str, Any]) -> str:
    lines = [
        "## Phase 1 — Single-Step Accuracy",
        "",
        f"- States tested: {data['n_states']}",
        f"- dt: {data['dt_s']} s",
        "",
        "| Method | max_abs | rmse | relative_l2 | per-dim max_abs (s, df, dhbo, dhb, r) |",
        "|--------|---------|------|-------------|-----------------------------------------|",
    ]
    for name in ["euler", "heun", "rk4", "torch_euler", "torch_heun", "torch_rk4", "torch_exact_vs_scipy"]:
        if name not in data:
            continue
        d = data[name]
        pdm = d.get("per_dim_max_abs", [0] * 5)
        lines.append(
            f"| {name} | {d['max_abs']:.2e} | {d['rmse']:.2e} | {d['relative_l2']:.2e}"
            f" | {pdm[0]:.2e}, {pdm[1]:.2e}, {pdm[2]:.2e}, {pdm[3]:.2e}, {pdm[4]:.2e} |"
        )
    lines.append("")
    return "\n".join(lines)


def _format_phase2_report(data: Dict[str, Any]) -> str:
    lines = [
        "## Phase 2 — Trajectory Error Accumulation",
        "",
        f"- Steps: {data['n_steps']}, dt: {data['dt_s']} s, total: {data['total_time_s']} s",
        "",
    ]
    for ic_name in sorted(data.keys()):
        if not isinstance(data[ic_name], dict) or "expm_wall_s" not in data[ic_name]:
            continue
        ic = data[ic_name]
        lines.append(f"### Initial condition: {ic_name}")
        lines.append(f"  initial state = [{', '.join(f'{v:.3f}' for v in ic['initial_state'])}]")
        lines.append(f"  expm wall = {ic['expm_wall_s']:.4f} s")
        lines.append("")
        lines.append("| Method | wall_s | speedup | rmse | per-dim RMSE (s,df,dhbo,dhb,r) | error @ 1%/10%/50%/100% |")
        lines.append("|--------|--------|---------|------|-------------------------------|--------------------------|")
        for name in ["euler", "heun", "rk4"]:
            if name not in ic:
                continue
            d = ic[name]
            sp = ic["expm_wall_s"] / max(d["wall_s"], 1e-12)
            pdr = ",".join(f"{v:.2e}" for v in d["per_dim_rmse"])
            keys = sorted(d["error_growth"].keys())
            growth = " / ".join(f"{d['error_growth'][k]:.2e}" for k in keys)
            lines.append(
                f"| {name} | {d['wall_s']:.4f} | {sp:.2f}x | {d['rmse']:.2e} | {pdr} | {growth} |"
            )
        lines.append("")
    return "\n".join(lines)


def _format_phase3_report(data: Dict[str, Any]) -> str:
    lines = [
        "## Phase 3 — Full Particle Filter Comparison",
        "",
        f"- Seeds: {data['seeds']}, particles: {data['num_particles']}",
        f"- sigma_prop: {data['sigma_prop']}, device: {data['torch_device']}",
        "",
        "| Method | mean_wall_s | speedup | mean_LL | std_LL | mean_ESS | max_LL_delta_vs_expm |",
        "|--------|-------------|---------|---------|--------|----------|----------------------|",
    ]
    expm_mean = data.get("expm", {}).get("mean_wall_s", 1.0)
    for name in ["expm", "euler", "heun", "rk4"]:
        if name not in data:
            continue
        d = data[name]
        sp = expm_mean / max(d["mean_wall_s"], 1e-12) if name != "expm" else 1.0
        delta = f"{d.get('mean_log_likelihood_delta', 0):.4f}" if name != "expm" else "—"
        lines.append(
            f"| {name} | {d['mean_wall_s']:.4f} | {sp:.2f}x"
            f" | {d['mean_log_likelihood']:.4f} | {d['std_log_likelihood']:.4f}"
            f" | {d['mean_ess']:.1f} | {delta} |"
        )
    lines.append("")
    return "\n".join(lines)


def _format_phase4_report(data: Dict[str, Any]) -> str:
    lines = [
        "## Phase 4 — End-to-End Target Cache Consistency",
        "",
        f"- Seed: {data['seed']}",
        "",
        "| Method | LL delta | state_est rmse | ESS rmse | r_est rmse | per-dim RMSE (s,df,dhbo,dhb,r) |",
        "|--------|----------|---------------|----------|------------|-------------------------------|",
    ]
    for name in ["euler", "heun", "rk4"]:
        if name not in data:
            continue
        d = data[name]
        state = d.get("state_estimates", {})
        ess = d.get("ess_trace", {})
        r_est = d.get("r_estimates_eeg_diff", {})
        pdr = ",".join(f"{v:.2e}" for v in d.get("state_per_dim_rmse", [float("nan")] * 5))
        lines.append(
            f"| {name} | {d['log_likelihood_delta']:.4e}"
            f" | {state.get('rmse', float('nan')):.2e}"
            f" | {ess.get('rmse', float('nan')):.2e}"
            f" | {r_est.get('rmse', float('nan')):.2e}"
            f" | {pdr} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_full_report(
    args: argparse.Namespace,
    phase1: Optional[Dict],
    phase2: Optional[Dict],
    phase3: Optional[Dict],
    phase4: Optional[Dict],
    versions: Dict[str, str],
) -> str:
    parts = [
        "# Numerical Integration vs Matrix Exponential — Benchmark Report",
        "",
        "## Configuration",
        "",
        f"- Subject {args.subject_id}, session {args.session_idx}, anchor {args.anchor_fnirs_channel}",
        f"- Segment: start {args.segment_start_s}s, duration {args.segment_duration_s}s",
        f"- Particles: {args.num_particles}, sigma_prop={args.sigma_prop}, sigma_nirs={args.sigma_nirs}",
        f"- Torch device: {args.torch_device}",
        f"- Trajectory steps (Phase 2): {args.trajectory_steps}",
        "",
        "## Environment",
        "",
        f"- Python {versions.get('python', '?')}, NumPy {versions.get('numpy', '?')}, "
        f"SciPy {versions.get('scipy', '?')}, PyTorch {versions.get('torch', '?')}",
        "",
    ]
    if phase1:
        parts.append(_format_phase1_report(phase1))
    if phase2:
        parts.append(_format_phase2_report(phase2))
    if phase3:
        parts.append(_format_phase3_report(phase3))
    if phase4:
        parts.append(_format_phase4_report(phase4))
    return "\n".join(parts)


def _parse_versions() -> Dict[str, str]:
    import scipy
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__ if torch is not None else "unavailable",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    audit = _load_audit_module()
    params = audit.ModelParams()
    dt = 0.005  # standard Croce 2017 integration step at 200 Hz EEG
    versions = _parse_versions()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"numint_benchmark_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    need_data = args.phase in ("3", "4", "all")
    bundle = None
    filter_config = None
    seed_list = [int(1000 + i) for i in range(args.num_seeds)]

    if need_data:
        sys.path.insert(0, str(PROJECT_ROOT / "croce_validation" / "scripts"))
        from benchmark_solver_optimizations import build_benchmark_context
        bundle, filter_config, params_from_data, _null_bundle = build_benchmark_context(audit, args)
        params = params_from_data
        dt = float(filter_config.integration_dt_s)
        print(f"dt = {dt:.4f}s, eeg_fs = {bundle.eeg_fs_hz}Hz, fnirs_fs = {bundle.fnirs_fs_hz}Hz")
        print(f"EEG steps = {bundle.eeg_time_s.shape[0]}, fNIRS steps = {bundle.time_s.shape[0]}")

    phase1_results = None
    phase2_results = None
    phase3_results = None
    phase4_results = None

    if args.phase in ("1", "all"):
        print("\n=== Phase 1: Single-Step Accuracy ===")
        phase1_results = run_phase1(audit, params, dt)
        print(_format_phase1_report(phase1_results))

    if args.phase in ("2", "all"):
        print("\n=== Phase 2: Trajectory Error Accumulation ===")
        phase2_results = run_phase2(audit, params, dt, args.trajectory_steps)
        print(_format_phase2_report(phase2_results))

    if args.phase in ("3", "all"):
        print("\n=== Phase 3: Full Particle Filter Comparison ===")
        if torch is None:
            print("  SKIPPED: PyTorch not available")
        elif bundle is None:
            print("  SKIPPED: data not loaded (use --phase 3 with real data)")
        else:
            phase3_results = run_phase3(audit, bundle, filter_config, params, seed_list, args.torch_device)
            print(_format_phase3_report(phase3_results))

    if args.phase in ("4", "all"):
        print("\n=== Phase 4: End-to-End Target Cache ===")
        if torch is None:
            print("  SKIPPED: PyTorch not available")
        elif bundle is None:
            print("  SKIPPED: data not loaded (use --phase 4 with real data)")
        else:
            phase4_results = run_phase4(audit, bundle, filter_config, params, seed_list, args.torch_device)
            print(_format_phase4_report(phase4_results))

    # Write outputs
    report = build_full_report(args, phase1_results, phase2_results, phase3_results, phase4_results, versions)
    (output_dir / "benchmark_report.md").write_text(report, encoding="utf-8")

    all_results = {
        "config": {
            "subject_id": args.subject_id,
            "session_idx": args.session_idx,
            "anchor": args.anchor_fnirs_channel,
            "segment_start_s": args.segment_start_s,
            "segment_duration_s": args.segment_duration_s,
            "num_particles": args.num_particles,
            "sigma_prop": args.sigma_prop,
            "sigma_nirs": args.sigma_nirs,
            "num_seeds": args.num_seeds,
            "trajectory_steps": args.trajectory_steps,
            "dt_s": dt,
            "torch_device": args.torch_device,
        },
        "versions": versions,
        "phase1": phase1_results,
        "phase2": phase2_results,
        "phase3": phase3_results,
        "phase4": phase4_results,
    }
    (output_dir / "benchmark_results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"\nResults written to {output_dir}")


if __name__ == "__main__":
    main()
