"""Fine-grained profiler for the Croce SMC particle filter pipeline.

Instruments every computational step of run_particle_filter_torch_exact to
measure wall time, CPU time, memory pressure, and numpy/torch transfer overhead.
Designed to identify optimization entry points for full-dataset cache generation.

Usage:
    python croce_validation/scripts/profile_particle_filter_pipeline.py \
        --data-root "data/EEG+NIRS Single-Trial" \
        --subject-id 1 --session-idx 0 \
        --anchor-fnirs-channel AF7Fp1 \
        --segment-start-s 60.0 --segment-duration-s 30.0 \
        --sigma-prop 5.0 --sigma-nirs 0.15 \
        --num-particles 224 --seed 11 \
        --profile-seconds 15.0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import tracemalloc
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / 'croce_validation' / 'results'
AUDIT_SCRIPT = PROJECT_ROOT / 'croce_validation' / 'scripts' / 'run_local_neighborhood_solver_audit.py'
AUDIT_MODULE_NAME = 'croce_pf_profile_target'

try:
    import torch
except ImportError:
    torch = None

try:
    import psutil
    _PROCESS = psutil.Process()
except ImportError:
    psutil = None
    _PROCESS = None


# ---------------------------------------------------------------------------
# Profiler infrastructure
# ---------------------------------------------------------------------------

class StepTimer:
    """Context manager that records wall time, CPU time, and optional memory delta."""

    def __init__(self, registry: ProfileRegistry, step_name: str, group: str = ""):
        self._registry = registry
        self._step_name = step_name
        self._group = group
        self._wall_start: float = 0.0
        self._cpu_start: float = 0.0
        self._rss_start: int = 0

    def __enter__(self) -> "StepTimer":
        self._wall_start = time.perf_counter()
        self._cpu_start = time.process_time()
        if _PROCESS is not None:
            self._rss_start = _PROCESS.memory_info().rss
        return self

    def __exit__(self, *args: Any) -> None:
        wall = time.perf_counter() - self._wall_start
        cpu = time.process_time() - self._cpu_start
        rss_delta = 0
        if _PROCESS is not None:
            rss_delta = _PROCESS.memory_info().rss - self._rss_start
        self._registry.record(self._step_name, self._group, wall, cpu, rss_delta)


class ProfileRegistry:
    """Accumulates timing records and produces summary reports."""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        self._group_agg: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"wall": 0.0, "cpu": 0.0, "rss_delta": 0, "count": 0}
        )

    def record(self, step: str, group: str, wall: float, cpu: float, rss_delta: int) -> None:
        self.records.append(
            {"step": step, "group": group, "wall_s": wall, "cpu_s": cpu, "rss_delta_b": rss_delta}
        )
        key = f"{group}/{step}" if group else step
        agg = self._group_agg[key]
        agg["wall"] += wall
        agg["cpu"] += cpu
        agg["rss_delta"] += rss_delta
        agg["count"] += 1

    def summary(self, total_wall: float) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for key, agg in sorted(self._group_agg.items(), key=lambda x: -x[1]["wall"]):
            rows.append(
                {
                    "step": key,
                    "wall_total_s": round(agg["wall"], 6),
                    "wall_pct": round(100.0 * agg["wall"] / max(total_wall, 1e-12), 2),
                    "cpu_total_s": round(agg["cpu"], 6),
                    "count": agg["count"],
                    "wall_per_call_ms": round(1000.0 * agg["wall"] / max(agg["count"], 1), 4),
                    "rss_delta_mb": round(agg["rss_delta"] / (1024 * 1024), 4),
                }
            )
        return rows


# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-grained profiler for the Croce PF pipeline."
    )
    parser.add_argument("--data-root", default="data/EEG+NIRS Single-Trial")
    parser.add_argument("--subject-id", type=int, default=1)
    parser.add_argument("--session-idx", type=int, default=0)
    parser.add_argument("--anchor-fnirs-channel", default="AF7Fp1")
    parser.add_argument("--segment-start-s", type=float, default=60.0)
    parser.add_argument("--segment-duration-s", type=float, default=30.0)
    parser.add_argument("--use-artifact-eeg", action="store_true")

    parser.add_argument("--eeg-neighbors", type=int, default=6)
    parser.add_argument("--fnirs-neighbors", type=int, default=4)
    parser.add_argument("--eeg-radius-mm", type=float, default=60.0)
    parser.add_argument("--fnirs-radius-mm", type=float, default=45.0)
    parser.add_argument("--eeg-sigma-mm", type=float, default=30.0)
    parser.add_argument("--fnirs-sigma-mm", type=float, default=22.0)
    parser.add_argument("--eeg-sign-mode", choices=("covariance", "geometric_x"), default="covariance")

    parser.add_argument("--num-particles", type=int, default=224)
    parser.add_argument("--resample-fraction", type=float, default=0.5)
    parser.add_argument("--prior-std", default="0.05,0.05,0.05,0.05,0.0")
    parser.add_argument("--state-noise-std", default="0.02,0.015,0.015,0.015,0.0")
    parser.add_argument("--sigma-prop", type=float, default=5.0)
    parser.add_argument("--sigma-nirs", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=11)

    parser.add_argument("--torch-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--profile-seconds", type=float, default=0.0,
                        help="Cap profiling to first N seconds of data (0 = full segment)")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--trace-malloc", action="store_true",
                        help="Enable tracemalloc for Python-level allocation tracing")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Module loading helpers
# ---------------------------------------------------------------------------

def load_audit_module() -> Any:
    spec = importlib.util.spec_from_file_location(AUDIT_MODULE_NAME, AUDIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load audit module from {AUDIT_SCRIPT}")
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = RESULTS_ROOT / f"profile_pf_pipeline_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ---------------------------------------------------------------------------
# Environment snapshot
# ---------------------------------------------------------------------------

def capture_environment(device: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "torch": "unavailable",
        "torch_cuda_available": False,
        "cpu_count_logical": os.cpu_count(),
    }
    if torch is not None:
        info["torch"] = torch.__version__
        info["torch_cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["torch_cuda_device_count"] = torch.cuda.device_count()
            info["torch_cuda_device_name"] = torch.cuda.get_device_name(0)
            info["torch_cuda_mem_total_mb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1
            )
    if psutil is not None:
        mem = psutil.virtual_memory()
        info["system_ram_total_gb"] = round(mem.total / (1024**3), 1)
        info["system_ram_available_gb"] = round(mem.available / (1024**3), 1)
    info["selected_device"] = device
    return info


# ---------------------------------------------------------------------------
# Data loading (reuses audit module)
# ---------------------------------------------------------------------------

def build_configs(audit: Any, args: argparse.Namespace) -> Tuple[Any, Any, Any, Any]:
    spatial_config = audit.SpatialConfig(
        eeg_neighbors=int(args.eeg_neighbors),
        fnirs_neighbors=int(args.fnirs_neighbors),
        eeg_radius_mm=float(args.eeg_radius_mm),
        fnirs_radius_mm=float(args.fnirs_radius_mm),
        eeg_sigma_mm=float(args.eeg_sigma_mm),
        fnirs_sigma_mm=float(args.fnirs_sigma_mm),
        eeg_sign_mode=str(args.eeg_sign_mode),
    )
    dataset_args = argparse.Namespace(
        data_root=args.data_root,
        subject_id=int(args.subject_id),
        session_idx=int(args.session_idx),
        segment_start_s=float(args.segment_start_s),
        segment_duration_s=float(args.segment_duration_s),
        anchor_fnirs_channel=str(args.anchor_fnirs_channel),
        use_artifact_eeg=bool(args.use_artifact_eeg),
        eeg_unit="uV",
        fnirs_primary_unit="V",
        fnirs_secondary_unit="V",
    )
    bundle = audit.load_dataset_bundle(dataset_args, spatial_config)
    filter_config = audit.FilterConfig(
        integration_dt_s=float(1.0 / bundle.eeg_fs_hz),
        observation_fs_hz=float(bundle.fnirs_fs_hz),
        num_particles=int(args.num_particles),
        resample_fraction=float(args.resample_fraction),
        prior_std=audit.parse_vector(str(args.prior_std), name="prior-std"),
        state_noise_std=audit.parse_vector(str(args.state_noise_std), name="state-noise-std"),
        sigma_prop=float(args.sigma_prop),
        sigma_nirs=float(args.sigma_nirs),
        seed_list=(int(args.seed),),
        time_shift_null_s=8.0,
        run_spatial_null=False,
        solver_backend="torch_exact",
        torch_device=str(args.torch_device),
    )
    filter_config.prior_std[4] = 0.0
    filter_config.state_noise_std[4] = 0.0
    params = audit.ModelParams()
    return bundle, filter_config, params, spatial_config


# ---------------------------------------------------------------------------
# Instrumented particle filter
# ---------------------------------------------------------------------------

def run_particle_filter_profiled(
    audit: Any,
    bundle: Any,
    filter_config: Any,
    params: Any,
    seed: int,
    device: str,
    registry: ProfileRegistry,
    max_eeg_steps: int = 0,
) -> Dict[str, Any]:
    """Identical to run_particle_filter_torch_exact but with StepTimer probes."""

    rng = np.random.default_rng(seed)
    num_particles = int(filter_config.num_particles)
    timer = StepTimer

    # --- Initialization ---
    with timer(registry, "pf_init", "00.init"):
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
    if max_eeg_steps > 0:
        num_eeg_steps = min(num_eeg_steps, max_eeg_steps)
        num_fnirs_steps = max(num_eeg_steps // int(bundle.eeg_substeps_per_fnirs), 1)
        num_eeg_steps = num_fnirs_steps * int(bundle.eeg_substeps_per_fnirs)

    if num_fnirs_steps * int(bundle.eeg_substeps_per_fnirs) != num_eeg_steps:
        raise ValueError("EEG / fNIRS alignment mismatch")

    estimates = np.zeros((num_fnirs_steps, 5), dtype=np.float64)
    state_std = np.zeros((num_fnirs_steps, 5), dtype=np.float64)
    ess_trace = np.zeros(num_fnirs_steps, dtype=np.float64)
    r_estimates_eeg = np.zeros(num_eeg_steps, dtype=np.float64)
    r_std_eeg = np.zeros(num_eeg_steps, dtype=np.float64)
    log_likelihood_total = 0.0

    dt = float(filter_config.integration_dt_s)
    hemo_scale = np.asarray(filter_config.state_noise_std[:4], dtype=np.float64) * np.sqrt(dt)
    sigma_nirs_sq = max(float(filter_config.sigma_nirs), 1e-8) ** 2

    # --- Tensor constant upload (one-time) ---
    with timer(registry, "tensor_upload_constants", "00.init"):
        lead_field_t = torch.from_numpy(np.asarray(bundle.lead_field, dtype=np.float64)).to(
            device=device, dtype=torch.float64
        )
        jac_primary_t = torch.from_numpy(np.asarray(bundle.jac_primary, dtype=np.float64)).to(
            device=device, dtype=torch.float64
        )
        jac_secondary_t = torch.from_numpy(np.asarray(bundle.jac_secondary, dtype=np.float64)).to(
            device=device, dtype=torch.float64
        )
        fnirs_primary_t = torch.from_numpy(np.asarray(bundle.fnirs_primary_obs, dtype=np.float64)).to(
            device=device, dtype=torch.float64
        )
        fnirs_secondary_t = torch.from_numpy(np.asarray(bundle.fnirs_secondary_obs, dtype=np.float64)).to(
            device=device, dtype=torch.float64
        )

    if device == "cuda":
        torch.cuda.synchronize()

    # --- Main PF loop ---
    for step in range(num_fnirs_steps):
        eeg_start = step * int(bundle.eeg_substeps_per_fnirs)
        eeg_stop = eeg_start + int(bundle.eeg_substeps_per_fnirs)

        # --- EEG substeps ---
        for eeg_idx in range(eeg_start, eeg_stop):
            # Step A: numpy RNG for proposal noise
            with timer(registry, "proposal_noise_numpy", "01.substep"):
                proposal_center = float(bundle.r_eeg_projection[eeg_idx])
                proposal_noise = rng.normal(size=num_particles)

            # Step B: numpy -> torch transfer + r proposal
            with timer(registry, "proposal_transfer", "01.substep"):
                particles[:, 4] = (
                    proposal_center
                    + float(filter_config.sigma_prop)
                    * torch.from_numpy(proposal_noise).to(device=device, dtype=torch.float64)
                )

            # Step C: local linearized step (drift + jacobian + matrix_exp)
            with timer(registry, "local_linearized_step", "01.substep"):
                particles = audit.torch_local_linearized_step_batch(particles, dt, params)

            # Step D: numpy RNG for process noise
            with timer(registry, "process_noise_numpy", "01.substep"):
                process_noise = rng.normal(
                    loc=0.0,
                    scale=hemo_scale.reshape(1, 4),
                    size=(num_particles, 4),
                )

            # Step E: process noise transfer + addition + clipping
            with timer(registry, "process_noise_transfer_clip", "01.substep"):
                particles[:, 0:4] += torch.from_numpy(process_noise).to(
                    device=device, dtype=torch.float64
                )
                particles[:, 1:4] = torch.clamp(particles[:, 1:4], min=-0.95)

            # Step F: weighted r statistics
            with timer(registry, "r_statistics", "01.substep"):
                r_mean = torch.sum(particles[:, 4] * weights)
                centered_r = particles[:, 4] - r_mean
                r_estimates_eeg[eeg_idx] = float(r_mean.detach().cpu().item())
                r_std_eeg[eeg_idx] = float(
                    torch.sqrt(torch.sum(torch.square(centered_r) * weights)).detach().cpu().item()
                )

        # Step G: observation prediction
        with timer(registry, "predict_observations", "02.fnirs_step"):
            _, pred_primary_t, pred_secondary_t = audit.torch_predict_observations(
                particles, lead_field_t, jac_primary_t, jac_secondary_t, bundle.pair_mode,
            )

        # Step H: log-weight computation
        with timer(registry, "log_weight_compute", "02.fnirs_step"):
            log_weights = torch.log(torch.clamp(weights, min=1e-300))
            log_weights = log_weights + (
                -0.5
                * torch.sum(
                    torch.square(fnirs_primary_t[step].reshape(1, -1) - pred_primary_t), dim=1
                )
                / sigma_nirs_sq
            )
            log_weights = log_weights + (
                -0.5
                * torch.sum(
                    torch.square(fnirs_secondary_t[step].reshape(1, -1) - pred_secondary_t), dim=1
                )
                / sigma_nirs_sq
            )

        # Step I: weight normalization
        with timer(registry, "weight_normalize", "02.fnirs_step"):
            max_log_weight = torch.max(log_weights)
            stable = torch.exp(log_weights - max_log_weight)
            stable_sum = torch.clamp(torch.sum(stable), min=1e-12)
            weights = stable / stable_sum
            log_likelihood_total += float(
                (max_log_weight + torch.log(stable_sum)).detach().cpu().item()
            )

        # Step J: state estimates + CPU transfer
        with timer(registry, "state_estimate_transfer", "02.fnirs_step"):
            estimate_t = torch.sum(particles * weights.reshape(-1, 1), dim=0)
            centered = particles - estimate_t.reshape(1, -1)
            std_t = torch.sqrt(torch.sum(torch.square(centered) * weights.reshape(-1, 1), dim=0))
            estimates[step] = estimate_t.detach().cpu().numpy()
            state_std[step] = std_t.detach().cpu().numpy()
            r_estimates_eeg[eeg_stop - 1] = estimates[step, 4]
            r_std_eeg[eeg_stop - 1] = state_std[step, 4]

        # Step K: ESS + resampling (conditional)
        with timer(registry, "ess_resample", "02.fnirs_step"):
            ess = float((1.0 / torch.sum(torch.square(weights))).detach().cpu().item())
            ess_trace[step] = ess
            if ess < filter_config.resample_fraction * num_particles:
                indices = audit.systematic_resample(weights.detach().cpu().numpy(), rng)
                index_t = torch.from_numpy(indices).to(device=device, dtype=torch.int64)
                particles = particles[index_t]
                weights.fill_(1.0 / float(num_particles))

    if device == "cuda":
        torch.cuda.synchronize()

    return {
        "seed": seed,
        "state_estimates": estimates,
        "state_std": state_std,
        "ess_trace": ess_trace,
        "log_likelihood": float(log_likelihood_total),
        "r_estimates_eeg": r_estimates_eeg,
        "r_std_eeg": r_std_eeg,
    }


# ---------------------------------------------------------------------------
# Micro-benchmarks: isolate specific hotspots
# ---------------------------------------------------------------------------

def micro_benchmark_matrix_exp(device: str, num_particles: int, n_warmup: int = 5, n_repeat: int = 50) -> Dict[str, float]:
    """Profile torch.linalg.matrix_exp on batched 6x6 matrices."""
    if torch is None:
        return {"error": "torch unavailable"}
    augmented = torch.randn(num_particles, 6, 6, dtype=torch.float64, device=device) * 0.01
    # warmup
    for _ in range(n_warmup):
        _ = torch.linalg.matrix_exp(augmented)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        _ = torch.linalg.matrix_exp(augmented)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "operation": "matrix_exp (N,6,6)",
        "num_particles": num_particles,
        "device": device,
        "n_repeat": n_repeat,
        "total_s": round(elapsed, 6),
        "per_call_ms": round(1000.0 * elapsed / n_repeat, 4),
    }


def micro_benchmark_numpy_torch_transfer(device: str, num_particles: int, n_repeat: int = 500) -> Dict[str, float]:
    """Profile numpy->torch transfer overhead for particle-sized arrays."""
    if torch is None:
        return {"error": "torch unavailable"}
    arr_np = np.random.randn(num_particles).astype(np.float64)
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        t = torch.from_numpy(arr_np).to(device=device, dtype=torch.float64)
        _ = t + 1.0  # force a small op so transfer isn't optimised away
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "operation": "numpy->torch transfer + add",
        "num_particles": num_particles,
        "device": device,
        "n_repeat": n_repeat,
        "total_s": round(elapsed, 6),
        "per_call_ms": round(1000.0 * elapsed / n_repeat, 4),
    }


def micro_benchmark_numpy_rng(num_particles: int, n_repeat: int = 2000) -> Dict[str, float]:
    """Profile numpy random number generation for proposal + process noise."""
    rng = np.random.default_rng(20260528)
    # proposal noise
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        _ = rng.normal(size=num_particles)
    proposal_s = time.perf_counter() - t0
    # process noise (4D)
    rng2 = np.random.default_rng(20260528)
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        _ = rng2.normal(loc=0.0, scale=0.015, size=(num_particles, 4))
    process_s = time.perf_counter() - t0
    return {
        "proposal_rng_per_call_ms": round(1000.0 * proposal_s / n_repeat, 6),
        "process_rng_per_call_ms": round(1000.0 * process_s / n_repeat, 6),
        "n_repeat": n_repeat,
        "num_particles": num_particles,
    }


def micro_benchmark_drift_jacobian(device: str, num_particles: int, n_repeat: int = 200) -> Dict[str, float]:
    """Profile the drift+jacobian assembly (excluding matrix_exp) using the audit module."""
    audit = load_audit_module()
    params = audit.ModelParams()
    particles = torch.randn(num_particles, 5, dtype=torch.float64, device=device)
    dt = 0.005
    # warmup
    for _ in range(5):
        _ = audit.torch_local_linearized_step_batch(particles, dt, params)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        _ = audit.torch_local_linearized_step_batch(particles, dt, params)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "operation": "torch_local_linearized_step_batch (full, incl matrix_exp)",
        "num_particles": num_particles,
        "device": device,
        "n_repeat": n_repeat,
        "total_s": round(elapsed, 6),
        "per_call_ms": round(1000.0 * elapsed / n_repeat, 4),
    }


# ---------------------------------------------------------------------------
# Memory profiling via tracemalloc
# ---------------------------------------------------------------------------

def run_tracemalloc_snapshot(
    audit: Any,
    bundle: Any,
    filter_config: Any,
    params: Any,
    seed: int,
    device: str,
    max_eeg_steps: int,
) -> Dict[str, Any]:
    """Run a short PF segment under tracemalloc to capture allocation hotspots."""
    tracemalloc.start()
    t0 = time.perf_counter()

    result = run_particle_filter_profiled(
        audit, bundle, filter_config, params, seed, device,
        ProfileRegistry(),  # discard timing in this pass
        max_eeg_steps=max_eeg_steps,
    )

    elapsed = time.perf_counter() - t0
    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    top_stats = snapshot.statistics("lineno")[:30]
    top_frames: List[Dict[str, Any]] = []
    for stat in top_stats:
        frame = stat.traceback[0] if stat.traceback else None
        top_frames.append({
            "file": frame.filename if frame else "",
            "line": frame.lineno if frame else 0,
            "size_mb": round(stat.size / (1024 * 1024), 4),
            "count": stat.count,
        })

    # Domain-grouped summary
    domain_totals: Dict[str, float] = defaultdict(float)
    for stat in snapshot.statistics("lineno"):
        frame = stat.traceback[0] if stat.traceback else None
        if frame is None:
            continue
        fname = frame.filename
        if "torch" in fname:
            domain = "torch"
        elif "numpy" in fname:
            domain = "numpy"
        elif "scipy" in fname:
            domain = "scipy"
        elif "run_local_neighborhood" in fname or "profile_particle" in fname:
            domain = "audit_script"
        else:
            domain = "other"
        domain_totals[domain] += stat.size

    return {
        "wall_s": round(elapsed, 4),
        "top_frames": top_frames,
        "domain_totals_mb": {k: round(v / (1024 * 1024), 4) for k, v in sorted(domain_totals.items(), key=lambda x: -x[1])},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    audit = load_audit_module()
    output_dir = resolve_output_dir(str(args.output_dir))
    device = str(args.torch_device)
    env = capture_environment(device)

    print("=" * 72)
    print("Croce PF Pipeline Profiler")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    # ---- Phase 1: Data loading profile ----
    print("\n--- Phase 1: Data loading ---")
    t0 = time.perf_counter()
    bundle, filter_config, params, spatial_config = build_configs(audit, args)
    load_s = time.perf_counter() - t0
    print(f"  load_dataset_bundle: {load_s:.4f}s")

    num_fnirs = int(bundle.time_s.shape[0])
    num_eeg = int(bundle.eeg_time_s.shape[0])
    substeps = int(bundle.eeg_substeps_per_fnirs)
    print(f"  Segment: {num_fnirs} fNIRS steps x {substeps} substeps = {num_eeg} EEG steps")
    print(f"  Duration: {num_fnirs / bundle.fnirs_fs_hz:.1f}s fNIRS, {num_eeg / bundle.eeg_fs_hz:.1f}s EEG")
    print(f"  Particles: {filter_config.num_particles}")
    print(f"  sigma_prop={filter_config.sigma_prop}, sigma_nirs={filter_config.sigma_nirs}")

    max_eeg = 0
    if args.profile_seconds > 0:
        max_eeg = int(args.profile_seconds * bundle.eeg_fs_hz)
        print(f"  Profiling capped to first {args.profile_seconds}s ({max_eeg} EEG steps)")

    # ---- Phase 2: Instrumented PF run ----
    print("\n--- Phase 2: Instrumented particle filter ---")
    registry = ProfileRegistry()
    t0 = time.perf_counter()
    pf_result = run_particle_filter_profiled(
        audit, bundle, filter_config, params, int(args.seed), device, registry, max_eeg,
    )
    total_wall = time.perf_counter() - t0
    print(f"  Total PF wall time: {total_wall:.4f}s")
    print(f"  Log-likelihood: {pf_result['log_likelihood']:.4f}")

    # ---- Phase 3: Micro-benchmarks ----
    print("\n--- Phase 3: Micro-benchmarks ---")
    n_particles = int(filter_config.num_particles)

    micro_matrix_exp = micro_benchmark_matrix_exp(device, n_particles)
    print(f"  matrix_exp (N,6,6): {micro_matrix_exp['per_call_ms']:.4f}ms/call")

    micro_transfer = micro_benchmark_numpy_torch_transfer(device, n_particles)
    print(f"  numpy->torch transfer: {micro_transfer['per_call_ms']:.4f}ms/call")

    micro_rng = micro_benchmark_numpy_rng(n_particles)
    print(f"  numpy proposal RNG: {micro_rng['proposal_rng_per_call_ms']:.4f}ms/call")
    print(f"  numpy process RNG:  {micro_rng['process_rng_per_call_ms']:.4f}ms/call")

    micro_drift = micro_benchmark_drift_jacobian(device, n_particles)
    print(f"  full step (drift+jac+expm): {micro_drift['per_call_ms']:.4f}ms/call")

    # ---- Phase 4: Tracemalloc (optional) ----
    tracemalloc_result = None
    if args.trace_malloc:
        print("\n--- Phase 4: Tracemalloc allocation profile ---")
        tracemalloc_result = run_tracemalloc_snapshot(
            audit, bundle, filter_config, params, int(args.seed), device,
            max_eeg if max_eeg > 0 else min(num_eeg, 2000),
        )
        print(f"  Tracemalloc PF segment: {tracemalloc_result['wall_s']:.4f}s")
        print("  Domain totals:")
        for domain, mb in tracemalloc_result["domain_totals_mb"].items():
            print(f"    {domain}: {mb:.2f} MB")

    # ---- Phase 5: Summary & extrapolation ----
    print("\n--- Phase 5: Summary ---")
    summary = registry.summary(total_wall)
    print(f"\n{'Step':<45s} {'Wall(s)':>10s} {'%':>7s} {'Count':>8s} {'ms/call':>10s}")
    print("-" * 85)
    for row in summary:
        print(
            f"{row['step']:<45s} {row['wall_total_s']:>10.4f} {row['wall_pct']:>6.1f}% "
            f"{row['count']:>8d} {row['wall_per_call_ms']:>10.4f}"
        )

    # Extrapolation
    n_substeps_profiled = max_eeg if max_eeg > 0 else num_eeg
    n_fnirs_profiled = n_substeps_profiled // substeps
    substep_calls = n_substeps_profiled

    # Aggregate by group
    group_totals: Dict[str, float] = defaultdict(float)
    for rec in registry.records:
        group_totals[rec["group"]] += rec["wall_s"]

    print(f"\n  Group breakdown:")
    for grp in sorted(group_totals, key=lambda g: -group_totals[g]):
        pct = 100.0 * group_totals[grp] / max(total_wall, 1e-12)
        print(f"    {grp:<30s} {group_totals[grp]:>8.4f}s ({pct:>5.1f}%)")

    # Estimate per-anchor and full-dataset cost
    per_substep_ms = 1000.0 * total_wall / max(n_substeps_profiled, 1)
    per_fnirs_step_ms = 1000.0 * total_wall / max(n_fnirs_profiled, 1)
    print(f"\n  Cost per EEG substep: {per_substep_ms:.4f}ms")
    print(f"  Cost per fNIRS step:  {per_fnirs_step_ms:.4f}ms")

    # Estimate for full segment (120s)
    full_segment_substeps = int(120.0 * bundle.eeg_fs_hz)  # 24,000
    est_full_segment_s = per_substep_ms * full_segment_substeps / 1000.0
    print(f"  Estimated 120s segment: {est_full_segment_s:.1f}s ({est_full_segment_s/60:.1f}min)")

    # Estimate for 1 anchor, 1 subject, all sessions (~1h data)
    est_one_anchor_one_subject_s = per_substep_ms * int(3600.0 * bundle.eeg_fs_hz) / 1000.0
    print(f"  Estimated 1 anchor, 1h data: {est_one_anchor_one_subject_s:.1f}s ({est_one_anchor_one_subject_s/3600:.1f}h)")

    # Estimate for all anchors (36), one subject
    est_all_anchors_one_subject_s = est_one_anchor_one_subject_s * 36
    print(f"  Estimated 36 anchors, 1h data: {est_all_anchors_one_subject_s:.1f}s ({est_all_anchors_one_subject_s/3600:.1f}h)")

    # Estimate GPU transfer overhead
    np_transfer_calls = 2 * substep_calls  # proposal + process noise per substep
    if "01.substep" in group_totals:
        substep_total = group_totals["01.substep"]
        np_transfer_total = sum(
            rec["wall_s"] for rec in registry.records
            if rec["step"] in ("proposal_transfer", "process_noise_transfer_clip")
        )
        print(f"\n  numpy<->torch transfer overhead: {np_transfer_total:.4f}s "
              f"({100.0*np_transfer_total/max(total_wall,1e-12):.1f}% of PF time)")

    # ---- Write outputs ----
    profile_report = {
        "config": {
            "data_root": str(args.data_root),
            "subject_id": int(args.subject_id),
            "session_idx": int(args.session_idx),
            "anchor_fnirs_channel": str(args.anchor_fnirs_channel),
            "segment_start_s": float(args.segment_start_s),
            "segment_duration_s": float(args.segment_duration_s),
            "sigma_prop": float(args.sigma_prop),
            "sigma_nirs": float(args.sigma_nirs),
            "num_particles": int(filter_config.num_particles),
            "device": device,
            "seed": int(args.seed),
            "profile_seconds": float(args.profile_seconds),
        },
        "data_dims": {
            "num_fnirs_steps": num_fnirs,
            "num_eeg_steps": num_eeg,
            "eeg_substeps_per_fnirs": substeps,
            "eeg_fs_hz": float(bundle.eeg_fs_hz),
            "fnirs_fs_hz": float(bundle.fnirs_fs_hz),
            "num_eeg_channels": int(bundle.eeg_obs.shape[1]),
            "num_fnirs_channels": int(bundle.fnirs_primary_obs.shape[1]),
        },
        "environment": env,
        "data_loading_s": round(load_s, 4),
        "pf_total_wall_s": round(total_wall, 4),
        "pf_log_likelihood": float(pf_result["log_likelihood"]),
        "step_summary": summary,
        "group_totals_s": {grp: round(v, 6) for grp, v in sorted(group_totals.items(), key=lambda x: -x[1])},
        "cost_estimates": {
            "per_substep_ms": round(per_substep_ms, 4),
            "per_fnirs_step_ms": round(per_fnirs_step_ms, 4),
            "est_120s_segment_s": round(est_full_segment_s, 1),
            "est_120s_segment_h": round(est_full_segment_s / 3600, 2),
            "est_1h_data_s": round(est_one_anchor_one_subject_s, 1),
            "est_1h_data_h": round(est_one_anchor_one_subject_s / 3600, 2),
            "est_36anchors_1h_data_h": round(est_all_anchors_one_subject_s / 3600, 2),
        },
        "micro_benchmarks": {
            "matrix_exp": micro_matrix_exp,
            "numpy_torch_transfer": micro_transfer,
            "numpy_rng": micro_rng,
            "drift_jacobian_full": micro_drift,
        },
    }

    if tracemalloc_result is not None:
        profile_report["tracemalloc"] = tracemalloc_result

    (output_dir / "profile_report.json").write_text(
        json.dumps(profile_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Human-readable summary
    lines = [
        "# Croce PF Pipeline Profile Report",
        "",
        f"**Date**: {datetime.now().isoformat()}",
        f"**Device**: {device}",
        "",
        "## Configuration",
        "",
        f"- sigma_prop={args.sigma_prop}, sigma_nirs={args.sigma_nirs}",
        f"- N_particles={filter_config.num_particles}",
        f"- Segment: {num_fnirs} fNIRS steps x {substeps} substeps ({num_eeg} EEG steps)",
        f"- Data loading: {load_s:.4f}s",
        f"- PF total wall: {total_wall:.4f}s",
        "",
        "## Step Timing Breakdown",
        "",
        "| Step | Wall (s) | % | Count | ms/call |",
        "|------|----------|---|-------|---------|",
    ]
    for row in summary:
        lines.append(
            f"| {row['step']} | {row['wall_total_s']:.4f} | {row['wall_pct']:.1f} | "
            f"{row['count']} | {row['wall_per_call_ms']:.4f} |"
        )
    lines.extend([
        "",
        "## Group Totals",
        "",
        "| Group | Wall (s) | % |",
        "|-------|----------|---|",
    ])
    for grp, wall_s in sorted(group_totals.items(), key=lambda x: -x[1]):
        pct = 100.0 * wall_s / max(total_wall, 1e-12)
        lines.append(f"| {grp} | {wall_s:.4f} | {pct:.1f} |")
    lines.extend([
        "",
        "## Cost Estimates",
        "",
        f"- Per EEG substep: {per_substep_ms:.4f}ms",
        f"- Per fNIRS step: {per_fnirs_step_ms:.4f}ms",
        f"- Estimated 120s segment: {est_full_segment_s:.1f}s ({est_full_segment_s/60:.1f}min)",
        f"- Estimated 1 anchor, 1h data: {est_one_anchor_one_subject_s:.1f}s ({est_one_anchor_one_subject_s/3600:.1f}h)",
        f"- Estimated 36 anchors, 1h data: **{est_all_anchors_one_subject_s/3600:.1f}h**",
        "",
        "## Micro-benchmarks",
        "",
        f"- matrix_exp (N,6,6): {micro_matrix_exp.get('per_call_ms', 'N/A')}ms/call",
        f"- numpy->torch transfer: {micro_transfer.get('per_call_ms', 'N/A')}ms/call",
        f"- numpy proposal RNG: {micro_rng.get('proposal_rng_per_call_ms', 'N/A')}ms/call",
        f"- numpy process RNG: {micro_rng.get('process_rng_per_call_ms', 'N/A')}ms/call",
        f"- full drift+jac+expm: {micro_drift.get('per_call_ms', 'N/A')}ms/call",
    ])
    (output_dir / "profile_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nReport saved to {output_dir}")
    print(f"  {output_dir / 'profile_report.json'}")
    print(f"  {output_dir / 'profile_summary.md'}")


if __name__ == "__main__":
    main()
