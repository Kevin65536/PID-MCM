"""Visual comparison of numerical integration methods vs matrix exponential.

Runs the particle filter with all 4 state-propagation kernels (expm, euler,
heun, rk4) using the same RNG seed, then plots time-domain overlays to visually
confirm that numerical integration produces equivalent state estimates.

Plots generated:
  1. state_timecourses_comparison.png  — 5 Croce states, all methods overlaid
  2. observation_comparison.png        — EEG/fNIRS predictions, all methods
  3. r_state_detail.png               — r(t) at EEG rate, methods overlaid
  4. state_differences.png            — method-minus-expm residual per state
  5. ess_and_weights.png              — ESS trace + final weight histogram

Output directory: croce_validation/results/numint_viz_<timestamp>/
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except ImportError:
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_SCRIPT = PROJECT_ROOT / "croce_validation" / "scripts" / "run_local_neighborhood_solver_audit.py"

# Import numerical kernels from benchmark script
BENCH_SCRIPT = PROJECT_ROOT / "croce_validation" / "scripts" / "benchmark_numerical_integration.py"


def _load_module(script_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Color / style conventions (matching existing audit plots)
# ---------------------------------------------------------------------------

METHOD_STYLES = {
    "expm": {"color": "#111111", "linewidth": 1.6, "linestyle": "-", "label": "expm (baseline)"},
    "euler": {"color": "#D62728", "linewidth": 1.2, "linestyle": "--", "label": "Euler"},
    "heun": {"color": "#1F77B4", "linewidth": 1.2, "linestyle": "--", "label": "Heun (RK2)"},
    "rk4": {"color": "#2CA02C", "linewidth": 1.2, "linestyle": "-.", "label": "RK4"},
}

DIFF_COLORS = {
    "euler": "#D62728",
    "heun": "#1F77B4",
    "rk4": "#2CA02C",
}

STATE_LABELS = ["s (vasodilatory)", "Δf (CBF)", "ΔHbO", "ΔHb", "r (neural)"]
STATE_UNITS = ["a.u.", "a.u.", "a.u.", "a.u.", "EEG μV"]


# ---------------------------------------------------------------------------
# Data loading and PF running
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visual comparison of numerical integration methods.")
    p.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "EEG+NIRS Single-Trial"))
    p.add_argument("--subject-id", type=int, default=1)
    p.add_argument("--session-idx", type=int, default=0)
    p.add_argument("--anchor-fnirs-channel", default="AF7Fp1")
    p.add_argument("--segment-start-s", type=float, default=30.0)
    p.add_argument("--segment-duration-s", type=float, default=30.0)
    p.add_argument("--num-particles", type=int, default=224)
    p.add_argument("--sigma-prop", type=float, default=6.0)
    p.add_argument("--sigma-nirs", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--torch-device", default="cpu")
    p.add_argument("--output-dir", default=str(PROJECT_ROOT / "croce_validation" / "results"))
    p.add_argument("--use-artifact-eeg", action="store_true", default=False)
    # Spatial config
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
    p.add_argument("--time-shift-null-s", type=float, default=8.0)
    p.add_argument("--seed-list", default="1000")
    return p.parse_args()


def run_all_methods(
    audit: Any,
    bench: Any,
    bundle: Any,
    filter_config: Any,
    params: Any,
    seed: int,
    torch_device: str,
) -> Dict[str, Any]:
    """Run PF with all 4 methods, return dict of results."""
    torch_config = replace(filter_config, solver_backend="torch_exact", torch_device=torch_device)
    results: Dict[str, Any] = {}

    # 1. expm baseline (unpatched)
    print("Running expm baseline...")
    t0 = time.perf_counter()
    results["expm"] = audit.run_particle_filter(bundle, torch_config, params, seed=seed)
    print(f"  expm: {time.perf_counter() - t0:.1f}s, LL={results['expm']['log_likelihood']:.2f}")

    # 2-4. Numerical methods (patched)
    original_fn = audit.torch_local_linearized_step_batch
    dt = float(filter_config.integration_dt_s)

    for name, kernel_fn in [
        ("euler", bench._torch_euler_step_batch),
        ("heun", bench._torch_heun_step_batch),
        ("rk4", bench._torch_rk4_step_batch),
    ]:
        def _make_wrapper(kfn):
            return lambda particles, _dt, _params: kfn(particles, dt, params)

        audit.torch_local_linearized_step_batch = _make_wrapper(kernel_fn)
        print(f"Running {name}...")
        t0 = time.perf_counter()
        results[name] = audit.run_particle_filter(bundle, torch_config, params, seed=seed)
        print(f"  {name}: {time.perf_counter() - t0:.1f}s, LL={results[name]['log_likelihood']:.2f}")

    audit.torch_local_linearized_step_batch = original_fn
    return results


# ---------------------------------------------------------------------------
# Plot 1: State timecourses — 5 panels, all methods overlaid
# ---------------------------------------------------------------------------

def plot_state_timecourses_comparison(
    output_path: Path,
    bundle: Any,
    results: Dict[str, Any],
) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

    time_s = bundle.time_s  # fNIRS rate

    for dim in range(5):
        ax = axes[dim]
        for method_name, style in METHOD_STYLES.items():
            estimates = results[method_name]["state_estimates"]
            ax.plot(time_s, estimates[:, dim], **style, alpha=0.9)

        ax.axhline(0.0, color="#BDBDBD", linewidth=0.8, linestyle="--")
        ax.set_ylabel(f"{STATE_LABELS[dim]}\n({STATE_UNITS[dim]})")
        ax.grid(alpha=0.25)
        if dim == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=2)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Croce State Estimates — Numerical Integration vs Matrix Exponential",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: Observation predictions — EEG + fNIRS
# ---------------------------------------------------------------------------

def plot_observation_comparison(
    output_path: Path,
    bundle: Any,
    results: Dict[str, Any],
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    # Row 0: EEG (first channel mean)
    ax = axes[0]
    eeg_obs_mean = bundle.eeg_obs.mean(axis=1)
    eeg_time = bundle.eeg_time_s
    ax.plot(eeg_time, eeg_obs_mean, color="#BDBDBD", linewidth=1.0, alpha=0.9, label="EEG observed (mean)")
    for method_name, style in METHOD_STYLES.items():
        pred_eeg_mean = results[method_name]["pred_eeg"].mean(axis=1)
        ax.plot(eeg_time, pred_eeg_mean, **style, alpha=0.85)
    ax.set_ylabel("EEG (μV)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=7, ncol=3)

    # Row 1: fNIRS primary (mean)
    ax = axes[1]
    fnirs_time = bundle.time_s
    primary_obs_mean = bundle.fnirs_primary_obs.mean(axis=1)
    ax.plot(fnirs_time, primary_obs_mean, color="#BDBDBD", linewidth=1.0, alpha=0.9, label="fNIRS primary observed (mean)")
    for method_name, style in METHOD_STYLES.items():
        pred_primary_mean = results[method_name]["pred_primary"].mean(axis=1)
        ax.plot(fnirs_time, pred_primary_mean, **style, alpha=0.85)
    ax.set_ylabel("fNIRS primary (a.u.)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=7, ncol=3)

    # Row 2: fNIRS secondary (mean)
    ax = axes[2]
    secondary_obs_mean = bundle.fnirs_secondary_obs.mean(axis=1)
    ax.plot(fnirs_time, secondary_obs_mean, color="#BDBDBD", linewidth=1.0, alpha=0.9, label="fNIRS secondary observed (mean)")
    for method_name, style in METHOD_STYLES.items():
        pred_secondary_mean = results[method_name]["pred_secondary"].mean(axis=1)
        ax.plot(fnirs_time, pred_secondary_mean, **style, alpha=0.85)
    ax.set_ylabel("fNIRS secondary (a.u.)")
    ax.set_xlabel("Time (s)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=7, ncol=3)

    fig.suptitle("Observation Predictions — Numerical Integration vs Matrix Exponential",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 3: r(t) at EEG rate — detail view
# ---------------------------------------------------------------------------

def plot_r_state_detail(
    output_path: Path,
    bundle: Any,
    results: Dict[str, Any],
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # Panel 1: r(t) at EEG rate, all methods
    ax = axes[0]
    eeg_time = bundle.eeg_time_s
    for method_name, style in METHOD_STYLES.items():
        ax.plot(eeg_time, results[method_name]["r_estimates_eeg"], **style, alpha=0.85)

    # Overlay EEG pseudoinverse projection for reference
    ax.plot(eeg_time, bundle.r_eeg_projection, color="#BDBDBD", linewidth=0.8,
            alpha=0.6, label="EEG pseudoinverse")
    ax.axhline(0.0, color="#BDBDBD", linewidth=0.8, linestyle="--")
    ax.set_ylabel("r(t) (EEG μV)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=7, ncol=3)

    # Panel 2: Zoom — first 5 seconds
    ax = axes[1]
    mask = eeg_time <= 5.0
    zoom_time = eeg_time[mask]
    for method_name, style in METHOD_STYLES.items():
        ax.plot(zoom_time, results[method_name]["r_estimates_eeg"][mask], **style, alpha=0.85)
    ax.plot(zoom_time, bundle.r_eeg_projection[mask], color="#BDBDBD", linewidth=0.8,
            alpha=0.6, label="EEG pseudoinverse")
    ax.axhline(0.0, color="#BDBDBD", linewidth=0.8, linestyle="--")
    ax.set_ylabel("r(t) — first 5s")
    ax.set_xlabel("Time (s)")
    ax.grid(alpha=0.25)

    fig.suptitle("Neural Driver r(t) at EEG Rate — Numerical Integration vs Matrix Exponential",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 4: State differences (method - expm) to visualize divergence
# ---------------------------------------------------------------------------

def plot_state_differences(
    output_path: Path,
    bundle: Any,
    results: Dict[str, Any],
) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

    expm_estimates = results["expm"]["state_estimates"]
    time_s = bundle.time_s
    num_methods = ["euler", "heun", "rk4"]

    for dim in range(5):
        ax = axes[dim]
        ax.axhline(0.0, color="#111111", linewidth=0.6, linestyle="-")
        for method_name in num_methods:
            diff = results[method_name]["state_estimates"][:, dim] - expm_estimates[:, dim]
            ax.plot(time_s, diff, color=DIFF_COLORS[method_name], linewidth=1.0,
                    alpha=0.85, label=f"{method_name} − expm")

        ax.set_ylabel(f"Δ {STATE_LABELS[dim]}")
        ax.grid(alpha=0.25)
        if dim == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("State Estimate Differences (Method − expm Baseline)",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 5: ESS trace + weight histogram
# ---------------------------------------------------------------------------

def plot_ess_and_weights(
    output_path: Path,
    bundle: Any,
    results: Dict[str, Any],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    time_s = bundle.time_s

    # ESS trace (zoom to see differences)
    ax = axes[0, 0]
    for method_name, style in METHOD_STYLES.items():
        ax.plot(time_s, results[method_name]["ess_trace"], **style, alpha=0.85)
    ax.axhline(y=results["expm"]["ess_trace"].mean(), color="#BDBDBD",
               linewidth=0.8, linestyle="--")
    ax.set_ylabel("Effective Sample Size")
    ax.set_xlabel("Time (s)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=7, ncol=2)

    # ESS histogram
    ax = axes[0, 1]
    for method_name, style in METHOD_STYLES.items():
        ax.hist(results[method_name]["ess_trace"], bins=30, alpha=0.35,
                color=style["color"], label=style["label"])
    ax.set_xlabel("ESS")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=7)

    # Log-likelihood bar chart
    ax = axes[1, 0]
    names = list(METHOD_STYLES.keys())
    lls = [results[n]["log_likelihood"] for n in names]
    colors = [METHOD_STYLES[n]["color"] for n in names]
    bars = ax.bar(names, lls, color=colors, alpha=0.75, edgecolor="#111111", linewidth=0.8)
    ax.set_ylabel("Log-Likelihood")
    ax.grid(alpha=0.25, axis="y")
    # Annotate with values
    for bar, ll in zip(bars, lls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{ll:.1f}", ha="center", fontsize=8)

    # LL delta vs expm
    ax = axes[1, 1]
    expm_ll = results["expm"]["log_likelihood"]
    deltas = [abs(results[n]["log_likelihood"] - expm_ll) for n in ["euler", "heun", "rk4"]]
    bars = ax.bar(["euler", "heun", "rk4"], deltas,
                  color=[DIFF_COLORS[n] for n in ["euler", "heun", "rk4"]],
                  alpha=0.75, edgecolor="#111111", linewidth=0.8)
    ax.set_ylabel("|LL − expm_LL|")
    ax.grid(alpha=0.25, axis="y")
    for bar, delta in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{delta:.1f}", ha="center", fontsize=8)

    fig.suptitle("Particle Filter Diagnostics — Numerical Integration vs Matrix Exponential",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if torch is None:
        raise RuntimeError("PyTorch is required for this script")

    audit = _load_module(AUDIT_SCRIPT, "viz_audit")
    bench = _load_module(BENCH_SCRIPT, "viz_bench")

    # Build context
    sys.path.insert(0, str(PROJECT_ROOT / "croce_validation" / "scripts"))
    from benchmark_solver_optimizations import build_benchmark_context
    bundle, filter_config, params, _ = build_benchmark_context(audit, args)

    print(f"Data: subject {args.subject_id}, anchor {args.anchor_fnirs_channel}")
    print(f"Segment: {args.segment_start_s}s–{args.segment_start_s + args.segment_duration_s}s")
    print(f"EEG {bundle.eeg_fs_hz}Hz, fNIRS {bundle.fnirs_fs_hz}Hz, {args.num_particles} particles")
    print(f"Seed: {args.seed}")

    # Run all methods
    results = run_all_methods(audit, bench, bundle, filter_config, params, args.seed, args.torch_device)

    # Output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"numint_viz_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Generate plots
    print("\nGenerating plots...")
    plot_state_timecourses_comparison(output_dir / "state_timecourses_comparison.png", bundle, results)
    plot_observation_comparison(output_dir / "observation_comparison.png", bundle, results)
    plot_r_state_detail(output_dir / "r_state_detail.png", bundle, results)
    plot_state_differences(output_dir / "state_differences.png", bundle, results)
    plot_ess_and_weights(output_dir / "ess_and_weights.png", bundle, results)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
