"""Generate EEG/fNIRS source/observation target cache for one subject.

Runs the Croce SMC particle filter on all fNIRS anchors for a single subject,
computes source targets (physiological signal) and observation targets (residual),
and saves them as a single .npz cache file.

Supports --threads N to control torch intra-op parallelism. The script runs
anchors sequentially when --threads is set (for fair timing comparison across
thread counts), or in parallel when --parallel-workers > 1.

Usage:
    # Baseline: default 52-thread torch, sequential anchors
    python croce_validation/scripts/generate_target_cache.py \
        --subject-id 1 --sigma-prop 5.0 --sigma-nirs 0.15 \
        --segment-duration-s 120.0 --threads 52 --output-dir /tmp/cache_test

    # Optimized: single-threaded torch
    python croce_validation/scripts/generate_target_cache.py \
        --subject-id 1 --sigma-prop 5.0 --sigma-nirs 0.15 \
        --segment-duration-s 120.0 --threads 1 --output-dir /tmp/cache_test

    # Full parallel: 1 thread per worker, fork pool
    python croce_validation/scripts/generate_target_cache.py \
        --subject-id 1 --sigma-prop 5.0 --sigma-nirs 0.15 \
        --segment-duration-s 120.0 --threads 2 --parallel-workers 18
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_SCRIPT = PROJECT_ROOT / "croce_validation" / "scripts" / "run_local_neighborhood_solver_audit.py"
AUDIT_MODULE_NAME = "croce_cache_gen"

try:
    import torch
except ImportError:
    torch = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate EEG/fNIRS source/observation target cache for one subject."
    )
    parser.add_argument("--data-root", default="data/EEG+NIRS Single-Trial")
    parser.add_argument("--subject-id", type=int, default=1)
    parser.add_argument("--session-idx", type=int, default=0)
    parser.add_argument("--segment-start-s", type=float, default=60.0)
    parser.add_argument("--segment-duration-s", type=float, default=120.0)
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

    parser.add_argument("--threads", type=int, default=2,
                        help="OMP/MKL threads per worker (1-2 recommended for 6x6 matrix)")
    parser.add_argument("--parallel-workers", type=int, default=1,
                        help="Number of parallel anchor workers (1=sequential)")
    parser.add_argument("--anchor-list", default="",
                        help="Comma-separated anchor base names (default: all 36)")
    parser.add_argument("--max-anchors", type=int, default=0,
                        help="Cap number of anchors processed (0=all)")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--torch-device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
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
        output_dir = PROJECT_ROOT / "croce_validation" / "cache" / f"subject_cache_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def configure_torch_threads(n_threads: int) -> None:
    """Set thread counts BEFORE any torch parallel work."""
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
    if torch is not None:
        torch.set_num_threads(n_threads)
        try:
            torch.set_num_interop_threads(n_threads)
        except RuntimeError:
            pass  # may have been set already by prior torch op


# ---------------------------------------------------------------------------
# Single-anchor worker (runs in subprocess for parallel mode)
# ---------------------------------------------------------------------------

# Module-level cache for the audit module (loaded once per forked worker)
_worker_audit: Any = None


def _init_worker(n_threads: int) -> None:
    """Called once per forked worker to set thread count and load audit module."""
    global _worker_audit
    configure_torch_threads(n_threads)
    _worker_audit = load_audit_module()


def _process_anchor(payload: Tuple[str, argparse.Namespace, Any, Any, int]) -> Dict[str, Any]:
    """Run PF for one anchor and return source/observation targets.

    Designed to be called via multiprocessing. Each call gets a fresh RNG seed.
    """
    anchor_name, ds_args, spatial_cfg, filter_cfg_template, n_threads = payload
    global _worker_audit
    if _worker_audit is None:
        configure_torch_threads(n_threads)
        _worker_audit = load_audit_module()

    audit = _worker_audit
    ds_args.anchor_fnirs_channel = anchor_name

    t0 = time.perf_counter()
    bundle = audit.load_dataset_bundle(ds_args, spatial_cfg)
    load_s = time.perf_counter() - t0

    num_fnirs = int(bundle.time_s.shape[0])
    num_eeg = int(bundle.eeg_time_s.shape[0])

    # Build filter config with correct rates
    fc = audit.FilterConfig(
        integration_dt_s=float(1.0 / bundle.eeg_fs_hz),
        observation_fs_hz=float(bundle.fnirs_fs_hz),
        num_particles=int(filter_cfg_template.num_particles),
        resample_fraction=float(filter_cfg_template.resample_fraction),
        prior_std=np.asarray(filter_cfg_template.prior_std, dtype=np.float64),
        state_noise_std=np.asarray(filter_cfg_template.state_noise_std, dtype=np.float64),
        sigma_prop=float(filter_cfg_template.sigma_prop),
        sigma_nirs=float(filter_cfg_template.sigma_nirs),
        seed_list=(int(filter_cfg_template.seed_list[0]),),
        time_shift_null_s=float(filter_cfg_template.time_shift_null_s),
        run_spatial_null=False,
        solver_backend="torch_exact",
        torch_device=str(filter_cfg_template.torch_device),
    )
    fc.prior_std[4] = 0.0
    fc.state_noise_std[4] = 0.0

    params = audit.ModelParams()
    seed = int(filter_cfg_template.seed_list[0])

    t0 = time.perf_counter()
    pf_result = audit.run_particle_filter(bundle, fc, params, seed=seed)
    pf_s = time.perf_counter() - t0

    # Compute source/observation targets in raw space
    t0 = time.perf_counter()
    cache_entry = _build_cache_entry(bundle, pf_result, audit)
    post_s = time.perf_counter() - t0

    return {
        "anchor": anchor_name,
        "load_s": round(load_s, 4),
        "pf_s": round(pf_s, 4),
        "post_s": round(post_s, 4),
        "total_s": round(load_s + pf_s + post_s, 4),
        "num_fnirs_steps": num_fnirs,
        "num_eeg_steps": num_eeg,
        "n_eeg_channels": int(bundle.eeg_obs.shape[1]),
        "n_fnirs_channels": int(bundle.fnirs_primary_obs.shape[1]),
        "log_likelihood": float(pf_result["log_likelihood"]),
        "cache_entry": cache_entry,
    }


def _build_cache_entry(bundle: Any, pf_result: Dict[str, Any], audit: Any) -> Dict[str, np.ndarray]:
    """Compute source/observation targets from PF result in raw measurement space."""
    r_estimates_eeg = np.asarray(pf_result["r_estimates_eeg"], dtype=np.float64)
    estimates = np.asarray(pf_result["state_estimates"], dtype=np.float64)

    # Source targets in raw units
    if bundle.normalization.get("mode") == "per_channel_zscore_after_local_selection":
        pred_eeg_norm, pred_primary_norm, pred_secondary_norm = audit.predict_observations(
            np.column_stack([
                np.zeros_like(r_estimates_eeg),
                np.zeros_like(r_estimates_eeg),
                np.zeros_like(r_estimates_eeg),
                np.zeros_like(r_estimates_eeg),
                r_estimates_eeg,
            ]),
            bundle.lead_field, bundle.jac_primary, bundle.jac_secondary, bundle.pair_mode,
        )
        _, pred_primary_norm, pred_secondary_norm = audit.predict_observations(
            estimates,
            bundle.lead_field, bundle.jac_primary, bundle.jac_secondary, bundle.pair_mode,
        )
        pred_eeg_raw = audit.destandardize_matrix(pred_eeg_norm, bundle.normalization["eeg"])
        pred_primary_raw = audit.destandardize_matrix(pred_primary_norm, bundle.normalization["fnirs_primary"])
        pred_secondary_raw = audit.destandardize_matrix(pred_secondary_norm, bundle.normalization["fnirs_secondary"])
    else:
        pred_eeg_raw, pred_primary_raw, pred_secondary_raw = audit.predict_observations(
            np.column_stack([
                np.zeros_like(r_estimates_eeg),
                np.zeros_like(r_estimates_eeg),
                np.zeros_like(r_estimates_eeg),
                np.zeros_like(r_estimates_eeg),
                r_estimates_eeg,
            ]),
            bundle.lead_field, bundle.jac_primary, bundle.jac_secondary, bundle.pair_mode,
        )
        _, pred_primary_raw, pred_secondary_raw = audit.predict_observations(
            estimates,
            bundle.lead_field, bundle.jac_primary, bundle.jac_secondary, bundle.pair_mode,
        )

    # Observation targets = raw - source
    eeg_raw = np.asarray(bundle.eeg_obs_raw, dtype=np.float64)
    fnirs_primary_raw = np.asarray(bundle.fnirs_primary_obs_raw, dtype=np.float64)
    fnirs_secondary_raw = np.asarray(bundle.fnirs_secondary_obs_raw, dtype=np.float64)

    obs_eeg = eeg_raw - pred_eeg_raw
    obs_primary = fnirs_primary_raw - pred_primary_raw
    obs_secondary = fnirs_secondary_raw - pred_secondary_raw

    return {
        "source_eeg": pred_eeg_raw.astype(np.float32),
        "source_fnirs_primary": pred_primary_raw.astype(np.float32),
        "source_fnirs_secondary": pred_secondary_raw.astype(np.float32),
        "obs_eeg": obs_eeg.astype(np.float32),
        "obs_fnirs_primary": obs_primary.astype(np.float32),
        "obs_fnirs_secondary": obs_secondary.astype(np.float32),
        "r_estimates_eeg": r_estimates_eeg.astype(np.float32),
        "state_estimates": estimates.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Sequential runner (for fair single-thread timing)
# ---------------------------------------------------------------------------

def run_sequential(
    audit: Any,
    anchor_names: List[str],
    ds_args: argparse.Namespace,
    spatial_cfg: Any,
    filter_cfg_template: Any,
    n_threads: int,
) -> List[Dict[str, Any]]:
    """Process anchors one at a time in the current process."""
    configure_torch_threads(n_threads)
    results: List[Dict[str, Any]] = []

    for i, anchor in enumerate(anchor_names):
        print(f"  [{i+1}/{len(anchor_names)}] {anchor} ...", end=" ", flush=True)
        payload = (anchor, ds_args, spatial_cfg, filter_cfg_template, n_threads)
        result = _process_anchor(payload)
        results.append(result)
        print(
            f"load={result['load_s']:.1f}s pf={result['pf_s']:.1f}s "
            f"post={result['post_s']:.2f}s total={result['total_s']:.1f}s "
            f"LL={result['log_likelihood']:.1f}"
        )
    return results


# ---------------------------------------------------------------------------
# Parallel runner (fork pool)
# ---------------------------------------------------------------------------

def run_parallel(
    anchor_names: List[str],
    ds_args: argparse.Namespace,
    spatial_cfg: Any,
    filter_cfg_template: Any,
    n_workers: int,
    n_threads: int,
) -> List[Dict[str, Any]]:
    """Process anchors in parallel using a fork-based multiprocessing pool."""
    payloads = [(name, ds_args, spatial_cfg, filter_cfg_template, n_threads)
                for name in anchor_names]
    effective_workers = min(n_workers, len(anchor_names))

    ctx = mp.get_context("fork")
    t0 = time.perf_counter()
    with ctx.Pool(processes=effective_workers) as pool:
        all_results = pool.map(_process_anchor, payloads)
    wall_s = time.perf_counter() - t0

    total_anchor_s = sum(r["total_s"] for r in all_results)
    print(f"  Parallel wall: {wall_s:.1f}s (sum of anchor times: {total_anchor_s:.1f}s, "
          f"speedup: {total_anchor_s / max(wall_s, 1e-12):.2f}x)")
    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    audit = load_audit_module()

    # Resolve anchor list
    dataset_args_no_anchor = argparse.Namespace(
        data_root=args.data_root, subject_id=int(args.subject_id),
        session_idx=int(args.session_idx),
        segment_start_s=float(args.segment_start_s),
        segment_duration_s=float(args.segment_duration_s),
        anchor_fnirs_channel="", use_artifact_eeg=bool(args.use_artifact_eeg),
        eeg_unit="uV", fnirs_primary_unit="a.u.", fnirs_secondary_unit="a.u.",
    )
    spatial_config = audit.SpatialConfig(
        eeg_neighbors=int(args.eeg_neighbors), fnirs_neighbors=int(args.fnirs_neighbors),
        eeg_radius_mm=float(args.eeg_radius_mm), fnirs_radius_mm=float(args.fnirs_radius_mm),
        eeg_sigma_mm=float(args.eeg_sigma_mm), fnirs_sigma_mm=float(args.fnirs_sigma_mm),
        eeg_sign_mode=str(args.eeg_sign_mode),
    )

    # Create a temporary dataset to get channel names, then build adjacency
    from src.data.channel_adjacency import build_channel_adjacency
    from src.data.eeg_fnirs_dataset import MultiModalEEGfNIRSDataset
    tmp_dataset = MultiModalEEGfNIRSDataset(
        data_root=args.data_root, subject_ids=[int(args.subject_id)],
        task="motor_imagery", window_duration_s=2.5,
        normalize=False, normalization_mode="none",
        eeg_preprocessing={"bandpass": [0.5, 45.0]},
        fnirs_preprocessing={"lowpass": 0.2},
        use_artifact_data=bool(args.use_artifact_eeg),
        exclude_eog=True, hbo_only=False, hbr_only=False,
    )
    adjacency = build_channel_adjacency(
        "eeg_fnirs_single_trial", args.data_root,
        tmp_dataset.get_eeg_channel_names(),
        tmp_dataset.get_fnirs_channel_names(),
        reference_subject_id=int(args.subject_id),
        use_artifact_data=bool(args.use_artifact_eeg),
    )
    paired_bases, _, _ = audit.build_fnirs_pair_maps(adjacency.fnirs_channel_names)

    if args.anchor_list:
        requested = [n.strip() for n in args.anchor_list.split(",") if n.strip()]
        anchor_names = [n for n in requested if any(
            audit.canonicalize_channel_label(n) == audit.canonicalize_channel_label(b)
            for b in paired_bases
        )]
    else:
        anchor_names = list(paired_bases)

    if args.max_anchors > 0:
        anchor_names = anchor_names[:args.max_anchors]

    output_dir = resolve_output_dir(str(args.output_dir))

    # Build filter config template
    filter_cfg_template = audit.FilterConfig(
        integration_dt_s=0.005, observation_fs_hz=10.0,
        num_particles=int(args.num_particles),
        resample_fraction=float(args.resample_fraction),
        prior_std=audit.parse_vector(args.prior_std, name="prior-std"),
        state_noise_std=audit.parse_vector(args.state_noise_std, name="state-noise-std"),
        sigma_prop=float(args.sigma_prop), sigma_nirs=float(args.sigma_nirs),
        seed_list=(int(args.seed),), time_shift_null_s=8.0,
        run_spatial_null=False,
        solver_backend="torch_exact", torch_device=str(args.torch_device),
    )
    filter_cfg_template.prior_std[4] = 0.0
    filter_cfg_template.state_noise_std[4] = 0.0

    # ---- Header ----
    print("=" * 72)
    print("Target Cache Generator")
    print("=" * 72)
    print(f"Subject: {args.subject_id}, Session: {args.session_idx}")
    print(f"Segment: {args.segment_start_s}s + {args.segment_duration_s}s")
    print(f"Config: sp={args.sigma_prop}, sn={args.sigma_nirs}, N={args.num_particles}")
    print(f"Anchors: {len(anchor_names)} ({anchor_names[0]} ... {anchor_names[-1]})")
    print(f"Threads/worker: {args.threads}, Parallel workers: {args.parallel_workers}")
    print(f"Device: {args.torch_device}")
    print(f"Output: {output_dir}")
    print()

    # ---- Run ----
    t_total_start = time.perf_counter()

    if args.parallel_workers > 1:
        results = run_parallel(
            anchor_names, dataset_args_no_anchor, spatial_config,
            filter_cfg_template, args.parallel_workers, args.threads,
        )
    else:
        results = run_sequential(
            audit, anchor_names, dataset_args_no_anchor, spatial_config,
            filter_cfg_template, args.threads,
        )

    total_wall = time.perf_counter() - t_total_start

    # ---- Aggregate statistics ----
    total_pf_s = sum(r["pf_s"] for r in results)
    total_load_s = sum(r["load_s"] for r in results)
    total_eeg_steps = sum(r["num_eeg_steps"] for r in results)
    total_fnirs_steps = sum(r["num_fnirs_steps"] for r in results)
    avg_pf_per_anchor = total_pf_s / max(len(results), 1)
    avg_total_per_anchor = sum(r["total_s"] for r in results) / max(len(results), 1)
    per_substep_ms = 1000.0 * total_pf_s / max(total_eeg_steps, 1)

    # ---- Save cache ----
    print(f"\nSaving cache to {output_dir} ...", end=" ", flush=True)
    cache: Dict[str, Any] = {}
    for r in results:
        entry = r.pop("cache_entry")
        anchor_key = r["anchor"].replace(" ", "_").replace("-", "_")
        for field_name, array in entry.items():
            cache[f"{anchor_key}/{field_name}"] = array

    cache_path = output_dir / f"subject{args.subject_id}_cache.npz"
    np.savez_compressed(cache_path, **cache)
    cache_size_mb = cache_path.stat().st_size / (1024 * 1024)

    # ---- Save manifest ----
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "subject_id": int(args.subject_id),
            "session_idx": int(args.session_idx),
            "segment_start_s": float(args.segment_start_s),
            "segment_duration_s": float(args.segment_duration_s),
            "sigma_prop": float(args.sigma_prop),
            "sigma_nirs": float(args.sigma_nirs),
            "num_particles": int(args.num_particles),
            "threads_per_worker": int(args.threads),
            "parallel_workers": int(args.parallel_workers),
            "torch_device": str(args.torch_device),
            "seed": int(args.seed),
        },
        "anchors_processed": len(results),
        "anchor_names": [r["anchor"] for r in results],
        "timing": {
            "total_wall_s": round(total_wall, 2),
            "total_pf_s": round(total_pf_s, 2),
            "total_load_s": round(total_load_s, 2),
            "avg_pf_per_anchor_s": round(avg_pf_per_anchor, 2),
            "avg_total_per_anchor_s": round(avg_total_per_anchor, 2),
            "per_substep_ms": round(per_substep_ms, 4),
            "total_eeg_steps": total_eeg_steps,
            "total_fnirs_steps": total_fnirs_steps,
        },
        "cache_file": str(cache_path.name),
        "cache_size_mb": round(cache_size_mb, 2),
        "per_anchor_results": [
            {k: v for k, v in r.items()}
            for r in results
        ],
    }
    manifest_path = output_dir / "cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ---- Report ----
    print(f"done ({cache_size_mb:.1f} MB)")
    print(f"\n{'='*72}")
    print("TIMING SUMMARY")
    print(f"{'='*72}")
    print(f"  Anchors processed:    {len(results)}")
    print(f"  Total wall time:      {total_wall:.1f}s ({total_wall/60:.1f}min)")
    print(f"  Total PF time:        {total_pf_s:.1f}s ({total_pf_s/60:.1f}min)")
    print(f"  Total data loading:   {total_load_s:.1f}s")
    print(f"  Avg PF per anchor:    {avg_pf_per_anchor:.1f}s")
    print(f"  Avg total per anchor: {avg_total_per_anchor:.1f}s")
    print(f"  Per EEG substep:      {per_substep_ms:.4f}ms")
    print(f"  Total EEG substeps:   {total_eeg_steps}")
    print(f"  Effective throughput: {total_eeg_steps / max(total_wall, 1e-12):.0f} substeps/s")

    # Extrapolation
    if len(results) > 0:
        est_full_36 = (avg_total_per_anchor * 36)
        est_6_sessions = est_full_36 * 6
        est_29_subjects = est_6_sessions * 29
        print(f"\n  Estimated full subject (36 anchors): {est_full_36:.0f}s = {est_full_36/60:.1f}min = {est_full_36/3600:.1f}h")
        print(f"  Estimated 6 sessions:                {est_6_sessions:.0f}s = {est_6_sessions/3600:.1f}h")
        print(f"  Estimated 29 subjects:               {est_29_subjects:.0f}s = {est_29_subjects/3600:.1f}h = {est_29_subjects/86400:.1f}days")
        if args.parallel_workers > 1:
            parallel_ratio = sum(r["total_s"] for r in results) / max(total_wall, 1e-12)
            est_29_parallel = est_29_subjects / parallel_ratio
            print(f"  With ~{parallel_ratio:.1f}x parallelism:            {est_29_parallel/3600:.1f}h = {est_29_parallel/86400:.1f}days")

    print(f"\nCache saved to: {cache_path}")
    print(f"Manifest:       {manifest_path}")


if __name__ == "__main__":
    main()
