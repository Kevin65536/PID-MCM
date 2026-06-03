#!/usr/bin/env python3
"""Batch generate source/observation target cache from simultaneous optical bundles.

Processes all pre-segmented optical-proxy bundles from the Simultaneous EEG&NIRS
dataset through the Croce SMC particle filter, producing anchor-scoped
source/observation target caches suitable for downstream tokenizer training.

Each bundle (one event window for one subject/task) is processed across all 36
fNIRS anchors. The PF uses the same "EEG proposes, fNIRS selects" mechanics as
the main cache generation pipeline.

Usage:
    # Dry-run: list what would be processed
    python croce_validation/scripts/batch_generate_simultaneous_optical_cache.py \
        --dry-run

    # Process nback task, subject 1 only, 2 events max
    python croce_validation/scripts/batch_generate_simultaneous_optical_cache.py \
        --tasks nback --start-subject 1 --end-subject 1 --max-events-per-subject 2 \
        --parallel-workers 36 --threads 1

    # Full run with resume support
    python croce_validation/scripts/batch_generate_simultaneous_optical_cache.py \
        --parallel-workers 36 --threads 1 --resume
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
AUDIT_MODULE_NAME = "croce_simultaneous_batch"

BUNDLES_ROOT_DEFAULT = PROJECT_ROOT / "croce_validation" / "cache" / "simultaneous_optical_bundles_20260601_171915"
OUTPUT_ROOT_DEFAULT = PROJECT_ROOT / "croce_validation" / "cache" / "simultaneous_optical_pf_cache"

FNIRS_SOURCE_CHANNEL0_FIELD = "source_fnirs_optical_channel_0"
FNIRS_SOURCE_CHANNEL1_FIELD = "source_fnirs_optical_channel_1"
FNIRS_OBS_CHANNEL0_FIELD = "obs_fnirs_optical_channel_0"
FNIRS_OBS_CHANNEL1_FIELD = "obs_fnirs_optical_channel_1"

try:
    import torch
except ImportError:
    torch = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch generate source/observation target cache from simultaneous optical bundles."
    )
    parser.add_argument(
        "--bundles-root",
        default=str(BUNDLES_ROOT_DEFAULT),
        help="Path to the simultaneous optical bundles cache root.",
    )
    parser.add_argument(
        "--tasks",
        default="nback,wg",
        help="Comma-separated task list. Supported: nback,wg",
    )
    parser.add_argument(
        "--start-subject", type=int, default=1,
        help="First subject ID to process (1-indexed, inclusive).",
    )
    parser.add_argument(
        "--end-subject", type=int, default=26,
        help="Last subject ID to process (1-indexed, inclusive).",
    )
    parser.add_argument(
        "--max-events-per-subject", type=int, default=0,
        help="Cap the number of event windows per subject (0=all).",
    )
    parser.add_argument(
        "--event-indices", default="",
        help="Optional comma-separated event indices to keep (applies to all subjects).",
    )

    # PF configuration
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

    # Parallelism
    parser.add_argument("--threads", type=int, default=1,
                        help="OMP/MKL threads per worker.")
    parser.add_argument("--parallel-workers", type=int, default=36,
                        help="Number of parallel anchor workers per bundle.")
    parser.add_argument("--torch-device", choices=("cpu", "cuda"), default="cpu")

    # Output
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT_DEFAULT))
    parser.add_argument("--resume", action="store_true",
                        help="Skip bundles with existing complete cache files.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List bundles that would be processed and exit.")
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


def configure_torch_threads(n_threads: int) -> None:
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


def parse_event_indices(spec: str) -> Optional[List[int]]:
    cleaned = [item.strip() for item in str(spec).split(",") if item.strip()]
    if not cleaned:
        return None
    return sorted({int(item) for item in cleaned})


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def bundle_output_exists(output_dir: Path) -> bool:
    cache_file = output_dir / "cache.npz"
    manifest_file = output_dir / "cache_manifest.json"
    return cache_file.exists() and manifest_file.exists()


# ---------------------------------------------------------------------------
# Bundle discovery
# ---------------------------------------------------------------------------

def discover_bundles(
    bundles_root: Path,
    tasks: List[str],
    start_subject: int,
    end_subject: int,
    max_events_per_subject: int,
    event_indices_spec: str,
) -> List[Dict[str, Any]]:
    """Discover all bundles to process, returning a list of bundle descriptors."""
    requested_indices = parse_event_indices(event_indices_spec)
    bundles: List[Dict[str, Any]] = []

    for task in tasks:
        task_dir = bundles_root / task
        if not task_dir.is_dir():
            print(f"[WARN] Task directory not found: {task_dir}")
            continue

        for subject_id in range(start_subject, end_subject + 1):
            subject_dir = task_dir / f"subject_{subject_id:03d}"
            if not subject_dir.is_dir():
                continue

            bundles_dir = subject_dir / "bundles"
            if not bundles_dir.is_dir():
                continue

            bundle_files = sorted(bundles_dir.glob("*.npz"))
            for bundle_path in bundle_files:
                event_idx = None
                try:
                    parts = bundle_path.stem.split("_")
                    for i, part in enumerate(parts):
                        if part == "window" and i + 1 < len(parts):
                            event_idx = int(parts[i + 1])
                            break
                except (ValueError, IndexError):
                    pass

                if requested_indices is not None and event_idx is not None and event_idx not in requested_indices:
                    continue

                bundles.append({
                    "task": task,
                    "subject_id": subject_id,
                    "bundle_path": str(bundle_path),
                    "event_idx": event_idx,
                })

            if max_events_per_subject > 0:
                subject_bundles = [b for b in bundles if b["task"] == task and b["subject_id"] == subject_id]
                if len(subject_bundles) > max_events_per_subject:
                    keep = {b["bundle_path"] for b in subject_bundles[:max_events_per_subject]}
                    bundles = [b for b in bundles if not (b["task"] == task and b["subject_id"] == subject_id and b["bundle_path"] not in keep)]

    return bundles


# ---------------------------------------------------------------------------
# Single-anchor worker
# ---------------------------------------------------------------------------

_worker_audit: Any = None


def _init_worker(n_threads: int) -> None:
    global _worker_audit
    configure_torch_threads(n_threads)
    _worker_audit = load_audit_module()


def _process_anchor(payload: Tuple[str, str, int, Any, Any, Any, int]) -> Dict[str, Any]:
    """Run PF for one anchor on one bundle and return source/observation targets."""
    bundle_path, anchor_name, anchor_index, spatial_cfg, filter_cfg_template, params, n_threads = payload
    global _worker_audit
    if _worker_audit is None:
        configure_torch_threads(n_threads)
        _worker_audit = load_audit_module()

    audit = _worker_audit

    # Load the bundle for this specific anchor
    ds_args = argparse.Namespace(
        mode="npz",
        input_npz=bundle_path,
        anchor_fnirs_channel=anchor_name,
        eeg_unit="uV",
        fnirs_primary_unit="V",
        fnirs_secondary_unit="V",
    )
    t0 = time.perf_counter()
    bundle = audit.load_real_bundle(ds_args, spatial_cfg)
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

    seed = int(filter_cfg_template.seed_list[0])

    t0 = time.perf_counter()
    pf_result = audit.run_particle_filter(bundle, fc, params, seed=seed)
    pf_s = time.perf_counter() - t0

    # Build cache entry
    t0 = time.perf_counter()
    cache_entry = _build_cache_entry(bundle, pf_result, audit)
    post_s = time.perf_counter() - t0

    return {
        "anchor": anchor_name,
        "anchor_index": anchor_index,
        "load_s": round(load_s, 4),
        "pf_s": round(pf_s, 4),
        "post_s": round(post_s, 4),
        "total_s": round(load_s + pf_s + post_s, 4),
        "num_fnirs_steps": num_fnirs,
        "num_eeg_steps": num_eeg,
        "n_eeg_channels": int(bundle.eeg_obs.shape[1]),
        "n_fnirs_channels": 1,
        "eeg_fs_hz": float(bundle.eeg_fs_hz),
        "fnirs_fs_hz": float(bundle.fnirs_fs_hz),
        "pair_mode": str(bundle.pair_mode),
        "pair_labels": [str(bundle.pair_labels[0]), str(bundle.pair_labels[1])],
        "log_likelihood": float(pf_result["log_likelihood"]),
        "cache_entry": cache_entry,
    }


def _build_cache_entry(bundle: Any, pf_result: Dict[str, Any], audit: Any) -> Dict[str, np.ndarray]:
    """Compute anchor-scoped source/observation targets in raw measurement space."""
    r_estimates_eeg = np.asarray(pf_result["r_estimates_eeg"], dtype=np.float64)
    estimates = np.asarray(pf_result["state_estimates"], dtype=np.float64)

    lead_eeg = np.asarray(bundle.lead_field, dtype=np.float64)
    jac_p = np.asarray(bundle.jac_primary, dtype=np.float64)
    jac_s = np.asarray(bundle.jac_secondary, dtype=np.float64)
    eeg_stats = bundle.normalization["eeg"]
    fnirs_p_stats = bundle.normalization["fnirs_primary"]
    fnirs_s_stats = bundle.normalization["fnirs_secondary"]
    eeg_raw = np.asarray(bundle.eeg_obs_raw, dtype=np.float64)
    fnirs_p_raw = np.asarray(bundle.fnirs_primary_obs_raw, dtype=np.float64)
    fnirs_s_raw = np.asarray(bundle.fnirs_secondary_obs_raw, dtype=np.float64)

    eeg_substeps = eeg_raw.shape[0] // estimates.shape[0]

    r_eeg_fnirs = estimates[:, 4]
    if eeg_substeps > 1:
        r_eeg_eeg = np.repeat(r_eeg_fnirs, eeg_substeps)[: eeg_raw.shape[0]]
    else:
        r_eeg_eeg = r_estimates_eeg
    pred_eeg_norm = r_eeg_eeg.reshape(-1, 1) * lead_eeg.reshape(1, -1)

    _jac_p0 = jac_p.reshape(1, -1)[:, 0:1]
    _jac_s0 = jac_s.reshape(1, -1)[:, 0:1]
    pred_primary_norm = estimates[:, 2:3] * _jac_p0
    pred_secondary_norm = estimates[:, 3:4] * _jac_s0

    _fnirs_p_stats0 = {"mean": [fnirs_p_stats["mean"][0]], "std": [fnirs_p_stats["std"][0]]}
    _fnirs_s_stats0 = {"mean": [fnirs_s_stats["mean"][0]], "std": [fnirs_s_stats["std"][0]]}
    pred_eeg_raw = audit.destandardize_matrix(pred_eeg_norm, eeg_stats)
    pred_primary_raw = audit.destandardize_matrix(pred_primary_norm, _fnirs_p_stats0)
    pred_secondary_raw = audit.destandardize_matrix(pred_secondary_norm, _fnirs_s_stats0)

    obs_eeg = eeg_raw - pred_eeg_raw
    obs_primary = fnirs_p_raw[:, 0:1] - pred_primary_raw
    obs_secondary = fnirs_s_raw[:, 0:1] - pred_secondary_raw

    return {
        "source_eeg": pred_eeg_raw.astype(np.float32),
        FNIRS_SOURCE_CHANNEL0_FIELD: pred_primary_raw.astype(np.float32),
        FNIRS_SOURCE_CHANNEL1_FIELD: pred_secondary_raw.astype(np.float32),
        "obs_eeg": obs_eeg.astype(np.float32),
        FNIRS_OBS_CHANNEL0_FIELD: obs_primary.astype(np.float32),
        FNIRS_OBS_CHANNEL1_FIELD: obs_secondary.astype(np.float32),
        "r_estimates_eeg": r_estimates_eeg.astype(np.float32),
        "state_estimates": estimates.astype(np.float32),
        "anchor_primary_channel": str(bundle.fnirs_primary_channel_names[0]),
        "anchor_secondary_channel": str(bundle.fnirs_secondary_channel_names[0]),
        "local_eeg_channel_names": list(bundle.eeg_channel_names),
    }


# ---------------------------------------------------------------------------
# Bundle processing
# ---------------------------------------------------------------------------

def process_bundle(
    bundle_desc: Dict[str, Any],
    audit: Any,
    spatial_config: Any,
    filter_cfg_template: Any,
    params: Any,
    anchor_names: List[str],
    n_threads: int,
    n_workers: int,
    output_base: Path,
) -> Dict[str, Any]:
    """Process all anchors for a single bundle and save the cache."""
    task = bundle_desc["task"]
    subject_id = bundle_desc["subject_id"]
    bundle_path = bundle_desc["bundle_path"]
    event_idx = bundle_desc["event_idx"]

    event_dir_name = f"event_{event_idx:03d}" if event_idx is not None else "segment_000"
    output_dir = output_base / task / f"subject_{subject_id:03d}" / event_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build job payloads (one per anchor)
    payloads: List[Tuple[str, str, int, Any, Any, Any, int]] = []
    for anchor_idx, anchor_name in enumerate(anchor_names):
        payloads.append((bundle_path, anchor_name, anchor_idx, spatial_config, filter_cfg_template, params, n_threads))

    effective_workers = max(1, min(int(n_workers), len(payloads)))

    t0 = time.perf_counter()
    if effective_workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=effective_workers, initializer=_init_worker, initargs=(n_threads,)) as pool:
            results = pool.map(_process_anchor, payloads)
    else:
        configure_torch_threads(n_threads)
        results = []
        for i, payload in enumerate(payloads):
            results.append(_process_anchor(payload))
    wall_s = time.perf_counter() - t0

    # Assemble cache
    cache: Dict[str, Any] = {}
    STRING_META_FIELDS = {"anchor_primary_channel", "anchor_secondary_channel", "local_eeg_channel_names"}
    assembly_meta: Dict[str, Dict[str, Any]] = {}

    for r in results:
        entry = r.pop("cache_entry")
        anchor_key = r["anchor"].replace(" ", "_").replace("-", "_")
        for field_name, value in entry.items():
            if field_name in STRING_META_FIELDS:
                assembly_meta.setdefault(anchor_key, {})[field_name] = value
            else:
                cache[f"{anchor_key}/{field_name}"] = value

    cache_path = output_dir / "cache.npz"
    np.savez_compressed(cache_path, **cache)
    cache_size_mb = cache_path.stat().st_size / (1024 * 1024)

    # Save manifest
    total_pf_s = sum(r["pf_s"] for r in results)
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "task": task,
            "subject_id": subject_id,
            "event_idx": event_idx,
            "bundle_path": bundle_path,
            "sigma_prop": float(filter_cfg_template.sigma_prop),
            "sigma_nirs": float(filter_cfg_template.sigma_nirs),
            "num_particles": int(filter_cfg_template.num_particles),
            "threads_per_worker": int(n_threads),
            "parallel_workers": int(effective_workers),
            "torch_device": str(filter_cfg_template.torch_device),
            "solver_backend": "torch_exact",
            "seed": int(filter_cfg_template.seed_list[0]),
            "pair_mode": str(results[0]["pair_mode"]) if results else "unknown",
            "pair_labels": list(results[0]["pair_labels"]) if results else [],
        },
        "cache_layout": {
            "result_granularity": "anchor",
            "key_pattern": "<anchor>/<field>",
            "field_names": [
                "source_eeg", FNIRS_SOURCE_CHANNEL0_FIELD, FNIRS_SOURCE_CHANNEL1_FIELD,
                "obs_eeg", FNIRS_OBS_CHANNEL0_FIELD, FNIRS_OBS_CHANNEL1_FIELD,
                "r_estimates_eeg", "state_estimates",
            ],
        },
        "anchors_processed": len(anchor_names),
        "jobs_processed": len(results),
        "anchor_names": list(anchor_names),
        "assembly_meta": assembly_meta,
        "timing": {
            "total_wall_s": round(wall_s, 2),
            "total_pf_s": round(total_pf_s, 2),
            "avg_pf_per_job_s": round(total_pf_s / max(len(results), 1), 2),
            "avg_total_per_job_s": round(sum(r["total_s"] for r in results) / max(len(results), 1), 2),
        },
        "cache_file": str(cache_path.name),
        "cache_size_mb": round(cache_size_mb, 2),
        "per_job_results": [to_jsonable({k: v for k, v in r.items()}) for r in results],
    }
    manifest_path = output_dir / "cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "task": task,
        "subject_id": subject_id,
        "event_idx": event_idx,
        "wall_s": round(wall_s, 2),
        "total_pf_s": round(total_pf_s, 2),
        "anchors": len(results),
        "cache_size_mb": round(cache_size_mb, 2),
        "avg_pf_per_job_s": round(total_pf_s / max(len(results), 1), 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    bundles_root = Path(args.bundles_root)
    if not bundles_root.is_absolute():
        bundles_root = PROJECT_ROOT / bundles_root
    output_base = Path(args.output_dir)
    if not output_base.is_absolute():
        output_base = PROJECT_ROOT / output_base

    tasks = [t.strip().lower() for t in args.tasks.split(",") if t.strip()]
    valid_tasks = {"nback", "wg"}
    invalid = [t for t in tasks if t not in valid_tasks]
    if invalid:
        raise ValueError(f"Unsupported tasks: {invalid}")

    # Discover bundles
    bundles = discover_bundles(
        bundles_root, tasks,
        args.start_subject, args.end_subject,
        args.max_events_per_subject, args.event_indices,
    )

    print("=" * 72)
    print("SIMULTANEOUS OPTICAL BUNDLE BATCH CACHE GENERATION")
    print("=" * 72)
    print(f"Bundles root: {bundles_root}")
    print(f"Tasks: {tasks}")
    print(f"Subject range: {args.start_subject}–{args.end_subject}")
    print(f"Bundles discovered: {len(bundles)}")
    print(f"Config: sp={args.sigma_prop}, sn={args.sigma_nirs}, N={args.num_particles}")
    print(f"Workers: {args.parallel_workers}, Threads/worker: {args.threads}")
    print(f"Output: {output_base}")

    if args.dry_run:
        print(f"\n--- Dry Run: would process {len(bundles)} bundles ---")
        for b in bundles[:20]:
            print(f"  {b['task']}/subject_{b['subject_id']:03d}/event_{b['event_idx']:03d}")
        if len(bundles) > 20:
            print(f"  ... and {len(bundles) - 20} more")
        return

    # Resume check
    skipped = 0
    if args.resume:
        new_bundles = []
        for b in bundles:
            event_dir_name = f"event_{b['event_idx']:03d}" if b['event_idx'] is not None else "segment_000"
            out_dir = output_base / b["task"] / f"subject_{b['subject_id']:03d}" / event_dir_name
            if bundle_output_exists(out_dir):
                skipped += 1
            else:
                new_bundles.append(b)
        if skipped:
            print(f"Skipping {skipped} bundles with existing cache files")
        bundles = new_bundles

    if not bundles:
        print("No bundles to process.")
        return

    print(f"Bundles to process: {len(bundles)}")
    print()

    # Load the audit module once
    configure_torch_threads(int(args.threads))
    audit = load_audit_module()

    # Build shared config objects
    spatial_config = audit.SpatialConfig(
        eeg_neighbors=int(args.eeg_neighbors),
        fnirs_neighbors=int(args.fnirs_neighbors),
        eeg_radius_mm=float(args.eeg_radius_mm),
        fnirs_radius_mm=float(args.fnirs_radius_mm),
        eeg_sigma_mm=float(args.eeg_sigma_mm),
        fnirs_sigma_mm=float(args.fnirs_sigma_mm),
        eeg_sign_mode=str(args.eeg_sign_mode),
    )

    filter_cfg_template = audit.FilterConfig(
        integration_dt_s=0.005,
        observation_fs_hz=10.0,
        num_particles=int(args.num_particles),
        resample_fraction=float(args.resample_fraction),
        prior_std=audit.parse_vector(args.prior_std, name="prior-std"),
        state_noise_std=audit.parse_vector(args.state_noise_std, name="state-noise-std"),
        sigma_prop=float(args.sigma_prop),
        sigma_nirs=float(args.sigma_nirs),
        seed_list=(int(args.seed),),
        time_shift_null_s=8.0,
        run_spatial_null=False,
        solver_backend="torch_exact",
        torch_device=str(args.torch_device),
    )
    filter_cfg_template.prior_std[4] = 0.0
    filter_cfg_template.state_noise_std[4] = 0.0
    params = audit.ModelParams()

    # Load anchor names from the first bundle (all bundles share the same 36 fNIRS channels)
    first_bundle = bundles[0]
    with np.load(first_bundle["bundle_path"], allow_pickle=False) as npz:
        fnirs_channel_names = [
            str(n) for n in np.asarray(npz["fnirs_channel_names"]).reshape(-1).tolist()
        ]
    anchor_names = fnirs_channel_names  # In wavelength space, each channel is its own anchor

    print(f"fNIRS anchors: {len(anchor_names)}")
    print(f"Total jobs: {len(bundles)} bundles × {len(anchor_names)} anchors = {len(bundles) * len(anchor_names)}")
    print()

    # Process bundles sequentially
    total_start = time.perf_counter()
    all_results: List[Dict[str, Any]] = []
    success_count = 0
    fail_count = 0

    try:
        for i, bundle_desc in enumerate(bundles):
            label = (
                f"{bundle_desc['task']}/subject_{bundle_desc['subject_id']:03d}"
                f"/event_{bundle_desc['event_idx']:03d}"
            )
            print(f"[{i+1}/{len(bundles)}] {label} ...", end=" ", flush=True)

            try:
                result = process_bundle(
                    bundle_desc, audit, spatial_config, filter_cfg_template, params,
                    anchor_names, int(args.threads), int(args.parallel_workers), output_base,
                )
                all_results.append(result)
                success_count += 1
                print(
                    f"OK | {result['wall_s']:.1f}s wall, "
                    f"{result['avg_pf_per_job_s']:.1f}s/job, "
                    f"{result['cache_size_mb']:.1f}MB"
                )
            except Exception as exc:
                fail_count += 1
                all_results.append({
                    "task": bundle_desc["task"],
                    "subject_id": bundle_desc["subject_id"],
                    "event_idx": bundle_desc["event_idx"],
                    "success": False,
                    "error": str(exc),
                })
                print(f"FAIL: {exc}")
    except KeyboardInterrupt:
        print(f"\n\nInterrupted after {success_count} successes, {fail_count} failures")

    total_wall = time.perf_counter() - total_start

    # Save batch manifest
    batch_manifest = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "solver_backend": "torch_exact",
            "state_propagation": "matrix_exponential_exact",
            "parallel_workers": int(args.parallel_workers),
            "threads": int(args.threads),
            "num_particles": int(args.num_particles),
            "sigma_prop": float(args.sigma_prop),
            "sigma_nirs": float(args.sigma_nirs),
            "seed": int(args.seed),
            "tasks": tasks,
            "subject_start": int(args.start_subject),
            "subject_end": int(args.end_subject),
            "max_events_per_subject": int(args.max_events_per_subject),
            "event_indices": args.event_indices,
        },
        "summary": {
            "total_wall_s": round(total_wall, 1),
            "total_wall_h": round(total_wall / 3600, 2),
            "bundles_discovered": len(bundles) + skipped,
            "bundles_processed": len(bundles),
            "bundles_skipped": skipped,
            "success": success_count,
            "failed": fail_count,
        },
        "per_bundle": all_results,
    }
    manifest_path = output_base / "batch_manifest.json"
    manifest_path.write_text(json.dumps(batch_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n{'='*72}")
    print(f"BATCH COMPLETE")
    print(f"{'='*72}")
    print(f"Total wall time: {total_wall:.0f}s = {total_wall/60:.1f}min = {total_wall/3600:.2f}h")
    print(f"Success: {success_count}, Failed: {fail_count}, Skipped: {skipped}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
