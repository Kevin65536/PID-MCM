"""Generate EEG/fNIRS source/observation target cache for one subject or one NPZ bundle.

Runs the Croce SMC particle filter on all fNIRS anchors for a single subject,
computes source targets (physiological signal) and observation targets (residual),
and saves them as a single .npz cache file.

Cache generation keeps fNIRS targets in optical measurement space. For the
EEG+NIRS Single-Trial dataset this means the cache stores paired `highWL` /
`lowWL` optical tracks rather than silently relabeling them as HbO / HbR.
Datasets that already expose oxy/deoxy concentration traces must be explicitly
projected into an optical pair before being merged into the same cache contract.

Supports --threads N to control torch intra-op parallelism. The script runs
anchors sequentially when --threads is set (for fair timing comparison across
thread counts), or in parallel when --parallel-workers > 1.

In `--mode npz`, the script consumes one pre-segmented bundle that already
contains EEG plus paired optical tracks. This is the bridge used for datasets
such as Simultaneous EEG&NIRS after explicit oxy/deoxy -> wavelength projection.

Usage:
    # Event-window exact PF with single-thread workers
    python croce_validation/scripts/generate_target_cache.py \
        --subject-id 1 --sigma-prop 5.0 --sigma-nirs 0.15 \
        --threads 1 --parallel-workers 36 --output-dir /tmp/cache_test

    # Override to a continuous debugging segment when needed
    python croce_validation/scripts/generate_target_cache.py \
        --subject-id 1 --sigma-prop 5.0 --sigma-nirs 0.15 \
        --segment-mode continuous --segment-start-s 60.0 --segment-duration-s 120.0 \
        --threads 1 --parallel-workers 36
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


FNIRS_SOURCE_CHANNEL0_FIELD = "source_fnirs_optical_channel_0"
FNIRS_SOURCE_CHANNEL1_FIELD = "source_fnirs_optical_channel_1"
FNIRS_OBS_CHANNEL0_FIELD = "obs_fnirs_optical_channel_0"
FNIRS_OBS_CHANNEL1_FIELD = "obs_fnirs_optical_channel_1"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate EEG/fNIRS source/observation target cache for one subject."
    )
    parser.add_argument("--mode", choices=("dataset", "npz"), default="dataset")
    parser.add_argument("--input-npz", default="",
                        help="Required when --mode=npz. Pre-segmented local bundle to cache.")
    parser.add_argument("--data-root", default="data/EEG+NIRS Single-Trial")
    parser.add_argument("--task", choices=("motor_imagery", "mental_arithmetic"), default="motor_imagery",
                        help="Task paradigm to load (motor_imagery or mental_arithmetic).")
    parser.add_argument("--subject-id", type=int, default=1)
    parser.add_argument("--session-idx", type=int, default=0,
                        help="Task-relative session index: 0..2 maps to MI raw sessions 0/2/4 or MA raw sessions 1/3/5.")
    parser.add_argument("--segment-mode", choices=("continuous", "event_windows"), default="event_windows",
                        help="Use one continuous segment or one event-aligned window per valid event")
    parser.add_argument("--segment-start-s", type=float, default=0.0,
                        help="Continuous segment start in seconds; ignored in event window mode")
    parser.add_argument("--segment-duration-s", type=float, default=0.0,
                        help="Continuous segment duration in seconds; <= 0 means use the full selected session")
    parser.add_argument("--event-window-pre-s", type=float, default=10.0,
                        help="Seconds kept before each event in event window mode")
    parser.add_argument("--event-window-post-s", type=float, default=40.0,
                        help="Seconds kept after each event in event window mode")
    parser.add_argument("--event-indices", default="",
                        help="Optional comma-separated event indices to keep in event window mode")
    parser.add_argument("--max-events", type=int, default=0,
                        help="Cap the number of valid events processed in event window mode (0=all valid)")
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

    parser.add_argument("--threads", type=int, default=1,
                        help="OMP/MKL threads per worker (1-2 recommended for 6x6 matrix)")
    parser.add_argument("--parallel-workers", type=int, default=36,
                        help="Number of parallel anchor workers (1=sequential, capped to anchor count)")
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


def parse_event_indices(spec: str) -> Optional[List[int]]:
    cleaned = [item.strip() for item in str(spec).split(",") if item.strip()]
    if not cleaned:
        return None
    values = sorted({int(item) for item in cleaned})
    return values


def to_jsonable_dict(values: Dict[str, Any]) -> Dict[str, Any]:
    serializable: Dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, (np.floating, np.integer)):
            serializable[key] = value.item()
        elif isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        else:
            serializable[key] = value
    return serializable


def select_event_windows(
    event_windows: List[Dict[str, Any]],
    event_indices_spec: str,
    max_events: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    requested_indices = parse_event_indices(event_indices_spec)
    windows_by_idx = {int(window["event_idx"]): window for window in event_windows}

    if requested_indices is None:
        candidate_windows = list(event_windows)
    else:
        missing = [idx for idx in requested_indices if idx not in windows_by_idx]
        if missing:
            raise ValueError(f"Requested event indices are out of range: {missing}")
        candidate_windows = [windows_by_idx[idx] for idx in requested_indices]

    valid_windows: List[Dict[str, Any]] = []
    skipped_windows: List[Dict[str, Any]] = []
    for window in candidate_windows:
        if bool(window.get("is_valid", False)):
            valid_windows.append(window)
        else:
            skipped_windows.append(window)

    if max_events > 0:
        valid_windows = valid_windows[:max_events]

    if not valid_windows:
        raise ValueError("No valid event windows were selected for cache generation")
    return valid_windows, skipped_windows


def clone_namespace(values: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**vars(values))


def build_job_label(ds_args: argparse.Namespace) -> str:
    anchor_name = str(ds_args.anchor_fnirs_channel)
    if str(getattr(ds_args, "segment_mode", "continuous")) == "event_windows":
        return f"{anchor_name}/event_{int(getattr(ds_args, 'event_idx', -1)):03d}"
    return anchor_name


def build_job_payloads(
    base_ds_args: argparse.Namespace,
    anchor_names: List[str],
    event_windows: Optional[List[Dict[str, Any]]],
    spatial_cfg: Any,
    filter_cfg_template: Any,
    n_threads: int,
) -> List[Tuple[argparse.Namespace, Any, Any, int]]:
    payloads: List[Tuple[argparse.Namespace, Any, Any, int]] = []
    for anchor_name in anchor_names:
        if event_windows is None:
            ds_args = clone_namespace(base_ds_args)
            ds_args.anchor_fnirs_channel = anchor_name
            payloads.append((ds_args, spatial_cfg, filter_cfg_template, n_threads))
            continue

        for window in event_windows:
            ds_args = clone_namespace(base_ds_args)
            ds_args.anchor_fnirs_channel = anchor_name
            ds_args.event_idx = int(window["event_idx"])
            ds_args.eeg_segment_start_s_raw = float(window["eeg_start_s"])
            ds_args.eeg_segment_end_s_raw = float(window["eeg_end_s"])
            ds_args.fnirs_segment_start_s_raw = float(window["fnirs_start_s"])
            ds_args.fnirs_segment_end_s_raw = float(window["fnirs_end_s"])
            ds_args.eeg_event_onset_s = float(window["eeg_onset_s"])
            ds_args.fnirs_event_onset_s = float(window["fnirs_onset_s"])
            ds_args.aligned_window_start_s = float(window["aligned_window_start_s"])
            ds_args.aligned_window_end_s = float(window["aligned_window_end_s"])
            ds_args.event_label_index = int(window["eeg_label"])
            ds_args.event_label_name_eeg = str(window["eeg_label_name"])
            ds_args.event_label_name_fnirs = str(window["fnirs_label_name"])
            ds_args.event_label_index_match = bool(window["label_index_match"])
            payloads.append((ds_args, spatial_cfg, filter_cfg_template, n_threads))
    return payloads


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


def _process_anchor(payload: Tuple[argparse.Namespace, Any, Any, int]) -> Dict[str, Any]:
    """Run PF for one anchor/job and return source/observation targets."""
    ds_args, spatial_cfg, filter_cfg_template, n_threads = payload
    anchor_name = str(ds_args.anchor_fnirs_channel)
    global _worker_audit
    if _worker_audit is None:
        configure_torch_threads(n_threads)
        _worker_audit = load_audit_module()

    audit = _worker_audit

    t0 = time.perf_counter()
    if str(getattr(ds_args, 'mode', 'dataset')) == 'npz':
        bundle = audit.load_real_bundle(ds_args, spatial_cfg)
    else:
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
        "n_fnirs_channels": 1,
        "eeg_fs_hz": float(bundle.eeg_fs_hz),
        "fnirs_fs_hz": float(bundle.fnirs_fs_hz),
        "pair_mode": str(bundle.pair_mode),
        "pair_labels": [str(bundle.pair_labels[0]), str(bundle.pair_labels[1])],
        "fnirs_units": {
            "primary": str(bundle.units.get("fnirs_primary", "a.u.")),
            "secondary": str(bundle.units.get("fnirs_secondary", "a.u.")),
        },
        "fnirs_signal_semantics": str(bundle.metadata.get("fnirs_signal_semantics", "unknown")),
        "bundle_task": str(bundle.metadata.get("task", "")),
        "raw_session_idx": int(bundle.metadata.get("raw_session_idx", getattr(ds_args, "session_idx", 0))),
        "bundle_segment_kind": str(bundle.metadata.get("bundle_segment_kind", "")),
        "bundle_segment_label": str(bundle.metadata.get("segment_label_name", "")),
        "segment_mode": str(bundle.metadata.get("segment_mode", getattr(ds_args, "segment_mode", "continuous"))),
        "segment_start_s": float(bundle.metadata.get("segment_start_s", 0.0)),
        "segment_duration_s": float(bundle.metadata.get("segment_duration_s", 0.0)),
        "full_session_used": bool(bundle.metadata.get("full_session_used", False)),
        "event_idx": int(bundle.metadata["event_idx"]) if bundle.metadata and bundle.metadata.get("event_idx") is not None else None,
        "event_label_index": int(bundle.metadata["event_label_index"]) if bundle.metadata and bundle.metadata.get("event_label_index") is not None else None,
        "event_label_name_eeg": str(bundle.metadata.get("event_label_name_eeg", "")) if bundle.metadata else "",
        "event_label_name_fnirs": str(bundle.metadata.get("event_label_name_fnirs", "")) if bundle.metadata else "",
        "event_window_pre_s": float(bundle.metadata.get("event_window_pre_s", 0.0)) if bundle.metadata else 0.0,
        "event_window_post_s": float(bundle.metadata.get("event_window_post_s", 0.0)) if bundle.metadata else 0.0,
        "event_window_duration_s": float(bundle.metadata.get("event_window_duration_s", 0.0)) if bundle.metadata else 0.0,
        "eeg_event_onset_s": float(bundle.metadata.get("eeg_event_onset_s", 0.0)) if bundle.metadata else 0.0,
        "fnirs_event_onset_s": float(bundle.metadata.get("fnirs_event_onset_s", 0.0)) if bundle.metadata else 0.0,
        "log_likelihood": float(pf_result["log_likelihood"]),
        "cache_entry": cache_entry,
    }


def _build_cache_entry(bundle: Any, pf_result: Dict[str, Any], audit: Any) -> Dict[str, np.ndarray]:
    """Compute anchor-scoped source/observation targets in raw measurement space.

    Each anchor contributes:
    - fNIRS: the anchor's own channel only (column 0 in each local neighbourhood),
      stored as (T_fnirs, 1).  The 36 anchors collectively cover all 36 spatial
      positions during global assembly.
    - EEG: the local EEG neighbourhood (6 channels), stored as (T_eeg, N).
      Overlaps between anchors are resolved by nearest-anchor rule at assembly
      time.
    """
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

    # EEG source: use the EEG-rate latent estimate instead of repeating the
    # fNIRS-rate state estimate. Repeating estimates[:, 4] creates 10 Hz
    # stair-steps in the 200 Hz EEG source target.
    if r_estimates_eeg.shape[0] == eeg_raw.shape[0]:
        r_eeg_eeg = r_estimates_eeg
    else:
        source_time = np.linspace(0.0, 1.0, num=r_estimates_eeg.shape[0], endpoint=False)
        target_time = np.linspace(0.0, 1.0, num=eeg_raw.shape[0], endpoint=False)
        r_eeg_eeg = np.interp(target_time, source_time, r_estimates_eeg)
    pred_eeg_norm = r_eeg_eeg.reshape(-1, 1) * lead_eeg.reshape(1, -1)

    # fNIRS source: anchor's own channel only (distance=0 → index 0).
    # State estimates → normalised prediction at anchor channel.
    _jac_p0 = jac_p.reshape(1, -1)[:, 0:1]
    _jac_s0 = jac_s.reshape(1, -1)[:, 0:1]
    pred_primary_norm = estimates[:, 2:3] * _jac_p0
    pred_secondary_norm = estimates[:, 3:4] * _jac_s0

    # Destandardize to raw measurement space
    _fnirs_p_stats0 = {"mean": [fnirs_p_stats["mean"][0]], "std": [fnirs_p_stats["std"][0]]}
    _fnirs_s_stats0 = {"mean": [fnirs_s_stats["mean"][0]], "std": [fnirs_s_stats["std"][0]]}
    pred_eeg_raw = audit.destandardize_matrix(pred_eeg_norm, eeg_stats)
    pred_primary_raw = audit.destandardize_matrix(pred_primary_norm, _fnirs_p_stats0)
    pred_secondary_raw = audit.destandardize_matrix(pred_secondary_norm, _fnirs_s_stats0)

    # Observation residuals
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
        # Assembly metadata: which channels this anchor covers
        "anchor_primary_channel": str(bundle.fnirs_primary_channel_names[0]),
        "anchor_secondary_channel": str(bundle.fnirs_secondary_channel_names[0]),
        "local_eeg_channel_names": list(bundle.eeg_channel_names),
    }


# ---------------------------------------------------------------------------
# Sequential runner (for fair single-thread timing)
# ---------------------------------------------------------------------------

def run_sequential(
    payloads: List[Tuple[argparse.Namespace, Any, Any, int]],
    n_threads: int,
) -> List[Dict[str, Any]]:
    """Process jobs one at a time in the current process."""
    configure_torch_threads(n_threads)
    results: List[Dict[str, Any]] = []

    for i, payload in enumerate(payloads):
        job_label = build_job_label(payload[0])
        print(f"  [{i+1}/{len(payloads)}] {job_label} ...", end=" ", flush=True)
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
    payloads: List[Tuple[argparse.Namespace, Any, Any, int]],
    n_workers: int,
    n_threads: int,
) -> List[Dict[str, Any]]:
    """Process jobs in parallel using a fork-based multiprocessing pool."""
    effective_workers = max(1, min(int(n_workers), len(payloads)))

    ctx = mp.get_context("fork")
    t0 = time.perf_counter()
    with ctx.Pool(processes=effective_workers, initializer=_init_worker, initargs=(n_threads,)) as pool:
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

    if str(args.mode) == 'npz':
        if not args.input_npz:
            raise ValueError('--input-npz is required when --mode=npz')
        input_npz = Path(args.input_npz)
        if not input_npz.is_absolute():
            input_npz = (PROJECT_ROOT / input_npz).resolve()
        if not input_npz.exists():
            raise FileNotFoundError(f'NPZ bundle not found: {input_npz}')
        args.input_npz = str(input_npz)

    # Resolve anchor list
    dataset_args_no_anchor = argparse.Namespace(
        mode=str(args.mode), input_npz=str(args.input_npz),
        data_root=args.data_root, task=str(args.task), subject_id=int(args.subject_id),
        session_idx=int(args.session_idx),
        segment_mode='continuous' if str(args.mode) == 'npz' else str(args.segment_mode),
        segment_start_s=float(args.segment_start_s),
        segment_duration_s=float(args.segment_duration_s),
        event_window_pre_s=float(args.event_window_pre_s),
        event_window_post_s=float(args.event_window_post_s),
        event_idx=-1,
        anchor_fnirs_channel="", use_artifact_eeg=bool(args.use_artifact_eeg),
        eeg_unit="uV", fnirs_primary_unit="V", fnirs_secondary_unit="V",
    )
    spatial_config = audit.SpatialConfig(
        eeg_neighbors=int(args.eeg_neighbors), fnirs_neighbors=int(args.fnirs_neighbors),
        eeg_radius_mm=float(args.eeg_radius_mm), fnirs_radius_mm=float(args.fnirs_radius_mm),
        eeg_sigma_mm=float(args.eeg_sigma_mm), fnirs_sigma_mm=float(args.fnirs_sigma_mm),
        eeg_sign_mode=str(args.eeg_sign_mode),
    )

    if str(args.mode) == 'npz':
        with np.load(args.input_npz, allow_pickle=False) as bundle_npz:
            if 'fnirs_channel_names' not in bundle_npz:
                raise ValueError('NPZ mode requires fnirs_channel_names for anchor selection')
            paired_bases = [
                str(name) for name in np.asarray(bundle_npz['fnirs_channel_names']).reshape(-1).tolist()
            ]
    else:
        # Create a temporary dataset to get channel names, then build adjacency
        from src.data.channel_adjacency import build_channel_adjacency
        from src.data.eeg_fnirs_dataset import MultiModalEEGfNIRSDataset
        tmp_dataset = MultiModalEEGfNIRSDataset(
            data_root=args.data_root, subject_ids=[int(args.subject_id)],
            task=args.task, window_duration_s=2.5,
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

    selected_event_windows: Optional[List[Dict[str, Any]]] = None
    skipped_event_windows: List[Dict[str, Any]] = []
    all_event_windows: List[Dict[str, Any]] = []
    if str(args.mode) == 'dataset' and str(args.segment_mode) == "event_windows":
        all_event_windows = audit.resolve_dataset_event_windows(dataset_args_no_anchor)
        selected_event_windows, skipped_event_windows = select_event_windows(
            all_event_windows,
            args.event_indices,
            int(args.max_events),
        )

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

    payloads = build_job_payloads(
        dataset_args_no_anchor,
        anchor_names,
        selected_event_windows,
        spatial_config,
        filter_cfg_template,
        int(args.threads),
    )
    effective_workers = max(1, min(int(args.parallel_workers), len(payloads)))

    output_dir = resolve_output_dir(str(args.output_dir))

    # ---- Header ----
    print("=" * 72)
    print("Target Cache Generator")
    print("=" * 72)
    print(f"Subject: {args.subject_id}, Task: {args.task}, Task session: {args.session_idx}")
    processed_segments = 1 if str(args.mode) == 'npz' else len(selected_event_windows or [])
    if str(args.mode) == 'npz':
        print("Segmentation: pre-segmented NPZ bundle")
        print(f"Input bundle: {args.input_npz}")
    elif str(args.segment_mode) == "event_windows":
        print(
            f"Segmentation: event windows ({args.event_window_pre_s:.1f}s pre + "
            f"{args.event_window_post_s:.1f}s post)"
        )
        print(
            f"Events: {len(selected_event_windows or [])} valid / {len(all_event_windows)} total"
        )
        if skipped_event_windows:
            skipped_ids = [int(window["event_idx"]) for window in skipped_event_windows]
            print(f"Skipped invalid events: {skipped_ids}")
    else:
        if float(args.segment_duration_s) <= 0.0:
            print("Segment: full selected session")
        else:
            print(f"Segment: {args.segment_start_s}s + {args.segment_duration_s}s")
    print(f"Config: sp={args.sigma_prop}, sn={args.sigma_nirs}, N={args.num_particles}")
    print(f"Anchors: {len(anchor_names)} ({anchor_names[0]} ... {anchor_names[-1]})")
    print(f"Jobs: {len(payloads)}")
    print(f"Threads/worker: {args.threads}, Parallel workers: {effective_workers}")
    if effective_workers != int(args.parallel_workers):
        print(f"Requested workers capped from {args.parallel_workers} to {effective_workers} (job count)")
    print(f"Device: {args.torch_device}, State propagation: exact matrix exponential")
    print(f"Output: {output_dir}")
    print()

    # ---- Run ----
    t_total_start = time.perf_counter()

    if args.parallel_workers > 1:
        results = run_parallel(payloads, effective_workers, int(args.threads))
    else:
        results = run_sequential(payloads, int(args.threads))

    total_wall = time.perf_counter() - t_total_start

    # ---- Aggregate statistics ----
    total_pf_s = sum(r["pf_s"] for r in results)
    total_load_s = sum(r["load_s"] for r in results)
    total_eeg_steps = sum(r["num_eeg_steps"] for r in results)
    total_fnirs_steps = sum(r["num_fnirs_steps"] for r in results)
    avg_pf_per_job = total_pf_s / max(len(results), 1)
    avg_total_per_job = sum(r["total_s"] for r in results) / max(len(results), 1)
    per_substep_ms = 1000.0 * total_pf_s / max(total_eeg_steps, 1)

    # ---- Save cache ----
    print(f"\nSaving cache to {output_dir} ...", end=" ", flush=True)
    cache: Dict[str, Any] = {}
    STRING_META_FIELDS = {"anchor_primary_channel", "anchor_secondary_channel", "local_eeg_channel_names"}
    assembly_meta: Dict[str, Dict[str, Any]] = {}
    for r in results:
        entry = r.pop("cache_entry")
        anchor_key = r["anchor"].replace(" ", "_").replace("-", "_")
        prefix = anchor_key
        if str(r.get("segment_mode", "continuous")) == "event_windows" and r.get("event_idx") is not None:
            prefix = f"{anchor_key}/event_{int(r['event_idx']):03d}"
        assm: Dict[str, Any] = {}
        for field_name, value in entry.items():
            if field_name in STRING_META_FIELDS:
                assm[field_name] = value
            else:
                cache[f"{prefix}/{field_name}"] = value
        assembly_meta[prefix] = assm

    cache_path = output_dir / f"subject{args.subject_id}_cache.npz"
    np.savez_compressed(cache_path, **cache)
    cache_size_mb = cache_path.stat().st_size / (1024 * 1024)

    # ---- Save manifest ----
    selected_event_indices = [int(window["event_idx"]) for window in (selected_event_windows or [])]
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "mode": str(args.mode),
            "input_npz": str(args.input_npz) if str(args.mode) == 'npz' else None,
            "task": str(args.task),
            "subject_id": int(args.subject_id),
            "session_idx": int(args.session_idx),
            "raw_session_idx": int(results[0].get("raw_session_idx", args.session_idx)) if results else int(args.session_idx),
            "segment_mode": str(results[0]["segment_mode"]) if results else ("npz_bundle" if str(args.mode) == 'npz' else str(args.segment_mode)),
            "segment_start_s": float(results[0]["segment_start_s"]) if results else float(args.segment_start_s),
            "segment_duration_s": float(results[0]["segment_duration_s"]) if results else float(args.segment_duration_s),
            "full_session_used": bool(results[0]["full_session_used"]) if results else bool(str(args.mode) != 'npz' and str(args.segment_mode) == "continuous" and float(args.segment_duration_s) <= 0.0),
            "event_window_pre_s": float(args.event_window_pre_s) if str(args.mode) == 'dataset' and str(args.segment_mode) == "event_windows" else None,
            "event_window_post_s": float(args.event_window_post_s) if str(args.mode) == 'dataset' and str(args.segment_mode) == "event_windows" else None,
            "event_window_duration_s": float(args.event_window_pre_s + args.event_window_post_s) if str(args.mode) == 'dataset' and str(args.segment_mode) == "event_windows" else None,
            "event_selection_policy": "single_presegmented_npz_bundle" if str(args.mode) == 'npz' else ("full_cross_modal_window_only" if str(args.segment_mode) == "event_windows" else "single_continuous_segment"),
            "event_indices_requested": parse_event_indices(args.event_indices),
            "event_indices_selected": selected_event_indices if selected_event_indices else None,
            "bundle_task": str(results[0].get("bundle_task", "")) if results else "",
            "bundle_segment_kind": str(results[0].get("bundle_segment_kind", "")) if results else "",
            "bundle_segment_label": str(results[0].get("bundle_segment_label", "")) if results else "",
            "sigma_prop": float(args.sigma_prop),
            "sigma_nirs": float(args.sigma_nirs),
            "num_particles": int(args.num_particles),
            "threads_per_worker": int(args.threads),
            "parallel_workers": int(effective_workers),
            "parallel_workers_requested": int(args.parallel_workers),
            "torch_device": str(args.torch_device),
            "solver_backend": "torch_exact",
            "state_propagation": "matrix_exponential_exact",
            "seed": int(args.seed),
            "eeg_fs_hz": float(results[0]["eeg_fs_hz"]) if results else None,
            "fnirs_fs_hz": float(results[0]["fnirs_fs_hz"]) if results else None,
            "pair_mode": str(results[0]["pair_mode"]) if results else "unknown",
            "pair_labels": list(results[0]["pair_labels"]) if results else [],
            "fnirs_target_semantics": "optical_measurement_space",
            "fnirs_target_labels": list(results[0]["pair_labels"]) if results else [],
            "fnirs_target_field_names": [
                FNIRS_SOURCE_CHANNEL0_FIELD,
                FNIRS_SOURCE_CHANNEL1_FIELD,
                FNIRS_OBS_CHANNEL0_FIELD,
                FNIRS_OBS_CHANNEL1_FIELD,
            ],
            "fnirs_units": dict(results[0]["fnirs_units"]) if results else {},
            "fnirs_signal_semantics": str(results[0]["fnirs_signal_semantics"]) if results else "unknown",
        },
        "cache_layout": {
            "result_granularity": "anchor_event_window" if str(args.mode) == 'dataset' and str(args.segment_mode) == "event_windows" else "anchor",
            "key_pattern": "<anchor>/event_<idx>/<field>" if str(args.mode) == 'dataset' and str(args.segment_mode) == "event_windows" else "<anchor>/<field>",
            "field_names": [
                "source_eeg",
                FNIRS_SOURCE_CHANNEL0_FIELD,
                FNIRS_SOURCE_CHANNEL1_FIELD,
                "obs_eeg",
                FNIRS_OBS_CHANNEL0_FIELD,
                FNIRS_OBS_CHANNEL1_FIELD,
                "r_estimates_eeg",
                "state_estimates",
            ],
        },
        "anchors_processed": len(anchor_names),
        "events_processed": processed_segments,
        "jobs_processed": len(results),
        "anchor_names": list(anchor_names),
        "assembly_meta": {k: {mk: mv for mk, mv in v.items()} for k, v in assembly_meta.items()},
        "event_windows_selected": [to_jsonable_dict(window) for window in (selected_event_windows or [])],
        "event_windows_skipped": [to_jsonable_dict(window) for window in skipped_event_windows],
        "timing": {
            "total_wall_s": round(total_wall, 2),
            "total_pf_s": round(total_pf_s, 2),
            "total_load_s": round(total_load_s, 2),
            "avg_pf_per_job_s": round(avg_pf_per_job, 2),
            "avg_total_per_job_s": round(avg_total_per_job, 2),
            "avg_pf_per_anchor_s": round(avg_pf_per_job, 2),
            "avg_total_per_anchor_s": round(avg_total_per_job, 2),
            "per_substep_ms": round(per_substep_ms, 4),
            "total_eeg_steps": total_eeg_steps,
            "total_fnirs_steps": total_fnirs_steps,
        },
        "cache_file": str(cache_path.name),
        "cache_size_mb": round(cache_size_mb, 2),
        "per_job_results": [to_jsonable_dict({k: v for k, v in r.items()}) for r in results],
    }
    manifest_path = output_dir / "cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ---- Report ----
    print(f"done ({cache_size_mb:.1f} MB)")
    print(f"\n{'='*72}")
    print("TIMING SUMMARY")
    print(f"{'='*72}")
    print(f"  Anchors processed:    {len(anchor_names)}")
    print(f"  Events processed:     {processed_segments}")
    print(f"  Jobs processed:       {len(results)}")
    print(f"  Total wall time:      {total_wall:.1f}s ({total_wall/60:.1f}min)")
    print(f"  Total PF time:        {total_pf_s:.1f}s ({total_pf_s/60:.1f}min)")
    print(f"  Total data loading:   {total_load_s:.1f}s")
    print(f"  Avg PF per job:       {avg_pf_per_job:.1f}s")
    print(f"  Avg total per job:    {avg_total_per_job:.1f}s")
    print(f"  Per EEG substep:      {per_substep_ms:.4f}ms")
    print(f"  Total EEG substeps:   {total_eeg_steps}")
    print(f"  Effective throughput: {total_eeg_steps / max(total_wall, 1e-12):.0f} substeps/s")

    # Extrapolation
    if len(results) > 0 and str(args.segment_mode) == "continuous":
        est_full_36 = (avg_total_per_job * 36)
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
