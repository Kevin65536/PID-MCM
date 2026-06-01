#!/usr/bin/env python3
"""Prepare Simultaneous EEG&NIRS optical bundles for later cache generation.

This script focuses on the two Simultaneous tasks that are currently viable for
Croce-style cache preparation:

- nback: aligned at session scale
- wg: aligned at trial scale

For each selected subject/task pair it:

1. aligns EEG/fNIRS marker sequences with the same skip-one-extra-marker logic
   used by the repository loader,
2. summarizes the natural onset spacing at the task's effective segmentation
   scale,
3. forward-projects MATLAB oxy/deoxy concentration exports into paired 850/760
   optical proxy tracks using the current Croce design coefficients,
4. writes one full-task projected fNIRS file plus one cache-ready NPZ bundle per
    aligned event/session, depending on the selected bundle mode.

The output bundles remain explicit about their provenance: they are wavelength-
aligned optical proxies derived from oxy/deoxy, not restored vendor raw files.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.eeg_fnirs_dataset import get_eeg_channel_mask, load_mat_struct  # noqa: E402
from src.data.simultaneous_eeg_nirs_dataset import (  # noqa: E402
    SimultaneousCognitiveLoader,
    _select_best_skip_alignment,
    classify_alignment_pattern,
    detect_offset_blocks,
    resolve_marker_event_label_names,
    resolve_segmentation_mode,
)


SUPPORTED_TASKS = ("nback", "wg")
DEFAULT_HEAD_RADIUS_MM = 95.0
DEFAULT_PROJECTION_MATRIX = np.asarray(
    [
        [1.00, 0.25],  # 850 nm / highWL
        [0.35, 1.00],  # 760 nm / lowWL
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Simultaneous EEG&NIRS optical proxy bundles for cache generation.",
    )
    parser.add_argument(
        "--data-root",
        default="data/Simultaneous EEG&NIRS",
        help="Path to the Simultaneous EEG&NIRS dataset root.",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(SUPPORTED_TASKS),
        help="Comma-separated task list. Supported: nback,wg",
    )
    parser.add_argument(
        "--subject-ids",
        default="all",
        help="Comma-separated subject ids, or 'all' to process every discovered subject.",
    )
    parser.add_argument(
        "--head-radius-mm",
        type=float,
        default=DEFAULT_HEAD_RADIUS_MM,
        help="Pseudo-mm head radius used to map task montage x/y coordinates onto a cache-friendly scale.",
    )
    parser.add_argument(
        "--projection-kappa",
        type=float,
        default=1.0,
        help="Global scale factor applied after the HbO/HbR -> optical forward projection.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional explicit output directory. Defaults to croce_validation/cache/simultaneous_optical_bundles_<timestamp>/.",
    )
    parser.add_argument(
        "--bundle-mode",
        choices=("event_windows", "natural_segments"),
        default="event_windows",
        help="Bundle extraction strategy. event_windows exports fixed -pre/+post windows around each aligned event; natural_segments exports session/trial spans bounded by the next aligned onset.",
    )
    parser.add_argument(
        "--event-window-pre-s",
        type=float,
        default=10.0,
        help="Seconds included before each aligned event onset when bundle_mode=event_windows.",
    )
    parser.add_argument(
        "--event-window-post-s",
        type=float,
        default=40.0,
        help="Seconds included after each aligned event onset when bundle_mode=event_windows.",
    )
    return parser.parse_args()


def parse_tasks(spec: str) -> List[str]:
    values = [item.strip().lower() for item in str(spec).split(",") if item.strip()]
    if not values:
        return list(SUPPORTED_TASKS)
    invalid = [value for value in values if value not in SUPPORTED_TASKS]
    if invalid:
        raise ValueError(f"Unsupported tasks: {invalid}")
    return values


def discover_subject_ids(data_root: Path) -> List[int]:
    eeg_subjects = {
        int(match.group(1))
        for path in data_root.glob("VP*-EEG")
        for match in [re.fullmatch(r"VP(\d+)-EEG", path.name)]
        if match is not None
    }
    fnirs_subjects = {
        int(match.group(1))
        for path in data_root.glob("VP*-NIRS")
        for match in [re.fullmatch(r"VP(\d+)-NIRS", path.name)]
        if match is not None
    }
    return sorted(eeg_subjects & fnirs_subjects)


def parse_subject_ids(spec: str, discovered_subject_ids: Sequence[int]) -> List[int]:
    if str(spec).strip().lower() in {"", "all", "*"}:
        return list(discovered_subject_ids)
    requested = sorted({int(item.strip()) for item in str(spec).split(",") if item.strip()})
    missing = [subject_id for subject_id in requested if subject_id not in discovered_subject_ids]
    if missing:
        raise ValueError(f"Requested subject ids are not available in the dataset root: {missing}")
    return requested


def resolve_output_dir(spec: str) -> Path:
    if spec:
        output_dir = Path(spec)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "croce_validation" / "cache" / f"simultaneous_optical_bundles_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip()).strip("_") or "segment"


def load_cnt_struct(data_root: Path, subject_id: int, modality: str, task: str) -> Any:
    suffix = "EEG" if modality == "eeg" else "NIRS"
    path = data_root / f"VP{subject_id:03d}-{suffix}" / f"cnt_{task}.mat"
    mat = load_mat_struct(str(path))
    key = next(name for name in mat if not name.startswith("__"))
    return mat[key]


def load_mnt_struct(data_root: Path, subject_id: int, modality: str, task: str) -> Any:
    suffix = "EEG" if modality == "eeg" else "NIRS"
    path = data_root / f"VP{subject_id:03d}-{suffix}" / f"mnt_{task}.mat"
    mat = load_mat_struct(str(path))
    key = next(name for name in mat if not name.startswith("__"))
    return mat[key]


def mnt_positions_to_mm(
    mnt_struct: Any,
    head_radius_mm: float,
    *,
    channel_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = np.asarray(getattr(mnt_struct, "x"), dtype=np.float64).reshape(-1)
    y = np.asarray(getattr(mnt_struct, "y"), dtype=np.float64).reshape(-1)
    positions = np.column_stack([x, y])
    if channel_mask is not None:
        positions = positions[np.asarray(channel_mask, dtype=bool)]
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"Unexpected montage position shape: {positions.shape}")
    if not np.isfinite(positions).all():
        raise ValueError("Montage positions contain non-finite values")

    max_abs = float(np.max(np.abs(positions))) if positions.size else 0.0
    scale_applied = float(head_radius_mm) if max_abs <= 2.5 else 1.0
    scaled = positions * scale_applied
    radii = np.linalg.norm(scaled, axis=1)
    return scaled.astype(np.float32), {
        "input_max_abs_xy": max_abs,
        "scale_applied": scale_applied,
        "radius_mean": float(np.mean(radii)) if radii.size else None,
        "radius_max": float(np.max(radii)) if radii.size else None,
    }


def align_marker_sequences(
    eeg_markers: Mapping[str, Any],
    fnirs_markers: Mapping[str, Any],
    *,
    jump_threshold_ms: float = 20_000.0,
) -> Dict[str, Any]:
    eeg_times = np.asarray(eeg_markers.get("time", []), dtype=np.float64)
    fnirs_times = np.asarray(fnirs_markers.get("time", []), dtype=np.float64)
    eeg_labels = resolve_marker_event_label_names(dict(eeg_markers))
    fnirs_labels = resolve_marker_event_label_names(dict(fnirs_markers))

    skipped = {"eeg_indices": [], "fnirs_indices": []}
    if len(eeg_times) == len(fnirs_times):
        aligned_eeg_times = eeg_times
        aligned_fnirs_times = fnirs_times
        aligned_eeg_labels = eeg_labels
        aligned_fnirs_labels = fnirs_labels
    elif len(eeg_times) == len(fnirs_times) + 1:
        skip_index, _ = _select_best_skip_alignment(eeg_times, fnirs_times)
        aligned_eeg_times = np.delete(eeg_times, skip_index)
        aligned_fnirs_times = fnirs_times
        aligned_eeg_labels = [label for index, label in enumerate(eeg_labels) if index != skip_index]
        aligned_fnirs_labels = fnirs_labels
        skipped["eeg_indices"] = [int(skip_index)]
    elif len(fnirs_times) == len(eeg_times) + 1:
        skip_index, _ = _select_best_skip_alignment(fnirs_times, eeg_times)
        aligned_eeg_times = eeg_times
        aligned_fnirs_times = np.delete(fnirs_times, skip_index)
        aligned_eeg_labels = eeg_labels
        aligned_fnirs_labels = [label for index, label in enumerate(fnirs_labels) if index != skip_index]
        skipped["fnirs_indices"] = [int(skip_index)]
    else:
        common = min(len(eeg_times), len(fnirs_times))
        aligned_eeg_times = eeg_times[:common]
        aligned_fnirs_times = fnirs_times[:common]
        aligned_eeg_labels = eeg_labels[:common]
        aligned_fnirs_labels = fnirs_labels[:common]

    residual_ms = aligned_fnirs_times - aligned_eeg_times
    offset_blocks = detect_offset_blocks(residual_ms, jump_threshold_ms=jump_threshold_ms)
    offset_pattern = classify_alignment_pattern(
        residual_ms,
        offset_blocks,
        skipped_marker_indices=skipped,
    )
    paired_onsets_ms = 0.5 * (aligned_eeg_times + aligned_fnirs_times)

    return {
        "aligned_eeg_times_ms": aligned_eeg_times,
        "aligned_fnirs_times_ms": aligned_fnirs_times,
        "aligned_eeg_labels": aligned_eeg_labels,
        "aligned_fnirs_labels": aligned_fnirs_labels,
        "paired_onsets_ms": paired_onsets_ms,
        "skipped_marker_indices": skipped,
        "label_sequence_match": aligned_eeg_labels == aligned_fnirs_labels,
        "residual_series_ms": residual_ms,
        "residual_mean_ms": float(np.mean(residual_ms)) if residual_ms.size else None,
        "residual_std_ms": float(np.std(residual_ms)) if residual_ms.size else None,
        "offset_blocks": offset_blocks,
        "offset_pattern": offset_pattern,
    }


def summarize_intervals(interval_ms: np.ndarray) -> Dict[str, Any]:
    interval_ms = np.asarray(interval_ms, dtype=np.float64)
    if interval_ms.size == 0:
        return {
            "count": 0,
            "median_s": None,
            "mean_s": None,
            "min_s": None,
            "max_s": None,
            "p10_s": None,
            "p90_s": None,
            "scale_label": "single_segment_only",
        }

    interval_s = interval_ms / 1000.0
    median_s = float(np.median(interval_s))
    if median_s < 1.0:
        scale_label = "subsecond"
    elif median_s < 10.0:
        scale_label = "few_seconds"
    elif median_s < 120.0:
        scale_label = "tens_of_seconds"
    else:
        scale_label = "minutes"

    return {
        "count": int(interval_s.size),
        "median_s": median_s,
        "mean_s": float(np.mean(interval_s)),
        "min_s": float(np.min(interval_s)),
        "max_s": float(np.max(interval_s)),
        "p10_s": float(np.quantile(interval_s, 0.10)),
        "p90_s": float(np.quantile(interval_s, 0.90)),
        "scale_label": scale_label,
    }


def forward_project_optical(
    oxy: np.ndarray,
    deoxy: np.ndarray,
    *,
    projection_kappa: float,
    projection_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    stacked = np.stack([np.asarray(oxy, dtype=np.float64), np.asarray(deoxy, dtype=np.float64)], axis=-1)
    projected = float(projection_kappa) * np.einsum("...c,wc->...w", stacked, projection_matrix, optimize=True)
    high_wl = np.asarray(projected[..., 0], dtype=np.float32)
    low_wl = np.asarray(projected[..., 1], dtype=np.float32)
    return high_wl, low_wl


def build_segment_specs(
    alignment: Mapping[str, Any],
    *,
    segmentation_mode: str,
    eeg_num_samples: int,
    fnirs_num_samples: int,
    eeg_fs_hz: float,
    fnirs_fs_hz: float,
) -> List[Dict[str, Any]]:
    eeg_times = np.asarray(alignment["aligned_eeg_times_ms"], dtype=np.float64)
    fnirs_times = np.asarray(alignment["aligned_fnirs_times_ms"], dtype=np.float64)
    labels = list(alignment["aligned_eeg_labels"])
    segment_kind = "session" if segmentation_mode == "session" else "trial"
    eeg_total_duration_ms = 1000.0 * float(eeg_num_samples) / float(eeg_fs_hz)
    fnirs_total_duration_ms = 1000.0 * float(fnirs_num_samples) / float(fnirs_fs_hz)

    segments: List[Dict[str, Any]] = []
    for index in range(len(eeg_times)):
        eeg_start_ms = float(eeg_times[index])
        fnirs_start_ms = float(fnirs_times[index])
        if index + 1 < len(eeg_times):
            eeg_end_ms = float(eeg_times[index + 1])
            fnirs_end_ms = float(fnirs_times[index + 1])
        else:
            eeg_end_ms = eeg_total_duration_ms
            fnirs_end_ms = fnirs_total_duration_ms

        eeg_start_sample = max(int(round(eeg_start_ms * eeg_fs_hz / 1000.0)), 0)
        eeg_end_sample = min(int(round(eeg_end_ms * eeg_fs_hz / 1000.0)), eeg_num_samples)
        fnirs_start_sample = max(int(round(fnirs_start_ms * fnirs_fs_hz / 1000.0)), 0)
        fnirs_end_sample = min(int(round(fnirs_end_ms * fnirs_fs_hz / 1000.0)), fnirs_num_samples)

        if eeg_end_sample - eeg_start_sample < 32:
            continue
        if fnirs_end_sample - fnirs_start_sample < 16:
            continue

        label_name = str(labels[index]) if index < len(labels) else f"{segment_kind}_{index}"
        segments.append(
            {
                "segment_index": int(index),
                "segment_kind": segment_kind,
                "label_name": label_name,
                "eeg_start_ms": eeg_start_ms,
                "eeg_end_ms": eeg_end_ms,
                "fnirs_start_ms": fnirs_start_ms,
                "fnirs_end_ms": fnirs_end_ms,
                "paired_duration_s": 0.5 * ((eeg_end_ms - eeg_start_ms) / 1000.0 + (fnirs_end_ms - fnirs_start_ms) / 1000.0),
                "onset_residual_ms": float(fnirs_start_ms - eeg_start_ms),
                "eeg_start_sample": eeg_start_sample,
                "eeg_end_sample": eeg_end_sample,
                "fnirs_start_sample": fnirs_start_sample,
                "fnirs_end_sample": fnirs_end_sample,
            }
        )
    return segments


def compute_fixed_window_bounds(
    *,
    onset_ms: float,
    pre_s: float,
    post_s: float,
    fs_hz: float,
    num_samples: int,
) -> Tuple[int, int]:
    nominal_length = max(int(round((float(pre_s) + float(post_s)) * float(fs_hz))), 1)
    event_sample = int(round(float(onset_ms) * float(fs_hz) / 1000.0))
    pre_samples = int(round(float(pre_s) * float(fs_hz)))
    start_sample = event_sample - pre_samples
    end_sample = start_sample + nominal_length

    if start_sample < 0:
        end_sample = min(num_samples, end_sample - start_sample)
        start_sample = 0
    if end_sample > num_samples:
        start_sample = max(0, start_sample - (end_sample - num_samples))
        end_sample = num_samples

    if end_sample - start_sample < nominal_length and num_samples >= nominal_length:
        if start_sample == 0:
            end_sample = nominal_length
        elif end_sample == num_samples:
            start_sample = num_samples - nominal_length

    return int(start_sample), int(end_sample)


def build_event_window_specs(
    alignment: Mapping[str, Any],
    *,
    eeg_num_samples: int,
    fnirs_num_samples: int,
    eeg_fs_hz: float,
    fnirs_fs_hz: float,
    event_window_pre_s: float,
    event_window_post_s: float,
) -> List[Dict[str, Any]]:
    eeg_times = np.asarray(alignment["aligned_eeg_times_ms"], dtype=np.float64)
    fnirs_times = np.asarray(alignment["aligned_fnirs_times_ms"], dtype=np.float64)
    labels = list(alignment["aligned_eeg_labels"])

    segments: List[Dict[str, Any]] = []
    for index, (eeg_onset_ms, fnirs_onset_ms) in enumerate(zip(eeg_times.tolist(), fnirs_times.tolist())):
        eeg_start_sample, eeg_end_sample = compute_fixed_window_bounds(
            onset_ms=float(eeg_onset_ms),
            pre_s=event_window_pre_s,
            post_s=event_window_post_s,
            fs_hz=eeg_fs_hz,
            num_samples=eeg_num_samples,
        )
        fnirs_start_sample, fnirs_end_sample = compute_fixed_window_bounds(
            onset_ms=float(fnirs_onset_ms),
            pre_s=event_window_pre_s,
            post_s=event_window_post_s,
            fs_hz=fnirs_fs_hz,
            num_samples=fnirs_num_samples,
        )

        eeg_start_ms = 1000.0 * float(eeg_start_sample) / float(eeg_fs_hz)
        eeg_end_ms = 1000.0 * float(eeg_end_sample) / float(eeg_fs_hz)
        fnirs_start_ms = 1000.0 * float(fnirs_start_sample) / float(fnirs_fs_hz)
        fnirs_end_ms = 1000.0 * float(fnirs_end_sample) / float(fnirs_fs_hz)
        label_name = str(labels[index]) if index < len(labels) else f"event_{index}"

        segments.append(
            {
                "segment_index": int(index),
                "event_idx": int(index),
                "segment_kind": "event_window",
                "label_name": label_name,
                "eeg_event_onset_ms": float(eeg_onset_ms),
                "fnirs_event_onset_ms": float(fnirs_onset_ms),
                "eeg_start_ms": eeg_start_ms,
                "eeg_end_ms": eeg_end_ms,
                "fnirs_start_ms": fnirs_start_ms,
                "fnirs_end_ms": fnirs_end_ms,
                "paired_duration_s": 0.5 * ((eeg_end_ms - eeg_start_ms) / 1000.0 + (fnirs_end_ms - fnirs_start_ms) / 1000.0),
                "onset_residual_ms": float(fnirs_onset_ms - eeg_onset_ms),
                "eeg_start_sample": int(eeg_start_sample),
                "eeg_end_sample": int(eeg_end_sample),
                "fnirs_start_sample": int(fnirs_start_sample),
                "fnirs_end_sample": int(fnirs_end_sample),
                "event_window_pre_s": float(event_window_pre_s),
                "event_window_post_s": float(event_window_post_s),
                "aligned_window_start_s": float(eeg_start_ms - float(eeg_onset_ms)) / 1000.0,
                "aligned_window_end_s": float(eeg_end_ms - float(eeg_onset_ms)) / 1000.0,
            }
        )
    return segments


def projected_signal_stats(values: np.ndarray) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "shape": list(array.shape),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p01": float(np.quantile(array, 0.01)),
        "p99": float(np.quantile(array, 0.99)),
        "finite": bool(np.isfinite(array).all()),
    }


def save_segment_bundle(
    output_path: Path,
    *,
    subject_id: int,
    task: str,
    segment: Mapping[str, Any],
    eeg_segment: np.ndarray,
    fnirs_850_segment: np.ndarray,
    fnirs_760_segment: np.ndarray,
    eeg_fs_hz: float,
    fnirs_fs_hz: float,
    eeg_positions_mm: np.ndarray,
    fnirs_positions_mm: np.ndarray,
    eeg_channel_names: Sequence[str],
    fnirs_channel_names: Sequence[str],
    projection_kappa: float,
    projection_matrix: np.ndarray,
    input_fnirs_unit: str,
) -> None:
    bundle_fields: Dict[str, Any] = {
        "eeg": np.asarray(eeg_segment, dtype=np.float32),
        "eeg_fs_hz": np.asarray(float(eeg_fs_hz), dtype=np.float64),
        "eeg_positions_mm": np.asarray(eeg_positions_mm, dtype=np.float32),
        "eeg_channel_names": np.asarray(list(eeg_channel_names), dtype=str),
        "fnirs_850": np.asarray(fnirs_850_segment, dtype=np.float32),
        "fnirs_760": np.asarray(fnirs_760_segment, dtype=np.float32),
        "fnirs_fs_hz": np.asarray(float(fnirs_fs_hz), dtype=np.float64),
        "fnirs_positions_mm": np.asarray(fnirs_positions_mm, dtype=np.float32),
        "fnirs_channel_names": np.asarray(list(fnirs_channel_names), dtype=str),
        "pair_labels": np.asarray(["highWL", "lowWL"], dtype=str),
        "wavelengths_nm": np.asarray([850, 760], dtype=np.int32),
        "subject_id": np.asarray(int(subject_id), dtype=np.int32),
        "task": np.asarray(str(task), dtype=str),
        "segment_kind": np.asarray(str(segment["segment_kind"]), dtype=str),
        "segment_index": np.asarray(int(segment["segment_index"]), dtype=np.int32),
        "label_name": np.asarray(str(segment["label_name"]), dtype=str),
        "eeg_start_ms": np.asarray(float(segment["eeg_start_ms"]), dtype=np.float64),
        "eeg_end_ms": np.asarray(float(segment["eeg_end_ms"]), dtype=np.float64),
        "fnirs_start_ms": np.asarray(float(segment["fnirs_start_ms"]), dtype=np.float64),
        "fnirs_end_ms": np.asarray(float(segment["fnirs_end_ms"]), dtype=np.float64),
        "onset_residual_ms": np.asarray(float(segment["onset_residual_ms"]), dtype=np.float64),
        "projection_kappa": np.asarray(float(projection_kappa), dtype=np.float64),
        "projection_matrix": np.asarray(projection_matrix, dtype=np.float64),
        "source_fnirs_unit": np.asarray(str(input_fnirs_unit), dtype=str),
        "projected_pair_unit": np.asarray("projected_optical_proxy_from_" + str(input_fnirs_unit), dtype=str),
        "optical_projection_kind": np.asarray("forward_projected_wavelength_proxy", dtype=str),
    }
    for optional_key in (
        "event_idx",
        "eeg_event_onset_ms",
        "fnirs_event_onset_ms",
        "event_window_pre_s",
        "event_window_post_s",
        "aligned_window_start_s",
        "aligned_window_end_s",
    ):
        if optional_key in segment:
            value = segment[optional_key]
            dtype = np.int32 if optional_key == "event_idx" else np.float64
            bundle_fields[optional_key] = np.asarray(value, dtype=dtype)
    np.savez_compressed(output_path, **bundle_fields)


def render_summary_markdown(report: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Simultaneous Optical Bundle Preparation")
    lines.append("")
    lines.append(f"- Generated at: {report['generated_at']}")
    lines.append(f"- Data root: {report['data_root']}")
    lines.append(f"- Subjects processed: {len(report['subject_ids'])}")
    lines.append(f"- Projection wavelengths: 850/760 nm")
    lines.append(f"- Projection kappa: {report['projection']['kappa']}")
    lines.append(f"- Bundle mode: {report['bundle_config']['mode']}")
    if report['bundle_config']['mode'] == 'event_windows':
        lines.append(
            f"- Event window: {report['bundle_config']['event_window_pre_s']} s pre + {report['bundle_config']['event_window_post_s']} s post"
        )
    lines.append("")
    lines.append("## Time Scale Summary")
    lines.append("")
    for task_summary in report["task_summary"]:
        interval = task_summary["pooled_interval_summary"]
        lines.append(f"### {task_summary['task']}")
        lines.append("")
        lines.append(f"- Segmentation mode: {task_summary['segmentation_mode']}")
        lines.append(f"- Subjects exported: {task_summary['subjects_exported']}")
        lines.append(f"- Bundles exported: {task_summary['segments_exported']}")
        lines.append(
            f"- Natural onset spacing: median={interval['median_s']} s, p10={interval['p10_s']} s, p90={interval['p90_s']} s, scale={interval['scale_label']}"
        )
        lines.append(f"- Offset pattern cases: {task_summary['offset_pattern_counts']}")
        lines.append("")

    lines.append("## Output Layout")
    lines.append("")
    lines.append("- Each task directory contains per-subject full projection files plus one NPZ bundle per aligned segment/event.")
    lines.append("- Bundle NPZ fields are explicit 850/760 wavelength channels, with EEG, channel names, positions, and segment timing metadata included.")
    lines.append("- These bundles are wavelength-aligned optical proxies derived from oxy/deoxy, not restored vendor raw optical files.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (PROJECT_ROOT / data_root).resolve()
    discovered_subject_ids = discover_subject_ids(data_root)
    subject_ids = parse_subject_ids(args.subject_ids, discovered_subject_ids)
    tasks = parse_tasks(args.tasks)
    output_dir = resolve_output_dir(args.output_dir)

    projection_matrix = np.asarray(DEFAULT_PROJECTION_MATRIX, dtype=np.float64)
    global_report: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(data_root),
        "subject_ids": subject_ids,
        "tasks": tasks,
        "bundle_config": {
            "mode": str(args.bundle_mode),
            "event_window_pre_s": float(args.event_window_pre_s),
            "event_window_post_s": float(args.event_window_post_s),
        },
        "projection": {
            "wavelengths_nm": [850, 760],
            "pair_labels": ["highWL", "lowWL"],
            "kappa": float(args.projection_kappa),
            "matrix_rows_hwl_lwl": projection_matrix.tolist(),
            "source_note": "Projected from MATLAB oxy/deoxy exports because raw wl1/wl2 files are unavailable in the checked local copy.",
        },
        "task_summary": [],
        "subject_task_manifests": [],
    }

    bundle_mode = str(args.bundle_mode).strip().lower()

    for task in tasks:
        segmentation_mode = resolve_segmentation_mode(task, "both", "auto")
        task_dir = output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        pooled_intervals_ms: List[float] = []
        offset_pattern_counts: MutableMapping[str, int] = {}
        subjects_exported = 0
        segments_exported = 0

        for subject_id in subject_ids:
            loader = SimultaneousCognitiveLoader(
                str(data_root),
                task=task,
                subject_ids=[subject_id],
                modality="both",
                allow_deprecated=True,
            )
            eeg_cnt, eeg_markers_full, eeg_info = loader.load_subject_data(subject_id, "eeg")
            fnirs_oxy_cnt, fnirs_markers, fnirs_oxy_info = SimultaneousCognitiveLoader(
                str(data_root),
                task=task,
                subject_ids=[subject_id],
                modality="fnirs",
                fnirs_signal="oxy",
                allow_deprecated=True,
            ).load_subject_data(subject_id, "fnirs")
            fnirs_deoxy_cnt, _, fnirs_deoxy_info = SimultaneousCognitiveLoader(
                str(data_root),
                task=task,
                subject_ids=[subject_id],
                modality="fnirs",
                fnirs_signal="deoxy",
                allow_deprecated=True,
            ).load_subject_data(subject_id, "fnirs")

            if segmentation_mode == "session":
                eeg_markers = loader.get_session_markers(subject_id, "eeg")
                fnirs_markers = loader.get_session_markers(subject_id, "fnirs")
            else:
                eeg_markers = eeg_markers_full

            eeg_mnt = load_mnt_struct(data_root, subject_id, "eeg", task)
            fnirs_mnt = load_mnt_struct(data_root, subject_id, "fnirs", task)
            eeg_channel_mask = get_eeg_channel_mask(list(eeg_info["clab"]), exclude_eog=True)
            eeg_cnt = np.asarray(eeg_cnt, dtype=np.float32)[:, eeg_channel_mask]
            eeg_channel_names = [name for index, name in enumerate(eeg_info["clab"]) if eeg_channel_mask[index]]
            eeg_positions_mm, eeg_layout_info = mnt_positions_to_mm(
                eeg_mnt,
                args.head_radius_mm,
                channel_mask=eeg_channel_mask,
            )
            fnirs_positions_mm, fnirs_layout_info = mnt_positions_to_mm(fnirs_mnt, args.head_radius_mm)
            eeg_info = dict(eeg_info)
            eeg_info["clab"] = eeg_channel_names

            alignment = align_marker_sequences(eeg_markers, fnirs_markers)
            interval_ms = np.diff(np.asarray(alignment["paired_onsets_ms"], dtype=np.float64))
            pooled_intervals_ms.extend(interval_ms.tolist())
            pattern_case = str(alignment["offset_pattern"]["case"])
            offset_pattern_counts[pattern_case] = offset_pattern_counts.get(pattern_case, 0) + 1

            fnirs_850_full, fnirs_760_full = forward_project_optical(
                fnirs_oxy_cnt,
                fnirs_deoxy_cnt,
                projection_kappa=float(args.projection_kappa),
                projection_matrix=projection_matrix,
            )

            if bundle_mode == "event_windows":
                segments = build_event_window_specs(
                    alignment,
                    eeg_num_samples=int(np.asarray(eeg_cnt).shape[0]),
                    fnirs_num_samples=int(np.asarray(fnirs_oxy_cnt).shape[0]),
                    eeg_fs_hz=float(eeg_info["fs"]),
                    fnirs_fs_hz=float(fnirs_oxy_info["fs"]),
                    event_window_pre_s=float(args.event_window_pre_s),
                    event_window_post_s=float(args.event_window_post_s),
                )
            else:
                segments = build_segment_specs(
                    alignment,
                    segmentation_mode=segmentation_mode,
                    eeg_num_samples=int(np.asarray(eeg_cnt).shape[0]),
                    fnirs_num_samples=int(np.asarray(fnirs_oxy_cnt).shape[0]),
                    eeg_fs_hz=float(eeg_info["fs"]),
                    fnirs_fs_hz=float(fnirs_oxy_info["fs"]),
                )

            subject_dir = task_dir / f"subject_{subject_id:03d}"
            bundles_dir = subject_dir / "bundles"
            subject_dir.mkdir(parents=True, exist_ok=True)
            bundles_dir.mkdir(parents=True, exist_ok=True)

            full_projection_path = subject_dir / f"subject_{subject_id:03d}_{task}_full_projection.npz"
            np.savez_compressed(
                full_projection_path,
                fnirs_850=np.asarray(fnirs_850_full, dtype=np.float32),
                fnirs_760=np.asarray(fnirs_760_full, dtype=np.float32),
                fnirs_positions_mm=np.asarray(fnirs_positions_mm, dtype=np.float32),
                fnirs_channel_names=np.asarray(list(fnirs_oxy_info["clab"]), dtype=str),
                fnirs_fs_hz=np.asarray(float(fnirs_oxy_info["fs"]), dtype=np.float64),
                pair_labels=np.asarray(["highWL", "lowWL"], dtype=str),
                wavelengths_nm=np.asarray([850, 760], dtype=np.int32),
                projection_kappa=np.asarray(float(args.projection_kappa), dtype=np.float64),
                projection_matrix=np.asarray(projection_matrix, dtype=np.float64),
                source_fnirs_unit=np.asarray(str(fnirs_oxy_info.get("title", "mmol/L")), dtype=str),
                optical_projection_kind=np.asarray("forward_projected_wavelength_proxy", dtype=str),
                aligned_eeg_marker_times_ms=np.asarray(alignment["aligned_eeg_times_ms"], dtype=np.float64),
                aligned_fnirs_marker_times_ms=np.asarray(alignment["aligned_fnirs_times_ms"], dtype=np.float64),
                aligned_label_names=np.asarray(list(alignment["aligned_eeg_labels"]), dtype=str),
                paired_onset_times_ms=np.asarray(alignment["paired_onsets_ms"], dtype=np.float64),
            )

            segment_records: List[Dict[str, Any]] = []
            for segment in segments:
                eeg_segment = np.asarray(
                    eeg_cnt[segment["eeg_start_sample"]: segment["eeg_end_sample"], :],
                    dtype=np.float32,
                )
                fnirs_850_segment = np.asarray(
                    fnirs_850_full[segment["fnirs_start_sample"]: segment["fnirs_end_sample"], :],
                    dtype=np.float32,
                )
                fnirs_760_segment = np.asarray(
                    fnirs_760_full[segment["fnirs_start_sample"]: segment["fnirs_end_sample"], :],
                    dtype=np.float32,
                )

                bundle_name = (
                    f"subject_{subject_id:03d}_{task}_{segment['segment_kind']}_{int(segment['segment_index']):03d}_"
                    f"{safe_slug(segment['label_name'])}.npz"
                )
                bundle_path = bundles_dir / bundle_name
                save_segment_bundle(
                    bundle_path,
                    subject_id=subject_id,
                    task=task,
                    segment=segment,
                    eeg_segment=eeg_segment,
                    fnirs_850_segment=fnirs_850_segment,
                    fnirs_760_segment=fnirs_760_segment,
                    eeg_fs_hz=float(eeg_info["fs"]),
                    fnirs_fs_hz=float(fnirs_oxy_info["fs"]),
                    eeg_positions_mm=eeg_positions_mm,
                    fnirs_positions_mm=fnirs_positions_mm,
                    eeg_channel_names=list(eeg_info["clab"]),
                    fnirs_channel_names=list(fnirs_oxy_info["clab"]),
                    projection_kappa=float(args.projection_kappa),
                    projection_matrix=projection_matrix,
                    input_fnirs_unit=str(getattr(load_cnt_struct(data_root, subject_id, "fnirs", task).oxy, "yUnit", "mmol/L")),
                )
                segment_record = dict(segment)
                segment_record["bundle_path"] = str(bundle_path.relative_to(output_dir))
                segment_records.append(segment_record)

            subject_manifest = {
                "subject_id": int(subject_id),
                "task": str(task),
                "segmentation_mode": str(segmentation_mode),
                "bundle_mode": bundle_mode,
                "event_window_pre_s": float(args.event_window_pre_s) if bundle_mode == "event_windows" else None,
                "event_window_post_s": float(args.event_window_post_s) if bundle_mode == "event_windows" else None,
                "eeg_sample_rate_hz": float(eeg_info["fs"]),
                "fnirs_sample_rate_hz": float(fnirs_oxy_info["fs"]),
                "num_eeg_channels": int(np.asarray(eeg_cnt).shape[1]),
                "num_fnirs_channels": int(np.asarray(fnirs_oxy_cnt).shape[1]),
                "projection": {
                    "wavelengths_nm": [850, 760],
                    "pair_labels": ["highWL", "lowWL"],
                    "kappa": float(args.projection_kappa),
                    "matrix_rows_hwl_lwl": projection_matrix.tolist(),
                    "source_unit": str(getattr(load_cnt_struct(data_root, subject_id, "fnirs", task).oxy, "yUnit", "mmol/L")),
                    "projected_pair_unit": f"projected_optical_proxy_from_{getattr(load_cnt_struct(data_root, subject_id, 'fnirs', task).oxy, 'yUnit', 'mmol/L')}",
                },
                "alignment_summary": {
                    "num_aligned_pairs": int(len(alignment["paired_onsets_ms"])),
                    "label_sequence_match": bool(alignment["label_sequence_match"]),
                    "residual_mean_ms": alignment["residual_mean_ms"],
                    "residual_std_ms": alignment["residual_std_ms"],
                    "offset_pattern": to_builtin(alignment["offset_pattern"]),
                    "skipped_marker_indices": to_builtin(alignment["skipped_marker_indices"]),
                },
                "interval_summary": summarize_intervals(interval_ms),
                "layout_info": {
                    "eeg": eeg_layout_info,
                    "fnirs": fnirs_layout_info,
                },
                "projected_signal_stats": {
                    "fnirs_850": projected_signal_stats(fnirs_850_full),
                    "fnirs_760": projected_signal_stats(fnirs_760_full),
                },
                "full_projection_file": str(full_projection_path.relative_to(output_dir)),
                "segments": segment_records,
            }
            manifest_path = subject_dir / "manifest.json"
            manifest_path.write_text(json.dumps(to_builtin(subject_manifest), indent=2), encoding="utf-8")

            global_report["subject_task_manifests"].append(str(manifest_path.relative_to(output_dir)))
            subjects_exported += 1
            segments_exported += len(segment_records)

        global_report["task_summary"].append(
            {
                "task": str(task),
                "segmentation_mode": str(segmentation_mode),
                "bundle_mode": bundle_mode,
                "subjects_exported": int(subjects_exported),
                "segments_exported": int(segments_exported),
                "pooled_interval_summary": summarize_intervals(np.asarray(pooled_intervals_ms, dtype=np.float64)),
                "offset_pattern_counts": dict(sorted(offset_pattern_counts.items())),
            }
        )

    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"
    summary_json_path.write_text(json.dumps(to_builtin(global_report), indent=2), encoding="utf-8")
    summary_md_path.write_text(render_summary_markdown(global_report), encoding="utf-8")

    print(f"[SimultaneousOpticalPrep] Saved summary: {summary_json_path}")
    print(f"[SimultaneousOpticalPrep] Saved markdown: {summary_md_path}")
    print(json.dumps(to_builtin(global_report["task_summary"]), indent=2))


if __name__ == "__main__":
    main()


def load_mnt_struct(data_root: Path, subject_id: int, modality: str, task: str) -> Any:
    return load_struct(data_root, subject_id, modality, "mnt", task)


def coerce_name_list(values: Any) -> List[str]:
    return [str(item) for item in np.asarray(values).tolist()]


def coerce_positions_2d_mm(mnt_struct: Any, head_radius_mm: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    x_coords = np.asarray(mnt_struct.x, dtype=np.float64).reshape(-1)
    y_coords = np.asarray(mnt_struct.y, dtype=np.float64).reshape(-1)
    positions = np.column_stack([x_coords, y_coords])
    if not np.all(np.isfinite(positions)):
        raise ValueError("Montage x/y coordinates contain non-finite values")

    scale_applied = False
    max_abs = float(np.max(np.abs(positions))) if positions.size else 0.0
    if max_abs <= 2.5:
        positions = positions * float(head_radius_mm)
        scale_applied = True

    return positions.astype(np.float32), {
        "coordinate_source": "mnt.xy",
        "scale_applied_from_unit_sphere": bool(scale_applied),
        "head_radius_mm": float(head_radius_mm) if scale_applied else None,
        "max_abs_before_scaling": max_abs,
    }


def align_marker_streams(
    eeg_markers: Mapping[str, Any],
    fnirs_markers: Mapping[str, Any],
) -> Dict[str, Any]:
    eeg_times = np.asarray(eeg_markers["time"], dtype=np.float64)
    fnirs_times = np.asarray(fnirs_markers["time"], dtype=np.float64)
    eeg_labels = resolve_marker_event_label_names(dict(eeg_markers))
    fnirs_labels = resolve_marker_event_label_names(dict(fnirs_markers))

    skipped = {"eeg_indices": [], "fnirs_indices": []}
    if len(eeg_times) == len(fnirs_times):
        aligned_eeg_times = eeg_times
        aligned_fnirs_times = fnirs_times
        aligned_eeg_labels = eeg_labels
        aligned_fnirs_labels = fnirs_labels
    elif len(eeg_times) == len(fnirs_times) + 1:
        skip_index, _ = _select_best_skip_alignment(eeg_times, fnirs_times)
        aligned_eeg_times = np.delete(eeg_times, skip_index)
        aligned_fnirs_times = fnirs_times
        aligned_eeg_labels = [label for index, label in enumerate(eeg_labels) if index != skip_index]
        aligned_fnirs_labels = fnirs_labels
        skipped["eeg_indices"] = [int(skip_index)]
    elif len(fnirs_times) == len(eeg_times) + 1:
        skip_index, _ = _select_best_skip_alignment(fnirs_times, eeg_times)
        aligned_eeg_times = eeg_times
        aligned_fnirs_times = np.delete(fnirs_times, skip_index)
        aligned_eeg_labels = eeg_labels
        aligned_fnirs_labels = [label for index, label in enumerate(fnirs_labels) if index != skip_index]
        skipped["fnirs_indices"] = [int(skip_index)]
    else:
        common = min(len(eeg_times), len(fnirs_times))
        aligned_eeg_times = eeg_times[:common]
        aligned_fnirs_times = fnirs_times[:common]
        aligned_eeg_labels = eeg_labels[:common]
        aligned_fnirs_labels = fnirs_labels[:common]

    residual_ms = aligned_fnirs_times - aligned_eeg_times
    blocks = detect_offset_blocks(residual_ms)
    return {
        "num_eeg_markers": int(len(eeg_times)),
        "num_fnirs_markers": int(len(fnirs_times)),
        "num_aligned_pairs": int(len(residual_ms)),
        "skipped_marker_indices": skipped,
        "label_sequence_match": aligned_eeg_labels == aligned_fnirs_labels,
        "eeg_times_ms": aligned_eeg_times,
        "fnirs_times_ms": aligned_fnirs_times,
        "labels": aligned_eeg_labels,
        "residual_series_ms": residual_ms,
        "offset_blocks": blocks,
        "offset_pattern": classify_alignment_pattern(residual_ms, blocks, skipped),
    }


def summarize_intervals(seconds: np.ndarray) -> Dict[str, Any]:
    if seconds.size == 0:
        return {
            "count": 0,
            "median_s": None,
            "mean_s": None,
            "std_s": None,
            "min_s": None,
            "max_s": None,
            "p10_s": None,
            "p90_s": None,
            "scale_label": "not_available",
        }

    median_s = float(np.median(seconds))
    if median_s < 1.0:
        scale_label = "sub_second"
    elif median_s < 10.0:
        scale_label = "few_seconds"
    elif median_s < 120.0:
        scale_label = "tens_of_seconds"
    else:
        scale_label = "minutes"

    return {
        "count": int(seconds.size),
        "median_s": median_s,
        "mean_s": float(np.mean(seconds)),
        "std_s": float(np.std(seconds)),
        "min_s": float(np.min(seconds)),
        "max_s": float(np.max(seconds)),
        "p10_s": float(np.quantile(seconds, 0.10)),
        "p90_s": float(np.quantile(seconds, 0.90)),
        "scale_label": scale_label,
    }


def summarize_time_scale(alignment: Mapping[str, Any]) -> Dict[str, Any]:
    eeg_times = np.asarray(alignment["eeg_times_ms"], dtype=np.float64)
    fnirs_times = np.asarray(alignment["fnirs_times_ms"], dtype=np.float64)
    paired_times = 0.5 * (eeg_times + fnirs_times)
    return {
        "paired_intervals_s": summarize_intervals(np.diff(paired_times) / 1000.0),
        "eeg_intervals_s": summarize_intervals(np.diff(eeg_times) / 1000.0),
        "fnirs_intervals_s": summarize_intervals(np.diff(fnirs_times) / 1000.0),
    }


def project_optical_from_concentration(oxy: np.ndarray, deoxy: np.ndarray, kappa: float) -> Tuple[np.ndarray, np.ndarray]:
    high_oxy, high_deoxy = PROJECTION_CONFIG["high_from_oxy_deoxy"]
    low_oxy, low_deoxy = PROJECTION_CONFIG["low_from_oxy_deoxy"]
    high = float(kappa) * (float(high_oxy) * np.asarray(oxy, dtype=np.float64) + float(high_deoxy) * np.asarray(deoxy, dtype=np.float64))
    low = float(kappa) * (float(low_oxy) * np.asarray(oxy, dtype=np.float64) + float(low_deoxy) * np.asarray(deoxy, dtype=np.float64))
    return high.astype(np.float32), low.astype(np.float32)


def build_segments(
    alignment: Mapping[str, Any],
    eeg_total_samples: int,
    eeg_fs_hz: float,
    fnirs_total_samples: int,
    fnirs_fs_hz: float,
) -> List[Dict[str, Any]]:
    eeg_times_ms = np.asarray(alignment["eeg_times_ms"], dtype=np.float64)
    fnirs_times_ms = np.asarray(alignment["fnirs_times_ms"], dtype=np.float64)
    labels = [str(label) for label in alignment["labels"]]
    eeg_end_record_s = float(eeg_total_samples) / float(eeg_fs_hz)
    fnirs_end_record_s = float(fnirs_total_samples) / float(fnirs_fs_hz)
    segments: List[Dict[str, Any]] = []

    for index, label_name in enumerate(labels):
        eeg_start_s = float(eeg_times_ms[index]) / 1000.0
        fnirs_start_s = float(fnirs_times_ms[index]) / 1000.0
        if index + 1 < len(labels):
            eeg_end_s = float(eeg_times_ms[index + 1]) / 1000.0
            fnirs_end_s = float(fnirs_times_ms[index + 1]) / 1000.0
        else:
            eeg_end_s = eeg_end_record_s
            fnirs_end_s = fnirs_end_record_s

        eeg_start_sample = max(int(round(eeg_start_s * eeg_fs_hz)), 0)
        eeg_end_sample = min(int(round(eeg_end_s * eeg_fs_hz)), eeg_total_samples)
        fnirs_start_sample = max(int(round(fnirs_start_s * fnirs_fs_hz)), 0)
        fnirs_end_sample = min(int(round(fnirs_end_s * fnirs_fs_hz)), fnirs_total_samples)
        if eeg_end_sample <= eeg_start_sample or fnirs_end_sample <= fnirs_start_sample:
            continue

        eeg_duration_s = float(eeg_end_sample - eeg_start_sample) / float(eeg_fs_hz)
        fnirs_duration_s = float(fnirs_end_sample - fnirs_start_sample) / float(fnirs_fs_hz)
        segments.append(
            {
                "segment_index": int(index),
                "label_name": label_name,
                "eeg_onset_s": eeg_start_s,
                "fnirs_onset_s": fnirs_start_s,
                "eeg_start_sample": int(eeg_start_sample),
                "eeg_end_sample": int(eeg_end_sample),
                "fnirs_start_sample": int(fnirs_start_sample),
                "fnirs_end_sample": int(fnirs_end_sample),
                "eeg_duration_s": float(eeg_duration_s),
                "fnirs_duration_s": float(fnirs_duration_s),
                "common_duration_s": float(min(eeg_duration_s, fnirs_duration_s)),
            }
        )
    return segments