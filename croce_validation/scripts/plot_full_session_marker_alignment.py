#!/usr/bin/env python3
"""Plot full-session cache representative traces with EEG/fNIRS marker overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.eeg_fnirs_dataset import BBCIDataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize full-session cache traces with EEG/fNIRS marker overlays.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        required=True,
        help="Path to subject cache npz file.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/EEG+NIRS Single-Trial"),
        help="Path to the EEG+NIRS Single-Trial dataset root.",
    )
    parser.add_argument("--subject-id", type=int, required=True, help="Subject id.")
    parser.add_argument("--session-idx", type=int, default=0, help="Session index.")
    parser.add_argument(
        "--align-mode",
        choices=("first_event", "recording_start"),
        default="first_event",
        help="Time-axis alignment mode. 'first_event' matches the continuous alignment reference script.",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default=None,
        help="Specific anchor name. Omit to average representative traces across all anchors.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the PNG figure and JSON summary.",
    )
    return parser.parse_args()


def load_manifest(cache_path: Path) -> Dict[str, object]:
    manifest_path = cache_path.with_name("cache_manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found next to cache: {manifest_path}")
    return json.loads(manifest_path.read_text())


def list_anchors(npz: np.lib.npyio.NpzFile) -> List[str]:
    return sorted({key.split("/", 1)[0] for key in npz.files if "/" in key})


def load_marker_stream(
    data_root: Path,
    subject_id: int,
    session_idx: int,
    modality: str,
) -> Dict[str, np.ndarray]:
    loader = BBCIDataLoader(
        data_root=str(data_root),
        subject_ids=[subject_id],
        task="motor_imagery",
        modality=modality,
        use_artifact_data=True,
    )
    _, marker_list, _ = loader.load_subject_data(subject_id, modality)
    marker = marker_list[session_idx]
    label_ids = np.argmax(marker["y"], axis=0).astype(int)
    class_names = marker.get("className")
    labels = []
    for label_id in label_ids:
        if class_names is not None and len(class_names) > label_id:
            labels.append(str(class_names[label_id]))
        else:
            labels.append(str(label_id))
    return {
        "time_s": np.asarray(marker["time"], dtype=float) / 1000.0,
        "label_ids": label_ids,
        "labels": np.asarray(labels, dtype=object),
    }


def _require_array(npz: np.lib.npyio.NpzFile, anchor_key: str, field_name: str) -> np.ndarray:
    key = f"{anchor_key}/{field_name}"
    if key not in npz:
        raise KeyError(f"Missing field '{field_name}' for anchor '{anchor_key}'")
    return np.asarray(npz[key], dtype=np.float32)


def _representative_eeg(series: np.ndarray) -> np.ndarray:
    if series.ndim == 1:
        return np.abs(series)
    return np.sqrt(np.mean(np.square(series), axis=1))


def _representative_fnirs(series: np.ndarray) -> np.ndarray:
    if series.ndim == 1:
        return series
    return np.mean(series, axis=1)


def _moving_average(series: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return series
    kernel = np.ones(window, dtype=np.float32) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(series, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _zscore(series: np.ndarray) -> np.ndarray:
    std = float(np.std(series))
    if std < 1e-8:
        return np.zeros_like(series)
    return (series - float(np.mean(series))) / std


def _resample_for_plot(
    time_axis: np.ndarray,
    series: np.ndarray,
    target_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(series) <= target_points:
        return time_axis, series

    edges = np.linspace(0, len(series), target_points + 1, dtype=int)
    resampled_time = np.empty(target_points, dtype=np.float32)
    resampled_series = np.empty(target_points, dtype=np.float32)
    for idx in range(target_points):
        start = edges[idx]
        stop = max(edges[idx + 1], start + 1)
        resampled_time[idx] = float(np.mean(time_axis[start:stop]))
        resampled_series[idx] = float(np.mean(series[start:stop]))
    return resampled_time, resampled_series


def _prepare_series(
    series: np.ndarray,
    duration_s: float,
    smooth_seconds: float,
    target_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    time_axis = np.linspace(0.0, duration_s, len(series), endpoint=False, dtype=np.float32)
    sampling_rate = len(series) / duration_s
    smooth_window = max(1, int(round(smooth_seconds * sampling_rate)))
    smoothed = _moving_average(series.astype(np.float32, copy=False), smooth_window)
    standardized = _zscore(smoothed)
    return _resample_for_plot(time_axis, standardized, target_points)


def align_time_axis(
    time_axis: np.ndarray,
    event_times_s: np.ndarray,
    align_mode: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if align_mode == "recording_start" or event_times_s.size == 0:
        return time_axis.copy(), event_times_s.copy(), 0.0
    anchor_s = float(event_times_s[0])
    return time_axis - anchor_s, event_times_s - anchor_s, anchor_s


def build_sync_summary(
    eeg_event_times_s: np.ndarray,
    fnirs_event_times_s: np.ndarray,
    eeg_anchor_s: float,
    fnirs_anchor_s: float,
    align_mode: str,
) -> Dict[str, float | int | str | None]:
    common_events = int(min(len(eeg_event_times_s), len(fnirs_event_times_s)))
    residual_ms = (fnirs_event_times_s[:common_events] - eeg_event_times_s[:common_events]) * 1000.0
    return {
        "align_mode": align_mode,
        "num_common_events_compared": common_events,
        "eeg_alignment_anchor_s": float(eeg_anchor_s),
        "fnirs_alignment_anchor_s": float(fnirs_anchor_s),
        "initial_raw_offset_ms": float((fnirs_anchor_s - eeg_anchor_s) * 1000.0),
        "residual_mean_ms": float(np.mean(residual_ms)) if common_events else None,
        "residual_std_ms": float(np.std(residual_ms)) if common_events else None,
        "residual_max_abs_ms": float(np.max(np.abs(residual_ms))) if common_events else None,
    }


def collect_cache_representatives(
    cache_path: Path,
    anchor_name: str | None,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    with np.load(cache_path, allow_pickle=False) as npz:
        anchors = list_anchors(npz)
        if not anchors:
            raise ValueError(f"No anchors found in cache: {cache_path}")

        if anchor_name is not None:
            anchor_key = anchor_name.replace(" ", "_").replace("-", "_")
            if anchor_key not in anchors:
                raise KeyError(f"Anchor '{anchor_name}' not found in cache")
            anchors = [anchor_key]

        eeg_raw_all: List[np.ndarray] = []
        eeg_source_all: List[np.ndarray] = []
        eeg_obs_all: List[np.ndarray] = []
        eeg_r_all: List[np.ndarray] = []
        fnirs0_raw_all: List[np.ndarray] = []
        fnirs0_source_all: List[np.ndarray] = []
        fnirs0_obs_all: List[np.ndarray] = []
        fnirs1_raw_all: List[np.ndarray] = []
        fnirs1_source_all: List[np.ndarray] = []
        fnirs1_obs_all: List[np.ndarray] = []

        for anchor_key in anchors:
            eeg_source = _require_array(npz, anchor_key, "source_eeg")
            eeg_obs = _require_array(npz, anchor_key, "obs_eeg")
            eeg_r = _require_array(npz, anchor_key, "r_estimates_eeg")
            fnirs0_source = _require_array(npz, anchor_key, "source_fnirs_optical_channel_0")
            fnirs0_obs = _require_array(npz, anchor_key, "obs_fnirs_optical_channel_0")
            fnirs1_source = _require_array(npz, anchor_key, "source_fnirs_optical_channel_1")
            fnirs1_obs = _require_array(npz, anchor_key, "obs_fnirs_optical_channel_1")

            eeg_source_all.append(_representative_eeg(eeg_source))
            eeg_obs_all.append(_representative_eeg(eeg_obs))
            eeg_raw_all.append(_representative_eeg(eeg_source + eeg_obs))
            eeg_r_all.append(np.asarray(eeg_r, dtype=np.float32))

            fnirs0_source_all.append(_representative_fnirs(fnirs0_source))
            fnirs0_obs_all.append(_representative_fnirs(fnirs0_obs))
            fnirs0_raw_all.append(_representative_fnirs(fnirs0_source + fnirs0_obs))

            fnirs1_source_all.append(_representative_fnirs(fnirs1_source))
            fnirs1_obs_all.append(_representative_fnirs(fnirs1_obs))
            fnirs1_raw_all.append(_representative_fnirs(fnirs1_source + fnirs1_obs))

    representatives = {
        "eeg_raw": np.mean(np.stack(eeg_raw_all, axis=0), axis=0),
        "eeg_source": np.mean(np.stack(eeg_source_all, axis=0), axis=0),
        "eeg_obs": np.mean(np.stack(eeg_obs_all, axis=0), axis=0),
        "eeg_r": np.mean(np.stack(eeg_r_all, axis=0), axis=0),
        "fnirs0_raw": np.mean(np.stack(fnirs0_raw_all, axis=0), axis=0),
        "fnirs0_source": np.mean(np.stack(fnirs0_source_all, axis=0), axis=0),
        "fnirs0_obs": np.mean(np.stack(fnirs0_obs_all, axis=0), axis=0),
        "fnirs1_raw": np.mean(np.stack(fnirs1_raw_all, axis=0), axis=0),
        "fnirs1_source": np.mean(np.stack(fnirs1_source_all, axis=0), axis=0),
        "fnirs1_obs": np.mean(np.stack(fnirs1_obs_all, axis=0), axis=0),
    }
    return representatives, anchors


def draw_marker_lines(
    ax: plt.Axes,
    times_s: np.ndarray,
    x_min: float,
    x_max: float,
    color: str,
    linestyle: str,
    alpha: float,
) -> int:
    visible = times_s[(times_s >= x_min) & (times_s <= x_max)]
    for event_time in visible:
        ax.axvline(event_time, color=color, linestyle=linestyle, linewidth=0.8, alpha=alpha)
    return int(len(visible))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.cache_path)
    config = manifest.get("config", {}) if isinstance(manifest, dict) else {}
    duration_s = float(config.get("segment_duration_s", 0.0))
    if duration_s <= 0.0:
        raise ValueError("This visualization expects a positive full-session segment duration")

    representatives, anchors_used = collect_cache_representatives(args.cache_path, args.anchor)
    eeg_markers = load_marker_stream(args.data_root, args.subject_id, args.session_idx, "eeg")
    fnirs_markers = load_marker_stream(args.data_root, args.subject_id, args.session_idx, "fnirs")

    raw_first_event_offset_s = float(fnirs_markers["time_s"][0] - eeg_markers["time_s"][0])

    eeg_time_raw, eeg_raw = _prepare_series(representatives["eeg_raw"], duration_s, smooth_seconds=1.0, target_points=6000)
    _, eeg_source = _prepare_series(representatives["eeg_source"], duration_s, smooth_seconds=1.0, target_points=6000)
    _, eeg_obs = _prepare_series(representatives["eeg_obs"], duration_s, smooth_seconds=1.0, target_points=6000)
    _, eeg_r = _prepare_series(representatives["eeg_r"], duration_s, smooth_seconds=1.0, target_points=6000)

    fnirs_time_raw, fnirs0_raw = _prepare_series(representatives["fnirs0_raw"], duration_s, smooth_seconds=1.0, target_points=6000)
    _, fnirs0_source = _prepare_series(representatives["fnirs0_source"], duration_s, smooth_seconds=1.0, target_points=6000)
    _, fnirs0_obs = _prepare_series(representatives["fnirs0_obs"], duration_s, smooth_seconds=1.0, target_points=6000)
    _, fnirs1_raw = _prepare_series(representatives["fnirs1_raw"], duration_s, smooth_seconds=1.0, target_points=6000)
    _, fnirs1_source = _prepare_series(representatives["fnirs1_source"], duration_s, smooth_seconds=1.0, target_points=6000)
    _, fnirs1_obs = _prepare_series(representatives["fnirs1_obs"], duration_s, smooth_seconds=1.0, target_points=6000)

    eeg_time, eeg_marker_times, eeg_anchor_s = align_time_axis(
        eeg_time_raw,
        eeg_markers["time_s"],
        args.align_mode,
    )
    fnirs_time, fnirs_marker_times, fnirs_anchor_s = align_time_axis(
        fnirs_time_raw,
        fnirs_markers["time_s"],
        args.align_mode,
    )
    sync_summary = build_sync_summary(
        eeg_marker_times,
        fnirs_marker_times,
        eeg_anchor_s,
        fnirs_anchor_s,
        args.align_mode,
    )
    x_min = float(min(eeg_time[0], fnirs_time[0], np.min(eeg_marker_times), np.min(fnirs_marker_times)))
    x_max = float(max(eeg_time[-1], fnirs_time[-1], np.max(eeg_marker_times), np.max(fnirs_marker_times)))
    eeg_marker_visible = eeg_marker_times[(eeg_marker_times >= x_min) & (eeg_marker_times <= x_max)]
    fnirs_marker_visible = fnirs_marker_times[(fnirs_marker_times >= x_min) & (fnirs_marker_times <= x_max)]
    eeg_visible = int(len(eeg_marker_visible))
    fnirs_visible = int(len(fnirs_marker_visible))

    figure_name = "full_session_marker_alignment"
    if args.anchor is not None:
        figure_name += f"_{args.anchor.replace(' ', '_').replace('-', '_')}"
    else:
        figure_name += "_all_anchors_mean"

    figure_path = args.output_dir / f"{figure_name}.png"
    summary_path = args.output_dir / f"{figure_name}.json"

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(16, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [0.7, 1.4, 1.2, 1.2]},
    )

    event_ax = axes[0]
    event_ax.scatter(eeg_marker_visible, np.ones_like(eeg_marker_visible), color="#2CA02C", s=22, label="EEG mrk")
    event_ax.scatter(fnirs_marker_visible, np.zeros_like(fnirs_marker_visible), color="#9467BD", marker="x", s=26, label="fNIRS mrk")
    event_ax.set_yticks([0.0, 1.0])
    event_ax.set_yticklabels(["fNIRS", "EEG"])
    event_ax.set_title(
        f"Marker Streams on the {args.align_mode} timeline | "
        f"raw first-event offset: {raw_first_event_offset_s:.2f}s"
    )
    event_ax.grid(alpha=0.2)
    event_ax.legend(loc="upper right", fontsize=9)

    eeg_ax = axes[1]
    draw_marker_lines(eeg_ax, eeg_marker_times, x_min, x_max, color="#2CA02C", linestyle="-", alpha=0.10)
    draw_marker_lines(eeg_ax, fnirs_marker_times, x_min, x_max, color="#9467BD", linestyle="--", alpha=0.10)
    eeg_ax.plot(eeg_time, eeg_raw, color="#4D4D4D", linewidth=1.1, label="Raw = source + observation")
    eeg_ax.plot(eeg_time, eeg_source, color="#D62728", linewidth=1.0, label="Source")
    eeg_ax.plot(eeg_time, eeg_obs, color="#1F77B4", linewidth=0.9, alpha=0.7, label="Observation")
    eeg_ax.plot(eeg_time, eeg_r, color="#111111", linewidth=1.0, linestyle=":", label="Latent r_estimates_eeg")
    eeg_ax.axhline(0.0, color="#BDBDBD", linewidth=0.8, linestyle="--")
    eeg_ax.set_ylabel("EEG rep.\n(z-score)")
    eeg_ax.set_title(
        "EEG representative trace "
        + (f"for anchor {anchors_used[0]}" if len(anchors_used) == 1 else f"(mean over {len(anchors_used)} anchors)")
    )
    eeg_ax.grid(alpha=0.25)
    eeg_ax.legend(loc="upper right", fontsize=8)

    fnirs0_ax = axes[2]
    draw_marker_lines(fnirs0_ax, eeg_marker_times, x_min, x_max, color="#2CA02C", linestyle="-", alpha=0.10)
    draw_marker_lines(fnirs0_ax, fnirs_marker_times, x_min, x_max, color="#9467BD", linestyle="--", alpha=0.10)
    fnirs0_ax.plot(fnirs_time, fnirs0_raw, color="#4D4D4D", linewidth=1.1, label="Raw = source + observation")
    fnirs0_ax.plot(fnirs_time, fnirs0_source, color="#17BECF", linewidth=1.0, label="Source")
    fnirs0_ax.plot(fnirs_time, fnirs0_obs, color="#1F77B4", linewidth=0.9, alpha=0.7, label="Observation")
    fnirs0_ax.axhline(0.0, color="#BDBDBD", linewidth=0.8, linestyle="--")
    fnirs0_ax.set_ylabel("Optical ch. 0\n(z-score)")
    fnirs0_ax.set_title("fNIRS optical channel 0 representative trace")
    fnirs0_ax.grid(alpha=0.25)
    fnirs0_ax.legend(loc="upper right", fontsize=8)

    fnirs1_ax = axes[3]
    draw_marker_lines(fnirs1_ax, eeg_marker_times, x_min, x_max, color="#2CA02C", linestyle="-", alpha=0.10)
    draw_marker_lines(fnirs1_ax, fnirs_marker_times, x_min, x_max, color="#9467BD", linestyle="--", alpha=0.10)
    fnirs1_ax.plot(fnirs_time, fnirs1_raw, color="#4D4D4D", linewidth=1.1, label="Raw = source + observation")
    fnirs1_ax.plot(fnirs_time, fnirs1_source, color="#FF9896", linewidth=1.0, label="Source")
    fnirs1_ax.plot(fnirs_time, fnirs1_obs, color="#9467BD", linewidth=0.9, alpha=0.7, label="Observation")
    fnirs1_ax.axhline(0.0, color="#BDBDBD", linewidth=0.8, linestyle="--")
    fnirs1_ax.set_ylabel("Optical ch. 1\n(z-score)")
    fnirs1_ax.set_title("fNIRS optical channel 1 representative trace")
    fnirs1_ax.grid(alpha=0.25)
    fnirs1_ax.legend(loc="upper right", fontsize=8)
    fnirs1_ax.set_xlabel("Aligned time (s)" if args.align_mode == "first_event" else "Time (s)")
    fnirs1_ax.set_xlim(x_min, x_max)

    marker_handles = [
        Line2D([0], [0], color="#2CA02C", linestyle="-", linewidth=1.0, label="EEG mrk"),
        Line2D([0], [0], color="#9467BD", linestyle="--", linewidth=1.0, label="fNIRS mrk"),
    ]
    fig.legend(handles=marker_handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.995), frameon=False)
    fig.suptitle(
        "Subject Full-Session Alignment Check with Dataset Markers",
        fontsize=14,
        fontweight="bold",
        y=0.998,
    )
    residual_mean_ms = sync_summary["residual_mean_ms"]
    residual_std_ms = sync_summary["residual_std_ms"]
    fig.text(
        0.5,
        0.012,
        f"align_mode={args.align_mode}; visible markers in plotted range: EEG {eeg_visible}/{len(eeg_marker_times)}, fNIRS {fnirs_visible}/{len(fnirs_marker_times)}; "
        f"post-alignment residual mean/std = {residual_mean_ms:.2f}/{residual_std_ms:.2f} ms. "
        f"Signals are representative traces derived from the exact PF cache and standardized for timing inspection.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.975))
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "cache_path": str(args.cache_path),
        "figure_path": str(figure_path),
        "subject_id": args.subject_id,
        "session_idx": args.session_idx,
        "align_mode": args.align_mode,
        "anchor_mode": anchors_used[0] if len(anchors_used) == 1 else "all_anchors_mean",
        "anchors_used": anchors_used,
        "segment_duration_s": duration_s,
        "eeg_marker_first_raw_s": float(eeg_markers["time_s"][0]),
        "fnirs_marker_first_raw_s": float(fnirs_markers["time_s"][0]),
        "eeg_to_fnirs_first_marker_raw_offset_s": raw_first_event_offset_s,
        "eeg_alignment_anchor_s": eeg_anchor_s,
        "fnirs_alignment_anchor_s": fnirs_anchor_s,
        "residual_mean_ms": sync_summary["residual_mean_ms"],
        "residual_std_ms": sync_summary["residual_std_ms"],
        "residual_max_abs_ms": sync_summary["residual_max_abs_ms"],
        "eeg_marker_count_total": int(len(eeg_markers["time_s"])),
        "fnirs_marker_count_total": int(len(fnirs_markers["time_s"])),
        "eeg_marker_count_visible": eeg_visible,
        "fnirs_marker_count_visible": fnirs_visible,
        "eeg_first_visible_marker_s": float(eeg_marker_visible[0]) if len(eeg_marker_visible) else None,
        "fnirs_first_visible_marker_s": float(fnirs_marker_visible[0]) if len(fnirs_marker_visible) else None,
        "eeg_last_visible_marker_s": float(eeg_marker_visible[-1]) if len(eeg_marker_visible) else None,
        "fnirs_last_visible_marker_s": float(fnirs_marker_visible[-1]) if len(fnirs_marker_visible) else None,
        "plot_x_min_s": x_min,
        "plot_x_max_s": x_max,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved figure: {figure_path}")
    print(f"Saved summary: {summary_path}")
    print(
        "Marker visibility inside common prefix: "
        f"EEG {eeg_visible}/{len(eeg_markers['time_s'])}, "
        f"fNIRS {fnirs_visible}/{len(fnirs_markers['time_s'])}"
    )


if __name__ == "__main__":
    main()