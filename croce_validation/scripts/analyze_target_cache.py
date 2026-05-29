"""Analyze and visualize source/observation targets from the generated cache.

Reads one or more subject cache .npz files and produces:
  1. target_distributions.png     — histogram + KDE of source/obs targets per modality
    2. target_timecourses.png       — sample time-domain source/obs target traces with real-time axes
  3. cross_subject_summary.png    — per-subject aggregate statistics
  4. analysis_report.json         — numerical summary statistics

Usage:
    # Analyze a single subject
    python croce_validation/scripts/analyze_target_cache.py \
        --cache-dir croce_validation/cache/pf_full/

    # Analyze with specific subject IDs
    python croce_validation/scripts/analyze_target_cache.py \
        --cache-dir croce_validation/cache/pf_full/ \
        --subject-ids 1,2,3,4,5
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Color palette (matching project conventions)
# ---------------------------------------------------------------------------

MODALITY_COLORS = {
    "eeg_source": "#D62728",
    "eeg_obs": "#1F77B4",
    "fnirs_optical_channel_0_source": "#17BECF",
    "fnirs_optical_channel_0_obs": "#1F77B4",
    "fnirs_optical_channel_1_source": "#FF9896",
    "fnirs_optical_channel_1_obs": "#9467BD",
}

MODALITY_LABELS = {
    "eeg_source": "EEG Source Target",
    "eeg_obs": "EEG Observation Target",
    "fnirs_optical_channel_0_source": "fNIRS Optical Channel 0 Source",
    "fnirs_optical_channel_0_obs": "fNIRS Optical Channel 0 Obs",
    "fnirs_optical_channel_1_source": "fNIRS Optical Channel 1 Source",
    "fnirs_optical_channel_1_obs": "fNIRS Optical Channel 1 Obs",
}

FIELD_ALIASES = {
    "source_eeg": ("source_eeg",),
    "obs_eeg": ("obs_eeg",),
    "source_fnirs_optical_channel_0": (
        "source_fnirs_optical_channel_0",
        "source_fnirs_optical_primary",
        "source_fnirs_primary",
        "source_fnirs_hbo",
    ),
    "obs_fnirs_optical_channel_0": (
        "obs_fnirs_optical_channel_0",
        "obs_fnirs_optical_primary",
        "obs_fnirs_primary",
        "obs_fnirs_hbo",
    ),
    "source_fnirs_optical_channel_1": (
        "source_fnirs_optical_channel_1",
        "source_fnirs_optical_secondary",
        "source_fnirs_secondary",
        "source_fnirs_hbr",
    ),
    "obs_fnirs_optical_channel_1": (
        "obs_fnirs_optical_channel_1",
        "obs_fnirs_optical_secondary",
        "obs_fnirs_secondary",
        "obs_fnirs_hbr",
    ),
}

CANONICAL_FIELDS = [
    "source_eeg",
    "obs_eeg",
    "source_fnirs_optical_channel_0",
    "obs_fnirs_optical_channel_0",
    "source_fnirs_optical_channel_1",
    "obs_fnirs_optical_channel_1",
]

PLOT_FIELD_LABEL_KEYS = {
    "source_eeg": "eeg_source",
    "obs_eeg": "eeg_obs",
    "source_fnirs_optical_channel_0": "fnirs_optical_channel_0_source",
    "obs_fnirs_optical_channel_0": "fnirs_optical_channel_0_obs",
    "source_fnirs_optical_channel_1": "fnirs_optical_channel_1_source",
    "obs_fnirs_optical_channel_1": "fnirs_optical_channel_1_obs",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze source/observation target cache.")
    p.add_argument(
        "--cache-dir",
        required=True,
        help="Directory containing subject_<id>/subject{id}_cache.npz files or legacy top-level subject{id}_cache.npz files",
    )
    p.add_argument("--subject-ids", default="", help="Comma-separated subject IDs (default: all found)")
    p.add_argument("--output-dir", default="")
    p.add_argument("--max-anchors-per-subject", type=int, default=3,
                   help="Max anchors to load per subject for analysis (default: 3)")
    p.add_argument("--sample-anchors", type=str, default="AF7Fp1",
                   help="Comma-separated anchor names for time-domain plots")
    return p.parse_args()


def parse_subject_id(cache_path: Path) -> Optional[int]:
    stem = cache_path.stem
    if not stem.startswith("subject"):
        return None
    try:
        return int(stem.replace("subject", "").split("_")[0])
    except ValueError:
        return None


def is_canonical_subject_cache(cache_path: Path, subject_id: int) -> bool:
    return cache_path.parent.name == f"subject_{subject_id}"


def resolve_manifest_path(cache_path: Path) -> Optional[Path]:
    subject_id = parse_subject_id(cache_path)
    if subject_id is None:
        return None

    if is_canonical_subject_cache(cache_path, subject_id):
        manifest_path = cache_path.parent / "cache_manifest.json"
        return manifest_path if manifest_path.exists() else None

    canonical_manifest = cache_path.parent / f"subject_{subject_id}" / "cache_manifest.json"
    if canonical_manifest.exists():
        return canonical_manifest

    manifest_path = cache_path.parent / "cache_manifest.json"
    return manifest_path if manifest_path.exists() else None


def load_manifest(cache_path: Path) -> Dict[str, Any]:
    manifest_path = resolve_manifest_path(cache_path)
    if manifest_path is None:
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def find_cache_files(cache_dir: Path) -> Dict[int, Path]:
    """Find subject caches, preferring canonical subject_<id>/ layout over legacy mirrors."""
    files: Dict[int, Path] = {}
    for pattern in ("subject_*/subject*_cache.npz", "subject*_cache.npz"):
        for cache_path in sorted(cache_dir.glob(pattern)):
            sid = parse_subject_id(cache_path)
            if sid is None or sid in files:
                continue
            files[sid] = cache_path
    return files


def load_cache(cache_path: Path, max_anchors: int) -> Dict[str, np.ndarray]:
    """Load a cache file and aggregate across anchors (and events) into flat arrays per modality.

    Supports both legacy ``anchor/field`` keys and the current
    ``anchor/event_XXX/field`` event-window layout.
    """
    with np.load(cache_path, allow_pickle=False) as data:
        data_keys = list(data.keys())
        # Normalise key structure to (anchor, field, optional event_part)
        key_triples: List[Tuple[str, str, Optional[str]]] = []
        for key in data_keys:
            parts = key.split("/")
            if len(parts) == 2:
                key_triples.append((parts[0], parts[1], None))
            elif len(parts) == 3:
                key_triples.append((parts[0], parts[2], parts[1]))

        anchors = sorted({t[0] for t in key_triples})
        if max_anchors > 0:
            anchors = anchors[:max_anchors]

        # Aggregate across anchors and events
        aggregated: Dict[str, List[np.ndarray]] = {}
        for anchor in anchors:
            for field in CANONICAL_FIELDS:
                for alias in FIELD_ALIASES[field]:
                    for anchor_name, field_name, event_part in key_triples:
                        if anchor_name != anchor or field_name != alias:
                            continue
                        aggregated.setdefault(field, []).append(np.asarray(data[f"{anchor}/{event_part}/{alias}" if event_part else f"{anchor}/{alias}"]))
                        break
                    else:
                        continue
                    break

    result: Dict[str, np.ndarray] = {}
    for field, arrays in aggregated.items():
        result[field] = np.concatenate([a.ravel() for a in arrays])

    return result


def build_time_axis(num_samples: int, duration_s: Optional[float], start_s: float = 0.0) -> np.ndarray:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if duration_s is None or duration_s <= 0.0:
        return start_s + np.arange(num_samples, dtype=np.float64)
    dt_s = float(duration_s) / float(num_samples)
    return start_s + np.arange(num_samples, dtype=np.float64) * dt_s


def pair_mode_uses_concentration_space(pair_mode: Optional[str]) -> bool:
    return str(pair_mode).strip().lower() in {"concentration", "chromophore"}


def infer_fnirs_channel_labels(
    pair_mode: Optional[str],
    pair_labels: Optional[Any] = None,
) -> Tuple[str, str]:
    defaults = ("HbO", "HbR") if pair_mode_uses_concentration_space(pair_mode) else ("Optical Channel 0", "Optical Channel 1")
    if isinstance(pair_labels, (list, tuple)) and len(pair_labels) >= 2:
        primary_label = str(pair_labels[0]).strip() or defaults[0]
        secondary_label = str(pair_labels[1]).strip() or defaults[1]
        return primary_label, secondary_label
    return defaults


def infer_fnirs_target_labels(
    pair_mode: Optional[str],
    pair_labels: Optional[Any] = None,
) -> Tuple[str, str, str]:
    primary_label, secondary_label = infer_fnirs_channel_labels(pair_mode, pair_labels)
    if pair_mode_uses_concentration_space(pair_mode):
        return (
            f"fNIRS {primary_label} Targets",
            f"fNIRS {secondary_label} Targets",
            "Legacy concentration-space caches store the two fNIRS target branches directly as oxy/deoxy-style concentration traces.",
        )
    return (
        f"fNIRS {primary_label} Targets",
        f"fNIRS {secondary_label} Targets",
        "Current cache contract keeps fNIRS targets in optical measurement space; concentration datasets should be forward-projected to an optical pair before caching.",
    )


# ---------------------------------------------------------------------------
# Plot 1: Distribution histograms with KDE
# ---------------------------------------------------------------------------

def plot_target_distributions(
    output_path: Path,
    all_subject_data: Dict[int, Dict[str, np.ndarray]],
) -> None:
    fields = CANONICAL_FIELDS
    colors = ["#D62728", "#1F77B4", "#17BECF", "#1F77B4", "#FF9896", "#9467BD"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, field in enumerate(fields):
        ax = axes[idx]
        # Collect data across all subjects
        all_vals = []
        for sid, data in all_subject_data.items():
            if field in data:
                vals = data[field]
                # Trim extreme outliers for visualization
                q01, q99 = np.percentile(vals, [0.1, 99.9])
                all_vals.append(np.clip(vals, q01, q99))

        if all_vals:
            combined = np.concatenate(all_vals)
            # Histogram
            ax.hist(combined, bins=80, density=True, alpha=0.5, color=colors[idx],
                    edgecolor="#111111", linewidth=0.3)
            # KDE
            if len(combined) > 1000:
                sample = np.random.choice(combined, size=min(50000, len(combined)), replace=False)
                kde = stats.gaussian_kde(sample)
                x_range = np.linspace(np.min(combined), np.max(combined), 200)
                ax.plot(x_range, kde(x_range), color=colors[idx], linewidth=2.0)

            ax.axvline(0.0, color="#111111", linewidth=0.8, linestyle="--", alpha=0.5)
            # Statistics annotation
            ax.text(0.02, 0.95,
                    f"μ={np.mean(combined):.3f}\nσ={np.std(combined):.3f}\nN={len(combined):,}",
                    transform=ax.transAxes, fontsize=8, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_title(MODALITY_LABELS[PLOT_FIELD_LABEL_KEYS[field]], fontsize=10, fontweight="bold")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)

    fig.suptitle("Source/Observation Target Distributions (All Subjects, Exact PF Solver)",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: Sample time-domain traces
# ---------------------------------------------------------------------------

def plot_target_timecourses(
    output_path: Path,
    cache_path: Path,
    anchor_name: str,
) -> None:
    """Plot source/obs target timecourses for a specific anchor."""
    manifest = load_manifest(cache_path)
    config = manifest.get("config", {}) if isinstance(manifest, dict) else {}
    segment_duration_s = config.get("segment_duration_s")
    segment_start_s = float(config.get("segment_start_s", 0.0) or 0.0)
    pair_mode = config.get("pair_mode")
    pair_labels = config.get("pair_labels")
    primary_label, secondary_label = infer_fnirs_channel_labels(pair_mode, pair_labels)
    primary_title, secondary_title, fnirs_note = infer_fnirs_target_labels(pair_mode, pair_labels)

    with np.load(cache_path, allow_pickle=False) as data:
        anchor_key = anchor_name.replace(" ", "_").replace("-", "_")
        fields_available = [k for k in data.keys() if k.startswith(anchor_key + "/")]

        if not fields_available:
            print(f"  Anchor '{anchor_name}' not found, skipping timecourse plot")
            return

        def _get_series(field_name: str) -> Optional[np.ndarray]:
            for alias in FIELD_ALIASES[field_name]:
                # Event-window layout (current): {anchor}/event_000/{field}
                for key in data.keys():
                    if not key.startswith(anchor_key + "/"):
                        continue
                    rest = key[len(anchor_key) + 1:]
                    if "/" in rest and rest.split("/")[-1] == alias:
                        arr = np.asarray(data[key])
                        return arr[:, 0] if arr.ndim > 1 else arr
                # Legacy layout: {anchor}/{field}
                key = f"{anchor_key}/{alias}"
                if key in data:
                    arr = np.asarray(data[key])
                    return arr[:, 0] if arr.ndim > 1 else arr
            return None

        eeg_src = _get_series("source_eeg")
        eeg_obs = _get_series("obs_eeg")
        fnirs_primary_src = _get_series("source_fnirs_optical_channel_0")
        fnirs_primary_obs = _get_series("obs_fnirs_optical_channel_0")
        fnirs_secondary_src = _get_series("source_fnirs_optical_channel_1")
        fnirs_secondary_obs = _get_series("obs_fnirs_optical_channel_1")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    use_seconds_axis = segment_duration_s is not None and float(segment_duration_s) > 0.0

    eeg_len = len(eeg_src) if eeg_src is not None else len(eeg_obs) if eeg_obs is not None else 0
    fnirs_primary_len = len(fnirs_primary_src) if fnirs_primary_src is not None else len(fnirs_primary_obs) if fnirs_primary_obs is not None else 0
    fnirs_secondary_len = len(fnirs_secondary_src) if fnirs_secondary_src is not None else len(fnirs_secondary_obs) if fnirs_secondary_obs is not None else 0

    eeg_time = build_time_axis(eeg_len, float(segment_duration_s) if use_seconds_axis and eeg_len > 0 else None, segment_start_s) if eeg_len > 0 else None
    fnirs_primary_time = build_time_axis(fnirs_primary_len, float(segment_duration_s) if use_seconds_axis and fnirs_primary_len > 0 else None, segment_start_s) if fnirs_primary_len > 0 else None
    fnirs_secondary_time = build_time_axis(fnirs_secondary_len, float(segment_duration_s) if use_seconds_axis and fnirs_secondary_len > 0 else None, segment_start_s) if fnirs_secondary_len > 0 else None

    # Row 0: EEG
    ax = axes[0]
    ax_src = ax
    ax_obs = ax.twinx()
    if eeg_src is not None and eeg_time is not None:
        ax_src.plot(eeg_time, eeg_src, color="#D62728", linewidth=1.2, label="Source (physiological)")
    if eeg_obs is not None and eeg_time is not None:
        ax_obs.plot(eeg_time, eeg_obs, color="#1F77B4", linewidth=1.1, label="Observation (residual)")
    ax_src.axhline(0.0, color="#BDBDBD", linewidth=0.8, linestyle="--")
    ax_src.set_ylabel("EEG (μV)  —  source", color="#D62728")
    ax_obs.set_ylabel("EEG (μV)  —  obs", color="#1F77B4")
    ax_src.tick_params(axis="y", colors="#D62728")
    ax_obs.tick_params(axis="y", colors="#1F77B4")
    lines_src, labels_src = ax_src.get_legend_handles_labels()
    lines_obs, labels_obs = ax_obs.get_legend_handles_labels()
    ax_src.legend(lines_src + lines_obs, labels_src + labels_obs, loc="upper right", fontsize=8)
    ax_src.set_title(f"EEG Targets — {anchor_name}")
    ax_src.grid(alpha=0.25)

    # Row 1: fNIRS Primary
    ax = axes[1]
    ax_src = ax
    ax_obs = ax.twinx()
    if fnirs_primary_src is not None and fnirs_primary_time is not None:
        ax_src.plot(fnirs_primary_time, fnirs_primary_src,
                color="#17BECF", linewidth=1.2, label="Source (physiological)")
    if fnirs_primary_obs is not None and fnirs_primary_time is not None:
        ax_obs.plot(fnirs_primary_time, fnirs_primary_obs,
                color="#1F77B4", linewidth=1.1, label="Observation (residual)")
    ax_src.axhline(0.0, color="#BDBDBD", linewidth=0.8, linestyle="--")
    ax_src.set_ylabel(f"fNIRS {primary_label}  —  source", color="#17BECF")
    ax_obs.set_ylabel(f"fNIRS {primary_label}  —  obs", color="#1F77B4")
    ax_src.tick_params(axis="y", colors="#17BECF")
    ax_obs.tick_params(axis="y", colors="#1F77B4")
    lines_src, labels_src = ax_src.get_legend_handles_labels()
    lines_obs, labels_obs = ax_obs.get_legend_handles_labels()
    ax_src.legend(lines_src + lines_obs, labels_src + labels_obs, loc="upper right", fontsize=8)
    ax_src.set_title(f"{primary_title} — {anchor_name}")
    ax_src.grid(alpha=0.25)

    # Row 2: fNIRS Secondary
    ax = axes[2]
    ax_src = ax
    ax_obs = ax.twinx()
    if fnirs_secondary_src is not None and fnirs_secondary_time is not None:
        ax_src.plot(fnirs_secondary_time, fnirs_secondary_src,
                color="#FF9896", linewidth=1.2, label="Source (physiological)")
    if fnirs_secondary_obs is not None and fnirs_secondary_time is not None:
        ax_obs.plot(fnirs_secondary_time, fnirs_secondary_obs,
                color="#9467BD", linewidth=1.1, label="Observation (residual)")
    ax_src.axhline(0.0, color="#BDBDBD", linewidth=0.8, linestyle="--")
    ax_src.set_ylabel(f"fNIRS {secondary_label}  —  source", color="#FF9896")
    ax_obs.set_ylabel(f"fNIRS {secondary_label}  —  obs", color="#9467BD")
    ax_src.tick_params(axis="y", colors="#FF9896")
    ax_obs.tick_params(axis="y", colors="#9467BD")
    lines_src, labels_src = ax_src.get_legend_handles_labels()
    lines_obs, labels_obs = ax_obs.get_legend_handles_labels()
    ax_src.legend(lines_src + lines_obs, labels_src + labels_obs, loc="upper right", fontsize=8)
    ax_src.set_xlabel("Time (s)" if use_seconds_axis else "Time (samples)")
    ax_src.set_title(f"{secondary_title} — {anchor_name}")
    ax_src.grid(alpha=0.25)

    if use_seconds_axis:
        axes[-1].set_xlim(segment_start_s, segment_start_s + float(segment_duration_s))
        fig.text(0.5, 0.012, fnirs_note, ha="center", fontsize=9)

    fig.suptitle("Source/Observation Target Timecourses (Exact PF Solver)",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.0, 0.03 if use_seconds_axis else 0.0, 1.0, 0.98))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 3: Cross-subject summary statistics
# ---------------------------------------------------------------------------

def plot_cross_subject_summary(
    output_path: Path,
    all_subject_stats: Dict[int, Dict[str, float]],
) -> None:
    """Per-subject aggregate statistics: mean, std, signal-to-residual ratio."""
    subjects = sorted(all_subject_stats.keys())
    n = len(subjects)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    modalities = [
        ("source_eeg", "EEG Source", "#D62728"),
        ("obs_eeg", "EEG Observation", "#1F77B4"),
        ("source_fnirs_optical_channel_0", "fNIRS Optical Channel 0 Source", "#17BECF"),
        ("obs_fnirs_optical_channel_0", "fNIRS Optical Channel 0 Obs", "#1F77B4"),
        ("source_fnirs_optical_channel_1", "fNIRS Optical Channel 1 Source", "#FF9896"),
        ("obs_fnirs_optical_channel_1", "fNIRS Optical Channel 1 Obs", "#9467BD"),
    ]

    for idx, (field, label, color) in enumerate(modalities):
        ax = axes[idx]
        means = [all_subject_stats[s].get(f"{field}_mean", np.nan) for s in subjects]
        stds = [all_subject_stats[s].get(f"{field}_std", np.nan) for s in subjects]

        x = np.arange(n)
        ax.bar(x, means, color=color, alpha=0.6, label="Mean")
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="#111111",
                    capsize=3, linewidth=0.8, label="±1σ")

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xticks(x[::5])
        ax.set_xticklabels([f"S{s}" for s in subjects[::5]], fontsize=8)
        ax.grid(alpha=0.25, axis="y")
        if idx == 0:
            ax.legend(loc="upper right", fontsize=7)

    # Signal-to-residual ratio panel
    ax = axes[5]  # Replace the last panel
    ax.clear()
    sr_ratios = []
    for s in subjects:
        src_var = all_subject_stats[s].get("source_eeg_var", 0)
        obs_var = all_subject_stats[s].get("obs_eeg_var", 1e-12)
        sr_ratios.append(src_var / max(obs_var, 1e-12))

    ax.bar(x, sr_ratios, color="#2CA02C", alpha=0.7, edgecolor="#111111", linewidth=0.5)
    ax.axhline(y=1.0, color="#BDBDBD", linewidth=0.8, linestyle="--", label="SRR=1")
    ax.set_title("EEG Signal-to-Residual Variance Ratio", fontsize=10, fontweight="bold")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Var(source) / Var(obs)")
    ax.set_xticks(x[::5])
    ax.set_xticklabels([f"S{s}" for s in subjects[::5]], fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper right", fontsize=7)

    fig.suptitle("Cross-Subject Target Statistics (Exact PF Solver)",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def compute_statistics(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute aggregate statistics for one subject's cache."""
    stats_dict: Dict[str, float] = {}
    for field, arr in data.items():
        arr_flat = arr.ravel()
        stats_dict[f"{field}_mean"] = float(np.mean(arr_flat))
        stats_dict[f"{field}_std"] = float(np.std(arr_flat))
        stats_dict[f"{field}_var"] = float(np.var(arr_flat))
        stats_dict[f"{field}_median"] = float(np.median(arr_flat))
        stats_dict[f"{field}_p01"] = float(np.percentile(arr_flat, 1))
        stats_dict[f"{field}_p99"] = float(np.percentile(arr_flat, 99))
        stats_dict[f"{field}_skew"] = float(stats.skew(arr_flat))
        stats_dict[f"{field}_kurtosis"] = float(stats.kurtosis(arr_flat))
    return stats_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    cache_files = find_cache_files(cache_dir)
    if not cache_files:
        raise FileNotFoundError(f"No subject*_cache.npz files found in {cache_dir}")

    # Filter by subject IDs if specified
    if args.subject_ids:
        requested = {int(s.strip()) for s in args.subject_ids.split(",") if s.strip()}
        cache_files = {k: v for k, v in cache_files.items() if k in requested}

    subjects = sorted(cache_files.keys())
    print(f"Found {len(subjects)} subject cache files: S{min(subjects)}–S{max(subjects)}")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = cache_dir / f"analysis_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Load data from all subjects
    print("\nLoading cache files...")
    all_subject_data: Dict[int, Dict[str, np.ndarray]] = {}
    all_subject_stats: Dict[int, Dict[str, float]] = {}

    for sid in subjects:
        print(f"  Subject {sid}...", end=" ", flush=True)
        try:
            data = load_cache(cache_files[sid], args.max_anchors_per_subject)
            stats_dict = compute_statistics(data)
            all_subject_data[sid] = data
            all_subject_stats[sid] = stats_dict
            n_fields = len(data)
            print(f"{n_fields} fields loaded")
        except Exception as e:
            print(f"ERROR: {e}")

    if not all_subject_data:
        raise RuntimeError("No subject data loaded. Check cache files.")

    # ---- Generate plots ----
    print("\nGenerating plots...")

    # 1. Distribution plots
    plot_target_distributions(
        output_dir / "target_distributions.png",
        all_subject_data,
    )

    # 2. Time-domain traces for a sample anchor on the first subject
    primary_subject = subjects[0]
    sample_anchors = [a.strip() for a in args.sample_anchors.split(",") if a.strip()]
    for anchor in sample_anchors:
        plot_target_timecourses(
            output_dir / f"timecourse_{anchor.replace(' ', '_')}_S{primary_subject}.png",
            cache_files[primary_subject],
            anchor,
        )

    # 3. Cross-subject summary
    plot_cross_subject_summary(
        output_dir / "cross_subject_summary.png",
        all_subject_stats,
    )

    # ---- Save numerical report ----
    report = {
        "generated_at": datetime.now().isoformat(),
        "cache_dir": str(cache_dir),
        "n_subjects": len(subjects),
        "subject_ids": subjects,
        "max_anchors_per_subject": args.max_anchors_per_subject,
        "fnirs_target_semantics": {},
        "per_subject_statistics": {
            str(sid): stats_dict for sid, stats_dict in all_subject_stats.items()
        },
        "aggregate_statistics": {},
    }

    reference_manifest = load_manifest(cache_files[subjects[0]]) if subjects else {}
    reference_config = reference_manifest.get("config", {}) if isinstance(reference_manifest, dict) else {}
    primary_title, secondary_title, fnirs_note = infer_fnirs_target_labels(
        reference_config.get("pair_mode"),
        reference_config.get("pair_labels"),
    )
    report["fnirs_target_semantics"] = {
        "pair_mode": reference_config.get("pair_mode", "unknown"),
        "pair_labels": reference_config.get("pair_labels", []),
        "primary_title": primary_title,
        "secondary_title": secondary_title,
        "description": fnirs_note,
    }

    # Compute aggregate across all subjects
    for field in CANONICAL_FIELDS:
        field_means = [all_subject_stats[s].get(f"{field}_mean", np.nan) for s in subjects]
        field_stds = [all_subject_stats[s].get(f"{field}_std", np.nan) for s in subjects]
        report["aggregate_statistics"][field] = {
            "mean_of_means": float(np.nanmean(field_means)),
            "std_of_means": float(np.nanstd(field_means)),
            "mean_of_stds": float(np.nanmean(field_stds)),
            "std_of_stds": float(np.nanstd(field_stds)),
        }

    report_path = output_dir / "analysis_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n  Report: {report_path}")

    # Print key summary
    print(f"\n{'='*60}")
    print("AGGREGATE STATISTICS (across all subjects)")
    print(f"{'='*60}")
    for field, agg in report["aggregate_statistics"].items():
        print(f"  {field:30s}: mean={agg['mean_of_means']:7.4f}  std={agg['mean_of_stds']:7.4f}")

    print(f"\nAll outputs in: {output_dir}")


if __name__ == "__main__":
    main()
