"""Analyze and visualize source/observation targets from the generated cache.

Reads one or more subject cache .npz files and produces:
  1. target_distributions.png     — histogram + KDE of source/obs targets per modality
  2. target_timecourses.png       — sample time-domain source/obs target traces
  3. cross_subject_summary.png    — per-subject aggregate statistics
  4. analysis_report.json         — numerical summary statistics

Usage:
    # Analyze a single subject
    python croce_validation/scripts/analyze_target_cache.py \
        --cache-dir croce_validation/cache/rk4_full/

    # Analyze with specific subject IDs
    python croce_validation/scripts/analyze_target_cache.py \
        --cache-dir croce_validation/cache/rk4_full/ \
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
    "fnirs_primary_source": "#17BECF",
    "fnirs_primary_obs": "#1F77B4",
    "fnirs_secondary_source": "#FF9896",
    "fnirs_secondary_obs": "#9467BD",
}

MODALITY_LABELS = {
    "eeg_source": "EEG Source Target",
    "eeg_obs": "EEG Observation Target",
    "fnirs_primary_source": "fNIRS Primary Source",
    "fnirs_primary_obs": "fNIRS Primary Obs",
    "fnirs_secondary_source": "fNIRS Secondary Source",
    "fnirs_secondary_obs": "fNIRS Secondary Obs",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze source/observation target cache.")
    p.add_argument("--cache-dir", required=True, help="Directory containing subject*_cache.npz files")
    p.add_argument("--subject-ids", default="", help="Comma-separated subject IDs (default: all found)")
    p.add_argument("--output-dir", default="")
    p.add_argument("--max-anchors-per-subject", type=int, default=3,
                   help="Max anchors to load per subject for analysis (default: 3)")
    p.add_argument("--sample-anchors", type=str, default="AF7Fp1",
                   help="Comma-separated anchor names for time-domain plots")
    return p.parse_args()


def find_cache_files(cache_dir: Path) -> Dict[int, Path]:
    """Find all subject*_cache.npz files in cache_dir."""
    files: Dict[int, Path] = {}
    for f in sorted(cache_dir.glob("subject*_cache.npz")):
        try:
            sid = int(f.stem.replace("subject", "").split("_")[0])
            files[sid] = f
        except ValueError:
            continue
    return files


def load_cache(cache_path: Path, max_anchors: int) -> Dict[str, np.ndarray]:
    """Load a cache file and aggregate across anchors into flat arrays per modality."""
    data = np.load(cache_path, allow_pickle=False)
    anchor_groups: Dict[str, List[str]] = {}
    for key in data.keys():
        parts = key.split("/", 1)
        if len(parts) == 2:
            anchor_groups.setdefault(parts[0], []).append(parts[1])

    anchors = sorted(anchor_groups.keys())
    if max_anchors > 0:
        anchors = anchors[:max_anchors]

    # Aggregate across anchors
    aggregated: Dict[str, List[np.ndarray]] = {}
    for anchor in anchors:
        for field in ["source_eeg", "source_fnirs_primary", "source_fnirs_secondary",
                       "obs_eeg", "obs_fnirs_primary", "obs_fnirs_secondary"]:
            key = f"{anchor}/{field}"
            if key in data:
                aggregated.setdefault(field, []).append(data[key])

    result: Dict[str, np.ndarray] = {}
    for field, arrays in aggregated.items():
        result[field] = np.concatenate([a.ravel() for a in arrays])

    return result


# ---------------------------------------------------------------------------
# Plot 1: Distribution histograms with KDE
# ---------------------------------------------------------------------------

def plot_target_distributions(
    output_path: Path,
    all_subject_data: Dict[int, Dict[str, np.ndarray]],
) -> None:
    fields = ["source_eeg", "obs_eeg", "source_fnirs_primary", "obs_fnirs_primary",
              "source_fnirs_secondary", "obs_fnirs_secondary"]
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

        label_key = field.replace("source_", "").replace("obs_", "")
        modality = "source" if "source" in field else "obs"
        ax.set_title(f"{label_key} ({modality})", fontsize=10, fontweight="bold")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)

    fig.suptitle("Source/Observation Target Distributions (All Subjects, RK4 Solver)",
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
    data = np.load(cache_path, allow_pickle=False)

    # Find all fields for this anchor
    anchor_key = anchor_name.replace(" ", "_").replace("-", "_")
    fields_available = [k for k in data.keys() if k.startswith(anchor_key + "/")]

    if not fields_available:
        print(f"  Anchor '{anchor_name}' not found, skipping timecourse plot")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Row 0: EEG
    ax = axes[0]
    src_key = f"{anchor_key}/source_eeg"
    obs_key = f"{anchor_key}/obs_eeg"
    if src_key in data:
        src = data[src_key]
        # Plot first channel
        ax.plot(src[:, 0] if src.ndim > 1 else src,
                color="#D62728", linewidth=1.2, label="Source (physiological)")
    if obs_key in data:
        obs = data[obs_key]
        ax.plot(obs[:, 0] if obs.ndim > 1 else obs,
                color="#1F77B4", linewidth=1.1, label="Observation (residual)")
    ax.axhline(0.0, color="#BDBDBD", linewidth=0.8, linestyle="--")
    ax.set_ylabel("EEG (μV)")
    ax.set_title(f"EEG Targets — {anchor_name}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    # Row 1: fNIRS Primary
    ax = axes[1]
    src_key = f"{anchor_key}/source_fnirs_primary"
    obs_key = f"{anchor_key}/obs_fnirs_primary"
    if src_key in data:
        ax.plot(data[src_key][:, 0] if data[src_key].ndim > 1 else data[src_key],
                color="#17BECF", linewidth=1.2, label="Source (physiological)")
    if obs_key in data:
        ax.plot(data[obs_key][:, 0] if data[obs_key].ndim > 1 else data[obs_key],
                color="#1F77B4", linewidth=1.1, label="Observation (residual)")
    ax.axhline(0.0, color="#BDBDBD", linewidth=0.8, linestyle="--")
    ax.set_ylabel("fNIRS Primary (a.u.)")
    ax.set_title(f"fNIRS Primary Targets — {anchor_name}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    # Row 2: fNIRS Secondary
    ax = axes[2]
    src_key = f"{anchor_key}/source_fnirs_secondary"
    obs_key = f"{anchor_key}/obs_fnirs_secondary"
    if src_key in data:
        ax.plot(data[src_key][:, 0] if data[src_key].ndim > 1 else data[src_key],
                color="#FF9896", linewidth=1.2, label="Source (physiological)")
    if obs_key in data:
        ax.plot(data[obs_key][:, 0] if data[obs_key].ndim > 1 else data[obs_key],
                color="#9467BD", linewidth=1.1, label="Observation (residual)")
    ax.axhline(0.0, color="#BDBDBD", linewidth=0.8, linestyle="--")
    ax.set_ylabel("fNIRS Secondary (a.u.)")
    ax.set_xlabel("Time (samples)")
    ax.set_title(f"fNIRS Secondary Targets — {anchor_name}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle("Source/Observation Target Timecourses (RK4 Solver)",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
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
        ("source_fnirs_primary", "fNIRS Pri. Source", "#17BECF"),
        ("obs_fnirs_primary", "fNIRS Pri. Obs", "#1F77B4"),
        ("source_fnirs_secondary", "fNIRS Sec. Source", "#FF9896"),
        ("obs_fnirs_secondary", "fNIRS Sec. Obs", "#9467BD"),
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

    fig.suptitle("Cross-Subject Target Statistics (RK4 Solver)",
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
        "per_subject_statistics": {
            str(sid): stats_dict for sid, stats_dict in all_subject_stats.items()
        },
        "aggregate_statistics": {},
    }

    # Compute aggregate across all subjects
    for field in ["source_eeg", "obs_eeg", "source_fnirs_primary", "obs_fnirs_primary",
                   "source_fnirs_secondary", "obs_fnirs_secondary"]:
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
