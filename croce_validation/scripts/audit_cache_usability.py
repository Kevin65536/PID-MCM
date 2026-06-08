"""Deep audit of EEG/fNIRS source/observation target caches for usability.

Compares original vs regenerated caches across all subjects with:
  1. Per-field distribution statistics (mean, std, skew, kurtosis, SNR)
  2. EEG source target artifact detection (stair-stepping from np.repeat)
  3. Power spectral density (PSD) comparison for EEG targets
  4. fNIRS source variance check (expected ~constant pathlength factor)
  5. Cross-subject variability summary
  6. Usability verdict per field

Usage:
    python croce_validation/scripts/audit_cache_usability.py
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
from scipy import integrate, signal, stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANONICAL_FIELDS = [
    "source_eeg",
    "obs_eeg",
    "source_fnirs_optical_channel_0",
    "obs_fnirs_optical_channel_0",
    "source_fnirs_optical_channel_1",
    "obs_fnirs_optical_channel_1",
]

FIELD_ALIASES = {
    "source_eeg": ("source_eeg",),
    "obs_eeg": ("obs_eeg",),
    "source_fnirs_optical_channel_0": (
        "source_fnirs_optical_channel_0", "source_fnirs_optical_primary",
        "source_fnirs_primary", "source_fnirs_hbo",
    ),
    "obs_fnirs_optical_channel_0": (
        "obs_fnirs_optical_channel_0", "obs_fnirs_optical_primary",
        "obs_fnirs_primary", "obs_fnirs_hbo",
    ),
    "source_fnirs_optical_channel_1": (
        "source_fnirs_optical_channel_1", "source_fnirs_optical_secondary",
        "source_fnirs_secondary", "source_fnirs_hbr",
    ),
    "obs_fnirs_optical_channel_1": (
        "obs_fnirs_optical_channel_1", "obs_fnirs_optical_secondary",
        "obs_fnirs_secondary", "obs_fnirs_hbr",
    ),
}

FIELD_LABELS = {
    "source_eeg": "EEG Source",
    "obs_eeg": "EEG Observation",
    "source_fnirs_optical_channel_0": "fNIRS Ch0 Source",
    "obs_fnirs_optical_channel_0": "fNIRS Ch0 Obs",
    "source_fnirs_optical_channel_1": "fNIRS Ch1 Source",
    "obs_fnirs_optical_channel_1": "fNIRS Ch1 Obs",
}

FIELD_COLORS = {
    "source_eeg": "#D62728",
    "obs_eeg": "#1F77B4",
    "source_fnirs_optical_channel_0": "#17BECF",
    "obs_fnirs_optical_channel_0": "#1F77B4",
    "source_fnirs_optical_channel_1": "#FF9896",
    "obs_fnirs_optical_channel_1": "#9467BD",
}

# Expected sampling rates
EEG_FS = 200.0   # Hz
FNIRS_FS = 10.0  # Hz


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_cache_files(cache_dir: Path) -> Dict[int, Path]:
    files: Dict[int, Path] = {}
    for pattern in ("subject_*/subject*_cache.npz", "subject*_cache.npz"):
        for cache_path in sorted(cache_dir.glob(pattern)):
            stem = cache_path.stem
            if not stem.startswith("subject"):
                continue
            try:
                sid = int(stem.replace("subject", "").split("_")[0])
            except ValueError:
                continue
            if sid not in files:
                files[sid] = cache_path
    return files


def load_cache_aggregated(cache_path: Path, max_anchors: int = 0) -> Dict[str, np.ndarray]:
    """Load cache, aggregate across anchors/events into flat arrays."""
    with np.load(cache_path, allow_pickle=False) as data:
        all_keys = list(data.keys())

        # Parse keys: anchor/event_XXX/field or anchor/field
        key_triples: List[Tuple[str, str, Optional[str]]] = []
        for key in all_keys:
            parts = key.split("/")
            if len(parts) == 2:
                key_triples.append((parts[0], parts[1], None))
            elif len(parts) == 3:
                key_triples.append((parts[0], parts[2], parts[1]))

        anchors = sorted({t[0] for t in key_triples})
        if max_anchors > 0:
            anchors = anchors[:max_anchors]

        aggregated: Dict[str, List[np.ndarray]] = {}
        for anchor in anchors:
            for field in CANONICAL_FIELDS:
                for alias in FIELD_ALIASES[field]:
                    found = False
                    for anchor_name, field_name, event_part in key_triples:
                        if anchor_name != anchor or field_name != alias:
                            continue
                        key = f"{anchor}/{event_part}/{alias}" if event_part else f"{anchor}/{alias}"
                        aggregated.setdefault(field, []).append(np.asarray(data[key]))
                        found = True
                        break
                    if found:
                        break

    result: Dict[str, np.ndarray] = {}
    for field, arrays in aggregated.items():
        result[field] = np.concatenate([a.ravel() for a in arrays])
    return result


def load_single_event_eeg(cache_path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load single EEG event for time-domain and PSD analysis."""
    with np.load(cache_path, allow_pickle=False) as data:
        # Find first event for first anchor
        all_keys = list(data.keys())
        eeg_keys = [k for k in all_keys if k.endswith("/source_eeg")]
        if not eeg_keys:
            return None

        key = eeg_keys[0]
        prefix = key.rsplit("/", 1)[0]  # anchor/event_XXX

        result = {}
        for field in CANONICAL_FIELDS:
            for alias in FIELD_ALIASES[field]:
                fkey = f"{prefix}/{alias}"
                if fkey in data:
                    arr = np.asarray(data[fkey])
                    result[field] = arr[:, 0] if arr.ndim > 1 else arr
                    break

        # Also get r_estimates_eeg if available
        for alias in ("r_estimates_eeg",):
            fkey = f"{prefix}/{alias}"
            if fkey in data:
                result["r_estimates_eeg"] = np.asarray(data[fkey]).ravel()
                break

        return result


# ---------------------------------------------------------------------------
# Stair-step artifact detection
# ---------------------------------------------------------------------------

def detect_stair_stepping(signal_1d: np.ndarray, eeg_fs: float = EEG_FS,
                          fnirs_fs: float = FNIRS_FS) -> Dict[str, float]:
    """Detect stair-step artifacts from naive np.repeat upsampling.

    If EEG source was generated by repeating fNIRS-rate values (10 Hz → 200 Hz),
    we'd see flat segments of length eeg_fs/fnirs_fs = 20 samples.
    """
    ratio = int(eeg_fs / fnirs_fs)  # 20

    # Compute first difference
    diff = np.diff(signal_1d)

    # Count zero-steps: consecutive identical values (within float32 precision)
    tol = max(1e-10, float(np.finfo(np.float32).eps * 10))
    zero_steps = np.sum(np.abs(diff) < tol)

    # Count runs of exactly ratio-1 consecutive zeros in diff
    # A stair-step pattern produces (ratio-1) zeros, then one non-zero
    run_lengths = []
    current_run = 0
    for d in diff:
        if np.abs(d) < tol:
            current_run += 1
        else:
            if current_run > 0:
                run_lengths.append(current_run)
            current_run = 0
    if current_run > 0:
        run_lengths.append(current_run)

    run_lengths = np.array(run_lengths, dtype=np.int64)

    metrics = {
        "zero_diff_fraction": float(zero_steps / len(diff)) if len(diff) > 0 else 0.0,
        "n_runs": len(run_lengths),
        "mean_run_length": float(np.mean(run_lengths)) if len(run_lengths) > 0 else 0.0,
        "median_run_length": float(np.median(run_lengths)) if len(run_lengths) > 0 else 0.0,
        "run_length_std": float(np.std(run_lengths)) if len(run_lengths) > 0 else 0.0,
        "expected_artifact_ratio": float(ratio - 1),
        "artifact_fraction": float(np.mean(run_lengths == (ratio - 1))) if len(run_lengths) > 0 else 0.0,
    }

    # Heuristic: if median run length ≈ ratio-1 (19) and >50% of runs are ratio-1,
    # stair-stepping is present
    metrics["stair_step_detected"] = float(
        metrics["artifact_fraction"] > 0.3 and
        abs(metrics["median_run_length"] - metrics["expected_artifact_ratio"]) < 3
    )

    return metrics


# ---------------------------------------------------------------------------
# PSD analysis
# ---------------------------------------------------------------------------

def compute_psd(signal_1d: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD."""
    nperseg = min(256, len(signal_1d) // 4)
    freqs, psd = signal.welch(
        signal_1d, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
        detrend="constant", scaling="density"
    )
    return freqs, psd


def find_psd_artifacts(freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
    """Check PSD for artifacts at fnirs_fs and its harmonics (10, 20, 30... Hz)."""
    harmonics = [10.0, 20.0, 30.0, 40.0]
    artifact_peaks = {}

    # Mean PSD across all frequencies
    mean_psd = np.mean(psd)

    for h in harmonics:
        idx = np.argmin(np.abs(freqs - h))
        # Check a small bandwidth around the harmonic
        bw = 2  # Hz
        mask = (freqs >= h - bw) & (freqs <= h + bw)
        local_psd = psd[mask]
        if len(local_psd) > 0 and mean_psd > 0:
            peak_ratio = float(np.max(local_psd) / mean_psd)
        else:
            peak_ratio = 1.0
        artifact_peaks[f"harmonic_{h:.0f}Hz_ratio"] = peak_ratio

    # Integrated power in 8-12 Hz band (near 10 Hz line)
    alpha_mask = (freqs >= 8) & (freqs <= 12)
    total_power = float(integrate.trapezoid(psd, freqs))
    if total_power > 0:
        band_power = float(integrate.trapezoid(psd[alpha_mask], freqs[alpha_mask]))
        artifact_peaks["alpha_band_power_fraction"] = float(band_power / total_power)
    else:
        artifact_peaks["alpha_band_power_fraction"] = 0.0

    return artifact_peaks


# ---------------------------------------------------------------------------
# Comprehensive per-field statistics
# ---------------------------------------------------------------------------

def compute_field_stats(arr: np.ndarray, field: str) -> Dict[str, Any]:
    """Compute comprehensive statistics for one field."""
    flat = arr.ravel().astype(np.float64)
    n = len(flat)

    q01, q05, q25, q50, q75, q95, q99 = np.percentile(flat, [1, 5, 25, 50, 75, 95, 99])

    stats_dict = {
        "n_samples": int(n),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "var": float(np.var(flat)),
        "median": float(q50),
        "p01": float(q01),
        "p05": float(q05),
        "p25": float(q25),
        "p75": float(q75),
        "p95": float(q95),
        "p99": float(q99),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "skewness": float(stats.skew(flat)),
        "kurtosis": float(stats.kurtosis(flat)),
        "any_nan": bool(np.any(np.isnan(flat))),
        "any_inf": bool(np.any(np.isinf(flat))),
    }

    # Coefficient of variation
    if abs(stats_dict["mean"]) > 1e-12:
        stats_dict["cv"] = float(stats_dict["std"] / abs(stats_dict["mean"]))
    else:
        stats_dict["cv"] = float("inf") if stats_dict["std"] > 1e-12 else 0.0

    # Dynamic range (p99 - p01)
    stats_dict["dynamic_range"] = float(q99 - q01)

    # Zero-crossing rate (for time-domain characterization)
    if n > 1:
        signs = np.sign(flat)
        zero_crossings = np.sum(np.abs(np.diff(signs)) > 0)
        stats_dict["zero_crossing_rate"] = float(zero_crossings / (n - 1))
    else:
        stats_dict["zero_crossing_rate"] = 0.0

    # EEG-specific: stair-step detection
    if field == "source_eeg":
        stair_metrics = detect_stair_stepping(flat)
        stats_dict["stair_step"] = stair_metrics

    return stats_dict


# ---------------------------------------------------------------------------
# Usability verdict
# ---------------------------------------------------------------------------

USABILITY_CRITERIA = {
    "source_eeg": {
        "expected_shape": "temporal EEG signal modulated by neural state estimate (r_eeg)",
        "expected_range": (-300, 300),  # μV
        "expected_std_min": 5.0,
        "expected_std_max": 200.0,
        "critical_checks": ["no_stair_step", "no_nan", "no_inf", "non_zero_variance"],
        "warning_checks": ["std_in_range", "non_zero_mean"],
    },
    "obs_eeg": {
        "expected_shape": "residual EEG after removing physiological component",
        "expected_range": (-300, 300),
        "expected_std_min": 2.0,
        "expected_std_max": 200.0,
        "critical_checks": ["no_nan", "no_inf", "non_zero_variance"],
        "warning_checks": ["std_in_range"],
    },
    "source_fnirs_optical_channel_0": {
        "expected_shape": "fNIRS pathlength factor (nearly constant per channel)",
        "expected_range": (0.1, 2.0),
        "expected_std_max": 0.1,  # Very low variance expected
        "critical_checks": ["no_nan", "no_inf", "non_negative"],
        "warning_checks": ["low_variance_as_expected"],
    },
    "obs_fnirs_optical_channel_0": {
        "expected_shape": "ΔHbO-like residual in optical space",
        "expected_range": (-0.5, 0.5),
        "expected_std_min": 0.001,
        "expected_std_max": 0.5,
        "critical_checks": ["no_nan", "no_inf", "non_zero_variance"],
        "warning_checks": ["std_in_range"],
    },
    "source_fnirs_optical_channel_1": {
        "expected_shape": "fNIRS pathlength factor (nearly constant per channel)",
        "expected_range": (0.1, 2.0),
        "expected_std_max": 0.1,
        "critical_checks": ["no_nan", "no_inf", "non_negative"],
        "warning_checks": ["low_variance_as_expected"],
    },
    "obs_fnirs_optical_channel_1": {
        "expected_shape": "ΔHbR-like residual in optical space",
        "expected_range": (-0.5, 0.5),
        "expected_std_min": 0.001,
        "expected_std_max": 0.5,
        "critical_checks": ["no_nan", "no_inf", "non_zero_variance"],
        "warning_checks": ["std_in_range"],
    },
}


def evaluate_usability(stats_dict: Dict[str, Any], field: str) -> Dict[str, Any]:
    """Evaluate whether a field's statistics indicate usable data."""
    criteria = USABILITY_CRITERIA[field]
    critical_pass = []
    critical_fail = []
    warning_pass = []
    warning_fail = []

    # Critical checks
    if "no_nan" in criteria["critical_checks"]:
        (critical_pass if not stats_dict["any_nan"] else critical_fail).append("no_nan")
    if "no_inf" in criteria["critical_checks"]:
        (critical_pass if not stats_dict["any_inf"] else critical_fail).append("no_inf")
    if "non_zero_variance" in criteria["critical_checks"]:
        (critical_pass if stats_dict["std"] > 1e-12 else critical_fail).append("non_zero_variance")
    if "no_stair_step" in criteria["critical_checks"]:
        stair = stats_dict.get("stair_step", {})
        has_stair = stair.get("stair_step_detected", 0) > 0.5
        (critical_pass if not has_stair else critical_fail).append("no_stair_step")
    if "non_negative" in criteria["critical_checks"]:
        (critical_pass if stats_dict["min"] >= 0 else critical_fail).append("non_negative")

    # Warning checks
    if "std_in_range" in criteria["warning_checks"]:
        lo = criteria.get("expected_std_min", 0)
        hi = criteria.get("expected_std_max", float("inf"))
        (warning_pass if lo <= stats_dict["std"] <= hi else warning_fail).append("std_in_range")
    if "low_variance_as_expected" in criteria["warning_checks"]:
        hi = criteria.get("expected_std_max", float("inf"))
        (warning_pass if stats_dict["std"] <= hi else warning_fail).append("low_variance_as_expected")
    if "non_zero_mean" in criteria["warning_checks"]:
        has_nonzero = abs(stats_dict["mean"]) > 1e-6
        (warning_pass if has_nonzero else warning_fail).append("non_zero_mean")

    usable = len(critical_fail) == 0

    return {
        "usable": usable,
        "critical_pass": critical_pass,
        "critical_fail": critical_fail,
        "warning_pass": warning_pass,
        "warning_fail": warning_fail,
        "verdict": "PASS" if usable and len(warning_fail) == 0 else
                   "PASS_WITH_WARNINGS" if usable else
                   "FAIL",
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_distribution_panel(
    output_path: Path,
    orig_data: Dict[str, np.ndarray],
    regen_data: Dict[str, np.ndarray],
) -> None:
    """Side-by-side distribution comparison: original vs regenerated."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    for idx, field in enumerate(CANONICAL_FIELDS):
        ax = axes[idx // 2, idx % 2]

        for label, data_dict, color, alpha, ls in [
            ("Original", orig_data, FIELD_COLORS[field], 0.5, "-"),
            ("Regenerated", regen_data, FIELD_COLORS[field], 0.7, "--"),
        ]:
            if field in data_dict and len(data_dict[field]) > 0:
                vals = data_dict[field].ravel()
                q01, q99 = np.percentile(vals, [0.1, 99.9])
                clipped = np.clip(vals, q01, q99)

                ax.hist(clipped, bins=80, density=True, alpha=alpha,
                        color=color, edgecolor="#333333", linewidth=0.3,
                        label=label)

                if len(clipped) > 1000:
                    sample = np.random.choice(clipped, size=min(30000, len(clipped)), replace=False)
                    kde = stats.gaussian_kde(sample)
                    x_range = np.linspace(np.min(clipped), np.max(clipped), 200)
                    ax.plot(x_range, kde(x_range), color=color, linewidth=2.0,
                            linestyle=ls)

                # Stats box
                mu, sigma = np.mean(vals), np.std(vals)
                ax.text(0.02, 0.97, f"μ={mu:.3f}\nσ={sigma:.3f}",
                        transform=ax.transAxes, fontsize=8, verticalalignment="top",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_title(FIELD_LABELS[field], fontsize=11, fontweight="bold")
        ax.axvline(0.0, color="#666666", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.grid(alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Target Distribution Comparison: Original vs Regenerated Cache\n(All Subjects, First Event Per Anchor)",
                 fontsize=14, fontweight="bold", y=0.998)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_per_subject_heatmap(
    output_path: Path,
    all_subject_stats: Dict[int, Dict[str, Dict[str, Any]]],
    stat_key: str,
    title: str,
    cmap: str = "RdBu_r",
) -> None:
    """Heatmap of one statistic across subjects × fields."""
    subjects = sorted(all_subject_stats.keys())
    fields = CANONICAL_FIELDS

    data_matrix = np.zeros((len(fields), len(subjects)))
    for fi, field in enumerate(fields):
        for si, sid in enumerate(subjects):
            if field in all_subject_stats[sid]:
                data_matrix[fi, si] = all_subject_stats[sid][field].get(stat_key, np.nan)
            else:
                data_matrix[fi, si] = np.nan

    fig, ax = plt.subplots(figsize=(max(14, len(subjects) * 0.35), 4))
    im = ax.imshow(data_matrix, aspect="auto", cmap=cmap)

    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels([f"S{s}" for s in subjects], fontsize=7, rotation=45)
    ax.set_yticks(range(len(fields)))
    ax.set_yticklabels([FIELD_LABELS[f] for f in fields], fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Annotate cells
    for fi in range(len(fields)):
        for si in range(len(subjects)):
            val = data_matrix[fi, si]
            if not np.isnan(val):
                ax.text(si, fi, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.5, color="black" if abs(val) < np.nanmax(np.abs(data_matrix)) * 0.6 else "white")

    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_eeg_timecourse_comparison(
    output_path: Path,
    orig_event: Optional[Dict[str, np.ndarray]],
    regen_event: Optional[Dict[str, np.ndarray]],
    cache_name_orig: str,
    cache_name_regen: str,
) -> None:
    """Compare EEG source/obs timecourses between original and regenerated."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # Row 0: EEG Source comparison
    ax = axes[0]
    if orig_event and "source_eeg" in orig_event:
        t_orig = np.arange(len(orig_event["source_eeg"])) / EEG_FS
        ax.plot(t_orig, orig_event["source_eeg"], color="#D62728", linewidth=0.8,
                alpha=0.7, label=f"Original ({cache_name_orig})")
    if regen_event and "source_eeg" in regen_event:
        t_regen = np.arange(len(regen_event["source_eeg"])) / EEG_FS
        ax.plot(t_regen, regen_event["source_eeg"], color="#1F77B4", linewidth=0.8,
                alpha=0.7, label=f"Regenerated ({cache_name_regen})")
    ax.set_ylabel("EEG Source (μV)")
    ax.set_title("EEG Source Target Timecourse", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Row 1: EEG Observation comparison
    ax = axes[1]
    if orig_event and "obs_eeg" in orig_event:
        t_orig = np.arange(len(orig_event["obs_eeg"])) / EEG_FS
        ax.plot(t_orig, orig_event["obs_eeg"], color="#FF9896", linewidth=0.8,
                alpha=0.7, label=f"Original ({cache_name_orig})")
    if regen_event and "obs_eeg" in regen_event:
        t_regen = np.arange(len(regen_event["obs_eeg"])) / EEG_FS
        ax.plot(t_regen, regen_event["obs_eeg"], color="#9467BD", linewidth=0.8,
                alpha=0.7, label=f"Regenerated ({cache_name_regen})")
    ax.set_ylabel("EEG Obs (μV)")
    ax.set_title("EEG Observation Target Timecourse", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Row 2: Zoomed difference (first 1 second)
    ax = axes[2]
    zoom_samples = int(EEG_FS * 1.0)  # 1 second
    if orig_event and "source_eeg" in orig_event:
        t_orig = np.arange(min(zoom_samples, len(orig_event["source_eeg"]))) / EEG_FS
        ax.plot(t_orig, orig_event["source_eeg"][:zoom_samples], color="#D62728",
                linewidth=1.2, marker=".", markersize=2, alpha=0.7, label="Original")
    if regen_event and "source_eeg" in regen_event:
        t_regen = np.arange(min(zoom_samples, len(regen_event["source_eeg"]))) / EEG_FS
        ax.plot(t_regen, regen_event["source_eeg"][:zoom_samples], color="#1F77B4",
                linewidth=1.2, marker=".", markersize=2, alpha=0.7, label="Regenerated")
    ax.set_ylabel("EEG Source (μV)")
    ax.set_xlabel("Time (s)")
    ax.set_title("EEG Source — First 1 Second Zoom (check for stair-steps)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle("EEG Target Timecourse: Original vs Regenerated", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_psd_comparison(
    output_path: Path,
    orig_event: Optional[Dict[str, np.ndarray]],
    regen_event: Optional[Dict[str, np.ndarray]],
) -> None:
    """PSD comparison for EEG source targets."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for idx, field in enumerate(["source_eeg", "obs_eeg"]):
        ax = axes[idx]
        for label, event, color, ls in [
            ("Original", orig_event, "#D62728", "-"),
            ("Regenerated", regen_event, "#1F77B4", "--"),
        ]:
            if event and field in event:
                sig = event[field]
                if len(sig) > 4:
                    freqs, psd = compute_psd(sig, EEG_FS)
                    ax.semilogy(freqs, psd, color=color, linewidth=1.2, linestyle=ls,
                                alpha=0.8, label=label)

        # Mark harmonics of fNIRS rate
        for h in [10, 20, 30, 40]:
            ax.axvline(h, color="#999999", linewidth=0.5, linestyle=":", alpha=0.6)
        ax.axvline(10, color="#FF6600", linewidth=0.8, linestyle="--", alpha=0.5,
                   label="fNIRS rate (10 Hz)")

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power Spectral Density")
        ax.set_title(f"PSD: {FIELD_LABELS[field]}", fontweight="bold")
        ax.set_xlim(0, 50)
        ax.grid(alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("EEG Target PSD Comparison: Original vs Regenerated\n(Dashed lines mark fNIRS rate harmonics)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_fnirs_diagnostics(
    output_path: Path,
    orig_event: Optional[Dict[str, np.ndarray]],
    regen_event: Optional[Dict[str, np.ndarray]],
) -> None:
    """fNIRS source/obs comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    plot_specs = [
        (0, 0, "source_fnirs_optical_channel_0", "fNIRS Ch0 Source", "#17BECF"),
        (0, 1, "obs_fnirs_optical_channel_0", "fNIRS Ch0 Obs", "#1F77B4"),
        (1, 0, "source_fnirs_optical_channel_1", "fNIRS Ch1 Source", "#FF9896"),
        (1, 1, "obs_fnirs_optical_channel_1", "fNIRS Ch1 Obs", "#9467BD"),
    ]

    for row, col, field, title, color in plot_specs:
        ax = axes[row, col]
        for label, event, c, ls in [
            ("Original", orig_event, color, "-"),
            ("Regenerated", regen_event, color, "--"),
        ]:
            if event and field in event:
                sig = event[field]
                t = np.arange(len(sig)) / FNIRS_FS
                ax.plot(t, sig, color=c, linewidth=1.0, linestyle=ls,
                        alpha=0.7, label=label)

                if len(sig) > 1:
                    mu, sigma = np.mean(sig), np.std(sig)
                    ax.text(0.02, 0.97,
                            f"μ={mu:.4f}\nσ={sigma:.6f}",
                            transform=ax.transAxes, fontsize=8, verticalalignment="top",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.2)
        if row == 0 and col == 0:
            ax.legend(fontsize=7)

    fig.suptitle("fNIRS Target Diagnostics: Original vs Regenerated",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_usability_summary(
    output_path: Path,
    orig_verdicts: Dict[str, Dict[str, Any]],
    regen_verdicts: Dict[str, Dict[str, Any]],
) -> None:
    """Summary heatmap of usability verdicts."""
    all_fields = CANONICAL_FIELDS
    verdict_map = {"PASS": 2, "PASS_WITH_WARNINGS": 1, "FAIL": 0}

    matrix = np.zeros((2, len(all_fields)))
    for fi, field in enumerate(all_fields):
        matrix[0, fi] = verdict_map.get(orig_verdicts.get(field, {}).get("verdict", "FAIL"), 0)
        matrix[1, fi] = verdict_map.get(regen_verdicts.get(field, {}).get("verdict", "FAIL"), 0)

    fig, ax = plt.subplots(figsize=(10, 3))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=2)

    ax.set_xticks(range(len(all_fields)))
    ax.set_xticklabels([FIELD_LABELS[f] for f in all_fields], fontsize=9, rotation=20)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Original", "Regenerated"], fontsize=10)

    for fi in range(len(all_fields)):
        for ri in range(2):
            val = int(matrix[ri, fi])
            label = ["FAIL", "WARN", "PASS"][val]
            color = "white" if val <= 1 else "black"
            ax.text(fi, ri, label, ha="center", va="center", fontsize=10,
                    fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2], shrink=0.8)
    cbar.set_ticklabels(["FAIL", "WARN", "PASS"])

    ax.set_title("Cache Usability Verdict Summary", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Deep audit of cache usability")
    parser.add_argument("--orig-dir", default="",
                        help="Original cache dir (default: auto-detect)")
    parser.add_argument("--regen-dir", default="",
                        help="Regenerated cache dir (default: auto-detect)")
    parser.add_argument("--output-dir", default="",
                        help="Output directory (default: auto-generated)")
    parser.add_argument("--max-subjects", type=int, default=0,
                        help="Max subjects to analyze (0=all)")
    parser.add_argument("--max-anchors", type=int, default=3,
                        help="Max anchors per subject (0=all)")
    args = parser.parse_args()

    # Auto-detect cache directories
    cache_root = PROJECT_ROOT / "croce_validation" / "cache"
    orig_dir = Path(args.orig_dir) if args.orig_dir else (
        cache_root / "EEG_fNIRS_single_trail_pf_full")
    regen_dir = Path(args.regen_dir) if args.regen_dir else (
        cache_root / "EEG_fNIRS_single_trail_pf_full_mental_arithmetic_regen_20260603_182938")

    if not orig_dir.exists():
        raise FileNotFoundError(f"Original cache not found: {orig_dir}")
    if not regen_dir.exists():
        raise FileNotFoundError(f"Regenerated cache not found: {regen_dir}")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "croce_validation" / "cache" / f"usability_audit_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find subjects
    orig_files = find_cache_files(orig_dir)
    regen_files = find_cache_files(regen_dir)

    # Common subjects
    common_subjects = sorted(set(orig_files.keys()) & set(regen_files.keys()))
    if args.max_subjects > 0:
        common_subjects = common_subjects[:args.max_subjects]

    print(f"Original cache: {orig_dir} ({len(orig_files)} subjects)")
    print(f"Regenerated cache: {regen_dir} ({len(regen_files)} subjects)")
    print(f"Common subjects for comparison: {len(common_subjects)}")
    print(f"Output: {output_dir}\n")

    # ---- Phase 1: Per-subject aggregate statistics ----
    print("=" * 70)
    print("PHASE 1: Per-subject aggregate statistics")
    print("=" * 70)

    orig_all_stats: Dict[int, Dict[str, Dict[str, Any]]] = {}
    regen_all_stats: Dict[int, Dict[str, Dict[str, Any]]] = {}

    for sid in common_subjects:
        print(f"\n  Subject {sid}:")
        try:
            orig_data = load_cache_aggregated(orig_files[sid], args.max_anchors)
            orig_all_stats[sid] = {}
            for field in CANONICAL_FIELDS:
                if field in orig_data:
                    orig_all_stats[sid][field] = compute_field_stats(orig_data[field], field)
            print(f"    Original: {len(orig_data)} fields loaded")

            regen_data = load_cache_aggregated(regen_files[sid], args.max_anchors)
            regen_all_stats[sid] = {}
            for field in CANONICAL_FIELDS:
                if field in regen_data:
                    regen_all_stats[sid][field] = compute_field_stats(regen_data[field], field)
            print(f"    Regenerated: {len(regen_data)} fields loaded")
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ---- Phase 2: EEG stair-step detection ----
    print("\n" + "=" * 70)
    print("PHASE 2: EEG source stair-step artifact detection")
    print("=" * 70)

    stair_results: Dict[int, Dict[str, Dict[str, float]]] = {}

    for sid in common_subjects[:min(5, len(common_subjects))]:  # Check first 5 subjects
        stair_results[sid] = {}
        for label, cache_path in [("original", orig_files[sid]), ("regenerated", regen_files[sid])]:
            event = load_single_event_eeg(cache_path)
            if event and "source_eeg" in event:
                stair_results[sid][label] = detect_stair_stepping(event["source_eeg"])
                detected = stair_results[sid][label]["stair_step_detected"] > 0.5
                print(f"  S{sid} {label:12s}: stair_step_detected={detected}, "
                      f"zero_diff={stair_results[sid][label]['zero_diff_fraction']:.4f}, "
                      f"artifact_fraction={stair_results[sid][label]['artifact_fraction']:.4f}")
            else:
                stair_results[sid][label] = {}
                print(f"  S{sid} {label:12s}: (no EEG data)")

    # ---- Phase 3: PSD analysis ----
    print("\n" + "=" * 70)
    print("PHASE 3: PSD analysis for 10 Hz harmonic artifacts")
    print("=" * 70)

    psd_results: Dict[int, Dict[str, Dict[str, Any]]] = {}

    for sid in common_subjects[:min(3, len(common_subjects))]:
        psd_results[sid] = {}
        for label, cache_path in [("original", orig_files[sid]), ("regenerated", regen_files[sid])]:
            event = load_single_event_eeg(cache_path)
            if event and "source_eeg" in event:
                freqs, psd = compute_psd(event["source_eeg"], EEG_FS)
                artifacts = find_psd_artifacts(freqs, psd)
                psd_results[sid][label] = artifacts
                print(f"  S{sid} {label:12s}: alpha_band_power={artifacts['alpha_band_power_fraction']:.4f}, "
                      f"10Hz_ratio={artifacts['harmonic_10Hz_ratio']:.2f}, "
                      f"20Hz_ratio={artifacts['harmonic_20Hz_ratio']:.2f}")
            else:
                print(f"  S{sid} {label:12s}: (no EEG data)")

    # ---- Phase 4: Generate visualizations ----
    print("\n" + "=" * 70)
    print("PHASE 4: Generating visualizations")
    print("=" * 70)

    # 4a. Distribution panel using first subject
    if common_subjects:
        sid0 = common_subjects[0]
        orig_data = load_cache_aggregated(orig_files[sid0], args.max_anchors)
        regen_data = load_cache_aggregated(regen_files[sid0], args.max_anchors)
        plot_distribution_panel(
            output_dir / "distribution_comparison.png",
            orig_data, regen_data,
        )

    # 4b. Per-subject heatmaps
    for field in CANONICAL_FIELDS:
        if field == "source_eeg" or field == "obs_eeg":
            # Mean
            field_stats_orig = {sid: {field: s[field]} for sid, s in orig_all_stats.items() if field in s}
            field_stats_regen = {sid: {field: s[field]} for sid, s in regen_all_stats.items() if field in s}

    plot_per_subject_heatmap(
        output_dir / "heatmap_source_eeg_std.png",
        orig_all_stats, "std",
        "EEG Source Standard Deviation per Subject (Original Cache)",
    )
    plot_per_subject_heatmap(
        output_dir / "heatmap_obs_eeg_std.png",
        {sid: {f: s[f] for f in CANONICAL_FIELDS if f == "obs_eeg" and f in s}
         for sid, s in orig_all_stats.items()},
        "std",
        "EEG Obs Standard Deviation per Subject (Original Cache)",
    )
    plot_per_subject_heatmap(
        output_dir / "heatmap_eeg_skewness.png",
        {sid: {f: s[f] for f in ["source_eeg", "obs_eeg"] if f in s}
         for sid, s in orig_all_stats.items()},
        "skewness",
        "EEG Skewness per Subject (Original Cache)",
    )

    # 4c. Timecourse and PSD comparisons
    if common_subjects:
        orig_event = load_single_event_eeg(orig_files[sid0])
        regen_event = load_single_event_eeg(regen_files[sid0])
        plot_eeg_timecourse_comparison(
            output_dir / "eeg_timecourse_comparison.png",
            orig_event, regen_event,
            orig_dir.name, regen_dir.name,
        )
        plot_psd_comparison(
            output_dir / "eeg_psd_comparison.png",
            orig_event, regen_event,
        )
        plot_fnirs_diagnostics(
            output_dir / "fnirs_diagnostics.png",
            orig_event, regen_event,
        )

    # ---- Phase 5: Usability evaluation ----
    print("\n" + "=" * 70)
    print("PHASE 5: Usability evaluation")
    print("=" * 70)

    orig_verdicts: Dict[str, Dict[str, Any]] = {}
    regen_verdicts: Dict[str, Dict[str, Any]] = {}

    for field in CANONICAL_FIELDS:
        # Aggregate across subjects for this field
        orig_field_stats_list = []
        for sid in common_subjects:
            if sid in orig_all_stats and field in orig_all_stats[sid]:
                orig_field_stats_list.append(orig_all_stats[sid][field])

        # Average statistics across subjects
        if orig_field_stats_list:
            avg_stats = {}
            for key in orig_field_stats_list[0]:
                val = orig_field_stats_list[0][key]
                if isinstance(val, (int, float, bool, np.bool_, np.integer, np.floating)):
                    avg_stats[key] = float(np.mean([s[key] for s in orig_field_stats_list if key in s]))
                elif key == "stair_step" and any(isinstance(s.get(key), dict) for s in orig_field_stats_list):
                    # Collect all stair_step keys from subjects that have it
                    all_sk = set()
                    for s in orig_field_stats_list:
                        if key in s and isinstance(s[key], dict):
                            all_sk.update(s[key].keys())
                    avg_stair = {}
                    for sk in all_sk:
                        vals = [s.get(key, {}).get(sk, 0) for s in orig_field_stats_list]
                        avg_stair[sk] = float(np.mean(vals))
                    avg_stats[key] = avg_stair
                else:
                    avg_stats[key] = val
            orig_verdicts[field] = evaluate_usability(avg_stats, field)
        else:
            orig_verdicts[field] = {"verdict": "NO_DATA", "usable": False,
                                    "critical_fail": ["no_data"], "warning_fail": []}

        # Same for regenerated
        regen_field_stats_list = []
        for sid in common_subjects:
            if sid in regen_all_stats and field in regen_all_stats[sid]:
                regen_field_stats_list.append(regen_all_stats[sid][field])

        if regen_field_stats_list:
            avg_stats = {}
            for key in regen_field_stats_list[0]:
                val = regen_field_stats_list[0][key]
                if isinstance(val, (int, float, bool, np.bool_, np.integer, np.floating)):
                    avg_stats[key] = float(np.mean([s[key] for s in regen_field_stats_list if key in s]))
                elif key == "stair_step" and any(isinstance(s.get(key), dict) for s in regen_field_stats_list):
                    all_sk = set()
                    for s in regen_field_stats_list:
                        if key in s and isinstance(s[key], dict):
                            all_sk.update(s[key].keys())
                    avg_stair = {}
                    for sk in all_sk:
                        vals = [s.get(key, {}).get(sk, 0) for s in regen_field_stats_list]
                        avg_stair[sk] = float(np.mean(vals))
                    avg_stats[key] = avg_stair
                else:
                    avg_stats[key] = val
            regen_verdicts[field] = evaluate_usability(avg_stats, field)
        else:
            regen_verdicts[field] = {"verdict": "NO_DATA", "usable": False,
                                     "critical_fail": ["no_data"], "warning_fail": []}

        print(f"  {FIELD_LABELS[field]:30s}  "
              f"Original: {orig_verdicts[field]['verdict']:20s}  "
              f"Regenerated: {regen_verdicts[field]['verdict']}")

    # Incorporate stair-step results from Phase 2
    # (Phase 2 stair detection runs on single-event data, avoiding the
    # concatenation artifact that masks stair-steps in aggregated stats)
    orig_has_stair = any(
        stair_results.get(sid, {}).get("original", {}).get("stair_step_detected", 0) > 0.5
        for sid in common_subjects
    )
    regen_has_stair = any(
        stair_results.get(sid, {}).get("regenerated", {}).get("stair_step_detected", 0) > 0.5
        for sid in common_subjects
    )

    if "source_eeg" in orig_verdicts:
        if orig_has_stair:
            orig_verdicts["source_eeg"]["critical_pass"] = [
                c for c in orig_verdicts["source_eeg"].get("critical_pass", [])
                if c != "no_stair_step"
            ]
            orig_verdicts["source_eeg"]["critical_fail"] = (
                orig_verdicts["source_eeg"].get("critical_fail", []) + ["no_stair_step"]
            )
            orig_verdicts["source_eeg"]["usable"] = False
            orig_verdicts["source_eeg"]["verdict"] = "FAIL"

    if "source_eeg" in regen_verdicts:
        if regen_has_stair:
            regen_verdicts["source_eeg"]["critical_pass"] = [
                c for c in regen_verdicts["source_eeg"].get("critical_pass", [])
                if c != "no_stair_step"
            ]
            regen_verdicts["source_eeg"]["critical_fail"] = (
                regen_verdicts["source_eeg"].get("critical_fail", []) + ["no_stair_step"]
            )
            regen_verdicts["source_eeg"]["usable"] = False
            regen_verdicts["source_eeg"]["verdict"] = "FAIL"

    # 5a. Usability summary heatmap
    plot_usability_summary(
        output_dir / "usability_summary.png",
        orig_verdicts, regen_verdicts,
    )

    # ---- Build comprehensive report ----
    report = {
        "generated_at": datetime.now().isoformat(),
        "original_cache": str(orig_dir),
        "regenerated_cache": str(regen_dir),
        "n_subjects_compared": len(common_subjects),
        "subject_ids": common_subjects,
        "stair_step_detection": {
            str(sid): data for sid, data in stair_results.items()
        },
        "psd_analysis": {
            str(sid): data for sid, data in psd_results.items()
        },
        "per_field_usability": {
            "original": orig_verdicts,
            "regenerated": regen_verdicts,
        },
        "aggregate_statistics_original": {
            field: {
                "mean_of_means": float(np.mean([
                    s[field]["mean"] for s in orig_all_stats.values() if field in s
                ])) if any(field in s for s in orig_all_stats.values()) else None,
                "std_of_means": float(np.std([
                    s[field]["mean"] for s in orig_all_stats.values() if field in s
                ])) if any(field in s for s in orig_all_stats.values()) else None,
                "mean_of_stds": float(np.mean([
                    s[field]["std"] for s in orig_all_stats.values() if field in s
                ])) if any(field in s for s in orig_all_stats.values()) else None,
                "mean_of_cv": float(np.mean([
                    s[field]["cv"] for s in orig_all_stats.values()
                    if field in s and np.isfinite(s[field]["cv"])
                ])) if any(field in s for s in orig_all_stats.values()) else None,
            }
            for field in CANONICAL_FIELDS
        },
        "aggregate_statistics_regenerated": {
            field: {
                "mean_of_means": float(np.mean([
                    s[field]["mean"] for s in regen_all_stats.values() if field in s
                ])) if any(field in s for s in regen_all_stats.values()) else None,
                "std_of_means": float(np.std([
                    s[field]["mean"] for s in regen_all_stats.values() if field in s
                ])) if any(field in s for s in regen_all_stats.values()) else None,
                "mean_of_stds": float(np.mean([
                    s[field]["std"] for s in regen_all_stats.values() if field in s
                ])) if any(field in s for s in regen_all_stats.values()) else None,
                "mean_of_cv": float(np.mean([
                    s[field]["cv"] for s in regen_all_stats.values()
                    if field in s and np.isfinite(s[field]["cv"])
                ])) if any(field in s for s in regen_all_stats.values()) else None,
            }
            for field in CANONICAL_FIELDS
        },
    }

    report_path = output_dir / "usability_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n  Report: {report_path}")

    # ---- Final summary ----
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    overall_orig = all(v.get("usable", False) for v in orig_verdicts.values())
    overall_regen = all(v.get("usable", False) for v in regen_verdicts.values())

    print(f"  Original cache overall usable:    {overall_orig}")
    print(f"  Regenerated cache overall usable: {overall_regen}")

    # Detailed per-field
    print("\n  Original cache details:")
    for field in CANONICAL_FIELDS:
        v = orig_verdicts.get(field, {})
        fails = v.get("critical_fail", [])
        warns = v.get("warning_fail", [])
        print(f"    {FIELD_LABELS[field]:30s}  {v.get('verdict', '?'):20s}  "
              f"fails={fails}  warnings={warns}")

    print("\n  Regenerated cache details:")
    for field in CANONICAL_FIELDS:
        v = regen_verdicts.get(field, {})
        fails = v.get("critical_fail", [])
        warns = v.get("warning_fail", [])
        print(f"    {FIELD_LABELS[field]:30s}  {v.get('verdict', '?'):20s}  "
              f"fails={fails}  warnings={warns}")

    print(f"\n  All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
