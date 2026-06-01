"""Comprehensive data quality audit for Croce EEG-fNIRS cache bundles.

The audit covers four slices:
    1. Cache inventory and storage layout consistency.
    2. Signal integrity for cached EEG and fNIRS source/observation targets.
    3. Representative and event-averaged waveform inspection plots.
    4. Problematic-subject flagging based on structural and scale anomalies.

Example:
    python croce_validation/scripts/check_cache_quality.py \
        --cache-dir croce_validation/cache/pf_full \
        --output-dir croce_validation/analysis/pf_full_quality_check
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SIGNAL_FIELDS = [
    "source_eeg",
    "obs_eeg",
    "source_fnirs_optical_channel_0",
    "source_fnirs_optical_channel_1",
    "obs_fnirs_optical_channel_0",
    "obs_fnirs_optical_channel_1",
]

LATENT_FIELDS = ["r_estimates_eeg", "state_estimates"]

EXPECTED_FIELDS = SIGNAL_FIELDS + LATENT_FIELDS

PROBLEMATIC_METRICS = {
    "eeg_source_std_mean": "EEG source std unusually large",
    "eeg_obs_std_mean": "EEG residual std unusually large",
    "fnirs_obs0_std_mean": "highWL residual std unusually large",
    "fnirs_obs1_std_mean": "lowWL residual std unusually large",
}

DELAY_ANALYSIS_NOTE = (
    "Cross-modal delay analysis is intentionally disabled for this cache audit because the current "
    "event windows are long and post-event EEG/fNIRS responses often contain multiple peaks, so a "
    "single peak-lag summary is not treated as meaningful."
)

WAVEFORM_COLORS = {
    "source_eeg_envelope": "#D62728",
    "obs_eeg_envelope": "#1F77B4",
    "source_fnirs_optical_channel_0": "#17BECF",
    "source_fnirs_optical_channel_1": "#FF9896",
    "obs_fnirs_optical_channel_0": "#2CA02C",
    "obs_fnirs_optical_channel_1": "#9467BD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a comprehensive QC audit on a Croce cache directory.")
    parser.add_argument("--cache-dir", required=True, help="Directory containing subject_<id>/ subdirectories.")
    parser.add_argument("--subject-ids", default="", help="Optional comma-separated subject IDs.")
    parser.add_argument("--output-dir", default="", help="Where plots and reports are written.")
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write JSON/Markdown only, without plots.",
    )
    return parser.parse_args()


def parse_subject_ids(spec: str) -> Optional[List[int]]:
    items = [item.strip() for item in str(spec).split(",") if item.strip()]
    if not items:
        return None
    return sorted({int(item) for item in items})


def resolve_output_dir(spec: str, cache_dir: Path) -> Path:
    if spec:
        output_dir = Path(spec)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
    else:
        output_dir = cache_dir / "quality_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_subject_id(subject_dir: Path) -> int:
    return int(subject_dir.name.replace("subject_", ""))


def find_subject_dirs(cache_dir: Path, subject_ids: Optional[Sequence[int]]) -> List[Path]:
    selected = set(subject_ids or [])
    subject_dirs: List[Path] = []
    for subject_dir in sorted(cache_dir.glob("subject_*"), key=lambda path: extract_subject_id(path)):
        subject_id = extract_subject_id(subject_dir)
        if selected and subject_id not in selected:
            continue
        cache_path = subject_dir / f"subject{subject_id}_cache.npz"
        manifest_path = subject_dir / "cache_manifest.json"
        if cache_path.exists() and manifest_path.exists():
            subject_dirs.append(subject_dir)
    return subject_dirs


def extract_job_prefixes(keys: Iterable[str]) -> List[str]:
    prefixes = set()
    for key in keys:
        parts = key.split("/")
        if len(parts) == 3:
            prefixes.add(f"{parts[0]}/{parts[1]}")
        elif len(parts) == 2:
            prefixes.add(parts[0])
    return sorted(prefixes)


def parse_event_idx(prefix: str) -> Optional[int]:
    parts = prefix.split("/")
    if len(parts) < 2 or not parts[1].startswith("event_"):
        return None
    return int(parts[1].replace("event_", ""))


def event_meta_map(manifest: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    mapping: Dict[int, Dict[str, Any]] = {}
    for entry in manifest.get("event_windows_selected", []):
        idx = entry.get("event_idx")
        if idx is None:
            continue
        mapping[int(idx)] = dict(entry)
    return mapping


def rms_envelope(matrix: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(matrix), axis=1, dtype=np.float64))


def downsample_mean(series: np.ndarray, target_len: int) -> np.ndarray:
    series = np.asarray(series, dtype=np.float64).reshape(-1)
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if series.shape[0] == target_len:
        return series.copy()
    if series.shape[0] % target_len == 0:
        factor = series.shape[0] // target_len
        return series.reshape(target_len, factor).mean(axis=1)
    src_x = np.linspace(0.0, 1.0, num=series.shape[0], endpoint=False)
    dst_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(dst_x, src_x, series)


def safe_zscore(series: np.ndarray) -> np.ndarray:
    series = np.asarray(series, dtype=np.float64).reshape(-1)
    mu = float(np.mean(series))
    sigma = float(np.std(series))
    if not np.isfinite(sigma) or sigma < 1e-12:
        return np.zeros_like(series)
    return (series - mu) / sigma


def baseline_zscore(series: np.ndarray, baseline_mask: np.ndarray) -> np.ndarray:
    series = np.asarray(series, dtype=np.float64).reshape(-1)
    if baseline_mask.shape[0] != series.shape[0]:
        raise ValueError("baseline mask must match series length")
    if not np.any(baseline_mask):
        return safe_zscore(series)
    baseline = series[baseline_mask]
    mu = float(np.mean(baseline))
    sigma = float(np.std(baseline))
    if not np.isfinite(sigma) or sigma < 1e-12:
        sigma = float(np.std(series))
    if not np.isfinite(sigma) or sigma < 1e-12:
        return np.zeros_like(series)
    return (series - mu) / sigma


def time_axis_from_manifest(manifest: Mapping[str, Any], num_samples: int, field: str) -> np.ndarray:
    config = manifest.get("config", {})
    event_windows = manifest.get("event_windows_selected", [])

    start_s = 0.0
    end_s = float(config.get("segment_duration_s", 0.0))
    if event_windows:
        start_s = float(event_windows[0].get("aligned_window_start_s", -float(config.get("event_window_pre_s", 0.0))))
        end_s = float(event_windows[0].get("aligned_window_end_s", float(config.get("event_window_post_s", 0.0))))
    elif float(config.get("segment_duration_s", 0.0)) > 0.0:
        start_s = float(config.get("segment_start_s", 0.0))
        end_s = start_s + float(config.get("segment_duration_s", 0.0))

    if field.startswith("source_eeg") or field.startswith("obs_eeg"):
        fs_hz = float(config.get("eeg_fs_hz", 0.0))
    else:
        fs_hz = float(config.get("fnirs_fs_hz", 0.0))

    if fs_hz > 0.0:
        duration_s = float(num_samples) / fs_hz
        return start_s + np.arange(num_samples, dtype=np.float64) / fs_hz

    duration_s = end_s - start_s
    if duration_s <= 0.0:
        return np.arange(num_samples, dtype=np.float64)
    dt_s = duration_s / float(num_samples)
    return start_s + np.arange(num_samples, dtype=np.float64) * dt_s


def expected_shape_map(
    manifest: Mapping[str, Any],
    prefix: str,
    assembly_meta: Mapping[str, Any],
) -> Dict[str, Tuple[int, ...]]:
    config = manifest.get("config", {})
    duration_s = float(config.get("event_window_duration_s") or config.get("segment_duration_s") or 0.0)
    eeg_fs = float(config.get("eeg_fs_hz") or 0.0)
    fnirs_fs = float(config.get("fnirs_fs_hz") or 0.0)
    eeg_samples = int(round(duration_s * eeg_fs)) if duration_s > 0.0 and eeg_fs > 0.0 else 0
    fnirs_samples = int(round(duration_s * fnirs_fs)) if duration_s > 0.0 and fnirs_fs > 0.0 else 0
    local_eeg_names = list(assembly_meta.get(prefix, {}).get("local_eeg_channel_names", []))
    n_eeg_channels = len(local_eeg_names) if local_eeg_names else 0
    return {
        "source_eeg": (eeg_samples, n_eeg_channels) if eeg_samples and n_eeg_channels else tuple(),
        "obs_eeg": (eeg_samples, n_eeg_channels) if eeg_samples and n_eeg_channels else tuple(),
        "source_fnirs_optical_channel_0": (fnirs_samples, 1) if fnirs_samples else tuple(),
        "source_fnirs_optical_channel_1": (fnirs_samples, 1) if fnirs_samples else tuple(),
        "obs_fnirs_optical_channel_0": (fnirs_samples, 1) if fnirs_samples else tuple(),
        "obs_fnirs_optical_channel_1": (fnirs_samples, 1) if fnirs_samples else tuple(),
        "r_estimates_eeg": (eeg_samples,) if eeg_samples else tuple(),
        "state_estimates": (fnirs_samples, 5) if fnirs_samples else tuple(),
    }


def percentile_summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def numeric_summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def detect_subject_outliers(rows: Sequence[Mapping[str, Any]], key: str, z_thresh: float = 2.5) -> List[Dict[str, Any]]:
    values = [float(row[key]) for row in rows if key in row and row[key] is not None]
    if len(values) < 3:
        return []
    mu = float(np.mean(values))
    sigma = float(np.std(values))
    if sigma < 1e-12:
        return []
    outliers: List[Dict[str, Any]] = []
    for row in rows:
        if key not in row or row[key] is None:
            continue
        z = (float(row[key]) - mu) / sigma
        if abs(z) >= z_thresh:
            outliers.append({
                "subject_id": int(row["subject_id"]),
                "metric": key,
                "value": float(row[key]),
                "z_score": float(z),
            })
    return sorted(outliers, key=lambda item: abs(item["z_score"]), reverse=True)


def format_float(value: Optional[float], digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def collect_problematic_subjects(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    medians: Dict[str, Optional[float]] = {}
    for key in PROBLEMATIC_METRICS:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        medians[key] = float(np.median(values)) if values else None

    flagged: Dict[int, Dict[str, Any]] = {}

    for metric_key, description in PROBLEMATIC_METRICS.items():
        for item in detect_subject_outliers(rows, metric_key):
            subject_id = int(item["subject_id"])
            reference = medians.get(metric_key)
            value = float(item["value"])
            ratio = None
            if reference is not None and np.isfinite(reference) and abs(reference) > 1e-12:
                ratio = value / reference
            entry = flagged.setdefault(
                subject_id,
                {
                    "subject_id": subject_id,
                    "cache_size_mb": None,
                    "jobs_expected": None,
                    "jobs_actual": None,
                    "event_label_mismatch_count": None,
                    "reasons": [],
                },
            )
            entry["reasons"].append(
                {
                    "type": "scale_outlier",
                    "metric": metric_key,
                    "description": description,
                    "value": value,
                    "z_score": float(item["z_score"]),
                    "median_reference": reference,
                    "ratio_to_median": ratio,
                }
            )

    for row in rows:
        subject_id = int(row["subject_id"])
        structural_reasons: List[Dict[str, Any]] = []
        if row.get("jobs_expected") is not None and row.get("jobs_actual") is not None:
            if int(row["jobs_expected"]) != int(row["jobs_actual"]):
                structural_reasons.append(
                    {
                        "type": "structural",
                        "metric": "jobs_actual",
                        "description": "Manifest job count does not match actual cache jobs",
                        "value": float(row["jobs_actual"]),
                        "expected": int(row["jobs_expected"]),
                    }
                )
        if int(row.get("event_label_mismatch_count") or 0) > 0:
            structural_reasons.append(
                {
                    "type": "structural",
                    "metric": "event_label_mismatch_count",
                    "description": "Event label index mismatch present in manifest",
                    "value": float(row["event_label_mismatch_count"]),
                }
            )
        if structural_reasons:
            entry = flagged.setdefault(
                subject_id,
                {
                    "subject_id": subject_id,
                    "cache_size_mb": None,
                    "jobs_expected": None,
                    "jobs_actual": None,
                    "event_label_mismatch_count": None,
                    "reasons": [],
                },
            )
            entry["reasons"].extend(structural_reasons)

        if subject_id in flagged:
            flagged[subject_id]["cache_size_mb"] = row.get("cache_size_mb")
            flagged[subject_id]["jobs_expected"] = row.get("jobs_expected")
            flagged[subject_id]["jobs_actual"] = row.get("jobs_actual")
            flagged[subject_id]["event_label_mismatch_count"] = row.get("event_label_mismatch_count")

    for entry in flagged.values():
        entry["reason_count"] = len(entry["reasons"])
        z_scores = [abs(float(reason["z_score"])) for reason in entry["reasons"] if reason.get("z_score") is not None]
        entry["max_abs_z_score"] = max(z_scores) if z_scores else 0.0

    return sorted(
        flagged.values(),
        key=lambda item: (-float(item["max_abs_z_score"]), -int(item["reason_count"]), int(item["subject_id"])),
    )


@dataclass
class RunningSeries:
    total: Optional[np.ndarray] = None
    count: int = 0

    def add(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64)
        if self.total is None:
            self.total = np.zeros_like(arr, dtype=np.float64)
        if self.total.shape != arr.shape:
            raise ValueError("running series shape mismatch")
        self.total += arr
        self.count += 1

    def mean(self) -> Optional[np.ndarray]:
        if self.total is None or self.count == 0:
            return None
        return self.total / float(self.count)


@dataclass
class RepresentativeJob:
    subject_id: int
    prefix: str
    eeg_time_s: np.ndarray
    fnirs_time_s: np.ndarray
    source_eeg: np.ndarray
    obs_eeg: np.ndarray
    source_fnirs_optical_channel_0: np.ndarray
    source_fnirs_optical_channel_1: np.ndarray
    obs_fnirs_optical_channel_0: np.ndarray
    obs_fnirs_optical_channel_1: np.ndarray
    meta: Dict[str, Any]


class CacheQualityAudit:
    def __init__(self, cache_dir: Path, subject_dirs: Sequence[Path]) -> None:
        self.cache_dir = cache_dir
        self.subject_dirs = list(subject_dirs)

        self.storage_counts: MutableMapping[str, int] = defaultdict(int)
        self.total_cache_size_mb = 0.0
        self.subject_rows: List[Dict[str, Any]] = []
        self.field_stats: Dict[str, Dict[str, Any]] = {
            field: {
                "mean": [],
                "std": [],
                "min": [],
                "max": [],
                "nonfinite_jobs": 0,
                "flat_jobs": 0,
                "flat_examples": [],
                "shapes": defaultdict(int),
            }
            for field in SIGNAL_FIELDS
        }
        self.latent_shape_samples: Dict[str, Dict[str, int]] = {field: defaultdict(int) for field in LATENT_FIELDS}
        self.integrity: Dict[str, Any] = {
            "missing_field_examples": [],
            "shape_mismatch_examples": [],
            "latent_shape_mismatch_examples": [],
            "label_index_mismatch_events": 0,
            "subjects_with_missing_jobs": 0,
            "nonfinite_signal_jobs": 0,
            "flat_signal_jobs": 0,
        }
        self.metadata_stats: Dict[str, List[float]] = {
            "raw_event_offset_s": [],
            "aligned_window_start_s": [],
            "aligned_window_end_s": [],
        }
        self.waveform_accum: Dict[str, RunningSeries] = {
            "source_eeg_envelope": RunningSeries(),
            "obs_eeg_envelope": RunningSeries(),
            "source_fnirs_optical_channel_0": RunningSeries(),
            "source_fnirs_optical_channel_1": RunningSeries(),
            "obs_fnirs_optical_channel_0": RunningSeries(),
            "obs_fnirs_optical_channel_1": RunningSeries(),
        }
        self.representative_job: Optional[RepresentativeJob] = None
        self.fnirs_time_axis_s: Optional[np.ndarray] = None
        self.jobs_processed = 0

    def run(self) -> Dict[str, Any]:
        for subject_dir in self.subject_dirs:
            self._process_subject(subject_dir)

        averaged_waveforms = {
            key: (series.mean().tolist() if series.mean() is not None else None)
            for key, series in self.waveform_accum.items()
        }
        problematic_subjects = collect_problematic_subjects(self.subject_rows)

        results = {
            "cache_dir": str(self.cache_dir),
            "subjects_analyzed": len(self.subject_rows),
            "jobs_processed": int(self.jobs_processed),
            "analysis_scope": {
                "cross_modal_delay_analysis": False,
                "delay_analysis_note": DELAY_ANALYSIS_NOTE,
            },
            "storage": {
                "total_cache_size_mb": round(self.total_cache_size_mb, 2),
                "artifact_counts": dict(sorted(self.storage_counts.items())),
                "per_subject": self.subject_rows,
            },
            "metadata": {
                "raw_event_offset_s": numeric_summary(self.metadata_stats["raw_event_offset_s"]),
                "aligned_window_start_s": numeric_summary(self.metadata_stats["aligned_window_start_s"]),
                "aligned_window_end_s": numeric_summary(self.metadata_stats["aligned_window_end_s"]),
            },
            "integrity": self.integrity,
            "signal_quality": {
                field: {
                    "mean": percentile_summary(stats["mean"]),
                    "std": percentile_summary(stats["std"]),
                    "min": percentile_summary(stats["min"]),
                    "max": percentile_summary(stats["max"]),
                    "nonfinite_jobs": int(stats["nonfinite_jobs"]),
                    "flat_jobs": int(stats["flat_jobs"]),
                    "flat_examples": list(stats["flat_examples"]),
                    "shapes": {str(shape): int(count) for shape, count in sorted(stats["shapes"].items())},
                }
                for field, stats in self.field_stats.items()
            },
            "latent_samples": {
                field: {str(shape): int(count) for shape, count in sorted(shape_counts.items())}
                for field, shape_counts in self.latent_shape_samples.items()
            },
            "averaged_waveforms": {
                "time_s": (self.fnirs_time_axis_s.tolist() if self.fnirs_time_axis_s is not None else None),
                "series": averaged_waveforms,
            },
            "problematic_subjects": problematic_subjects,
            "representative_job": self._representative_job_summary(),
        }
        return results

    def _representative_job_summary(self) -> Optional[Dict[str, Any]]:
        if self.representative_job is None:
            return None
        return {
            "subject_id": int(self.representative_job.subject_id),
            "prefix": self.representative_job.prefix,
            "meta": self.representative_job.meta,
            "eeg_shape": list(self.representative_job.source_eeg.shape),
            "fnirs_shape": list(self.representative_job.source_fnirs_optical_channel_0.shape),
        }

    def _process_subject(self, subject_dir: Path) -> None:
        subject_id = extract_subject_id(subject_dir)
        cache_path = subject_dir / f"subject{subject_id}_cache.npz"
        manifest_path = subject_dir / "cache_manifest.json"
        manifest = load_json(manifest_path)
        event_map = event_meta_map(manifest)
        assembly_meta = manifest.get("assembly_meta", {})
        cache_size_mb = cache_path.stat().st_size / (1024.0 * 1024.0)
        self.total_cache_size_mb += cache_size_mb

        subject_files = list(subject_dir.iterdir())
        heatmaps = list(subject_dir.glob("spatial_heatmap_event_*.png"))
        variances = list(subject_dir.glob("spatial_variance_event_*.png"))
        self.storage_counts["cache_manifest.json"] += int(manifest_path.exists())
        self.storage_counts["subject_cache.npz"] += int(cache_path.exists())
        self.storage_counts["global_targets.npz"] += int((subject_dir / "global_targets.npz").exists())
        self.storage_counts["global_targets.json"] += int((subject_dir / "global_targets.json").exists())
        self.storage_counts["spatial_heatmap_png"] += len(heatmaps)
        self.storage_counts["spatial_variance_png"] += len(variances)
        self.storage_counts["other_artifacts"] += max(len(subject_files) - 4 - len(heatmaps) - len(variances), 0)

        config = manifest.get("config", {})
        expected_jobs = int(manifest.get("jobs_processed", 0) or 0)
        events_selected = manifest.get("event_windows_selected", [])
        for window in events_selected:
            if not bool(window.get("label_index_match", False)):
                self.integrity["label_index_mismatch_events"] += 1
            for key in self.metadata_stats:
                value = window.get(key)
                if value is not None:
                    self.metadata_stats[key].append(float(value))

        subject_metrics: Dict[str, List[float]] = defaultdict(list)

        with np.load(cache_path, allow_pickle=False) as data:
            keys = list(data.keys())
            prefixes = extract_job_prefixes(keys)
            key_set = set(keys)
            if expected_jobs and len(prefixes) != expected_jobs:
                self.integrity["subjects_with_missing_jobs"] += 1

            latent_sampled = False
            for prefix in prefixes:
                expected_shapes = expected_shape_map(manifest, prefix, assembly_meta)
                missing_fields = [field for field in EXPECTED_FIELDS if f"{prefix}/{field}" not in key_set]
                if missing_fields:
                    self.integrity["missing_field_examples"].append(
                        {"subject_id": subject_id, "prefix": prefix, "missing_fields": missing_fields[:8]}
                    )
                    continue

                signal_arrays = {
                    field: np.asarray(data[f"{prefix}/{field}"], dtype=np.float64)
                    for field in SIGNAL_FIELDS
                }

                if not latent_sampled:
                    latent_sampled = True
                    for field in LATENT_FIELDS:
                        arr = np.asarray(data[f"{prefix}/{field}"], dtype=np.float64)
                        self.latent_shape_samples[field][str(tuple(arr.shape))] += 1
                        expected = expected_shapes.get(field, tuple())
                        if expected and tuple(arr.shape) != expected:
                            self.integrity["latent_shape_mismatch_examples"].append(
                                {
                                    "subject_id": subject_id,
                                    "prefix": prefix,
                                    "field": field,
                                    "expected": list(expected),
                                    "actual": list(arr.shape),
                                }
                            )

                has_nonfinite = False
                has_flat = False
                for field, arr in signal_arrays.items():
                    stats = self.field_stats[field]
                    stats["shapes"][str(tuple(arr.shape))] += 1
                    expected = expected_shapes.get(field, tuple())
                    if expected and tuple(arr.shape) != expected:
                        self.integrity["shape_mismatch_examples"].append(
                            {
                                "subject_id": subject_id,
                                "prefix": prefix,
                                "field": field,
                                "expected": list(expected),
                                "actual": list(arr.shape),
                            }
                        )

                    if not np.all(np.isfinite(arr)):
                        stats["nonfinite_jobs"] += 1
                        has_nonfinite = True

                    arr_std = float(np.std(arr))
                    stats["mean"].append(float(np.mean(arr)))
                    stats["std"].append(arr_std)
                    stats["min"].append(float(np.min(arr)))
                    stats["max"].append(float(np.max(arr)))

                    flat_threshold = 1e-4 if field.startswith("source_eeg") or field.startswith("obs_eeg") else 1e-7
                    if arr_std < flat_threshold:
                        stats["flat_jobs"] += 1
                        if len(stats["flat_examples"]) < 10:
                            stats["flat_examples"].append(f"subject {subject_id} {prefix}")
                        has_flat = True

                if has_nonfinite:
                    self.integrity["nonfinite_signal_jobs"] += 1
                if has_flat:
                    self.integrity["flat_signal_jobs"] += 1

                fnirs_len = signal_arrays["source_fnirs_optical_channel_0"].shape[0]
                fnirs_time_s = time_axis_from_manifest(manifest, fnirs_len, "source_fnirs_optical_channel_0")
                if self.fnirs_time_axis_s is None:
                    self.fnirs_time_axis_s = fnirs_time_s

                eeg_source_env = downsample_mean(rms_envelope(signal_arrays["source_eeg"]), fnirs_len)
                eeg_obs_env = downsample_mean(rms_envelope(signal_arrays["obs_eeg"]), fnirs_len)
                derived_series = {
                    "source_eeg_envelope": eeg_source_env,
                    "obs_eeg_envelope": eeg_obs_env,
                    "source_fnirs_optical_channel_0": signal_arrays["source_fnirs_optical_channel_0"].reshape(-1),
                    "source_fnirs_optical_channel_1": signal_arrays["source_fnirs_optical_channel_1"].reshape(-1),
                    "obs_fnirs_optical_channel_0": signal_arrays["obs_fnirs_optical_channel_0"].reshape(-1),
                    "obs_fnirs_optical_channel_1": signal_arrays["obs_fnirs_optical_channel_1"].reshape(-1),
                }
                baseline_mask = fnirs_time_s < 0.0
                for key, series in derived_series.items():
                    self.waveform_accum[key].add(baseline_zscore(series, baseline_mask))

                subject_metrics["eeg_source_std"].append(float(np.std(signal_arrays["source_eeg"])))
                subject_metrics["eeg_obs_std"].append(float(np.std(signal_arrays["obs_eeg"])))
                subject_metrics["fnirs_source0_std"].append(float(np.std(signal_arrays["source_fnirs_optical_channel_0"])))
                subject_metrics["fnirs_source1_std"].append(float(np.std(signal_arrays["source_fnirs_optical_channel_1"])))
                subject_metrics["fnirs_obs0_std"].append(float(np.std(signal_arrays["obs_fnirs_optical_channel_0"])))
                subject_metrics["fnirs_obs1_std"].append(float(np.std(signal_arrays["obs_fnirs_optical_channel_1"])))
                self.jobs_processed += 1

                if self.representative_job is None:
                    event_idx = parse_event_idx(prefix)
                    representative_meta = dict(event_map.get(event_idx or -1, {}))
                    representative_meta.update(assembly_meta.get(prefix, {}))
                    eeg_time_s = time_axis_from_manifest(manifest, signal_arrays["source_eeg"].shape[0], "source_eeg")
                    self.representative_job = RepresentativeJob(
                        subject_id=subject_id,
                        prefix=prefix,
                        eeg_time_s=eeg_time_s,
                        fnirs_time_s=fnirs_time_s,
                        source_eeg=signal_arrays["source_eeg"].copy(),
                        obs_eeg=signal_arrays["obs_eeg"].copy(),
                        source_fnirs_optical_channel_0=signal_arrays["source_fnirs_optical_channel_0"].copy(),
                        source_fnirs_optical_channel_1=signal_arrays["source_fnirs_optical_channel_1"].copy(),
                        obs_fnirs_optical_channel_0=signal_arrays["obs_fnirs_optical_channel_0"].copy(),
                        obs_fnirs_optical_channel_1=signal_arrays["obs_fnirs_optical_channel_1"].copy(),
                        meta=representative_meta,
                    )

        row = {
            "subject_id": subject_id,
            "cache_size_mb": round(cache_size_mb, 2),
            "jobs_expected": expected_jobs,
            "jobs_actual": len(prefixes),
            "anchors_processed": int(manifest.get("anchors_processed", 0)),
            "events_processed": int(manifest.get("events_processed", 0)),
            "event_label_mismatch_count": sum(1 for item in events_selected if not bool(item.get("label_index_match", False))),
            "raw_event_offset_mean_s": float(np.mean([float(item.get("raw_event_offset_s", np.nan)) for item in events_selected])) if events_selected else None,
            "eeg_source_std_mean": float(np.mean(subject_metrics["eeg_source_std"])) if subject_metrics["eeg_source_std"] else None,
            "eeg_obs_std_mean": float(np.mean(subject_metrics["eeg_obs_std"])) if subject_metrics["eeg_obs_std"] else None,
            "fnirs_source0_std_mean": float(np.mean(subject_metrics["fnirs_source0_std"])) if subject_metrics["fnirs_source0_std"] else None,
            "fnirs_source1_std_mean": float(np.mean(subject_metrics["fnirs_source1_std"])) if subject_metrics["fnirs_source1_std"] else None,
            "fnirs_obs0_std_mean": float(np.mean(subject_metrics["fnirs_obs0_std"])) if subject_metrics["fnirs_obs0_std"] else None,
            "fnirs_obs1_std_mean": float(np.mean(subject_metrics["fnirs_obs1_std"])) if subject_metrics["fnirs_obs1_std"] else None,
        }
        self.subject_rows.append(row)


def plot_subject_overview(results: Mapping[str, Any], output_path: Path) -> None:
    rows = list(results["storage"]["per_subject"])
    if not rows:
        return
    subject_ids = [int(row["subject_id"]) for row in rows]
    cache_sizes = [float(row["cache_size_mb"]) for row in rows]
    jobs_actual = [int(row["jobs_actual"]) for row in rows]
    eeg_std = [float(row["eeg_source_std_mean"] or 0.0) for row in rows]
    fnirs_obs_std = [float(row["fnirs_obs0_std_mean"] or 0.0) for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    axes = axes.ravel()
    axes[0].bar(subject_ids, cache_sizes, color="#1F77B4")
    axes[0].set_title("Per-subject cache size")
    axes[0].set_xlabel("Subject")
    axes[0].set_ylabel("MB")

    axes[1].bar(subject_ids, jobs_actual, color="#FF7F0E")
    axes[1].set_title("Jobs per subject")
    axes[1].set_xlabel("Subject")
    axes[1].set_ylabel("Anchor-event jobs")

    axes[2].plot(subject_ids, eeg_std, marker="o", color="#D62728")
    axes[2].set_title("Mean EEG source std by subject")
    axes[2].set_xlabel("Subject")
    axes[2].set_ylabel("Std")

    axes[3].plot(subject_ids, fnirs_obs_std, marker="o", color="#2CA02C")
    axes[3].set_title("Mean highWL residual std by subject")
    axes[3].set_xlabel("Subject")
    axes[3].set_ylabel("Std")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_waveform_overview(results: Mapping[str, Any], audit: CacheQualityAudit, output_path: Path) -> None:
    representative = audit.representative_job
    averaged = results["averaged_waveforms"]
    if representative is None or averaged["time_s"] is None:
        return

    avg_time_s = np.asarray(averaged["time_s"], dtype=np.float64)
    avg_series = {key: np.asarray(values, dtype=np.float64) for key, values in averaged["series"].items() if values is not None}

    fig, axes = plt.subplots(3, 2, figsize=(16, 11), constrained_layout=True)
    axes = axes.ravel()

    axes[0].plot(representative.eeg_time_s, representative.source_eeg, linewidth=0.8, alpha=0.75)
    axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_title("Representative source EEG channels")
    axes[0].set_xlabel("Aligned time (s)")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(representative.eeg_time_s, representative.obs_eeg, linewidth=0.8, alpha=0.75)
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title("Representative observation EEG channels")
    axes[1].set_xlabel("Aligned time (s)")
    axes[1].set_ylabel("Residual amplitude")

    axes[2].plot(representative.fnirs_time_s, representative.source_fnirs_optical_channel_0, color="#17BECF", label="highWL")
    axes[2].plot(representative.fnirs_time_s, representative.source_fnirs_optical_channel_1, color="#FF9896", label="lowWL")
    axes[2].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[2].set_title("Representative fNIRS source traces")
    axes[2].set_xlabel("Aligned time (s)")
    axes[2].set_ylabel("V")
    axes[2].legend(frameon=False)

    axes[3].plot(representative.fnirs_time_s, representative.obs_fnirs_optical_channel_0, color="#2CA02C", label="highWL residual")
    axes[3].plot(representative.fnirs_time_s, representative.obs_fnirs_optical_channel_1, color="#9467BD", label="lowWL residual")
    axes[3].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[3].set_title("Representative fNIRS residual traces")
    axes[3].set_xlabel("Aligned time (s)")
    axes[3].set_ylabel("Residual V")
    axes[3].legend(frameon=False)

    for key in ("source_eeg_envelope", "source_fnirs_optical_channel_0", "source_fnirs_optical_channel_1"):
        if key in avg_series:
            axes[4].plot(avg_time_s, avg_series[key], label=key, color=WAVEFORM_COLORS[key], linewidth=2.0)
    axes[4].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[4].set_title("Event-averaged source waveforms (baseline z-score)")
    axes[4].set_xlabel("Aligned time (s)")
    axes[4].set_ylabel("Baseline z-score")
    axes[4].legend(frameon=False)

    for key in ("obs_eeg_envelope", "obs_fnirs_optical_channel_0", "obs_fnirs_optical_channel_1"):
        if key in avg_series:
            axes[5].plot(avg_time_s, avg_series[key], label=key, color=WAVEFORM_COLORS[key], linewidth=2.0)
    axes[5].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[5].set_title("Event-averaged residual waveforms (baseline z-score)")
    axes[5].set_xlabel("Aligned time (s)")
    axes[5].set_ylabel("Baseline z-score")
    axes[5].legend(frameon=False)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_markdown_report(results: Mapping[str, Any], plot_files: Sequence[Path]) -> str:
    analysis_scope = results["analysis_scope"]
    storage = results["storage"]
    metadata = results["metadata"]
    integrity = results["integrity"]
    averaged = results["averaged_waveforms"]
    representative = results["representative_job"]
    signal_quality = results["signal_quality"]
    problematic_subjects = results["problematic_subjects"]

    lines = [
        "# pf_full cache data quality audit",
        "",
        "## Scope",
        f"- Cache directory: {results['cache_dir']}",
        f"- Subjects analyzed: {results['subjects_analyzed']}",
        f"- Anchor-event jobs analyzed: {results['jobs_processed']}",
        f"- Total subject cache size: {format_float(storage['total_cache_size_mb'], 2)} MB",
        "- fNIRS semantics: EEG+NIRS Single-Trial optical wavelength pairs highWL/lowWL in V, not HbO/HbR concentration traces.",
        "",
        "## Storage layout",
        f"- Per-subject artifacts: {storage['artifact_counts']}",
        f"- Median subject cache size: {format_float(median([row['cache_size_mb'] for row in storage['per_subject']]), 2)} MB",
        f"- Median jobs per subject: {format_float(median([row['jobs_actual'] for row in storage['per_subject']]), 1)}",
        f"- Representative cache entry: subject {representative['subject_id'] if representative else 'n/a'} {representative['prefix'] if representative else 'n/a'}",
        "",
        "## Integrity findings",
        f"- Missing signal field examples: {len(integrity['missing_field_examples'])}",
        f"- Shape mismatch examples: {len(integrity['shape_mismatch_examples'])}",
        f"- Latent shape mismatch examples: {len(integrity['latent_shape_mismatch_examples'])}",
        f"- Signal jobs with non-finite values: {integrity['nonfinite_signal_jobs']}",
        f"- Signal jobs flagged near-flat: {integrity['flat_signal_jobs']}",
        f"- Subjects with manifest/job-count mismatch: {integrity['subjects_with_missing_jobs']}",
        f"- Event label index mismatches: {integrity['label_index_mismatch_events']}",
        "",
        "## Signal quality snapshot",
        f"- Source EEG std: median {format_float(signal_quality['source_eeg']['std'].get('median'))}, p05-p95 {format_float(signal_quality['source_eeg']['std'].get('p05'))} to {format_float(signal_quality['source_eeg']['std'].get('p95'))}",
        f"- Observation EEG std: median {format_float(signal_quality['obs_eeg']['std'].get('median'))}, p05-p95 {format_float(signal_quality['obs_eeg']['std'].get('p05'))} to {format_float(signal_quality['obs_eeg']['std'].get('p95'))}",
        f"- highWL source std: median {format_float(signal_quality['source_fnirs_optical_channel_0']['std'].get('median'))}, highWL residual std: median {format_float(signal_quality['obs_fnirs_optical_channel_0']['std'].get('median'))}",
        f"- lowWL source std: median {format_float(signal_quality['source_fnirs_optical_channel_1']['std'].get('median'))}, lowWL residual std: median {format_float(signal_quality['obs_fnirs_optical_channel_1']['std'].get('median'))}",
        "",
        "## Window alignment",
        f"- Raw EEG-fNIRS event offset: mean {format_float(metadata['raw_event_offset_s'].get('mean'))} s, std {format_float(metadata['raw_event_offset_s'].get('std'))} s.",
        f"- Aligned event window: {format_float(metadata['aligned_window_start_s'].get('mean'))} s to {format_float(metadata['aligned_window_end_s'].get('mean'))} s.",
        f"- Cross-modal delay analysis enabled: {analysis_scope['cross_modal_delay_analysis']}",
        f"- Note: {analysis_scope['delay_analysis_note']}",
        "- Event-averaged waveforms are kept for qualitative inspection only; no single-delay statistic is reported.",
        "",
        "## Problematic subjects",
    ]

    if problematic_subjects:
        for item in problematic_subjects:
            reason_texts = []
            for reason in item["reasons"]:
                if reason["type"] == "scale_outlier":
                    ratio_text = ""
                    if reason.get("ratio_to_median") is not None:
                        ratio_text = f", ratio_to_median={format_float(reason['ratio_to_median'])}x"
                    reason_texts.append(
                        f"{reason['metric']}={format_float(reason['value'])}, z={format_float(reason.get('z_score'))}{ratio_text}"
                    )
                else:
                    reason_texts.append(reason["description"])
            lines.append(f"- Subject {item['subject_id']}: " + "; ".join(reason_texts))
    else:
        lines.append("- No subject crossed the configured structural/scale flagging rules.")

    lines.extend([
        "",
        "## Outputs",
    ])
    for plot_path in plot_files:
        lines.append(f"- {plot_path.name}")

    return "\n".join(lines) + "\n"


def build_problematic_subjects_markdown(results: Mapping[str, Any]) -> str:
    storage = results["storage"]
    problematic_subjects = results["problematic_subjects"]
    analysis_scope = results["analysis_scope"]
    total_subjects = int(results["subjects_analyzed"])
    flagged_ids = [int(item["subject_id"]) for item in problematic_subjects]
    clean_count = total_subjects - len(flagged_ids)

    lines = [
        "# pf_full problematic subjects",
        "",
        "当前阶段只做标记，不直接删除、修复或重生成缓存。",
        "",
        "## Decision",
        "- 本文档只记录需要重点关注的被试，不对缓存内容做自动处理。",
        f"- EEG-fNIRS 延迟分析已关闭：{analysis_scope['delay_analysis_note']}",
        f"- 本次检查共分析 {total_subjects} 个被试，其中 {len(flagged_ids)} 个被试被标记，{clean_count} 个被试暂未标记。",
        "",
        "## Flagged Subjects",
    ]

    if not problematic_subjects:
        lines.append("- 当前规则下没有被标记的被试。")
        return "\n".join(lines) + "\n"

    for item in problematic_subjects:
        subject_id = int(item["subject_id"])
        lines.extend([
            f"### Subject {subject_id}",
            f"- Cache size: {format_float(item.get('cache_size_mb'), 2)} MB",
            f"- Jobs: {item.get('jobs_actual', 'n/a')} / expected {item.get('jobs_expected', 'n/a')}",
            f"- Event label mismatch count: {item.get('event_label_mismatch_count', 'n/a')}",
            "- Reasons:",
        ])
        for reason in item["reasons"]:
            if reason["type"] == "scale_outlier":
                median_text = format_float(reason.get("median_reference"))
                ratio_text = format_float(reason.get("ratio_to_median"))
                lines.append(
                    "  - "
                    f"{reason['description']}: value={format_float(reason['value'])}, "
                    f"cohort_median={median_text}, z={format_float(reason.get('z_score'))}, "
                    f"ratio_to_median={ratio_text}x"
                )
            else:
                lines.append(f"  - {reason['description']}")
        lines.append("- Action: mark_only")
        lines.append("")

    flagged_id_text = ", ".join(str(subject_id) for subject_id in flagged_ids)
    lines.extend([
        "## Summary",
        f"- Flagged subject IDs: {flagged_id_text}",
        f"- Remaining unflagged subject count: {clean_count}",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    subject_ids = parse_subject_ids(args.subject_ids)
    subject_dirs = find_subject_dirs(cache_dir, subject_ids)
    if not subject_dirs:
        raise RuntimeError(f"No subject caches found in {cache_dir}")

    output_dir = resolve_output_dir(args.output_dir, cache_dir)
    audit = CacheQualityAudit(cache_dir=cache_dir, subject_dirs=subject_dirs)
    results = audit.run()

    report_json_path = output_dir / "quality_report.json"
    report_json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    problematic_json_path = output_dir / "problematic_subjects.json"
    problematic_json_path.write_text(
        json.dumps(results["problematic_subjects"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    stale_temporal_path = output_dir / "temporal_relationships.png"
    if stale_temporal_path.exists():
        stale_temporal_path.unlink()

    plot_files: List[Path] = []
    if not args.skip_plots:
        subject_overview_path = output_dir / "subject_overview.png"
        waveform_overview_path = output_dir / "waveform_overview.png"
        plot_subject_overview(results, subject_overview_path)
        plot_waveform_overview(results, audit, waveform_overview_path)
        plot_files.extend([subject_overview_path, waveform_overview_path])

    report_md_path = output_dir / "quality_report.md"
    report_md_path.write_text(build_markdown_report(results, plot_files), encoding="utf-8")

    problematic_md_path = output_dir / "problematic_subjects.md"
    problematic_md_path.write_text(build_problematic_subjects_markdown(results), encoding="utf-8")

    print(f"Wrote {report_json_path}")
    print(f"Wrote {problematic_json_path}")
    print(f"Wrote {report_md_path}")
    print(f"Wrote {problematic_md_path}")
    for plot_path in plot_files:
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()