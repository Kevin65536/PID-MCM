#!/usr/bin/env python3
"""Audit and materialize the admitted Single-Trial EEG v3 branch.

The five ``cnt_artifact`` recordings are calibration controls only.  They are
never emitted as task samples and never contribute labels or windows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, welch

from src.data.eeg_artifact_preprocessing import (
    EEGArtifactCleaningConfig,
    clean_single_trial_eeg,
    compute_channel_quality_metrics,
    correct_high_frequency_bursts,
    detect_high_frequency_mask,
)
from src.data.unified_physiology import (
    CANONICAL_EEG_BAND_HZ,
    CANONICAL_EEG_SAMPLE_RATE_HZ,
    _robust_standardize,
)

matplotlib.use("Agg")

CONTROL_CONDITIONS = ("EOG", "EMG", "Eye Blinking", "Teeth Clenching", "Mouth Opening")


def _cell(value: Any, index: int) -> Any:
    return np.asarray(value, dtype=object).reshape(-1)[index]


def _labels(value: Any) -> list[str]:
    return [str(item) for item in np.asarray(value).reshape(-1).tolist()]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _geometry(subject_dir: Path, channel_names: list[str]) -> tuple[np.ndarray, str]:
    path = subject_dir / "with occular artifact" / "mnt.mat"
    if not path.exists():
        return np.full((len(channel_names), 3), np.nan), "missing_mnt"
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    mnt = payload.get("mnt")
    if mnt is None:
        return np.full((len(channel_names), 3), np.nan), "missing_mnt_key"
    labels = _labels(mnt.clab)
    positions = np.asarray(mnt.pos_3d, dtype=np.float64)
    if positions.shape[0] == 3:
        positions = positions.T
    lookup = {name: positions[index] for index, name in enumerate(labels) if index < len(positions)}
    result = np.asarray([lookup.get(name, np.full(3, np.nan)) for name in channel_names])
    status = "available" if np.all(np.isfinite(result)) else "partial_or_missing"
    return result, status


def _band_power(values: np.ndarray, sample_rate_hz: float, low: float, high: float) -> np.ndarray:
    frequencies, density = welch(
        values,
        fs=sample_rate_hz,
        nperseg=min(len(values), max(256, int(round(sample_rate_hz * 4)))),
        axis=0,
    )
    selected = (frequencies >= low) & (frequencies <= high)
    return np.trapezoid(density[selected], frequencies[selected], axis=0)


def _psd(values: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    frequencies, density = welch(
        values,
        fs=sample_rate_hz,
        nperseg=min(len(values), max(256, int(round(sample_rate_hz * 4)))),
        axis=0,
    )
    return frequencies, np.median(density, axis=1)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, np.finfo(np.float64).eps))


def _circular_event_mask(
    marker_times_ms: np.ndarray,
    *,
    sample_count: int,
    sample_rate_hz: float,
    half_window_s: float,
) -> np.ndarray:
    output = np.zeros(sample_count, dtype=bool)
    radius = max(1, int(round(half_window_s * sample_rate_hz)))
    offsets = np.arange(-radius, radius + 1)
    for marker_ms in marker_times_ms:
        center = int(round(float(marker_ms) / 1000.0 * sample_rate_hz))
        output[(center + offsets) % sample_count] = True
    return output


def _high_frequency_signal(values: np.ndarray, sample_rate_hz: float, band_hz: tuple[float, float]) -> np.ndarray:
    nyquist = 0.5 * sample_rate_hz
    sos = butter(4, [band_hz[0] / nyquist, band_hz[1] / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, values, axis=0)


def _audit_controlled(
    subject_dir: Path,
    config: EEGArtifactCleaningConfig,
) -> tuple[list[dict[str, Any]], str]:
    path = subject_dir / "cnt_artifact.mat"
    marker_path = subject_dir / "mrk_artifact.mat"
    if not path.exists() or not marker_path.exists():
        return [], "missing_controlled_artifact_recordings"
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    records = np.atleast_1d(payload.get("cnt_artifact", []))
    markers = np.atleast_1d(
        loadmat(marker_path, squeeze_me=True, struct_as_record=False).get("mrk_artifact", [])
    )
    output = []
    for index, condition in enumerate(CONTROL_CONDITIONS):
        if index >= len(records):
            break
        record = _cell(records, index)
        values = np.asarray(record.x, dtype=np.float64)
        sample_rate_hz = float(record.fs)
        metrics = compute_channel_quality_metrics(values, sample_rate_hz)
        high_frequency_mask, mask_state = detect_high_frequency_mask(values, sample_rate_hz, config)
        corrected, correction_state = correct_high_frequency_bursts(
            values, high_frequency_mask, sample_rate_hz, config
        )
        marker_times_ms = np.asarray(_cell(markers, index).time, dtype=np.float64).reshape(-1)
        marker_intervals_s = np.diff(marker_times_ms) / 1000.0
        if len(marker_intervals_s):
            half_window_s = 0.25 * float(np.median(marker_intervals_s))
        else:
            half_window_s = 0.125 * len(values) / sample_rate_hz
        event_mask = _circular_event_mask(
            marker_times_ms,
            sample_count=len(values),
            sample_rate_hz=sample_rate_hz,
            half_window_s=half_window_s,
        )
        seed = int(subject_dir.name.rsplit(" ", 1)[-1]) * 100 + index
        rng = np.random.default_rng(seed)
        null_coverages = []
        duration_ms = len(values) / sample_rate_hz * 1000.0
        for _ in range(128):
            shifted_times = (marker_times_ms + rng.uniform(0.0, duration_ms)) % duration_ms
            shifted_event_mask = _circular_event_mask(
                shifted_times,
                sample_count=len(values),
                sample_rate_hz=sample_rate_hz,
                half_window_s=half_window_s,
            )
            null_coverages.append(float(np.mean(high_frequency_mask[shifted_event_mask])))
        shift_samples = max(1, int(round(0.5 * np.median(marker_intervals_s) * sample_rate_hz))) \
            if len(marker_intervals_s) else len(values) // 2
        sham_mask = np.roll(high_frequency_mask, shift_samples)
        sham_corrected, _ = correct_high_frequency_bursts(values, sham_mask, sample_rate_hz, config)
        high_before = _high_frequency_signal(values, sample_rate_hz, config.high_frequency_band_hz)
        high_after = _high_frequency_signal(corrected, sample_rate_hz, config.high_frequency_band_hz)
        high_sham = _high_frequency_signal(sham_corrected, sample_rate_hz, config.high_frequency_band_hz)
        event_power_before = float(np.mean(high_before[event_mask] ** 2))
        event_power_after = float(np.mean(high_after[event_mask] ** 2))
        event_power_sham = float(np.mean(high_sham[event_mask] ** 2))
        output.append({
            "condition": condition,
            "sample_count": int(len(values)),
            "channel_count": int(values.shape[1]),
            "low_frequency_ratio_median": float(np.median(metrics["low_frequency_ratio"])),
            "high_frequency_ratio_median": float(np.median(metrics["high_frequency_ratio"])),
            "line_noise_ratio_median": float(np.median(metrics["line_noise_ratio"])),
            "robust_scale_median": float(np.median(metrics["robust_scale"])),
            "marker_count": int(len(marker_times_ms)),
            "adaptive_event_half_window_s": half_window_s,
            "muscle_mask_fraction": float(mask_state["dilated_fraction"]),
            "event_mask_coverage": float(np.mean(high_frequency_mask[event_mask])),
            "time_shift_null_coverage_mean": float(np.mean(null_coverages)),
            "event_coverage_above_time_shift_null": bool(
                np.mean(high_frequency_mask[event_mask]) > np.mean(null_coverages)
            ),
            "event_high_frequency_reduction": float(
                1.0 - event_power_after / max(event_power_before, np.finfo(np.float64).eps)
            ),
            "sham_event_high_frequency_reduction": float(
                1.0 - event_power_sham / max(event_power_before, np.finfo(np.float64).eps)
            ),
            "target_reduction_above_sham": bool(event_power_after < event_power_sham),
            "correction_method": correction_state["method"],
        })
    return output, "available" if len(output) == len(CONTROL_CONDITIONS) else "incomplete"


def _write_artifact_cache_record(
    *,
    cache_root: Path,
    subject: str,
    session_index: int,
    cnt_path: Path,
    channel_names: list[str],
    result: Any,
    config: EEGArtifactCleaningConfig,
) -> dict[str, Any]:
    base_record_id = f"session_{session_index:02d}"
    join_key = f"eeg_fnirs_single_trial|{subject}|{base_record_id}"
    canonical, standardization_state = _robust_standardize(result.cleaned_values)
    source_stat = cnt_path.stat()
    preprocessing_state = dict(standardization_state)
    preprocessing_state.update({
        "native_unit": "uV",
        "native_sample_rate_hz": float(result.state["sample_rate_hz"]),
        "canonical_sample_rate_hz": CANONICAL_EEG_SAMPLE_RATE_HZ,
        "filter_band_hz": list(CANONICAL_EEG_BAND_HZ),
        "filter_input_repaired_nonfinite_samples": int(
            result.state["input_repaired_nonfinite_samples"]["eeg"]
        ),
        "source_path": str(cnt_path),
        "signal_branch": config.schema,
        "artifact_cleaning": result.state,
    })
    output_path = cache_root / subject / f"{base_record_id}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".npz.tmp")
    with temporary_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema=np.asarray("single_trial_eeg_artifact_cache_v3"),
            signal_branch=np.asarray(config.schema),
            join_key=np.asarray(join_key),
            source_path=np.asarray(str(cnt_path.relative_to(REPO_ROOT))),
            source_size_bytes=np.asarray(source_stat.st_size, dtype=np.int64),
            source_mtime_ns=np.asarray(source_stat.st_mtime_ns, dtype=np.int64),
            eeg=canonical,
            artifact_mask=np.asarray(result.artifact_mask, dtype=bool),
            bad_channel_mask=np.asarray(result.bad_channel_mask, dtype=bool),
            channel_names=np.asarray(channel_names),
            preprocessing_state_json=np.asarray(json.dumps(preprocessing_state, ensure_ascii=False)),
        )
    temporary_path.replace(output_path)
    return {
        "join_key": join_key,
        "subject": subject,
        "base_record_id": base_record_id,
        "cache_path": str(output_path.relative_to(REPO_ROOT)),
        "source_path": str(cnt_path.relative_to(REPO_ROOT)),
        "sample_count": int(canonical.shape[0]),
        "channel_count": int(canonical.shape[1]),
        "artifact_fraction": float(np.mean(result.artifact_mask)),
    }


def _audit_subject(
    subject_dir_text: str,
    config_payload: dict[str, Any],
    cache_root_text: str | None,
) -> dict[str, Any]:
    subject_dir = Path(subject_dir_text)
    subject = subject_dir.name.replace(" ", "_")
    source_dir = subject_dir / "with occular artifact"
    cnt_path = source_dir / "cnt.mat"
    mrk_path = source_dir / "mrk.mat"
    cnt = np.atleast_1d(loadmat(cnt_path, squeeze_me=True, struct_as_record=False)["cnt"])
    markers = np.atleast_1d(loadmat(mrk_path, squeeze_me=True, struct_as_record=False)["mrk"])
    config = EEGArtifactCleaningConfig(**config_payload)
    rows: list[dict[str, Any]] = []
    psd_rows: list[dict[str, Any]] = []
    cache_records: list[dict[str, Any]] = []
    for session_index in range(len(cnt)):
        record = _cell(cnt, session_index)
        values = np.asarray(record.x, dtype=np.float64)
        labels = _labels(record.clab)
        eeg_keep = np.asarray(["EOG" not in name.upper() for name in labels], dtype=bool)
        eog_keep = ~eeg_keep
        eeg_names = [name for name, keep in zip(labels, eeg_keep) if keep]
        eog_names = [name for name, keep in zip(labels, eog_keep) if keep]
        positions, geometry_status = _geometry(subject_dir, eeg_names)
        result = clean_single_trial_eeg(
            values[:, eeg_keep],
            values[:, eog_keep],
            sample_rate_hz=float(record.fs),
            channel_names=eeg_names,
            eog_channel_names=eog_names,
            channel_positions=positions,
            config=config,
        )
        if cache_root_text is not None:
            cache_records.append(_write_artifact_cache_record(
                cache_root=Path(cache_root_text),
                subject=subject,
                session_index=session_index,
                cnt_path=cnt_path,
                channel_names=eeg_names,
                result=result,
                config=config,
            ))
        raw_alpha = _band_power(result.filtered_raw_values, float(record.fs), 8.0, 13.0)
        clean_alpha = _band_power(result.cleaned_values, float(record.fs), 8.0, 13.0)
        topology_corr = float(np.corrcoef(np.log1p(raw_alpha), np.log1p(clean_alpha))[0, 1])
        nonfrontal = np.asarray([
            not (name.upper().startswith(("F", "AF")) or "FP" in name.upper())
            for name in eeg_names
        ])
        nonfrontal_topology_corr = float(
            np.corrcoef(np.log1p(raw_alpha[nonfrontal]), np.log1p(clean_alpha[nonfrontal]))[0, 1]
        )
        marker_count = 0
        if session_index < len(markers):
            marker = _cell(markers, session_index)
            marker_count = int(np.asarray(marker.time).size)
        removed = np.asarray(result.state["eog_regression"]["removed_variance_fraction"], dtype=float)
        before = float(result.state["median_eog_correlation_before"])
        after = float(result.state["median_eog_correlation_after"])
        row = {
            "schema": config.schema,
            "subject": subject,
            "session_index": session_index,
            "source_path": str(cnt_path.relative_to(REPO_ROOT)),
            "sample_rate_hz": float(record.fs),
            "sample_count": int(len(values)),
            "eeg_channel_count": int(np.count_nonzero(eeg_keep)),
            "eog_channel_count": int(np.count_nonzero(eog_keep)),
            "event_count": marker_count,
            "sample_count_unchanged": bool(result.cleaned_values.shape[0] == values.shape[0]),
            "channel_count_unchanged": bool(result.cleaned_values.shape[1] == np.count_nonzero(eeg_keep)),
            "geometry_status": geometry_status,
            "interpolation_method": result.state["interpolation"]["method"],
            "bad_channel_count": int(np.count_nonzero(result.bad_channel_mask)),
            "bad_channel_names": result.state["bad_channel_names"],
            "artifact_fraction": float(result.state["artifact_fraction"]),
            "ocular_fraction": float(result.state["ocular"]["dilated_fraction"]),
            "high_frequency_fraction": float(result.state["high_frequency"]["dilated_fraction"]),
            "median_eog_correlation_before": before,
            "median_eog_correlation_after": after,
            "eog_correlation_ratio": _safe_ratio(after, before),
            "median_removed_variance_fraction": float(np.median(removed)),
            "muscle_high_frequency_energy_reduction": float(
                result.state["muscle_correction"]["high_frequency_energy_reduction_in_mask"]
            ),
            "muscle_correction_method": str(result.state["muscle_correction"]["method"]),
            "alpha_power_ratio_median": float(np.median(clean_alpha / np.maximum(raw_alpha, 1e-18))),
            "alpha_topology_correlation": topology_corr,
            "nonfrontal_channel_count": int(np.count_nonzero(nonfrontal)),
            "nonfrontal_alpha_topology_correlation": nonfrontal_topology_corr,
            "config": result.state["config"],
        }
        rows.append(row)
        frequencies, raw_density = _psd(result.filtered_raw_values, float(record.fs))
        _, clean_density = _psd(result.cleaned_values, float(record.fs))
        psd_rows.append({
            "subject": subject,
            "session_index": session_index,
            "frequencies_hz": frequencies.tolist(),
            "raw_density": raw_density.tolist(),
            "clean_density": clean_density.tolist(),
        })
    controls, control_status = _audit_controlled(subject_dir, config)
    return {
        "subject": subject,
        "rows": rows,
        "psd": psd_rows,
        "controlled_artifacts": controls,
        "controlled_artifact_status": control_status,
        "cache_records": cache_records,
    }


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _write_rows(rows: list[dict[str, Any]], output_dir: Path) -> None:
    with (output_dir / "subject_session_qc.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_jsonable) + "\n")
    scalar_keys = [key for key, value in rows[0].items() if not isinstance(value, (dict, list))]
    with (output_dir / "subject_session_qc.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in scalar_keys} for row in rows)


def _plot_psd(psd_rows: list[dict[str, Any]], output_dir: Path, clean_branch: str) -> None:
    frequencies = np.asarray(psd_rows[0]["frequencies_hz"], dtype=float)
    raw = np.asarray([row["raw_density"] for row in psd_rows], dtype=float)
    clean = np.asarray([row["clean_density"] for row in psd_rows], dtype=float)
    selected = (frequencies >= 1.0) & (frequencies <= 45.0)
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.semilogy(frequencies[selected], np.median(raw[:, selected], axis=0), label="raw filtered")
    axis.semilogy(frequencies[selected], np.median(clean[:, selected], axis=0), label=clean_branch)
    axis.fill_between(
        frequencies[selected],
        np.quantile(clean[:, selected], 0.05, axis=0),
        np.quantile(clean[:, selected], 0.95, axis=0),
        alpha=0.2,
        label="clean session 5–95%",
    )
    axis.set(xlabel="Frequency (Hz)", ylabel="PSD (native unit²/Hz)", title="Single-Trial EEG raw vs cleaned")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "raw_cleaned_psd_comparison.png", dpi=180)
    figure.savefig(output_dir / "raw_cleaned_psd_comparison.svg")
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    data_root = (REPO_ROOT / args.data_root).resolve()
    subjects = sorted(path for path in (data_root / "EEG_01-29").glob("subject *") if path.is_dir())
    if args.subjects:
        selected = {f"subject {int(value):02d}" for value in args.subjects}
        subjects = [path for path in subjects if path.name in selected]
    if args.subject_limit:
        subjects = subjects[: args.subject_limit]
    if not subjects:
        raise RuntimeError("no subjects selected")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = EEGArtifactCleaningConfig()
    results = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(subjects))) as executor:
        cache_root = None if not args.cache_root else str(Path(args.cache_root).resolve())
        futures = {
            executor.submit(_audit_subject, str(path), asdict(config), cache_root): path.name
            for path in subjects
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"completed {result['subject']}", flush=True)
    results.sort(key=lambda item: item["subject"])
    rows = [row for result in results for row in result["rows"]]
    psd_rows = [row for result in results for row in result["psd"]]
    _write_rows(rows, output_dir)
    _plot_psd(psd_rows, output_dir, config.schema)

    controlled = [
        {"subject": result["subject"], **row}
        for result in results
        for row in result["controlled_artifacts"]
    ]
    by_condition = {}
    for condition in CONTROL_CONDITIONS:
        condition_rows = [row for row in controlled if row["condition"] == condition]
        by_condition[condition] = {
            "subject_count": len(condition_rows),
            "low_frequency_ratio": _quantiles(row["low_frequency_ratio_median"] for row in condition_rows),
            "high_frequency_ratio": _quantiles(row["high_frequency_ratio_median"] for row in condition_rows),
            "robust_scale": _quantiles(row["robust_scale_median"] for row in condition_rows),
            "event_mask_coverage": _quantiles(row["event_mask_coverage"] for row in condition_rows),
            "time_shift_null_coverage": _quantiles(
                row["time_shift_null_coverage_mean"] for row in condition_rows
            ),
            "event_high_frequency_reduction": _quantiles(
                row["event_high_frequency_reduction"] for row in condition_rows
            ),
            "sham_event_high_frequency_reduction": _quantiles(
                row["sham_event_high_frequency_reduction"] for row in condition_rows
            ),
            "event_coverage_above_null_subject_fraction": float(np.mean([
                row["event_coverage_above_time_shift_null"] for row in condition_rows
            ])),
            "target_reduction_above_sham_subject_fraction": float(np.mean([
                row["target_reduction_above_sham"] for row in condition_rows
            ])),
        } if condition_rows else {"subject_count": 0}
    calibration = {
        "schema": "single_trial_controlled_artifact_calibration_v3",
        "role": "detector_calibration_only_not_a_task_dataset",
        "conditions": by_condition,
        "subject_status": {result["subject"]: result["controlled_artifact_status"] for result in results},
        "rows": controlled,
    }
    blink_available = by_condition["Eye Blinking"].get("subject_count", 0) > 0
    clench_available = by_condition["Teeth Clenching"].get("subject_count", 0) > 0
    calibration["expected_signature_directions"] = {
        "eye_blinking_low_frequency_above_teeth_clenching": bool(
            blink_available and clench_available
            and by_condition["Eye Blinking"]["low_frequency_ratio"]["median"]
            > by_condition["Teeth Clenching"]["low_frequency_ratio"]["median"]
        ),
        "teeth_clenching_high_frequency_above_eye_blinking": bool(
            blink_available and clench_available
            and by_condition["Teeth Clenching"]["high_frequency_ratio"]["median"]
            > by_condition["Eye Blinking"]["high_frequency_ratio"]["median"]
        ),
    }
    (output_dir / "artifact_calibration.json").write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False, default=_jsonable) + "\n", encoding="utf-8"
    )

    preservation = {
        "schema": "single_trial_eeg_signal_preservation_v3",
        "record_count": len(rows),
        "subject_count": len(results),
        "artifact_fraction": _quantiles(row["artifact_fraction"] for row in rows),
        "ocular_fraction": _quantiles(row["ocular_fraction"] for row in rows),
        "high_frequency_fraction": _quantiles(row["high_frequency_fraction"] for row in rows),
        "eog_correlation_before": _quantiles(row["median_eog_correlation_before"] for row in rows),
        "eog_correlation_after": _quantiles(row["median_eog_correlation_after"] for row in rows),
        "eog_correlation_ratio": _quantiles(row["eog_correlation_ratio"] for row in rows),
        "alpha_power_ratio_median": _quantiles(row["alpha_power_ratio_median"] for row in rows),
        "alpha_topology_correlation": _quantiles(row["alpha_topology_correlation"] for row in rows),
        "nonfrontal_alpha_topology_correlation": _quantiles(
            row["nonfrontal_alpha_topology_correlation"] for row in rows
        ),
        "nonfrontal_alpha_topology_negative_record_count": int(np.count_nonzero([
            row["nonfrontal_alpha_topology_correlation"] < 0.0 for row in rows
        ])),
        "median_removed_variance_fraction": _quantiles(
            row["median_removed_variance_fraction"] for row in rows
        ),
        "muscle_high_frequency_energy_reduction": _quantiles(
            row["muscle_high_frequency_energy_reduction"] for row in rows
        ),
        "bad_channel_count": _quantiles(row["bad_channel_count"] for row in rows),
        "all_sample_counts_unchanged": all(row["sample_count_unchanged"] for row in rows),
        "all_channel_counts_unchanged": all(row["channel_count_unchanged"] for row in rows),
    }
    (output_dir / "signal_preservation.json").write_text(
        json.dumps(preservation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    expected_sessions = len(subjects) * 6
    muscle_conditions = [by_condition[name] for name in ("EMG", "Teeth Clenching", "Mouth Opening")]
    muscle_correction_validated = all(
        condition.get("target_reduction_above_sham_subject_fraction", 0.0) > 0.5
        and condition.get("event_coverage_above_null_subject_fraction", 0.0) > 0.5
        and condition["event_high_frequency_reduction"]["median"]
        > condition["sham_event_high_frequency_reduction"]["median"]
        for condition in muscle_conditions
    )
    gates = {
        "selected_subject_coverage": len(results) == len(subjects),
        "six_sessions_per_selected_subject": len(rows) == expected_sessions,
        "no_sample_or_channel_loss": preservation["all_sample_counts_unchanged"] and preservation["all_channel_counts_unchanged"],
        "eog_correlation_reduced_in_majority": float(np.mean([row["eog_correlation_ratio"] < 1.0 for row in rows])) > 0.5,
        "controlled_artifact_records_separated_from_task_data": calibration["role"] == "detector_calibration_only_not_a_task_dataset",
        "controlled_signature_directions_verified": all(calibration["expected_signature_directions"].values()),
        "bad_channel_limit_not_saturated_in_majority": float(np.mean([
            row["bad_channel_count"] < int(np.floor(row["eeg_channel_count"] * config.max_bad_channel_fraction))
            for row in rows
        ])) > 0.5,
        "nonfrontal_alpha_topology_has_no_negative_outliers": preservation[
            "nonfrontal_alpha_topology_negative_record_count"
        ] == 0,
        "muscle_correction_validated_against_sham": muscle_correction_validated,
        "full_29_subject_audit": len(results) == 29 and len(rows) == 174,
    }
    decision = {
        "schema": "single_trial_eeg_artifact_admission_v3",
        "decision": "admitted" if all(gates.values()) else "not_admitted",
        "default_eeg_signal_branch": config.schema if all(gates.values()) else "raw_with_ocular_artifact",
        "gates": gates,
        "blocking_reasons": [name for name, passed in gates.items() if not passed],
        "note": (
            "Muscle correction attenuates only 30-45 Hz content inside adaptively detected bursts; "
            "controlled event windows are compared with equal-mask circular-shift sham correction."
        ),
    }
    (output_dir / "admission_decision.yaml").write_text(
        yaml.safe_dump(decision, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    manifest = {
        "schema": "single_trial_eeg_artifact_audit_manifest_v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "data_root": str(data_root.relative_to(REPO_ROOT)),
        "subject_count": len(results),
        "record_count": len(rows),
        "workers": args.workers,
        "cleaning_config": config.to_dict(),
        "code_sha256": {
            "audit": _sha256(Path(__file__)),
            "cleaner": _sha256(REPO_ROOT / "src/data/eeg_artifact_preprocessing.py"),
        },
        "outputs": sorted(path.name for path in output_dir.iterdir()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.cache_root:
        artifact_cache_root = Path(args.cache_root).resolve()
        cache_records = [record for result in results for record in result["cache_records"]]
        cache_manifest = {
            "schema": "single_trial_eeg_artifact_cache_v3",
            "created_at": manifest["created_at"],
            "signal_branch": config.schema,
            "record_count": len(cache_records),
            "subject_count": len(results),
            "cleaning_config": config.to_dict(),
            "code_sha256": manifest["code_sha256"],
            "records": cache_records,
        }
        artifact_cache_root.mkdir(parents=True, exist_ok=True)
        (artifact_cache_root / "cache_manifest.json").write_text(
            json.dumps(cache_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/EEG+NIRS Single-Trial")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subjects", nargs="*", type=int)
    parser.add_argument("--subject-limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--cache-root",
        default="",
        help="Optional versioned EEG artifact-cache output root.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    output = run(parse_args())
    print(f"audit written to {output}")
