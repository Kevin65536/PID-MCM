#!/usr/bin/env python3
"""Audit the Single-Trial EEG artifact-cleaning candidate on complete records.

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
from scipy.signal import welch

from src.data.eeg_artifact_preprocessing import (
    EEGArtifactCleaningConfig,
    clean_single_trial_eeg,
    compute_channel_quality_metrics,
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


def _audit_controlled(subject_dir: Path) -> tuple[list[dict[str, Any]], str]:
    path = subject_dir / "cnt_artifact.mat"
    if not path.exists():
        return [], "missing_controlled_artifact_recordings"
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    records = np.atleast_1d(payload.get("cnt_artifact", []))
    output = []
    for index, condition in enumerate(CONTROL_CONDITIONS):
        if index >= len(records):
            break
        record = _cell(records, index)
        values = np.asarray(record.x, dtype=np.float64)
        sample_rate_hz = float(record.fs)
        metrics = compute_channel_quality_metrics(values, sample_rate_hz)
        output.append({
            "condition": condition,
            "sample_count": int(len(values)),
            "channel_count": int(values.shape[1]),
            "low_frequency_ratio_median": float(np.median(metrics["low_frequency_ratio"])),
            "high_frequency_ratio_median": float(np.median(metrics["high_frequency_ratio"])),
            "line_noise_ratio_median": float(np.median(metrics["line_noise_ratio"])),
            "robust_scale_median": float(np.median(metrics["robust_scale"])),
        })
    return output, "available" if len(output) == len(CONTROL_CONDITIONS) else "incomplete"


def _audit_subject(subject_dir_text: str, config_payload: dict[str, Any]) -> dict[str, Any]:
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
    controls, control_status = _audit_controlled(subject_dir)
    return {
        "subject": subject,
        "rows": rows,
        "psd": psd_rows,
        "controlled_artifacts": controls,
        "controlled_artifact_status": control_status,
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


def _plot_psd(psd_rows: list[dict[str, Any]], output_dir: Path) -> None:
    frequencies = np.asarray(psd_rows[0]["frequencies_hz"], dtype=float)
    raw = np.asarray([row["raw_density"] for row in psd_rows], dtype=float)
    clean = np.asarray([row["clean_density"] for row in psd_rows], dtype=float)
    selected = (frequencies >= 1.0) & (frequencies <= 45.0)
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.semilogy(frequencies[selected], np.median(raw[:, selected], axis=0), label="raw filtered")
    axis.semilogy(frequencies[selected], np.median(clean[:, selected], axis=0), label="artifact_clean_v2")
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
        futures = {executor.submit(_audit_subject, str(path), asdict(config)): path.name for path in subjects}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"completed {result['subject']}", flush=True)
    results.sort(key=lambda item: item["subject"])
    rows = [row for result in results for row in result["rows"]]
    psd_rows = [row for result in results for row in result["psd"]]
    _write_rows(rows, output_dir)
    _plot_psd(psd_rows, output_dir)

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
        } if condition_rows else {"subject_count": 0}
    calibration = {
        "schema": "single_trial_controlled_artifact_calibration_v2",
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
        "schema": "single_trial_eeg_signal_preservation_v2",
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
        "bad_channel_count": _quantiles(row["bad_channel_count"] for row in rows),
        "all_sample_counts_unchanged": all(row["sample_count_unchanged"] for row in rows),
        "all_channel_counts_unchanged": all(row["channel_count_unchanged"] for row in rows),
    }
    (output_dir / "signal_preservation.json").write_text(
        json.dumps(preservation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    expected_sessions = len(subjects) * 6
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
        "muscle_correction_validated_against_sham": False,
        "full_29_subject_audit": len(results) == 29 and len(rows) == 174,
    }
    decision = {
        "schema": "single_trial_eeg_artifact_admission_v2",
        "decision": "admitted" if all(gates.values()) else "not_admitted",
        "default_eeg_signal_branch": "artifact_clean_v2" if all(gates.values()) else "raw_with_ocular_artifact",
        "gates": gates,
        "blocking_reasons": [name for name, passed in gates.items() if not passed],
        "note": "High-frequency muscle activity remains mask-only until controlled-artifact and sham validation are complete.",
    }
    (output_dir / "admission_decision.yaml").write_text(
        yaml.safe_dump(decision, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    manifest = {
        "schema": "single_trial_eeg_artifact_audit_manifest_v2",
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
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/EEG+NIRS Single-Trial")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subjects", nargs="*", type=int)
    parser.add_argument("--subject-limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    output = run(parse_args())
    print(f"audit written to {output}")
