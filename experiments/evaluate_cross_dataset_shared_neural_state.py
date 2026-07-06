#!/usr/bin/env python3
"""Estimate lagged cross-modal shared neural information in four EEG-fNIRS datasets.

The diagnostic removes prediction from each modality's own history, trial phase,
and condition before fitting a low-dimensional CCA state to the remaining paired
innovations. Reciprocal leave-one-subject-out folds test whether an EEG-derived
state reconstructs future fNIRS innovation and whether an fNIRS-derived state
back-projects the corresponding EEG innovation in a held-out subject.

The reported fractions apply to standardized one-second raw-data-derived
features, not to waveform samples or universal mutual information.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import warnings
import zipfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.io import loadmat
from scipy.signal import detrend, resample_poly
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.eeg_fnirs_dataset import BBCIDataLoader  # noqa: E402
from src.data.fnirs_standardization import (  # noqa: E402
    DATASET_FNIRS_CONTRACTS,
    standardize_fnirs_record,
)
from src.data.simultaneous_eeg_nirs_dataset import SimultaneousCognitiveLoader  # noqa: E402


SCHEMA = "cross_dataset_shared_neural_state_v1"
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
DISPLAY_NAMES = {
    "eeg_fnirs_single_trial": "Single-Trial",
    "refed": "REFED",
    "visual_cognitive_motivation": "Visual motivation",
    "simultaneous_eeg_nirs": "Simultaneous",
}
DIRECTIONS = {
    "eeg_to_fnirs": "EEG → future fNIRS",
    "fnirs_to_eeg": "fNIRS → earlier EEG",
}
JOINT_DIRECTIONS = {
    "joint_to_fnirs": "Joint state → fNIRS",
    "joint_to_eeg": "Joint state → EEG",
}


@dataclass(frozen=True)
class Segment:
    dataset_id: str
    subject: str
    segment_id: str
    condition: str
    eeg: np.ndarray
    fnirs: np.ndarray


@dataclass(frozen=True)
class LagRows:
    eeg_target: np.ndarray
    fnirs_target: np.ndarray
    eeg_base: np.ndarray
    fnirs_base: np.ndarray
    groups: np.ndarray
    conditions: np.ndarray


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_json_value(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_record(path: Path, role: str, dataset_id: str, subject: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "dataset_id": dataset_id,
        "subject": subject,
        "role": role,
        "path": str(path.relative_to(REPO_ROOT)),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256(path),
    }


def _mat_payload(path: Path) -> Any:
    payload = loadmat(path, struct_as_record=False, squeeze_me=True)
    return payload[next(key for key in payload if not key.startswith("__"))]


def _repair_nonfinite(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    if output.ndim != 2:
        raise ValueError(f"expected [time, channel] array, got {output.shape}")
    indices = np.arange(output.shape[0], dtype=np.float64)
    for channel in range(output.shape[1]):
        series = output[:, channel]
        finite = np.isfinite(series)
        if finite.all():
            continue
        if finite.sum() < 2:
            output[:, channel] = 0.0
        else:
            output[:, channel] = np.interp(indices, indices[finite], series[finite])
    return output


def _eeg_features(values: np.ndarray, sample_rate_hz: float, bands: Mapping[str, Sequence[float]]) -> np.ndarray:
    values = _repair_nonfinite(values)
    samples = int(round(float(sample_rate_hz)))
    count = values.shape[0] // samples
    if count < 1:
        return np.empty((0, values.shape[1] * len(bands)), dtype=np.float64)
    windows = values[: count * samples].reshape(count, samples, values.shape[1])
    windows = detrend(windows, axis=1, type="linear")
    taper = np.hanning(samples)[None, :, None]
    spectrum = np.fft.rfft(windows * taper, axis=1)
    power = (spectrum.real ** 2 + spectrum.imag ** 2) / max(float(samples), 1.0)
    frequencies = np.fft.rfftfreq(samples, d=1.0 / float(sample_rate_hz))
    features = []
    for low, high in bands.values():
        mask = (frequencies >= float(low)) & (frequencies < float(high))
        if not mask.any():
            features.append(np.zeros((count, values.shape[1]), dtype=np.float64))
        else:
            features.append(np.log(power[:, mask, :].mean(axis=1) + 1e-20))
    return np.concatenate(features, axis=1)


def _fnirs_features(values: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    values = _repair_nonfinite(values)
    samples = int(round(float(sample_rate_hz)))
    count = values.shape[0] // samples
    if count < 1:
        return np.empty((0, values.shape[1] * 2), dtype=np.float64)
    windows = values[: count * samples].reshape(count, samples, values.shape[1])
    mean = windows.mean(axis=1)
    time = np.linspace(-1.0, 1.0, samples, dtype=np.float64)
    slope = np.einsum("tsc,s->tc", windows, time) / max(float(np.dot(time, time)), 1e-12)
    return np.concatenate((mean, slope), axis=1)


def _segment_features(
    dataset_id: str,
    subject: str,
    segment_id: str,
    condition: str,
    eeg: np.ndarray,
    eeg_fs: float,
    fnirs: np.ndarray,
    fnirs_fs: float,
    bands: Mapping[str, Sequence[float]],
) -> Segment | None:
    eeg_features = _eeg_features(eeg, eeg_fs, bands)
    fnirs_features = _fnirs_features(fnirs, fnirs_fs)
    length = min(len(eeg_features), len(fnirs_features))
    if length < 7:
        return None
    return Segment(
        dataset_id=dataset_id,
        subject=str(subject),
        segment_id=segment_id,
        condition=condition,
        eeg=eeg_features[:length],
        fnirs=fnirs_features[:length],
    )


def _robust_subject_adapter(segments: Sequence[Segment], clip_abs: float) -> tuple[list[Segment], dict[str, Any]]:
    adapted: list[Segment] = []
    report: dict[str, Any] = {}
    for subject in sorted({segment.subject for segment in segments}):
        selected = [segment for segment in segments if segment.subject == subject]
        modality_stats: dict[str, Any] = {}
        transformed: dict[str, list[np.ndarray]] = {}
        for modality in ("eeg", "fnirs"):
            matrix = np.concatenate([getattr(segment, modality) for segment in selected], axis=0)
            median = np.median(matrix, axis=0)
            mad = 1.4826 * np.median(np.abs(matrix - median), axis=0)
            std = matrix.std(axis=0)
            scale = np.where(mad > 1e-8, mad, np.where(std > 1e-8, std, 1.0))
            unbounded = [(getattr(segment, modality) - median) / scale for segment in selected]
            transformed[modality] = [np.clip(value, -float(clip_abs), float(clip_abs)) for value in unbounded]
            unbounded_matrix = np.concatenate(unbounded, axis=0)
            modality_stats[modality] = {
                "feature_count": int(matrix.shape[1]),
                "time_rows": int(matrix.shape[0]),
                "mad_fallback_count": int(np.sum(mad <= 1e-8)),
                "finite_fraction": float(np.isfinite(matrix).mean()),
                "clip_abs": float(clip_abs),
                "clipped_fraction": float(np.mean(np.abs(unbounded_matrix) > float(clip_abs))),
                "median_abs_after": float(np.median(np.abs(np.concatenate(transformed[modality], axis=0)))),
            }
        for index, segment in enumerate(selected):
            adapted.append(replace(segment, eeg=transformed["eeg"][index], fnirs=transformed["fnirs"][index]))
        report[subject] = modality_stats
    return adapted, report


def _slice_seconds(values: np.ndarray, start_ms: float, duration_s: float, fs: float) -> np.ndarray:
    start = int(round(float(start_ms) * float(fs) / 1000.0))
    length = int(round(float(duration_s) * float(fs)))
    start = max(start, 0)
    end = min(start + length, values.shape[0])
    return values[start:end]


def load_single_trial(
    root: Path, subjects: Sequence[int], cfg: Mapping[str, Any], bands: Mapping[str, Sequence[float]]
) -> tuple[list[Segment], list[dict[str, Any]], list[dict[str, Any]]]:
    loader = BBCIDataLoader(str(root), subject_ids=list(subjects), task="both", modality="both")
    contract = DATASET_FNIRS_CONTRACTS["eeg_fnirs_single_trial"]["wavelength_pair"]
    segments: list[Segment] = []
    inventory: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    duration = float(cfg["trial_duration_s"])
    for subject in subjects:
        subject_name = str(subject)
        eeg_dir = loader._get_subject_dir(int(subject), "eeg")
        fnirs_dir = loader._get_subject_dir(int(subject), "fnirs")
        for role, path in (("eeg_continuous", eeg_dir / "cnt.mat"), ("eeg_markers", eeg_dir / "mrk.mat"),
                           ("fnirs_continuous", fnirs_dir / "cnt.mat"), ("fnirs_markers", fnirs_dir / "mrk.mat")):
            inputs.append(_input_record(path, role, "eeg_fnirs_single_trial", subject_name))
        eeg_sessions, eeg_markers, eeg_info = loader.load_subject_data(int(subject), "eeg")
        fnirs_sessions, fnirs_markers, fnirs_info = loader.load_subject_data(int(subject), "fnirs")
        eeg_names = [str(value) for value in eeg_info["clab"]]
        eeg_mask = np.asarray(["EOG" not in name.upper() for name in eeg_names], dtype=bool)
        subject_count = 0
        for session_index in range(6):
            eeg = np.asarray(eeg_sessions[session_index], dtype=np.float64)[:, eeg_mask]
            fnirs_raw = np.asarray(fnirs_sessions[session_index], dtype=np.float64)
            standard = standardize_fnirs_record(
                fnirs_raw, sample_rate_hz=float(fnirs_info["fs"]), contract=contract
            )
            eeg_times = np.asarray(eeg_markers[session_index]["time"], dtype=np.float64)
            fnirs_times = np.asarray(fnirs_markers[session_index]["time"], dtype=np.float64)
            eeg_labels = np.argmax(np.asarray(eeg_markers[session_index]["y"]), axis=0)
            fnirs_labels = np.argmax(np.asarray(fnirs_markers[session_index]["y"]), axis=0)
            common = min(len(eeg_times), len(fnirs_times))
            task = "motor_imagery" if session_index % 2 == 0 else "mental_arithmetic"
            for trial in range(common):
                if int(eeg_labels[trial]) != int(fnirs_labels[trial]):
                    continue
                segment = _segment_features(
                    "eeg_fnirs_single_trial", subject_name, f"s{session_index}_trial_{trial:03d}",
                    f"{task}_class_{int(eeg_labels[trial])}",
                    _slice_seconds(eeg, eeg_times[trial], duration, float(eeg_info["fs"])), float(eeg_info["fs"]),
                    _slice_seconds(standard.values, fnirs_times[trial], duration, float(fnirs_info["fs"])), float(fnirs_info["fs"]),
                    bands,
                )
                if segment is not None:
                    segments.append(segment)
                    subject_count += 1
            inventory.append({
                "dataset_id": "eeg_fnirs_single_trial", "subject": subject_name,
                "record": f"session_{session_index}", "eeg_shape": list(eeg.shape),
                "fnirs_shape": list(fnirs_raw.shape), "events": int(common),
                "fnirs_repaired_samples": int(sum(standard.state.repaired_nonfinite)),
                "fnirs_residual_drift_sd_per_min": float(standard.quality["residual_drift_sd_per_min_median"]),
            })
        if subject_count == 0:
            raise RuntimeError(f"no Single-Trial segments loaded for subject {subject}")
    return segments, inventory, inputs


def load_refed(
    root: Path, subjects: Sequence[int], cfg: Mapping[str, Any], bands: Mapping[str, Sequence[float]]
) -> tuple[list[Segment], list[dict[str, Any]], list[dict[str, Any]]]:
    contract = DATASET_FNIRS_CONTRACTS["refed"][str(cfg["fnirs_signal"])]
    signal_indices = (0, 1) if str(cfg["fnirs_signal"]) == "hbo_hbr" else (3, 4, 5)
    segments: list[Segment] = []
    inventory: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for subject in subjects:
        subject_name = str(subject)
        directory = root / "data" / subject_name
        eeg_path = directory / "EEG_videos.mat"
        fnirs_path = directory / "fNIRS_videos.mat"
        inputs.extend((
            _input_record(eeg_path, "eeg_videos", "refed", subject_name),
            _input_record(fnirs_path, "fnirs_videos", "refed", subject_name),
        ))
        for video in [int(value) for value in cfg["videos"]]:
            key = f"video_{video}"
            eeg_payload = loadmat(eeg_path, variable_names=[key])
            fnirs_payload = loadmat(fnirs_path, variable_names=[key])
            if key not in eeg_payload or key not in fnirs_payload:
                continue
            eeg = np.asarray(eeg_payload[key], dtype=np.float64).T
            tensor = np.asarray(fnirs_payload[key], dtype=np.float64)
            fnirs_raw = tensor[list(signal_indices)].transpose(2, 1, 0).reshape(tensor.shape[2], -1)
            standard = standardize_fnirs_record(fnirs_raw, sample_rate_hz=47.62, contract=contract)
            duration = min(eeg.shape[0] / 1000.0, fnirs_raw.shape[0] / 47.62)
            segment = _segment_features(
                "refed", subject_name, key, key,
                eeg[: int(duration * 1000.0)], 1000.0,
                standard.values[: int(duration * 47.62)], 47.62, bands,
            )
            if segment is not None:
                segments.append(segment)
            inventory.append({
                "dataset_id": "refed", "subject": subject_name, "record": key,
                "eeg_shape": list(eeg.shape), "fnirs_shape": list(fnirs_raw.shape),
                "duration_s": float(duration),
                "fnirs_repaired_samples": int(sum(standard.state.repaired_nonfinite)),
                "fnirs_residual_drift_sd_per_min": float(standard.quality["residual_drift_sd_per_min_median"]),
            })
    return segments, inventory, inputs


def _read_edf_channels(path: Path) -> tuple[np.ndarray, float, list[str]]:
    with path.open("rb") as handle:
        fixed = handle.read(256)
        header_bytes = int(fixed[184:192].decode().strip())
        records = int(fixed[236:244].decode().strip())
        duration = float(fixed[244:252].decode().strip())
        signal_count = int(fixed[252:256].decode().strip())
        signal_header = handle.read(header_bytes - 256)
    cursor = 0
    widths = (16, 80, 8, 8, 8, 8, 8, 80, 8, 32)
    fields: list[list[str]] = []
    for width in widths:
        fields.append([
            signal_header[cursor + index * width:cursor + (index + 1) * width].decode(errors="replace").strip()
            for index in range(signal_count)
        ])
        cursor += width * signal_count
    labels = fields[0]
    physical_min = np.asarray([float(value) for value in fields[3]])
    physical_max = np.asarray([float(value) for value in fields[4]])
    digital_min = np.asarray([float(value) for value in fields[5]])
    digital_max = np.asarray([float(value) for value in fields[6]])
    samples_per_record = np.asarray([int(value) for value in fields[8]], dtype=int)
    candidate_indices = [
        index for index, label in enumerate(labels)
        if "ANNOTATION" not in label.upper() and label.upper() != "A64"
    ]
    candidate_rates = [samples_per_record[index] / duration for index in candidate_indices]
    eeg_rate = float(max(candidate_rates))
    eeg_indices = candidate_indices
    total_samples = int(samples_per_record.sum())
    raw = np.memmap(path, dtype="<i2", mode="r", offset=header_bytes, shape=(records, total_samples))
    offsets = np.concatenate(([0], np.cumsum(samples_per_record)))
    channels = []
    for index in eeg_indices:
        digital = np.asarray(raw[:, offsets[index]:offsets[index + 1]], dtype=np.float64).reshape(-1)
        scale = (physical_max[index] - physical_min[index]) / max(digital_max[index] - digital_min[index], 1.0)
        physical = (digital - digital_min[index]) * scale + physical_min[index]
        channel_rate = samples_per_record[index] / duration
        if abs(channel_rate - eeg_rate) > 1e-6:
            up = int(round(eeg_rate))
            down = int(round(channel_rate))
            physical = resample_poly(physical, up, down)
        channels.append(physical.astype(np.float32, copy=False))
    common_length = min(len(channel) for channel in channels)
    return np.stack([channel[:common_length] for channel in channels], axis=1).astype(np.float32), eeg_rate, [labels[index] for index in eeg_indices]


def _read_annotation_onsets(path: Path) -> np.ndarray:
    onsets = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 2:
            try:
                onsets.append(float(parts[1]))
            except ValueError:
                pass
    return np.asarray(onsets[::3], dtype=np.float64)


def _read_etg_csv(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    data_line = next(index for index, line in enumerate(lines) if line.strip() == "Data")
    sampling_line = next(line for line in lines[:data_line] if line.startswith("Sampling Period[s]"))
    sample_period = float(next(csv.reader([sampling_line]))[1])
    reader = csv.reader(lines[data_line + 1:])
    header = next(reader)
    channel_indices = [index for index, name in enumerate(header) if re.fullmatch(r"CH\d+", name.strip())]
    mark_index = next(index for index, name in enumerate(header) if name.strip() == "Mark")
    values: list[list[float]] = []
    marks: list[int] = []
    for row in reader:
        if len(row) <= max(max(channel_indices), mark_index):
            continue
        try:
            values.append([float(row[index]) for index in channel_indices])
            marks.append(int(float(row[mark_index] or 0)))
        except ValueError:
            continue
    return np.asarray(values, dtype=np.float64), np.asarray(marks, dtype=int), 1.0 / sample_period


def _read_xlsx_types(path: Path) -> list[str]:
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared_root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = ["".join(node.itertext()) for node in shared_root.findall("x:si", namespace)]
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    values: list[str] = []
    for row in sheet.findall(".//x:row", namespace):
        if int(row.attrib.get("r", "0")) <= 1:
            continue
        cell = next((cell for cell in row.findall("x:c", namespace) if cell.attrib.get("r", "").startswith("B")), None)
        if cell is None:
            continue
        value = cell.find("x:v", namespace)
        if value is None or value.text is None:
            continue
        values.append(shared[int(value.text)] if cell.attrib.get("t") == "s" else value.text)
    return values


def _visual_parts(subject_dir: Path) -> list[tuple[str, Path, Path, list[Path]]]:
    oxy_files = sorted((subject_dir / "fNIRS").glob("*Probe1_Oxy.csv"))
    output = []
    for oxy in oxy_files:
        prefix = oxy.name.replace("_Probe1_Oxy.csv", "")
        suffix = prefix[len(subject_dir.name):].strip("_")
        token = suffix.lower() if suffix else ""
        raw_dir = subject_dir / "EEG" / "raw"
        candidates = sorted(raw_dir.glob(f"{subject_dir.name}*{token}*.edf")) if token else sorted(raw_dir.glob(f"{subject_dir.name}.edf"))
        if not candidates:
            candidates = sorted(raw_dir.glob("*.edf"))
        edf = candidates[0]
        annotation = edf.with_name(edf.stem + "_annotations.txt")
        files = [
            subject_dir / "fNIRS" / f"{prefix}_Probe1_Oxy.csv",
            subject_dir / "fNIRS" / f"{prefix}_Probe1_Deoxy.csv",
            subject_dir / "fNIRS" / f"{prefix}_Probe2_Oxy.csv",
            subject_dir / "fNIRS" / f"{prefix}_Probe2_Deoxy.csv",
        ]
        output.append((prefix, edf, annotation, files))
    return output


def load_visual(
    root: Path, subjects: Sequence[str], cfg: Mapping[str, Any], bands: Mapping[str, Sequence[float]]
) -> tuple[list[Segment], list[dict[str, Any]], list[dict[str, Any]]]:
    contract = DATASET_FNIRS_CONTRACTS["visual_cognitive_motivation"]["oxy_deoxy"]
    duration = float(cfg["trial_duration_s"])
    segments: list[Segment] = []
    inventory: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for subject_value in subjects:
        subject = str(subject_value)
        directory = root / subject
        types_path = directory / f"{subject}_type.xlsx"
        labels = _read_xlsx_types(types_path)
        inputs.append(_input_record(types_path, "trial_types", "visual_cognitive_motivation", subject))
        label_offset = 0
        for part_index, (prefix, edf_path, annotation_path, fnirs_paths) in enumerate(_visual_parts(directory)):
            inputs.extend((
                _input_record(edf_path, "eeg_raw_edf", "visual_cognitive_motivation", subject),
                _input_record(annotation_path, "eeg_event_onsets", "visual_cognitive_motivation", subject),
            ))
            for path in fnirs_paths:
                inputs.append(_input_record(path, "fnirs_etg_csv", "visual_cognitive_motivation", subject))
            eeg, eeg_fs, _ = _read_edf_channels(edf_path)
            eeg_onsets = _read_annotation_onsets(annotation_path)
            arrays = [_read_etg_csv(path) for path in fnirs_paths]
            length = min(len(item[0]) for item in arrays)
            fnirs_raw = np.concatenate([item[0][:length] for item in arrays], axis=1)
            marks = arrays[0][1][:length]
            fnirs_fs = float(arrays[0][2])
            fnirs_onsets = np.flatnonzero(marks == 1) / fnirs_fs
            standard = standardize_fnirs_record(fnirs_raw, sample_rate_hz=fnirs_fs, contract=contract)
            common = min(len(eeg_onsets), len(fnirs_onsets), max(len(labels) - label_offset, 0))
            for trial in range(common):
                condition = labels[label_offset + trial]
                segment = _segment_features(
                    "visual_cognitive_motivation", subject, f"part_{part_index}_trial_{trial:03d}", condition,
                    _slice_seconds(eeg, eeg_onsets[trial] * 1000.0, duration, eeg_fs), eeg_fs,
                    _slice_seconds(standard.values, fnirs_onsets[trial] * 1000.0, duration, fnirs_fs), fnirs_fs,
                    bands,
                )
                if segment is not None:
                    segments.append(segment)
            inventory.append({
                "dataset_id": "visual_cognitive_motivation", "subject": subject, "record": prefix,
                "eeg_shape": list(eeg.shape), "fnirs_shape": list(fnirs_raw.shape),
                "eeg_events": int(len(eeg_onsets)), "fnirs_events": int(len(fnirs_onsets)),
                "paired_events": int(common),
                "fnirs_repaired_samples": int(sum(standard.state.repaired_nonfinite)),
                "fnirs_residual_drift_sd_per_min": float(standard.quality["residual_drift_sd_per_min_median"]),
            })
            label_offset += common
            del eeg, arrays, fnirs_raw, standard
            gc.collect()
    return segments, inventory, inputs


def load_simultaneous(
    root: Path, subjects: Sequence[int], cfg: Mapping[str, Any], bands: Mapping[str, Sequence[float]]
) -> tuple[list[Segment], list[dict[str, Any]], list[dict[str, Any]]]:
    contract = DATASET_FNIRS_CONTRACTS["simultaneous_eeg_nirs"]["oxy_deoxy"]
    duration = float(cfg["trial_duration_s"])
    task = str(cfg["task"])
    segments: list[Segment] = []
    inventory: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for subject in subjects:
        subject_name = str(subject)
        eeg_loader = SimultaneousCognitiveLoader(str(root), task=task, subject_ids=[int(subject)], modality="eeg")
        oxy_loader = SimultaneousCognitiveLoader(str(root), task=task, subject_ids=[int(subject)], modality="fnirs", fnirs_signal="oxy")
        deoxy_loader = SimultaneousCognitiveLoader(str(root), task=task, subject_ids=[int(subject)], modality="fnirs", fnirs_signal="deoxy")
        for modality in ("eeg", "fnirs"):
            cnt_path, marker_path = eeg_loader._get_file_paths(int(subject), modality)
            inputs.extend((
                _input_record(cnt_path, f"{modality}_continuous", "simultaneous_eeg_nirs", subject_name),
                _input_record(marker_path, f"{modality}_markers", "simultaneous_eeg_nirs", subject_name),
            ))
        eeg, eeg_markers, eeg_info = eeg_loader.load_subject_data(int(subject), "eeg")
        oxy, fnirs_markers, fnirs_info = oxy_loader.load_subject_data(int(subject), "fnirs")
        deoxy, _, _ = deoxy_loader.load_subject_data(int(subject), "fnirs")
        length = min(len(oxy), len(deoxy))
        fnirs_raw = np.concatenate((oxy[:length], deoxy[:length]), axis=1)
        standard = standardize_fnirs_record(fnirs_raw, sample_rate_hz=float(fnirs_info["fs"]), contract=contract)
        eeg_times = np.asarray(eeg_markers["time"], dtype=np.float64)
        fnirs_times = np.asarray(fnirs_markers["time"], dtype=np.float64)
        eeg_labels = np.argmax(np.asarray(eeg_markers["y"]), axis=0)
        fnirs_labels = np.argmax(np.asarray(fnirs_markers["y"]), axis=0)
        class_names = [str(value) for value in eeg_markers.get("className", [])]
        common = min(len(eeg_times), len(fnirs_times))
        for trial in range(common):
            if int(eeg_labels[trial]) != int(fnirs_labels[trial]):
                continue
            label = int(eeg_labels[trial])
            condition = class_names[label] if label < len(class_names) else str(label)
            segment = _segment_features(
                "simultaneous_eeg_nirs", subject_name, f"trial_{trial:03d}", condition,
                _slice_seconds(eeg, eeg_times[trial], duration, float(eeg_info["fs"])), float(eeg_info["fs"]),
                _slice_seconds(standard.values, fnirs_times[trial], duration, float(fnirs_info["fs"])), float(fnirs_info["fs"]),
                bands,
            )
            if segment is not None:
                segments.append(segment)
        inventory.append({
            "dataset_id": "simultaneous_eeg_nirs", "subject": subject_name, "record": task,
            "eeg_shape": list(eeg.shape), "fnirs_shape": list(fnirs_raw.shape),
            "eeg_events": int(len(eeg_times)), "fnirs_events": int(len(fnirs_times)),
            "paired_events": int(common),
            "initial_marker_offset_ms": float(fnirs_times[0] - eeg_times[0]) if common else None,
            "fnirs_repaired_samples": int(sum(standard.state.repaired_nonfinite)),
            "fnirs_residual_drift_sd_per_min": float(standard.quality["residual_drift_sd_per_min_median"]),
        })
    return segments, inventory, inputs


def _condition_matrix(conditions: Sequence[str], vocabulary: Sequence[str]) -> np.ndarray:
    mapping = {condition: index for index, condition in enumerate(vocabulary)}
    matrix = np.zeros((len(conditions), len(vocabulary)), dtype=np.float64)
    for row, condition in enumerate(conditions):
        if str(condition) in mapping:
            matrix[row, mapping[str(condition)]] = 1.0
    return matrix


def _fit_pca(train_segments: Sequence[Segment], modality: str, dimension: int) -> PCA:
    matrix = np.concatenate([getattr(segment, modality) for segment in train_segments], axis=0)
    count = min(int(dimension), matrix.shape[0] - 1, matrix.shape[1])
    return PCA(n_components=max(count, 1), whiten=True, svd_solver="full").fit(matrix)


def _lag_rows(
    segments: Sequence[Segment], eeg_pca: PCA, fnirs_pca: PCA, lag: int, history: int,
    condition_vocabulary: Sequence[str],
) -> LagRows:
    eeg_targets: list[np.ndarray] = []
    fnirs_targets: list[np.ndarray] = []
    eeg_bases: list[np.ndarray] = []
    fnirs_bases: list[np.ndarray] = []
    groups: list[str] = []
    conditions: list[str] = []
    for segment in segments:
        eeg_scores = eeg_pca.transform(segment.eeg)
        fnirs_scores = fnirs_pca.transform(segment.fnirs)
        length = min(len(eeg_scores), len(fnirs_scores))
        for time_index in range(history, length - int(lag)):
            fnirs_index = time_index + int(lag)
            eeg_history = np.concatenate([eeg_scores[time_index - offset] for offset in range(1, history + 1)])
            fnirs_history = np.concatenate([fnirs_scores[fnirs_index - offset] for offset in range(1, history + 1)])
            eeg_phase = time_index / max(length - 1, 1)
            fnirs_phase = fnirs_index / max(length - 1, 1)
            eeg_bases.append(np.concatenate((eeg_history, [eeg_phase, eeg_phase ** 2, math.sin(math.pi * eeg_phase), math.cos(math.pi * eeg_phase)])))
            fnirs_bases.append(np.concatenate((fnirs_history, [fnirs_phase, fnirs_phase ** 2, math.sin(math.pi * fnirs_phase), math.cos(math.pi * fnirs_phase)])))
            eeg_targets.append(segment.eeg[time_index])
            fnirs_targets.append(segment.fnirs[fnirs_index])
            groups.append(f"{segment.subject}:{segment.segment_id}")
            conditions.append(segment.condition)
    condition_features = _condition_matrix(conditions, condition_vocabulary)
    return LagRows(
        eeg_target=np.asarray(eeg_targets),
        fnirs_target=np.asarray(fnirs_targets),
        eeg_base=np.concatenate((np.asarray(eeg_bases), condition_features), axis=1),
        fnirs_base=np.concatenate((np.asarray(fnirs_bases), condition_features), axis=1),
        groups=np.asarray(groups),
        conditions=np.asarray(conditions),
    )


def _ridge_predictions(
    train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    scaler = StandardScaler().fit(train_x)
    model = Ridge(alpha=float(alpha)).fit(scaler.transform(train_x), train_y)
    return model.predict(scaler.transform(train_x)), model.predict(scaler.transform(val_x)), {
        "predictor_dimension": int(train_x.shape[1]), "target_dimension": int(train_y.shape[1])
    }


def _standardize_latent(train: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(train)
    return scaler.transform(train), scaler.transform(val)


def _safe_r2(truth: np.ndarray, prediction: np.ndarray, reference: np.ndarray | None = None) -> float:
    if reference is None:
        reference = np.zeros_like(truth)
    denominator = float(np.sum((truth - reference) ** 2))
    numerator = float(np.sum((truth - prediction) ** 2))
    return float(1.0 - numerator / denominator) if denominator > 1e-12 else float("nan")


def _corr_columns(first: np.ndarray, second: np.ndarray) -> list[float]:
    output = []
    for column in range(min(first.shape[1], second.shape[1])):
        a = first[:, column]
        b = second[:, column]
        if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
            output.append(0.0)
        else:
            output.append(float(np.corrcoef(a, b)[0, 1]))
    return output


def _group_bootstrap(
    groups: np.ndarray, target: np.ndarray, base_prediction: np.ndarray, combined_prediction: np.ndarray,
    iterations: int, rng: np.random.Generator,
) -> dict[str, float]:
    if int(iterations) <= 0:
        return {
            "innovation_ci_low": float("nan"), "innovation_ci_high": float("nan"),
            "total_ci_low": float("nan"), "total_ci_high": float("nan"),
        }
    unique = np.unique(groups)
    innovation_values = []
    total_values = []
    mean_reference = np.zeros_like(target)
    for _ in range(int(iterations)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        truth = target[indices]
        base = base_prediction[indices]
        combined = combined_prediction[indices]
        sse_zero = float(np.sum((truth - mean_reference[indices]) ** 2))
        sse_base = float(np.sum((truth - base) ** 2))
        sse_combined = float(np.sum((truth - combined) ** 2))
        if sse_base > 1e-12:
            innovation_values.append(1.0 - sse_combined / sse_base)
        if sse_zero > 1e-12:
            total_values.append((sse_base - sse_combined) / sse_zero)
    return {
        "innovation_ci_low": float(np.quantile(innovation_values, 0.025)),
        "innovation_ci_high": float(np.quantile(innovation_values, 0.975)),
        "total_ci_low": float(np.quantile(total_values, 0.025)),
        "total_ci_high": float(np.quantile(total_values, 0.975)),
    }


def _shift_latent_by_group(latent: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shifted = latent.copy()
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        if len(indices) < 2:
            continue
        offset = int(rng.integers(1, len(indices)))
        shifted[indices] = latent[np.roll(indices, offset)]
    return shifted


def _direction_metrics(
    direction: str, truth: np.ndarray, base_prediction: np.ndarray, residual_truth: np.ndarray,
    latent_train: np.ndarray, latent_val: np.ndarray, residual_train: np.ndarray,
    groups: np.ndarray, alpha: float, bootstrap_iterations: int, null_iterations: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    decoder = Ridge(alpha=float(alpha)).fit(latent_train, residual_train)
    residual_prediction = decoder.predict(latent_val)
    combined = base_prediction + residual_prediction
    sse_zero = float(np.sum(truth ** 2))
    sse_base = float(np.sum(residual_truth ** 2))
    sse_combined = float(np.sum((truth - combined) ** 2))
    innovation_fraction = 1.0 - sse_combined / sse_base if sse_base > 1e-12 else float("nan")
    total_fraction = (sse_base - sse_combined) / sse_zero if sse_zero > 1e-12 else float("nan")
    information_gain = 0.5 * math.log(max(sse_base, 1e-12) / max(sse_combined, 1e-12)) / max(truth.shape[1], 1)
    null_rows = []
    null_fractions = []
    for iteration in range(int(null_iterations)):
        shifted = _shift_latent_by_group(latent_val, groups, rng)
        shifted_combined = base_prediction + decoder.predict(shifted)
        shifted_sse = float(np.sum((truth - shifted_combined) ** 2))
        shifted_fraction = 1.0 - shifted_sse / sse_base if sse_base > 1e-12 else float("nan")
        null_fractions.append(shifted_fraction)
        null_rows.append({"direction": direction, "iteration": iteration, "innovation_fraction": shifted_fraction})
    empirical_p = (
        (1.0 + sum(value >= innovation_fraction for value in null_fractions)) / (len(null_fractions) + 1.0)
        if null_fractions else float("nan")
    )
    bootstrap = _group_bootstrap(
        groups, truth, base_prediction, combined, bootstrap_iterations, rng
    )
    return {
        "direction": direction,
        "target_dimension": int(truth.shape[1]),
        "validation_rows": int(truth.shape[0]),
        "self_history_r2": _safe_r2(truth, base_prediction),
        "combined_r2": _safe_r2(truth, combined),
        "shared_innovation_fraction": innovation_fraction,
        "shared_innovation_fraction_clipped": max(0.0, innovation_fraction),
        "shared_total_variance_fraction": total_fraction,
        "shared_total_variance_fraction_clipped": max(0.0, total_fraction),
        "gaussian_log_mse_gain_nats_per_feature": information_gain,
        "alignment_null_median": float(np.median(null_fractions)) if null_fractions else float("nan"),
        "alignment_null_p95": float(np.quantile(null_fractions, 0.95)) if null_fractions else float("nan"),
        "alignment_empirical_p": float(empirical_p),
        **bootstrap,
    }, null_rows


def evaluate_fold_lag(
    dataset_id: str, train_subject: str, validation_subject: str,
    train_segments: Sequence[Segment], val_segments: Sequence[Segment], lag: int,
    config: Mapping[str, Any], rng: np.random.Generator, compute_uncertainty: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model_cfg = config["model"]
    condition_vocabulary = sorted({segment.condition for segment in [*train_segments, *val_segments]})
    eeg_pca = _fit_pca(train_segments, "eeg", int(model_cfg["residual_pca_dimension"]))
    fnirs_pca = _fit_pca(train_segments, "fnirs", int(model_cfg["residual_pca_dimension"]))
    train = _lag_rows(train_segments, eeg_pca, fnirs_pca, lag, int(model_cfg["self_history_s"]), condition_vocabulary)
    val = _lag_rows(val_segments, eeg_pca, fnirs_pca, lag, int(model_cfg["self_history_s"]), condition_vocabulary)
    alpha = float(model_cfg["ridge_alpha"])
    train_eeg_base, val_eeg_base, _ = _ridge_predictions(train.eeg_base, train.eeg_target, val.eeg_base, alpha)
    train_fnirs_base, val_fnirs_base, _ = _ridge_predictions(train.fnirs_base, train.fnirs_target, val.fnirs_base, alpha)
    train_eeg_residual = train.eeg_target - train_eeg_base
    train_fnirs_residual = train.fnirs_target - train_fnirs_base
    val_eeg_residual = val.eeg_target - val_eeg_base
    val_fnirs_residual = val.fnirs_target - val_fnirs_base
    eeg_residual_pca = _fit_residual_pca(train_eeg_residual, int(model_cfg["residual_pca_dimension"]), 11)
    fnirs_residual_pca = _fit_residual_pca(train_fnirs_residual, int(model_cfg["residual_pca_dimension"]), 12)
    train_eeg_comp = eeg_residual_pca.transform(train_eeg_residual)
    train_fnirs_comp = fnirs_residual_pca.transform(train_fnirs_residual)
    val_eeg_comp = eeg_residual_pca.transform(val_eeg_residual)
    val_fnirs_comp = fnirs_residual_pca.transform(val_fnirs_residual)
    shared_dimension = min(
        int(model_cfg["shared_state_dimension"]), train_eeg_comp.shape[1], train_fnirs_comp.shape[1]
    )
    cca = CCA(
        n_components=shared_dimension, scale=False,
        max_iter=int(model_cfg["cca_max_iterations"]), tol=float(model_cfg["cca_tolerance"]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cca.fit(train_eeg_comp, train_fnirs_comp)
    train_z_eeg, train_z_fnirs = cca.transform(train_eeg_comp, train_fnirs_comp)
    val_z_eeg, val_z_fnirs = cca.transform(val_eeg_comp, val_fnirs_comp)
    train_z_eeg, val_z_eeg = _standardize_latent(train_z_eeg, val_z_eeg)
    train_z_fnirs, val_z_fnirs = _standardize_latent(train_z_fnirs, val_z_fnirs)
    train_z_joint = 0.5 * (train_z_eeg + train_z_fnirs)
    val_z_joint = 0.5 * (val_z_eeg + val_z_fnirs)
    correlations_train = _corr_columns(train_z_eeg, train_z_fnirs)
    correlations_val = _corr_columns(val_z_eeg, val_z_fnirs)
    uncertainty_cfg = config["uncertainty"]
    bootstrap_iterations = int(uncertainty_cfg["segment_bootstrap_iterations"]) if compute_uncertainty else 0
    null_iterations = int(uncertainty_cfg["alignment_null_iterations"]) if compute_uncertainty else 0
    fnirs_metrics, fnirs_null = _direction_metrics(
        "eeg_to_fnirs", val.fnirs_target, val_fnirs_base, val_fnirs_residual,
        train_z_eeg, val_z_eeg, train_fnirs_residual, val.groups, alpha,
        bootstrap_iterations, null_iterations, rng,
    )
    eeg_metrics, eeg_null = _direction_metrics(
        "fnirs_to_eeg", val.eeg_target, val_eeg_base, val_eeg_residual,
        train_z_fnirs, val_z_fnirs, train_eeg_residual, val.groups, alpha,
        bootstrap_iterations, null_iterations, rng,
    )
    joint_fnirs_metrics, _ = _direction_metrics(
        "joint_to_fnirs", val.fnirs_target, val_fnirs_base, val_fnirs_residual,
        train_z_joint, val_z_joint, train_fnirs_residual, val.groups, alpha,
        bootstrap_iterations, 0, rng,
    )
    joint_eeg_metrics, _ = _direction_metrics(
        "joint_to_eeg", val.eeg_target, val_eeg_base, val_eeg_residual,
        train_z_joint, val_z_joint, train_eeg_residual, val.groups, alpha,
        bootstrap_iterations, 0, rng,
    )
    common = {
        "dataset_id": dataset_id,
        "train_subject": train_subject,
        "validation_subject": validation_subject,
        "lag_s": int(lag),
        "train_rows": int(train.eeg_target.shape[0]),
        "validation_rows": int(val.eeg_target.shape[0]),
        "shared_state_dimension": int(shared_dimension),
        "mean_train_canonical_correlation": float(np.mean(correlations_train)),
        "mean_validation_canonical_correlation": float(np.mean(correlations_val)),
        "minimum_validation_canonical_correlation": float(np.min(correlations_val)),
        "eeg_feature_dimension": int(train.eeg_target.shape[1]),
        "fnirs_feature_dimension": int(train.fnirs_target.shape[1]),
        "eeg_residual_pca_retained_variance": float(np.sum(eeg_residual_pca.explained_variance_ratio_)),
        "fnirs_residual_pca_retained_variance": float(np.sum(fnirs_residual_pca.explained_variance_ratio_)),
    }
    rows = [
        common | fnirs_metrics, common | eeg_metrics,
        common | joint_fnirs_metrics, common | joint_eeg_metrics,
    ]
    null_rows = [common | row for row in [*fnirs_null, *eeg_null]]
    return {"rows": rows, "correlations_train": correlations_train, "correlations_val": correlations_val}, null_rows


def _fit_residual_pca(values: np.ndarray, dimension: int, seed: int) -> PCA:
    count = min(int(dimension), values.shape[0] - 1, values.shape[1])
    return PCA(n_components=max(count, 1), whiten=True, svd_solver="full", random_state=seed).fit(values)


def summarize(rows: Sequence[Mapping[str, Any]], primary_lag: int) -> list[dict[str, Any]]:
    output = []
    datasets = sorted({str(row["dataset_id"]) for row in rows})
    for dataset_id in datasets:
        primary = [row for row in rows if row["dataset_id"] == dataset_id and int(row["lag_s"]) == int(primary_lag)]
        direction_summaries: dict[str, dict[str, Any]] = {}
        for direction in DIRECTIONS:
            selected = [row for row in primary if row["direction"] == direction]
            direction_summaries[direction] = {
                "folds": len(selected),
                "shared_innovation_fraction_median": float(np.median([row["shared_innovation_fraction"] for row in selected])),
                "shared_innovation_fraction_min": float(np.min([row["shared_innovation_fraction"] for row in selected])),
                "shared_innovation_fraction_max": float(np.max([row["shared_innovation_fraction"] for row in selected])),
                "shared_total_variance_fraction_median": float(np.median([row["shared_total_variance_fraction"] for row in selected])),
                "information_gain_nats_per_feature_median": float(np.median([row["gaussian_log_mse_gain_nats_per_feature"] for row in selected])),
                "null_exceedance_fold_count": int(sum(row["alignment_empirical_p"] <= 0.05 for row in selected)),
            }
        eeg_fraction = max(0.0, direction_summaries["fnirs_to_eeg"]["shared_innovation_fraction_median"])
        fnirs_fraction = max(0.0, direction_summaries["eeg_to_fnirs"]["shared_innovation_fraction_median"])
        eeg_total = max(0.0, direction_summaries["fnirs_to_eeg"]["shared_total_variance_fraction_median"])
        fnirs_total = max(0.0, direction_summaries["eeg_to_fnirs"]["shared_total_variance_fraction_median"])
        joint_summaries: dict[str, dict[str, float]] = {}
        for direction in JOINT_DIRECTIONS:
            selected = [row for row in primary if row["direction"] == direction]
            joint_summaries[direction] = {
                "shared_innovation_fraction_median": float(np.median([row["shared_innovation_fraction"] for row in selected])),
                "shared_total_variance_fraction_median": float(np.median([row["shared_total_variance_fraction"] for row in selected])),
            }
        joint_eeg = max(0.0, joint_summaries["joint_to_eeg"]["shared_innovation_fraction_median"])
        joint_fnirs = max(0.0, joint_summaries["joint_to_fnirs"]["shared_innovation_fraction_median"])
        joint_eeg_total = max(0.0, joint_summaries["joint_to_eeg"]["shared_total_variance_fraction_median"])
        joint_fnirs_total = max(0.0, joint_summaries["joint_to_fnirs"]["shared_total_variance_fraction_median"])
        output.append({
            "dataset_id": dataset_id,
            "primary_lag_s": int(primary_lag),
            "subjects": 2,
            "eeg_innovation_shared_fraction": eeg_fraction,
            "fnirs_innovation_shared_fraction": fnirs_fraction,
            "balanced_shared_innovation_fraction": min(eeg_fraction, fnirs_fraction),
            "eeg_total_feature_variance_shared_fraction": eeg_total,
            "fnirs_total_feature_variance_shared_fraction": fnirs_total,
            "balanced_total_feature_variance_shared_fraction": min(eeg_total, fnirs_total),
            "joint_teacher_eeg_innovation_fraction": joint_eeg,
            "joint_teacher_fnirs_innovation_fraction": joint_fnirs,
            "joint_teacher_balanced_innovation_fraction": min(joint_eeg, joint_fnirs),
            "joint_teacher_eeg_total_feature_variance_fraction": joint_eeg_total,
            "joint_teacher_fnirs_total_feature_variance_fraction": joint_fnirs_total,
            "joint_teacher_balanced_total_feature_variance_fraction": min(joint_eeg_total, joint_fnirs_total),
            "eeg_direction": direction_summaries["fnirs_to_eeg"],
            "fnirs_direction": direction_summaries["eeg_to_fnirs"],
            "joint_directions": joint_summaries,
        })
    return output


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def plot_results(rows: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]], run_dir: Path) -> list[dict[str, Any]]:
    _style()
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    artifacts = []
    datasets = [item["dataset_id"] for item in summaries]
    x = np.arange(len(datasets), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
    width = 0.34
    for index, direction in enumerate(DIRECTIONS):
        values = []
        for dataset_id in datasets:
            selected = [row for row in rows if row["dataset_id"] == dataset_id and row["direction"] == direction and int(row["lag_s"]) == 5]
            values.append(float(np.median([row["shared_innovation_fraction"] for row in selected])))
            for fold_index, row in enumerate(selected):
                axes[0].scatter(x[datasets.index(dataset_id)] + (index - 0.5) * width + (fold_index - 0.5) * 0.035,
                                row["shared_innovation_fraction"], color="black", s=10, zorder=3)
        axes[0].bar(x + (index - 0.5) * width, values, width=width, color=OKABE_ITO[index], label=DIRECTIONS[direction], alpha=0.88)
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_xticks(x, [DISPLAY_NAMES[item] for item in datasets], rotation=20, ha="right")
    axes[0].set_ylabel("Shared fraction of modality innovation")
    axes[0].set_title("A  Cross-inferable state at 5 s lag")
    axes[0].legend(frameon=False)
    balanced = [item["balanced_shared_innovation_fraction"] for item in summaries]
    joint_balanced = [item["joint_teacher_balanced_innovation_fraction"] for item in summaries]
    joint_total = [item["joint_teacher_balanced_total_feature_variance_fraction"] for item in summaries]
    narrow = 0.24
    axes[1].bar(x - narrow, balanced, width=narrow, color=OKABE_ITO[2], label="Cross-inferable")
    axes[1].bar(x, joint_balanced, width=narrow, color=OKABE_ITO[4], label="Joint innovation")
    axes[1].bar(x + narrow, joint_total, width=narrow, color=OKABE_ITO[5], label="Joint total variance")
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_xticks(x, [DISPLAY_NAMES[item] for item in datasets], rotation=20, ha="right")
    axes[1].set_ylabel("Balanced fraction of modality innovation")
    axes[1].set_title("B  Strict estimate vs joint-input ceiling")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    artifacts.extend(_save_figure(fig, figure_dir / "shared_information_summary"))
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True, sharey=True)
    for axis, dataset_id in zip(axes.flat, datasets):
        for index, direction in enumerate(DIRECTIONS):
            selected = [row for row in rows if row["dataset_id"] == dataset_id and row["direction"] == direction]
            lags = sorted({int(row["lag_s"]) for row in selected})
            medians = [float(np.median([row["shared_innovation_fraction"] for row in selected if int(row["lag_s"]) == lag])) for lag in lags]
            minima = [float(np.min([row["shared_innovation_fraction"] for row in selected if int(row["lag_s"]) == lag])) for lag in lags]
            maxima = [float(np.max([row["shared_innovation_fraction"] for row in selected if int(row["lag_s"]) == lag])) for lag in lags]
            axis.plot(lags, medians, color=OKABE_ITO[index], marker=("o" if index == 0 else "s"), markersize=3, label=DIRECTIONS[direction])
            axis.fill_between(lags, minima, maxima, color=OKABE_ITO[index], alpha=0.15)
        axis.axhline(0, color="#555555", linewidth=0.7)
        axis.axvline(5, color="#777777", linewidth=0.7, linestyle="--")
        axis.set_title(DISPLAY_NAMES[dataset_id])
        axis.set_xlabel("EEG-leading lag (s)")
        axis.set_ylabel("Shared innovation fraction")
    axes[0, 0].legend(frameon=False, loc="best")
    fig.tight_layout()
    artifacts.extend(_save_figure(fig, figure_dir / "lag_profiles"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for direction_index, direction in enumerate(DIRECTIONS):
        axis = axes[direction_index]
        for dataset_index, dataset_id in enumerate(datasets):
            selected = [row for row in rows if row["dataset_id"] == dataset_id and row["direction"] == direction and int(row["lag_s"]) == 5]
            history = float(np.median([row["self_history_r2"] for row in selected]))
            shared = float(np.median([row["shared_total_variance_fraction"] for row in selected]))
            combined = float(np.median([row["combined_r2"] for row in selected]))
            axis.bar(dataset_index, history, color="#999999", label="Self-history + phase" if dataset_index == 0 else None)
            axis.bar(dataset_index, shared, bottom=history, color=OKABE_ITO[2], label="Cross-modal shared increment" if dataset_index == 0 else None)
            axis.plot([dataset_index - 0.32, dataset_index + 0.32], [combined, combined], color="black", linewidth=1.2)
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.set_xticks(x, [DISPLAY_NAMES[item] for item in datasets], rotation=20, ha="right")
        axis.set_ylabel("Held-out feature variance fraction")
        axis.set_title(DIRECTIONS[direction])
        axis.legend(frameon=False)
    fig.tight_layout()
    artifacts.extend(_save_figure(fig, figure_dir / "variance_attribution"))
    plt.close(fig)
    return artifacts


def _save_figure(fig: plt.Figure, stem: Path) -> list[dict[str, Any]]:
    output = []
    for suffix, kwargs in ((".svg", {}), (".png", {"dpi": 300})):
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        output.append({"path": str(path), "sha256": _sha256(path)})
    return output


def _report_markdown(summaries: Sequence[Mapping[str, Any]], run_dir: Path) -> str:
    lines = [
        "# Cross-dataset shared neural state diagnostic", "",
        "Two subjects per dataset; reciprocal one-subject train/one-subject validation folds; diagnostic only.", "",
        "The primary number is the fraction of self-history residual (innovation) reconstructed from the other modality's CCA state at a fixed 5 s EEG-leading lag. "
        "The total-feature fraction is the corresponding improvement relative to total standardized raw-data-derived feature variance.", "",
        "| Dataset | EEG innovation from fNIRS | fNIRS innovation from EEG | Balanced cross-inferable | Joint innovation ceiling | Strict total variance | Joint total variance ceiling |", 
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {DISPLAY_NAMES[row['dataset_id']]} | {100 * row['eeg_innovation_shared_fraction']:.2f}% | "
            f"{100 * row['fnirs_innovation_shared_fraction']:.2f}% | "
            f"{100 * row['balanced_shared_innovation_fraction']:.2f}% | "
            f"{100 * row['joint_teacher_balanced_innovation_fraction']:.2f}% | "
            f"{100 * row['balanced_total_feature_variance_shared_fraction']:.2f}% | "
            f"{100 * row['joint_teacher_balanced_total_feature_variance_fraction']:.2f}% |"
        )
    lines.extend([
        "", "## Interpretation boundary", "",
        "- Negative un-clipped fractions mean the cross-modal state generalized worse than the self-history baseline; summary percentages clip them to zero.",
        "- The joint-teacher ceiling uses both modalities to construct the CCA state and is therefore an optimistic compression bound, not cross-modal evidence.",
        "- Bootstrap intervals resample trials/videos within one held-out subject and are not population confidence intervals.",
        "- The Gaussian log-MSE gain is a model-based diagnostic, not a nonparametric mutual-information estimate.",
        "- REFED and Visual use experiment-local read adapters because their registered paired loaders remain planned.",
        "- No physiology-semantic protected subjects were opened, and this run cannot pass E0.",
        "", "## Figures", "",
        "- `figures/shared_information_summary.svg`", "- `figures/lag_profiles.svg`", "- `figures/variance_attribution.svg`", "",
        f"Run directory: `{run_dir}`", "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments/configs/physiology_semantic_tokenizer/cross_dataset_shared_neural_state.yaml"))
    parser.add_argument("--output-dir")
    parser.add_argument("--datasets", nargs="*", choices=list(DISPLAY_NAMES))
    parser.add_argument("--resource-snapshot", default="/tmp/pid_mcm_resources_20260706.json")
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir).resolve() if args.output_dir else (
        REPO_ROOT / "experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity" /
        f"{stamp}_{config['experiment']['name']}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir()
    (run_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    resource_path = Path(args.resource_snapshot)
    if resource_path.exists():
        shutil.copy2(resource_path, run_dir / "resources.json")

    requested = set(args.datasets or config["datasets"].keys())
    bands = config["features"]["eeg_frequency_bands_hz"]
    all_segments: list[Segment] = []
    inventory: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    adapter_reports: dict[str, Any] = {}
    loaders = {
        "eeg_fnirs_single_trial": load_single_trial,
        "refed": load_refed,
        "visual_cognitive_motivation": load_visual,
        "simultaneous_eeg_nirs": load_simultaneous,
    }
    for dataset_id, dataset_cfg in config["datasets"].items():
        if dataset_id not in requested:
            continue
        root = (REPO_ROOT / dataset_cfg["root"]).resolve()
        segments, records, inputs = loaders[dataset_id](root, dataset_cfg["subjects"], dataset_cfg, bands)
        segments, adapter_report = _robust_subject_adapter(
            segments, float(config["features"]["subject_adapter_clip_abs"])
        )
        if len({segment.subject for segment in segments}) != 2:
            raise RuntimeError(f"{dataset_id} did not yield exactly two subjects")
        all_segments.extend(segments)
        inventory.extend(records)
        input_records.extend(inputs)
        adapter_reports[dataset_id] = adapter_report

    seed = int(config["uncertainty"]["seed"])
    rng = np.random.default_rng(seed)
    metric_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    for dataset_id in sorted(requested):
        selected = [segment for segment in all_segments if segment.dataset_id == dataset_id]
        subjects = sorted({segment.subject for segment in selected})
        for validation_subject in subjects:
            train_subject = next(subject for subject in subjects if subject != validation_subject)
            train_segments = [segment for segment in selected if segment.subject == train_subject]
            val_segments = [segment for segment in selected if segment.subject == validation_subject]
            for lag in [int(value) for value in config["model"]["lag_grid_s"]]:
                history = int(config["model"]["self_history_s"])
                train_has_rows = any(min(len(segment.eeg), len(segment.fnirs)) > history + lag for segment in train_segments)
                val_has_rows = any(min(len(segment.eeg), len(segment.fnirs)) > history + lag for segment in val_segments)
                if not (train_has_rows and val_has_rows):
                    continue
                result, fold_null = evaluate_fold_lag(
                    dataset_id, train_subject, validation_subject,
                    train_segments, val_segments, lag, config, rng,
                    compute_uncertainty=(lag == int(config["model"]["primary_lag_s"])),
                )
                metric_rows.extend(result["rows"])
                if lag == int(config["model"]["primary_lag_s"]):
                    null_rows.extend(fold_null)
                gc.collect()
    summaries = summarize(metric_rows, int(config["model"]["primary_lag_s"]))
    figure_artifacts = plot_results(metric_rows, summaries, run_dir)

    _write_csv(run_dir / "dataset_inventory.csv", inventory)
    _write_json(run_dir / "subject_adapter_report.json", adapter_reports)
    _write_csv(run_dir / "lag_metrics.csv", metric_rows)
    _write_csv(run_dir / "alignment_null_metrics.csv", null_rows)
    _write_json(run_dir / "dataset_summary.json", {"datasets": summaries})
    _write_csv(run_dir / "dataset_summary.csv", [{key: value for key, value in row.items() if not isinstance(value, dict)} for row in summaries])
    _write_json(run_dir / "input_manifest.json", {"files": input_records})
    _write_json(run_dir / "metric_registry.json", {
        "schema": SCHEMA,
        "primary": ["shared_innovation_fraction", "shared_total_variance_fraction"],
        "supportive": ["joint_teacher_balanced_innovation_fraction"],
        "secondary": ["gaussian_log_mse_gain_nats_per_feature", "mean_validation_canonical_correlation"],
        "diagnostic": ["self_history_r2", "combined_r2", "alignment_empirical_p", "lag_profile"],
    })
    (run_dir / "decision_protocol.yaml").write_text(yaml.safe_dump({
        "schema": SCHEMA,
        "metric_role": "diagnostic_non_gate",
        "primary_lag_s": int(config["model"]["primary_lag_s"]),
        "primary_estimator": "cross-subject cross-inference of self-history residual through lagged CCA state",
        "shared_fraction_definition": "1 - SSE(self_history_plus_cross_state) / SSE(self_history)",
        "balanced_definition": "minimum of non-negative EEG and fNIRS directional fractions",
        "population_inference": "prohibited with two subjects per dataset",
        "e0_status_change": "prohibited",
    }, sort_keys=False), encoding="utf-8")
    _write_json(run_dir / "evidence_calibration.json", {
        "schema": SCHEMA,
        "outer_split": "reciprocal leave-one-subject-out",
        "segment_bootstrap_iterations": int(config["uncertainty"]["segment_bootstrap_iterations"]),
        "alignment_null_iterations": int(config["uncertainty"]["alignment_null_iterations"]),
        "thresholds": "none; empirical nulls and reciprocal-fold consistency are reported diagnostically",
        "protected_test_used": False,
    })
    _write_json(run_dir / "environment.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "sklearn": __import__("sklearn").__version__,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "git_status_porcelain": subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.splitlines(),
    })
    summary_payload = {
        "schema": SCHEMA,
        "status": "formal_complete_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "subjects_per_dataset": 2,
        "datasets": summaries,
        "figures": [{"path": str(Path(item["path"]).relative_to(run_dir)), "sha256": item["sha256"]} for item in figure_artifacts],
        "protected_test_used": False,
        "interpretation": config["validation"]["interpretation"],
    }
    _write_json(run_dir / "summary.json", summary_payload)
    (run_dir / "summary.md").write_text(_report_markdown(summaries, run_dir), encoding="utf-8")
    _write_json(run_dir / "manifest.json", {
        "schema": SCHEMA, "status": "formal_complete", "metric_role": "diagnostic_non_gate",
        "config": "config.yaml", "summary": "summary.json", "metrics": "lag_metrics.csv",
        "null_metrics": "alignment_null_metrics.csv", "protected_test_used": False,
    })
    return run_dir


if __name__ == "__main__":
    print(run(parse_args()))
