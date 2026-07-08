"""Unified event, label, and cross-modal timing contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Mapping, Sequence
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np


EVENT_ALIGNMENT_SCHEMA = "physiology_event_alignment_v1"


@dataclass(frozen=True)
class CanonicalEvent:
    dataset_id: str
    subject: str
    record_id: str
    event_index: int
    event_type: str
    label: str
    label_index: int | None = None
    eeg_time_ms: float | None = None
    fnirs_time_ms: float | None = None
    onset_ms: float | None = None
    duration_ms: float | None = None
    alignment_role: str = "native"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = EVENT_ALIGNMENT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = _jsonable(payload["metadata"])
        return payload


@dataclass(frozen=True)
class EventAlignmentReport:
    dataset_id: str
    subject: str
    record_id: str
    num_eeg_events: int
    num_fnirs_events: int
    num_aligned_events: int
    alignment_case: str
    label_sequence_match: bool | None
    offset_mean_ms: float | None
    offset_std_ms: float | None
    drift_slope_ms_per_min: float | None
    offset_blocks: tuple[Mapping[str, Any], ...] = ()
    skipped_marker_indices: Mapping[str, Sequence[int]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = EVENT_ALIGNMENT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["offset_blocks"] = [_jsonable(block) for block in payload["offset_blocks"]]
        payload["skipped_marker_indices"] = _jsonable(payload["skipped_marker_indices"])
        payload["metadata"] = _jsonable(payload["metadata"])
        return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalize_class_names(value: Any) -> list[str]:
    if value is None:
        return []
    array = np.asarray(value, dtype=object)
    if array.ndim == 0:
        return [str(array.item())]
    return [str(item) for item in array.ravel().tolist()]


def normalize_marker_targets(marker_y: Any, n_events: int) -> np.ndarray:
    y = np.asarray(marker_y)
    if y.ndim == 2 and y.shape[1] == n_events:
        return y.astype(np.float32, copy=False)
    if y.ndim == 2 and y.shape[0] == n_events:
        return y.T.astype(np.float32, copy=False)
    if y.ndim == 1 and y.shape[0] == n_events:
        unique = list(dict.fromkeys(int(item) for item in y.tolist()))
        matrix = np.zeros((len(unique), n_events), dtype=np.float32)
        lookup = {value: index for index, value in enumerate(unique)}
        for event_index, value in enumerate(y.tolist()):
            matrix[lookup[int(value)], event_index] = 1.0
        return matrix
    return np.ones((1, n_events), dtype=np.float32)


def normalize_marker_struct(marker_struct: Any) -> dict[str, Any]:
    time = np.asarray(getattr(marker_struct, "time"), dtype=np.float64).reshape(-1)
    event = getattr(marker_struct, "event", None)
    event_desc = getattr(event, "desc", None)
    if event_desc is not None:
        event_desc = np.asarray(event_desc).reshape(-1)
    y = normalize_marker_targets(getattr(marker_struct, "y", None), len(time))
    return {
        "time": time,
        "y": y,
        "className": normalize_class_names(getattr(marker_struct, "className", None)),
        "event_desc": event_desc,
    }


def marker_label_indices(marker_info: Mapping[str, Any]) -> np.ndarray:
    y = np.asarray(marker_info.get("y"))
    if y.ndim == 2 and y.shape[1] == len(marker_info.get("time", [])):
        return np.argmax(y, axis=0).astype(np.int64)
    return np.zeros(len(marker_info.get("time", [])), dtype=np.int64)


def marker_label_names(marker_info: Mapping[str, Any]) -> list[str]:
    indices = marker_label_indices(marker_info)
    class_names = [str(item) for item in marker_info.get("className", [])]
    output = []
    for index in indices.tolist():
        output.append(class_names[index] if 0 <= index < len(class_names) else str(index))
    return output


def detect_offset_blocks(residual_ms: np.ndarray, jump_threshold_ms: float = 20_000.0) -> list[dict[str, Any]]:
    residual = np.asarray(residual_ms, dtype=np.float64).reshape(-1)
    if residual.size == 0:
        return []
    start = 0
    blocks: list[dict[str, Any]] = []
    for index in range(1, len(residual)):
        if abs(float(residual[index] - residual[index - 1])) > jump_threshold_ms:
            blocks.append(_offset_block(start, index - 1, residual[start:index]))
            start = index
    blocks.append(_offset_block(start, len(residual) - 1, residual[start:]))
    return blocks


def _offset_block(start: int, end: int, residual: np.ndarray) -> dict[str, Any]:
    return {
        "start_index": int(start),
        "end_index": int(end),
        "count": int(len(residual)),
        "offset_mean_ms": float(np.mean(residual)),
        "offset_std_ms": float(np.std(residual)),
    }


def drift_slope_ms_per_min(eeg_time_ms: np.ndarray, residual_ms: np.ndarray) -> float | None:
    if len(eeg_time_ms) < 2 or len(residual_ms) < 2:
        return None
    x_min = (np.asarray(eeg_time_ms, dtype=np.float64) - float(eeg_time_ms[0])) / 60_000.0
    x0 = x_min - float(np.mean(x_min))
    denom = float(np.dot(x0, x0))
    if denom <= 0:
        return None
    y = np.asarray(residual_ms, dtype=np.float64)
    return float(np.dot(x0, y - float(np.mean(y))) / denom)


def classify_alignment(
    residual_ms: np.ndarray,
    blocks: Sequence[Mapping[str, Any]],
    *,
    skipped_marker_indices: Mapping[str, Sequence[int]] | None = None,
    stable_block_std_threshold_ms: float = 100.0,
    continuous_drift_slope_threshold_ms_per_min: float = 10.0,
) -> str:
    residual = np.asarray(residual_ms, dtype=np.float64).reshape(-1)
    if residual.size == 0:
        return "no_common_events"
    stable_blocks = all(float(block["offset_std_ms"]) <= stable_block_std_threshold_ms for block in blocks)
    skipped = bool(skipped_marker_indices and any(skipped_marker_indices.values()))
    if len(blocks) == 1 and stable_blocks:
        return "stable_fixed_offset"
    if len(blocks) > 1 and stable_blocks:
        return "skip_aligned_piecewise_constant_offset" if skipped else "piecewise_constant_offset"
    slope = drift_slope_ms_per_min(np.arange(len(residual), dtype=np.float64), residual)
    if slope is not None and abs(slope) >= continuous_drift_slope_threshold_ms_per_min:
        return "continuous_drift"
    return "mixed_or_unstable_offset"


def _select_best_skip(longer: np.ndarray, shorter: np.ndarray) -> tuple[int, np.ndarray]:
    best_skip = 0
    best_residual = shorter - np.delete(longer, 0)
    best_score = float("inf")
    for skip_index in range(len(longer)):
        residual = shorter - np.delete(longer, skip_index)
        score = sum(float(block["offset_std_ms"]) for block in detect_offset_blocks(residual))
        if score < best_score:
            best_score = score
            best_skip = skip_index
            best_residual = residual
    return best_skip, best_residual


def align_paired_marker_streams(
    *,
    dataset_id: str,
    subject: str,
    record_id: str,
    eeg_marker: Mapping[str, Any],
    fnirs_marker: Mapping[str, Any],
    event_type: str = "trial",
    jump_threshold_ms: float = 20_000.0,
) -> tuple[list[CanonicalEvent], EventAlignmentReport]:
    eeg_times = np.asarray(eeg_marker.get("time", []), dtype=np.float64).reshape(-1)
    fnirs_times = np.asarray(fnirs_marker.get("time", []), dtype=np.float64).reshape(-1)
    eeg_labels = marker_label_names(eeg_marker)
    fnirs_labels = marker_label_names(fnirs_marker)
    eeg_indices = marker_label_indices(eeg_marker)
    fnirs_indices = marker_label_indices(fnirs_marker)
    skipped: dict[str, list[int]] = {"eeg_indices": [], "fnirs_indices": []}

    if len(eeg_times) == len(fnirs_times):
        aligned_eeg_times = eeg_times
        aligned_fnirs_times = fnirs_times
        aligned_eeg_labels = eeg_labels
        aligned_fnirs_labels = fnirs_labels
        aligned_label_indices = eeg_indices
    elif len(eeg_times) == len(fnirs_times) + 1:
        skip, _ = _select_best_skip(eeg_times, fnirs_times)
        aligned_eeg_times = np.delete(eeg_times, skip)
        aligned_fnirs_times = fnirs_times
        aligned_eeg_labels = [label for index, label in enumerate(eeg_labels) if index != skip]
        aligned_fnirs_labels = fnirs_labels
        aligned_label_indices = np.delete(eeg_indices, skip)
        skipped["eeg_indices"] = [int(skip)]
    elif len(fnirs_times) == len(eeg_times) + 1:
        skip, _ = _select_best_skip(fnirs_times, eeg_times)
        aligned_eeg_times = eeg_times
        aligned_fnirs_times = np.delete(fnirs_times, skip)
        aligned_eeg_labels = eeg_labels
        aligned_fnirs_labels = [label for index, label in enumerate(fnirs_labels) if index != skip]
        aligned_label_indices = eeg_indices
        skipped["fnirs_indices"] = [int(skip)]
    else:
        common = min(len(eeg_times), len(fnirs_times))
        aligned_eeg_times = eeg_times[:common]
        aligned_fnirs_times = fnirs_times[:common]
        aligned_eeg_labels = eeg_labels[:common]
        aligned_fnirs_labels = fnirs_labels[:common]
        aligned_label_indices = eeg_indices[:common]

    residual = aligned_fnirs_times - aligned_eeg_times
    blocks = detect_offset_blocks(residual, jump_threshold_ms=jump_threshold_ms)
    label_match = aligned_eeg_labels == aligned_fnirs_labels if len(residual) else None
    report = EventAlignmentReport(
        dataset_id=dataset_id,
        subject=subject,
        record_id=record_id,
        num_eeg_events=int(len(eeg_times)),
        num_fnirs_events=int(len(fnirs_times)),
        num_aligned_events=int(len(residual)),
        alignment_case=classify_alignment(residual, blocks, skipped_marker_indices=skipped),
        label_sequence_match=label_match,
        offset_mean_ms=float(np.mean(residual)) if residual.size else None,
        offset_std_ms=float(np.std(residual)) if residual.size else None,
        drift_slope_ms_per_min=drift_slope_ms_per_min(aligned_eeg_times, residual),
        offset_blocks=tuple(blocks),
        skipped_marker_indices=skipped,
        metadata={"residual_series_ms": residual.tolist()},
    )
    events = [
        CanonicalEvent(
            dataset_id=dataset_id,
            subject=subject,
            record_id=record_id,
            event_index=int(index),
            event_type=event_type,
            label=str(aligned_eeg_labels[index] if label_match or index >= len(aligned_fnirs_labels) else aligned_fnirs_labels[index]),
            label_index=int(aligned_label_indices[index]) if index < len(aligned_label_indices) else None,
            eeg_time_ms=float(aligned_eeg_times[index]),
            fnirs_time_ms=float(aligned_fnirs_times[index]),
            onset_ms=float(aligned_fnirs_times[index]),
            alignment_role="paired_eeg_fnirs_marker",
            metadata={
                "eeg_label": aligned_eeg_labels[index] if index < len(aligned_eeg_labels) else None,
                "fnirs_label": aligned_fnirs_labels[index] if index < len(aligned_fnirs_labels) else None,
                "offset_ms": float(residual[index]),
            },
        )
        for index in range(len(residual))
    ]
    return events, report


def read_xlsx_rows(path: str) -> list[dict[str, str]]:
    """Read the first worksheet of a small xlsx file using only stdlib."""
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        shared: list[str] = []
        ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", ns):
                shared.append("".join(text.text or "" for text in item.findall(".//a:t", ns)))
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: list[list[str]] = []
        for row in root.findall(".//a:row", ns):
            values: dict[int, str] = {}
            for cell in row.findall("a:c", ns):
                ref = cell.get("r", "")
                column = _excel_column_index(ref)
                value_node = cell.find("a:v", ns)
                value = "" if value_node is None else str(value_node.text or "")
                if cell.get("t") == "s" and value:
                    value = shared[int(value)]
                values[column] = value
            if values:
                max_col = max(values)
                rows.append([values.get(index, "") for index in range(max_col + 1)])
    if not rows:
        return []
    header = [str(item).strip() for item in rows[0]]
    return [
        {header[index]: row[index] if index < len(row) else "" for index in range(len(header)) if header[index]}
        for row in rows[1:]
    ]


def _excel_column_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        return 0
    value = 0
    for char in match.group(1):
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value - 1
