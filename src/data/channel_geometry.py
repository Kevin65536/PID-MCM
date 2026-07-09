"""Channel geometry normalization for clean multimodal physiology caches."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.io import loadmat

from .clean_physiology_cache import base_record_id, canonical_subject_id
from .event_alignment import read_xlsx_rows


CHANNEL_GEOMETRY_SCHEMA = "physiology_channel_geometry_v1"


@dataclass(frozen=True)
class ChannelGeometryRecord:
    dataset_id: str
    subject: str
    record_id: str
    modality: str
    channel_name: str
    channel_role: str
    coordinate_system: str
    coordinate_units: str
    x: float | None = None
    y: float | None = None
    z: float | None = None
    source_index: int | None = None
    detector_index: int | None = None
    source_name: str | None = None
    detector_name: str | None = None
    source_file: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = CHANNEL_GEOMETRY_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["canonical_subject_id"] = canonical_subject_id(self.dataset_id, self.subject)
        payload["base_record_id"] = base_record_id(self.dataset_id, self.record_id)
        payload["geometry_key"] = "|".join(
            (
                self.dataset_id,
                payload["canonical_subject_id"],
                payload["base_record_id"],
                self.modality,
                self.channel_name,
            )
        )
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


def _as_list(value: Any) -> list[Any]:
    array = np.asarray(value, dtype=object)
    if array.ndim == 0:
        return [array.item()]
    return array.ravel().tolist()


def _labels(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value)]


def _coords_from_mnt(mnt: Any, index: int) -> tuple[float | None, float | None, float | None]:
    pos = getattr(mnt, "pos_3d", None)
    if pos is not None:
        arr = np.asarray(pos, dtype=np.float64)
        if arr.ndim == 2:
            if arr.shape[0] == 3 and index < arr.shape[1]:
                return float(arr[0, index]), float(arr[1, index]), float(arr[2, index])
            if arr.shape[1] == 3 and index < arr.shape[0]:
                return float(arr[index, 0]), float(arr[index, 1]), float(arr[index, 2])
    x = np.asarray(getattr(mnt, "x", []), dtype=np.float64).reshape(-1)
    y = np.asarray(getattr(mnt, "y", []), dtype=np.float64).reshape(-1)
    return (
        float(x[index]) if index < len(x) else None,
        float(y[index]) if index < len(y) else None,
        None,
    )


def _load_first_matstruct(path: Path) -> Any:
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    key = next(name for name in payload if not name.startswith("__"))
    value = payload[key]
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        return value.item()
    return value


def records_from_mnt(
    path: Path,
    *,
    dataset_id: str,
    subject: str,
    record_id: str,
    modality: str,
    channel_role: str,
    coordinate_system: str = "dataset_mnt_pos_3d",
    coordinate_units: str = "normalized_head_unit",
    source_file: str | None = None,
) -> list[ChannelGeometryRecord]:
    mnt = _load_first_matstruct(path)
    labels = _labels(getattr(mnt, "clab", []))
    sd = np.asarray(getattr(mnt, "sd", np.empty((0, 2))), dtype=np.float64)
    rows = []
    for index, label in enumerate(labels):
        x, y, z = _coords_from_mnt(mnt, index)
        source_index = detector_index = None
        if sd.ndim == 2 and index < sd.shape[0] and sd.shape[1] >= 2:
            source_index = int(sd[index, 0])
            detector_index = int(sd[index, 1])
        rows.append(
            ChannelGeometryRecord(
                dataset_id=dataset_id,
                subject=subject,
                record_id=record_id,
                modality=modality,
                channel_name=label,
                channel_role=channel_role,
                coordinate_system=coordinate_system,
                coordinate_units=coordinate_units,
                x=x,
                y=y,
                z=z,
                source_index=source_index,
                detector_index=detector_index,
                source_file=source_file or str(path),
                metadata={"mnt_index": index},
            )
        )
    return rows


def records_from_refed_coordinates(path: Path, *, source_file: str | None = None) -> list[ChannelGeometryRecord]:
    rows: list[ChannelGeometryRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for item in csv.DictReader(handle):
            channel = str(item.get("Channel", "")).strip()
            rows.append(
                ChannelGeometryRecord(
                    dataset_id="refed",
                    subject="all",
                    record_id="global_fnirs_coordinates",
                    modality="fnirs",
                    channel_name=f"CH{int(float(channel))}" if channel else "",
                    channel_role="fnirs_channel_midpoint",
                    coordinate_system="dataset_head_coordinates",
                    coordinate_units="unknown_native",
                    x=_float_or_none(item.get("X")),
                    y=_float_or_none(item.get("Y")),
                    z=_float_or_none(item.get("Z")),
                    source_index=_int_or_none(item.get("Source")),
                    detector_index=_int_or_none(item.get("Detector")),
                    source_file=source_file or str(path),
                    metadata={"raw_channel": channel},
                )
            )
    return rows


def records_from_visual_ced(path: Path, *, source_file: str | None = None) -> list[ChannelGeometryRecord]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    header = [item.strip() for item in lines[0].split("\t") if item.strip()]
    output: list[ChannelGeometryRecord] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        values = [item.strip() for item in line.split("\t")]
        row = {header[index]: values[index] if index < len(values) else "" for index in range(len(header))}
        label = row.get("labels", "")
        if not label:
            continue
        output.append(
            ChannelGeometryRecord(
                dataset_id="visual_cognitive_motivation",
                subject="all",
                record_id="global_eeg_location_ced",
                modality="eeg",
                channel_name=label,
                channel_role="eeg_electrode",
                coordinate_system="EEGLAB_CED_cartesian",
                coordinate_units="normalized_head_unit",
                x=_float_or_none(row.get("X")),
                y=_float_or_none(row.get("Y")),
                z=_float_or_none(row.get("Z")),
                source_file=source_file or str(path),
                metadata={key: value for key, value in row.items() if key not in {"labels", "X", "Y", "Z"}},
            )
        )
    return output


def records_from_visual_fnirs_reference(path: Path, *, source_file: str | None = None) -> list[ChannelGeometryRecord]:
    raw_rows = read_xlsx_rows(str(path))
    output: list[ChannelGeometryRecord] = []
    for index, row in enumerate(raw_rows):
        pairs = []
        values = list(row.values())
        if len(values) >= 2:
            pairs.append(("Probe1", values[0], values[1]))
        if len(values) >= 4:
            pairs.append(("Probe2", values[2], values[3]))
        for probe, fnirs_channel, eeg_label in pairs:
            fnirs_channel = str(fnirs_channel).strip()
            if not re.fullmatch(r"CH\d+", fnirs_channel):
                continue
            output.append(
                ChannelGeometryRecord(
                    dataset_id="visual_cognitive_motivation",
                    subject="all",
                    record_id=probe,
                    modality="fnirs",
                    channel_name=fnirs_channel,
                    channel_role="fnirs_channel_to_eeg_reference",
                    coordinate_system="referenced_to_visual_eeg_ced",
                    coordinate_units="label_reference",
                    source_file=source_file or str(path),
                    metadata={"nearest_eeg_label": str(eeg_label).strip(), "row_index": index},
                )
            )
    return output


def _float_or_none(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    parsed = _float_or_none(value)
    return int(parsed) if parsed is not None else None
