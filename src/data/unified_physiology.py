"""Unified, provenance-preserving access to the four raw physiology datasets.

The Croce source/observation caches are deliberately absent from this module:
they are derived supervision targets, not an additional dataset.  This loader
joins the clean fNIRS cache to the original EEG recordings, applies one
canonical preprocessing contract, and returns event-aligned multimodal
windows with a common schema.

Canonical numerical coordinates are dimensionless robust standard deviations.
Native units and measurement families remain in metadata; the transform does
not claim that volts and chromophore concentration are physically identical.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from fractions import Fraction
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt

from .clean_physiology_cache import CleanCacheRecord, CleanPhysiologyCacheIndex


RAW_DATASET_IDS: tuple[str, ...] = (
    "eeg_fnirs_single_trial",
    "refed",
    "visual_cognitive_motivation",
    "simultaneous_eeg_nirs",
)

UNIFIED_PHYSIOLOGY_SCHEMA = "unified_physiology_window_v1"
CANONICAL_UNIT = "robust_standard_deviation"
CANONICAL_EEG_SAMPLE_RATE_HZ = 200.0
CANONICAL_FNIRS_SAMPLE_RATE_HZ = 10.0
CANONICAL_EEG_BAND_HZ = (1.0, 45.0)
CANONICAL_FNIRS_BAND_HZ = (0.01, 0.2)
CANONICAL_FNIRS_COMPONENTS = ("HbO", "HbR")
DEFAULT_UNIFIED_WINDOW_DURATION_S = 20.0
DEFAULT_ADMISSIBLE_ALIGNMENT_CASES = frozenset({
    "stable_fixed_offset",
    "piecewise_constant_offset",
    "skip_aligned_piecewise_constant_offset",
    "shared_segment_index_no_marker_stream",
})

VISUAL_CONDITION_INDICES = {"RR": 0, "RF": 1, "FF": 2, "FR": 3, "unknown": -1}


@dataclass(frozen=True)
class CanonicalPreprocessingContract:
    canonical_unit: str = CANONICAL_UNIT
    eeg_sample_rate_hz: float = CANONICAL_EEG_SAMPLE_RATE_HZ
    fnirs_sample_rate_hz: float = CANONICAL_FNIRS_SAMPLE_RATE_HZ
    eeg_band_hz: tuple[float, float] = CANONICAL_EEG_BAND_HZ
    fnirs_band_hz: tuple[float, float] = CANONICAL_FNIRS_BAND_HZ
    eeg_steps: tuple[str, ...] = (
        "finite_interpolation",
        "bandpass_1_45_hz",
        "resample_200_hz",
        "full_record_channel_median_mad",
    )
    fnirs_steps: tuple[str, ...] = (
        "dataset_available_homer2_aligned_hbo_hbr",
        "resample_10_hz",
        "full_record_channel_median_mad",
    )
    schema: str = UNIFIED_PHYSIOLOGY_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("eeg_band_hz", "fnirs_band_hz", "eeg_steps", "fnirs_steps"):
            payload[key] = list(payload[key])
        return payload


CANONICAL_PREPROCESSING = CanonicalPreprocessingContract()


@dataclass(frozen=True)
class NativeEEGRecord:
    values: np.ndarray
    sample_rate_hz: float
    channel_names: tuple[str, ...]
    native_unit: str
    source_path: Path


@dataclass(frozen=True)
class UnifiedWindowRef:
    record: CleanCacheRecord
    event: Mapping[str, Any]


def _first_mat_value(path: Path, key: str | None = None) -> Any:
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    if key is None:
        key = next(name for name in payload if not name.startswith("__"))
    value = payload[key]
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        value = value.item()
    return value


def _labels(value: Any) -> tuple[str, ...]:
    return tuple(str(item).strip() for item in np.asarray(value, dtype=object).reshape(-1).tolist())


def canonical_channel_name(value: str) -> str:
    """Return a stable channel label without destroying 10-20 case semantics."""
    return re.sub(r"\s+", "", str(value).strip())


def canonical_fnirs_channel_names(names: Sequence[str]) -> tuple[str, ...]:
    output = []
    for name in names:
        value = canonical_channel_name(name)
        value = re.sub(r"_(Oxy|Hbo)$", "_HbO", value, flags=re.IGNORECASE)
        value = re.sub(r"_(Deoxy|Hbr|Hb)$", "_HbR", value, flags=re.IGNORECASE)
        output.append(value)
    return tuple(output)


def fnirs_component_roles(names: Sequence[str]) -> tuple[str, ...]:
    roles = []
    for name in canonical_fnirs_channel_names(names):
        if name.endswith("_HbO"):
            roles.append("HbO")
        elif name.endswith("_HbR"):
            roles.append("HbR")
        else:
            roles.append("unknown")
    return tuple(roles)


def _as_time_channels(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"expected [time, channels], got {array.shape}")
    return array


def _interpolate_nonfinite(values: np.ndarray) -> tuple[np.ndarray, int]:
    output = _as_time_channels(values).copy()
    repaired = 0
    time = np.arange(output.shape[0], dtype=np.float64)
    for channel in range(output.shape[1]):
        finite = np.isfinite(output[:, channel])
        repaired += int(np.count_nonzero(~finite))
        if not np.any(finite):
            output[:, channel] = 0.0
        elif not np.all(finite):
            output[:, channel] = np.interp(time, time[finite], output[finite, channel])
    return output, repaired


def _bandpass(values: np.ndarray, sample_rate_hz: float, band_hz: tuple[float, float]) -> np.ndarray:
    low, high = band_hz
    nyquist = float(sample_rate_hz) * 0.5
    high = min(float(high), nyquist * 0.95)
    if low <= 0 or high <= low or values.shape[0] < 32:
        return values
    sos = butter(4, [float(low) / nyquist, high / nyquist], btype="bandpass", output="sos")
    try:
        return sosfiltfilt(sos, values, axis=0)
    except ValueError:
        return values


def _resample(values: np.ndarray, source_hz: float, target_hz: float) -> np.ndarray:
    if abs(float(source_hz) - float(target_hz)) < 1e-9:
        return values
    ratio = Fraction(float(target_hz) / float(source_hz)).limit_denominator(10_000)
    return resample_poly(values, ratio.numerator, ratio.denominator, axis=0)


def _robust_standardize(values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    array, repaired = _interpolate_nonfinite(values)
    median = np.median(array, axis=0)
    mad = 1.482602218505602 * np.median(np.abs(array - median[None, :]), axis=0)
    q25, q75 = np.quantile(array, [0.25, 0.75], axis=0)
    iqr = (q75 - q25) / 1.3489795003921634
    std = np.std(array, axis=0)
    scale = np.where(np.isfinite(mad) & (mad > 1e-8), mad, iqr)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, std)
    flat = ~(np.isfinite(scale) & (scale > 1e-8))
    scale = np.where(flat, 1.0, scale)
    canonical = (array - median[None, :]) / scale[None, :]
    return canonical.astype(np.float32), {
        "repaired_nonfinite_samples": int(repaired),
        "flat_channel_count": int(np.count_nonzero(flat)),
        "channel_location": median.astype(float).tolist(),
        "channel_scale": scale.astype(float).tolist(),
        "canonical_unit": CANONICAL_UNIT,
    }


def preprocess_eeg_record(record: NativeEEGRecord) -> tuple[np.ndarray, dict[str, Any]]:
    finite, repaired = _interpolate_nonfinite(record.values)
    filtered = _bandpass(finite, record.sample_rate_hz, CANONICAL_EEG_BAND_HZ)
    resampled = _resample(filtered, record.sample_rate_hz, CANONICAL_EEG_SAMPLE_RATE_HZ)
    canonical, state = _robust_standardize(resampled)
    state.update({
        "native_unit": record.native_unit,
        "native_sample_rate_hz": float(record.sample_rate_hz),
        "canonical_sample_rate_hz": CANONICAL_EEG_SAMPLE_RATE_HZ,
        "filter_band_hz": list(CANONICAL_EEG_BAND_HZ),
        "filter_input_repaired_nonfinite_samples": int(repaired),
        "source_path": str(record.source_path),
    })
    return canonical, state


def preprocess_fnirs_record(
    values: np.ndarray,
    *,
    sample_rate_hz: float,
    native_contract: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    # The clean-cache branch has already applied the best available common
    # 0.01-0.2 Hz/motion/HbO-HbR path.  Re-filtering would double-filter it.
    resampled = _resample(_as_time_channels(values), sample_rate_hz, CANONICAL_FNIRS_SAMPLE_RATE_HZ)
    canonical, state = _robust_standardize(resampled)
    state.update({
        "native_contract": dict(native_contract),
        "native_sample_rate_hz": float(sample_rate_hz),
        "canonical_sample_rate_hz": CANONICAL_FNIRS_SAMPLE_RATE_HZ,
        "filter_band_hz": list(CANONICAL_FNIRS_BAND_HZ),
        "input_branch": "homer2_aligned_fnirs",
    })
    return canonical, state


def _single_trial_eeg(project_root: Path, record: CleanCacheRecord) -> NativeEEGRecord:
    subject = record.canonical_subject_id.replace("subject_", "subject ")
    directory = project_root / "data/EEG+NIRS Single-Trial/EEG_01-29" / subject / "with occular artifact"
    if not directory.exists():
        directory = directory.parent
    path = directory / "cnt.mat"
    sessions = np.atleast_1d(_first_mat_value(path, "cnt"))
    session_index = int(record.base_record_id.rsplit("_", 1)[-1])
    session = sessions[session_index]
    values = _as_time_channels(np.asarray(session.x, dtype=np.float64))
    names = _labels(session.clab)
    keep = np.asarray(["EOG" not in name.upper() for name in names], dtype=bool)
    return NativeEEGRecord(
        values=values[:, keep],
        sample_rate_hz=float(session.fs),
        channel_names=tuple(canonical_channel_name(name) for name, selected in zip(names, keep) if selected),
        native_unit=str(getattr(session, "yUnit", "uV") or "uV"),
        source_path=path,
    )


def _simultaneous_eeg(project_root: Path, record: CleanCacheRecord) -> NativeEEGRecord:
    path = project_root / "data/Simultaneous EEG&NIRS" / f"{record.canonical_subject_id}-EEG" / f"{record.base_record_id}.mat"
    payload = _first_mat_value(path)
    return NativeEEGRecord(
        values=_as_time_channels(np.asarray(payload.x, dtype=np.float64)),
        sample_rate_hz=float(payload.fs),
        channel_names=tuple(canonical_channel_name(name) for name in _labels(payload.clab)),
        native_unit=str(getattr(payload, "yUnit", "uV") or "uV"),
        source_path=path,
    )


def _refed_eeg_names(project_root: Path, count: int) -> tuple[str, ...]:
    path = project_root / "data/REFED-dataset/EEG_channels.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        names = [canonical_channel_name(row.get("ch_name", "")) for row in csv.DictReader(handle)]
    return tuple(names[:count]) if len(names) >= count else tuple(f"EEG{index + 1}" for index in range(count))


def _refed_eeg(project_root: Path, record: CleanCacheRecord) -> NativeEEGRecord:
    path = project_root / "data/REFED-dataset/data" / record.canonical_subject_id / "EEG_videos.mat"
    values = np.asarray(_first_mat_value(path, record.base_record_id), dtype=np.float64)
    if values.shape[0] <= values.shape[1]:
        values = values.T
    return NativeEEGRecord(
        values=_as_time_channels(values),
        sample_rate_hz=1000.0,
        channel_names=_refed_eeg_names(project_root, values.shape[1]),
        native_unit="V",
        source_path=path,
    )


def _read_edf(path: Path) -> tuple[np.ndarray, float, list[str], list[str]]:
    with path.open("rb") as handle:
        fixed = handle.read(256)
        header_bytes = int(fixed[184:192].decode().strip())
        records = int(fixed[236:244].decode().strip())
        duration = float(fixed[244:252].decode().strip())
        signal_count = int(fixed[252:256].decode().strip())
        header = handle.read(header_bytes - 256)
    cursor = 0
    fields: list[list[str]] = []
    for width in (16, 80, 8, 8, 8, 8, 8, 80, 8, 32):
        fields.append([
            header[cursor + index * width : cursor + (index + 1) * width].decode(errors="replace").strip()
            for index in range(signal_count)
        ])
        cursor += width * signal_count
    labels, units = fields[0], fields[2]
    physical_min = np.asarray([float(value) for value in fields[3]])
    physical_max = np.asarray([float(value) for value in fields[4]])
    digital_min = np.asarray([float(value) for value in fields[5]])
    digital_max = np.asarray([float(value) for value in fields[6]])
    samples_per_record = np.asarray([int(value) for value in fields[8]], dtype=int)
    candidates = [
        index for index, label in enumerate(labels)
        if "ANNOTATION" not in label.upper() and label.upper() not in {"A64", "DC9", "DC09"}
    ]
    sample_rate = float(max(samples_per_record[index] / duration for index in candidates))
    offsets = np.concatenate(([0], np.cumsum(samples_per_record)))
    raw = np.memmap(path, dtype="<i2", mode="r", offset=header_bytes, shape=(records, int(samples_per_record.sum())))
    channels = []
    for index in candidates:
        digital = np.asarray(raw[:, offsets[index] : offsets[index + 1]], dtype=np.float64).reshape(-1)
        denominator = max(digital_max[index] - digital_min[index], 1.0)
        physical = (digital - digital_min[index]) * (physical_max[index] - physical_min[index]) / denominator + physical_min[index]
        channel_rate = samples_per_record[index] / duration
        physical = _resample(physical[:, None], channel_rate, sample_rate)[:, 0]
        channels.append(physical)
    length = min(map(len, channels))
    return (
        np.stack([channel[:length] for channel in channels], axis=1),
        sample_rate,
        [labels[index] for index in candidates],
        [units[index] or "unknown" for index in candidates],
    )


def _visual_eeg_name_map(project_root: Path) -> dict[str, str]:
    path = project_root / "data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults/Location.ced"
    rows = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    header = [value.strip() for value in rows[0].split("\t")]
    mapping: dict[str, str] = {}
    for line in rows[1:]:
        values = line.split("\t")
        row = {key: values[index].strip() if index < len(values) else "" for index, key in enumerate(header)}
        if row.get("Number") and row.get("labels"):
            mapping[f"A{int(float(row['Number']))}"] = canonical_channel_name(row["labels"])
    return mapping


def _visual_eeg(project_root: Path, record: CleanCacheRecord) -> NativeEEGRecord:
    root = project_root / "data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults"
    raw_dir = root / record.canonical_subject_id / "EEG/raw"
    part = re.search(r"Part(\d+)", record.base_record_id, flags=re.IGNORECASE)
    if part:
        candidates = sorted(raw_dir.glob(f"*part{part.group(1)}.edf"))
    else:
        candidates = sorted(raw_dir.glob(f"{record.canonical_subject_id}.edf"))
    if not candidates:
        candidates = sorted(raw_dir.glob("*.edf"))
    if not candidates:
        raise FileNotFoundError(f"no EDF file for {record.join_key}")
    values, sample_rate, names, units = _read_edf(candidates[0])
    name_map = _visual_eeg_name_map(project_root)
    canonical_names = tuple(name_map.get(name, canonical_channel_name(name)) for name in names)
    unique_units = sorted(set(units))
    return NativeEEGRecord(
        values=values,
        sample_rate_hz=sample_rate,
        channel_names=canonical_names,
        native_unit=unique_units[0] if len(unique_units) == 1 else "/".join(unique_units),
        source_path=candidates[0],
    )


def load_native_eeg_record(project_root: Path, record: CleanCacheRecord) -> NativeEEGRecord:
    if record.dataset_id == "eeg_fnirs_single_trial":
        return _single_trial_eeg(project_root, record)
    if record.dataset_id == "simultaneous_eeg_nirs":
        return _simultaneous_eeg(project_root, record)
    if record.dataset_id == "refed":
        return _refed_eeg(project_root, record)
    if record.dataset_id == "visual_cognitive_motivation":
        return _visual_eeg(project_root, record)
    raise KeyError(f"not one of the four raw datasets: {record.dataset_id}")


def canonical_label(event: Mapping[str, Any], dataset_id: str) -> dict[str, Any]:
    metadata = dict(event.get("metadata", {}))
    task = str(metadata.get("task") or event.get("event_type") or "unknown")
    event_role = str(metadata.get("event_role") or event.get("event_type") or "unknown")
    condition = str(metadata.get("condition_label") or event.get("label") or "unknown")
    if dataset_id == "visual_cognitive_motivation":
        condition = str(metadata.get("epoch_type") or condition or "unknown")
        class_index = VISUAL_CONDITION_INDICES.get(condition, -1)
    else:
        raw_index = event.get("label_index")
        class_index = int(raw_index) if raw_index is not None else -1
    return {
        "schema": "canonical_task_label_v1",
        "namespace": f"{dataset_id}:{task}",
        "dataset_id": dataset_id,
        "task": task,
        "condition": condition,
        "class_index": class_index,
        "event_role": event_role,
    }


def _slice_window(values: np.ndarray, onset_ms: float, duration_s: float, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    length = max(1, int(round(float(duration_s) * float(sample_rate_hz))))
    start = int(round(float(onset_ms) / 1000.0 * float(sample_rate_hz)))
    stop = start + length
    output = np.zeros((length, values.shape[1]), dtype=np.float32)
    mask = np.zeros(length, dtype=bool)
    src_start = max(start, 0)
    src_stop = min(stop, values.shape[0])
    if src_stop > src_start:
        dst_start = src_start - start
        dst_stop = dst_start + (src_stop - src_start)
        output[dst_start:dst_stop] = values[src_start:src_stop]
        mask[dst_start:dst_stop] = True
    return output.T, mask


class ChannelGeometryIndex:
    """Read the common channel-geometry sidecar and emit one row per channel."""

    def __init__(self, cache_root: Path):
        path = cache_root / "channel_geometry/channels.jsonl"
        self.rows = []
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                self.rows = [json.loads(line) for line in handle if line.strip()]

    def for_channels(
        self,
        *,
        record: CleanCacheRecord,
        modality: str,
        channel_names: Sequence[str],
    ) -> list[dict[str, Any]]:
        candidates = [row for row in self.rows if row.get("dataset_id") == record.dataset_id and row.get("modality") == modality]
        subject_rows = [
            row for row in candidates
            if row.get("canonical_subject_id") in {record.canonical_subject_id, "all"}
        ]
        if subject_rows:
            candidates = subject_rows
        exact_record = [row for row in candidates if row.get("base_record_id") == record.base_record_id]
        if exact_record:
            candidates = exact_record
        lookup = {canonical_channel_name(str(row.get("channel_name", ""))): row for row in candidates}

        # Visual fNIRS positions are label references.  Resolve them onto the
        # common visual EEG CED coordinates when possible.
        visual_eeg = {
            canonical_channel_name(str(row.get("channel_name", ""))): row
            for row in self.rows
            if row.get("dataset_id") == "visual_cognitive_motivation" and row.get("modality") == "eeg"
        }
        output = []
        for channel_name in channel_names:
            canonical = canonical_channel_name(channel_name)
            base = re.sub(r"_(HbO|HbR)$", "", canonical)
            row = dict(lookup.get(canonical) or lookup.get(base) or {})
            if record.dataset_id == "visual_cognitive_motivation" and modality == "fnirs" and row:
                reference = canonical_channel_name(str(row.get("metadata", {}).get("nearest_eeg_label", "")))
                reference_row = visual_eeg.get(reference, {})
                for axis in ("x", "y", "z"):
                    if row.get(axis) is None and reference_row.get(axis) is not None:
                        row[axis] = reference_row[axis]
                row["coordinate_system"] = "referenced_visual_eeg_head_coordinates"
            output.append({
                "schema": "canonical_channel_geometry_v1",
                "channel_name": canonical,
                "base_channel_name": base,
                "component": canonical.rsplit("_", 1)[-1] if canonical.endswith(("_HbO", "_HbR")) else modality,
                "modality": modality,
                "x": row.get("x"),
                "y": row.get("y"),
                "z": row.get("z"),
                "coordinate_system": row.get("coordinate_system", "unavailable"),
                "coordinate_units": row.get("coordinate_units", "unavailable"),
                "source_index": row.get("source_index"),
                "detector_index": row.get("detector_index"),
                "position_available": any(row.get(axis) is not None for axis in ("x", "y", "z")),
            })
        return output


class UnifiedPhysiologyWindowDataset:
    """Event-aligned EEG/fNIRS windows for the four original datasets.

    The 20-second default is an observation-context contract, not a claim that
    every event lasts 20 seconds.  Event labels remain anchored at the event
    timestamp and ``valid_mask`` identifies record-boundary padding.  Models
    may subdivide the returned context into shorter patches without changing
    the loader contract.
    """

    def __init__(
        self,
        cache_root: str | Path = "data/cache/physiology_semantic_clean_v1",
        *,
        dataset_ids: Sequence[str] = RAW_DATASET_IDS,
        window_duration_s: float = DEFAULT_UNIFIED_WINDOW_DURATION_S,
        window_offset_s: float = 0.0,
        require_paired_timestamps: bool = True,
        include_event_types: set[str] | None = None,
        admissible_alignment_cases: set[str] | frozenset[str] | None = DEFAULT_ADMISSIBLE_ALIGNMENT_CASES,
    ) -> None:
        requested = tuple(str(value) for value in dataset_ids)
        invalid = sorted(set(requested) - set(RAW_DATASET_IDS))
        if invalid:
            raise ValueError(f"only the four original datasets are supported; invalid={invalid}")
        self.cache_root = Path(cache_root)
        self.project_root = Path(__file__).resolve().parents[2]
        self.index = CleanPhysiologyCacheIndex(self.cache_root)
        self.geometry_index = ChannelGeometryIndex(self.cache_root)
        self.dataset_ids = requested
        self.window_duration_s = float(window_duration_s)
        self.window_offset_s = float(window_offset_s)
        self.require_paired_timestamps = bool(require_paired_timestamps)
        self.include_event_types = include_event_types
        self.admissible_alignment_cases = None if admissible_alignment_cases is None else frozenset(admissible_alignment_cases)
        self.excluded_alignment_records: dict[str, str] = {}
        self.windows = self._build_windows()
        self._record_cache: dict[str, dict[str, Any]] = {}

    def _selected_records(self) -> Iterable[CleanCacheRecord]:
        for record in self.index.records:
            if record.dataset_id not in self.dataset_ids:
                continue
            if record.dataset_id == "refed" and record.signal_branch != "hbo_hbr":
                continue
            yield record

    def _build_windows(self) -> list[UnifiedWindowRef]:
        windows = []
        for record in self._selected_records():
            reports = self.index.reports_by_join_key.get(record.join_key, [])
            if self.admissible_alignment_cases is not None:
                cases = {str(report.get("alignment_case", "")) for report in reports}
                label_match = all(report.get("label_sequence_match") is not False for report in reports)
                if not reports or not cases.intersection(self.admissible_alignment_cases) or not label_match:
                    self.excluded_alignment_records[record.join_key] = ",".join(sorted(cases)) or "missing_alignment_report"
                    continue
            for event in self.index.events_by_join_key.get(record.join_key, []):
                if self.include_event_types is not None and str(event.get("event_type")) not in self.include_event_types:
                    continue
                eeg_time = event.get("eeg_time_ms")
                fnirs_time = event.get("fnirs_time_ms", event.get("onset_ms"))
                if eeg_time is None and not self.require_paired_timestamps:
                    eeg_time = event.get("onset_ms")
                if fnirs_time is None or (self.require_paired_timestamps and eeg_time is None):
                    continue
                windows.append(UnifiedWindowRef(record=record, event=event))
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def _load_canonical_record(self, record: CleanCacheRecord) -> dict[str, Any]:
        cached = self._record_cache.get(record.join_key)
        if cached is not None:
            return cached
        arrays = self.index.load_record_arrays(record)
        if "homer2_aligned_fnirs" not in arrays:
            raise KeyError(f"missing homer2_aligned_fnirs: {record.npz_path}")
        fnirs_names = canonical_fnirs_channel_names(
            [str(value) for value in arrays.get("homer2_channel_names", record.manifest.get("homer2_channel_names", []))]
        )
        roles = fnirs_component_roles(fnirs_names)
        if set(roles) != set(CANONICAL_FNIRS_COMPONENTS):
            raise ValueError(f"fNIRS component contract failed for {record.join_key}: {sorted(set(roles))}")
        fnirs, fnirs_state = preprocess_fnirs_record(
            arrays["homer2_aligned_fnirs"],
            sample_rate_hz=record.sample_rate_hz,
            native_contract=record.manifest.get("native_contract", {}),
        )
        eeg_native = load_native_eeg_record(self.project_root, record)
        eeg, eeg_state = preprocess_eeg_record(eeg_native)
        payload = {
            "eeg": eeg,
            "fnirs": fnirs,
            "eeg_channel_names": eeg_native.channel_names,
            "fnirs_channel_names": fnirs_names,
            "fnirs_component_roles": roles,
            "eeg_preprocessing_state": eeg_state,
            "fnirs_preprocessing_state": fnirs_state,
        }
        # Keep memory bounded while allowing repeated events from one record.
        if len(self._record_cache) >= 2:
            self._record_cache.pop(next(iter(self._record_cache)))
        self._record_cache[record.join_key] = payload
        return payload

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.windows[index]
        record_data = self._load_canonical_record(ref.record)
        offset_ms = self.window_offset_s * 1000.0
        eeg_time_ms = float(ref.event.get("eeg_time_ms", ref.event.get("onset_ms"))) + offset_ms
        fnirs_time_ms = float(ref.event.get("fnirs_time_ms", ref.event.get("onset_ms"))) + offset_ms
        eeg, eeg_mask = _slice_window(
            record_data["eeg"], eeg_time_ms, self.window_duration_s, CANONICAL_EEG_SAMPLE_RATE_HZ
        )
        fnirs, fnirs_mask = _slice_window(
            record_data["fnirs"], fnirs_time_ms, self.window_duration_s, CANONICAL_FNIRS_SAMPLE_RATE_HZ
        )
        label = canonical_label(ref.event, ref.record.dataset_id)
        return {
            "schema": UNIFIED_PHYSIOLOGY_SCHEMA,
            "eeg": eeg,
            "fnirs": fnirs,
            "valid_mask": {"eeg": eeg_mask, "fnirs": fnirs_mask},
            "modality_available": {"eeg": True, "fnirs": True},
            "sample_rate_hz": {
                "eeg": CANONICAL_EEG_SAMPLE_RATE_HZ,
                "fnirs": CANONICAL_FNIRS_SAMPLE_RATE_HZ,
            },
            "unit": {"eeg": CANONICAL_UNIT, "fnirs": CANONICAL_UNIT},
            "channel_names": {
                "eeg": list(record_data["eeg_channel_names"]),
                "fnirs": list(record_data["fnirs_channel_names"]),
            },
            "component_roles": {
                "eeg": ["electrical_potential"] * eeg.shape[0],
                "fnirs": list(record_data["fnirs_component_roles"]),
            },
            "channel_geometry": {
                "eeg": self.geometry_index.for_channels(
                    record=ref.record, modality="eeg", channel_names=record_data["eeg_channel_names"]
                ),
                "fnirs": self.geometry_index.for_channels(
                    record=ref.record, modality="fnirs", channel_names=record_data["fnirs_channel_names"]
                ),
            },
            "label": label,
            "dataset_id": ref.record.dataset_id,
            "subject": ref.record.canonical_subject_id,
            "record_id": ref.record.base_record_id,
            "signal_branch": ref.record.signal_branch,
            "join_key": ref.record.join_key,
            "event": dict(ref.event),
            "alignment": {
                "eeg_time_ms": eeg_time_ms,
                "fnirs_time_ms": fnirs_time_ms,
                "offset_ms": fnirs_time_ms - eeg_time_ms,
                "separate_modality_clocks_used": True,
            },
            "preprocessing_contract": CANONICAL_PREPROCESSING.to_dict(),
            "preprocessing_state": {
                "eeg": record_data["eeg_preprocessing_state"],
                "fnirs": record_data["fnirs_preprocessing_state"],
            },
        }

    def contract_summary(self) -> dict[str, Any]:
        counts = {dataset_id: 0 for dataset_id in self.dataset_ids}
        for window in self.windows:
            counts[window.record.dataset_id] += 1
        return {
            "schema": UNIFIED_PHYSIOLOGY_SCHEMA,
            "dataset_ids": list(self.dataset_ids),
            "derived_targets_excluded": ["croce_local_cache"],
            "window_count_by_dataset": counts,
            "admissible_alignment_cases": sorted(self.admissible_alignment_cases or []),
            "excluded_alignment_record_count": len(self.excluded_alignment_records),
            "excluded_alignment_records": dict(self.excluded_alignment_records),
            "preprocessing": CANONICAL_PREPROCESSING.to_dict(),
            "fnirs_components": list(CANONICAL_FNIRS_COMPONENTS),
            "label_schema": "canonical_task_label_v1",
            "geometry_schema": "canonical_channel_geometry_v1",
        }
