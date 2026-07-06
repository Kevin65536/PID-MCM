"""Cross-dataset fNIRS measurement audit used by E0-v2.

Only train and validation subjects are opened.  The returned adapter metadata
records the original measurement semantics and never relabels optical voltage
or absorbance as HbO/HbR concentration.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import yaml
from scipy.io import loadmat

from src.data.physiology_measurement_adapter import PhysiologyMeasurementAdapter, robust_location_scale


@dataclass
class RawRecord:
    dataset: str
    subject: str
    values: np.ndarray
    sample_rate_hz: float
    channel_names: tuple[str, str]
    semantics: str
    unit: str
    transform: str
    source_path: str


def _bounded(values: np.ndarray, max_samples: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] <= max_samples:
        return values
    indices = np.linspace(0, values.shape[0] - 1, max_samples).round().astype(int)
    return values[indices]


def _mat_struct(path: Path, key: str | None = None) -> Any:
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    if key is None:
        key = next(name for name in payload if not name.startswith("__"))
    value = payload[key]
    if isinstance(value, np.ndarray):
        value = value.flat[0]
    return value


def read_single_trial(root: Path, subject: str, max_samples: int) -> RawRecord:
    path = root / f"subject {int(subject):02d}" / "cnt.mat"
    cnt = _mat_struct(path, "cnt")
    values = np.asarray(cnt.x, dtype=np.float64)
    labels = np.asarray(cnt.clab).astype(str).ravel()
    low = np.asarray(["lowWL" in label for label in labels])
    high = np.asarray(["highWL" in label for label in labels])
    pair = np.column_stack((np.nanmedian(values[:, low], axis=1), np.nanmedian(values[:, high], axis=1)))
    return RawRecord(
        dataset="single_trial_voltage",
        subject=subject,
        values=_bounded(pair, max_samples),
        sample_rate_hz=float(cnt.fs),
        channel_names=("760_nm_voltage", "850_nm_voltage"),
        semantics="paired raw optical detector voltage",
        unit=str(getattr(cnt, "yUnit", "V")),
        transform="relative_change",
        source_path=str(path),
    )


def read_simultaneous(root: Path, subject: str, max_samples: int) -> RawRecord:
    path = root / f"VP{int(subject):03d}-NIRS" / "cnt_nback.mat"
    cnt = _mat_struct(path, "cnt_nback")
    oxy = cnt.oxy
    deoxy = cnt.deoxy
    pair = np.column_stack(
        (
            np.nanmedian(np.asarray(oxy.x, dtype=np.float64), axis=1),
            np.nanmedian(np.asarray(deoxy.x, dtype=np.float64), axis=1),
        )
    )
    return RawRecord(
        dataset="simultaneous_concentration",
        subject=subject,
        values=_bounded(pair, max_samples),
        sample_rate_hz=float(oxy.fs),
        channel_names=("oxy", "deoxy"),
        semantics="reported relative oxy/deoxy concentration",
        unit=str(getattr(oxy, "yUnit", "mmol/L")),
        transform="center",
        source_path=str(path),
    )


def read_refed(root: Path, subject: str, max_samples: int) -> RawRecord:
    path = root / str(int(subject)) / "fNIRS_baselines.mat"
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = sorted(name for name in payload if name.startswith("video_"))
    if not keys:
        raise ValueError(f"no fNIRS video arrays in {path}")
    chunks = []
    for key in keys[:3]:
        array = np.asarray(payload[key], dtype=np.float64)
        if array.ndim != 3 or array.shape[0] < 2:
            continue
        chunks.append(np.column_stack((np.nanmedian(array[0], axis=0), np.nanmedian(array[1], axis=0))))
    pair = np.concatenate(chunks, axis=0)
    return RawRecord(
        dataset="refed_reported_chromophore",
        subject=subject,
        values=_bounded(pair, max_samples),
        sample_rate_hz=47.62,
        channel_names=("HbO", "HbR"),
        semantics="reported relative HbO/HbR channels",
        unit="dataset-relative concentration unit (not globally declared)",
        transform="center",
        source_path=str(path),
    )


def _read_visual_csv(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        rows = list(csv.reader(handle))
    data_index = next(index for index, row in enumerate(rows) if row and row[0].strip() == "Data")
    numeric = []
    for row in rows[data_index + 2 :]:
        if len(row) < 25:
            continue
        try:
            numeric.append([float(value) for value in row[1:25]])
        except ValueError:
            continue
    if not numeric:
        raise ValueError(f"no numeric fNIRS rows in {path}")
    return np.asarray(numeric, dtype=np.float64)


def read_visual(root: Path, subject: str, max_samples: int) -> RawRecord:
    directory = root / f"S{int(subject):02d}" / "fNIRS"
    oxy_paths = sorted(directory.glob("*_Probe1_Oxy.csv"))
    if not oxy_paths:
        oxy_paths = sorted(directory.glob("*_Oxy.csv"))
    if not oxy_paths:
        raise FileNotFoundError(f"no Oxy CSV under {directory}")
    oxy_path = oxy_paths[0]
    deoxy_path = Path(str(oxy_path).replace("_Oxy.csv", "_Deoxy.csv"))
    oxy = _read_visual_csv(oxy_path)
    deoxy = _read_visual_csv(deoxy_path)
    length = min(len(oxy), len(deoxy))
    pair = np.column_stack((np.nanmedian(oxy[:length], axis=1), np.nanmedian(deoxy[:length], axis=1)))
    return RawRecord(
        dataset="visual_exported_oxy_deoxy",
        subject=subject,
        values=_bounded(pair, max_samples),
        sample_rate_hz=10.0,
        channel_names=("Oxy", "Deoxy"),
        semantics="instrument-exported relative Oxy/Deoxy traces",
        unit="export unit not declared",
        transform="center",
        source_path=str(oxy_path),
    )


READERS: dict[str, Callable[[Path, str, int], RawRecord]] = {
    "single_trial_voltage": read_single_trial,
    "simultaneous_concentration": read_simultaneous,
    "refed_reported_chromophore": read_refed,
    "visual_exported_oxy_deoxy": read_visual,
}


def _summary(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    location, scale = robust_location_scale(finite)
    q01, q25, q75, q99 = np.quantile(finite, [0.01, 0.25, 0.75, 0.99])
    return {
        "median": location,
        "robust_scale": scale,
        "q01": float(q01),
        "q25": float(q25),
        "q75": float(q75),
        "q99": float(q99),
        "finite_fraction": float(np.mean(np.isfinite(values))),
    }


def _split_subjects(subjects: Iterable[str], train_fraction: float, val_fraction: float) -> dict[str, list[str]]:
    ordered = sorted((str(item) for item in subjects), key=lambda item: int(item))
    n = len(ordered)
    train_end = max(1, int(np.floor(n * train_fraction)))
    val_end = min(n - 1, max(train_end + 1, int(np.floor(n * (train_fraction + val_fraction)))))
    return {"train": ordered[:train_end], "validation": ordered[train_end:val_end], "protected_test": ordered[val_end:]}


def run_measurement_audit(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    measurement = config["measurement_audit"]
    max_samples = int(measurement.get("max_samples_per_subject", 20000))
    train_fraction = float(measurement.get("train_fraction", 0.6))
    val_fraction = float(measurement.get("validation_fraction", 0.2))
    trace_points = int(measurement.get("trace_points", 1200))
    summary_rows: list[dict[str, Any]] = []
    adapter_payload: dict[str, Any] = {}
    trace_payload: dict[str, Any] = {}
    split_payload: dict[str, Any] = {}
    joint_validation_scales: dict[str, float] = {}

    for dataset_cfg in measurement["datasets"]:
        name = str(dataset_cfg["name"])
        root = Path(dataset_cfg["root"])
        subjects = [str(item) for item in dataset_cfg["subjects"]]
        split = _split_subjects(subjects, train_fraction, val_fraction)
        split_payload[name] = split
        opened: dict[str, list[RawRecord]] = {"train": [], "validation": []}
        for split_name in ("train", "validation"):
            for subject in split[split_name]:
                opened[split_name].append(READERS[name](root, subject, max_samples))
        exemplar = opened["train"][0]
        adapter = PhysiologyMeasurementAdapter.fit(
            [record.values for record in opened["train"]],
            dataset=name,
            modality="fnirs",
            original_semantics=exemplar.semantics,
            original_unit=exemplar.unit,
            canonical_semantics="baseline-relative paired optical measurement",
            transform=exemplar.transform,
            channel_names=exemplar.channel_names,
            fit_subjects=split["train"],
        )
        adapter_payload[name] = adapter.spec.to_dict()
        for split_name, records in opened.items():
            raw_values = np.concatenate([record.values for record in records], axis=0)
            canonical_values = np.concatenate([adapter.transform(record.values) for record in records], axis=0)
            if split_name == "validation":
                _, joint_validation_scales[name] = robust_location_scale(canonical_values.ravel())
            for channel_index, channel_name in enumerate(exemplar.channel_names):
                for space, values in (("raw", raw_values), ("canonical", canonical_values)):
                    summary_rows.append(
                        {
                            "dataset": name,
                            "split": split_name,
                            "space": space,
                            "channel": channel_name,
                            "unit": exemplar.unit if space == "raw" else "train robust SD",
                            "subjects": len(records),
                            **_summary(values[:, channel_index]),
                        }
                    )

        record = opened["validation"][0]
        baseline = adapter.record_baseline(record.values)
        canonical = adapter.transform(record.values, baseline=baseline)
        positions = np.linspace(0, len(record.values) - 1, min(trace_points, len(record.values))).round().astype(int)
        full_canonical = adapter.transform(record.values, baseline=baseline)
        midpoint = len(record.values) // 2
        crop_start = max(0, midpoint - min(100, midpoint))
        crop_end = min(len(record.values), crop_start + 200)
        crop_delta = np.max(
            np.abs(adapter.transform(record.values[crop_start:crop_end], baseline=baseline) - full_canonical[crop_start:crop_end])
        )
        trace_payload[name] = {
            "subject": record.subject,
            "sample_rate_hz": record.sample_rate_hz,
            "source_path": record.source_path,
            "channel_names": list(record.channel_names),
            "raw_unit": record.unit,
            "indices": positions.tolist(),
            "time_s": (positions / record.sample_rate_hz).tolist(),
            "raw": record.values[positions].tolist(),
            "canonical": canonical[positions].tolist(),
            "crop_position_max_abs_delta": float(crop_delta),
        }

    validation_scales = list(joint_validation_scales.values())
    scale_ratio = float(max(validation_scales) / max(min(validation_scales), 1e-12))
    result = {
        "schema": "physiology_semantic_e0_v2_measurement_audit",
        "protected_test_opened": False,
        "subject_splits": split_payload,
        "adapters": adapter_payload,
        "summary_rows": summary_rows,
        "representative_traces": trace_payload,
        "validation_canonical_scale_max_ratio": scale_ratio,
        "validation_joint_canonical_scales": joint_validation_scales,
        "crop_position_invariance_max_abs_delta": float(
            max(trace["crop_position_max_abs_delta"] for trace in trace_payload.values())
        ),
    }
    (output_dir / "measurement_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output_dir / "measurement_adapters.yaml").write_text(yaml.safe_dump(adapter_payload, sort_keys=False), encoding="utf-8")
    return result


__all__ = ["RawRecord", "run_measurement_audit"]
