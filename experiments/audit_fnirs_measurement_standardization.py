#!/usr/bin/env python3
"""Audit native and canonical fNIRS scales across all four registered datasets.

This is intentionally a measurement audit, not a multimodal task loader.  It
can inspect REFED and Visual Cognitive Motivation before their full paired EEG
loaders exist.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Iterator

import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fnirs_standardization import (  # noqa: E402
    DATASET_FNIRS_CONTRACTS,
    FNIRSMeasurementContract,
    standardize_fnirs_record,
)


DATA_ROOTS = {
    "eeg_fnirs_single_trial": PROJECT_ROOT / "data/EEG+NIRS Single-Trial",
    "refed": PROJECT_ROOT / "data/REFED-dataset",
    "visual_cognitive_motivation": PROJECT_ROOT / "data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults",
    "simultaneous_eeg_nirs": PROJECT_ROOT / "data/Simultaneous EEG&NIRS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects-per-dataset", type=int, default=3)
    parser.add_argument("--records-per-subject", type=int, default=2)
    parser.add_argument("--output-dir", default="experiments/runs/fnirs_measurement_audit/latest")
    return parser.parse_args()


def _mat_payload(path: Path) -> Any:
    payload = loadmat(path, struct_as_record=False, squeeze_me=True)
    return payload[next(key for key in payload if not key.startswith("__"))]


def iter_single_trial(root: Path, subject_limit: int, record_limit: int) -> Iterator[dict[str, Any]]:
    subjects = sorted((root / "NIRS_01-29").glob("subject *"))[:subject_limit]
    contract = DATASET_FNIRS_CONTRACTS["eeg_fnirs_single_trial"]["wavelength_pair"]
    for subject in subjects:
        sessions = np.atleast_1d(_mat_payload(subject / "cnt.mat"))
        for index, session in enumerate(sessions[:record_limit]):
            yield {
                "dataset_id": contract.dataset_id,
                "subject": subject.name,
                "record": f"session_{index}",
                "values": np.asarray(session.x, dtype=np.float64),
                "sample_rate_hz": float(session.fs),
                "contract": contract,
                "metadata_unit": str(session.yUnit),
                "metadata_signal": str(session.signal),
            }


def iter_refed(root: Path, subject_limit: int, record_limit: int) -> Iterator[dict[str, Any]]:
    subjects = sorted((root / "data").glob("[0-9]*"), key=lambda item: int(item.name))[:subject_limit]
    contracts = DATASET_FNIRS_CONTRACTS["refed"]
    for subject in subjects:
        payload = loadmat(subject / "fNIRS_videos.mat", variable_names=[f"video_{i}" for i in range(1, record_limit + 1)])
        for key in sorted((name for name in payload if name.startswith("video_")), key=lambda name: int(name.split("_")[1])):
            tensor = np.asarray(payload[key], dtype=np.float64)
            for signal_key, indices in (("hbo_hbr", (0, 1)), ("absorbance_780_805_830", (3, 4, 5))):
                selected = tensor[list(indices)].transpose(2, 1, 0).reshape(tensor.shape[2], -1)
                yield {
                    "dataset_id": "refed",
                    "subject": subject.name,
                    "record": f"{key}_{signal_key}",
                    "values": selected,
                    "sample_rate_hz": 47.62,
                    "contract": contracts[signal_key],
                    "metadata_unit": contracts[signal_key].native_unit,
                    "metadata_signal": signal_key,
                }


def _read_etg_csv(path: Path) -> tuple[np.ndarray, float]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    data_line = next(index for index, line in enumerate(lines) if line.strip() == "Data")
    sampling_line = next(line for line in lines[:data_line] if line.startswith("Sampling Period[s]"))
    sample_period = float(next(csv.reader([sampling_line]))[1])
    rows = list(csv.reader(lines[data_line + 1 :]))
    header = rows[0]
    channel_indices = [index for index, name in enumerate(header) if re.fullmatch(r"CH\d+", name.strip())]
    values = []
    for row in rows[1:]:
        if len(row) <= max(channel_indices):
            continue
        try:
            values.append([float(row[index]) for index in channel_indices])
        except ValueError:
            continue
    return np.asarray(values, dtype=np.float64), 1.0 / sample_period


def iter_visual(root: Path, subject_limit: int, record_limit: int) -> Iterator[dict[str, Any]]:
    subjects = sorted(path for path in root.glob("S[0-9][0-9]") if (path / "fNIRS").exists())[:subject_limit]
    contract = DATASET_FNIRS_CONTRACTS["visual_cognitive_motivation"]["oxy_deoxy"]
    for subject in subjects:
        oxy_files = sorted((subject / "fNIRS").glob("*Oxy.csv"))[:record_limit]
        for oxy_path in oxy_files:
            deoxy_path = Path(str(oxy_path).replace("_Oxy.csv", "_Deoxy.csv"))
            if not deoxy_path.exists():
                continue
            oxy, fs_oxy = _read_etg_csv(oxy_path)
            deoxy, fs_deoxy = _read_etg_csv(deoxy_path)
            length = min(len(oxy), len(deoxy))
            values = np.stack((oxy[:length], deoxy[:length]), axis=2).reshape(length, -1)
            yield {
                "dataset_id": contract.dataset_id,
                "subject": subject.name,
                "record": oxy_path.stem.replace("_Oxy", ""),
                "values": values,
                "sample_rate_hz": min(fs_oxy, fs_deoxy),
                "contract": contract,
                "metadata_unit": contract.native_unit,
                "metadata_signal": "ETG-7100 Oxy/Deoxy export",
            }


def iter_simultaneous(root: Path, subject_limit: int, record_limit: int) -> Iterator[dict[str, Any]]:
    subjects = sorted(root.glob("VP*-NIRS"))[:subject_limit]
    contract = DATASET_FNIRS_CONTRACTS["simultaneous_eeg_nirs"]["oxy_deoxy"]
    for subject in subjects:
        for path in sorted(subject.glob("cnt_*.mat"))[:record_limit]:
            payload = _mat_payload(path)
            oxy = payload.oxy
            deoxy = payload.deoxy
            length = min(len(oxy.x), len(deoxy.x))
            values = np.stack((np.asarray(oxy.x)[:length], np.asarray(deoxy.x)[:length]), axis=2).reshape(length, -1)
            yield {
                "dataset_id": contract.dataset_id,
                "subject": subject.name,
                "record": path.stem,
                "values": values,
                "sample_rate_hz": float(oxy.fs),
                "contract": contract,
                "metadata_unit": str(oxy.yUnit),
                "metadata_signal": str(oxy.signal),
            }


def _summary(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    channel_std = np.nanstd(values, axis=0)
    return {
        "minimum": float(np.min(finite)),
        "p01": float(np.quantile(finite, 0.01)),
        "median": float(np.median(finite)),
        "p99": float(np.quantile(finite, 0.99)),
        "maximum": float(np.max(finite)),
        "channel_std_median": float(np.median(channel_std)),
        "channel_std_p90": float(np.quantile(channel_std, 0.9)),
    }


def audit_record(record: dict[str, Any]) -> dict[str, Any]:
    result = standardize_fnirs_record(
        record["values"],
        sample_rate_hz=record["sample_rate_hz"],
        contract=record["contract"],
    )
    return {
        key: value for key, value in record.items() if key not in {"values", "contract"}
    } | {
        "contract": record["contract"].to_dict(),
        "shape": list(record["values"].shape),
        "raw": _summary(record["values"]),
        "canonical": _summary(result.values),
        "quality": dict(result.quality),
        "standardization_state": result.state.to_dict(),
    }


def render_report(records: list[dict[str, Any]]) -> str:
    lines = ["# Cross-dataset fNIRS measurement audit", "", "All canonical values are dimensionless deviations; native measurement semantics remain explicit.", ""]
    for dataset_id in DATA_ROOTS:
        selected = [record for record in records if record["dataset_id"] == dataset_id]
        lines.extend([f"## {dataset_id}", ""])
        if not selected:
            lines.extend(["No records were discovered.", ""])
            continue
        units = sorted({record["metadata_unit"] for record in selected})
        families = sorted({record["contract"]["measurement_family"] for record in selected})
        raw_scales = [record["raw"]["channel_std_median"] for record in selected]
        canonical_scales = [record["canonical"]["channel_std_median"] for record in selected]
        residual_drift = [record["quality"]["residual_drift_sd_per_min_median"] for record in selected]
        lines.extend([
            f"- Records audited: {len(selected)}",
            f"- Native unit labels: `{units}`",
            f"- Measurement families: `{families}`",
            f"- Median native channel SD: {np.median(raw_scales):.6g}",
            f"- Median canonical channel SD: {np.median(canonical_scales):.6g}",
            f"- Median residual linear drift (canonical SD/min): {np.median(residual_drift):.6g}",
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    iterators: Iterable[Iterator[dict[str, Any]]] = (
        iter_single_trial(DATA_ROOTS["eeg_fnirs_single_trial"], args.subjects_per_dataset, args.records_per_subject),
        iter_refed(DATA_ROOTS["refed"], args.subjects_per_dataset, args.records_per_subject),
        iter_visual(DATA_ROOTS["visual_cognitive_motivation"], args.subjects_per_dataset, args.records_per_subject),
        iter_simultaneous(DATA_ROOTS["simultaneous_eeg_nirs"], args.subjects_per_dataset, args.records_per_subject),
    )
    audited = [audit_record(record) for iterator in iterators for record in iterator]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(json.dumps({"records": audited}, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(render_report(audited), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "records": len(audited)}, indent=2))


if __name__ == "__main__":
    main()
