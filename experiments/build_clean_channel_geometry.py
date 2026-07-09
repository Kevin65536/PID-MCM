#!/usr/bin/env python3
"""Build a normalized channel-geometry sidecar for the clean physiology cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.channel_geometry import (  # noqa: E402
    CHANNEL_GEOMETRY_SCHEMA,
    ChannelGeometryRecord,
    records_from_mnt,
    records_from_refed_coordinates,
    records_from_visual_ced,
)
from src.utils.io import write_json  # noqa: E402


DATA_ROOTS = {
    "eeg_fnirs_single_trial": PROJECT_ROOT / "data/EEG+NIRS Single-Trial",
    "refed": PROJECT_ROOT / "data/REFED-dataset",
    "visual_cognitive_motivation": PROJECT_ROOT / "data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults",
    "simultaneous_eeg_nirs": PROJECT_ROOT / "data/Simultaneous EEG&NIRS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATA_ROOTS), choices=list(DATA_ROOTS))
    parser.add_argument("--output-dir", default="data/cache/physiology_semantic_clean_v1/channel_geometry")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path)


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


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def _subject_id_from_name(name: str) -> int:
    return int(name.split()[-1])


def iter_single_trial(root: Path) -> Iterable[ChannelGeometryRecord]:
    for subject_dir in sorted((root / "NIRS_01-29").glob("subject *"), key=lambda item: _subject_id_from_name(item.name)):
        mnt = subject_dir / "mnt.mat"
        if mnt.exists():
            yield from records_from_mnt(
                mnt,
                dataset_id="eeg_fnirs_single_trial",
                subject=subject_dir.name,
                record_id="global_nirs_mnt",
                modality="fnirs",
                channel_role="fnirs_channel_midpoint",
                source_file=_rel(mnt),
            )
    for subject_dir in sorted((root / "EEG_01-29").glob("subject *"), key=lambda item: _subject_id_from_name(item.name)):
        for name in ("mnt.mat", "mnt_artifact.mat"):
            mnt = subject_dir / name
            if mnt.exists():
                yield from records_from_mnt(
                    mnt,
                    dataset_id="eeg_fnirs_single_trial",
                    subject=subject_dir.name,
                    record_id=name.replace(".mat", ""),
                    modality="eeg",
                    channel_role="eeg_electrode",
                    source_file=_rel(mnt),
                )
                break


def iter_simultaneous(root: Path) -> Iterable[ChannelGeometryRecord]:
    for subject_dir in sorted(root.glob("VP*-EEG")):
        subject = subject_dir.name.split("-")[0]
        for mnt in sorted(subject_dir.glob("mnt_*.mat")):
            yield from records_from_mnt(
                mnt,
                dataset_id="simultaneous_eeg_nirs",
                subject=subject,
                record_id=mnt.stem.replace("mnt_", "cnt_"),
                modality="eeg",
                channel_role="eeg_electrode",
                source_file=_rel(mnt),
            )
    for subject_dir in sorted(root.glob("VP*-NIRS")):
        subject = subject_dir.name.split("-")[0]
        for mnt in sorted(subject_dir.glob("mnt_*.mat")):
            yield from records_from_mnt(
                mnt,
                dataset_id="simultaneous_eeg_nirs",
                subject=subject,
                record_id=mnt.stem.replace("mnt_", "cnt_"),
                modality="fnirs",
                channel_role="fnirs_channel_midpoint",
                source_file=_rel(mnt),
            )


def _xlsx_rows(path: Path) -> list[list[str]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", ns):
                shared.append("".join(text.text or "" for text in item.findall(".//a:t", ns)))
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: list[list[str]] = []
        for row in root.findall(".//a:row", ns):
            values: dict[int, str] = {}
            for cell in row.findall("a:c", ns):
                ref = cell.get("r", "")
                col = _excel_column_index(ref)
                node = cell.find("a:v", ns)
                value = "" if node is None else str(node.text or "")
                if cell.get("t") == "s" and value:
                    value = shared[int(value)]
                values[col] = value
            if values:
                rows.append([values.get(index, "") for index in range(max(values) + 1)])
    return rows


def _excel_column_index(cell_ref: str) -> int:
    col = "".join(ch for ch in cell_ref if ch.isalpha())
    value = 0
    for char in col:
        value = value * 26 + (ord(char.upper()) - ord("A") + 1)
    return max(value - 1, 0)


def iter_visual(root: Path) -> Iterable[ChannelGeometryRecord]:
    ced = root / "Location.ced"
    if ced.exists():
        yield from records_from_visual_ced(ced, source_file=_rel(ced))

    reference = root / "fNIRS_to_EEG_channel_reference.xlsx"
    if reference.exists():
        for row_index, row in enumerate(_xlsx_rows(reference)[2:]):
            for probe, fnirs_idx, eeg_idx in (("Probe1", 0, 1), ("Probe2", 2, 3)):
                if len(row) <= fnirs_idx:
                    continue
                channel = str(row[fnirs_idx]).strip()
                if not channel.startswith("CH"):
                    continue
                yield ChannelGeometryRecord(
                    dataset_id="visual_cognitive_motivation",
                    subject="all",
                    record_id=probe,
                    modality="fnirs",
                    channel_name=channel,
                    channel_role="fnirs_channel_to_eeg_reference",
                    coordinate_system="referenced_to_visual_eeg_ced",
                    coordinate_units="label_reference",
                    source_file=_rel(reference),
                    metadata={
                        "nearest_eeg_label": str(row[eeg_idx]).strip() if len(row) > eeg_idx else "",
                        "row_index": row_index,
                    },
                )


def iter_records(datasets: list[str]) -> Iterable[ChannelGeometryRecord]:
    if "eeg_fnirs_single_trial" in datasets:
        yield from iter_single_trial(DATA_ROOTS["eeg_fnirs_single_trial"])
    if "refed" in datasets:
        path = DATA_ROOTS["refed"] / "fNIRS_coordinates.csv"
        if path.exists():
            yield from records_from_refed_coordinates(path, source_file=_rel(path))
    if "visual_cognitive_motivation" in datasets:
        yield from iter_visual(DATA_ROOTS["visual_cognitive_motivation"])
    if "simultaneous_eeg_nirs" in datasets:
        yield from iter_simultaneous(DATA_ROOTS["simultaneous_eeg_nirs"])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output directory exists; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [record.to_dict() for record in iter_records(args.datasets)]
    channels_path = output_dir / "channels.jsonl"
    count = _write_jsonl(channels_path, rows)
    counts: dict[str, int] = {}
    modality_counts: dict[str, int] = {}
    for row in rows:
        counts[row["dataset_id"]] = counts.get(row["dataset_id"], 0) + 1
        key = f'{row["dataset_id"]}:{row["modality"]}'
        modality_counts[key] = modality_counts.get(key, 0) + 1
    manifest = {
        "schema": CHANNEL_GEOMETRY_SCHEMA,
        "files": {"channels_jsonl": _rel(channels_path)},
        "datasets": args.datasets,
        "record_count": count,
        "counts": counts,
        "modality_counts": modality_counts,
        "notes": [
            "Coordinates preserve each dataset's native coordinate system and units.",
            "Visual fNIRS rows are channel-to-EEG label references, not measured 3D optode coordinates.",
        ],
    }
    write_json(output_dir / "geometry_manifest.json", manifest, ensure_ascii=False)
    print(json.dumps({"output_dir": str(output_dir), "record_count": count, "counts": counts}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
