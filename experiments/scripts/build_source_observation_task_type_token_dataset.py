#!/usr/bin/env python
"""Augment exported source/observation token datasets with task-internal labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Mapping

import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.simultaneous_eeg_nirs_dataset import (  # noqa: E402
    SimultaneousCognitiveLoader,
    resolve_marker_event_label_names,
)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_nback_session(label: str) -> str:
    text = str(label).strip()
    for prefix in ("0-back", "2-back", "3-back"):
        if text.startswith(prefix):
            return prefix
    return text


def simultaneous_event_label_maps(data_root: Path, subject_ids: set[int]) -> Dict[tuple[str, int], Dict[int, str]]:
    maps: Dict[tuple[str, int], Dict[int, str]] = {}
    for task in ("nback", "wg"):
        loader = SimultaneousCognitiveLoader(
            str(data_root),
            task=task,
            subject_ids=sorted(subject_ids),
            modality="eeg",
            allow_deprecated=False,
        )
        for subject_id in sorted(subject_ids):
            _, markers, _ = loader.load_subject_data(subject_id, "eeg")
            labels = resolve_marker_event_label_names(markers)
            if task == "nback":
                session_labels = [normalize_nback_session(label) for label in labels if str(label).endswith("session")]
                maps[(task, subject_id)] = {idx: label for idx, label in enumerate(session_labels)}
            else:
                maps[(task, subject_id)] = {idx: str(label) for idx, label in enumerate(labels)}
    return maps


def infer_task_type_labels(
    arrays: Mapping[str, np.ndarray],
    event_maps: Mapping[tuple[str, int], Mapping[int, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    source_names = np.asarray(arrays["source_name"]).astype(str)
    source_tasks = np.asarray(arrays["source_task"]).astype(str)
    base_labels = np.asarray(arrays["label_name"]).astype(str)
    subject_ids = np.asarray(arrays["subject_id"]).astype(int)
    event_indices = np.asarray(arrays["event_idx"]).astype(int)

    labels: list[str] = []
    families: list[str] = []
    provenance: list[str] = []
    unresolved: Counter[str] = Counter()

    for source_name, source_task, base_label, subject_id, event_idx in zip(
        source_names,
        source_tasks,
        base_labels,
        subject_ids,
        event_indices,
    ):
        if source_name == "simultaneous_cognitive" and source_task == "cognitive":
            task = base_label.lower()
            mapped = event_maps.get((task, int(subject_id)), {}).get(int(event_idx))
            if mapped is None:
                mapped = base_label
                unresolved[f"{task}/subject_{int(subject_id):03d}/event_{int(event_idx):03d}"] += 1
                provenance.append("fallback_base_label")
            else:
                provenance.append("raw_simultaneous_marker")
            labels.append(str(mapped))
            families.append("nback_load" if task == "nback" else "wg_state" if task == "wg" else "cognitive")
        else:
            labels.append(str(base_label))
            families.append(str(source_task))
            provenance.append("cache_label_name")

    unique = sorted(set(labels))
    label_to_id = {label: idx for idx, label in enumerate(unique)}
    ids = np.asarray([label_to_id[label] for label in labels], dtype=np.int64)
    return (
        np.asarray(labels, dtype=str),
        ids,
        np.asarray(families, dtype=str),
        dict(unresolved),
    )


def augment_split(
    input_path: Path,
    output_path: Path,
    event_maps: Mapping[tuple[str, int], Mapping[int, str]],
) -> Dict[str, Any]:
    with np.load(input_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}

    labels, ids, families, unresolved = infer_task_type_labels(arrays, event_maps)
    arrays["task_type_label_name"] = labels
    arrays["task_type_label"] = ids
    arrays["task_type_family"] = families

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    counts = Counter(labels.tolist())
    family_counts = Counter(families.tolist())
    return {
        "split": input_path.name[: -len("_tokens.npz")],
        "input_path": str(input_path),
        "output_path": str(output_path),
        "samples": int(labels.shape[0]),
        "task_type_label_counts": dict(sorted(counts.items())),
        "task_type_family_counts": dict(sorted(family_counts.items())),
        "unresolved_mappings": unresolved,
        "arrays": {key: list(value.shape) for key, value in arrays.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-run-dir", required=True, help="Existing token export run directory")
    parser.add_argument("--output-dir", required=True, help="New downstream dataset run directory")
    parser.add_argument("--simultaneous-data-root", default="data/Simultaneous EEG&NIRS")
    args = parser.parse_args()

    token_run_dir = Path(args.token_run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    data_root = Path(args.simultaneous_data_root)
    if not data_root.is_absolute():
        data_root = project_root / data_root

    token_paths = sorted((token_run_dir / "tokens").glob("*_tokens.npz"))
    if not token_paths:
        raise FileNotFoundError(f"No token splits found under {token_run_dir / 'tokens'}")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    subject_ids: set[int] = set()
    for path in token_paths:
        with np.load(path, allow_pickle=False) as data:
            source_names = np.asarray(data["source_name"]).astype(str)
            mask = source_names == "simultaneous_cognitive"
            subject_ids.update(int(value) for value in np.asarray(data["subject_id"])[mask].tolist())

    event_maps = simultaneous_event_label_maps(data_root, subject_ids)
    output_dir.mkdir(parents=True)
    if (token_run_dir / "manifest.json").exists():
        shutil.copy2(token_run_dir / "manifest.json", output_dir / "source_token_manifest.json")
    if (token_run_dir / "config.yaml").exists():
        shutil.copy2(token_run_dir / "config.yaml", output_dir / "source_export_config.yaml")

    split_summaries = [
        augment_split(path, output_dir / "tokens" / path.name, event_maps)
        for path in token_paths
    ]
    summary = {
        "schema_version": "source_observation_task_type_token_dataset_v1",
        "source_token_run_dir": str(token_run_dir),
        "run_dir": str(output_dir),
        "simultaneous_data_root": str(data_root),
        "mapping_policy": {
            "simultaneous_cognitive/nback": "map cache event_idx 0..26 to raw nback session markers: 0-back, 2-back, 3-back",
            "simultaneous_cognitive/wg": "map cache event_idx 0..59 to raw wg markers: BL or WG",
            "single_trial_motor_imagery": "reuse cache label_name",
            "single_trial_mental_arithmetic": "reuse cache label_name",
        },
        "splits": split_summaries,
    }
    write_json(output_dir / "manifest.json", summary)
    print(json.dumps({"output_dir": str(output_dir), "splits": len(split_summaries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
