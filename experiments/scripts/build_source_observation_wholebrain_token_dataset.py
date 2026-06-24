#!/usr/bin/env python
"""Group single-anchor source/observation token rows into whole-brain event samples."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import scipy.io as sio

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


BRANCHES = (
    "eeg_source_tokens",
    "fnirs_source_tokens",
    "eeg_observation_tokens",
    "fnirs_observation_tokens",
)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mat_first_struct(path: Path) -> Any:
    data = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    key = next(key for key in data if not key.startswith("__"))
    return data[key]


def _names(value: Any) -> list[str]:
    return [str(item) for item in np.asarray(value, dtype=object).reshape(-1).tolist()]


def _positions_3d(mnt: Any) -> np.ndarray:
    pos = np.asarray(mnt.pos_3d, dtype=np.float32)
    if pos.ndim == 2 and pos.shape[0] == 3:
        pos = pos.T
    return pos


def _canonical(name: str) -> str:
    return "".join(char.lower() for char in str(name) if char.isalnum())


def _nearest_ownership(
    *,
    eeg_names: list[str],
    eeg_positions: np.ndarray,
    fnirs_names: list[str],
    fnirs_positions: np.ndarray,
    exclude_eog: bool,
) -> dict[str, Any]:
    usable_eeg = []
    usable_pos = []
    for name, pos in zip(eeg_names, eeg_positions):
        if exclude_eog and "eog" in name.lower():
            continue
        usable_eeg.append(name)
        usable_pos.append(pos)
    usable_pos_arr = np.asarray(usable_pos, dtype=np.float32)
    owners: dict[str, list[str]] = {name: [] for name in fnirs_names}
    for eeg_name, eeg_pos in zip(usable_eeg, usable_pos_arr):
        distances = np.sqrt(np.sum((fnirs_positions - eeg_pos.reshape(1, 3)) ** 2, axis=1))
        owner = fnirs_names[int(np.argmin(distances))]
        owners[owner].append(eeg_name)
    return {
        "eeg_channel_names": eeg_names,
        "usable_eeg_channel_names": usable_eeg,
        "fnirs_channel_names": fnirs_names,
        "exclusive_eeg_owner_by_fnirs": owners,
    }


def single_trial_channel_metadata(data_root: Path) -> dict[str, Any]:
    eeg_mnt = _mat_first_struct(data_root / "EEG_01-29" / "subject 01" / "mnt_artifact.mat")
    nirs_mnt = _mat_first_struct(data_root / "NIRS_01-29" / "subject 01" / "mnt.mat")
    eeg_names = _names(eeg_mnt.clab)
    fnirs_names = _names(nirs_mnt.clab)
    source_names = _names(nirs_mnt.source.clab)
    detector_names = _names(nirs_mnt.detector.clab)
    ownership = _nearest_ownership(
        eeg_names=eeg_names,
        eeg_positions=_positions_3d(eeg_mnt),
        fnirs_names=fnirs_names,
        fnirs_positions=_positions_3d(nirs_mnt),
        exclude_eog=True,
    )
    ownership.update(
        {
            "dataset_id": "eeg_fnirs_single_trial",
            "raw_eeg_channel_count_note": "dataset documentation: 32 total = 30 EEG + 2 EOG; artifact montage used here exposes 30 EEG channels",
            "fnirs_track_count_note": "72 raw optical channels = 36 spatial tracks x highWL/lowWL; active token export uses highWL only",
            "fnirs_source_names": source_names,
            "fnirs_detector_names": detector_names,
        }
    )
    return ownership


def simultaneous_channel_metadata(data_root: Path) -> dict[str, Any]:
    eeg_mnt = _mat_first_struct(data_root / "VP001-EEG" / "mnt_nback.mat")
    nirs_mnt = _mat_first_struct(data_root / "VP001-NIRS" / "mnt_nback.mat")
    eeg_names = _names(eeg_mnt.clab)
    fnirs_names = _names(nirs_mnt.clab)
    ownership = _nearest_ownership(
        eeg_names=eeg_names,
        eeg_positions=_positions_3d(eeg_mnt),
        fnirs_names=fnirs_names,
        fnirs_positions=_positions_3d(nirs_mnt),
        exclude_eog=True,
    )
    ownership.update(
        {
            "dataset_id": "simultaneous_eeg_nirs_cognitive",
            "raw_eeg_channel_count_note": "30 EEG file channels including HEOG and VEOG; 28 neural EEG channels after EOG exclusion",
            "fnirs_track_count_note": "36 oxy/deoxy fNIRS channels; current token export uses oxy/highWL-compatible branch",
        }
    )
    return ownership


def load_split(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def group_key(arrays: Mapping[str, np.ndarray], index: int) -> tuple[Any, ...]:
    return (
        str(arrays["source_name"][index]),
        str(arrays["source_task"][index]),
        int(arrays["subject_id"][index]),
        int(arrays["event_idx"][index]),
        int(arrays["crop_start_fnirs"][index]),
        str(arrays["label_name"][index]),
        str(arrays["task_type_label_name"][index]),
    )


def build_split(
    arrays: Mapping[str, np.ndarray],
    *,
    anchor_order: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    anchor_to_index = {anchor: index for index, anchor in enumerate(anchor_order)}
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index in range(int(arrays[BRANCHES[0]].shape[0])):
        groups[group_key(arrays, index)].append(index)

    keys = sorted(groups)
    n_samples = len(keys)
    n_anchors = len(anchor_order)
    n_branches = len(BRANCHES)
    token_count = int(arrays[BRANCHES[0]].shape[1])
    tokens = np.full((n_samples, n_anchors, n_branches, token_count), -1, dtype=np.int64)
    anchor_mask = np.zeros((n_samples, n_anchors), dtype=np.int64)
    source_name = []
    source_task = []
    label_name = []
    task_type_label_name = []
    task_type_family = []
    subject_id = np.zeros(n_samples, dtype=np.int64)
    event_idx = np.zeros(n_samples, dtype=np.int64)
    crop_start_fnirs = np.zeros(n_samples, dtype=np.int64)
    group_size = np.zeros(n_samples, dtype=np.int64)

    for sample_index, key in enumerate(keys):
        row_indices = groups[key]
        group_size[sample_index] = len(row_indices)
        first = row_indices[0]
        source_name.append(str(arrays["source_name"][first]))
        source_task.append(str(arrays["source_task"][first]))
        label_name.append(str(arrays["label_name"][first]))
        task_type_label_name.append(str(arrays["task_type_label_name"][first]))
        task_type_family.append(str(arrays["task_type_family"][first]))
        subject_id[sample_index] = int(arrays["subject_id"][first])
        event_idx[sample_index] = int(arrays["event_idx"][first])
        crop_start_fnirs[sample_index] = int(arrays["crop_start_fnirs"][first])
        for row_index in row_indices:
            anchor = str(arrays["anchor"][row_index])
            if anchor not in anchor_to_index:
                continue
            anchor_index = anchor_to_index[anchor]
            anchor_mask[sample_index, anchor_index] = 1
            for branch_index, branch in enumerate(BRANCHES):
                tokens[sample_index, anchor_index, branch_index, :] = arrays[branch][row_index]

    output = {
        "wholebrain_tokens": tokens,
        "anchor_mask": anchor_mask,
        "subject_id": subject_id,
        "event_idx": event_idx,
        "crop_start_fnirs": crop_start_fnirs,
        "group_size": group_size,
        "source_name": np.asarray(source_name, dtype=str),
        "source_task": np.asarray(source_task, dtype=str),
        "label_name": np.asarray(label_name, dtype=str),
        "task_type_label_name": np.asarray(task_type_label_name, dtype=str),
        "task_type_family": np.asarray(task_type_family, dtype=str),
    }
    summary = {
        "samples": int(n_samples),
        "anchor_slots": int(n_anchors),
        "token_count": int(token_count),
        "group_size_counts": dict(Counter(group_size.tolist())),
        "label_counts": dict(Counter(label_name)),
        "task_type_label_counts": dict(Counter(task_type_label_name)),
        "task_type_family_counts": dict(Counter(task_type_family)),
    }
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--single-trial-root", default="data/EEG+NIRS Single-Trial")
    parser.add_argument("--simultaneous-root", default="data/Simultaneous EEG&NIRS")
    args = parser.parse_args()

    token_run_dir = Path(args.token_run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    split_paths = sorted((token_run_dir / "tokens").glob("*_tokens.npz"))
    if not split_paths:
        raise FileNotFoundError(f"No token splits found under {token_run_dir / 'tokens'}")

    all_anchors: set[str] = set()
    split_arrays = {}
    for path in split_paths:
        split_name = path.name[: -len("_tokens.npz")]
        arrays = load_split(path)
        split_arrays[split_name] = arrays
        all_anchors.update(str(anchor) for anchor in np.asarray(arrays["anchor"]).astype(str).tolist())
    anchor_order = sorted(all_anchors)

    output_dir.mkdir(parents=True)
    split_summaries = []
    for split_name, arrays in split_arrays.items():
        split_output, split_summary = build_split(arrays, anchor_order=anchor_order)
        out_path = output_dir / "tokens" / f"{split_name}_wholebrain_tokens.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **split_output)
        split_summary["split"] = split_name
        split_summary["output_path"] = str(out_path)
        split_summaries.append(split_summary)

    channel_metadata = {
        "single_trial": single_trial_channel_metadata((project_root / args.single_trial_root).resolve()),
        "simultaneous": simultaneous_channel_metadata((project_root / args.simultaneous_root).resolve()),
    }
    manifest = {
        "schema_version": "source_observation_wholebrain_token_dataset_v1",
        "source_token_run_dir": str(token_run_dir),
        "run_dir": str(output_dir),
        "branch_order": list(BRANCHES),
        "anchor_order": anchor_order,
        "anchor_count": len(anchor_order),
        "split_summaries": split_summaries,
        "channel_metadata": channel_metadata,
        "group_key": [
            "source_name",
            "source_task",
            "subject_id",
            "event_idx",
            "crop_start_fnirs",
            "label_name",
            "task_type_label_name",
        ],
        "notes": [
            "Rows are grouped from single-anchor token exports into event-level whole-brain samples.",
            "EEG non-overlap is represented as exclusive channel ownership metadata; current tokens were already produced from overlapping local EEG neighborhoods.",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), "anchor_count": len(anchor_order)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
