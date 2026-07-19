"""Immutable, leakage-audited split manifests for STA-Net experiments."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold

from .data import STANetUnifiedTaskDataset

SPLIT_SCHEMA = "sta_net_split_registry_v2"
PROTECTED_SCHEMA = "sta_net_protected_test_v1"


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metadata(dataset: STANetUnifiedTaskDataset) -> list[dict[str, Any]]:
    rows = [dataset.lightweight_metadata(index) for index in range(len(dataset))]
    for index, row in enumerate(rows):
        row["dataset_index"] = index
    return rows


def metadata_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    stable = [
        {
            "dataset_index": int(row["dataset_index"]),
            "subject": str(row["subject"]),
            "record_id": str(row["record_id"]),
            "trial_group": str(row["trial_group"]),
            "condition": str(row["condition"]),
            "window_offset_s": float(row["window_offset_s"]),
        }
        for row in rows
    ]
    return _sha256_json(stable)


def _public_manifest(
    dataset: STANetUnifiedTaskDataset,
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: str,
    fold_id: str,
    seed: int,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    protected_descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    train = sorted(int(value) for value in train_indices)
    validation = sorted(int(value) for value in validation_indices)
    if not train or not validation:
        raise ValueError(f"{fold_id} has an empty train or validation partition")
    if set(train).intersection(validation):
        raise ValueError(f"{fold_id} train/validation partitions overlap")
    payload = {
        "schema": SPLIT_SCHEMA,
        "task": dataset.spec.key,
        "protocol": protocol,
        "fold_id": fold_id,
        "seed": int(seed),
        "metadata_sha256": metadata_fingerprint(rows),
        "train_indices": train,
        "validation_indices": validation,
        "train_sample_count": len(train),
        "validation_sample_count": len(validation),
        "train_subjects": sorted({str(rows[index]["subject"]) for index in train}),
        "validation_subjects": sorted({str(rows[index]["subject"]) for index in validation}),
        "protected_test": dict(protected_descriptor),
        "protected_test_opened": False,
    }
    payload["split_sha256"] = _sha256_json(payload)
    return payload


def validate_public_manifest(
    dataset: STANetUnifiedTaskDataset,
    manifest: Mapping[str, Any],
) -> tuple[list[int], list[int]]:
    if manifest.get("schema") != SPLIT_SCHEMA:
        raise ValueError(f"expected split schema {SPLIT_SCHEMA}")
    if manifest.get("task") != dataset.spec.key:
        raise ValueError("split task does not match the requested dataset")
    forbidden = {"test_indices", "reserved_test_indices", "protected_indices"}.intersection(manifest)
    if forbidden:
        raise ValueError(f"public training manifest exposes protected indices: {sorted(forbidden)}")
    rows = _metadata(dataset)
    if manifest.get("metadata_sha256") != metadata_fingerprint(rows):
        raise RuntimeError("split metadata fingerprint drifted from the current dataset")
    train = [int(value) for value in manifest["train_indices"]]
    validation = [int(value) for value in manifest["validation_indices"]]
    if not train or not validation or set(train).intersection(validation):
        raise RuntimeError("split has empty or overlapping train/validation indices")
    if min(train + validation) < 0 or max(train + validation) >= len(dataset):
        raise IndexError("split contains an out-of-range dataset index")
    return train, validation


def development_subject_split(
    dataset: STANetUnifiedTaskDataset,
    seed: int,
) -> tuple[list[int], list[int], dict[str, Any]]:
    rows = _metadata(dataset)
    subjects = sorted({str(row["subject"]) for row in rows})
    if len(subjects) < 3:
        raise RuntimeError(f"{dataset.spec.key} requires at least three subjects")
    rng = random.Random(seed)
    rng.shuffle(subjects)
    protected_count = max(1, round(len(subjects) * 0.15))
    validation_count = max(1, round(len(subjects) * 0.15))
    protected_subjects = sorted(subjects[:protected_count])
    validation_subjects = sorted(subjects[protected_count : protected_count + validation_count])
    train_subjects = sorted(subjects[protected_count + validation_count :])
    train_set, validation_set = set(train_subjects), set(validation_subjects)
    protected_set = set(protected_subjects)
    train = [int(row["dataset_index"]) for row in rows if row["subject"] in train_set]
    validation = [int(row["dataset_index"]) for row in rows if row["subject"] in validation_set]
    protected_indices = [int(row["dataset_index"]) for row in rows if row["subject"] in protected_set]
    protected = {
        "scope": "subject_holdout",
        "subject_count": len(protected_subjects),
        "sample_count": len(protected_indices),
        "subject_ids_sha256": _sha256_json(protected_subjects),
        "indices_sha256": _sha256_json(protected_indices),
    }
    public = _public_manifest(
        dataset, rows, protocol="development_cross_subject", fold_id="dev_holdout",
        seed=seed, train_indices=train, validation_indices=validation,
        protected_descriptor=protected,
    )
    public.update({
        "group_key": "canonical_subject_id",
        "reserved_test_subjects": protected_subjects,
        "reserved_test_sample_count": len(protected_indices),
        "regression_target_scaler": None,
    })
    return train, validation, public


def build_cross_subject_registry(
    dataset: STANetUnifiedTaskDataset,
    *,
    seed: int = 42,
    outer_folds: int = 5,
    inner_folds: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _metadata(dataset)
    subjects = np.asarray(sorted({str(row["subject"]) for row in rows}), dtype=object)
    if len(subjects) < outer_folds:
        raise ValueError("outer_folds exceeds the available subject count")
    outer = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    public, protected = [], []
    for outer_index, (outer_train_pos, test_pos) in enumerate(outer.split(subjects)):
        outer_train_subjects = subjects[outer_train_pos]
        test_subjects = sorted(str(value) for value in subjects[test_pos])
        inner_count = min(inner_folds, len(outer_train_subjects))
        inner = KFold(n_splits=inner_count, shuffle=True, random_state=seed + outer_index + 1)
        test_indices = [
            int(row["dataset_index"]) for row in rows if str(row["subject"]) in set(test_subjects)
        ]
        descriptor = {
            "scope": "unseen_subjects",
            "outer_fold": outer_index,
            "subject_count": len(test_subjects),
            "sample_count": len(test_indices),
            "subject_ids_sha256": _sha256_json(test_subjects),
            "indices_sha256": _sha256_json(test_indices),
        }
        for inner_index, (train_pos, validation_pos) in enumerate(inner.split(outer_train_subjects)):
            train_subjects = set(str(value) for value in outer_train_subjects[train_pos])
            validation_subjects = set(str(value) for value in outer_train_subjects[validation_pos])
            train_indices = [int(row["dataset_index"]) for row in rows if row["subject"] in train_subjects]
            validation_indices = [
                int(row["dataset_index"]) for row in rows if row["subject"] in validation_subjects
            ]
            public.append(_public_manifest(
                dataset, rows, protocol="cross_subject_nested_cv",
                fold_id=f"outer{outer_index}_inner{inner_index}", seed=seed,
                train_indices=train_indices, validation_indices=validation_indices,
                protected_descriptor=descriptor,
            ))
        protected.append({
            "schema": PROTECTED_SCHEMA,
            "task": dataset.spec.key,
            "protocol": "cross_subject_nested_cv",
            "outer_fold": outer_index,
            "test_subjects": test_subjects,
            "test_indices": sorted(test_indices),
            "indices_sha256": _sha256_json(sorted(test_indices)),
        })
    return public, protected


def _single_subject_groups(rows: Sequence[Mapping[str, Any]], task: str) -> np.ndarray:
    record_count = len({str(row["record_id"]) for row in rows})
    if task in {"motor_imagery", "mental_arithmetic", "visual", "refed_regression"} and record_count >= 2:
        return np.asarray([str(row["join_key"]) for row in rows], dtype=object)
    return np.asarray([str(row["trial_group"]) for row in rows], dtype=object)


def build_single_subject_registry(
    dataset: STANetUnifiedTaskDataset,
    *,
    seed: int = 42,
    max_outer_folds: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows = _metadata(dataset)
    public, protected = [], []
    for subject in sorted({str(row["subject"]) for row in all_rows}):
        subject_rows = [row for row in all_rows if str(row["subject"]) == subject]
        indices = np.asarray([int(row["dataset_index"]) for row in subject_rows], dtype=np.int64)
        groups = _single_subject_groups(subject_rows, dataset.spec.key)
        unique_groups = np.unique(groups)
        fold_count = min(max_outer_folds, len(unique_groups))
        if fold_count < 3:
            continue
        labels = np.asarray([
            int(row["class_index"] if row["class_index"] is not None else 0) for row in subject_rows
        ])
        if dataset.spec.task_type == "classification":
            splitter = StratifiedGroupKFold(n_splits=fold_count, shuffle=True, random_state=seed)
            try:
                outer_splits = list(splitter.split(indices, labels, groups))
            except ValueError:
                outer_splits = list(GroupKFold(n_splits=fold_count).split(indices, groups=groups))
        else:
            outer_splits = list(GroupKFold(n_splits=fold_count).split(indices, groups=groups))
        for outer_index, (development_pos, test_pos) in enumerate(outer_splits):
            development_pos = np.asarray(development_pos)
            test_indices = sorted(int(value) for value in indices[test_pos])
            development_groups = groups[development_pos]
            validation_group = np.unique(development_groups)[outer_index % len(np.unique(development_groups))]
            validation_mask = development_groups == validation_group
            train_indices = [int(value) for value in indices[development_pos[~validation_mask]]]
            validation_indices = [int(value) for value in indices[development_pos[validation_mask]]]
            descriptor = {
                "scope": "within_subject_held_out_group",
                "subject_ids_sha256": _sha256_json([subject]),
                "sample_count": len(test_indices),
                "indices_sha256": _sha256_json(test_indices),
            }
            fold_id = f"subject_{subject}_outer{outer_index}"
            public.append(_public_manifest(
                dataset, all_rows, protocol="single_subject_nested_cv", fold_id=fold_id,
                seed=seed, train_indices=train_indices, validation_indices=validation_indices,
                protected_descriptor=descriptor,
            ))
            protected.append({
                "schema": PROTECTED_SCHEMA,
                "task": dataset.spec.key,
                "protocol": "single_subject_nested_cv",
                "fold_id": fold_id,
                "subject": subject,
                "test_indices": test_indices,
                "indices_sha256": _sha256_json(test_indices),
            })
    return public, protected


def write_registry(
    public: Sequence[Mapping[str, Any]],
    protected: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    public_dir = output_dir / "public"
    protected_dir = output_dir / "protected"
    public_dir.mkdir(parents=True, exist_ok=True)
    protected_dir.mkdir(parents=True, exist_ok=True)
    for row in public:
        (public_dir / f"{row['fold_id']}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    for index, row in enumerate(protected):
        name = str(row.get("fold_id", f"outer{row.get('outer_fold', index)}"))
        (protected_dir / f"{name}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
