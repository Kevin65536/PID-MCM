#!/usr/bin/env python3
"""Group-safe label/identity probes for public feature caches.

This diagnostic estimates three different quantities and keeps them separate:

* ``task``: task-label predictability under a strict subject-group split;
* ``session``: session predictability under a strict subject-group split;
* ``subject_closed_set``: *closed-set row-split subject-ID decodability*.  The
  latter deliberately puts rows from the same subject in train and test,
  because a subject-ID class that is absent from training is an unknown class
  and has no closed-set accuracy estimand.  This is an upper-bound retention
  diagnostic: record/session overlap is not excluded, so it is not evidence of
  a deployable within-subject or cross-subject identity classifier.  A
  cross-subject subject-ID probe is therefore reported as ``not_estimable``
  rather than scored.

Dataset identity is read from the first component of ``sample_ids``.  A cache
containing one dataset only is reported as ``unavailable_constant``.  This is
important: a classifier cannot establish dataset shortcuts when the target is
constant, and silently replacing the target with a task name would answer a
different question.

All preprocessing state (feature scaling and the linear classifier) is fit on
the training rows of each split.  The script is public-only and fails closed
when a cache's sidecar says that the protected test was opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning


SCHEMA_VERSION = "identity_label_probes_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "comparative_methods/runs/performance_analysis/20260816_p0/identity_probes"
DEFAULT_TASKS = ("dsr", "mental_arithmetic", "motor_imagery", "nback", "visual", "wg")
METHODS = ("biot", "cbramod", "reve", "normwear")


@dataclass(frozen=True)
class CacheRecord:
    method: str
    task: str
    feature_path: Path
    metadata_path: Path
    sidecar_path: Path | None
    cache_id: str
    outer_fold: int | None
    schema: str | None


@dataclass(frozen=True)
class CacheData:
    record: CacheRecord
    features: np.ndarray
    targets: np.ndarray
    subjects: np.ndarray
    sessions: np.ndarray
    dataset_ids: np.ndarray
    sample_ids: np.ndarray
    dataset_indices: np.ndarray
    sidecar: Mapping[str, Any]


def _json_load(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_session(sample_id: str) -> str:
    for component in sample_id.split("|"):
        if component.startswith("session_"):
            return component
    return "unknown_session"


def _parse_dataset(sample_id: str) -> str:
    pieces = sample_id.split("|")
    return pieces[0] if pieces and pieces[0] else "unknown_dataset"


def _check_string_array(name: str, values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    if len(values) != n:
        raise ValueError(f"{name} has {len(values)} rows, expected {n}")
    return values.astype(str)


def discover_caches(
    root: Path,
    methods: Sequence[str],
    tasks: Sequence[str],
    *,
    all_cache_replicates: bool = False,
) -> list[CacheRecord]:
    """Discover sidecar-bound public caches without opening protected outputs."""

    records: list[CacheRecord] = []
    for method in methods:
        method = method.lower()
        for task in tasks:
            if method in {"biot", "cbramod", "reve"}:
                method_root = {
                    "biot": root / "comparative_methods/BIOT",
                    "cbramod": root / "comparative_methods/CBraMod",
                    "reve": root / "comparative_methods/REVE",
                }[method]
                for sidecar_path in sorted((method_root / "runs/feature_cache_v2" / task).glob("*.json")):
                    feature_path = sidecar_path.with_suffix(".npz")
                    if not feature_path.is_file():
                        continue
                    sidecar = _json_load(sidecar_path)
                    if not all_cache_replicates and sidecar.get("outer_fold") not in (0, None):
                        continue
                    records.append(
                        CacheRecord(
                            method=method,
                            task=task,
                            feature_path=feature_path,
                            metadata_path=feature_path,
                            sidecar_path=sidecar_path,
                            cache_id=str(sidecar.get("feature_cache_key", feature_path.stem)),
                            outer_fold=(
                                int(sidecar["outer_fold"])
                                if sidecar.get("outer_fold") is not None
                                else None
                            ),
                            schema=(str(sidecar["schema"]) if sidecar.get("schema") else None),
                        )
                    )
            elif method == "normwear":
                method_root = root / "comparative_methods/NormWear/runs/public_feature_cache_v2" / task
                for identity_path in sorted(method_root.glob("*/identity.json")):
                    directory = identity_path.parent
                    feature_path = directory / "features.npy"
                    metadata_path = directory / "metadata.npz"
                    if not feature_path.is_file() or not metadata_path.is_file():
                        continue
                    sidecar = _json_load(identity_path)
                    records.append(
                        CacheRecord(
                            method=method,
                            task=task,
                            feature_path=feature_path,
                            metadata_path=metadata_path,
                            sidecar_path=identity_path,
                            cache_id=str(sidecar.get("feature_cache_key", directory.name)),
                            outer_fold=None,
                            schema=(str(sidecar["schema"]) if sidecar.get("schema") else None),
                        )
                    )
            else:
                raise ValueError(f"unsupported method: {method}")
    return records


def load_cache(record: CacheRecord) -> CacheData:
    """Load one cache and enforce its public-only identity boundary."""

    sidecar = _json_load(record.sidecar_path) if record.sidecar_path else {}
    if bool(sidecar.get("protected_test_opened", False)):
        raise RuntimeError(
            f"refusing protected-opened cache {record.feature_path}; "
            "identity probes are public-only"
        )
    if record.method == "normwear":
        feature_data = np.load(record.feature_path, mmap_mode="r")
        metadata = np.load(record.metadata_path, allow_pickle=True)
        features = np.asarray(feature_data)
    else:
        feature_data = np.load(record.feature_path, allow_pickle=True)
        metadata = feature_data
        if "features" not in feature_data.files:
            raise ValueError(f"cache has no features array: {record.feature_path}")
        features = np.asarray(feature_data["features"])
    if features.ndim != 2 or len(features) == 0:
        raise ValueError(f"features must be non-empty 2-D: {features.shape}")
    n = len(features)
    source = metadata
    required = ("targets", "subjects", "sample_ids", "dataset_indices")
    missing = [key for key in required if key not in source.files]
    if missing:
        raise ValueError(f"cache metadata missing {missing}: {record.feature_path}")
    targets = np.asarray(source["targets"]).reshape(-1)
    if len(targets) != n:
        raise ValueError(f"targets has {len(targets)} rows, expected {n}")
    subjects = _check_string_array("subjects", source["subjects"], n)
    sample_ids = _check_string_array("sample_ids", source["sample_ids"], n)
    dataset_indices = np.asarray(source["dataset_indices"]).reshape(-1)
    if len(dataset_indices) != n:
        raise ValueError(f"dataset_indices has {len(dataset_indices)} rows, expected {n}")
    sessions = np.asarray([_parse_session(s) for s in sample_ids], dtype=str)
    dataset_ids = np.asarray([_parse_dataset(s) for s in sample_ids], dtype=str)
    # Sample IDs are the authoritative audit trail.  A mismatch means that a
    # feature cache cannot be safely joined to metadata and is rejected.
    sid_subjects = np.asarray(
        [next((piece for piece in s.split("|") if piece.startswith("subject_")), "") for s in sample_ids],
        dtype=str,
    )
    if np.any((sid_subjects != "") & (sid_subjects != subjects)):
        bad = int(np.flatnonzero((sid_subjects != "") & (sid_subjects != subjects))[0])
        raise ValueError(
            f"subject/sample_id mismatch at row {bad}: {subjects[bad]} vs {sample_ids[bad]}"
        )
    return CacheData(
        record=record,
        features=features.astype(np.float32, copy=False),
        targets=targets,
        subjects=subjects,
        sessions=sessions,
        dataset_ids=dataset_ids,
        sample_ids=sample_ids,
        dataset_indices=dataset_indices,
        sidecar=sidecar,
    )


def _class_counts(values: np.ndarray) -> dict[str, int]:
    keys, counts = np.unique(values.astype(str), return_counts=True)
    return {str(key): int(count) for key, count in zip(keys, counts)}


def _metric_row(
    *,
    cache: CacheData,
    probe: str,
    split_kind: str,
    split_index: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y_train: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    estimator_name: str,
    scale_center: bool,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "method": cache.record.method,
        "task": cache.record.task,
        "cache_id": cache.record.cache_id,
        "outer_fold": cache.record.outer_fold,
        "probe": probe,
        "estimand": (
            "strict_cross_subject_task_label"
            if probe == "task"
            else "strict_cross_subject_session_label"
            if probe == "session"
            else "closed_set_row_split_subject_identity_decodability"
        ),
        "split_kind": split_kind,
        "split_index": int(split_index),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_train_subjects": int(len(np.unique(cache.subjects[train_idx]))),
        "n_test_subjects": int(len(np.unique(cache.subjects[test_idx]))),
        "train_class_count": len(np.unique(y_train)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "estimator": estimator_name,
        "standardizer_fit_scope": "train_only",
        "standardizer_center": bool(scale_center),
        "seed": int(seed),
    }


def _fit_predict(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    seed: int,
    center: bool = False,
) -> tuple[np.ndarray, str]:
    """Fit scaler/classifier on train rows only and predict test rows."""

    unique_train = np.unique(y[train_idx])
    if len(unique_train) < 2:
        raise ValueError("training split has fewer than two classes")
    scaler = StandardScaler(with_mean=center, with_std=True, copy=False)
    # The NormWear cache is a 78k-dimensional memmap.  Scaling without
    # centering avoids a second several-hundred-MB dense copy while retaining
    # a train-fitted variance normalization.  For all methods the exact same
    # rule is used, so the metric remains a linear-probe comparison.
    X_train = scaler.fit_transform(np.asarray(X[train_idx], dtype=np.float32))
    X_test = scaler.transform(np.asarray(X[test_idx], dtype=np.float32))
    estimator = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        # This is a diagnostic probe, not a downstream model-selection run.
        # A short, fixed optimization budget keeps the high-dimensional
        # NormWear cache tractable and is recorded in the manifest below.
        max_iter=30,
        tol=1e-2,
        average=True,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        estimator.fit(X_train, y[train_idx])
    return estimator.predict(X_test), "sgd_log_loss_average"


def _subject_within_splits(subjects: np.ndarray, *, repeats: int, test_fraction: float, seed: int) -> Iterable[tuple[int, np.ndarray, np.ndarray]]:
    """Create repeated within-subject splits for the fingerprint estimand."""

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        train: list[int] = []
        test: list[int] = []
        for subject in np.unique(subjects):
            rows = np.flatnonzero(subjects == subject)
            rows = rows[rng.permutation(len(rows))]
            n_test = max(1, int(round(len(rows) * test_fraction)))
            test.extend(rows[:n_test].tolist())
            train.extend(rows[n_test:].tolist())
        yield repeat, np.asarray(sorted(train), dtype=np.int64), np.asarray(sorted(test), dtype=np.int64)


def run_cache(
    cache: CacheData,
    *,
    cv_splits: int,
    subject_repeats: int,
    subject_test_fraction: float,
    seed: int,
    center: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run estimable probes and return metric rows plus capability rows."""

    rows: list[dict[str, Any]] = []
    capabilities: list[dict[str, Any]] = []
    X = cache.features
    groups = cache.subjects
    n_groups = len(np.unique(groups))
    n_splits = min(cv_splits, n_groups)
    if n_splits < 2:
        raise ValueError(f"need at least 2 subjects for GroupKFold, got {n_groups}")
    splitter = GroupKFold(n_splits=n_splits)

    for probe, y in (("task", cache.targets), ("session", cache.sessions)):
        class_counts = _class_counts(y)
        if len(class_counts) < 2:
            capabilities.append(
                {
                    "method": cache.record.method,
                    "task": cache.record.task,
                    "cache_id": cache.record.cache_id,
                    "probe": probe,
                    "status": "unavailable_constant",
                    "estimand": f"strict_cross_subject_{probe}_label",
                    "n_classes": len(class_counts),
                    "class_counts": class_counts,
                    "reason": "target has one class in this cache",
                }
            )
            continue
        capabilities.append(
            {
                "method": cache.record.method,
                "task": cache.record.task,
                "cache_id": cache.record.cache_id,
                "probe": probe,
                "status": "available",
                "estimand": f"strict_cross_subject_{probe}_label",
                "n_classes": len(class_counts),
                "class_counts": class_counts,
                "split": "GroupKFold(subjects)",
            }
        )
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
            if len(np.unique(y[train_idx])) < 2:
                capabilities[-1]["status"] = "unavailable_fold_class_missing"
                continue
            y_pred, estimator_name = _fit_predict(
                X,
                y,
                train_idx,
                test_idx,
                seed=seed + fold,
                center=center,
            )
            rows.append(
                _metric_row(
                    cache=cache,
                    probe=probe,
                    split_kind="group_kfold_subject",
                    split_index=fold,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    y_train=y[train_idx],
                    y_true=y[test_idx],
                    y_pred=y_pred,
                    estimator_name=estimator_name,
                    scale_center=center,
                    seed=seed + fold,
                )
            )

    # A cross-subject subject-ID target is not a closed-set estimand: every
    # held-out subject is an unseen category.  Report that fact explicitly.
    capabilities.append(
        {
            "method": cache.record.method,
            "task": cache.record.task,
            "cache_id": cache.record.cache_id,
            "probe": "subject_cross_subject",
            "status": "not_estimable",
            "estimand": "strict_cross_subject_subject_identity",
            "n_classes": int(n_groups),
            "reason": "held-out subject IDs are unknown classes under group-safe split",
        }
    )
    subject_counts = _class_counts(cache.subjects)
    capabilities.append(
        {
            "method": cache.record.method,
            "task": cache.record.task,
            "cache_id": cache.record.cache_id,
            "probe": "subject_closed_set",
            "status": "available",
            "estimand": "closed_set_row_split_subject_identity_decodability",
            "n_classes": int(n_groups),
            "class_counts": subject_counts,
            "split": (
                f"{subject_repeats} repeated row splits stratified within subject "
                f"({1 - subject_test_fraction:.2f}/{subject_test_fraction:.2f}); "
                "record/session overlap is not excluded"
            ),
        }
    )
    for repeat, train_idx, test_idx in _subject_within_splits(
        groups,
        repeats=subject_repeats,
        test_fraction=subject_test_fraction,
        seed=seed,
    ):
        y_pred, estimator_name = _fit_predict(
            X,
            groups,
            train_idx,
            test_idx,
            seed=seed + 1000 + repeat,
            center=center,
        )
        rows.append(
            _metric_row(
                cache=cache,
                probe="subject_closed_set",
                split_kind="closed_set_row_split_subject",
                split_index=repeat,
                train_idx=train_idx,
                test_idx=test_idx,
                y_train=groups[train_idx],
                y_true=groups[test_idx],
                y_pred=y_pred,
                estimator_name=estimator_name,
                scale_center=center,
                seed=seed + 1000 + repeat,
            )
        )

    dataset_counts = _class_counts(cache.dataset_ids)
    capabilities.append(
        {
            "method": cache.record.method,
            "task": cache.record.task,
            "cache_id": cache.record.cache_id,
            "probe": "dataset",
            "status": "available" if len(dataset_counts) >= 2 else "unavailable_constant",
            "estimand": "strict_cross_subject_dataset_identity",
            "n_classes": len(dataset_counts),
            "class_counts": dataset_counts,
            "reason": None if len(dataset_counts) >= 2 else "all rows belong to one dataset; dataset shortcut cannot be tested within one cache",
        }
    )
    return rows, capabilities


def _summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["method"]), str(row["task"]), str(row["probe"]))
        grouped.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    for (method, task, probe), values in sorted(grouped.items()):
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            arr = np.asarray([float(value[metric]) for value in values], dtype=float)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            # 1.96 is deliberately a descriptive normal approximation here;
            # the row-level CV folds are not independent subjects.  The report
            # labels this as CV spread, not an inferential CI.
            half = float(1.96 * std / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
            result.append(
                {
                    "schema": SCHEMA_VERSION,
                    "method": method,
                    "task": task,
                    "probe": probe,
                    "metric": metric,
                    "n_splits": len(arr),
                    "mean": mean,
                    "std": std,
                    "cv_spread_low": max(0.0, mean - half),
                    "cv_spread_high": min(1.0, mean + half),
                    "interpretation": "descriptive CV spread; not an independent-subject confidence interval",
                }
            )
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _make_report(
    *,
    output: Path,
    records: Sequence[CacheRecord],
    metrics: Sequence[Mapping[str, Any]],
    capabilities: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> str:
    lines = [
        "# P0 表征身份/标签探针报告",
        "",
        f"- schema: `{SCHEMA_VERSION}`",
        f"- generated_utc: `{datetime.now(timezone.utc).isoformat()}`",
        "- 数据边界：仅读取 `protected_test_opened=false` 的 public feature cache；没有读取 protected predictions。",
        "- 标准化：每个 split 只在 train rows 上拟合方差；为了支持 NormWear 的 78,336 维 memmap，默认不中心化（`with_mean=false`），所有方法一致。",
        "- 统计解释：下表是 CV 的描述性均值、标准差和均值的正态近似 spread；不能把 CV folds 当成独立被试置信区间。",
        "- 探针优化：固定 `SGDClassifier(log_loss, average=True, max_iter=30, tol=1e-2, class_weight=balanced)`；这是跨方法一致的诊断预算，不是原论文下游微调结果。",
        "",
        "## Estimand",
        "",
        "- `task`: subject-group holdout 下的任务标签可预测性。",
        "- `session`: subject-group holdout 下的 session 标签可预测性。",
        "- `subject_closed_set`: closed-set row-split subject-ID decodability；同一 subject 的 rows 分入 train/test。这是 identity-retention 的上界诊断，未排除同一 record/session，因此不能解释为可部署的被试内分类器，也不能证明模型使用了 subject shortcut。跨被试 subject-ID 分类被标记为 `not_estimable`，因为 test 的 subject 类别在 train 中不存在。",
        "- `dataset`: 每个 cache 的 dataset ID 来自 sample ID 的第一段。当前 single-dataset cache 不人为替换为 task 名称，因此会 `unavailable_constant`。",
        "",
        "## Cache support",
        "",
        f"- cache records: `{len(records)}`",
        f"- metric rows: `{len(metrics)}`",
        f"- capability rows: `{len(capabilities)}`",
        "",
        "## CV summary",
        "",
        "| method | task | probe | metric | n | mean | std | descriptive spread |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['method']} | {row['task']} | {row['probe']} | {row['metric']} | "
            f"{row['n_splits']} | {row['mean']:.4f} | {row['std']:.4f} | "
            f"[{row['cv_spread_low']:.4f}, {row['cv_spread_high']:.4f}] |"
        )
    lines.extend(["", "## Capability statuses", "", "| method | task | probe | status | estimand/reason |", "|---|---|---|---|---|"])
    for row in capabilities:
        reason = row.get("reason") or row.get("estimand") or ""
        lines.append(f"| {row['method']} | {row['task']} | {row['probe']} | {row['status']} | {reason} |")
    lines.extend(
        [
            "",
            "## Reading the result",
            "",
            "- 若 task probe 接近 chance、`subject_closed_set` 较高，说明表示包含可重复的被试特异性而任务可迁移性弱；这只是 identity-retention 相关性证据，不单凭该 row-split probe 证明模型在训练时使用了 subject ID。",
            "- 若 session probe 明显高于 chance，需将采集 session/state shift 作为性能劣化候选因素；session probe 仍是相关性证据，不是因果证据。",
            "- dataset probe 在单一数据集 cache 内不能估计；应在后续跨数据集 public cache 合并设计中单独实现，并保持 subject-safe split。",
            "",
            f"输出目录：`{output}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(
    *,
    root: Path = REPO_ROOT,
    output: Path = DEFAULT_OUTPUT,
    methods: Sequence[str] = METHODS,
    tasks: Sequence[str] = DEFAULT_TASKS,
    all_cache_replicates: bool = False,
    cv_splits: int = 5,
    subject_repeats: int = 5,
    subject_test_fraction: float = 0.2,
    seed: int = 20260816,
    center: bool = False,
) -> dict[str, Any]:
    records = discover_caches(
        root,
        methods,
        tasks,
        all_cache_replicates=all_cache_replicates,
    )
    metrics: list[dict[str, Any]] = []
    capabilities: list[dict[str, Any]] = []
    supports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in records:
        try:
            cache = load_cache(record)
            rows, caps = run_cache(
                cache,
                cv_splits=cv_splits,
                subject_repeats=subject_repeats,
                subject_test_fraction=subject_test_fraction,
                seed=seed,
                center=center,
            )
            metrics.extend(rows)
            capabilities.extend(caps)
            supports.append(
                {
                    "method": record.method,
                    "task": record.task,
                    "cache_id": record.cache_id,
                    "outer_fold": record.outer_fold,
                    "feature_path": str(record.feature_path.relative_to(root)),
                    "metadata_or_sidecar": str(record.metadata_path.relative_to(root)),
                    "sidecar_path": str(record.sidecar_path.relative_to(root)) if record.sidecar_path else None,
                    "feature_sha256": _sha256_file(record.feature_path),
                    "sidecar_sha256": _sha256_file(record.sidecar_path) if record.sidecar_path else None,
                    "protected_test_opened": False,
                    "n_rows": int(len(cache.features)),
                    "n_features": int(cache.features.shape[1]),
                    "n_subjects": int(len(np.unique(cache.subjects))),
                    "n_sessions": int(len(np.unique(cache.sessions))),
                    "dataset_counts": _class_counts(cache.dataset_ids),
                }
            )
        except Exception as exc:  # fail closed per cache, preserving a report
            failures.append(
                {
                    "method": record.method,
                    "task": record.task,
                    "cache_id": record.cache_id,
                    "feature_path": str(record.feature_path.relative_to(root)),
                    "status": "failed_closed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    summaries = _summary(metrics)
    args = argparse.Namespace(
        all_cache_replicates=all_cache_replicates,
        cv_splits=cv_splits,
        subject_repeats=subject_repeats,
        subject_test_fraction=subject_test_fraction,
        seed=seed,
        center=center,
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "support.json", {"schema": SCHEMA_VERSION, "records": supports, "failures": failures})
    _write_jsonl(output / "metrics.jsonl", metrics)
    _write_csv(output / "metrics.csv", metrics)
    _write_jsonl(output / "capabilities.jsonl", capabilities)
    _write_json(output / "cv_summary.json", summaries)
    report = _make_report(
        output=output,
        records=records,
        metrics=metrics,
        capabilities=capabilities,
        summaries=summaries,
        args=args,
    )
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    manifest = {
        "schema": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "methods": list(methods),
        "tasks": list(tasks),
        "all_cache_replicates": all_cache_replicates,
        "cv_splits": cv_splits,
        "subject_repeats": subject_repeats,
        "subject_test_fraction": subject_test_fraction,
        "seed": seed,
        "standardizer": {"with_mean": center, "with_std": True, "fit_scope": "train_only"},
        "probe_estimator": {"class": "SGDClassifier", "loss": "log_loss", "alpha": 1e-4, "max_iter": 30, "tol": 1e-2, "average": True, "class_weight": "balanced", "n_jobs": 4},
        "cache_selection": "all outer_fold records" if all_cache_replicates else "outer_fold=0 when available; one NormWear cache per task",
        "protected_boundary": "all sidecars checked protected_test_opened=false; errors fail closed",
        "n_support_records": len(supports),
        "n_failures": len(failures),
        "n_metric_rows": len(metrics),
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--tasks", nargs="+", choices=DEFAULT_TASKS, default=list(DEFAULT_TASKS))
    parser.add_argument("--all-cache-replicates", action="store_true")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--subject-repeats", type=int, default=5)
    parser.add_argument("--subject-test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--center", action="store_true", help="center features as well as scaling (uses more memory)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = run(
        root=args.root,
        output=args.output,
        methods=args.methods,
        tasks=args.tasks,
        all_cache_replicates=args.all_cache_replicates,
        cv_splits=args.cv_splits,
        subject_repeats=args.subject_repeats,
        subject_test_fraction=args.subject_test_fraction,
        seed=args.seed,
        center=args.center,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["n_support_records"] > 0 and manifest["n_failures"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
