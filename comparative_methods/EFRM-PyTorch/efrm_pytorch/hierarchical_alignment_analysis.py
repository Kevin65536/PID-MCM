"""Duplicate-aware, hierarchical EFRM alignment analysis.

This module consumes only the explicitly exported *full public-validation*
alignment evidence.  It never opens a protected split.  The original EFRM
export uses an identity diagonal as the positive mask, but the validation
sampler may repeat a ``sample_id``.  Such repeated rows are exact duplicate
positives, not negative instances.  Therefore the report has two views:

``unique_sample``
    Stable first-occurrence de-duplication by ``sample_id``.  This is the
    primary retrieval view.
``raw_duplicate_aware``
    All exported rows are retained; every candidate with the query's
    ``sample_id`` is treated as a positive and is removed from all negative
    pools.  Retrieval uses the best supported positive rank.

For each view and direction, metrics are reported for the exact positive
against all negatives and four explicit negative relations:

* same-subject wrong-time (same dataset/subject/record, different sample),
* same-dataset different-subject,
* cross-dataset, and
* same-class wrong-instance (same dataset/task/condition, different record).

The relation pools are intentionally allowed to overlap: they are mechanism
diagnostics, not a mutually-exclusive partition.  Subject and record block
permutations are run on the de-duplicated view, where each sample has one
candidate row.  They preserve the block while breaking exact sample
correspondence.

The CLI is fail-closed.  Missing schema, metadata, manifest, or a non-full
validation artifact produces a machine-readable failure report and a
non-zero exit status; no fabricated metrics or plots are written.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EVIDENCE_SCHEMA = "efrm_clip_alignment_evidence_v1"
ANALYSIS_SCHEMA = "efrm_hierarchical_alignment_analysis_v1"
REQUIRED_METADATA_FIELDS = (
    "condition",
    "crop_start_s",
    "dataset_id",
    "join_key",
    "record_id",
    "sample_id",
    "subject",
    "task_namespace",
)
RELATIONS = (
    "exact_pair",
    "same_subject_wrong_time",
    "same_dataset_different_subject",
    "cross_dataset",
    "same_class_wrong_instance",
)
BLOCK_TYPES = ("subject_block", "record_block")
DEFAULT_PERMUTATIONS = 200
DEFAULT_SEED = 20260816


class EvidenceError(ValueError):
    """Raised when evidence cannot be used without making assumptions."""


@dataclass(frozen=True)
class Evidence:
    path: Path
    run_id: str
    manifest: dict[str, Any]
    cosine: np.ndarray
    eeg_embeddings: np.ndarray
    fnirs_embeddings: np.ndarray
    metadata: tuple[dict[str, Any], ...]
    positive_mask: np.ndarray


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "run"


def _read_manifest_for_evidence(path: Path) -> tuple[Path, dict[str, Any]]:
    run_dir = path.parent.parent
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise EvidenceError(
            f"missing run manifest {manifest_path}; re-export a complete run before "
            f"analysis: python comparative_methods/EFRM-PyTorch/build_alignment_evidence_v2.py "
            f"--help"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EvidenceError(f"invalid JSON run manifest: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise EvidenceError(f"run manifest is not an object: {manifest_path}")
    if str(manifest.get("status")) != "completed":
        raise EvidenceError(
            f"run manifest status is {manifest.get('status')!r}, expected 'completed': "
            f"{manifest_path}"
        )
    if bool(manifest.get("protected_test_opened", True)):
        raise EvidenceError(
            f"evidence manifest does not certify protected_test_opened=false: {manifest_path}"
        )
    return manifest_path, manifest


def load_evidence(path: str | Path, *, require_full_validation: bool = True) -> Evidence:
    """Load and validate one complete public-validation evidence artifact."""

    evidence_path = Path(path).resolve()
    if not evidence_path.is_file():
        raise EvidenceError(f"evidence file does not exist: {evidence_path}")
    if require_full_validation and evidence_path.name != "full_validation_clip_alignment_evidence.npz":
        raise EvidenceError(
            f"refusing non-full-validation artifact {evidence_path.name}; expected "
            "full_validation_clip_alignment_evidence.npz"
        )
    manifest_path, manifest = _read_manifest_for_evidence(evidence_path)
    try:
        payload = np.load(evidence_path, allow_pickle=False)
    except Exception as error:  # pragma: no cover - numpy emits several subclasses
        raise EvidenceError(f"cannot open NPZ evidence {evidence_path}: {error}") from error
    with payload:
        missing = [
            key
            for key in (
                "schema",
                "eeg_embeddings",
                "fnirs_embeddings",
                "cosine_similarity",
                "positive_pair_mask",
                "metadata_json",
            )
            if key not in payload.files
        ]
        if missing:
            raise EvidenceError(
                f"{evidence_path} is missing fields {missing}; re-export with the existing "
                "export_alignment_evidence(...) path before running this analysis"
            )
        schema = str(np.asarray(payload["schema"]).item())
        if schema != EVIDENCE_SCHEMA:
            raise EvidenceError(f"unsupported evidence schema {schema!r} in {evidence_path}")
        cosine = np.asarray(payload["cosine_similarity"], dtype=np.float64)
        eeg = np.asarray(payload["eeg_embeddings"])
        fnirs = np.asarray(payload["fnirs_embeddings"])
        positive_mask = np.asarray(payload["positive_pair_mask"], dtype=bool)
        raw_metadata = np.asarray(payload["metadata_json"])
        try:
            metadata = tuple(json.loads(str(value)) for value in raw_metadata.tolist())
        except (TypeError, json.JSONDecodeError) as error:
            raise EvidenceError(f"metadata_json is not a valid list of objects: {evidence_path}") from error

    if cosine.ndim != 2 or cosine.shape[0] != cosine.shape[1]:
        raise EvidenceError(f"cosine_similarity must be square, got {cosine.shape}")
    n = cosine.shape[0]
    if eeg.ndim != 2 or fnirs.ndim != 2 or eeg.shape[0] != n or fnirs.shape[0] != n:
        raise EvidenceError(
            f"embedding row counts do not match cosine matrix: eeg={eeg.shape}, fnirs={fnirs.shape}, n={n}"
        )
    if positive_mask.shape != (n, n):
        raise EvidenceError(f"positive_pair_mask shape {positive_mask.shape} != {(n, n)}")
    if not np.all(np.isfinite(cosine)):
        raise EvidenceError("cosine_similarity contains non-finite values")
    if len(metadata) != n:
        raise EvidenceError(f"metadata_json length {len(metadata)} != cosine rows {n}")
    missing_fields = sorted({field for row in metadata for field in REQUIRED_METADATA_FIELDS if field not in row})
    if missing_fields:
        raise EvidenceError(
            f"metadata is missing fields {missing_fields}; re-export pair metadata with fields "
            f"{list(REQUIRED_METADATA_FIELDS)}"
        )
    if not np.array_equal(positive_mask, np.eye(n, dtype=bool)):
        raise EvidenceError(
            "positive_pair_mask is not an identity matrix; this analysis assumes one synchronized "
            "diagonal pair and derives duplicate-aware positives from sample_id"
        )
    run_id = str(manifest.get("run_id") or run_dir_from_evidence(evidence_path).name)
    # Keep the certification path visible in the in-memory provenance.
    manifest = dict(manifest)
    manifest["manifest_path"] = str(manifest_path)
    return Evidence(
        evidence_path,
        run_id,
        manifest,
        cosine,
        np.asarray(eeg, dtype=np.float64),
        np.asarray(fnirs, dtype=np.float64),
        metadata,
        positive_mask,
    )


def run_dir_from_evidence(path: Path) -> Path:
    return path.parent.parent


def _metadata_arrays(metadata: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    return {
        field: np.asarray([str(row.get(field, "")) for row in metadata], dtype=object)
        for field in REQUIRED_METADATA_FIELDS
    }


def stable_unique_indices(metadata: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Return first occurrence of every sample_id, preserving export order."""

    seen: set[str] = set()
    selected: list[int] = []
    for index, row in enumerate(metadata):
        sample_id = str(row["sample_id"])
        if sample_id not in seen:
            seen.add(sample_id)
            selected.append(index)
    return np.asarray(selected, dtype=np.int64)


def duplicate_summary(
    metadata: Sequence[Mapping[str, Any]],
    eeg_embeddings: np.ndarray | None = None,
    fnirs_embeddings: np.ndarray | None = None,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(metadata):
        groups.setdefault(str(row["sample_id"]), []).append(index)
    duplicate_groups = [indices for indices in groups.values() if len(indices) > 1]
    n = len(metadata)
    off_diagonal_duplicate_pairs = int(sum(len(indices) * (len(indices) - 1) for indices in duplicate_groups))
    all_off_diagonal_pairs = n * (n - 1)
    inconsistent: list[str] = []
    comparable_fields = ("dataset_id", "subject", "join_key", "record_id", "condition", "task_namespace", "crop_start_s")
    for sample_id, indices in groups.items():
        first = metadata[indices[0]]
        if any(
            tuple(str(metadata[index].get(field, "")) for field in comparable_fields)
            != tuple(str(first.get(field, "")) for field in comparable_fields)
            for index in indices[1:]
        ):
            inconsistent.append(sample_id)
    summary = {
        "row_count": n,
        "unique_sample_count": len(groups),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_row_count_excess": int(sum(len(indices) - 1 for indices in duplicate_groups)),
        "max_duplicate_group_size": int(max((len(indices) for indices in duplicate_groups), default=1)),
        "off_diagonal_duplicate_positive_pairs": off_diagonal_duplicate_pairs,
        "all_off_diagonal_pairs": all_off_diagonal_pairs,
        "false_negative_pair_rate_if_diagonal_only": float(
            off_diagonal_duplicate_pairs / all_off_diagonal_pairs
            if all_off_diagonal_pairs else 0.0
        ),
        "inconsistent_duplicate_group_count": len(inconsistent),
        "inconsistent_duplicate_sample_ids": inconsistent[:20],
    }
    if eeg_embeddings is not None or fnirs_embeddings is not None:
        if eeg_embeddings is None or fnirs_embeddings is None:
            raise ValueError("provide both EEG and fNIRS embeddings for duplicate identity audit")
        eeg = np.asarray(eeg_embeddings)
        fnirs = np.asarray(fnirs_embeddings)
        if eeg.shape[0] != n or fnirs.shape[0] != n:
            raise ValueError("embedding rows do not match metadata for duplicate identity audit")
        eeg_diffs: list[float] = []
        fnirs_diffs: list[float] = []
        for indices in duplicate_groups:
            eeg_diffs.append(float(np.max(np.abs(eeg[indices] - eeg[indices[0]]))))
            fnirs_diffs.append(float(np.max(np.abs(fnirs[indices] - fnirs[indices[0]]))))
        summary.update({
            "duplicate_eeg_embedding_max_abs_diff": float(max(eeg_diffs, default=0.0)),
            "duplicate_fnirs_embedding_max_abs_diff": float(max(fnirs_diffs, default=0.0)),
        })
    return summary


def embedding_geometry(embeddings: np.ndarray) -> dict[str, float | int]:
    """Return centered effective rank and angular concentration diagnostics."""

    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(f"embedding geometry requires [n,d] with n>=2, got {values.shape}")
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular * singular
    total = float(energy.sum())
    if total <= 0.0:
        effective_rank = 0.0
        first_axis = 1.0
    else:
        probability = energy / total
        nonzero = probability[probability > 0.0]
        effective_rank = float(np.exp(-np.sum(nonzero * np.log(nonzero))))
        first_axis = float(probability[0])
    normalized = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    # The full within-modality cosine matrix is unnecessary for this scalar;
    # compute its off-diagonal moments in deterministic chunks.
    n = len(normalized)
    sum_cosine = 0.0
    sum_square = 0.0
    count = 0
    for start in range(0, n, 512):
        block = normalized[start : start + 512] @ normalized.T
        rows = np.arange(start, min(start + 512, n))
        block[np.arange(len(rows)), rows] = np.nan
        finite = block[np.isfinite(block)]
        sum_cosine += float(finite.sum())
        sum_square += float(np.square(finite).sum())
        count += int(finite.size)
    mean = sum_cosine / count if count else math.nan
    variance = max(sum_square / count - mean * mean, 0.0) if count else math.nan
    return {
        "sample_count": int(n),
        "embedding_dim": int(values.shape[1]),
        "effective_rank": effective_rank,
        "first_axis_energy_fraction": first_axis,
        "rank_one_warning": bool(effective_rank < 2.0 and first_axis > 0.95),
        "off_diagonal_cosine_mean": float(mean),
        "off_diagonal_cosine_sd": float(math.sqrt(variance)) if np.isfinite(variance) else math.nan,
    }


def _positive_groups(
    query_metadata: Sequence[Mapping[str, Any]],
    candidate_metadata: Sequence[Mapping[str, Any]],
) -> list[np.ndarray]:
    candidates: dict[str, list[int]] = {}
    for index, row in enumerate(candidate_metadata):
        candidates.setdefault(str(row["sample_id"]), []).append(index)
    return [
        np.asarray(candidates.get(str(row["sample_id"]), []), dtype=np.int64)
        for row in query_metadata
    ]


def _relation_mask(
    relation: str,
    query: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> bool:
    if relation in ("exact_pair", "all_negative"):
        return True
    query_dataset = str(query["dataset_id"])
    candidate_dataset = str(candidate["dataset_id"])
    if relation == "same_subject_wrong_time":
        return (
            query_dataset == candidate_dataset
            and str(query["subject"]) == str(candidate["subject"])
            and str(query["join_key"]) == str(candidate["join_key"])
            and str(query["sample_id"]) != str(candidate["sample_id"])
        )
    if relation == "same_dataset_different_subject":
        return (
            query_dataset == candidate_dataset
            and str(query["subject"]) != str(candidate["subject"])
        )
    if relation == "cross_dataset":
        return query_dataset != candidate_dataset
    if relation == "same_class_wrong_instance":
        return (
            query_dataset == candidate_dataset
            and str(query["task_namespace"]) == str(candidate["task_namespace"])
            and str(query["condition"]) == str(candidate["condition"])
            and str(query["join_key"]) != str(candidate["join_key"])
        )
    raise ValueError(f"unknown relation {relation}")


def _relation_mask_for_query(
    relation: str,
    query_index: int,
    arrays: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Vectorized candidate relation mask for one query row."""

    n = len(arrays["dataset_id"])
    if relation in ("exact_pair", "all_negative"):
        return np.ones(n, dtype=bool)
    dataset = arrays["dataset_id"]
    subject = arrays["subject"]
    join_key = arrays["join_key"]
    task = arrays["task_namespace"]
    condition = arrays["condition"]
    same_dataset = dataset == dataset[query_index]
    if relation == "same_subject_wrong_time":
        return (
            same_dataset
            & (subject == subject[query_index])
            & (join_key == join_key[query_index])
            & (arrays["sample_id"] != arrays["sample_id"][query_index])
        )
    if relation == "same_dataset_different_subject":
        return same_dataset & (subject != subject[query_index])
    if relation == "cross_dataset":
        return ~same_dataset
    if relation == "same_class_wrong_instance":
        return (
            same_dataset
            & (task == task[query_index])
            & (condition == condition[query_index])
            & (join_key != join_key[query_index])
        )
    raise ValueError(f"unknown relation {relation}")


def _auc_single(positive_score: float, negative_scores: np.ndarray) -> float:
    negatives = np.asarray(negative_scores, dtype=np.float64)
    if negatives.size == 0:
        return math.nan
    sorted_negatives = np.sort(negatives)
    below = int(np.searchsorted(sorted_negatives, positive_score, side="left"))
    at_or_below = int(np.searchsorted(sorted_negatives, positive_score, side="right"))
    return float((below + 0.5 * (at_or_below - below)) / negatives.size)


def _aggregate(values: Sequence[float]) -> tuple[float, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return math.nan, math.nan
    return float(finite.mean()), float(finite.std(ddof=1)) if finite.size > 1 else 0.0


def relation_metrics(
    cosine: np.ndarray,
    query_metadata: Sequence[Mapping[str, Any]],
    candidate_metadata: Sequence[Mapping[str, Any]],
    positive_groups: Sequence[np.ndarray],
    relation: str,
) -> dict[str, Any]:
    """Compute macro-query metrics against one explicit negative relation."""

    matrix = np.asarray(cosine, dtype=np.float64)
    if matrix.shape != (len(query_metadata), len(candidate_metadata)):
        raise ValueError("cosine and metadata shapes do not agree")
    if len(positive_groups) != len(query_metadata):
        raise ValueError("positive_groups length does not match query count")
    n_candidates = len(candidate_metadata)
    all_indices = np.arange(n_candidates, dtype=np.int64)
    candidate_arrays = _metadata_arrays(candidate_metadata)
    values: dict[str, list[float]] = {
        "auc": [],
        "mrr": [],
        "recall_at_1": [],
        "recall_at_5": [],
        "mean_rank": [],
        "hardest_margin": [],
        "positive_score": [],
        "negative_score": [],
    }
    negative_pair_count = 0
    eligible_queries = 0
    positive_support_sizes: list[int] = []
    for query_index, query in enumerate(query_metadata):
        positive_indices = np.asarray(positive_groups[query_index], dtype=np.int64)
        if positive_indices.size == 0:
            continue
        positive_indices = positive_indices[(positive_indices >= 0) & (positive_indices < n_candidates)]
        if positive_indices.size == 0:
            continue
        candidate_relation = _relation_mask_for_query(
            relation, query_index, candidate_arrays
        )
        # Duplicate sample IDs are positive support even when their metadata
        # happens to match a relation; never count them as negatives.
        candidate_relation[positive_indices] = False
        negative_indices = all_indices[candidate_relation]
        if negative_indices.size == 0:
            continue
        positive_score = float(np.max(matrix[query_index, positive_indices]))
        negative_scores = matrix[query_index, negative_indices]
        rank = 1 + int(np.count_nonzero(negative_scores > positive_score))
        values["auc"].append(_auc_single(positive_score, negative_scores))
        values["mrr"].append(1.0 / rank)
        values["recall_at_1"].append(float(rank <= 1))
        values["recall_at_5"].append(float(rank <= 5))
        values["mean_rank"].append(float(rank))
        values["hardest_margin"].append(float(positive_score - np.max(negative_scores)))
        values["positive_score"].append(positive_score)
        values["negative_score"].extend(float(score) for score in negative_scores.tolist())
        negative_pair_count += int(negative_indices.size)
        eligible_queries += 1
        positive_support_sizes.append(int(positive_indices.size))
    result: dict[str, Any] = {
        "relation": relation,
        "query_count": int(len(query_metadata)),
        "eligible_query_count": int(eligible_queries),
        "negative_pair_count": int(negative_pair_count),
        "positive_support_mean": float(np.mean(positive_support_sizes)) if positive_support_sizes else math.nan,
        "positive_support_max": int(max(positive_support_sizes, default=0)),
    }
    for key in ("auc", "mrr", "recall_at_1", "recall_at_5", "mean_rank", "hardest_margin", "positive_score"):
        mean, sd = _aggregate(values[key])
        result[key] = mean
        result[f"{key}_sd_across_queries"] = sd
    # Negative cosine is pair-weighted, while retrieval metrics are macro-query.
    result["negative_score_mean_pair_weighted"] = (
        float(np.mean(values["negative_score"])) if values["negative_score"] else math.nan
    )
    return result


def _global_scalar_metrics(
    cosine: np.ndarray,
    positive_groups: Sequence[np.ndarray],
    *,
    positive_assignments: Sequence[np.ndarray] | None = None,
) -> dict[str, float]:
    """Metrics used for block nulls; supports one or many positives per query."""

    matrix = np.asarray(cosine, dtype=np.float64)
    n_query, n_candidate = matrix.shape
    all_indices = np.arange(n_candidate, dtype=np.int64)
    aucs: list[float] = []
    mrrs: list[float] = []
    r1: list[float] = []
    r5: list[float] = []
    margins: list[float] = []
    positives: list[float] = []
    for i in range(n_query):
        if positive_assignments is not None:
            positive_indices = np.asarray(positive_assignments[i], dtype=np.int64)
        else:
            positive_indices = np.asarray(positive_groups[i], dtype=np.int64)
        if positive_indices.size == 0:
            continue
        positive_indices = positive_indices[(positive_indices >= 0) & (positive_indices < n_candidate)]
        if positive_indices.size == 0:
            continue
        mask = ~np.isin(all_indices, positive_indices)
        negative_scores = matrix[i, mask]
        if negative_scores.size == 0:
            continue
        positive_score = float(np.max(matrix[i, positive_indices]))
        rank = 1 + int(np.count_nonzero(negative_scores > positive_score))
        aucs.append(_auc_single(positive_score, negative_scores))
        mrrs.append(1.0 / rank)
        r1.append(float(rank <= 1))
        r5.append(float(rank <= 5))
        margins.append(positive_score - float(np.max(negative_scores)))
        positives.append(positive_score)
    return {
        "auc": float(np.mean(aucs)) if aucs else math.nan,
        "mrr": float(np.mean(mrrs)) if mrrs else math.nan,
        "recall_at_1": float(np.mean(r1)) if r1 else math.nan,
        "recall_at_5": float(np.mean(r5)) if r5 else math.nan,
        "hardest_margin": float(np.mean(margins)) if margins else math.nan,
        "positive_score": float(np.mean(positives)) if positives else math.nan,
    }


def _rank_cache(cosine: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare rank and top-two caches for repeated block-null evaluations.

    ``argsort`` is paid once per direction.  The inverse rank matrix is int32,
    so hundreds of block permutations only require vectorized gathers rather
    than repeated O(n log n) row sorting.
    """

    matrix = np.asarray(cosine, dtype=np.float64)
    n_query, n_candidate = matrix.shape
    order = np.argsort(-matrix, axis=1, kind="stable")
    rank = np.empty(order.shape, dtype=np.int32)
    rows = np.arange(n_query, dtype=np.int64)[:, None]
    rank[rows, order] = np.arange(n_candidate, dtype=np.int32)[None, :] + 1
    if n_candidate == 1:
        top_idx = np.zeros((n_query, 2), dtype=np.int64)
        top_scores = np.repeat(matrix[:, :1], 2, axis=1)
    else:
        top_idx = order[:, :2].astype(np.int64, copy=True)
        top_scores = np.take_along_axis(matrix, top_idx, axis=1)
    del order
    return rank, top_idx, top_scores


def _global_scalar_metrics_fast(
    cosine: np.ndarray,
    rank: np.ndarray,
    top_idx: np.ndarray,
    top_scores: np.ndarray,
    assignments: np.ndarray,
) -> dict[str, float]:
    """Fast one-positive metrics for a single block permutation."""

    matrix = np.asarray(cosine, dtype=np.float64)
    assignments = np.asarray(assignments, dtype=np.int64)
    rows = np.arange(len(assignments), dtype=np.int64)
    ranks = rank[rows, assignments].astype(np.float64)
    positive = matrix[rows, assignments]
    if matrix.shape[1] > 1:
        hardest = np.where(
            assignments == top_idx[:, 0], top_scores[:, 1], top_scores[:, 0]
        )
    else:
        hardest = np.full(len(assignments), -np.inf, dtype=np.float64)
    denominator = max(matrix.shape[1] - 1, 1)
    # Ties are vanishingly rare for non-duplicated cosine values.  The rank
    # based AUC is deterministic and is exact under strict ordering.
    auc = (matrix.shape[1] - ranks) / denominator
    return {
        "auc": float(np.mean(auc)),
        "mrr": float(np.mean(1.0 / ranks)),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "hardest_margin": float(np.mean(positive - hardest)),
        "positive_score": float(np.mean(positive)),
    }


def _block_groups(metadata: Sequence[Mapping[str, Any]], block_type: str) -> list[np.ndarray]:
    if block_type == "subject_block":
        keys = [f"{row['dataset_id']}|{row['subject']}" for row in metadata]
    elif block_type == "record_block":
        keys = [f"{row['dataset_id']}|{row['subject']}|{row['join_key']}" for row in metadata]
    else:
        raise ValueError(f"unknown block type {block_type}")
    groups: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    return [np.asarray(indices, dtype=np.int64) for indices in groups.values()]


def _within_block_permutation(groups: Sequence[np.ndarray], rng: np.random.Generator, n: int) -> np.ndarray:
    permutation = np.arange(n, dtype=np.int64)
    for group in groups:
        if len(group) > 1:
            permutation[group] = rng.permutation(group)
    return permutation


def block_permutation_metrics(
    cosine: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    observed_positive_groups: Sequence[np.ndarray],
    *,
    permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Run subject/record within-block nulls on a unique-sample matrix."""

    matrix = np.asarray(cosine, dtype=np.float64)
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != len(metadata):
        raise ValueError("block permutation requires a square unique-sample matrix")
    n = matrix.shape[0]
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for direction, directional_matrix in (("eeg_to_fnirs", matrix), ("fnirs_to_eeg", matrix.T)):
        rank, top_idx, top_scores = _rank_cache(directional_matrix)
        identity = np.arange(n, dtype=np.int64)
        observed = _global_scalar_metrics_fast(
            directional_matrix, rank, top_idx, top_scores, identity
        )
        for block_type in BLOCK_TYPES:
            groups = _block_groups(metadata, block_type)
            null_values = {metric: [] for metric in observed}
            no_op_count = 0
            for _ in range(int(permutations)):
                permutation = _within_block_permutation(groups, rng, n)
                if np.array_equal(permutation, np.arange(n)):
                    no_op_count += 1
                values = _global_scalar_metrics_fast(
                    directional_matrix, rank, top_idx, top_scores, permutation
                )
                for metric, value in values.items():
                    null_values[metric].append(value)
            for metric, observed_value in observed.items():
                null = np.asarray(null_values[metric], dtype=np.float64)
                finite = null[np.isfinite(null)]
                if finite.size:
                    p_greater = (1 + int(np.count_nonzero(finite >= observed_value))) / (finite.size + 1)
                    p_less = (1 + int(np.count_nonzero(finite <= observed_value))) / (finite.size + 1)
                    p_two = min(1.0, 2.0 * min(p_greater, p_less))
                    null_mean = float(np.mean(finite))
                    null_sd = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
                else:
                    p_greater = p_less = p_two = null_mean = null_sd = math.nan
                rows.append({
                    "direction": direction,
                    "block_type": block_type,
                    "metric": metric,
                    "observed": observed_value,
                    "null_mean": null_mean,
                    "null_sd": null_sd,
                    "p_greater": p_greater,
                    "p_less": p_less,
                    "p_two_sided": p_two,
                    "permutations": int(permutations),
                    "no_op_permutations": int(no_op_count),
                    "block_count": len(groups),
                    "block_size_min": int(min((len(group) for group in groups), default=0)),
                    "block_size_max": int(max((len(group) for group in groups), default=0)),
                })
    return rows


def _csv_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames = sorted({str(key) for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key)) for key in fieldnames})


def _plot_report(
    run_id: str,
    metrics_rows: Sequence[Mapping[str, Any]],
    permutation_rows: Sequence[Mapping[str, Any]],
    output_base: Path,
) -> None:
    relations = list(RELATIONS)
    relation_labels = {
        "exact_pair": "exact",
        "same_subject_wrong_time": "same-subj-time",
        "same_dataset_different_subject": "other-subj",
        "cross_dataset": "cross-data",
        "same_class_wrong_instance": "same-class",
    }
    short_label = (
        "exclude-ST"
        if "exclude_eeg_fnirs_single_trial" in run_id
        else "exclude-Sim"
        if "exclude_simultaneous_eeg_nirs" in run_id
        else _safe_name(run_id)[:24]
    )
    directions = ("eeg_to_fnirs", "fnirs_to_eeg")
    view = "unique_sample"
    selected = [row for row in metrics_rows if row.get("view") == view]
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.0), constrained_layout=True)
    colors = {"eeg_to_fnirs": "#0072B2", "fnirs_to_eeg": "#D55E00"}
    x = np.arange(len(relations))
    width = 0.36
    for offset, direction in ((-width / 2, directions[0]), (width / 2, directions[1])):
        values = [next((float(row["auc"]) for row in selected if row["direction"] == direction and row["relation"] == relation), math.nan) for relation in relations]
        axes[0, 0].bar(
            x + offset,
            values,
            width,
            label="EEG→fNIRS" if direction == "eeg_to_fnirs" else "fNIRS→EEG",
            color=colors[direction],
        )
    axes[0, 0].set_xticks(x, [relation_labels[r] for r in relations], rotation=27, ha="right")
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].set_ylabel("macro-query AUC")
    axes[0, 0].set_title("Pair separation by hierarchical negative relation")
    axes[0, 0].axhline(0.5, color="#666666", ls="--", lw=0.8)
    axes[0, 0].legend(fontsize=8)
    for offset, direction in ((-width / 2, directions[0]), (width / 2, directions[1])):
        values = [next((float(row["mrr"]) for row in selected if row["direction"] == direction and row["relation"] == relation), math.nan) for relation in relations]
        axes[0, 1].bar(
            x + offset,
            values,
            width,
            label="EEG→fNIRS" if direction == "eeg_to_fnirs" else "fNIRS→EEG",
            color=colors[direction],
        )
    axes[0, 1].set_xticks(x, [relation_labels[r] for r in relations], rotation=27, ha="right")
    axes[0, 1].set_ylabel("MRR (relation-restricted)")
    axes[0, 1].set_title("Exact-pair retrieval against each negative pool")
    for block_type, color in (("subject_block", "#009E73"), ("record_block", "#CC79A7")):
        values = [float(row["observed"]) for row in permutation_rows if row["direction"] == "eeg_to_fnirs" and row["block_type"] == block_type and row["metric"] == "mrr"]
        nulls = [float(row["null_mean"]) for row in permutation_rows if row["direction"] == "eeg_to_fnirs" and row["block_type"] == block_type and row["metric"] == "mrr"]
        if values:
            axes[1, 0].bar(
                block_type,
                values[0],
                color=color,
                alpha=0.85,
                label="observed" if block_type == "subject_block" else None,
            )
            axes[1, 0].scatter([block_type], [nulls[0]], color="#222222", marker="x", s=55, label="null mean" if block_type == "subject_block" else None)
    axes[1, 0].set_ylabel("EEG→fNIRS MRR")
    axes[1, 0].set_title("Within-block permutation null")
    axes[1, 0].legend(fontsize=8)
    for direction, color in colors.items():
        margins = [float(row["hardest_margin"]) for row in selected if row["direction"] == direction]
        labels = [str(row["relation"]) for row in selected if row["direction"] == direction]
        if margins:
            axes[1, 1].plot(
                labels,
                margins,
                marker="o",
                color=color,
                label="EEG→fNIRS" if direction == "eeg_to_fnirs" else "fNIRS→EEG",
            )
    axes[1, 1].axhline(0.0, color="#666666", ls="--", lw=0.8)
    axes[1, 1].set_xticks(
        range(len(relations)),
        [relation_labels[r] for r in relations],
        rotation=27,
        ha="right",
    )
    axes[1, 1].set_ylabel("positive − hardest negative cosine")
    axes[1, 1].set_title("Hardest-negative margin")
    axes[1, 1].legend(fontsize=8)
    figure.suptitle(f"EFRM alignment | {short_label}", fontsize=13)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _alt_text(
    run_id: str,
    duplicate: Mapping[str, Any],
    metric_rows: Sequence[Mapping[str, Any]],
    permutation_rows: Sequence[Mapping[str, Any]],
) -> str:
    unique = next((row for row in metric_rows if row.get("view") == "unique_sample" and row.get("direction") == "eeg_to_fnirs" and row.get("relation") == "exact_pair"), None)
    duplicate_rate = float(duplicate.get("false_negative_pair_rate_if_diagonal_only", 0.0))
    lines = [
        f"# Alt text: EFRM hierarchical alignment — {run_id}",
        "",
        "The figure has four panels. Top-left shows macro-query AUC for the exact synchronized pair against all negatives and four hierarchical negative relations, with EEG-to-fNIRS and fNIRS-to-EEG bars. Top-right shows the corresponding relation-restricted MRR. Bottom-left compares observed EEG-to-fNIRS MRR with subject-block and record-block permutation null means. Bottom-right shows positive-minus-hardest-negative cosine margins by relation.",
        "",
        f"The exported validation contains {duplicate['row_count']} rows but only {duplicate['unique_sample_count']} unique sample IDs; {duplicate['duplicate_row_count_excess']} rows are repeated. If the diagonal alone were treated as positive, the off-diagonal duplicate-positive pair rate would be {duplicate_rate:.6f}.",
    ]
    if unique:
        lines.append(
        f"After stable sample-ID de-duplication, exact-pair EEG-to-fNIRS AUC against all non-positive candidates is {float(unique['auc']):.4f}, MRR is {float(unique['mrr']):.4f}, and the hardest-negative margin is {float(unique['hardest_margin']):.4f}."
        )
    lines.append("The relation pools can overlap; they are mechanism diagnostics rather than a mutually exclusive partition. Block permutation p-values are exploratory and are computed on the unique-sample view.")
    return "\n".join(lines) + "\n"


def analyze_one(
    evidence: Evidence,
    output_dir: Path,
    *,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = list(evidence.metadata)
    duplicate = duplicate_summary(
        metadata, evidence.eeg_embeddings, evidence.fnirs_embeddings
    )
    unique_indices = stable_unique_indices(metadata)
    unique_metadata = [metadata[int(index)] for index in unique_indices.tolist()]
    unique_cosine = evidence.cosine[np.ix_(unique_indices, unique_indices)]
    geometry_rows: list[dict[str, Any]] = []
    for view, indices in (("unique_sample", unique_indices), ("raw_duplicate_aware", np.arange(len(metadata), dtype=np.int64))):
        for modality, embeddings in (("eeg", evidence.eeg_embeddings), ("fnirs", evidence.fnirs_embeddings)):
            geometry = embedding_geometry(embeddings[indices])
            geometry.update({"run_id": evidence.run_id, "view": view, "modality": modality})
            geometry_rows.append(geometry)
    raw_positive_groups = _positive_groups(metadata, metadata)
    unique_positive_groups = _positive_groups(unique_metadata, unique_metadata)
    metric_rows: list[dict[str, Any]] = []
    for view, matrix, view_metadata, positive_groups in (
        ("unique_sample", unique_cosine, unique_metadata, unique_positive_groups),
        ("raw_duplicate_aware", evidence.cosine, metadata, raw_positive_groups),
    ):
        for direction, directional_matrix in (("eeg_to_fnirs", matrix), ("fnirs_to_eeg", matrix.T)):
            for relation in RELATIONS:
                result = relation_metrics(
                    directional_matrix,
                    view_metadata,
                    view_metadata,
                    positive_groups,
                    relation,
                )
                result.update({"run_id": evidence.run_id, "view": view, "direction": direction})
                metric_rows.append(result)
    permutation_rows = block_permutation_metrics(
        unique_cosine,
        unique_metadata,
        unique_positive_groups,
        permutations=permutations,
        seed=seed,
    )
    for row in permutation_rows:
        row["run_id"] = evidence.run_id
        row["view"] = "unique_sample"
    _csv_write(output_dir / "hierarchical_metrics.csv", metric_rows)
    _csv_write(output_dir / "geometry.csv", geometry_rows)
    _csv_write(output_dir / "block_permutation.csv", permutation_rows)
    metrics_json = {
        "schema": ANALYSIS_SCHEMA,
        "run_id": evidence.run_id,
        "evidence": {
            "path": str(evidence.path),
            "sha256": _sha256(evidence.path),
            "manifest_path": evidence.manifest.get("manifest_path"),
            "manifest_run_id": evidence.manifest.get("run_id"),
            "protected_test_opened": evidence.manifest.get("protected_test_opened"),
            "status": evidence.manifest.get("status"),
            "excluded_target_dataset": evidence.manifest.get("excluded_target_dataset"),
        },
        "definitions": {
            "positive": "all candidates with matching sample_id; unique_sample retains first occurrence only",
            "exact_pair": "synchronized sample_id positive support compared with every non-positive candidate",
            "same_subject_wrong_time": "same dataset, subject, and join_key; different sample_id",
            "same_dataset_different_subject": "same dataset_id; different subject",
            "cross_dataset": "different dataset_id",
            "same_class_wrong_instance": "same dataset/task/condition; different join_key",
            "metric_scope": "relation pools are independently evaluated and may overlap; AUC/MRR are macro-query",
            "rank_one_warning_threshold": "effective_rank < 2.0 AND first_axis_energy_fraction > 0.95; descriptive diagnostic, not a hypothesis test",
            "permutation": "within-block candidate permutation on stable unique sample IDs; preserves dataset-subject or dataset-subject-record blocks",
        },
        "duplicate_summary": duplicate,
        "geometry": geometry_rows,
        "row_count": len(metadata),
        "unique_sample_count": len(unique_metadata),
        "permutation_config": {"permutations": int(permutations), "seed": int(seed)},
        "metrics": metric_rows,
        "block_permutation": permutation_rows,
    }
    (output_dir / "hierarchical_metrics.json").write_text(
        json.dumps(_jsonable(metrics_json), indent=2, sort_keys=True), encoding="utf-8"
    )
    _plot_report(evidence.run_id, metric_rows, permutation_rows, output_dir / "hierarchical_alignment")
    (output_dir / "alt_text.md").write_text(
        _alt_text(evidence.run_id, duplicate, metric_rows, permutation_rows), encoding="utf-8"
    )
    output_files = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name not in {"manifest.json", "hierarchical_metrics.json"}
    )
    run_output_manifest = {
        "schema": f"{ANALYSIS_SCHEMA}_files_v1",
        "run_id": evidence.run_id,
        "files": [
            {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in output_files
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(_jsonable(run_output_manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics_json["output_manifest"] = run_output_manifest
    # Keep the JSON self-contained while retaining a hash for every generated
    # non-JSON artifact.  The JSON and manifest deliberately do not hash each
    # other, avoiding a self-referential provenance cycle.
    (output_dir / "hierarchical_metrics.json").write_text(
        json.dumps(_jsonable(metrics_json), indent=2, sort_keys=True), encoding="utf-8"
    )
    return metrics_json


def _failure_report(output_dir: Path, error: Exception, inputs: Sequence[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": ANALYSIS_SCHEMA,
        "status": "failed_closed",
        "error_type": type(error).__name__,
        "error": str(error),
        "inputs": list(inputs),
        "next_action": "Provide a completed full_validation_clip_alignment_evidence.npz with the required fields and a sibling manifest.json certifying protected_test_opened=false; then rerun this script.",
    }
    (output_dir / "failure.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output_dir / "failure.md").write_text(
        "# EFRM hierarchical analysis failed closed\n\n"
        f"- Error: `{type(error).__name__}: {error}`\n"
        f"- Inputs: {', '.join(inputs)}\n"
        "- No metric, plot, or inferred result was generated.\n"
        "- Next action: export a complete full-validation evidence NPZ with metadata_json fields "
        "condition, crop_start_s, dataset_id, join_key, record_id, sample_id, subject, "
        "task_namespace, and a completed manifest certifying protected_test_opened=false.\n",
        encoding="utf-8",
    )


def refresh_existing_manifests(output_root: str | Path) -> None:
    """Refresh output hashes after an approved plot-only re-render.

    This is intentionally separate from metric computation: it never opens
    evidence and only re-hashes files already inside the analysis output root.
    """

    root = Path(output_root)
    manifest_path = root / "manifest.json"
    root_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in root_manifest.get("inputs", []):
        run_dir = root / str(item["output_subdir"])
        run_manifest_path = run_dir / "manifest.json"
        metrics_path = run_dir / "hierarchical_metrics.json"
        if metrics_path.is_file():
            metrics_json = json.loads(metrics_path.read_text(encoding="utf-8"))
            metric_files = sorted(
                path
                for path in run_dir.iterdir()
                if path.is_file() and path.name not in {"manifest.json", "hierarchical_metrics.json"}
            )
            metrics_json["output_manifest"]["files"] = [
                {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
                for path in metric_files
            ]
            metrics_path.write_text(
                json.dumps(_jsonable(metrics_json), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        if run_manifest_path.is_file():
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            run_files = sorted(
                path
                for path in run_dir.iterdir()
                if path.is_file() and path.name not in {"manifest.json", "hierarchical_metrics.json"}
            )
            run_manifest["files"] = [
                {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
                for path in run_files
            ]
            run_manifest_path.write_text(
                json.dumps(_jsonable(run_manifest), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        output_files = sorted(path for path in run_dir.iterdir() if path.is_file())
        item["output_files"] = [
            {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in output_files
        ]
    root_manifest["root_outputs"] = [
        {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "manifest.json"
    ]
    manifest_path.write_text(
        json.dumps(_jsonable(root_manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", action="append", required=True, help="full_validation_clip_alignment_evidence.npz; repeat twice for the two Stage-A runs")
    parser.add_argument("--output-dir", required=True, help="root output directory")
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    output_root = Path(args.output_dir).resolve()
    if args.permutations < 1:
        error = EvidenceError("--permutations must be >= 1")
        _failure_report(output_root, error, args.evidence)
        return 2
    try:
        evidences = [load_evidence(path) for path in args.evidence]
        if not evidences:
            raise EvidenceError("at least one evidence file is required")
        root_manifest: dict[str, Any] = {
            "schema": ANALYSIS_SCHEMA,
            "status": "completed",
            "analysis": "duplicate-aware hierarchical alignment",
            "inputs": [],
            "permutations": int(args.permutations),
            "seed": int(args.seed),
            "views": ["unique_sample", "raw_duplicate_aware"],
            "relations": list(RELATIONS),
            "block_types": list(BLOCK_TYPES),
        }
        for offset, evidence in enumerate(evidences):
            run_name = _safe_name(evidence.run_id)
            if run_name in {str(item.get("run_id")) for item in root_manifest["inputs"]}:
                run_name = f"{run_name}_{offset + 1}"
            result = analyze_one(
                evidence,
                output_root / run_name,
                permutations=args.permutations,
                seed=args.seed + offset,
            )
            run_output_dir = output_root / run_name
            output_files = sorted(path for path in run_output_dir.iterdir() if path.is_file())
            root_manifest["inputs"].append({
                "run_id": evidence.run_id,
                "output_subdir": run_name,
                "evidence_path": str(evidence.path),
                "evidence_sha256": _sha256(evidence.path),
                "row_count": result["row_count"],
                "unique_sample_count": result["unique_sample_count"],
                "duplicate_summary": result["duplicate_summary"],
                "output_files": [
                    {
                        "path": path.name,
                        "sha256": _sha256(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in output_files
                ],
            })
        root_manifest["input_count"] = len(root_manifest["inputs"])
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "alt_text.md").write_text(
            "# Alt text: EFRM hierarchical alignment batch\n\n"
            "This directory contains one duplicate-aware analysis per Stage-A full public-validation artifact. "
            "Each run subdirectory has CSV/JSON metrics, PNG/PDF plots, and a run-specific alt-text description. "
            "The primary view de-duplicates by stable sample_id; the sensitivity view retains all rows while excluding duplicate sample IDs from negative pools.\n",
            encoding="utf-8",
        )
        summary_lines = [
            "# EFRM hierarchical alignment analysis",
            "",
            "This is a public-validation, duplicate-aware diagnostic. The primary view uses stable first-occurrence sample_id de-duplication; raw_duplicate_aware is a sensitivity view that excludes repeated sample IDs from negative pools.",
            "",
        ]
        for item in root_manifest["inputs"]:
            result_path = output_root / str(item["output_subdir"]) / "hierarchical_metrics.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            dup = result["duplicate_summary"]
            exact = next(
                row for row in result["metrics"]
                if row["view"] == "unique_sample" and row["direction"] == "eeg_to_fnirs" and row["relation"] == "exact_pair"
            )
            geometry = [
                row for row in result["geometry"]
                if row["view"] == "unique_sample"
            ]
            summary_lines.extend([
                f"## {item['run_id']}",
                "",
                f"- Rows: {dup['row_count']}; unique sample IDs: {dup['unique_sample_count']}; duplicate-row excess: {dup['duplicate_row_count_excess']}",
                f"- Diagonal-only false-negative pair rate: {dup['false_negative_pair_rate_if_diagonal_only']:.6f}",
                f"- Exact-pair EEG→fNIRS AUC / MRR / Recall@1 / Recall@5: {exact['auc']:.4f} / {exact['mrr']:.6f} / {exact['recall_at_1']:.6f} / {exact['recall_at_5']:.6f}",
                *[
                    f"- {row['modality']} unique-sample effective rank / first-axis fraction / rank-one warning: {row['effective_rank']:.3f} / {row['first_axis_energy_fraction']:.4f} / {row['rank_one_warning']}"
                    for row in geometry
                ],
                "",
            ])
        summary_lines.extend([
            "Block-permutation p-values are exploratory and use the unique-sample view. Relation pools may overlap; see each run's JSON definitions for exact negative-pool rules.",
            "",
        ])
        (output_root / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
        root_manifest["root_outputs"] = [
            {
                "path": path.name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(output_root.iterdir())
            if path.is_file() and path.name != "manifest.json"
        ]
        (output_root / "manifest.json").write_text(
            json.dumps(_jsonable(root_manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0
    except Exception as error:
        _failure_report(output_root, error, args.evidence)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run_cli())
