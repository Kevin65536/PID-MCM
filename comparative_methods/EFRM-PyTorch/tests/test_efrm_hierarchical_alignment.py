from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from efrm_pytorch.hierarchical_alignment_analysis import (
    EvidenceError,
    _positive_groups,
    block_permutation_metrics,
    duplicate_summary,
    embedding_geometry,
    relation_metrics,
    stable_unique_indices,
)


def _metadata() -> list[dict[str, str | float]]:
    return [
        {
            "condition": "A",
            "crop_start_s": 0.0,
            "dataset_id": "d1",
            "join_key": "d1|s1|r1",
            "record_id": "r1",
            "sample_id": "a",
            "subject": "s1",
            "task_namespace": "d1:task",
        },
        {
            "condition": "A",
            "crop_start_s": 0.0,
            "dataset_id": "d1",
            "join_key": "d1|s1|r1",
            "record_id": "r1",
            "sample_id": "a",
            "subject": "s1",
            "task_namespace": "d1:task",
        },
        {
            "condition": "A",
            "crop_start_s": 1.0,
            "dataset_id": "d1",
            "join_key": "d1|s1|r2",
            "record_id": "r2",
            "sample_id": "b",
            "subject": "s1",
            "task_namespace": "d1:task",
        },
        {
            "condition": "B",
            "crop_start_s": 0.0,
            "dataset_id": "d2",
            "join_key": "d2|s2|r1",
            "record_id": "r1",
            "sample_id": "c",
            "subject": "s2",
            "task_namespace": "d2:task",
        },
    ]


def test_stable_sample_dedup_and_false_negative_audit() -> None:
    metadata = _metadata()
    assert stable_unique_indices(metadata).tolist() == [0, 2, 3]
    summary = duplicate_summary(metadata)
    assert summary["row_count"] == 4
    assert summary["unique_sample_count"] == 3
    assert summary["duplicate_row_count_excess"] == 1
    assert summary["off_diagonal_duplicate_positive_pairs"] == 2
    assert summary["inconsistent_duplicate_group_count"] == 0
    embeddings = np.asarray([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    identity_summary = duplicate_summary(metadata, embeddings, embeddings)
    assert identity_summary["duplicate_eeg_embedding_max_abs_diff"] == 0.0


def test_embedding_geometry_reports_rank_and_axis_concentration() -> None:
    embeddings = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=np.float64,
    )
    result = embedding_geometry(embeddings)
    assert result["sample_count"] == 4
    assert result["embedding_dim"] == 2
    assert 1.0 < result["effective_rank"] <= 2.0
    assert 0.4 <= result["first_axis_energy_fraction"] <= 0.6


def test_relation_metrics_excludes_duplicate_ids_from_negative_pool() -> None:
    metadata = _metadata()
    cosine = np.asarray(
        [
            [1.0, 1.0, 0.2, -0.2],
            [1.0, 1.0, 0.2, -0.2],
            [0.2, 0.2, 1.0, -0.1],
            [-0.2, -0.2, -0.1, 1.0],
        ],
        dtype=np.float64,
    )
    groups = _positive_groups(metadata, metadata)
    result = relation_metrics(cosine, metadata, metadata, groups, "all_negative")
    # Query 0 has candidate 1 as a duplicate positive, not a negative.
    assert result["negative_pair_count"] == 10
    assert result["mrr"] == pytest.approx(1.0)
    assert result["recall_at_1"] == pytest.approx(1.0)
    wrong_time = relation_metrics(
        cosine, metadata, metadata, groups, "same_subject_wrong_time"
    )
    assert wrong_time["eligible_query_count"] == 0


def test_block_permutation_is_reproducible_and_reports_both_blocks() -> None:
    source_metadata = _metadata()
    metadata = [source_metadata[index] for index in (0, 2, 3)]  # three unique samples: a, b, c
    cosine = np.eye(3, dtype=np.float64)
    groups = _positive_groups(metadata, metadata)
    first = block_permutation_metrics(
        cosine, metadata, groups, permutations=8, seed=9
    )
    second = block_permutation_metrics(
        cosine, metadata, groups, permutations=8, seed=9
    )
    assert first == second
    assert {row["block_type"] for row in first} == {
        "subject_block",
        "record_block",
    }
    assert {row["direction"] for row in first} == {
        "eeg_to_fnirs",
        "fnirs_to_eeg",
    }


def test_missing_evidence_error_is_explicit() -> None:
    with pytest.raises(EvidenceError, match="does not exist"):
        # This assertion exercises the fail-closed error class without opening
        # any protected or synthetic test split.
        from efrm_pytorch.hierarchical_alignment_analysis import load_evidence

        load_evidence("/definitely/missing/full_validation_clip_alignment_evidence.npz")
