from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "analyze_e1_tokenizer_health_and_coupling.py"
SPEC = importlib.util.spec_from_file_location("e1_health_coupling", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
from src.tokenizers.physiology_semantic_tokenizer import PhysiologySemanticTokenizer


def test_distribution_metrics_separates_active_and_effective_usage() -> None:
    metrics = MODULE.distribution_metrics(np.asarray([90, 5, 5, 0], dtype=float))
    assert metrics["active_codes"] == 3
    assert metrics["effective_codes"] < metrics["active_codes"]
    assert metrics["top1_mass"] == 0.9
    assert 0.0 < metrics["gini"] < 1.0


def test_pair_statistics_detects_identity_association() -> None:
    left = np.tile(np.arange(8), 40)
    right = left.copy()
    metrics, counts = MODULE.pair_statistics(left, right, left_vocab=8, right_vocab=8)
    assert counts.shape == (8, 8)
    assert metrics["normalized_mi"] > 0.99
    assert metrics["conditional_top1_accuracy"] == 1.0
    assert metrics["conditional_accuracy_delta"] > 0.8


def test_lagged_pairs_positive_lag_means_eeg_leads_fnirs() -> None:
    arrays = {
        "eeg_tokens": np.asarray([[0, 1, 2, 3]]),
        "fnirs_tokens": np.asarray([[9, 0, 1, 2]]),
        "eeg_mask": np.ones((1, 4), dtype=bool),
        "fnirs_mask": np.ones((1, 4), dtype=bool),
        "subject": np.asarray(["s1"]),
        "label": np.asarray([0]),
    }
    left, right, _, _ = MODULE._lagged_pairs(arrays, lag=1)
    np.testing.assert_array_equal(left, np.asarray([0, 1, 2]))
    np.testing.assert_array_equal(right, np.asarray([0, 1, 2]))


def test_within_group_permutation_null_is_deterministic() -> None:
    left = np.tile(np.arange(4), 20)
    right = left.copy()
    subjects = np.repeat(np.asarray(["s1", "s2"]), 40)
    labels = np.tile(np.repeat(np.asarray([0, 1]), 20), 2)
    first = MODULE.within_group_permutation_null(
        left,
        right,
        subjects,
        labels,
        permutations=8,
        seed=17,
        left_vocab=4,
        right_vocab=4,
    )
    second = MODULE.within_group_permutation_null(
        left,
        right,
        subjects,
        labels,
        permutations=8,
        seed=17,
        left_vocab=4,
        right_vocab=4,
    )
    np.testing.assert_allclose(first, second)
    assert float(first.mean()) < 1.0


def test_leave_one_subject_out_uses_only_other_subjects() -> None:
    left = np.tile(np.arange(4), 40)
    right = left.copy()
    subjects = np.repeat(np.asarray(["s1", "s2", "s3", "s4"]), 40)
    result = MODULE.leave_one_subject_out_accuracy(
        left,
        right,
        subjects,
        left_vocab=4,
        right_vocab=4,
    )
    assert result["loso_subjects"] == 4
    assert result["loso_conditional_accuracy"] == 1.0
    assert result["loso_accuracy_delta"] > 0.7


def test_checkpoint_loader_allows_only_registered_legacy_buffers() -> None:
    model = PhysiologySemanticTokenizer(
        eeg_encoder_dim=16,
        fnirs_encoder_dim=16,
        semantic_dim=8,
        eeg_residual_dim=4,
        fnirs_residual_dim=4,
        codebook_size=16,
        history_tokens=2,
    )
    state = model.state_dict()
    for key in (
        "eeg_branch.quantizer.initialized",
        "eeg_branch.quantizer.quantization_strength",
        "fnirs_branch.quantizer.initialized",
        "fnirs_branch.quantizer.quantization_strength",
    ):
        state.pop(key)
    MODULE._load_checkpoint_model_state(model, state)

    invalid = dict(state)
    invalid.pop("eeg_branch.quantizer.codebook")
    with pytest.raises(RuntimeError, match="outside the registered E1 buffer migration"):
        MODULE._load_checkpoint_model_state(model, invalid)
