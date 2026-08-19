from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.evaluate_ssm_reconstruction_reliability import (
    _visual_pair_id,
    aggregate_metrics,
    aggregate_timecourses,
    assign_group_folds,
    build_jobs,
)


def _window_row(subject: str, group: str, unit: str, value: float) -> dict[str, object]:
    return {
        "task_id": "task",
        "family": "family",
        "stage": "core",
        "dataset_id": "dataset",
        "role": "development_validation",
        "subject": subject,
        "stratum": "one",
        "dependency_group": group,
        "unit_id": unit,
        "fold_index": 0,
        "model": "adaptive_eeg_only",
        "spatial_mode": "local",
        "selected_fnirs_channels": "hbo|hbr",
        "selected_eeg_channels": "eeg",
        "hbo_trajectory_deviation_nrmse": value,
    }


def test_group_folds_are_deterministic_balanced_and_never_split_groups():
    groups = ["g1", "g1", "g2", "g3", "g4", "g5", "g6"]
    first = assign_group_folds(groups, 5, seed=17)
    second = assign_group_folds(list(reversed(groups)), 5, seed=17)

    assert first == second
    assert set(first) == set(groups)
    counts = np.bincount(list(first.values()))
    assert int(np.max(counts) - np.min(counts)) <= 1


def test_visual_probe_pair_uses_shared_part_and_epoch_identity():
    assert _visual_pair_id("S01_Part1_Probe1", 7) == _visual_pair_id(
        "S01_Part1_Probe2", 7
    )
    assert _visual_pair_id("S01_Part1_Probe1", 7) != _visual_pair_id(
        "S01_Part2_Probe1", 7
    )


def test_aggregation_is_dependency_equal_then_subject_equal_and_deterministic():
    rows = [
        _window_row("s1", "g1", "u1", 1.0),
        _window_row("s1", "g1", "u2", 3.0),
        _window_row("s1", "g2", "u3", 8.0),
        _window_row("s2", "g3", "u4", 20.0),
    ]
    first = aggregate_metrics(rows, bootstrap_iterations=100, seed=9)
    second = aggregate_metrics(rows, bootstrap_iterations=100, seed=9)
    dependency, subjects, summary, draws, keys = first

    assert [row["hbo_trajectory_deviation_nrmse"] for row in dependency] == [2.0, 8.0, 20.0]
    assert [row["hbo_trajectory_deviation_nrmse"] for row in subjects] == [5.0, 20.0]
    assert summary[0]["hbo_trajectory_deviation_nrmse"] == pytest.approx(12.5)
    np.testing.assert_array_equal(draws, second[3])
    assert keys == second[4]


def test_timecourse_bootstrap_keeps_subject_as_replication_unit():
    rows = []
    for subject, offset in (("s1", 0.0), ("s2", 2.0)):
        for time in (0.0, 0.1):
            rows.append(
                {
                    "task_id": "task",
                    "family": "family",
                    "stage": "core",
                    "dataset_id": "dataset",
                    "role": "development_validation",
                    "subject": subject,
                    "model": "adaptive_eeg_only",
                    "spatial_mode": "local",
                    "modality": "hbo",
                    "time_s": time,
                    "observed": offset + time,
                    "reconstructed": offset + time + 0.5,
                    "posterior_predictive_sd": 0.25,
                }
            )
    result = aggregate_timecourses(rows, bootstrap_iterations=100, seed=3)
    assert [row["observed_mean"] for row in result] == pytest.approx([1.0, 1.1])
    assert all(row["subjects"] == 2 for row in result)


def test_full_job_matrix_never_contains_protected_or_unused_subjects():
    path = Path(
        "experiments/configs/physiology_semantic_tokenizer/ssm_reconstruction_reliability.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    jobs = build_jobs(config, smoke=False)
    forbidden = {
        value
        for values in config["data"]["protected_or_unused"].values()
        for value in values
    }
    assert len(jobs) == 279
    assert not {job.subject for job in jobs}.intersection(forbidden)
