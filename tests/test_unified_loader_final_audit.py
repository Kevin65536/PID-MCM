from types import SimpleNamespace

import numpy as np

from experiments.audit_unified_loader_final import RunningSignalStats, semantic_sample_key


def test_running_signal_stats_respects_time_mask_and_reports_exact_moments():
    stats = RunningSignalStats(quantile_points_per_window=8)
    signal = np.asarray([[1.0, 2.0, 100.0], [3.0, 4.0, 200.0]], dtype=np.float32)

    stats.update(signal, np.asarray([True, True, False]))
    summary = stats.to_dict()

    assert summary["finite_fraction"] == 1.0
    assert summary["mean"] == 2.5
    assert summary["variance"] == 1.25
    assert summary["std"] == np.sqrt(1.25)
    assert summary["max"] == 4.0


def test_running_signal_stats_merge_matches_pooled_exact_moments():
    left = RunningSignalStats(quantile_points_per_window=8)
    right = RunningSignalStats(quantile_points_per_window=8)
    pooled = RunningSignalStats(quantile_points_per_window=8)
    mask = np.asarray([True, True])

    left.update(np.asarray([[1.0, 2.0]], dtype=np.float32), mask)
    right.update(np.asarray([[3.0, 6.0]], dtype=np.float32), mask)
    pooled.update(np.asarray([[1.0, 2.0, 3.0, 6.0]], dtype=np.float32), np.ones(4, dtype=bool))
    left.merge(right)

    merged = left.to_dict()
    expected = pooled.to_dict()
    for key in ("value_count", "finite_value_count"):
        assert merged[key] == expected[key]
    for key in ("mean", "variance", "std", "min", "max"):
        assert np.isclose(merged[key], expected[key])


def test_semantic_sample_key_collapses_visual_probes_but_not_other_records():
    label = {"condition": "RR"}
    visual_one = SimpleNamespace(
        record=SimpleNamespace(
            dataset_id="visual_cognitive_motivation",
            canonical_subject_id="S01",
            base_record_id="S01_Part1_Probe1",
        ),
        event={"event_index": 0, "metadata": {"epoch_id": 1}},
    )
    visual_two = SimpleNamespace(
        record=SimpleNamespace(
            dataset_id="visual_cognitive_motivation",
            canonical_subject_id="S01",
            base_record_id="S01_Part1_Probe2",
        ),
        event={"event_index": 0, "metadata": {"epoch_id": 1}},
    )
    simultaneous = SimpleNamespace(
        record=SimpleNamespace(
            dataset_id="simultaneous_eeg_nirs",
            canonical_subject_id="VP001",
            base_record_id="cnt_wg",
        ),
        event={"event_index": 0, "metadata": {}},
    )

    assert semantic_sample_key(visual_one, label) == semantic_sample_key(visual_two, label)
    assert semantic_sample_key(visual_one, label) != semantic_sample_key(simultaneous, label)
