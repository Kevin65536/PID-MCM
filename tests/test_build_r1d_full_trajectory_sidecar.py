import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.scripts.build_r1d_full_trajectory_sidecar import (
    EEG_ONLY_MODEL,
    JOINT_MODEL,
    FoldKey,
    MeasuredEvent,
    build_r1d_bundle,
    reshape_full_trajectory,
)
from src.data.shared_driver_targets import (
    PhysiologyRawViewRegistry,
    SharedDriverTrajectorySidecar,
)


def _trajectory_rows(offset=0.0):
    return [
        {
            "time_s": str(-5.0 + index / 10.0),
            "target_shared_driver": str(offset + index),
            "target_shared_driver_std": "0.25",
            "gauge_shared_driver_scale": "1.0",
            "gauge_shared_driver_offset": "0.0",
        }
        for index in range(200)
    ]


def test_full_trajectory_reshape_preserves_exact_two_second_slices():
    payload = reshape_full_trajectory(
        list(reversed(_trajectory_rows())),
        expected_start_s=-5.0,
        sample_rate_hz=10.0,
    )

    assert payload["target"].shape == (10, 20)
    assert payload["point_mask"].all()
    np.testing.assert_array_equal(payload["target"][3], np.arange(60, 80))
    np.testing.assert_allclose(payload["time_s"][3, [0, -1]], [1.0, 2.9])


class _FakeMeasuredDataset:
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, index):
        return self.samples[index]


def _measured_sample():
    eeg = ["E1", "E2", "E3", "E4", "E5", "E6"]
    fnirs = ["A_HbO", "A_HbR"]
    return {
        "channel_names": {"eeg": eeg, "fnirs": fnirs},
        "bad_channel_mask": {
            "eeg": np.zeros(6, dtype=bool),
            "fnirs": np.zeros(2, dtype=bool),
        },
        "component_roles": {"fnirs": ["HbO", "HbR"]},
        "channel_geometry": {
            "fnirs": [
                {"channel_name": name, "base_channel_name": "A"}
                for name in fnirs
            ]
        },
    }


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_source(root: Path):
    source = root / "source"
    base = source / "base_model"
    base.mkdir(parents=True)
    subjects = ["subject_01", "subject_19"]
    config = {
        "data": {
            "cache_root": "unused-in-injected-test",
            "window_duration_s": 20.0,
            "window_offset_s": -5.0,
            "conditions": [
                {
                    "condition_id": "synthetic",
                    "dataset_id": "eeg_fnirs_single_trial",
                    "subjects": subjects,
                    "record_id": "session_01",
                    "target_label": "MA",
                    "eeg_signal_branch": "single_trial_eeg_artifact_clean_v4",
                    "max_trials_per_subject": 1,
                }
            ],
        },
        "analysis": {"ssm": {"fs_hz": 10.0}},
    }
    (base / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    trajectory_rows = []
    fit_rows = []
    for subject_index, subject in enumerate(subjects):
        joint_offset = 1000.0 * subject_index
        for model, model_offset in (
            (JOINT_MODEL, joint_offset),
            (EEG_ONLY_MODEL, joint_offset - 10.0),
        ):
            for row in _trajectory_rows(model_offset):
                trajectory_rows.append(
                    {
                        "condition_id": "synthetic",
                        "subject": subject,
                        "heldout_trial": "0",
                        "model": model,
                        "spatial_mode": "local",
                        **row,
                    }
                )
        fit_rows.append(
            {
                "condition_id": "synthetic",
                "subject": subject,
                "heldout_trial": "0",
                "spatial_mode": "local",
                "selected_fnirs_channels": "A_HbO|A_HbR",
                "selected_eeg_channels": "E1|E2|E3|E4|E5|E6",
                "optimizer_success": "True",
            }
        )
    _write_csv(base / "trajectories.csv", trajectory_rows)
    _write_csv(base / "fit_parameters.csv", fit_rows)

    split = {
        "data": {
            "split": {
                "train_subject_keys": [
                    "eeg_fnirs_single_trial|subject_01",
                ],
                "val_subject_keys": [
                    "eeg_fnirs_single_trial|subject_19",
                ],
                "test_subject_keys": [
                    f"eeg_fnirs_single_trial|subject_{index:02d}"
                    for index in range(24, 30)
                ],
            }
        }
    }
    split_path = root / "split.yaml"
    split_path.write_text(yaml.safe_dump(split), encoding="utf-8")

    measured = _FakeMeasuredDataset([_measured_sample(), _measured_sample()])
    event_lookup = {
        FoldKey("synthetic", "subject_01", 0): MeasuredEvent(
            "eeg_fnirs_single_trial", "session_01", 10, "MA", measured, 0
        ),
        FoldKey("synthetic", "subject_19", 0): MeasuredEvent(
            "eeg_fnirs_single_trial", "session_01", 11, "MA", measured, 1
        ),
    }
    return source, split_path, event_lookup


def test_builder_separates_targets_and_raw_view_and_uses_train_only_scalar(tmp_path):
    source, split_path, event_lookup = _synthetic_source(tmp_path)
    output = tmp_path / "r1_d_development_v1"

    build_r1d_bundle(
        source,
        split_path,
        output,
        event_lookup_override=event_lookup,
    )

    teacher = SharedDriverTrajectorySidecar(
        output / "trajectory_targets",
        expected_scope="development_crossfit",
        expected_family="adaptive_joint_full_trajectory",
    )
    raw_view = PhysiologyRawViewRegistry(output / "raw_view_registry")
    key = "eeg_fnirs_single_trial|subject_01|session_01|event=10"
    target = teacher.lookup(key)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    expected_scale = float(np.std(np.arange(200, dtype=np.float64)))
    assert len(teacher) == len(raw_view) == 2
    assert manifest["promotion_eligible"] is False
    assert manifest["protected_open"] is False
    assert manifest["normalization"]["mean"] == pytest.approx(99.5)
    assert manifest["normalization"]["scale"] == pytest.approx(expected_scale)
    assert target["joint_correction"].numpy().mean() == pytest.approx(
        10.0 / expected_scale
    )
    assert raw_view.lookup(key)["selected_eeg_channels"] == (
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "E6",
    )
    assert "selected_eeg_channels" not in target
    coverage = output / "data_coverage_by_subject_session_condition_patch.csv"
    assert len(coverage.read_text(encoding="utf-8").splitlines()) == 21

    with pytest.raises(FileExistsError, match="overwrite"):
        build_r1d_bundle(
            source,
            split_path,
            output,
            event_lookup_override=event_lookup,
        )
