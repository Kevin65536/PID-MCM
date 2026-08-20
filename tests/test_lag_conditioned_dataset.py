from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.data.lag_conditioned_dataset import (
    CANONICAL_LC_CACHE_ROOT,
    CANONICAL_PROTECTED_SUBJECTS,
    REVIEWED_EEG_SIGNAL_BRANCH,
    TASK_SPECS,
    LagConditionedSampleIndex,
    LagConditionedTaskDataset,
    build_admitted_index,
    get_task_spec,
    make_group_derangement,
)
from src.data.unified_physiology import (
    DEFAULT_ADMISSIBLE_ALIGNMENT_CASES,
    UnifiedPhysiologyWindowDataset,
)


def _event(index: int, condition: str) -> dict:
    return {
        "event_index": index,
        "onset_ms": float(index * 30_000),
        "event_type": "motor_imagery",
        "label": condition,
        "label_index": 0 if condition == "LMI" else 1,
        "metadata": {
            "task": "motor_imagery",
            "condition_label": condition,
            "event_role": "trial",
        },
    }


def _ref(subject: str, event_index: int, condition: str):
    return SimpleNamespace(
        record=SimpleNamespace(
            canonical_subject_id=subject,
            base_record_id="session_00",
        ),
        event=_event(event_index, condition),
    )


def _sample(subject: str, condition: str, event_index: int) -> dict:
    return {
        "eeg": np.ones((3, 4000), dtype=np.float32) * (event_index + 1),
        "fnirs": np.ones((4, 200), dtype=np.float32) * (event_index + 1),
        "valid_mask": {
            "eeg": np.ones(4000, dtype=bool),
            "fnirs": np.ones(200, dtype=bool),
        },
        "analysis_valid_mask": {
            "eeg": np.ones(4000, dtype=bool),
            "fnirs": np.ones(200, dtype=bool),
        },
        "bad_channel_mask": {
            "eeg": np.zeros(3, dtype=bool),
            "fnirs": np.zeros(4, dtype=bool),
        },
        "channel_names": {
            "eeg": ["e0", "e1", "e2"],
            "fnirs": ["a_HbO", "a_HbR", "b_HbO", "b_HbR"],
        },
        "component_roles": {
            "eeg": ["electrical_potential"] * 3,
            "fnirs": ["HbO", "HbR", "HbO", "HbR"],
        },
        "dataset_id": "eeg_fnirs_single_trial",
        "eeg_signal_branch": REVIEWED_EEG_SIGNAL_BRANCH,
        "sample_rate_hz": {"eeg": 200.0, "fnirs": 10.0},
        "subject": subject,
        "record_id": "session_00",
        "label": {
            "condition": condition,
            "namespace": "eeg_fnirs_single_trial:motor_imagery",
        },
        "event": _event(event_index, condition),
        "alignment": {
            "eeg_time_ms": float(event_index * 30_000 - 5_000),
            "fnirs_time_ms": float(event_index * 30_000 - 5_000),
        },
    }


class _FakeBase(UnifiedPhysiologyWindowDataset):
    def __init__(self):
        self.project_root = CANONICAL_LC_CACHE_ROOT.parents[2]
        self.cache_root = CANONICAL_LC_CACHE_ROOT
        self.eeg_artifact_cache_root = CANONICAL_LC_CACHE_ROOT / "eeg_artifact_clean_v4"
        self.simultaneous_eeg_cache_root = (
            CANONICAL_LC_CACHE_ROOT / "simultaneous_eeg_eog_clean_v1"
        )
        self.eeg_artifact_config = None
        self.dataset_ids = ("eeg_fnirs_single_trial",)
        self.window_duration_s = 20.0
        self.window_offset_s = -5.0
        self.eeg_signal_branch = REVIEWED_EEG_SIGNAL_BRANCH
        self.require_eeg_artifact_cache = True
        self.require_paired_timestamps = True
        self.admissible_alignment_cases = DEFAULT_ADMISSIBLE_ALIGNMENT_CASES
        self.include_event_types = None
        self.windows = []
        self.samples = []
        self.accessed: list[int] = []
        for subject in ("subject_01", "subject_24"):
            for condition in ("LMI", "RMI"):
                for repeat in range(2):
                    event_index = len(self.windows)
                    self.windows.append(_ref(subject, event_index, condition))
                    self.samples.append(_sample(subject, condition, event_index))

    def __getitem__(self, index: int):
        self.accessed.append(int(index))
        return self.samples[int(index)]


def test_task_specs_use_twenty_second_ten_token_contract():
    for task in ("motor_imagery", "mental_arithmetic", "word_generation", "n_back"):
        spec = get_task_spec(task)
        assert spec.window_duration_s == 20.0
        assert spec.window_offset_s == -5.0
        assert spec.num_tokens == 10
        assert spec.eeg_patch_samples == 400
        assert spec.fnirs_patch_samples == 20


def test_task_specs_and_measured_source_contract_are_immutable(tmp_path):
    with pytest.raises(TypeError):
        TASK_SPECS["motor_imagery"] = TASK_SPECS["motor_imagery"]  # type: ignore[index]
    with pytest.raises(PermissionError, match="canonical cache"):
        LagConditionedTaskDataset(
            task_id="motor_imagery",
            admitted_subjects=("subject_01",),
            cache_root=tmp_path,
            base_dataset=_FakeBase(),
        )
    base = _FakeBase()
    base.window_duration_s = 19.0
    with pytest.raises(PermissionError, match="window_duration_s"):
        LagConditionedTaskDataset(
            task_id="motor_imagery",
            admitted_subjects=("subject_01",),
            base_dataset=base,
        )
    relative = _FakeBase()
    relative.cache_root = Path("data/cache/physiology_semantic_clean_v1")
    with pytest.raises(PermissionError, match="cache_root_absolute"):
        LagConditionedTaskDataset(
            task_id="motor_imagery",
            admitted_subjects=("subject_01",),
            base_dataset=relative,
        )
    alternate = _FakeBase()
    alternate.eeg_artifact_cache_root = tmp_path / "alternate-eeg"
    with pytest.raises(PermissionError, match="eeg_artifact_cache_root"):
        LagConditionedTaskDataset(
            task_id="motor_imagery",
            admitted_subjects=("subject_01",),
            base_dataset=alternate,
        )


def test_loaded_sample_event_timing_must_match_metadata_index():
    branch_drift = _FakeBase()
    branch_drift.samples[0]["eeg_signal_branch"] = "unreviewed"
    branch_dataset = LagConditionedTaskDataset(
        task_id="motor_imagery",
        admitted_subjects=("subject_01",),
        base_dataset=branch_drift,
    )
    with pytest.raises(RuntimeError, match="branch identity drift"):
        branch_dataset[0]
    base = _FakeBase()
    base.samples[0]["alignment"]["eeg_time_ms"] += 1.0
    dataset = LagConditionedTaskDataset(
        task_id="motor_imagery",
        admitted_subjects=("subject_01",),
        base_dataset=base,
    )
    with pytest.raises(RuntimeError, match="event timing drift"):
        dataset[0]


def test_metadata_filter_prevents_forbidden_measured_access():
    base = _FakeBase()
    dataset = LagConditionedTaskDataset(
        task_id="motor_imagery",
        admitted_subjects=("subject_01",),
        forbidden_subjects=CANONICAL_PROTECTED_SUBJECTS["eeg_fnirs_single_trial"],
        base_dataset=base,
    )

    assert len(dataset) == 4
    assert all(row.subject == "subject_01" for row in dataset.rows)
    item = dataset[0]
    assert item["eeg"].shape == (3, 4000)
    assert item["fnirs"].shape == (4, 200)
    assert item["eeg_token_valid_mask"].shape == (10,)
    assert item["fnirs_token_valid_mask"].shape == (10,)
    assert all(base.windows[index].record.canonical_subject_id == "subject_01" for index in base.accessed)
    assert dataset.contract_summary()["forbidden_measured_access"] == 0


def test_admitted_forbidden_overlap_fails_closed_before_access():
    base = _FakeBase()
    with pytest.raises(PermissionError, match="overlap"):
        LagConditionedTaskDataset(
            task_id="motor_imagery",
            admitted_subjects=("subject_24",),
            forbidden_subjects=CANONICAL_PROTECTED_SUBJECTS["eeg_fnirs_single_trial"],
            base_dataset=base,
        )
    assert base.accessed == []


def test_default_dataset_boundary_blocks_canonical_protected_subject():
    base = _FakeBase()
    with pytest.raises(PermissionError, match="overlap"):
        LagConditionedTaskDataset(
            task_id="motor_imagery",
            admitted_subjects=("subject_24",),
            base_dataset=base,
        )
    assert base.accessed == []


def test_derangement_is_stable_nonidentity_and_group_preserving():
    rows = tuple(
        LagConditionedSampleIndex(
            base_index=index,
            sample_id=f"sample-{index}",
            dataset_id="eeg_fnirs_single_trial",
            task_id="motor_imagery",
            subject="s1" if index < 4 else "s2",
            record_id="r",
            condition="a" if index % 4 < 2 else "b",
            class_index=0,
            event_index=index,
            event_time_ms=float(index * 30_000),
            fnirs_event_time_ms=float(index * 30_000),
        )
        for index in range(8)
    )
    first = make_group_derangement(rows, seed=17)
    second = make_group_derangement(rows, seed=17)

    np.testing.assert_array_equal(first, second)
    assert np.all(first != np.arange(len(rows)))
    for target, donor in enumerate(first):
        assert rows[target].subject == rows[int(donor)].subject
        assert rows[target].condition == rows[int(donor)].condition


def test_derangement_rejects_overlapping_window_donors():
    rows = tuple(
        LagConditionedSampleIndex(
            base_index=index,
            sample_id=f"overlap-{index}",
            dataset_id="eeg_fnirs_single_trial",
            task_id="motor_imagery",
            subject="subject_01",
            record_id="session_00",
            condition="LMI",
            class_index=0,
            event_index=index,
            event_time_ms=float(index * 30_000),
            fnirs_event_time_ms=float(index * 10_000),
        )
        for index in range(2)
    )
    with pytest.raises(ValueError, match="nonoverlapping donor"):
        make_group_derangement(rows, seed=1)


def test_dataset_can_return_same_group_donor_arrays():
    base = _FakeBase()
    dataset = LagConditionedTaskDataset(
        task_id="motor_imagery",
        admitted_subjects=("subject_01",),
        forbidden_subjects=CANONICAL_PROTECTED_SUBJECTS["eeg_fnirs_single_trial"],
        base_dataset=base,
    )
    dataset.set_derangement(seed=3)
    item = dataset[0]

    donor = int(item["donor_index"])
    assert donor != 0
    assert dataset.rows[donor].subject == dataset.rows[0].subject
    assert dataset.rows[donor].condition == dataset.rows[0].condition
    assert item["donor_fnirs"].shape == item["fnirs"].shape
    assert item["donor_fnirs_token_valid_mask"].shape == (10,)


def test_build_index_rejects_missing_class_support():
    spec = get_task_spec("motor_imagery")
    windows = [_ref("subject_01", index, "LMI") for index in range(2)]
    with pytest.raises(RuntimeError, match="class support"):
        build_admitted_index(
            windows,
            spec,
            admitted_subjects=("subject_01",),
            forbidden_subjects=CANONICAL_PROTECTED_SUBJECTS["eeg_fnirs_single_trial"],
        )
