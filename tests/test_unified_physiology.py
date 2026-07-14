import numpy as np
import hashlib
import json
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.data.eeg_artifact_preprocessing import EEGArtifactCleaningConfig, clean_single_trial_eeg
from src.data.unified_physiology import (
    CANONICAL_EEG_SAMPLE_RATE_HZ,
    CANONICAL_FNIRS_COMPONENTS,
    CANONICAL_FNIRS_SAMPLE_RATE_HZ,
    CANONICAL_UNIT,
    DEFAULT_UNIFIED_WINDOW_DURATION_S,
    NativeEEGRecord,
    UnifiedPhysiologyWindowDataset,
    canonical_fnirs_channel_names,
    canonical_label,
    fnirs_component_roles,
    preprocess_eeg_record,
    preprocess_fnirs_record,
)


def test_unified_loader_default_observation_window_is_twenty_seconds():
    default = inspect.signature(UnifiedPhysiologyWindowDataset).parameters["window_duration_s"].default
    assert default == DEFAULT_UNIFIED_WINDOW_DURATION_S == 20.0


def test_unified_loader_default_eeg_branch_is_admitted_single_trial_v3():
    default = inspect.signature(UnifiedPhysiologyWindowDataset).parameters["eeg_signal_branch"].default
    assert default == "single_trial_eeg_artifact_clean_v3"


def test_fnirs_names_are_unified_to_hbo_hbr_components():
    names = canonical_fnirs_channel_names(["CH1_Oxy", "CH1_Deoxy", "CH2_HbO", "CH2_HbR"])
    assert names == ("CH1_HbO", "CH1_HbR", "CH2_HbO", "CH2_HbR")
    assert set(fnirs_component_roles(names)) == set(CANONICAL_FNIRS_COMPONENTS)


def test_common_preprocessing_unifies_rates_and_robust_units(tmp_path):
    time = np.arange(5000) / 500.0
    eeg_native = np.column_stack((20 + 5 * np.sin(2 * np.pi * 10 * time), -8 + np.cos(2 * np.pi * 6 * time)))
    eeg, eeg_state = preprocess_eeg_record(
        NativeEEGRecord(
            values=eeg_native,
            sample_rate_hz=500.0,
            channel_names=("Fz", "Cz"),
            native_unit="uV",
            source_path=tmp_path / "test.edf",
        )
    )
    assert eeg.shape[0] == int(round(eeg_native.shape[0] * CANONICAL_EEG_SAMPLE_RATE_HZ / 500.0))
    assert eeg_state["canonical_unit"] == CANONICAL_UNIT
    assert np.max(np.abs(np.median(eeg, axis=0))) < 0.05

    fnirs_time = np.arange(1000) / 20.0
    fnirs_native = np.column_stack((3 + np.sin(2 * np.pi * 0.08 * fnirs_time), -2 + 0.5 * np.cos(2 * np.pi * 0.05 * fnirs_time)))
    fnirs, fnirs_state = preprocess_fnirs_record(
        fnirs_native,
        sample_rate_hz=20.0,
        native_contract={"native_unit": "mmol/L"},
    )
    assert fnirs.shape[0] == int(round(fnirs_native.shape[0] * CANONICAL_FNIRS_SAMPLE_RATE_HZ / 20.0))
    assert fnirs_state["canonical_unit"] == CANONICAL_UNIT
    assert np.max(np.abs(np.median(fnirs, axis=0))) < 0.05


def test_visual_label_separates_condition_from_event_role():
    label = canonical_label(
        {
            "event_type": "trial",
            "label": "RR",
            "label_index": 0,
            "metadata": {
                "task": "visual_cognitive_motivation",
                "event_role": "stimulus_onset",
                "epoch_type": "RR",
            },
        },
        "visual_cognitive_motivation",
    )
    assert label["schema"] == "canonical_task_label_v1"
    assert label["condition"] == "RR"
    assert label["class_index"] == 0
    assert label["event_role"] == "stimulus_onset"


def test_alignment_admission_filter_excludes_unstable_records(tmp_path):
    cache = tmp_path / "cache"
    (cache / "event_index").mkdir(parents=True)
    record = {
        "dataset_id": "refed",
        "subject": "1",
        "record_id": "video_1_hbo_hbr",
        "sample_rate_hz": 10.0,
        "record_npz": str(cache / "unused.npz"),
        "metadata": {},
    }
    (cache / "cache_manifest.json").write_text(json.dumps({"records": [record]}), encoding="utf-8")
    (cache / "event_index/event_manifest.json").write_text("{}", encoding="utf-8")
    event = {
        "dataset_id": "refed",
        "subject": "1",
        "record_id": "video_1",
        "event_type": "trial",
        "label": "x",
        "eeg_time_ms": 0.0,
        "fnirs_time_ms": 0.0,
        "onset_ms": 0.0,
    }
    report = {
        "dataset_id": "refed",
        "subject": "1",
        "record_id": "video_1",
        "alignment_case": "continuous_drift",
        "label_sequence_match": True,
    }
    (cache / "event_index/events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    (cache / "event_index/alignment_reports.jsonl").write_text(json.dumps(report) + "\n", encoding="utf-8")

    from src.data.unified_physiology import UnifiedPhysiologyWindowDataset

    strict = UnifiedPhysiologyWindowDataset(cache, dataset_ids=["refed"])
    diagnostic = UnifiedPhysiologyWindowDataset(cache, dataset_ids=["refed"], admissible_alignment_cases=None)
    assert len(strict) == 0
    assert len(strict.excluded_alignment_records) == 1
    assert len(diagnostic) == 1


def test_v3_artifact_cache_loads_only_when_source_stat_and_join_key_match(tmp_path):
    source = tmp_path / "cnt.mat"
    source.write_bytes(b"source")
    cache_root = tmp_path / "cache"
    path = cache_root / "subject_01" / "session_00.npz"
    path.parent.mkdir(parents=True)
    state = {"signal_branch": "single_trial_eeg_artifact_clean_v3"}
    np.savez_compressed(
        path,
        schema=np.asarray("single_trial_eeg_artifact_cache_v3"),
        join_key=np.asarray("eeg_fnirs_single_trial|subject_01|session_00"),
        source_path=np.asarray("cnt.mat"),
        source_size_bytes=np.asarray(source.stat().st_size, dtype=np.int64),
        source_mtime_ns=np.asarray(source.stat().st_mtime_ns, dtype=np.int64),
        eeg=np.zeros((20, 2), dtype=np.float32),
        artifact_mask=np.zeros(20, dtype=bool),
        bad_channel_mask=np.zeros(2, dtype=bool),
        channel_names=np.asarray(["F3", "F4"]),
        preprocessing_state_json=np.asarray(json.dumps(state)),
    )
    repo_root = Path(__file__).resolve().parents[1]
    code_sha256 = {
        "audit": hashlib.sha256(
            (repo_root / "experiments/audit_single_trial_eeg_artifact_v2.py").read_bytes()
        ).hexdigest(),
        "cleaner": hashlib.sha256(
            Path(clean_single_trial_eeg.__code__.co_filename).read_bytes()
        ).hexdigest(),
    }
    (cache_root / "cache_manifest.json").write_text(json.dumps({
        "schema": "single_trial_eeg_artifact_cache_v3",
        "signal_branch": "single_trial_eeg_artifact_clean_v3",
        "cleaning_config": EEGArtifactCleaningConfig().to_dict(),
        "code_sha256": code_sha256,
        "records": [{"join_key": "eeg_fnirs_single_trial|subject_01|session_00"}],
    }))
    dataset = object.__new__(UnifiedPhysiologyWindowDataset)
    dataset.project_root = tmp_path
    dataset.eeg_artifact_cache_root = cache_root
    dataset.eeg_artifact_config = None
    dataset._artifact_cache_manifest = None
    record = SimpleNamespace(
        dataset_id="eeg_fnirs_single_trial",
        canonical_subject_id="subject_01",
        base_record_id="session_00",
        join_key="eeg_fnirs_single_trial|subject_01|session_00",
    )
    loaded = dataset._load_cached_single_trial_eeg(record, "single_trial_eeg_artifact_clean_v3")
    assert loaded is not None
    eeg, names, loaded_state, quality = loaded
    assert eeg.shape == (20, 2)
    assert names == ("F3", "F4")
    assert loaded_state["artifact_cache"]["used"] is True
    assert quality["artifact_mask"].shape == (20,)
    record.join_key = "eeg_fnirs_single_trial|subject_01|session_wrong"
    with pytest.raises(RuntimeError, match="manifest has no record"):
        dataset._load_cached_single_trial_eeg(record, "single_trial_eeg_artifact_clean_v3")
    record.join_key = "eeg_fnirs_single_trial|subject_01|session_00"
    source.write_bytes(b"changed source")
    with pytest.raises(RuntimeError, match="source EEG changed"):
        dataset._load_cached_single_trial_eeg(record, "single_trial_eeg_artifact_clean_v3")


def test_v3_artifact_cache_rejects_stale_code_hash(tmp_path):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    (cache_root / "cache_manifest.json").write_text(json.dumps({
        "schema": "single_trial_eeg_artifact_cache_v3",
        "signal_branch": "single_trial_eeg_artifact_clean_v3",
        "cleaning_config": EEGArtifactCleaningConfig().to_dict(),
        "code_sha256": {"audit": "stale", "cleaner": "stale"},
        "records": [],
    }))
    dataset = object.__new__(UnifiedPhysiologyWindowDataset)
    dataset.eeg_artifact_cache_root = cache_root
    dataset._artifact_cache_manifest = None
    with pytest.raises(RuntimeError, match="code hash mismatch"):
        dataset._validated_artifact_cache_manifest()
