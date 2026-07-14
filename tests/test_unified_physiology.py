import numpy as np
import json

from src.data.unified_physiology import (
    CANONICAL_EEG_SAMPLE_RATE_HZ,
    CANONICAL_FNIRS_COMPONENTS,
    CANONICAL_FNIRS_SAMPLE_RATE_HZ,
    CANONICAL_UNIT,
    NativeEEGRecord,
    canonical_fnirs_channel_names,
    canonical_label,
    fnirs_component_roles,
    preprocess_eeg_record,
    preprocess_fnirs_record,
)


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
