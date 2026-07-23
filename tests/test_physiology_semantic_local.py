from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest

from src.data.physiology_semantic_local import (
    LOCAL_VIEW_SCHEMA,
    UnifiedPhysiologyLocalViewDataset,
)
from src.data.physiology_semantic_targets import TARGET_ARRAY_SCHEMA, TARGET_SIDECAR_SCHEMA


def _base_dataset(*, fnirs_units="normalized_head_unit"):
    record = SimpleNamespace(
        dataset_id="eeg_fnirs_single_trial",
        canonical_subject_id="subject_01",
        base_record_id="session_00",
        join_key="eeg_fnirs_single_trial|subject_01|session_00",
    )
    event = {
        "event_index": 3,
        "event_type": "trial",
        "label": "LMI",
        "metadata": {"task": "motor_imagery"},
    }
    eeg_rows = [
        {
            "channel_name": f"E{index}",
            "base_channel_name": f"E{index}",
            "x": float(index),
            "y": 0.0,
            "z": 0.0,
            "coordinate_units": "normalized_head_unit",
        }
        for index in range(8)
    ]
    fnirs_rows = []
    for anchor, x in (("A", 1.0), ("B", 6.0)):
        for component in ("HbO", "HbR"):
            fnirs_rows.append(
                {
                    "channel_name": f"{anchor}_{component}",
                    "base_channel_name": anchor,
                    "component": component,
                    "x": x,
                    "y": 0.0,
                    "z": 0.0,
                    "coordinate_units": fnirs_units,
                }
            )
    eeg_valid = np.ones(4000, dtype=bool)
    eeg_valid[:400] = False
    sample = {
        "eeg": np.ones((8, 4000), dtype=np.float32),
        "fnirs": np.ones((4, 200), dtype=np.float32),
        "analysis_valid_mask": {
            "eeg": eeg_valid,
            "fnirs": np.ones(200, dtype=bool),
        },
        "bad_channel_mask": {
            "eeg": np.asarray([True] + [False] * 7),
            "fnirs": np.zeros(4, dtype=bool),
        },
        "component_roles": {"fnirs": ["HbO", "HbR", "HbO", "HbR"]},
        "channel_geometry": {"eeg": eeg_rows, "fnirs": fnirs_rows},
        "channel_names": {
            "eeg": [row["channel_name"] for row in eeg_rows],
            "fnirs": [row["channel_name"] for row in fnirs_rows],
        },
        "label": {
            "namespace": "eeg_fnirs_single_trial:motor_imagery",
            "class_index": 0,
            "condition": "LMI",
        },
        "dataset_id": record.dataset_id,
        "subject": record.canonical_subject_id,
        "record_id": record.base_record_id,
        "join_key": record.join_key,
        "event": event,
    }
    return SimpleNamespace(windows=[SimpleNamespace(record=record, event=event)], __getitem__=None), sample


class _Base:
    def __init__(self, namespace, sample):
        self.windows = namespace.windows
        self.sample = sample

    def __getitem__(self, index):
        return self.sample


def test_unified_local_view_consumes_masks_and_bad_channels():
    namespace, sample = _base_dataset()
    dataset = UnifiedPhysiologyLocalViewDataset(
        base_dataset=_Base(namespace, sample),
        subject_keys=["eeg_fnirs_single_trial|subject_01"],
        task_namespaces=["eeg_fnirs_single_trial:motor_imagery"],
        reject_unknown_labels=False,
    )

    item = dataset[0]

    assert item["schema"] == LOCAL_VIEW_SCHEMA
    assert tuple(item["eeg"].shape) == (6, 4000)
    assert tuple(item["fnirs"].shape) == (2, 200)
    assert not item["token_valid_mask"]["eeg"][0]
    assert item["token_valid_mask"]["eeg"][1:].all()
    assert item["token_valid_mask"]["fnirs"].all()
    assert not item["eeg"][:, :400].any()
    assert "E0" not in item["selected_eeg_channels"]


def test_unified_local_view_rejects_unadmitted_coordinate_mix():
    namespace, sample = _base_dataset(fnirs_units="unknown_native")
    dataset = UnifiedPhysiologyLocalViewDataset(
        base_dataset=_Base(namespace, sample),
        subject_keys=["eeg_fnirs_single_trial|subject_01"],
        reject_unknown_labels=False,
    )
    with pytest.raises(ValueError, match="matching coordinate units"):
        dataset[0]


def test_unified_local_view_never_selects_a_bad_fnirs_component():
    namespace, sample = _base_dataset()
    sample["bad_channel_mask"]["fnirs"][:2] = True
    dataset = UnifiedPhysiologyLocalViewDataset(
        base_dataset=_Base(namespace, sample),
        subject_keys=["eeg_fnirs_single_trial|subject_01"],
        reject_unknown_labels=False,
    )

    item = dataset[0]

    assert item["anchor"] == "B"


def _write_sidecar(root):
    root.mkdir()
    sample_key = "eeg_fnirs_single_trial|subject_01|session_00|event=3"
    valid = np.ones((1, 10), dtype=bool)
    invalid = np.zeros((1, 10), dtype=bool)
    arrays = root / "targets.npz"
    np.savez_compressed(
        arrays,
        schema=np.asarray(TARGET_ARRAY_SCHEMA),
        sample_key=np.asarray([sample_key]),
        selected_eeg_channels=np.asarray([[f"E{index}" for index in range(1, 7)]]),
        selected_fnirs_channels=np.asarray([["B_HbO", "B_HbR"]]),
        eeg_target=np.ones((1, 10, 6), dtype=np.float32),
        eeg_uncertainty=np.ones((1, 10, 6), dtype=np.float32),
        fnirs_target=np.ones((1, 10, 9), dtype=np.float32),
        fnirs_uncertainty=np.ones((1, 10, 9), dtype=np.float32),
        eeg_local_valid_mask=valid,
        eeg_prototype_valid_mask=valid,
        eeg_context_valid_mask=invalid,
        eeg_coupling_valid_mask=invalid,
        fnirs_local_valid_mask=valid,
        fnirs_prototype_valid_mask=valid,
        fnirs_context_valid_mask=invalid,
        fnirs_coupling_valid_mask=invalid,
    )
    digest = hashlib.sha256(arrays.read_bytes()).hexdigest()
    order = hashlib.sha256(sample_key.encode("utf-8")).hexdigest()
    (root / "manifest.json").write_text(json.dumps({
        "schema": TARGET_SIDECAR_SCHEMA,
        "target_family": "adaptive_multimodal_consensus_proxy",
        "target_version": "adaptive_ssm_gauge_corrected_patch_v1",
        "protected_test_included": False,
        "arrays_file": "targets.npz",
        "arrays_sha256": digest,
        "sample_count": 1,
        "sample_order_sha256": order,
    }), encoding="utf-8")


def test_unified_local_view_joins_target_by_anchor_independent_identity(tmp_path):
    sidecar = tmp_path / "sidecar"
    _write_sidecar(sidecar)
    namespace, sample = _base_dataset()
    dataset = UnifiedPhysiologyLocalViewDataset(
        base_dataset=_Base(namespace, sample),
        subject_keys=["eeg_fnirs_single_trial|subject_01"],
        reject_unknown_labels=False,
        auxiliary_target_root=str(sidecar),
        auxiliary_target_family="adaptive_multimodal_consensus_proxy",
        auxiliary_target_version="adaptive_ssm_gauge_corrected_patch_v1",
        require_auxiliary_target=True,
    )

    item = dataset[0]

    assert item["has_auxiliary_target"]
    assert item["anchor"] == "B"
    assert item["selected_eeg_channels"] == [f"E{index}" for index in range(1, 7)]
    assert item["teacher"]["eeg_target"].shape == (10, 6)
    assert item["teacher"]["eeg_local_valid_mask"].all()
