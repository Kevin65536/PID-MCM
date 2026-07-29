from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest
import torch

from src.data.shared_driver_dataset import (
    SHARED_DRIVER_VIEW_SCHEMA,
    SharedDriverWindowDataset,
)
from src.data.shared_driver_targets import (
    RAW_VIEW_ARRAY_SCHEMA,
    RAW_VIEW_REGISTRY_SCHEMA,
    SHARED_DRIVER_ARRAY_SCHEMA,
    SHARED_DRIVER_SIDECAR_SCHEMA,
)


SAMPLE_KEY = "eeg_fnirs_single_trial|subject_01|session_01|event=7"


class _Base:
    window_offset_s = -5.0

    def __init__(self):
        record = SimpleNamespace(
            dataset_id="eeg_fnirs_single_trial",
            canonical_subject_id="subject_01",
            base_record_id="session_01",
            join_key="eeg_fnirs_single_trial|subject_01|session_01",
        )
        event = {
            "event_index": 7,
            "label": "MA",
            "onset_ms": 10000.0,
            "metadata": {"task": "mental_arithmetic"},
        }
        self.windows = [SimpleNamespace(record=record, event=event)]
        eeg_rows = [
            {
                "channel_name": f"F{index}",
                "base_channel_name": f"F{index}",
                "x": float(index),
                "y": 0.0,
                "z": 0.0,
                "coordinate_units": "normalized_head_unit",
            }
            for index in range(1, 8)
        ]
        fnirs_rows = [
            {
                "channel_name": f"A_{role}",
                "base_channel_name": "A",
                "x": 2.0,
                "y": 0.0,
                "z": 0.0,
                "coordinate_units": "normalized_head_unit",
            }
            for role in ("HbO", "HbR")
        ]
        self.sample = {
            "eeg": np.arange(7 * 4000, dtype=np.float32).reshape(7, 4000),
            "fnirs": np.arange(2 * 200, dtype=np.float32).reshape(2, 200),
            "valid_mask": {
                "eeg": np.ones(4000, dtype=bool),
                "fnirs": np.ones(200, dtype=bool),
            },
            "bad_channel_mask": {
                "eeg": np.zeros(7, dtype=bool),
                "fnirs": np.zeros(2, dtype=bool),
            },
            "component_roles": {"fnirs": ["HbO", "HbR"]},
            "channel_geometry": {"eeg": eeg_rows, "fnirs": fnirs_rows},
            "channel_names": {
                "eeg": [row["channel_name"] for row in eeg_rows],
                "fnirs": [row["channel_name"] for row in fnirs_rows],
            },
            "label": {
                "namespace": "eeg_fnirs_single_trial:mental_arithmetic",
                "class_index": 0,
                "condition": "MA",
            },
            "dataset_id": record.dataset_id,
            "subject": record.canonical_subject_id,
            "record_id": record.base_record_id,
            "event": event,
        }

    def __getitem__(self, index):
        return self.sample


def _write_manifest(root, schema, arrays, **extra):
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": schema,
                "arrays_file": arrays.name,
                "arrays_sha256": hashlib.sha256(arrays.read_bytes()).hexdigest(),
                "sample_count": 1,
                "sample_order_sha256": hashlib.sha256(
                    SAMPLE_KEY.encode("utf-8")
                ).hexdigest(),
                "protected_test_included": False,
                **extra,
            }
        ),
        encoding="utf-8",
    )


def _write_view(root):
    root.mkdir()
    arrays = root / "arrays.npz"
    np.savez_compressed(
        arrays,
        schema=np.asarray(RAW_VIEW_ARRAY_SCHEMA),
        sample_key=np.asarray([SAMPLE_KEY]),
        selected_eeg_channels=np.asarray(
            [["F1", "F2", "F3", "F4", "F5", "F6"]]
        ),
        selected_fnirs_channels=np.asarray([["A_HbO", "A_HbR"]]),
        anchor_id=np.asarray(["A"]),
        selection_fold=np.asarray(["subject_01:loto=7"]),
        selection_source_hash=np.asarray(["view-source"]),
    )
    _write_manifest(root, RAW_VIEW_REGISTRY_SCHEMA, arrays)


def _write_teacher(root):
    root.mkdir()
    arrays = root / "arrays.npz"
    values = np.ones((1, 10, 20), dtype=np.float32)
    masks = np.ones_like(values, dtype=bool)
    np.savez_compressed(
        arrays,
        schema=np.asarray(SHARED_DRIVER_ARRAY_SCHEMA),
        sample_key=np.asarray([SAMPLE_KEY]),
        target_shared_driver=values,
        target_point_valid_mask=masks,
        target_eeg_only_driver=values * 0.5,
        eeg_only_point_valid_mask=masks,
        teacher_scope=np.asarray(["development_crossfit"]),
        teacher_parameter_fold=np.asarray(["subject_01:loto=7"]),
        teacher_gauge_hash=np.asarray(["gauge"]),
        teacher_source_hash=np.asarray(["source"]),
    )
    _write_manifest(
        root,
        SHARED_DRIVER_SIDECAR_SCHEMA,
        arrays,
        target_family="adaptive_joint_full_trajectory",
        teacher_scope="development_crossfit",
    )


def _dataset(view, teacher=None, *, require=False):
    return SharedDriverWindowDataset(
        raw_view_registry_root=str(view),
        trajectory_sidecar_root=None if teacher is None else str(teacher),
        expected_teacher_scope=(
            None if teacher is None else "development_crossfit"
        ),
        expected_target_family=(
            None if teacher is None else "adaptive_joint_full_trajectory"
        ),
        require_trajectory_target=require,
        base_dataset=_Base(),
        subject_keys=["eeg_fnirs_single_trial|subject_01"],
        task_namespaces=["eeg_fnirs_single_trial:mental_arithmetic"],
        reject_unknown_labels=False,
    )


def test_teacher_join_cannot_change_frozen_raw_view(tmp_path):
    view = tmp_path / "view"
    teacher = tmp_path / "teacher"
    _write_view(view)
    _write_teacher(teacher)

    without_teacher = _dataset(view)[0]
    with_teacher = _dataset(view, teacher, require=True)[0]

    assert with_teacher["schema"] == SHARED_DRIVER_VIEW_SCHEMA
    assert torch.equal(without_teacher["eeg"], with_teacher["eeg"])
    assert torch.equal(without_teacher["fnirs"], with_teacher["fnirs"])
    assert (
        without_teacher["raw_view_sha256"]
        == with_teacher["raw_view_sha256"]
    )
    assert with_teacher["teacher"]["target_shared_driver"].shape == (10, 20)


def test_shared_driver_view_uses_boundary_mask_only(tmp_path):
    view = tmp_path / "view"
    _write_view(view)
    dataset = _dataset(view)
    dataset.base.sample["valid_mask"]["eeg"][:400] = False

    item = dataset[0]

    assert not item["token_valid_mask"]["eeg"][0]
    assert item["token_valid_mask"]["eeg"][1:].all()
    assert item["eeg"][:, :400].eq(0).all()


def test_shared_driver_view_requires_subject_allowlist_and_rejects_protected(
    tmp_path,
):
    view = tmp_path / "view"
    _write_view(view)

    with pytest.raises(ValueError, match="explicit subject_keys allowlist"):
        SharedDriverWindowDataset(
            raw_view_registry_root=str(view),
            require_trajectory_target=False,
            base_dataset=_Base(),
            subject_keys=None,
            reject_unknown_labels=False,
        )

    with pytest.raises(PermissionError, match="protected-test"):
        SharedDriverWindowDataset(
            raw_view_registry_root=str(view),
            require_trajectory_target=False,
            base_dataset=_Base(),
            subject_keys=["eeg_fnirs_single_trial|subject_24"],
            reject_unknown_labels=False,
        )
