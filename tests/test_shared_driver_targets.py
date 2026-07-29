import hashlib
import json

import numpy as np
import pytest
import torch

from src.data.shared_driver_targets import (
    PhysiologyRawViewRegistry,
    RAW_VIEW_ARRAY_SCHEMA,
    RAW_VIEW_REGISTRY_SCHEMA,
    SHARED_DRIVER_ARRAY_SCHEMA,
    SHARED_DRIVER_SIDECAR_SCHEMA,
    SharedDriverTrajectorySidecar,
)


SAMPLE_KEY = "eeg_fnirs_single_trial|subject_01|session_01|event=7"


def _manifest(root, *, schema, arrays, **extra):
    order = hashlib.sha256(SAMPLE_KEY.encode("utf-8")).hexdigest()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": schema,
                "arrays_file": arrays.name,
                "arrays_sha256": hashlib.sha256(arrays.read_bytes()).hexdigest(),
                "sample_count": 1,
                "sample_order_sha256": order,
                "protected_test_included": False,
                **extra,
            }
        ),
        encoding="utf-8",
    )


def _write_teacher(root, *, scope="development_crossfit"):
    root.mkdir()
    arrays = root / "arrays.npz"
    joint = np.arange(200, dtype=np.float32).reshape(1, 10, 20)
    eeg_only = joint - 2.0
    point_mask = np.ones((1, 10, 20), dtype=bool)
    point_mask[:, 3, 7:] = False
    np.savez_compressed(
        arrays,
        schema=np.asarray(SHARED_DRIVER_ARRAY_SCHEMA),
        sample_key=np.asarray([SAMPLE_KEY]),
        target_shared_driver=joint,
        target_point_valid_mask=point_mask,
        target_eeg_only_driver=eeg_only,
        eeg_only_point_valid_mask=np.ones_like(point_mask),
        teacher_scope=np.asarray([scope]),
        teacher_parameter_fold=np.asarray(["subject_01:loto=7"]),
        teacher_gauge_hash=np.asarray(["gauge"]),
        teacher_source_hash=np.asarray(["source"]),
    )
    _manifest(
        root,
        schema=SHARED_DRIVER_SIDECAR_SCHEMA,
        arrays=arrays,
        target_family="adaptive_joint_full_trajectory",
        teacher_scope=scope,
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
    _manifest(root, schema=RAW_VIEW_REGISTRY_SCHEMA, arrays=arrays)


def test_shared_driver_sidecar_preserves_full_point_support(tmp_path):
    root = tmp_path / "teacher"
    _write_teacher(root)
    sidecar = SharedDriverTrajectorySidecar(
        root,
        expected_scope="development_crossfit",
        expected_family="adaptive_joint_full_trajectory",
    )

    row = sidecar.lookup(SAMPLE_KEY)

    assert row is not None
    assert row["target_shared_driver"].shape == (10, 20)
    assert row["joint_correction"].eq(2.0).all()
    assert row["teacher_mask"].all()
    assert not row["target_point_valid_mask"][3, 7:].any()


def test_point_loss_mask_is_exact_three_way_intersection():
    measurement = torch.tensor([[True, True, False]])
    teacher = torch.tensor([[True, False, True]])
    points = torch.ones(1, 3, 4, dtype=torch.bool)
    points[:, 0, -1] = False

    mask = SharedDriverTrajectorySidecar.point_loss_mask(
        measurement, teacher, points
    )

    assert mask.sum().item() == 3
    assert mask[0, 0, :3].all()
    assert not mask[0, 0, 3]
    assert not mask[0, 1:].any()


def test_raw_view_registry_is_independent_artifact(tmp_path):
    teacher_root = tmp_path / "teacher"
    view_root = tmp_path / "view"
    _write_teacher(teacher_root)
    _write_view(view_root)

    teacher = SharedDriverTrajectorySidecar(teacher_root)
    view = PhysiologyRawViewRegistry(view_root)

    assert teacher.contains(SAMPLE_KEY)
    assert view.lookup(SAMPLE_KEY)["selected_eeg_channels"] == (
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
    )
    assert "selected_eeg_channels" not in teacher.lookup(SAMPLE_KEY)


def test_development_sidecar_rejects_protected_payload(tmp_path):
    root = tmp_path / "teacher"
    _write_teacher(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protected_test_included"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="protected-test"):
        SharedDriverTrajectorySidecar(root)
