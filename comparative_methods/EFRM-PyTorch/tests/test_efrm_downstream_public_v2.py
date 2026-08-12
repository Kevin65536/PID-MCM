from __future__ import annotations

import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
for path in (REPO_ROOT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_downstream_public_matrix_v2 import build_matrix
from run_downstream_public_v2 import (
    DEFAULT_CONFIG,
    CHECKPOINT_SCHEMA,
    PublicFold,
    PublicSurface,
    fit_target_scaler,
    feature_cache_identity,
    frozen_checkpoint_identity,
    load_config,
    load_public_surface,
    mask_sha256,
    select_and_refit,
    write_json,
)


def _fast_config() -> tuple[dict, Path]:
    config, path = load_config(DEFAULT_CONFIG)
    value = copy.deepcopy(config)
    value["selection"].update(
        {
            "epoch_cap": 2,
            "batch_size": 8,
            "learning_rate": 0.01,
            "minimum_learning_rate": 1.0e-5,
            "weight_decay": 0.0,
            "dropout": 0.0,
        }
    )
    value["smoke"]["epoch_cap"] = 2
    return value, path


def _fold(tmp_path: Path, *, task: str = "motor_imagery") -> PublicFold:
    return PublicFold(
        task=task,
        outer_fold=0,
        public_manifest_path=tmp_path / "public.json",
        public_manifest_sha256="a" * 64,
        public_split_sha256="b" * 64,
        train_indices=tuple([*range(8), *range(12, 20)]),
        validation_indices=tuple([*range(8, 12), *range(20, 24)]),
    )


def test_config_and_candidate_matrix_freeze_public_only_105_jobs() -> None:
    config, _path = load_config(DEFAULT_CONFIG)
    assert config["protected_test_default"] == "locked"
    assert config["job_matrix"]["expected_public_jobs"] == 105
    assert config["job_matrix"]["seeds"] == [17, 42, 73]
    assert config["protocol"]["target_dataset_exposure_allowed"] is False
    matrix = build_matrix(DEFAULT_CONFIG)
    assert matrix["job_count"] == 105
    assert matrix["max_concurrent_jobs"] == 1
    assert matrix["automatic_retry_count"] == 0
    assert matrix["public_matrix_launch_authorized"] is False
    assert matrix["protected_evaluation_authorized"] is False
    assert matrix["target_dataset_exposure"] is False
    assert matrix["protected_test_opened"] is False
    assert all("protected" not in " ".join(job["command"]).lower() for job in matrix["jobs"])


def test_support_mask_digest_preserves_shape_and_values() -> None:
    mask = torch.tensor([[True, False], [False, True]])
    assert mask_sha256(mask) == mask_sha256(mask.clone())
    assert mask_sha256(mask) != mask_sha256(mask.reshape(4))
    assert mask_sha256(mask) != mask_sha256(~mask)


def test_public_surface_covers_complete_task_without_protected_read() -> None:
    config, _path = load_config(DEFAULT_CONFIG)
    surface = load_public_surface(config, task="motor_imagery")
    assert len(surface.folds) == 5
    assert surface.full_public_indices == tuple(range(len(surface.dataset)))
    assert len(surface.public_inventory_sha256) == 64
    for fold in surface.folds.values():
        assert set(fold.train_indices).isdisjoint(fold.validation_indices)
        assert "protected" not in {part.lower() for part in fold.public_manifest_path.parts}

    checkpoint = frozen_checkpoint_identity(
        config, dataset_id=surface.dataset.spec.dataset_id
    )
    identity = feature_cache_identity(config, surface, checkpoint)
    assert len(identity["source_sha256"]["feature_materializer"]) == 64
    assert set(identity["data_branch_sha256"]) == {
        "measurement_adapter",
        "cache_manifest",
        "event_manifest",
        "geometry_manifest",
        "single_trial_eeg_branch",
        "simultaneous_eeg_branch",
    }
    assert all(len(value) == 64 for value in identity["data_branch_sha256"].values())


def test_classification_probe_selects_refits_and_weights_only_reloads(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    features = np.concatenate(
        (rng.normal(-1.0, 0.2, (12, 16)), rng.normal(1.0, 0.2, (12, 16)))
    ).astype(np.float32)
    arrays = {
        "features": features,
        "targets": np.asarray([0] * 12 + [1] * 12, dtype=np.int64),
        "target_valid_mask": np.ones(24, dtype=bool),
        "dataset_indices": np.arange(24, dtype=np.int64),
        "subjects": np.asarray([f"s{index // 4}" for index in range(24)]),
        "sample_ids": np.asarray([f"sample-{index}" for index in range(24)]),
    }
    spec = SimpleNamespace(
        task_type="classification",
        class_names=("left", "right"),
        target_names=(),
        output_dim=2,
        target_length=1,
    )
    surface = PublicSurface(
        task="motor_imagery",
        dataset=SimpleNamespace(spec=spec),  # type: ignore[arg-type]
        folds={},
        full_public_indices=tuple(range(24)),
        public_inventory_sha256="c" * 64,
        split_registry_sha256="d" * 64,
    )
    config, config_path = _fast_config()
    report, prediction = select_and_refit(
        arrays=arrays,
        surface=surface,
        fold=_fold(tmp_path),
        train_indices=_fold(tmp_path).train_indices,
        validation_indices=_fold(tmp_path).validation_indices,
        config=config,
        config_path=config_path,
        cache_identity={"feature_cache_key": "synthetic"},
        seed=17,
        device=torch.device("cpu"),
        smoke=True,
        output_dir=tmp_path,
    )
    assert prediction.shape == (8, 2)
    assert report["selection_metric"] == "macro_f1"
    assert report["public_refit"]["weights_only_reload_match"] is True
    checkpoint = torch.load(
        tmp_path / "checkpoint_public_refit.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint["schema"] == CHECKPOINT_SCHEMA
    assert checkpoint["embedding_dim"] == 16
    assert checkpoint["target_dataset_exposure"] is False
    assert checkpoint["protected_test_opened"] is False


def test_regression_scalers_are_fit_on_selection_and_refit_memberships(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    arrays = {
        "features": rng.normal(size=(24, 12)).astype(np.float32),
        "targets": rng.normal(size=(24, 2, 20)).astype(np.float32),
        "target_valid_mask": np.ones((24, 2, 20), dtype=bool),
        "dataset_indices": np.arange(24, dtype=np.int64),
        "subjects": np.asarray([f"s{index // 4}" for index in range(24)]),
        "sample_ids": np.asarray([f"sample-{index}" for index in range(24)]),
    }
    arrays["target_valid_mask"][-1, :, -3:] = False
    spec = SimpleNamespace(
        task_type="regression",
        class_names=(),
        target_names=("valence", "arousal"),
        output_dim=2,
        target_length=20,
    )
    surface = PublicSurface(
        task="refed_regression",
        dataset=SimpleNamespace(spec=spec),  # type: ignore[arg-type]
        folds={},
        full_public_indices=tuple(range(24)),
        public_inventory_sha256="c" * 64,
        split_registry_sha256="d" * 64,
    )
    fold = _fold(tmp_path, task="refed_regression")
    config, config_path = _fast_config()
    report, prediction = select_and_refit(
        arrays=arrays,
        surface=surface,
        fold=fold,
        train_indices=fold.train_indices,
        validation_indices=fold.validation_indices,
        config=config,
        config_path=config_path,
        cache_identity={"feature_cache_key": "synthetic-regression"},
        seed=42,
        device=torch.device("cpu"),
        smoke=True,
        output_dir=tmp_path,
    )
    assert prediction.shape == (8, 2, 20)
    assert report["selection_metric"] == "masked_rmse_scaled"
    assert report["selection_mode"] == "min"
    train_rows = np.asarray(fold.train_indices)
    expected_center, _expected_scale, _counts = fit_target_scaler(
        arrays["targets"][train_rows], arrays["target_valid_mask"][train_rows]
    )
    np.testing.assert_allclose(
        report["selection_target_scaler"]["center"], expected_center, rtol=0, atol=1e-7
    )
    assert report["selection_target_scaler"]["fit_membership"] == "outer_train_only"
    assert report["public_refit"]["target_scaler"]["fit_membership"] == (
        "outer_train_plus_public_validation"
    )
    assert report["public_refit"]["weights_only_reload_match"] is True


def test_evidence_writer_refuses_protected_output(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="protected EFRM output"):
        write_json(tmp_path / "protected" / "result.json", {"status": "forbidden"})
