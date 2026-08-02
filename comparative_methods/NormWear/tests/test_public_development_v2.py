from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import comparative_methods.NormWear.run_public_development_v2 as runner_module
from comparative_methods.NormWear.alignment_data import SUPPORTED_TASKS
from comparative_methods.NormWear.build_public_job_matrix_v2 import build_matrix
from comparative_methods.NormWear.run_public_development_v2 import (
    DEFAULT_CONFIG,
    PublicFold,
    class_weights,
    load_public_fold,
    load_runner_config,
    load_verified_feature_cache,
    run,
    select_and_refit,
    standardizer,
)
from comparative_methods.NormWear.run_public_matrix_v2 import execute, validate_jobs


METHOD_ROOT = Path(__file__).resolve().parents[1]
ALIGNMENT_CONTRACT = METHOD_ROOT.parent / "adapter_alignment_gate_contract_v2.yaml"


@pytest.fixture(autouse=True)
def _freeze_normwear_as_active_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = yaml.safe_load(ALIGNMENT_CONTRACT.read_text(encoding="utf-8"))
    frozen = copy.deepcopy(contract)
    frozen["execution_policy"]["active_delivery_method"] = (
        "normwear_eeg_fnirs_adapted"
    )
    path = tmp_path / "normwear_active_contract.yaml"
    path.write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(runner_module, "ALIGNMENT_CONTRACT", path)


def test_public_runner_config_freezes_serial_public_only_matrix() -> None:
    config, _config_path, alignment, _alignment_path = load_runner_config(DEFAULT_CONFIG)
    assert config["method_id"] == "normwear_eeg_fnirs_adapted"
    assert config["protected_test_default"] == "locked"
    assert tuple(config["job_matrix"]["tasks"]) == SUPPORTED_TASKS
    assert config["job_matrix"]["outer_folds"] == [0, 1, 2, 3, 4]
    assert config["job_matrix"]["seeds"] == [17, 42, 73]
    assert config["job_matrix"]["expected_public_jobs"] == 90
    assert config["failure_policy"]["automatic_retry_count"] == 0
    assert alignment["method_id"] == "normwear_eeg_fnirs_adapted"


def test_a7_feature_cache_is_bound_to_exact_nback_public_inventory() -> None:
    config, _config_path, alignment, _alignment_path = load_runner_config(DEFAULT_CONFIG)
    fold = load_public_fold(alignment, task="nback", outer_fold=0)
    arrays, identity, cache_dir, verification = load_verified_feature_cache(
        config=config, fold=fold
    )
    assert arrays["features"].shape == (702, 76800)
    assert arrays["features"].dtype == np.float32
    assert identity["task"] == "nback"
    assert identity["protected_test_opened"] is False
    assert cache_dir.is_dir()
    assert verification["global_target_metadata_loaded"] is False
    assert verification["selected_public_rows_materialized"] == (
        len(fold.train_indices) + len(fold.validation_indices)
    )
    assert set(fold.train_indices).isdisjoint(fold.validation_indices)


def test_standardizer_uses_only_the_array_it_is_given() -> None:
    train = np.asarray([[0.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    validation = np.asarray([[1000.0, -1000.0]], dtype=np.float32)
    mean, scale = standardizer(train, 1.0e-6)
    np.testing.assert_array_equal(mean, np.asarray([1.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(scale, np.asarray([1.0, 1.0], dtype=np.float32))
    validation[:] = 0.0
    mean_after, scale_after = standardizer(train, 1.0e-6)
    np.testing.assert_array_equal(mean_after, mean)
    np.testing.assert_array_equal(scale_after, scale)


def test_inverse_frequency_weights_reject_empty_training_class() -> None:
    np.testing.assert_allclose(class_weights(np.asarray([0, 0, 1]), 2), [0.75, 1.5])
    with pytest.raises(RuntimeError, match="empty class"):
        class_weights(np.asarray([0, 0]), 2)


def test_public_probe_selects_refits_and_weights_only_reloads(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    features = np.concatenate(
        (
            rng.normal(-1.0, 0.1, (12, 32)),
            rng.normal(1.0, 0.1, (12, 32)),
        )
    ).astype(np.float32)
    arrays = {
        "features": features,
        "targets": np.asarray([0] * 12 + [1] * 12, dtype=np.int64),
        "dataset_indices": np.arange(24, dtype=np.int64),
        "subjects": np.asarray([f"s{index // 4}" for index in range(24)]),
        "sample_ids": np.asarray([f"sample-{index}" for index in range(24)]),
    }
    inventory = SimpleNamespace(
        task="nback",
        dataset=SimpleNamespace(spec=SimpleNamespace(class_names=("zero", "two"))),
    )
    fold = PublicFold(
        inventory=inventory,  # type: ignore[arg-type]
        outer_fold=0,
        public_manifest_path=tmp_path / "public.json",
        public_manifest_sha256="a" * 64,
        train_indices=tuple([*range(8), *range(12, 20)]),
        validation_indices=tuple([*range(8, 12), *range(20, 24)]),
    )
    config, config_path, _alignment, alignment_path = load_runner_config(DEFAULT_CONFIG)
    report, logits = select_and_refit(
        arrays=arrays,
        fold=fold,
        train_indices=fold.train_indices,
        validation_indices=fold.validation_indices,
        config=config,
        config_path=config_path,
        alignment_path=alignment_path,
        cache_identity={"feature_cache_key": "synthetic"},
        seed=17,
        device=torch.device("cpu"),
        smoke=True,
        output_dir=tmp_path,
    )
    assert logits.shape == (8, 2)
    assert report["selection_standardizer"]["fit_membership"] == "outer_train_only"
    assert report["public_refit"]["membership"] == (
        "smoke_train_plus_validation_subset"
    )
    assert report["public_refit"]["weights_only_reload_match"] is True
    checkpoint = torch.load(
        tmp_path / "checkpoint_public_refit.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint["protected_test_opened"] is False
    assert checkpoint["head_state"]["weight"].shape == (2, 32)


def test_public_runner_refuses_output_outside_normwear_run_root(tmp_path: Path) -> None:
    args = SimpleNamespace(
        config=DEFAULT_CONFIG,
        task="nback",
        outer_fold=0,
        seed=17,
        output_dir=tmp_path / "outside",
        smoke=True,
    )
    with pytest.raises(PermissionError, match="must remain under"):
        run(args)


def test_candidate_matrix_is_serial_public_only_and_not_self_authorizing() -> None:
    matrix = build_matrix()
    assert matrix["job_count"] == 90
    assert matrix["max_concurrent_jobs"] == 1
    assert matrix["automatic_retry_count"] == 0
    assert matrix["public_matrix_launch_authorized"] is False
    assert matrix["protected_evaluation_authorized"] is False
    assert matrix["protected_test_opened"] is False
    assert all(job["initial_status"] == "queued_not_authorized" for job in matrix["jobs"])
    assert all("protected" not in " ".join(job["command"]).lower() for job in matrix["jobs"])
    jobs = validate_jobs(
        matrix,
        run_root=(METHOD_ROOT / "runs/public_development_v2/matrix_v2").resolve(),
    )
    assert [job["order"] for job in jobs] == list(range(90))


def test_reviewed_launch_authorizes_only_serial_public_matrix() -> None:
    report = execute(METHOD_ROOT / "configs/public_matrix_launch_v2.yaml", dry_run=True)
    assert report["status"] == "pass"
    assert report["job_count"] == 90
    assert report["max_concurrent_jobs"] == 1
    assert report["automatic_retry_count"] == 0
    assert report["public_matrix_launch_authorized"] is True
    assert report["protected_evaluation_authorized"] is False
    assert report["protected_test_opened"] is False


def test_completed_public_matrix_closes_delivery_queue_with_protected_locked() -> None:
    import json

    completion = json.loads(
        (METHOD_ROOT / "evidence/public_development_v2/matrix_completion_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert completion["status"] == "pass"
    assert completion["job_count"] == 90
    assert completion["completed_job_count"] == 90
    assert completion["failed_job_count"] == 0
    assert completion["max_concurrent_jobs"] == 1
    assert completion["automatic_retry_count"] == 0
    assert len(completion["tasks"]) == 6
    assert all(task["job_count"] == 15 for task in completion["tasks"])
    assert completion["protected_evaluation_authorized"] is False
    assert completion["protected_test_opened"] is False

    final = json.loads(
        (METHOD_ROOT / "evidence/alignment_v2/summary_final.json").read_text(
            encoding="utf-8"
        )
    )
    assert final["status"] == "public_development_complete_A0_A8_pass_protected_locked"
    assert final["completed_public_job_count"] == 90
    assert final["protected_evaluation_authorized"] is False
    assert final["protected_test_opened"] is False

    contract = yaml.safe_load(ALIGNMENT_CONTRACT.read_text(encoding="utf-8"))
    assert contract["execution_policy"]["active_delivery_method"] == (
        "none_public_delivery_queue_complete"
    )
    queue = contract["execution_policy"]["ordered_queue"]
    assert queue[-1]["method_id"] == "normwear_eeg_fnirs_adapted"
    assert queue[-1]["current_state"] == "public_development_complete_protected_locked"
