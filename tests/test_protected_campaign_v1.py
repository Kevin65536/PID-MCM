from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from comparative_methods.aggregate_protected_campaign import _ccc, _verify_job
from comparative_methods.protected_campaign_common import (
    CampaignError,
    read_json,
    sha256_file,
    verify_authorization,
    verify_candidate_file,
    verify_file,
    write_json_atomic,
)
from comparative_methods.protected_campaign_controller import campaign_status
from comparative_methods.protected_campaign_worker import (
    _device_uuid,
    _rows,
    _validate_predictions,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    REPO_ROOT
    / "comparative_methods/evidence/protected_campaign/joint_release_candidate_v1.json"
)


@pytest.fixture
def non_authorizing_template(tmp_path: Path) -> Path:
    """Build a real template without reading the completed campaign's signed record."""
    output = tmp_path / "authorization_template_v1.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "comparative_methods/prepare_protected_authorization.py"),
            "--candidate",
            str(CANDIDATE),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return output


@pytest.mark.sealed_evidence
def test_release_candidate_has_exact_frozen_matrix_without_protected_artifacts() -> None:
    candidate, _digest = verify_candidate_file(CANDIDATE, verify_artifacts=False)
    assert candidate["state"] in {"DRAFT", "REVIEWED"}
    assert candidate["protected_evaluation_authorized"] is False
    assert candidate["protected_test_opened"] is False
    assert candidate["disposition_counts"] == {
        "direct": 34,
        "overlap": 2,
        "supported": 36,
        "unsupported": 6,
        "jobs": 540,
    }
    assert len({row["job_id"] for row in candidate["jobs"]}) == 540
    assert {row["outer_fold"] for row in candidate["jobs"]} == set(range(5))
    assert {row["seed"] for row in candidate["jobs"]} == {17, 42, 73}
    assert all(
        "protected" not in Path(artifact["path"]).parts
        for job in candidate["jobs"]
        for artifact in job["artifacts"]
    )
    assert all(
        "protected_manifest_path" in job["input_contract"]
        and "protected_manifest_sha256" in job["input_contract"]
        for job in candidate["jobs"]
    )


@pytest.mark.sealed_evidence
def test_authorization_template_is_separate_and_strictly_non_authorizing(
    non_authorizing_template: Path,
) -> None:
    candidate, candidate_sha256 = verify_candidate_file(
        CANDIDATE, verify_artifacts=False
    )
    template = read_json(non_authorizing_template)
    assert template["candidate_sha256"] == candidate_sha256
    assert template["protected_evaluation_authorized"] is False
    with pytest.raises(CampaignError, match="not authorized"):
        verify_authorization(
            non_authorizing_template,
            candidate=candidate,
            candidate_sha256=candidate_sha256,
        )


@pytest.mark.sealed_evidence
def test_authorization_template_hash_is_not_candidate_hash(
    non_authorizing_template: Path,
) -> None:
    assert sha256_file(non_authorizing_template) != sha256_file(CANDIDATE)


@pytest.mark.sealed_evidence
def test_unsupported_cells_have_zero_jobs_and_supported_cells_have_fifteen() -> None:
    candidate = read_json(CANDIDATE)
    counts: dict[tuple[str, str], int] = {}
    for row in candidate["jobs"]:
        key = (row["method_id"], row["task"])
        counts[key] = counts.get(key, 0) + 1
    for cell in candidate["cells"]:
        key = (cell["method_id"], cell["task_id"])
        expected = 0 if cell["campaign_disposition"] == "unsupported" else 15
        assert counts.get(key, 0) == expected


@pytest.mark.sealed_evidence
def test_each_supported_cell_has_exact_fold_seed_product() -> None:
    candidate = read_json(CANDIDATE)
    grouped: dict[tuple[str, str], set[tuple[int, int]]] = {}
    for row in candidate["jobs"]:
        grouped.setdefault((row["method_id"], row["task"]), set()).add(
            (row["outer_fold"], row["seed"])
        )
    expected = {(fold, seed) for fold in range(5) for seed in (17, 42, 73)}
    assert grouped and all(values == expected for values in grouped.values())


@pytest.mark.sealed_evidence
def test_two_retained_cpu_shadow_passes_are_bitwise_identical_and_redacted() -> None:
    candidate = read_json(CANDIDATE)
    pre_lane_candidate_sha256 = candidate["pre_lane_candidate_sha256"]
    roots = (
        REPO_ROOT
        / "comparative_methods/evidence/protected_campaign/shadow_cpu_pass_v1",
        REPO_ROOT
        / "comparative_methods/evidence/protected_campaign/shadow_cpu_pass_v1_repeat",
    )
    for method in ("biot", "cbramod", "reve", "efrm", "normwear", "brainfusion"):
        directories = [root / method for root in roots]
        for directory in directories:
            manifest = read_json(directory / "job_manifest.json")
            assert manifest["candidate_sha256"] == pre_lane_candidate_sha256
            assert manifest["protected_test_opened"] is False
            assert manifest["performance_computed"] is False
            for json_path in directory.glob("*.json"):
                serialized = json_path.read_text(encoding="utf-8").lower()
                assert not any(
                    token in serialized
                    for token in ("target", "logits", "metric", "confusion", "sample_id")
                )
        with np.load(
            directories[0] / "shadow_predictions.npz", allow_pickle=False
        ) as left, np.load(
            directories[1] / "shadow_predictions.npz", allow_pickle=False
        ) as right:
            assert left.files == right.files
            for name in left.files:
                assert left[name].dtype == right[name].dtype
                assert left[name].shape == right[name].shape
                assert np.array_equal(left[name], right[name])


@pytest.mark.sealed_evidence
def test_single_modal_formal_jobs_freeze_live_encoder_artifacts() -> None:
    candidate = read_json(CANDIDATE)
    expected = {
        "biot": ("biot_live_eeg", {"encoder_checkpoint"}),
        "cbramod": ("cbramod_live_eeg", {"encoder_checkpoint"}),
        "reve": ("reve_live_eeg", {"encoder_checkpoint", "position_bank"}),
    }
    for method, (worker_kind, live_roles) in expected.items():
        jobs = [row for row in candidate["jobs"] if row["method_slug"] == method]
        assert jobs
        assert {row["worker_kind"] for row in jobs} == {worker_kind}
        for job in jobs:
            roles = {artifact["role"] for artifact in job["artifacts"]}
            assert live_roles <= roles
            assert (
                job["frozen_inference_contract"]["protected_feature_source"]
                == "hash_pinned_frozen_encoder_over_exact_authorized_indices"
            )


@pytest.mark.sealed_evidence
def test_user_authorized_single_gpu_lane_is_exact_and_has_no_fallback() -> None:
    candidate = read_json(CANDIDATE)
    lane = candidate["lane_manifest"]["value"]
    expected_uuid = "GPU-130bb706-1e2e-4523-d23e-6d98c8d9854c"
    assert lane["execution_policy"] == "single_frozen_gpu_user_override"
    assert lane["minimum_healthy_idle_gpus"] == 1
    assert len(lane["gpu_snapshot"]) == 1
    assert lane["gpu_snapshot"][0]["uuid"] == expected_uuid
    assert {row["gpu_uuid"] for row in lane["assignments"]} == {expected_uuid}
    assert sum(row["job_count"] for row in lane["assignments"]) == 540
    assert lane["backup_gpu_uuids"] == []
    assert lane["single_gpu_policy_authorization"] == {
        "protocol_owner": "Hukaiwen",
        "run_owner": "Jiaminmin",
        "all_540_jobs_fixed_to_one_gpu": True,
        "automatic_gpu_migration_forbidden": True,
    }
    assert all(
        row["comparison_mode"] == "single_frozen_gpu_self_consistency"
        and row["equivalent"] is True
        and row["maximum_absolute_difference"] == 0.0
        for row in lane["gpu_equivalence"].values()
    )


@pytest.mark.sealed_evidence
def test_quarantine_is_a_durable_incomplete_terminal(tmp_path: Path) -> None:
    candidate = read_json(CANDIDATE)
    job_id = candidate["jobs"][0]["job_id"]
    quarantine = tmp_path / "quarantine" / f"{job_id}.attempt1.injected"
    quarantine.mkdir(parents=True)
    write_json_atomic(
        quarantine / "status.json",
        {
            "attempt": 1,
            "failure_code": "FAILED_TECHNICAL",
            "protected_test_opened": False,
        },
    )
    status = campaign_status(candidate, tmp_path)
    assert status["state"] == "INCOMPLETE_TECHNICAL"
    assert status["failed_job_count"] == 1
    assert status["missing_job_count"] == 539
    assert status["protected_test_opened"] is False


@pytest.mark.sealed_evidence
def test_candidate_checker_rejects_unsafe_job_id_and_wrong_seed(tmp_path: Path) -> None:
    candidate = read_json(CANDIDATE)
    unsafe = copy.deepcopy(candidate)
    unsafe["jobs"][0]["job_id"] = "../outside"
    unsafe_path = tmp_path / "unsafe.json"
    write_json_atomic(unsafe_path, unsafe)
    with pytest.raises(CampaignError, match="unsafe or non-canonical"):
        verify_candidate_file(unsafe_path, verify_artifacts=False)

    wrong_seed = copy.deepcopy(candidate)
    wrong_seed["jobs"][0]["seed"] = 99
    wrong_seed_path = tmp_path / "wrong_seed.json"
    write_json_atomic(wrong_seed_path, wrong_seed)
    with pytest.raises(CampaignError, match="job ID and job identity"):
        verify_candidate_file(wrong_seed_path, verify_artifacts=False)


@pytest.mark.sealed_evidence
def test_candidate_checker_rejects_input_contract_and_cell_routing_drift(
    tmp_path: Path,
) -> None:
    candidate = read_json(CANDIDATE)
    contract_drift = copy.deepcopy(candidate)
    contract_drift["jobs"][0]["input_contract"]["protected_sample_count"] += 1
    contract_path = tmp_path / "contract_drift.json"
    write_json_atomic(contract_path, contract_drift)
    with pytest.raises(CampaignError, match="input contract identity"):
        verify_candidate_file(contract_path, verify_artifacts=False)

    routing_drift = copy.deepcopy(candidate)
    routing_drift["cells"][0]["campaign_disposition"] = "unsupported"
    routing_path = tmp_path / "routing_drift.json"
    write_json_atomic(routing_path, routing_drift)
    with pytest.raises(CampaignError, match="disposition routing"):
        verify_candidate_file(routing_path, verify_artifacts=False)


def test_runtime_audit_redacts_payload_field_names_and_values() -> None:
    arrays = {
        "schema_version": np.asarray(
            ["joint_protected_predictions_v1"] * 2, dtype="<U30"
        ),
        "dataset_index": np.asarray([1, 2], dtype=np.int64),
        "identity": np.asarray(["a", "b"]),
        "logits": np.asarray([[0.1, 0.9], [0.7, 0.3]], dtype=np.float32),
        "prediction": np.asarray([1, 0], dtype=np.int64),
        "target": np.asarray([1, 1], dtype=np.int64),
    }
    report = _validate_predictions(arrays, expected_count=2)
    serialized = json.dumps(report).lower()
    for forbidden in ("target", "logits", "metric", "confusion", "sample_id"):
        assert forbidden not in serialized
    assert report["sample_count"] == 2
    assert report["unique_identity_count"] == 2
    assert report["all_numeric_finite"] is True


def test_runtime_audit_rejects_duplicate_identity_and_nonfinite_output() -> None:
    duplicate = {
        "schema_version": np.asarray(
            ["joint_protected_predictions_v1"] * 2, dtype="<U30"
        ),
        "dataset_index": np.asarray([1, 2], dtype=np.int64),
        "identity": np.asarray(["a", "a"]),
        "prediction": np.asarray([0, 1], dtype=np.int64),
        "target": np.asarray([0, 1], dtype=np.int64),
    }
    with pytest.raises(CampaignError, match="duplicate"):
        _validate_predictions(duplicate, expected_count=2)
    nonfinite = {
        **duplicate,
        "identity": np.asarray(["a", "b"]),
        "score": np.asarray([0.0, np.nan]),
    }
    with pytest.raises(CampaignError, match="non-finite"):
        _validate_predictions(nonfinite, expected_count=2)


def test_mask_aware_ccc_golden_data() -> None:
    target = np.asarray(
        [
            [[0.0, 1.0, 2.0]],
            [[3.0, 4.0, 5.0]],
        ],
        dtype=np.float32,
    )
    valid = np.ones_like(target, dtype=bool)
    assert _ccc(target, target.copy(), valid) == pytest.approx(1.0)
    reversed_prediction = target[:, :, ::-1].copy()
    assert _ccc(target, reversed_prediction, valid) < 1.0


def test_hash_fault_and_missing_cache_index_are_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"frozen")
    with pytest.raises(CampaignError, match="drifted"):
        verify_file(str(artifact), "0" * 64, label="fault injection")
    with pytest.raises(CampaignError, match="absent"):
        _rows(np.asarray([1, 2], dtype=np.int64), [3])


def test_atomic_commit_failure_does_not_publish_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "status.json"

    def fail_replace(_source: str, _destination: Path) -> None:
        raise OSError("injected disk failure")

    monkeypatch.setattr(
        "comparative_methods.protected_campaign_common.os.replace", fail_replace
    )
    with pytest.raises(OSError, match="disk failure"):
        write_json_atomic(destination, {"status": "test"})
    assert not destination.exists()


def test_gpu_uuid_failure_is_not_silently_recovered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_gpu(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("injected GPU failure")

    monkeypatch.setattr(
        "comparative_methods.protected_campaign_worker.subprocess.check_output", fail_gpu
    )
    with pytest.raises(RuntimeError, match="GPU failure"):
        _device_uuid(torch.device("cuda:0"))


def test_aggregator_rejects_unbound_job_before_opening_predictions(
    tmp_path: Path,
) -> None:
    job = {"job_id": "example__job"}
    directory = tmp_path / job["job_id"]
    directory.mkdir()
    write_json_atomic(
        directory / "status.json",
        {
            "schema": "joint_protected_campaign_job_v1",
            "job_id": job["job_id"],
            "attempt": 1,
            "status": "COMPLETED",
            "surface": "protected",
            "protected_test_opened": False,
        },
    )
    with pytest.raises(CampaignError, match="not a completed protected terminal"):
        _verify_job(
            directory,
            job=job,
            candidate_sha256="0" * 64,
            authorization_sha256="1" * 64,
            environment_sha256="2" * 64,
        )
