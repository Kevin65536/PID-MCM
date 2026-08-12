#!/usr/bin/env python3
"""Run one frozen campaign job without calculating or displaying metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comparative_methods.protected_campaign_common import (  # noqa: E402
    JOB_SCHEMA,
    CampaignError,
    append_jsonl,
    artifact_map,
    index_jobs,
    portable_path,
    read_json,
    repo_path,
    sha256_file,
    stable_hash,
    utc_now,
    verify_authorization,
    verify_candidate_file,
    verify_file,
    verify_runtime_environment,
    write_json_atomic,
)


def _device_uuid(device: torch.device) -> str:
    if device.type != "cuda":
        return "CPU"
    index = int(device.index if device.index is not None else torch.cuda.current_device())
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    if not output:
        raise CampaignError(f"could not resolve UUID for CUDA device {index}")
    return output.splitlines()[0].strip()


def _indices_sha256(indices: Sequence[int]) -> str:
    return stable_hash(sorted(int(value) for value in indices))


def _load_surface_indices(
    job: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    surface: str,
) -> tuple[list[int], str, str]:
    if surface == "protected":
        contract = job["input_contract"]
        path = verify_file(
            str(contract["protected_manifest_path"]),
            str(contract["protected_manifest_sha256"]),
            label="authorized protected manifest",
        )
        manifest = read_json(path)
        values = manifest.get("test_indices")
        if not isinstance(values, list):
            raise CampaignError("protected manifest does not contain test_indices")
        indices = [int(value) for value in values]
        expected = str(contract["protected_indices_sha256"])
        retained = str(manifest.get("test_indices_sha256", expected))
        if retained != expected or _indices_sha256(indices) != expected:
            raise CampaignError("protected test identity hash drifted")
        if len(indices) != int(contract["protected_sample_count"]):
            raise CampaignError("protected test sample count drifted")
        return indices, portable_path(path), sha256_file(path)

    public_run = read_json(repo_path(str(artifacts["public_run_manifest"]["path"])))
    path_value = public_run.get("public_manifest_path")
    if not path_value:
        raise CampaignError("public run does not identify its public manifest")
    split_artifact = artifacts["public_split_manifest"]
    path = verify_file(
        str(split_artifact["path"]),
        str(split_artifact["sha256"]),
        label="frozen public split manifest",
    )
    if path != repo_path(str(path_value)):
        raise CampaignError("public run and frozen public split paths differ")
    manifest = read_json(path)
    values = manifest.get("validation_indices")
    if not isinstance(values, list) or not values:
        raise CampaignError("public shadow manifest lacks validation_indices")
    indices = [int(value) for value in values]
    return indices, portable_path(path), sha256_file(path)


def _rows(dataset_indices: np.ndarray, selected: Sequence[int]) -> np.ndarray:
    lookup = {int(value): row for row, value in enumerate(dataset_indices.tolist())}
    missing = [int(value) for value in selected if int(value) not in lookup]
    if missing:
        raise CampaignError(f"selected evaluation indices are absent from cache: {missing[:5]}")
    return np.asarray([lookup[int(value)] for value in selected], dtype=np.int64)


def _linear_inference(
    job: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    indices: Sequence[int],
    device: torch.device,
) -> dict[str, np.ndarray]:
    checkpoint = torch.load(
        repo_path(str(artifacts["downstream_checkpoint"]["path"])),
        map_location="cpu",
        weights_only=True,
    )
    with np.load(
        repo_path(str(artifacts["feature_cache"]["path"])), allow_pickle=False
    ) as payload:
        dataset_indices = payload["dataset_indices"].astype(np.int64, copy=False)
        selected_rows = _rows(dataset_indices, indices)
        features = payload["features"][selected_rows].astype(np.float32, copy=False)
        targets = payload["targets"][selected_rows]
    mean = checkpoint["feature_mean"].numpy().astype(np.float32, copy=False)
    scale = checkpoint["feature_scale"].numpy().astype(np.float32, copy=False)
    standardized = (features - mean) / scale
    if not np.isfinite(standardized).all():
        raise CampaignError("frozen standardization produced non-finite values")
    weight = checkpoint["head_state"]["weight"].to(device)
    bias = checkpoint["head_state"]["bias"].to(device)
    with torch.inference_mode():
        logits = F.linear(torch.from_numpy(standardized).to(device), weight, bias)
    prediction = logits.argmax(dim=1)
    return {
        "dataset_index": np.asarray(indices, dtype=np.int64),
        "identity": np.asarray(
            [f"{job['task']}|dataset_index={value}" for value in indices], dtype=str
        ),
        "logits": logits.float().cpu().numpy(),
        "prediction": prediction.cpu().numpy().astype(np.int64, copy=False),
        "target": targets.astype(np.int64, copy=False),
    }


def _normwear_inference(
    job: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    indices: Sequence[int],
    device: torch.device,
) -> dict[str, np.ndarray]:
    checkpoint = torch.load(
        repo_path(str(artifacts["downstream_checkpoint"]["path"])),
        map_location="cpu",
        weights_only=True,
    )
    features = np.load(
        repo_path(str(artifacts["feature_cache"]["path"])),
        mmap_mode="r",
        allow_pickle=False,
    )
    with np.load(
        repo_path(str(artifacts["feature_metadata"]["path"])), allow_pickle=False
    ) as metadata:
        dataset_indices = metadata["dataset_indices"].astype(np.int64, copy=False)
        selected_rows = _rows(dataset_indices, indices)
        targets = metadata["targets"][selected_rows].astype(np.int64, copy=False)
    mean = checkpoint["feature_mean"].numpy().astype(np.float32, copy=False)
    scale = checkpoint["feature_scale"].numpy().astype(np.float32, copy=False)
    weight = checkpoint["head_state"]["weight"].to(device)
    bias = checkpoint["head_state"]["bias"].to(device)
    parts: list[torch.Tensor] = []
    for start in range(0, len(selected_rows), 8):
        block = np.asarray(features[selected_rows[start : start + 8]], dtype=np.float32)
        block = (block - mean) / scale
        if not np.isfinite(block).all():
            raise CampaignError("NormWear frozen standardization produced non-finite values")
        with torch.inference_mode():
            parts.append(F.linear(torch.from_numpy(block).to(device), weight, bias).cpu())
    logits = torch.cat(parts, dim=0)
    return {
        "dataset_index": np.asarray(indices, dtype=np.int64),
        "identity": np.asarray(
            [f"{job['task']}|dataset_index={value}" for value in indices], dtype=str
        ),
        "logits": logits.numpy().astype(np.float32, copy=False),
        "prediction": logits.argmax(dim=1).numpy().astype(np.int64, copy=False),
        "target": targets,
    }


def _efrm_inference(
    job: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    indices: Sequence[int],
    device: torch.device,
) -> dict[str, np.ndarray]:
    checkpoint = torch.load(
        repo_path(str(artifacts["downstream_checkpoint"]["path"])),
        map_location="cpu",
        weights_only=True,
    )
    with np.load(
        repo_path(str(artifacts["feature_cache"]["path"])), allow_pickle=False
    ) as payload:
        dataset_indices = payload["dataset_indices"].astype(np.int64, copy=False)
        selected_rows = _rows(dataset_indices, indices)
        features = payload["features"][selected_rows].astype(np.float32, copy=False)
        targets = payload["targets"][selected_rows]
        valid = payload["target_valid_mask"][selected_rows].astype(bool, copy=False)
    state = checkpoint["probe_state"]
    values = torch.from_numpy(features).to(device)
    with torch.inference_mode():
        normalized = F.layer_norm(
            values,
            (int(checkpoint["embedding_dim"]),),
            state["norm.weight"].to(device),
            state["norm.bias"].to(device),
        )
        output = F.linear(
            normalized,
            state["head.weight"].to(device),
            state["head.bias"].to(device),
        )
        if int(checkpoint["target_length"]) > 1:
            output = output.reshape(
                -1, int(checkpoint["output_dim"]), int(checkpoint["target_length"])
            )
    prediction = output.float().cpu().numpy()
    if checkpoint["task_type"] == "classification":
        return {
            "dataset_index": np.asarray(indices, dtype=np.int64),
            "identity": np.asarray(
                [f"{job['task']}|dataset_index={value}" for value in indices], dtype=str
            ),
            "logits": prediction,
            "prediction": prediction.argmax(axis=1).astype(np.int64, copy=False),
            "target": targets.astype(np.int64, copy=False),
        }
    center = checkpoint["target_center"].numpy().astype(np.float32, copy=False)
    scale = checkpoint["target_scale"].numpy().astype(np.float32, copy=False)
    native = prediction * scale[None, :, None] + center[None, :, None]
    return {
        "dataset_index": np.asarray(indices, dtype=np.int64),
        "identity": np.asarray(
            [f"{job['task']}|dataset_index={value}" for value in indices], dtype=str
        ),
        "prediction": native.astype(np.float32, copy=False),
        "target": targets.astype(np.float32, copy=False),
        "target_valid_mask": valid,
    }


def _brainfusion_inference(
    job: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    indices: Sequence[int],
    device: torch.device,
) -> dict[str, np.ndarray]:
    adapter_root = (
        REPO_ROOT
        / "comparative_methods/BrainFusion-NVC-CSP-Stacking/adapters"
    )
    if str(adapter_root) not in sys.path:
        sys.path.insert(0, str(adapter_root))
    from brainfusion_gpu.pipeline import BrainFusionFoldPipeline

    payload = torch.load(
        repo_path(str(artifacts["feature_cache"]["path"])),
        map_location="cpu",
        weights_only=True,
    )
    dataset_indices = payload["dataset_indices"].numpy().astype(np.int64, copy=False)
    selected_rows = _rows(dataset_indices, indices)
    rows_tensor = torch.from_numpy(selected_rows)
    checkpoint_dir = repo_path(str(artifacts["pipeline_manifest"]["path"])).parent
    pipeline = BrainFusionFoldPipeline.load(checkpoint_dir, device=device)
    tensors = tuple(
        payload[name].index_select(0, rows_tensor) for name in ("eeg", "hbo", "hbr")
    )
    predictions = np.asarray(pipeline.predict(*tensors), dtype=np.int64)
    decisions = np.asarray(pipeline.decision_function(*tensors), dtype=np.float32)
    targets = payload["targets"].index_select(0, rows_tensor).numpy().astype(np.int64)
    return {
        "dataset_index": np.asarray(indices, dtype=np.int64),
        "identity": np.asarray(
            [f"{job['task']}|dataset_index={value}" for value in indices], dtype=str
        ),
        "decision_score": decisions,
        "prediction": predictions,
        "target": targets,
    }


def _infer(
    job: Mapping[str, Any], indices: Sequence[int], device: torch.device
) -> dict[str, np.ndarray]:
    artifacts = artifact_map(job)
    kind = str(job["worker_kind"])
    if kind == "linear_npz":
        return _linear_inference(job, artifacts, indices, device)
    if kind == "normwear_memmap":
        return _normwear_inference(job, artifacts, indices, device)
    if kind == "efrm_npz":
        return _efrm_inference(job, artifacts, indices, device)
    if kind == "brainfusion_pipeline":
        return _brainfusion_inference(job, artifacts, indices, device)
    raise CampaignError(f"unsupported worker kind: {kind}")


def _configure_determinism(seed: int) -> dict[str, Any]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "seed": seed,
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def _validate_predictions(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_count: int,
    expected_indices: Sequence[int] | None = None,
    job: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    required = {"schema_version", "dataset_index", "identity", "prediction", "target"}
    missing = required - set(arrays)
    if missing:
        raise CampaignError(f"prediction payload lacks fields: {sorted(missing)}")
    if any(len(value) != expected_count for value in arrays.values()):
        raise CampaignError("prediction payload arrays have inconsistent coverage")
    identities = arrays["identity"].astype(str).tolist()
    if len(identities) != len(set(identities)):
        raise CampaignError("prediction payload contains duplicate identities")
    if not np.all(arrays["schema_version"].astype(str) == "joint_protected_predictions_v1"):
        raise CampaignError("prediction payload schema version differs")
    if arrays["dataset_index"].dtype.kind not in "iu" or arrays["dataset_index"].ndim != 1:
        raise CampaignError("prediction dataset identity schema differs")
    if expected_indices is not None:
        expected_array = np.asarray(expected_indices, dtype=np.int64)
        if not np.array_equal(arrays["dataset_index"].astype(np.int64), expected_array):
            raise CampaignError("prediction payload coverage differs from the authorized view")
        if job is not None:
            expected_identity = [
                f"{job['task']}|dataset_index={int(value)}" for value in expected_array
            ]
            if identities != expected_identity:
                raise CampaignError("prediction identities differ from the authorized view")
    prediction = arrays["prediction"]
    target = arrays["target"]
    if job is not None and job.get("metric_target") == "native_coordinate_masked_ccc":
        mask = arrays.get("target_valid_mask")
        if (
            mask is None
            or mask.dtype.kind != "b"
            or prediction.ndim != 3
            or prediction.shape != target.shape
            or mask.shape != target.shape
            or not bool(mask.any())
            or not np.isfinite(prediction).all()
            or not np.isfinite(target[mask]).all()
        ):
            raise CampaignError("REFED prediction/mask shape or finite contract differs")
    elif job is not None:
        if (
            prediction.ndim != 1
            or target.ndim != 1
            or prediction.dtype.kind not in "iu"
            or target.dtype.kind not in "iu"
        ):
            raise CampaignError("classification prediction schema differs")
        scores = [arrays[name] for name in ("logits", "decision_score") if name in arrays]
        if len(scores) != 1 or scores[0].shape[0] != expected_count:
            raise CampaignError("classification score schema differs")
    finite_arrays = [
        value
        for name, value in arrays.items()
        if value.dtype.kind in "fciu" and name != "target"
    ]
    if any(not np.isfinite(value).all() for value in finite_arrays):
        raise CampaignError("prediction payload contains non-finite numeric values")
    return {
        "schema": "protected_job_audit_report_v1",
        "status": "pass",
        "sample_count": expected_count,
        "identity_count": len(identities),
        "unique_identity_count": len(set(identities)),
        "array_count": len(arrays),
        "payload_schema_sha256": stable_hash(
            [
                {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
                for name, value in sorted(arrays.items())
            ]
        ),
        "all_numeric_finite": True,
        "performance_computed": False,
    }


def _save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidate_path = args.candidate.resolve()
    candidate, candidate_sha256 = verify_candidate_file(
        candidate_path, verify_artifacts=False
    )
    if args.expected_candidate_sha256 is not None and candidate_sha256 != args.expected_candidate_sha256:
        raise CampaignError("candidate differs from the controller-pinned SHA-256")
    environment_sha256 = verify_runtime_environment(candidate)
    jobs = index_jobs(candidate)
    if args.job_id not in jobs:
        raise CampaignError(f"job is outside the frozen candidate: {args.job_id}")
    job = jobs[args.job_id]
    if args.surface == "protected":
        if args.authorization is None:
            raise CampaignError("protected surface requires an authorization manifest")
        if args.audit_log is None:
            raise CampaignError("protected surface requires an append-only audit log")
        authorization, authorization_sha256 = verify_authorization(
            args.authorization.resolve(),
            candidate=candidate,
            candidate_sha256=candidate_sha256,
        )
        if (
            args.expected_authorization_sha256 is None
            or authorization_sha256 != args.expected_authorization_sha256
        ):
            raise CampaignError("authorization differs from the controller-pinned SHA-256")
    else:
        authorization = None
        authorization_sha256 = "not_applicable_public_shadow"
    artifacts = artifact_map(job)
    for artifact in artifacts.values():
        verify_file(
            str(artifact["path"]),
            str(artifact["sha256"]),
            label=f"frozen {artifact['role']}",
        )

    determinism = _configure_determinism(int(job["seed"]))
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise CampaignError("CUDA worker requested but CUDA is unavailable")
        torch.cuda.set_device(device)
    device_uuid = _device_uuid(device)
    lane = next(
        (
            row
            for row in (candidate.get("lane_manifest") or {}).get("value", {}).get("assignments", [])
            if row.get("method_slug") == job["method_slug"]
        ),
        None,
    )
    if args.surface == "protected":
        if lane is None:
            raise CampaignError("candidate has no frozen lane assignment for this method")
        if device_uuid != lane.get("gpu_uuid"):
            raise CampaignError("worker device UUID differs from frozen lane assignment")

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"completed output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.attempt{args.attempt}.{os.getpid()}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    started_at = utc_now()
    started = time.perf_counter()
    write_json_atomic(
        temporary / "status.json",
        {
            "schema": JOB_SCHEMA,
            "job_id": args.job_id,
            "attempt": args.attempt,
            "status": "RUNNING",
            "surface": args.surface,
            "protected_test_opened": False,
            "candidate_sha256": candidate_sha256,
            "authorization_sha256": authorization_sha256,
            "started_at": started_at,
        },
    )

    protected_opened = args.surface == "protected"
    if protected_opened:
        # Conservatively record access before the first protected-manifest open.
        write_json_atomic(
            temporary / "status.json",
            {
                "schema": JOB_SCHEMA,
                "job_id": args.job_id,
                "attempt": args.attempt,
                "status": "RUNNING",
                "surface": args.surface,
                "protected_test_opened": True,
                "candidate_sha256": candidate_sha256,
                "authorization_sha256": authorization_sha256,
                "started_at": started_at,
            },
        )
        append_jsonl(
            args.audit_log.resolve(),
            {
                "event": "PROTECTED_VIEW_OPEN",
                "job_id": args.job_id,
                "attempt": args.attempt,
                "candidate_sha256": candidate_sha256,
                "authorization_sha256": authorization_sha256,
                "device_uuid": device_uuid,
                "at": utc_now(),
            },
        )
    indices, input_path, input_sha256 = _load_surface_indices(
        job, artifacts, surface=args.surface
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    arrays = _infer(job, indices, device)
    arrays["schema_version"] = np.full(
        len(indices), "joint_protected_predictions_v1", dtype="<U30"
    )
    audit = _validate_predictions(
        arrays, expected_count=len(indices), expected_indices=indices, job=job
    )
    prediction_name = (
        "protected_predictions.npz" if protected_opened else "shadow_predictions.npz"
    )
    prediction_path = temporary / prediction_name
    _save_npz(prediction_path, arrays)
    job_manifest = {
        "schema": "protected_job_manifest_v1",
        "campaign_id": candidate["campaign_id"],
        "candidate_path": portable_path(candidate_path),
        "candidate_sha256": candidate_sha256,
        "authorization_sha256": authorization_sha256,
        "job_id": args.job_id,
        "method_id": job["method_id"],
        "task": job["task"],
        "outer_fold": job["outer_fold"],
        "seed": job["seed"],
        "attempt": args.attempt,
        "surface": args.surface,
        "device": str(device),
        "device_uuid": device_uuid,
        "input_manifest_sha256": input_sha256,
        "input_contract_sha256": job["input_contract"]["sha256"],
        "frozen_inference_contract_sha256": stable_hash(
            job["frozen_inference_contract"]
        ),
        "environment_sha256": environment_sha256,
        "determinism_sha256": stable_hash(determinism),
        "artifact_sha256": {
            role: value["sha256"] for role, value in sorted(artifacts.items())
        },
        "performance_computed": False,
        "protected_test_opened": protected_opened,
    }
    write_json_atomic(temporary / "job_manifest.json", job_manifest)
    audit.update(
        {
            "job_id": args.job_id,
            "surface": args.surface,
            "protected_test_opened": protected_opened,
        }
    )
    write_json_atomic(temporary / "audit_report.json", audit)
    status = {
        "schema": JOB_SCHEMA,
        "job_id": args.job_id,
        "attempt": args.attempt,
        "status": "COMPLETED",
        "surface": args.surface,
        "protected_test_opened": protected_opened,
        "candidate_sha256": candidate_sha256,
        "authorization_sha256": authorization_sha256,
        "started_at": started_at,
        "completed_at": utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "device_uuid": device_uuid,
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "cuda_peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
        "failure_code": None,
        "performance_computed": False,
    }
    write_json_atomic(temporary / "status.json", status)
    checksummed = [
        "job_manifest.json",
        prediction_name,
        "audit_report.json",
        "status.json",
    ]
    checksums = {
        "schema": "protected_job_artifact_checksums_v1",
        "job_id": args.job_id,
        "files": {name: sha256_file(temporary / name) for name in checksummed},
    }
    write_json_atomic(temporary / "artifact_checksums.json", checksums)
    os.replace(temporary, output_dir)
    if protected_opened:
        append_jsonl(
            args.audit_log.resolve(),
            {
                "event": "JOB_ATOMIC_COMMIT",
                "job_id": args.job_id,
                "attempt": args.attempt,
                "candidate_sha256": candidate_sha256,
                "authorization_sha256": authorization_sha256,
                "at": utc_now(),
            },
        )
    print(
        json.dumps(
            {
                "status": "COMPLETED",
                "job_id": args.job_id,
                "attempt": args.attempt,
                "surface": args.surface,
                "sample_count": len(indices),
                "performance_computed": False,
                "protected_test_opened": protected_opened,
                "output": portable_path(output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--surface", choices=("shadow", "protected"), default="shadow")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256")
    parser.add_argument("--expected-authorization-sha256")
    parser.add_argument("--audit-log", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run(args)
        return 0
    except Exception as exc:  # sanitized fail-closed boundary
        output_dir = args.output_dir.resolve()
        prefix = f".{output_dir.name}.attempt{args.attempt}."
        temporary = None
        protected_opened = False
        try:
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            temporary = next(
                (
                    path
                    for path in output_dir.parent.glob(f"{prefix}*.tmp")
                    if path.is_dir() and path.parent == output_dir.parent
                ),
                None,
            )
            if temporary is not None and (temporary / "status.json").is_file():
                protected_opened = bool(
                    read_json(temporary / "status.json").get("protected_test_opened", False)
                )
            if temporary is None:
                temporary = output_dir.parent / (
                    f"{prefix}{os.getpid()}.failure.tmp"
                )
                temporary.mkdir()
        except Exception:
            temporary = None
        technical = isinstance(
            exc,
            (OSError, subprocess.CalledProcessError, torch.cuda.OutOfMemoryError),
        ) or (
            isinstance(exc, RuntimeError)
            and any(token in str(exc).lower() for token in ("cuda", "cudnn", "gpu"))
        )
        failure_code = "FAILED_TECHNICAL" if technical else "FAILED_INVALID_OUTPUT"
        if temporary is not None:
            try:
                write_json_atomic(
                    temporary / "status.json",
                    {
                        "schema": JOB_SCHEMA,
                        "job_id": args.job_id,
                        "attempt": args.attempt,
                        "status": failure_code,
                        "surface": args.surface,
                        "protected_test_opened": protected_opened,
                        "candidate_sha256": args.expected_candidate_sha256,
                        "authorization_sha256": args.expected_authorization_sha256,
                        "failure_code": failure_code,
                        "performance_computed": False,
                        "completed_at": utc_now(),
                    },
                )
                quarantine_root = output_dir.parent / "quarantine"
                quarantine_root.mkdir(parents=True, exist_ok=True)
                suffix = utc_now().replace(":", "").replace("+", "_")
                quarantine = quarantine_root / (
                    f"{output_dir.name}.attempt{args.attempt}.{suffix}"
                )
                os.replace(temporary, quarantine)
            except Exception:
                pass
        if args.surface == "protected" and args.audit_log is not None:
            try:
                append_jsonl(
                    args.audit_log.resolve(),
                    {
                        "event": "JOB_FAILURE",
                        "job_id": args.job_id,
                        "attempt": args.attempt,
                        "failure_code": failure_code,
                        "protected_test_opened": protected_opened,
                        "at": utc_now(),
                    },
                )
            except Exception:
                pass
        print(
            json.dumps(
                {
                    "status": failure_code,
                    "job_id": args.job_id,
                    "attempt": args.attempt,
                    "surface": args.surface,
                    "performance_computed": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
