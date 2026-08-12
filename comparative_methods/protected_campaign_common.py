"""Shared, metric-blind primitives for the joint protected campaign.

This module deliberately contains no metric implementation.  Code imported by
the controller and worker may validate identities and artifacts, but only the
sealed aggregator is allowed to calculate protected performance numbers.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import tempfile
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_SCHEMA = "joint_protected_campaign_release_candidate_v1"
AUTHORIZATION_SCHEMA = "joint_protected_campaign_authorization_v1"
UNBLIND_SCHEMA = "joint_protected_campaign_unblind_v1"
JOB_SCHEMA = "joint_protected_campaign_job_v1"

ALLOWED_JOB_TERMINALS = {"COMPLETED", "FAILED_TECHNICAL", "FAILED_INVALID_OUTPUT"}
SUCCESS_TERMINAL = "COMPLETED"
FOLDS = frozenset(range(5))
SEEDS = frozenset({17, 42, 73})
METHOD_SLUGS = frozenset(
    {"biot", "cbramod", "reve", "efrm", "normwear", "brainfusion"}
)
CAMPAIGN_ID = "joint-comparison-protected-20260812-v1"
METHOD_IDENTITIES = {
    "biot": "biot",
    "cbramod": "cbramod",
    "reve": "reve",
    "efrm": "efrm_sync_200_10_variable_channel_v1",
    "normwear": "normwear_eeg_fnirs_adapted",
    "brainfusion": "brainfusion_nvc_csp_stacking_reimplementation",
}
ALL_TASKS = frozenset(
    {
        "motor_imagery",
        "mental_arithmetic",
        "wg",
        "nback",
        "dsr",
        "visual",
        "refed_regression",
    }
)
SUPPORTED_TASKS = {
    "biot": ALL_TASKS - {"refed_regression"},
    "cbramod": ALL_TASKS - {"refed_regression"},
    "reve": ALL_TASKS - {"refed_regression"},
    "efrm": ALL_TASKS,
    "normwear": ALL_TASKS - {"refed_regression"},
    "brainfusion": ALL_TASKS - {"dsr", "refed_regression"},
}
JOB_ID_PATTERN = re.compile(
    r"^(biot|cbramod|reve|efrm|normwear|brainfusion)__"
    r"([a-z0-9_]+)__outer([0-4])__seed(17|42|73)$"
)
EXPECTED_OUTPUTS = frozenset(
    {
        "job_manifest.json",
        "status.json",
        "protected_predictions.npz",
        "artifact_checksums.json",
        "audit_report.json",
    }
)
ARTIFACT_ROLES = {
    "linear_npz": frozenset(
        {
            "public_run_manifest",
            "public_split_manifest",
            "downstream_checkpoint",
            "feature_cache",
            "feature_cache_manifest",
        }
    ),
    "efrm_npz": frozenset(
        {
            "public_run_manifest",
            "public_split_manifest",
            "downstream_checkpoint",
            "feature_cache",
            "feature_cache_manifest",
        }
    ),
    "normwear_memmap": frozenset(
        {
            "public_run_manifest",
            "public_split_manifest",
            "downstream_checkpoint",
            "feature_cache",
            "feature_metadata",
            "feature_cache_manifest",
        }
    ),
    "brainfusion_pipeline": frozenset(
        {
            "public_run_manifest",
            "public_split_manifest",
            "pipeline_manifest",
            "pipeline_feature_state",
            "pipeline_stacking",
            "feature_cache",
            "feature_cache_manifest",
        }
    ),
}


class CampaignError(RuntimeError):
    """Raised when a campaign boundary or immutable identity is invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def environment_fingerprint() -> dict[str, Any]:
    """Return the execution environment identity frozen into the candidate."""
    import torch

    packages: dict[str, str] = {}
    for name in ("numpy", "torch", "scikit-learn", "scipy", "pyyaml", "joblib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    value = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    return {**value, "sha256": stable_hash(value)}


def verify_runtime_environment(candidate: Mapping[str, Any]) -> str:
    expected = candidate.get("environment", {})
    if not isinstance(expected, Mapping):
        raise CampaignError("candidate environment contract is absent")
    without_hash = {key: value for key, value in expected.items() if key != "sha256"}
    if stable_hash(without_hash) != expected.get("sha256"):
        raise CampaignError("candidate environment identity drifted")
    observed = environment_fingerprint()
    if observed != expected:
        raise CampaignError("runtime environment differs from the frozen candidate")
    return str(observed["sha256"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CampaignError(f"JSON object required: {path}")
    return value


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary_path = Path(temporary)
        if temporary_path.exists():
            temporary_path.unlink()


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o640)
    try:
        os.write(descriptor, (line + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_file(path_value: str, expected_sha256: str, *, label: str) -> Path:
    path = repo_path(path_value)
    if not path.is_file():
        raise CampaignError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise CampaignError(
            f"{label} SHA-256 drifted: expected {expected_sha256}, observed {actual}"
        )
    return path


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _safe_repo_relative(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return False
    try:
        (REPO_ROOT / path).resolve().relative_to(REPO_ROOT)
    except ValueError:
        return False
    return True


def _safe_public_shadow_evidence(value: Any) -> bool:
    if not _safe_repo_relative(value):
        return False
    evidence_root = (
        REPO_ROOT / "comparative_methods/evidence/protected_campaign"
    ).resolve()
    try:
        relative = (REPO_ROOT / str(value)).resolve().relative_to(evidence_root)
    except ValueError:
        return False
    return bool(relative.parts) and relative.parts[0] in {
        "shadow_cpu_pass_v1",
        "shadow_cpu_pass_v1_repeat",
        "shadow_gpu_v1",
    }


def _validate_lane_manifest(
    value: Mapping[str, Any], *, campaign: Mapping[str, Any], jobs: list[Mapping[str, Any]]
) -> None:
    if value.get("schema") != "joint_protected_campaign_lane_manifest_v1":
        raise CampaignError("unexpected frozen lane schema")
    if value.get("campaign_id") != campaign.get("campaign_id"):
        raise CampaignError("frozen lane campaign identity differs")
    if value.get("status") != "pass" or value.get("protected_test_opened") is not False:
        raise CampaignError("frozen lane is not a protected-closed pass")
    pre_lane_sha256 = value.get("candidate_sha256_before_lane_freeze")
    if (
        not _is_sha256(pre_lane_sha256)
        or campaign.get("pre_lane_candidate_sha256") != pre_lane_sha256
    ):
        raise CampaignError("frozen lane is not bound to this pre-lane candidate")
    assignments = value.get("assignments", [])
    if not isinstance(assignments, list) or value.get("assignment_sha256") != stable_hash(assignments):
        raise CampaignError("frozen lane assignment identity drifted")
    methods = {str(row["method_slug"]) for row in jobs}
    assignment_methods = [str(row.get("method_slug")) for row in assignments]
    if methods != METHOD_SLUGS or set(assignment_methods) != methods or len(assignment_methods) != len(methods):
        raise CampaignError("frozen lanes must cover each campaign method exactly once")
    snapshots = value.get("gpu_snapshot", [])
    if not isinstance(snapshots, list) or len(snapshots) < 2:
        raise CampaignError("frozen lanes require at least two benchmarked GPUs")
    if value.get("torch_cuda") != campaign.get("environment", {}).get("torch_cuda"):
        raise CampaignError("frozen GPU lane CUDA runtime differs from candidate")
    by_uuid = {str(row.get("uuid")): row for row in snapshots}
    if len(by_uuid) != len(snapshots) or "" in by_uuid:
        raise CampaignError("frozen GPU UUIDs are missing or duplicated")
    indices = [int(row.get("index", -1)) for row in snapshots]
    if len(indices) != len(set(indices)) or any(index < 0 for index in indices):
        raise CampaignError("frozen GPU indices are missing or duplicated")
    expected_counts: dict[str, int] = {}
    for job in jobs:
        slug = str(job["method_slug"])
        expected_counts[slug] = expected_counts.get(slug, 0) + 1
    for row in assignments:
        uuid = str(row.get("gpu_uuid", ""))
        snapshot = by_uuid.get(uuid)
        if snapshot is None or int(row.get("gpu_index", -1)) != int(snapshot["index"]):
            raise CampaignError("lane GPU index/UUID does not match its frozen snapshot")
        if int(row.get("job_count", -1)) != expected_counts[str(row["method_slug"])]:
            raise CampaignError("frozen lane job count differs from the candidate")
    backup = value.get("backup_gpu_uuids", [])
    if set(map(str, backup)) != set(by_uuid) or len(backup) != len(by_uuid):
        raise CampaignError("backup GPU list differs from equivalence-tested GPUs")
    equivalence = value.get("gpu_equivalence", {})
    if set(equivalence) != methods or any(
        row.get("equivalent") is not True for row in equivalence.values()
    ):
        raise CampaignError("not every method passed frozen GPU equivalence")
    cpu_repeatability = value.get("cpu_repeatability", {})
    if set(cpu_repeatability) != methods or any(
        row.get("bitwise_equal") is not True for row in cpu_repeatability.values()
    ):
        raise CampaignError("not every method passed two-pass CPU repeatability")
    import numpy as np

    shadow_jobs = {
        str(job["method_slug"]): job
        for job in jobs
        if job["task"] == "motor_imagery"
        and int(job["outer_fold"]) == 0
        and int(job["seed"]) == 17
    }

    def validate_shadow_manifests(
        runtime_paths: list[Path], *, method: str, expected_device_uuid: str
    ) -> None:
        job = shadow_jobs[method]
        manifests = [path for path in runtime_paths if path.name == "job_manifest.json"]
        if not manifests:
            raise CampaignError("shadow job manifest evidence is absent")
        expected_artifacts = {
            str(artifact["role"]): artifact["sha256"]
            for artifact in sorted(job["artifacts"], key=lambda row: str(row["role"]))
        }
        determinism = {
            "seed": int(job["seed"]),
            "torch_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cublas_workspace_config": ":4096:8",
        }
        for manifest_path in manifests:
            manifest = read_json(manifest_path)
            if (
                manifest.get("candidate_sha256") != pre_lane_sha256
                or manifest.get("authorization_sha256")
                != "not_applicable_public_shadow"
                or manifest.get("job_id") != job["job_id"]
                or manifest.get("method_id") != job["method_id"]
                or manifest.get("task") != job["task"]
                or int(manifest.get("outer_fold", -1)) != int(job["outer_fold"])
                or int(manifest.get("seed", -1)) != int(job["seed"])
                or manifest.get("surface") != "shadow"
                or manifest.get("device_uuid") != expected_device_uuid
                or manifest.get("input_contract_sha256")
                != job["input_contract"]["sha256"]
                or manifest.get("frozen_inference_contract_sha256")
                != stable_hash(job["frozen_inference_contract"])
                or manifest.get("environment_sha256")
                != campaign["environment"]["sha256"]
                or manifest.get("determinism_sha256") != stable_hash(determinism)
                or manifest.get("artifact_sha256") != expected_artifacts
                or manifest.get("protected_test_opened") is not False
                or manifest.get("performance_computed") is not False
            ):
                raise CampaignError("shadow job manifest is not candidate-bound")

    for method, row in cpu_repeatability.items():
        if (
            row.get("candidate_sha256_before_lane_freeze")
            != value.get("candidate_sha256_before_lane_freeze")
            or row.get("runtime_json_redaction_pass") is not True
        ):
            raise CampaignError("CPU repeatability evidence is not candidate-bound")
        for field in ("first_prediction_path", "second_prediction_path"):
            if not _safe_public_shadow_evidence(row.get(field)):
                raise CampaignError("CPU repeatability path is outside public evidence")
        first = verify_file(
            str(row.get("first_prediction_path")),
            str(row.get("first_prediction_sha256")),
            label=f"first CPU shadow for {method}",
        )
        second = verify_file(
            str(row.get("second_prediction_path")),
            str(row.get("second_prediction_sha256")),
            label=f"second CPU shadow for {method}",
        )
        with np.load(first, allow_pickle=False) as left, np.load(
            second, allow_pickle=False
        ) as right:
            if left.files != right.files or any(
                left[name].dtype != right[name].dtype
                or left[name].shape != right[name].shape
                or not np.array_equal(left[name], right[name])
                for name in left.files
            ):
                raise CampaignError("retained CPU shadow evidence is not bitwise equal")
        runtime_files = row.get("runtime_json_files", [])
        if not isinstance(runtime_files, list) or not runtime_files:
            raise CampaignError("CPU runtime log evidence is absent")
        runtime_paths: list[Path] = []
        for descriptor in runtime_files:
            if not _safe_public_shadow_evidence(descriptor.get("path")):
                raise CampaignError("CPU runtime JSON path is outside public evidence")
            runtime_path = verify_file(
                str(descriptor.get("path")),
                str(descriptor.get("sha256")),
                label=f"CPU runtime JSON for {method}",
            )
            runtime_paths.append(runtime_path)
            serialized = runtime_path.read_text(encoding="utf-8").lower()
            if any(
                token in serialized
                for token in ("target", "logits", "metric", "confusion", "sample_id")
            ):
                raise CampaignError("retained CPU runtime JSON violates redaction")
        validate_shadow_manifests(
            runtime_paths, method=method, expected_device_uuid="CPU"
        )
    benchmarks = value.get("benchmarks", {})
    if not isinstance(benchmarks, Mapping) or set(benchmarks) != methods:
        raise CampaignError("GPU benchmark coverage differs from campaign methods")
    snapshot_uuids = set(by_uuid)
    for method, rows in benchmarks.items():
        if (
            not isinstance(rows, list)
            or len(rows) != len(snapshot_uuids)
            or {str(row.get("gpu_uuid")) for row in rows} != snapshot_uuids
        ):
            raise CampaignError("GPU benchmark does not cover every frozen GPU")
        predictions: list[Path] = []
        for row in rows:
            if (
                row.get("candidate_sha256_before_lane_freeze") != pre_lane_sha256
                or row.get("job_id")
                != f"{method}__motor_imagery__outer0__seed17"
                or not _safe_public_shadow_evidence(row.get("prediction_path"))
            ):
                raise CampaignError("GPU benchmark identity is not candidate-bound")
            prediction = verify_file(
                str(row.get("prediction_path")),
                str(row.get("prediction_sha256")),
                label=f"GPU shadow for {method}",
            )
            predictions.append(prediction)
            runtime_files = row.get("runtime_json_files", [])
            if not isinstance(runtime_files, list) or not runtime_files:
                raise CampaignError("GPU runtime log evidence is absent")
            runtime_paths: list[Path] = []
            for descriptor in runtime_files:
                if not _safe_public_shadow_evidence(descriptor.get("path")):
                    raise CampaignError("GPU runtime JSON path is outside public evidence")
                runtime_path = verify_file(
                    str(descriptor.get("path")),
                    str(descriptor.get("sha256")),
                    label=f"GPU runtime JSON for {method}",
                )
                runtime_paths.append(runtime_path)
                serialized = runtime_path.read_text(encoding="utf-8").lower()
                if any(
                    token in serialized
                    for token in ("target", "logits", "metric", "confusion", "sample_id")
                ):
                    raise CampaignError("retained GPU runtime JSON violates redaction")
            validate_shadow_manifests(
                runtime_paths,
                method=method,
                expected_device_uuid=str(row["gpu_uuid"]),
            )
        with np.load(predictions[0], allow_pickle=False) as left, np.load(
            predictions[1], allow_pickle=False
        ) as right:
            if left.files != right.files:
                raise CampaignError("GPU shadow fields differ")
            for name in left.files:
                if left[name].dtype != right[name].dtype or left[name].shape != right[name].shape:
                    raise CampaignError("GPU shadow array contract differs")
                if left[name].dtype.kind in "f":
                    equivalent = np.allclose(
                        left[name], right[name], rtol=1e-5, atol=1e-6
                    )
                else:
                    equivalent = np.array_equal(left[name], right[name])
                if not equivalent:
                    raise CampaignError("retained GPU shadow evidence is not equivalent")
    method_benchmarks = value.get("method_benchmarks", {})
    if set(method_benchmarks) != methods or any(
        float(row.get("median_wall_seconds", 0.0)) <= 0.0
        or int(row.get("peak_cuda_allocated_bytes", -1)) < 0
        or int(row.get("peak_cuda_reserved_bytes", -1)) < 0
        for row in method_benchmarks.values()
    ):
        raise CampaignError("frozen method timing/memory benchmarks are incomplete")


def verify_candidate_file(
    path: Path, *, verify_artifacts: bool = True
) -> tuple[dict[str, Any], str]:
    candidate = read_json(path)
    if candidate.get("schema") != CAMPAIGN_SCHEMA:
        raise CampaignError(f"unexpected candidate schema: {candidate.get('schema')!r}")
    if candidate.get("campaign_id") != CAMPAIGN_ID:
        raise CampaignError("unexpected campaign identity")
    digest = sha256_file(path)
    if candidate.get("protected_evaluation_authorized") is not False:
        raise CampaignError("release candidate must never authorize protected evaluation")
    if candidate.get("protected_test_opened") is not False:
        raise CampaignError("release candidate reports protected access")
    has_lane = candidate.get("lane_manifest") is not None
    if (
        (has_lane and candidate.get("state") != "REVIEWED")
        or (not has_lane and candidate.get("state") != "DRAFT")
        or (
            candidate.get("orr_decision")
            != ("PENDING_DUAL_GO" if has_lane else "NO_GO_PENDING_SHADOW_LANE")
        )
        or (
            has_lane
            and not _is_sha256(candidate.get("pre_lane_candidate_sha256"))
        )
        or (
            not has_lane
            and candidate.get("pre_lane_candidate_sha256") is not None
        )
    ):
        raise CampaignError("candidate state/lane transition identity drifted")
    cells = candidate.get("cells", [])
    jobs = candidate.get("jobs", [])
    if len(cells) != 42 or len(jobs) != 540:
        raise CampaignError(
            f"campaign matrix drifted: cells={len(cells)}, jobs={len(jobs)}"
        )
    if not all(isinstance(row, Mapping) for row in cells + jobs):
        raise CampaignError("candidate cells/jobs must be JSON objects")
    job_ids = [str(row.get("job_id")) for row in jobs]
    if len(job_ids) != len(set(job_ids)):
        raise CampaignError("candidate contains duplicate job IDs")
    if any(JOB_ID_PATTERN.fullmatch(job_id) is None for job_id in job_ids):
        raise CampaignError("candidate contains an unsafe or non-canonical job ID")
    dispositions = candidate.get("disposition_counts", {})
    expected = {
        "direct": 34,
        "overlap": 2,
        "supported": 36,
        "unsupported": 6,
        "jobs": 540,
    }
    if dispositions != expected:
        raise CampaignError(f"candidate disposition counts drifted: {dispositions}")
    cell_keys = [(str(row.get("method_id")), str(row.get("task_id"))) for row in cells]
    if len(cell_keys) != len(set(cell_keys)):
        raise CampaignError("candidate contains duplicate method-task cells")
    expected_cell_keys = {
        (method_id, task)
        for method_id in METHOD_IDENTITIES.values()
        for task in ALL_TASKS
    }
    if set(cell_keys) != expected_cell_keys:
        raise CampaignError("candidate method-task cell identities drifted")
    cell_lookup = {key: row for key, row in zip(cell_keys, cells)}
    observed_dispositions = {
        disposition: sum(
            row.get("campaign_disposition") == disposition for row in cells
        )
        for disposition in ("direct", "overlap", "unsupported")
    }
    if observed_dispositions != {"direct": 34, "overlap": 2, "unsupported": 6}:
        raise CampaignError("cell disposition routing differs from the frozen counts")
    overlap_cells = {
        key for key, row in cell_lookup.items() if row.get("campaign_disposition") == "overlap"
    }
    if overlap_cells != {
        ("reve", "motor_imagery"),
        ("reve", "mental_arithmetic"),
    }:
        raise CampaignError("overlap routing differs from the frozen REVE appendix")
    expected_unsupported = {
        ("biot", "refed_regression"),
        ("cbramod", "refed_regression"),
        ("reve", "refed_regression"),
        ("normwear_eeg_fnirs_adapted", "refed_regression"),
        ("brainfusion_nvc_csp_stacking_reimplementation", "dsr"),
        ("brainfusion_nvc_csp_stacking_reimplementation", "refed_regression"),
    }
    unsupported_cells = {
        key
        for key, row in cell_lookup.items()
        if row.get("campaign_disposition") == "unsupported"
    }
    if unsupported_cells != expected_unsupported:
        raise CampaignError("unsupported cell routing differs from the frozen matrix")
    split_fingerprint = candidate.get("split_fingerprint", {})
    if (
        not isinstance(split_fingerprint, Mapping)
        or not isinstance(split_fingerprint.get("entries"), list)
        or stable_hash(split_fingerprint.get("entries")) != split_fingerprint.get("sha256")
    ):
        raise CampaignError("candidate split fingerprint identity drifted")
    split_lookup = {
        (str(row.get("task")), int(row.get("outer_fold", -1))): row
        for row in split_fingerprint["entries"]
        if isinstance(row, Mapping)
    }
    if len(split_lookup) != 35:
        raise CampaignError("candidate split fingerprint coverage drifted")
    supported_products: dict[tuple[str, str], set[tuple[int, int]]] = {}
    slug_to_method: dict[str, str] = {}
    for row in jobs:
        job_id = str(row["job_id"])
        match = JOB_ID_PATTERN.fullmatch(job_id)
        assert match is not None
        slug, encoded_task, encoded_fold, encoded_seed = match.groups()
        identity = (slug, str(row.get("method_id")))
        if identity[1] != METHOD_IDENTITIES[slug] or str(row.get("task")) not in SUPPORTED_TASKS[slug]:
            raise CampaignError("job method identity or supported task routing drifted")
        previous = slug_to_method.setdefault(identity[0], identity[1])
        if previous != identity[1]:
            raise CampaignError("one method slug maps to multiple method identities")
        if (
            row.get("method_slug") != slug
            or row.get("task") != encoded_task
            or int(row.get("outer_fold", -1)) != int(encoded_fold)
            or int(row.get("seed", -1)) != int(encoded_seed)
        ):
            raise CampaignError("job ID and job identity fields differ")
        key = (str(row.get("method_id")), str(row.get("task")))
        cell = cell_lookup.get(key)
        if cell is None or cell.get("campaign_disposition") == "unsupported":
            raise CampaignError("job is absent from, or unsupported by, the cell matrix")
        if any(
            row.get(field) != cell.get(cell_field)
            for field, cell_field in (
                ("track", "track"),
                ("campaign_disposition", "campaign_disposition"),
                ("metric_target", "metric_target"),
            )
        ):
            raise CampaignError("job routing differs from its frozen cell")
        if row.get("metric_target") not in {"macro_f1", "native_coordinate_masked_ccc"}:
            raise CampaignError("job metric target is outside the frozen metric set")
        supported_products.setdefault(key, set()).add(
            (int(row["outer_fold"]), int(row["seed"]))
        )
        contract = row.get("input_contract", {})
        if not isinstance(contract, Mapping):
            raise CampaignError("job input contract is absent")
        contract_value = {key: value for key, value in contract.items() if key != "sha256"}
        required_contract = {
            "registry_sha256",
            "split_fingerprint_sha256",
            "protected_manifest_path",
            "protected_manifest_sha256",
            "protected_indices_sha256",
            "protected_sample_count",
            "dataset_id",
            "protocol",
        }
        if set(contract_value) != required_contract or contract.get("sha256") != stable_hash(contract_value):
            raise CampaignError("job input contract identity drifted")
        if (
            contract.get("registry_sha256") != candidate.get("data_registry", {}).get("sha256")
            or contract.get("split_fingerprint_sha256") != candidate.get("split_fingerprint", {}).get("sha256")
            or contract.get("protocol") != "strict_cross_subject"
            or int(contract.get("protected_sample_count", 0)) <= 0
            or not _safe_repo_relative(contract.get("protected_manifest_path"))
            or not _is_sha256(contract.get("protected_manifest_sha256"))
            or not _is_sha256(contract.get("protected_indices_sha256"))
        ):
            raise CampaignError("job input contract differs from the campaign registry")
        split = split_lookup.get((str(row["task"]), int(row["outer_fold"])))
        if split is None or any(
            contract.get(contract_field) != split.get(split_field)
            for contract_field, split_field in (
                ("dataset_id", "dataset_id"),
                ("protocol", "protocol"),
                ("protected_manifest_sha256", "protected_manifest_sha256"),
                ("protected_indices_sha256", "protected_indices_sha256"),
                ("protected_sample_count", "protected_sample_count"),
            )
        ):
            raise CampaignError("job input contract differs from its split fingerprint")
        artifacts = row.get("artifacts", [])
        if not isinstance(artifacts, list):
            raise CampaignError("job artifacts must be a list")
        roles = [str(artifact.get("role")) for artifact in artifacts]
        expected_roles = ARTIFACT_ROLES.get(str(row.get("worker_kind")))
        if expected_roles is None or set(roles) != expected_roles or len(roles) != len(expected_roles):
            raise CampaignError("job artifact roles are incomplete or duplicated")
        for artifact in artifacts:
            artifact_path = str(artifact.get("path", ""))
            if (
                not _safe_repo_relative(artifact_path)
                or any(
                    part in {"protected", "protected_campaign"}
                    for part in Path(artifact_path).parts
                )
                or not _is_sha256(artifact.get("sha256"))
                or int(artifact.get("size_bytes", 0)) <= 0
            ):
                raise CampaignError("job artifact descriptor is unsafe or incomplete")
        if frozenset(row.get("expected_outputs", [])) != EXPECTED_OUTPUTS:
            raise CampaignError("job expected output contract drifted")
    expected_product = {(fold, seed) for fold in FOLDS for seed in SEEDS}
    for key, cell in cell_lookup.items():
        disposition = str(cell.get("campaign_disposition"))
        product = supported_products.get(key, set())
        if disposition == "unsupported":
            if product or int(cell.get("job_count", -1)) != 0:
                raise CampaignError("unsupported cell contains formal jobs")
        elif product != expected_product or int(cell.get("job_count", -1)) != 15:
            raise CampaignError("supported cell lacks the exact fold-seed product")
    if candidate.get("job_matrix_sha256") != stable_hash(jobs):
        raise CampaignError("candidate job matrix identity drifted")
    if candidate.get("sta_net") != {
        "new_job_count": 0,
        "disposition": "method_native_context_reference",
    }:
        raise CampaignError("STA-Net zero-job context routing drifted")
    state_machine = candidate.get("state_machine", {})
    if state_machine != {
        "states": [
            "DRAFT",
            "REVIEWED",
            "AUTHORIZED",
            "RUNNING",
            "INCOMPLETE_TECHNICAL",
            "SEALED_COMPLETE",
            "UNBLINDED",
            "AGGREGATED",
            "RELEASED",
        ],
        "fail_closed": True,
        "authorization_and_unblind_are_separate_dual_signature_transitions": True,
    }:
        raise CampaignError("campaign state machine drifted")
    failure_policy = candidate.get("failure_policy", {})
    if (
        failure_policy.get("maximum_attempts_per_job") != 2
        or failure_policy.get("invalid_output_is_not_retryable") is not True
        or failure_policy.get("performance_based_retry_forbidden") is not True
        or failure_policy.get("attempt_2_device_policy")
        != "same_frozen_gpu_uuid_only"
        or failure_policy.get("unavailable_assigned_gpu_terminal")
        != "INCOMPLETE_TECHNICAL_requires_new_candidate_and_dual_authorization"
        or failure_policy.get("second_technical_failure_terminal")
        != "INCOMPLETE_TECHNICAL"
    ):
        raise CampaignError("campaign failure policy drifted")
    blinding_policy = candidate.get("blinding_policy", {})
    if not isinstance(blinding_policy, Mapping) or any(
        blinding_policy.get(field) is not True
        for field in (
            "operator_prediction_access_forbidden_before_unblind",
            "aggregator_requires_sealed_complete_and_dual_unblind",
        )
    ) or any(
        blinding_policy.get(field) is not False
        for field in ("worker_computes_metrics", "controller_displays_metrics")
    ):
        raise CampaignError("campaign blinding policy drifted")
    source_snapshot = candidate.get("code_snapshot", {})
    source_files = source_snapshot.get("files", [])
    source_paths = [str(row.get("path")) for row in source_files]
    if (
        stable_hash(source_files) != source_snapshot.get("sha256")
        or len(source_paths) != len(set(source_paths))
        or any(
            not _safe_repo_relative(value) or not _is_sha256(row.get("sha256"))
            for value, row in zip(source_paths, source_files)
        )
    ):
        raise CampaignError("candidate code snapshot identity drifted")
    for source in source_files:
        verify_file(
            str(source["path"]),
            str(source["sha256"]),
            label="controlled campaign source",
        )
    data_registry = candidate.get("data_registry", {})
    if not _safe_repo_relative(data_registry.get("path")):
        raise CampaignError("method-neutral registry path is unsafe")
    verify_file(
        str(data_registry["path"]),
        str(data_registry["sha256"]),
        label="method-neutral data registry",
    )
    metric_targets = candidate.get("metric_targets", {})
    if not _safe_repo_relative(metric_targets.get("path")):
        raise CampaignError("metric target path is unsafe")
    verify_file(
        str(metric_targets["path"]),
        str(metric_targets["sha256"]),
        label="metric target contract",
    )
    environment = candidate.get("environment", {})
    if not isinstance(environment, Mapping) or stable_hash(
        {key: value for key, value in environment.items() if key != "sha256"}
    ) != environment.get("sha256"):
        raise CampaignError("candidate environment identity drifted")
    lane = candidate.get("lane_manifest")
    if lane is not None:
        if not _safe_repo_relative(lane.get("path")):
            raise CampaignError("frozen lane path is unsafe")
        lane_path = verify_file(
            str(lane["path"]), str(lane["sha256"]), label="frozen lane manifest"
        )
        retained_lane = read_json(lane_path)
        if retained_lane != lane.get("value"):
            raise CampaignError("embedded lane manifest differs from its frozen file")
        _validate_lane_manifest(retained_lane, campaign=candidate, jobs=jobs)
    if verify_artifacts:
        verified: dict[str, str] = {}
        for row in jobs:
            for artifact in row.get("artifacts", []):
                artifact_path = str(artifact["path"])
                artifact_sha256 = str(artifact["sha256"])
                previous = verified.get(artifact_path)
                if previous is not None:
                    if previous != artifact_sha256:
                        raise CampaignError(
                            f"candidate assigns two hashes to artifact: {artifact_path}"
                        )
                    continue
                verify_file(
                    artifact_path,
                    artifact_sha256,
                    label=f"artifact for {row['job_id']}",
                )
                verified[artifact_path] = artifact_sha256
    return candidate, digest


def _parse_time(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CampaignError(f"invalid {field}: {value!r}") from exc
    if parsed.tzinfo is None:
        raise CampaignError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def verify_authorization(
    path: Path,
    *,
    candidate: Mapping[str, Any],
    candidate_sha256: str,
    now: datetime | None = None,
    enforce_window: bool = True,
) -> tuple[dict[str, Any], str]:
    authorization = read_json(path)
    if authorization.get("schema") != AUTHORIZATION_SCHEMA:
        raise CampaignError("unexpected authorization schema")
    if authorization.get("campaign_id") != candidate.get("campaign_id"):
        raise CampaignError("authorization campaign identity differs")
    if authorization.get("candidate_sha256") != candidate_sha256:
        raise CampaignError("authorization does not reference the exact candidate SHA-256")
    if authorization.get("protected_evaluation_authorized") is not True:
        raise CampaignError("protected evaluation is not authorized")
    if (
        candidate.get("state") != "REVIEWED"
        or candidate.get("orr_decision") != "PENDING_DUAL_GO"
        or candidate.get("lane_manifest") is None
    ):
        raise CampaignError("candidate is not lane-frozen and ready for dual GO")
    scope = authorization.get("scope", {})
    if scope != {"supported_cells": 36, "jobs": 540}:
        raise CampaignError(f"authorization scope drifted: {scope}")
    window = authorization.get("authorized_window", {})
    starts = _parse_time(str(window.get("starts_at", "")), field="starts_at")
    ends = _parse_time(str(window.get("ends_at", "")), field="ends_at")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if enforce_window and not starts <= current <= ends:
        raise CampaignError("current time is outside the authorized execution window")
    signatures = authorization.get("signatures", [])
    if not isinstance(signatures, list) or len(signatures) != 2:
        raise CampaignError("authorization requires exactly two signatures")
    roles = {row.get("role") for row in signatures if isinstance(row, dict)}
    if roles != {"protocol_owner", "run_owner"}:
        raise CampaignError("authorization requires protocol_owner and run_owner signatures")
    signer_ids = {str(row.get("signer_id", "")).strip() for row in signatures}
    if len(signer_ids) != 2 or "" in signer_ids:
        raise CampaignError("authorization signers must be two distinct named identities")
    signed_times = []
    for row in signatures:
        if row.get("attestation") != "GO" or not row.get("signed_at"):
            raise CampaignError("both authorization signatures must attest GO")
        signed_times.append(_parse_time(str(row["signed_at"]), field="signed_at"))
    if any(signed < starts or signed > ends for signed in signed_times):
        raise CampaignError("authorization signatures must be made within the authorized window")
    policy = authorization.get("technical_recovery_policy", {})
    if policy != {
        "maximum_attempts_per_job": 2,
        "one_technical_recovery_only": True,
        "performance_based_retry_forbidden": True,
        "attempt_2_device_policy": "same_frozen_gpu_uuid_only",
        "unavailable_assigned_gpu_terminal": (
            "INCOMPLETE_TECHNICAL_requires_new_candidate_and_dual_authorization"
        ),
    }:
        raise CampaignError("authorization must freeze the one-recovery/two-attempt policy")
    return authorization, sha256_file(path)


def verify_unblind(
    path: Path,
    *,
    candidate: Mapping[str, Any],
    candidate_sha256: str,
    authorization_sha256: str,
    authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = read_json(path)
    if value.get("schema") != UNBLIND_SCHEMA:
        raise CampaignError("unexpected unblind schema")
    if value.get("campaign_id") != candidate.get("campaign_id"):
        raise CampaignError("unblind campaign identity differs")
    if value.get("candidate_sha256") != candidate_sha256:
        raise CampaignError("unblind candidate SHA-256 differs")
    if value.get("authorization_sha256") != authorization_sha256:
        raise CampaignError("unblind authorization SHA-256 differs")
    if value.get("unblind_authorized") is not True:
        raise CampaignError("unblind is not authorized")
    signatures = value.get("signatures", [])
    roles = {row.get("role") for row in signatures if isinstance(row, dict)}
    if not isinstance(signatures, list) or len(signatures) != 2 or roles != {"protocol_owner", "run_owner"}:
        raise CampaignError("unblind requires the two accountable roles")
    signer_ids = {str(row.get("signer_id", "")).strip() for row in signatures}
    if len(signer_ids) != 2 or "" in signer_ids:
        raise CampaignError("unblind requires two distinct named identities")
    if authorization is not None:
        authorized_signers = {
            (str(row.get("role")), str(row.get("signer_id", "")).strip())
            for row in authorization.get("signatures", [])
            if isinstance(row, Mapping)
        }
        unblind_signers = {
            (str(row.get("role")), str(row.get("signer_id", "")).strip())
            for row in signatures
            if isinstance(row, Mapping)
        }
        if unblind_signers != authorized_signers:
            raise CampaignError("unblind signers differ from authorization signers")
    if any(row.get("attestation") != "UNBLIND" for row in signatures):
        raise CampaignError("both unblind signatures must attest UNBLIND")
    for row in signatures:
        _parse_time(str(row.get("signed_at", "")), field="unblind signed_at")
    return value


def index_jobs(candidate: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["job_id"]): dict(row) for row in candidate["jobs"]}


def artifact_map(job: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["role"]): row for row in job.get("artifacts", [])}


def hash_rows(rows: Iterable[Mapping[str, Any]]) -> str:
    return stable_hash(list(rows))
