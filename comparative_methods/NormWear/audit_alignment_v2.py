#!/usr/bin/env python3
"""Run NormWear's resumable full-public A7 production adapter replay."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import yaml


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
for import_path in (REPO_ROOT, METHOD_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from adapters.normwear import NormWearFrozenEncoder, load_verified_normwear_encoder
from alignment_data import (
    METHOD_ID,
    SUPPORTED_TASKS,
    NormWearPublicView,
    PublicInventory,
    data_branch_fingerprints,
    load_config,
    load_public_inventory,
    resolve_repo_path,
    stable_hash,
)
from audit_data_boundary_v2 import load_alignment_contract, unsupported_refed_cell
from comparative_methods.audit_adapter_alignment import audit as audit_evidence
from comparative_methods.audit_public_preflight import sha256_file


DEFAULT_CONFIG = METHOD_ROOT / "configs/alignment_v2.yaml"
DEFAULT_OUTPUT_ROOT = METHOD_ROOT / "evidence/alignment_v2"
ALIGNMENT_CONTRACT = REPO_ROOT / "comparative_methods/adapter_alignment_gate_contract_v2.yaml"
EVIDENCE_SCHEMA = "adapter_alignment_cell_evidence_v2"
CACHE_SCHEMA = "normwear_full_public_feature_cache_v2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return portable_path(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    if "protected" in {part.lower() for part in resolved.parts}:
        raise PermissionError(f"refusing protected NormWear evidence path: {resolved}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@contextmanager
def exclusive_gpu_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"NormWear alignment GPU lane is already locked: {path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started={utc_now()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def parse_tasks(values: Sequence[str]) -> tuple[str, ...]:
    tasks = tuple(str(value) for value in values) if values else SUPPORTED_TASKS
    if len(tasks) != len(set(tasks)):
        raise ValueError("NormWear audit tasks must be unique")
    unknown = sorted(set(tasks) - set(SUPPORTED_TASKS))
    if unknown:
        raise ValueError(f"unknown or unsupported NormWear audit tasks: {unknown}")
    return tasks


def method_identity(
    *, metadata: Any, config: Mapping[str, Any], config_path: Path
) -> dict[str, Any]:
    source_paths = {
        "adapter": METHOD_ROOT / "adapters/normwear.py",
        "alignment_data": METHOD_ROOT / "alignment_data.py",
        "alignment_audit": Path(__file__),
        "identity_audit": METHOD_ROOT / "audit_identity_v2.py",
        "upstream_model": METHOD_ROOT / "upstream/modules/normwear.py",
        "upstream_patch_embed": METHOD_ROOT / "upstream/modules/patch_embed.py",
        "alignment_config": config_path,
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"NormWear identity source is missing: {missing}")
    identity = jsonable(asdict(metadata))
    identity.update(
        {
            "path": portable_path(Path(metadata.path)),
            "source_file_sha256": {
                name: sha256_file(path) for name, path in source_paths.items()
            },
            "model_input_sample_rate_hz": 65.0,
            "resampling": dict(config["adapter"]["resampling"]),
            "cwt": dict(config["adapter"]["cwt"]),
            "patch_embedding": dict(config["adapter"]["patch_embedding"]),
            "execution": dict(config["adapter"]["execution"]),
            "output_layer": str(config["adapter"]["representation_layer"]),
            "pooling": {
                "patch": str(config["adapter"]["patch_pooling"]),
                "channel": str(config["adapter"]["channel_pooling"]),
            },
            "trainable_parameter_boundary": (
                "official_encoder_frozen_outer_training_linear_probe_only"
            ),
            "source_deviation": [
                "fNIRS_is_an_explicit_cross_modality_adaptation",
                "canonical_shared_measurement_coordinate_replaces_source_preprocessing",
                "explicit_polyphase_200_and_10_to_65_rate_alignment",
                "bounded_memory_channel_attention_chunking_with_complete_cls_fusion",
            ],
            "target_corpus_exposure": "none_by_declared_dataset_identity",
        }
    )
    return identity


def feature_cache_identity(
    *,
    inventory: PublicInventory,
    method: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "schema": CACHE_SCHEMA,
        "method_identity": method,
        "task": inventory.task,
        "sample_inventory_sha256": inventory.sample_inventory_sha256,
        "split_fingerprint": inventory.split_fingerprint,
        "dataset_indices_sha256": stable_hash(list(inventory.indices)),
        "delivered_channel_order": list(inventory.delivered_channel_names),
        "delivered_channel_order_sha256": stable_hash(
            list(inventory.delivered_channel_names)
        ),
        "duration_s": inventory.duration_s,
        "data_branch_sha256": data_branch_fingerprints(config),
        "feature_extraction": {
            "canonical_rates_hz": {"eeg": 200.0, "fnirs": 10.0},
            "model_rate_hz": 65.0,
            "feature_dimension": len(inventory.delivered_channel_names) * 768,
            "dtype": str(config["resources"]["feature_cache_dtype"]),
            "format": str(config["resources"]["feature_cache_format"]),
            "feature_batch_size": int(config["resources"]["feature_batch_size"]),
            "channel_attention_chunk_size": int(
                config["adapter"]["execution"]["channel_attention_chunk_size"]
            ),
        },
        "protected_test_opened": False,
    }
    value["feature_cache_key"] = stable_hash(value)
    return value


def _cache_paths(config: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Path]:
    root = resolve_repo_path(config["resources"]["feature_cache_root"])
    directory = root / str(identity["task"]) / str(identity["feature_cache_key"])
    if "protected" in {part.lower() for part in directory.resolve().parts}:
        raise PermissionError("NormWear feature cache path crosses protected boundary")
    return {
        "directory": directory,
        "identity": directory / "identity.json",
        "status": directory / "status.json",
        "features": directory / "features.npy",
        "metadata": directory / "metadata.npz",
    }


def _batch_items(view: NormWearPublicView, indices: Sequence[int]) -> dict[str, Any]:
    items = [view[int(index)] for index in indices]
    return {
        "eeg": torch.stack([item["eeg"] for item in items]),
        "hbo": torch.stack([item["hbo"] for item in items]),
        "hbr": torch.stack([item["hbr"] for item in items]),
        "sample_ids": [str(item["sample_id"]) for item in items],
    }


def _feature_digest(sample_ids: Sequence[str], features: np.ndarray) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(sample_ids), 64):
        block = np.asarray(features[start : start + 64], dtype=np.float32)
        for identifier, row in zip(sample_ids[start : start + 64], block, strict=True):
            digest.update(str(identifier).encode("utf-8"))
            digest.update(b"\0")
            digest.update(np.ascontiguousarray(row).tobytes())
    return digest.hexdigest()


def extract_or_resume_task(
    *,
    inventory: PublicInventory,
    adapter: NormWearFrozenEncoder,
    identity: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Path]]:
    paths = _cache_paths(config, identity)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    count = len(inventory.indices)
    dimension = len(inventory.delivered_channel_names) * 768
    batch_size = int(config["resources"]["feature_batch_size"])
    commit_interval = int(config["resources"]["progress_commit_interval_batches"])
    if batch_size < 1 or commit_interval < 1:
        raise ValueError("NormWear feature replay batch/progress interval must be positive")

    if paths["identity"].is_file():
        retained_identity = json.loads(paths["identity"].read_text(encoding="utf-8"))
        if retained_identity != identity:
            raise RuntimeError("NormWear retained feature identity drifted")
    else:
        write_json(paths["identity"], identity)
    status = (
        json.loads(paths["status"].read_text(encoding="utf-8"))
        if paths["status"].is_file()
        else {
            "schema": "normwear_feature_cache_status_v2",
            "state": "in_progress",
            "feature_cache_key": identity["feature_cache_key"],
            "completed_rows": 0,
            "total_rows": count,
            "started_at": utc_now(),
            "protected_test_opened": False,
        }
    )
    if status.get("feature_cache_key") != identity["feature_cache_key"]:
        raise RuntimeError("NormWear feature status identity drifted")
    if int(status.get("total_rows", -1)) != count:
        raise RuntimeError("NormWear feature status sample count drifted")
    completed = int(status.get("completed_rows", 0))
    if completed < 0 or completed > count:
        raise RuntimeError("NormWear feature resume row is invalid")

    if paths["features"].is_file():
        features = np.load(paths["features"], mmap_mode="r+")
        if features.shape != (count, dimension) or features.dtype != np.float32:
            raise RuntimeError("NormWear feature memmap shape or dtype drifted")
    else:
        if completed:
            raise RuntimeError("NormWear feature status exists without its memmap")
        features = np.lib.format.open_memmap(
            paths["features"], mode="w+", dtype=np.float32, shape=(count, dimension)
        )

    view = NormWearPublicView(inventory)
    indices = inventory.indices
    started = time.perf_counter()
    initial_completed = completed
    batch_number = 0
    while completed < count:
        stop = min(completed + batch_size, count)
        batch = _batch_items(view, indices[completed:stop])
        with torch.inference_mode():
            embedding = adapter(
                batch["eeg"].to(device, non_blocking=True),
                batch["hbo"].to(device, non_blocking=True),
                batch["hbr"].to(device, non_blocking=True),
                eeg_sampling_rate_hz=200.0,
                fnirs_sampling_rate_hz=10.0,
                eeg_channel_names=inventory.eeg_channels,
                fnirs_location_names=inventory.fnirs_locations,
            )
        expected_shape = (stop - completed, dimension)
        if embedding.shape != expected_shape or not bool(torch.isfinite(embedding).all()):
            raise RuntimeError(
                f"NormWear feature shape/finite check failed: {tuple(embedding.shape)}"
            )
        features[completed:stop] = embedding.float().cpu().numpy()
        completed = stop
        batch_number += 1
        if batch_number % commit_interval == 0 or completed == count:
            features.flush()
            status.update(
                {
                    "state": "in_progress" if completed < count else "validating",
                    "completed_rows": completed,
                    "updated_at": utc_now(),
                }
            )
            write_json(paths["status"], status)
        if batch_number % 20 == 0 or completed == count:
            elapsed = time.perf_counter() - started
            rate = (completed - initial_completed) / max(elapsed, 1e-9)
            print(
                f"[{inventory.task}] production replay {completed}/{count} "
                f"({rate:.2f} samples/s)",
                flush=True,
            )

    sample_ids = tuple(inventory.sample_ids)
    rows = [inventory.dataset.lightweight_metadata(index) for index in indices]
    targets = np.asarray(
        [inventory.dataset.class_to_index[str(row["condition"])] for row in rows],
        dtype=np.int64,
    )
    subjects = np.asarray([str(row["subject"]) for row in rows], dtype=str)
    np.savez_compressed(
        paths["metadata"],
        dataset_indices=np.asarray(indices, dtype=np.int64),
        sample_ids=np.asarray(sample_ids, dtype=str),
        targets=targets,
        subjects=subjects,
    )

    feature_sum = np.zeros(dimension, dtype=np.float64)
    feature_square_sum = np.zeros(dimension, dtype=np.float64)
    minimum_row_std = float("inf")
    maximum_absolute_value = 0.0
    for start in range(0, count, 64):
        block = np.asarray(features[start : start + 64], dtype=np.float32)
        if not np.isfinite(block).all():
            raise RuntimeError("NormWear feature cache contains non-finite values")
        feature_sum += block.sum(axis=0, dtype=np.float64)
        feature_square_sum += np.square(block, dtype=np.float64).sum(axis=0)
        minimum_row_std = min(
            minimum_row_std,
            float(np.min(np.std(block, axis=1, dtype=np.float64))),
        )
        maximum_absolute_value = max(maximum_absolute_value, float(np.max(np.abs(block))))
    variance = np.maximum(feature_square_sum / count - np.square(feature_sum / count), 0.0)
    nonconstant = int((np.sqrt(variance) > 1e-8).sum())
    if minimum_row_std <= 1e-8 or nonconstant == 0:
        raise RuntimeError("NormWear feature cache is anomalously constant")

    replay_count = min(batch_size, count)
    replay_batch = _batch_items(view, indices[:replay_count])
    with torch.inference_mode():
        replay = adapter(
            replay_batch["eeg"].to(device),
            replay_batch["hbo"].to(device),
            replay_batch["hbr"].to(device),
            eeg_sampling_rate_hz=200.0,
            fnirs_sampling_rate_hz=10.0,
            eeg_channel_names=inventory.eeg_channels,
            fnirs_location_names=inventory.fnirs_locations,
        ).float().cpu().numpy()
    cache_replay_exact = bool(
        np.array_equal(replay, np.asarray(features[:replay_count]))
    )
    if not cache_replay_exact:
        raise RuntimeError("NormWear first public batch does not replay its cache bitwise")

    cache_report = {
        "feature_cache_key": identity["feature_cache_key"],
        "cache_directory": portable_path(paths["directory"]),
        "feature_shape": [count, dimension],
        "feature_dtype": "float32",
        "feature_sha256": _feature_digest(sample_ids, features),
        "feature_file_sha256": sha256_file(paths["features"]),
        "metadata_file_sha256": sha256_file(paths["metadata"]),
        "minimum_per_sample_feature_std": minimum_row_std,
        "maximum_absolute_feature_value": maximum_absolute_value,
        "nonconstant_coordinate_count": nonconstant,
        "cache_replay_exact": cache_replay_exact,
        "cache_replay_batch_size": replay_count,
        "resumed_from_row": initial_completed,
        "extracted_row_count_this_run": count - initial_completed,
        "protected_test_opened": False,
    }
    status.update(
        {
            "state": "complete",
            "completed_rows": count,
            "completed_at": utc_now(),
            "cache_report": cache_report,
        }
    )
    write_json(paths["status"], status)
    return cache_report, paths


def complete_cell(
    *,
    task: str,
    inventory: PublicInventory,
    method: Mapping[str, Any],
    cache_identity: Mapping[str, Any],
    cache_report: Mapping[str, Any],
    output_root: Path,
) -> tuple[dict[str, Any], Path]:
    retained_path = output_root / f"{task}.json"
    retained = json.loads(retained_path.read_text(encoding="utf-8"))
    if retained.get("method_id") != METHOD_ID or retained.get("task_id") != task:
        raise RuntimeError("NormWear retained data-boundary cell identity drifted")
    retained["adapter_identity"] = {
        **method,
        "delivered_channel_order": list(inventory.delivered_channel_names),
        "delivered_channel_order_sha256": stable_hash(
            list(inventory.delivered_channel_names)
        ),
        "input_shape": [
            "batch",
            len(inventory.delivered_channel_names),
            int(round(inventory.duration_s * 65.0)),
        ],
        "output_shape": ["batch", len(inventory.delivered_channel_names) * 768],
    }
    retained["gate_status"].update({"A5": "pass", "A6": "pass", "A7": "pass"})
    retained["public_adapter_audit"] = {
        "unique_sample_count": len(inventory.indices),
        "all_unique_public_samples_executed": True,
        "production_adapter_path": "NormWearFrozenEncoder.forward",
        "feature_cache_identity_sha256": stable_hash(cache_identity),
        **cache_report,
    }
    retained["cell_status"] = "pending"
    write_json(retained_path, retained)
    return retained, retained_path


def run(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    tasks: Sequence[str] = (),
    device: torch.device,
) -> dict[str, Any]:
    selected = parse_tasks(tasks)
    config, resolved_config = load_config(config_path)
    contract = load_alignment_contract()
    if contract["execution_policy"]["active_delivery_method"] != METHOD_ID:
        raise PermissionError("NormWear is not the active serial delivery method")
    output_root = output_root.resolve()
    if "protected" in {part.lower() for part in output_root.parts}:
        raise PermissionError("refusing protected NormWear alignment output")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NormWear full-public replay")

    task_reports: list[dict[str, Any]] = []
    cell_paths: list[Path] = []
    started_at = utc_now()
    lock_path = Path(str(config["resources"]["gpu_lock_path"]))
    with exclusive_gpu_lock(lock_path):
        backbone, upstream_module, metadata = load_verified_normwear_encoder(device=device)
        adapter = NormWearFrozenEncoder(
            backbone,
            upstream_module,
            channel_chunk_size=int(
                config["adapter"]["execution"]["channel_attention_chunk_size"]
            ),
        ).to(device)
        method = method_identity(
            metadata=metadata, config=config, config_path=resolved_config
        )
        for task in selected:
            inventory = load_public_inventory(config, task=task)
            identity = feature_cache_identity(
                inventory=inventory, method=method, config=config
            )
            cache_report, _ = extract_or_resume_task(
                inventory=inventory,
                adapter=adapter,
                identity=identity,
                config=config,
                device=device,
            )
            cell, cell_path = complete_cell(
                task=task,
                inventory=inventory,
                method=method,
                cache_identity=identity,
                cache_report=cache_report,
                output_root=output_root,
            )
            task_reports.append(
                {
                    "task": task,
                    "path": portable_path(cell_path),
                    "sample_count": len(inventory.indices),
                    "feature_sha256": cache_report["feature_sha256"],
                    "status": "A0_A7_pass_A8_pending",
                }
            )
            cell_paths.append(cell_path)
            del inventory
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    all_supported_complete = all(
        (output_root / f"{task}.json").is_file()
        and json.loads((output_root / f"{task}.json").read_text(encoding="utf-8"))[
            "gate_status"
        ]["A7"]
        == "pass"
        for task in SUPPORTED_TASKS
    )
    if all_supported_complete:
        refed = unsupported_refed_cell(config=config, alignment_contract=contract)
        refed_path = output_root / "refed_regression.json"
        write_json(refed_path, refed)
        cell_paths = [output_root / f"{task}.json" for task in SUPPORTED_TASKS]
        cell_paths.append(refed_path)
    schema_report = audit_evidence(ALIGNMENT_CONTRACT, cell_paths)
    summary = {
        "schema": "normwear_alignment_audit_summary_v2",
        "status": (
            "A0_A7_pass_A8_pending_protected_locked"
            if all_supported_complete
            else "partial_A7_replay_in_serial_progress_protected_locked"
        ),
        "method_id": METHOD_ID,
        "started_at": started_at,
        "completed_at": utc_now(),
        "tasks_completed_this_run": list(selected),
        "task_reports": task_reports,
        "all_supported_tasks_complete": all_supported_complete,
        "schema_audit": schema_report,
        "protected_test_opened": False,
    }
    summary_name = "summary.json" if all_supported_complete else "partial_summary.json"
    write_json(output_root / summary_name, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--task", action="append", default=[])
    args = parser.parse_args(argv)
    report = run(
        config_path=args.config,
        output_root=args.output_root,
        tasks=args.task,
        device=torch.device(args.device),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
