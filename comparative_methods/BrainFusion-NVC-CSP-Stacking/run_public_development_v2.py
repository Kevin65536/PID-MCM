#!/usr/bin/env python3
"""Run one public-only BrainFusion outer-fold development job."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import f1_score
import torch
import yaml


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
ADAPTER_ROOT = METHOD_ROOT / "adapters"
for import_path in (REPO_ROOT, METHOD_ROOT, ADAPTER_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from alignment_data import (
    METHOD_ID,
    SUPPORTED_TASKS,
    BrainFusionPublicView,
    PublicInventory,
    data_branch_fingerprints,
    load_config as load_alignment_config,
    load_public_inventory,
    resolve_repo_path,
    sample_id,
    stable_hash,
)
from brainfusion_gpu.features import BrainFusionFeaturePipeline, CSPConfig
from brainfusion_gpu.pipeline import BrainFusionFoldPipeline
from brainfusion_gpu.stacking import StackingConfig
from comparative_methods.audit_public_preflight import (
    public_json,
    registry_manifest,
    sha256_file,
    strict_public_entry,
)


CONFIG_SCHEMA = "brainfusion_public_development_v2"
TENSOR_CACHE_SCHEMA = "brainfusion_full_public_tensor_cache_v2"
DEFAULT_CONFIG = METHOD_ROOT / "configs/public_development_v2.yaml"
ALIGNMENT_CONTRACT = REPO_ROOT / "comparative_methods/adapter_alignment_gate_contract_v2.yaml"
RUN_ROOT = METHOD_ROOT / "runs/public_development_v2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    if "protected" in {part.lower() for part in resolved.parts}:
        raise PermissionError(f"refusing protected BrainFusion artifact: {resolved}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_torch_save(path: Path, payload: Mapping[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _active_contract() -> dict[str, Any]:
    contract = yaml.safe_load(ALIGNMENT_CONTRACT.read_text(encoding="utf-8"))
    active = contract.get("execution_policy", {}).get("active_delivery_method")
    if active != METHOD_ID:
        raise PermissionError(f"BrainFusion is not the active delivery method: {active!r}")
    if contract.get("authority", {}).get("protected_test_default") != "locked":
        raise PermissionError("protected test boundary is not locked")
    return contract


def load_runner_config(
    path: str | Path,
) -> tuple[dict[str, Any], Path, dict[str, Any], Path]:
    config_path = resolve_repo_path(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"expected {CONFIG_SCHEMA} config: {config_path}")
    if config.get("method_id") != METHOD_ID or config.get("mode") != "public_development_only":
        raise PermissionError("BrainFusion runner must remain public-development-only")
    if config.get("protected_test_default") != "locked":
        raise PermissionError("protected test must remain locked")
    matrix = config.get("job_matrix", {})
    tasks = tuple(str(value) for value in matrix.get("tasks", ()))
    folds = tuple(int(value) for value in matrix.get("outer_folds", ()))
    seeds = tuple(int(value) for value in matrix.get("seeds", ()))
    if tasks != SUPPORTED_TASKS or folds != tuple(range(5)) or seeds != (17, 42, 73):
        raise ValueError("BrainFusion public matrix identity drifted")
    if int(matrix.get("expected_public_jobs", -1)) != 75:
        raise ValueError("BrainFusion public matrix must contain 75 jobs")
    if int(matrix.get("max_concurrent_jobs", -1)) != 1:
        raise ValueError("BrainFusion public matrix must remain serial")
    if int(matrix.get("automatic_retry_count", -1)) != 0:
        raise ValueError("BrainFusion public jobs may not retry automatically")
    cache_root = resolve_repo_path(config["resources"]["tensor_cache_root"])
    method_runs = (METHOD_ROOT / "runs").resolve()
    if cache_root != method_runs and method_runs not in cache_root.parents:
        raise PermissionError("BrainFusion tensor cache must remain under method runs")
    if "protected" in {part.lower() for part in cache_root.parts}:
        raise PermissionError("BrainFusion tensor cache crossed the protected boundary")
    alignment_path = resolve_repo_path(config["alignment_config"])
    alignment, checked_path = load_alignment_config(alignment_path)
    if checked_path != alignment_path:
        raise RuntimeError("BrainFusion alignment config path drifted")
    _active_contract()
    return config, config_path, alignment, alignment_path


def _fold_membership(
    alignment: Mapping[str, Any], *, task: str, outer_fold: int
) -> tuple[Any, list[int], list[int], Path, str]:
    inventory = load_public_inventory(alignment, task=task)
    registry = registry_manifest(resolve_repo_path(alignment["registry"]["manifest"]))
    entry = strict_public_entry(registry, task=task, outer_fold=outer_fold)
    public_path = Path(str(entry["public_path"])).resolve()
    manifest = public_json(public_path)
    digest = sha256_file(public_path)
    if digest != str(entry["public_sha256"]):
        raise RuntimeError("BrainFusion public fold manifest hash drifted")
    train, validation = inventory.dataset.validate_shared_public_split(public_path)
    if set(train).intersection(validation):
        raise RuntimeError("BrainFusion public train and validation overlap")
    if len(train) != int(entry["train_sample_count"]) or len(validation) != int(
        entry["validation_sample_count"]
    ):
        raise RuntimeError("BrainFusion public fold membership count drifted")
    if manifest.get("protected_test_opened", False):
        raise PermissionError("public fold reports protected access")
    return inventory, train, validation, public_path, digest


def _targets_and_groups(inventory: Any, indices: Sequence[int]) -> tuple[np.ndarray, list[str]]:
    targets = []
    groups = []
    for index in indices:
        row = inventory.dataset.lightweight_metadata(int(index))
        targets.append(inventory.dataset.class_to_index[str(row["condition"])])
        groups.append(str(row["subject"]))
    return np.asarray(targets, dtype=np.int64), groups


def diverse_balanced_subset(
    inventory: Any, indices: Sequence[int], *, per_class: int, seed: int
) -> list[int]:
    by_class_group: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index in indices:
        row = inventory.dataset.lightweight_metadata(int(index))
        label = inventory.dataset.class_to_index[str(row["condition"])]
        by_class_group[int(label)][str(row["subject"])].append(int(index))
    rng = random.Random(int(seed))
    selected: list[int] = []
    for label in sorted(by_class_group):
        groups = sorted(by_class_group[label])
        rng.shuffle(groups)
        for values in by_class_group[label].values():
            values.sort()
        cursor = {group: 0 for group in groups}
        class_selected: list[int] = []
        while len(class_selected) < per_class:
            progressed = False
            for group in groups:
                position = cursor[group]
                values = by_class_group[label][group]
                if position < len(values):
                    class_selected.append(values[position])
                    cursor[group] += 1
                    progressed = True
                    if len(class_selected) == per_class:
                        break
            if not progressed:
                raise RuntimeError(f"class {label} lacks {per_class} public smoke samples")
        selected.extend(class_selected)
    return sorted(selected)


def _materialize(
    view: BrainFusionPublicView, inventory: Any, indices: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str]]:
    eeg: list[torch.Tensor] = []
    hbo: list[torch.Tensor] = []
    hbr: list[torch.Tensor] = []
    targets: list[int] = []
    sample_ids: list[str] = []
    groups: list[str] = []
    for index in indices:
        item = view[int(index)]
        row = inventory.dataset.lightweight_metadata(int(index))
        eeg.append(item["eeg"])
        hbo.append(item["hbo"])
        hbr.append(item["hbr"])
        targets.append(inventory.dataset.class_to_index[str(row["condition"])])
        sample_ids.append(sample_id(inventory.dataset, int(index)))
        groups.append(str(row["subject"]))
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("BrainFusion fold materialization contains duplicate identities")
    return (
        torch.stack(eeg),
        torch.stack(hbo),
        torch.stack(hbr),
        torch.tensor(targets, dtype=torch.long),
        sample_ids,
        groups,
    )


def tensor_cache_identity(
    inventory: PublicInventory,
    *,
    alignment: Mapping[str, Any],
    alignment_path: Path,
) -> dict[str, Any]:
    value = {
        "schema": TENSOR_CACHE_SCHEMA,
        "task": inventory.task,
        "sample_count": len(inventory.indices),
        "dataset_indices_sha256": stable_hash(list(inventory.indices)),
        "sample_identity_sha256": stable_hash(list(inventory.sample_ids)),
        "sample_inventory_sha256": inventory.sample_inventory_sha256,
        "split_fingerprint": inventory.split_fingerprint,
        "measured_channel_identity_sha256": inventory.measured_channel_identity_sha256,
        "alignment_config_sha256": sha256_file(alignment_path),
        "alignment_data_sha256": sha256_file(METHOD_ROOT / "alignment_data.py"),
        "data_branch_sha256": data_branch_fingerprints(alignment),
        "tensor_contract": {
            "dtype": "float32",
            "eeg_channels": list(inventory.eeg_channels),
            "fnirs_locations": list(inventory.fnirs_locations),
            "duration_s": inventory.duration_s,
            "eeg_sample_rate_hz": float(alignment["data"]["eeg_sample_rate_hz"]),
            "fnirs_sample_rate_hz": float(alignment["data"]["fnirs_sample_rate_hz"]),
            "channel_policy": "fixed_measured_inventory_no_copy_no_padding",
            "time_support_policy": "canonical_interval_only_full_recorded_and_analysis_valid",
        },
        "fitted_or_supervised_state_included": False,
        "protected_test_opened": False,
    }
    value["tensor_cache_key"] = stable_hash(value)
    return value


def validate_tensor_cache(
    payload: Mapping[str, torch.Tensor], inventory: PublicInventory
) -> None:
    required = {"eeg", "hbo", "hbr", "targets", "dataset_indices"}
    if set(payload) != required:
        raise RuntimeError(f"BrainFusion tensor cache fields differ: {sorted(payload)}")
    count = len(inventory.indices)
    eeg_samples = int(round(inventory.duration_s * 200.0))
    fnirs_samples = int(round(inventory.duration_s * 10.0))
    expected_shapes = {
        "eeg": (count, len(inventory.eeg_channels), eeg_samples),
        "hbo": (count, len(inventory.fnirs_locations), fnirs_samples),
        "hbr": (count, len(inventory.fnirs_locations), fnirs_samples),
        "targets": (count,),
        "dataset_indices": (count,),
    }
    for name, shape in expected_shapes.items():
        if not isinstance(payload[name], torch.Tensor) or tuple(payload[name].shape) != shape:
            raise RuntimeError(f"BrainFusion tensor cache shape drifted: {name}")
    if any(payload[name].dtype != torch.float32 for name in ("eeg", "hbo", "hbr")):
        raise RuntimeError("BrainFusion tensor cache modality dtype drifted")
    if payload["targets"].dtype != torch.long or payload["dataset_indices"].dtype != torch.long:
        raise RuntimeError("BrainFusion tensor cache index dtype drifted")
    if tuple(payload["dataset_indices"].tolist()) != inventory.indices:
        raise RuntimeError("BrainFusion tensor cache public index order drifted")
    expected_targets = []
    for index in inventory.indices:
        row = inventory.dataset.lightweight_metadata(int(index))
        expected_targets.append(inventory.dataset.class_to_index[str(row["condition"])])
    if payload["targets"].tolist() != expected_targets:
        raise RuntimeError("BrainFusion tensor cache target order drifted")
    for name in ("eeg", "hbo", "hbr"):
        values = payload[name]
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError(f"BrainFusion tensor cache contains non-finite {name}")
        flattened = values.reshape(count, -1)
        if bool((flattened.amax(dim=1) <= flattened.amin(dim=1)).any()):
            raise RuntimeError(f"BrainFusion tensor cache contains constant {name} trials")


def load_tensor_cache_payload(
    cache_path: Path,
    manifest_path: Path,
    *,
    expected_identity: Mapping[str, Any],
    inventory: PublicInventory,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("identity") != expected_identity:
        raise RuntimeError("BrainFusion tensor cache manifest identity drifted")
    if sha256_file(cache_path) != manifest.get("file_sha256"):
        raise RuntimeError("BrainFusion tensor cache file hash drifted")
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("BrainFusion tensor cache payload must be a mapping")
    validate_tensor_cache(payload, inventory)
    return payload, manifest


def materialize_or_load_tensor_cache(
    inventory: PublicInventory,
    *,
    alignment: Mapping[str, Any],
    alignment_path: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any], bool, Path, Path]:
    identity = tensor_cache_identity(
        inventory, alignment=alignment, alignment_path=alignment_path
    )
    cache_root = resolve_repo_path(config["resources"]["tensor_cache_root"])
    cache_path = cache_root / inventory.task / f"{identity['tensor_cache_key']}.pt"
    manifest_path = cache_path.with_suffix(".json")
    if cache_path.is_file() or manifest_path.is_file():
        if not cache_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("BrainFusion tensor cache is only partially present")
        payload, manifest = load_tensor_cache_payload(
            cache_path,
            manifest_path,
            expected_identity=identity,
            inventory=inventory,
        )
        return payload, manifest, True, cache_path, manifest_path

    view = BrainFusionPublicView(inventory)
    materialized = _materialize(view, inventory, inventory.indices)
    if tuple(materialized[4]) != inventory.sample_ids:
        raise RuntimeError("BrainFusion tensor cache materialization identity drifted")
    payload = {
        "eeg": materialized[0],
        "hbo": materialized[1],
        "hbr": materialized[2],
        "targets": materialized[3],
        "dataset_indices": torch.tensor(inventory.indices, dtype=torch.long),
    }
    validate_tensor_cache(payload, inventory)
    atomic_torch_save(cache_path, payload)
    manifest = {
        "schema": TENSOR_CACHE_SCHEMA,
        "identity": identity,
        "file_sha256": sha256_file(cache_path),
        "created_at": utc_now(),
        "protected_test_opened": False,
    }
    write_json(manifest_path, manifest)
    return payload, manifest, False, cache_path, manifest_path


def fold_data_from_cache(
    payload: Mapping[str, torch.Tensor],
    inventory: PublicInventory,
    indices: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str]]:
    lookup = {int(index): row for row, index in enumerate(inventory.indices)}
    missing = [int(index) for index in indices if int(index) not in lookup]
    if missing:
        raise RuntimeError(f"BrainFusion public fold index is absent from cache: {missing[:5]}")
    rows = torch.tensor([lookup[int(index)] for index in indices], dtype=torch.long)
    sample_ids = [sample_id(inventory.dataset, int(index)) for index in indices]
    groups = [
        str(inventory.dataset.lightweight_metadata(int(index))["subject"])
        for index in indices
    ]
    return (
        payload["eeg"].index_select(0, rows),
        payload["hbo"].index_select(0, rows),
        payload["hbr"].index_select(0, rows),
        payload["targets"].index_select(0, rows),
        sample_ids,
        groups,
    )


def _pipeline(config: Mapping[str, Any], *, seed: int, smoke: bool) -> BrainFusionFoldPipeline:
    feature = config["features"]
    stack = config["stacking"]
    smoke_config = config["smoke"]
    return BrainFusionFoldPipeline(
        features=BrainFusionFeaturePipeline(
            csp_config=CSPConfig(
                components_per_class=int(feature["csp_components_per_class"]),
                regularization=float(feature["csp_regularization"]),
            ),
            nvc_pair_count=int(feature["nvc_pair_count"]),
        ),
        stacking_config=StackingConfig(
            inner_folds=int(smoke_config["inner_folds"] if smoke else stack["inner_folds"]),
            seed=int(seed),
            linear_svm_c_values=tuple(float(value) for value in stack["linear_svm_c_values"]),
            rbf_svm_c_values=tuple(float(value) for value in stack["rbf_svm_c_values"]),
            random_forest_estimators=int(
                smoke_config["random_forest_estimators"]
                if smoke
                else stack["random_forest_estimators"]
            ),
            random_forest_max_depth=stack["random_forest_max_depth"],
            meta_svm_c=float(stack["meta_svm_c"]),
        ),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, config_path, alignment, alignment_path = load_runner_config(args.config)
    task = str(args.task)
    outer_fold = int(args.outer_fold)
    seed = int(args.seed)
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"unsupported BrainFusion public task: {task}")
    if outer_fold not in range(5) or seed not in (17, 42, 73):
        raise ValueError("BrainFusion fold or seed is outside the frozen matrix")
    output_dir = Path(args.output_dir).resolve()
    run_root = RUN_ROOT.resolve()
    if output_dir != run_root and run_root not in output_dir.parents:
        raise PermissionError(f"BrainFusion output must remain under {run_root}")
    if "protected" in {part.lower() for part in output_dir.parts}:
        raise PermissionError("BrainFusion public runner refuses protected output")
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for BrainFusion public run but unavailable")

    inventory, train, validation, manifest_path, manifest_sha256 = _fold_membership(
        alignment, task=task, outer_fold=outer_fold
    )
    if args.smoke:
        train = diverse_balanced_subset(
            inventory,
            train,
            per_class=int(config["smoke"]["train_samples_per_class"]),
            seed=seed,
        )
        validation = diverse_balanced_subset(
            inventory,
            validation,
            per_class=int(config["smoke"]["validation_samples_per_class"]),
            seed=seed + 1,
        )
    tensor_cache: dict[str, Any] | None = None
    if args.smoke:
        view = BrainFusionPublicView(inventory)
        train_data = _materialize(view, inventory, train)
        validation_data = _materialize(view, inventory, validation)
    else:
        payload, cache_manifest, cache_hit, cache_path, cache_manifest_path = (
            materialize_or_load_tensor_cache(
                inventory,
                alignment=alignment,
                alignment_path=alignment_path,
                config=config,
            )
        )
        train_data = fold_data_from_cache(payload, inventory, train)
        validation_data = fold_data_from_cache(payload, inventory, validation)
        tensor_cache = {
            "path": portable_path(cache_path),
            "manifest_path": portable_path(cache_manifest_path),
            "file_sha256": cache_manifest["file_sha256"],
            "manifest_sha256": sha256_file(cache_manifest_path),
            "identity_sha256": stable_hash(cache_manifest["identity"]),
            "tensor_cache_key": cache_manifest["identity"]["tensor_cache_key"],
            "cache_hit": cache_hit,
            "fitted_or_supervised_state_included": False,
            "protected_test_opened": False,
        }
    train_tensors = [value.to(device) for value in train_data[:4]]
    validation_tensors = [value.to(device) for value in validation_data[:3]]
    pipeline = _pipeline(config, seed=seed, smoke=bool(args.smoke)).fit(
        *train_tensors,
        groups=train_data[5],
        sample_ids=train_data[4],
    )
    predictions = pipeline.predict(*validation_tensors)
    decisions = pipeline.decision_function(*validation_tensors)
    target = validation_data[3].numpy()
    score = float(f1_score(target, predictions, average="macro"))
    checkpoint_dir = pipeline.save(output_dir / "checkpoint")
    restored = BrainFusionFoldPipeline.load(checkpoint_dir, device=device)
    reload_predictions = restored.predict(*validation_tensors)
    reload_decisions = restored.decision_function(*validation_tensors)
    reload_match = bool(
        np.array_equal(predictions, reload_predictions)
        and np.array_equal(decisions, reload_decisions)
    )
    if not reload_match:
        raise RuntimeError("BrainFusion checkpoint reload changed public predictions")
    predictions_payload = {
        "schema": "brainfusion_public_predictions_v2",
        "sample_ids": validation_data[4],
        "targets": target.tolist(),
        "predictions": predictions.tolist(),
        "decision_scores": decisions.tolist(),
        "protected_test_opened": False,
    }
    predictions_path = output_dir / "public_validation_predictions.json"
    write_json(predictions_path, predictions_payload)
    source_files = {
        "runner": Path(__file__),
        "alignment_data": METHOD_ROOT / "alignment_data.py",
        "alignment_audit": METHOD_ROOT / "audit_alignment_v2.py",
        "features": ADAPTER_ROOT / "brainfusion_gpu/features.py",
        "stacking": ADAPTER_ROOT / "brainfusion_gpu/stacking.py",
        "pipeline": ADAPTER_ROOT / "brainfusion_gpu/pipeline.py",
    }
    report = {
        "schema": "brainfusion_public_development_run_v2",
        "status": "pass",
        "method_id": METHOD_ID,
        "task": task,
        "track": str(alignment["tasks"][task]["track"]),
        "outer_fold": outer_fold,
        "seed": seed,
        "mode": "smoke_only" if args.smoke else "public_development",
        "created_at": utc_now(),
        "public_manifest_path": str(manifest_path),
        "public_manifest_sha256": manifest_sha256,
        "config_sha256": sha256_file(config_path),
        "alignment_config_sha256": sha256_file(alignment_path),
        "source_file_sha256": {
            name: sha256_file(path) for name, path in source_files.items()
        },
        "train_sample_count": len(train_data[4]),
        "validation_sample_count": len(validation_data[4]),
        "train_sample_identity_sha256": stable_hash(train_data[4]),
        "validation_sample_identity_sha256": stable_hash(validation_data[4]),
        "train_validation_overlap": bool(set(train_data[4]) & set(validation_data[4])),
        "fit_state": pipeline.audit_state(),
        "tensor_cache": tensor_cache,
        "checkpoint_fit_scope": "outer_training_only_public_validation_held_out",
        "validation_macro_f1": score,
        "predictions_path": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
        "checkpoint_manifest_sha256": sha256_file(checkpoint_dir / "manifest.json"),
        "checkpoint_reload_exact": reload_match,
        "table_admissible": False,
        "claim_boundary": "public_development_only_not_table_admissible",
        "protected_evaluation_authorized": False,
        "protected_test_opened": False,
    }
    if report["train_validation_overlap"]:
        raise RuntimeError("BrainFusion public train and validation identities overlap")
    write_json(output_dir / "run_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--task", required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
