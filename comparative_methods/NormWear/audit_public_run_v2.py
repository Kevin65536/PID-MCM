#!/usr/bin/env python3
"""Audit completed NormWear public-development v2 runs and artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comparative_methods.NormWear.run_public_development_v2 import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    DEFAULT_CONFIG,
    METHOD_ID,
    RUN_SCHEMA,
    balanced_subset,
    load_public_fold,
    load_runner_config,
    load_verified_feature_cache,
    portable_path,
    resolve_repo_path,
    sha256_file,
    stable_hash,
    write_json,
)
from efrm_pytorch.metrics import classification_metrics  # noqa: E402


AUDIT_SCHEMA = "normwear_public_development_audit_v2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if "protected" in {part.lower() for part in resolved.parts}:
        raise PermissionError(f"refusing protected audit input: {resolved}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"audit JSON must be an object: {path}")
    if value.get("protected_test_opened", False):
        raise PermissionError(f"artifact reports protected access: {path}")
    return value


def expected_membership(
    *, mode: str, fold: Any, config: Mapping[str, Any]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if mode == "public_selection_and_refit":
        return fold.train_indices, fold.validation_indices
    if mode != "smoke_only":
        raise RuntimeError(f"unknown NormWear public run mode: {mode}")
    train = balanced_subset(
        fold,
        fold.train_indices,
        samples_per_class=int(config["smoke"]["train_samples_per_class"]),
    )
    validation = balanced_subset(
        fold,
        fold.validation_indices,
        samples_per_class=int(config["smoke"]["validation_samples_per_class"]),
    )
    return train, validation


def expected_sample_ids(fold: Any, indices: Sequence[int]) -> tuple[str, ...]:
    lookup = dict(zip(fold.inventory.indices, fold.inventory.sample_ids, strict=True))
    return tuple(lookup[int(index)] for index in indices)


def audit_run(
    run_dir: Path,
    *,
    config: Mapping[str, Any],
    config_path: Path,
    alignment: Mapping[str, Any],
    alignment_path: Path,
) -> dict[str, Any]:
    resolved_run = run_dir.resolve()
    run_root = resolve_repo_path(config["resources"]["run_root"])
    try:
        resolved_run.relative_to(run_root)
    except ValueError as exc:
        raise PermissionError(
            f"run is outside the NormWear public root: {resolved_run}"
        ) from exc
    manifest_path = resolved_run / "manifest.json"
    status_path = resolved_run / "status.json"
    manifest = load_json(manifest_path)
    status = load_json(status_path)
    require(manifest == status, "completed manifest and status artifacts differ")
    require(manifest.get("schema") == RUN_SCHEMA, "unexpected NormWear run schema")
    require(manifest.get("status") == "completed", "NormWear public run is not complete")
    require(manifest.get("method_id") == METHOD_ID, "run method identity is not NormWear")
    require(manifest.get("table_admissible") is False, "public run claims table admission")
    require(
        manifest.get("runner_config_sha256") == sha256_file(config_path),
        "runner config hash differs from the audited file",
    )
    require(
        manifest.get("runner_sha256")
        == sha256_file(METHOD_ROOT / "run_public_development_v2.py"),
        "runner source hash differs from the audited file",
    )
    require(
        manifest.get("alignment_config_sha256") == sha256_file(alignment_path),
        "alignment config hash differs from the audited file",
    )
    task = str(manifest["task"])
    outer_fold = int(manifest["outer_fold"])
    seed = int(manifest["seed"])
    require(task in config["job_matrix"]["tasks"], "run task is outside frozen matrix")
    require(
        outer_fold in config["job_matrix"]["outer_folds"],
        "run fold is outside frozen matrix",
    )
    require(seed in config["job_matrix"]["seeds"], "run seed is outside frozen matrix")
    fold = load_public_fold(alignment, task=task, outer_fold=outer_fold)
    require(
        manifest["public_manifest_sha256"] == fold.public_manifest_sha256,
        "run public manifest fingerprint drifted",
    )
    train_indices, validation_indices = expected_membership(
        mode=str(manifest["mode"]), fold=fold, config=config
    )
    require(
        int(manifest["selection_train_sample_count"]) == len(train_indices),
        "selection train count differs from public membership",
    )
    require(
        int(manifest["selection_validation_sample_count"]) == len(validation_indices),
        "selection validation count differs from public membership",
    )

    predictions_path = resolved_run / "public_validation_predictions.npz"
    with np.load(predictions_path, allow_pickle=False) as payload:
        predictions = {name: payload[name] for name in payload.files}
    require(
        set(predictions) == {"logits", "target", "dataset_index", "subject", "sample_id"},
        "public prediction artifact has unexpected arrays",
    )
    require(
        tuple(predictions["dataset_index"].astype(int).tolist())
        == tuple(validation_indices),
        "public prediction indices differ from validation membership",
    )
    require(
        tuple(predictions["sample_id"].astype(str).tolist())
        == expected_sample_ids(fold, validation_indices),
        "public prediction sample identities differ from validation membership",
    )
    rows = [
        fold.inventory.dataset.lightweight_metadata(int(index))
        for index in validation_indices
    ]
    expected_target = np.asarray(
        [fold.inventory.dataset.class_to_index[str(row["condition"])] for row in rows],
        dtype=np.int64,
    )
    expected_subject = tuple(str(row["subject"]) for row in rows)
    require(
        tuple(predictions["subject"].astype(str).tolist()) == expected_subject,
        "public prediction subjects differ from validation membership",
    )
    require(
        np.array_equal(predictions["target"].astype(np.int64), expected_target),
        "public prediction targets differ from canonical public labels",
    )
    require(bool(np.isfinite(predictions["logits"]).all()), "public logits are non-finite")
    metrics = classification_metrics(expected_target, predictions["logits"], fold.class_names)
    retained_metrics = load_json(resolved_run / "public_selection_report.json")
    require(
        stable_hash(metrics) == stable_hash(retained_metrics["validation_metrics"]),
        "retained public metrics do not recompute from predictions",
    )
    require(
        stable_hash(retained_metrics) == stable_hash(manifest["probe"]),
        "selection report differs from completed run manifest",
    )
    require(
        retained_metrics["selection_standardizer"]["fit_membership"]
        == "outer_train_only",
        "selection standardizer was not fit on outer-train only",
    )

    checkpoint_path = resolve_repo_path(manifest["probe"]["public_refit"]["checkpoint_path"])
    require(checkpoint_path.parent == resolved_run, "refit checkpoint is outside run directory")
    require(
        sha256_file(checkpoint_path)
        == str(manifest["probe"]["public_refit"]["checkpoint_sha256"]),
        "public refit checkpoint hash drifted",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(checkpoint.get("schema") == CHECKPOINT_SCHEMA, "unexpected checkpoint schema")
    require(checkpoint.get("method_id") == METHOD_ID, "checkpoint method is not NormWear")
    require(
        checkpoint.get("runner_sha256") == manifest["runner_sha256"],
        "checkpoint runner identity differs from run manifest",
    )
    require(checkpoint.get("protected_test_opened") is False, "checkpoint crossed protected data")
    refit_indices = tuple(dict.fromkeys([*train_indices, *validation_indices]))
    require(
        tuple(checkpoint["refit_dataset_indices"].tolist()) == refit_indices,
        "checkpoint refit membership differs from public train+validation",
    )
    feature_dimension = len(fold.inventory.delivered_channel_names) * 768
    class_count = len(fold.class_names)
    require(
        tuple(checkpoint["head_state"]["weight"].shape)
        == (class_count, feature_dimension),
        "checkpoint linear head shape differs from NormWear representation",
    )
    require(
        tuple(checkpoint["feature_mean"].shape) == (feature_dimension,)
        and tuple(checkpoint["feature_scale"].shape) == (feature_dimension,),
        "checkpoint standardizer shape drifted",
    )
    require(
        bool(torch.isfinite(checkpoint["feature_mean"]).all())
        and bool(torch.isfinite(checkpoint["feature_scale"]).all())
        and bool((checkpoint["feature_scale"] > 0).all()),
        "checkpoint standardizer is invalid",
    )

    arrays, cache_identity, cache_dir, cache_verification = load_verified_feature_cache(
        config=config, fold=fold
    )
    retained_cache = manifest["feature_cache"]
    require(
        retained_cache["identity"] == cache_identity,
        "run feature cache identity differs from A7",
    )
    require(
        resolve_repo_path(retained_cache["directory"]) == cache_dir,
        "run feature cache directory differs from A7",
    )
    for field, expected in cache_verification.items():
        require(retained_cache.get(field) == expected, f"feature cache audit differs on {field}")
    require(
        retained_cache.get("global_target_metadata_loaded") is False,
        "run loaded global target metadata",
    )
    require(
        arrays["features"].shape[1] == feature_dimension,
        "audited feature dimension differs from checkpoint",
    )

    return {
        "run_dir": portable_path(resolved_run),
        "manifest_sha256": sha256_file(manifest_path),
        "mode": str(manifest["mode"]),
        "task": task,
        "outer_fold": outer_fold,
        "seed": seed,
        "selection_train_sample_count": len(train_indices),
        "selection_validation_sample_count": len(validation_indices),
        "validation_macro_f1": float(metrics["macro_f1"]),
        "selected_candidate": manifest["probe"]["selected_candidate"],
        "public_refit_checkpoint_sha256": sha256_file(checkpoint_path),
        "selected_public_feature_sha256": cache_verification[
            "selected_public_feature_sha256"
        ],
        "status": "pass",
        "table_admissible": False,
        "protected_test_opened": False,
    }


def audit(run_dirs: Sequence[Path], config_path: Path) -> dict[str, Any]:
    config, resolved_config, alignment, alignment_path = load_runner_config(config_path)
    reports = [
        audit_run(
            path,
            config=config,
            config_path=resolved_config,
            alignment=alignment,
            alignment_path=alignment_path,
        )
        for path in run_dirs
    ]
    return {
        "schema": AUDIT_SCHEMA,
        "status": "pass",
        "created_at": utc_now(),
        "auditor_path": portable_path(Path(__file__)),
        "auditor_sha256": sha256_file(Path(__file__)),
        "runner_config_path": portable_path(resolved_config),
        "runner_config_sha256": sha256_file(resolved_config),
        "run_reports": reports,
        "protected_test_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="+", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = audit(args.run_dir, args.config)
    if args.output is not None:
        write_json(resolve_repo_path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
