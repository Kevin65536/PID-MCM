#!/usr/bin/env python3
"""Independently audit one retained EFRM LODO-v2 public downstream job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
for import_path in (REPO_ROOT, METHOD_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from comparative_methods.audit_public_preflight import sha256_file  # noqa: E402
from efrm_pytorch.metrics import classification_metrics, regression_metrics  # noqa: E402
from run_downstream_public_v2 import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    METHOD_ID,
    PROTOCOL_ID,
    RUN_SCHEMA,
    load_config,
    load_public_surface,
    portable_path,
    resolve_repo_path,
    stable_hash,
    write_json,
)


AUDIT_SCHEMA = "efrm_lodo_downstream_public_run_audit_v2"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(left: Any, right: Any, *, tolerance: float = 1e-10) -> bool:
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=tolerance))


def audit(run_dir: str | Path, config_path: str | Path) -> dict[str, Any]:
    directory = resolve_repo_path(run_dir)
    if "protected" in {part.lower() for part in directory.parts}:
        raise PermissionError(f"refusing protected EFRM audit input: {directory}")
    config, resolved_config = load_config(config_path)
    manifest_path = directory / "manifest.json"
    report_path = directory / "public_selection_report.json"
    prediction_path = directory / "public_validation_predictions.npz"
    checkpoint_path = directory / "checkpoint_public_refit.pt"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == RUN_SCHEMA, "unexpected EFRM public-run schema")
    require(manifest.get("status") == "completed", "EFRM public run is not complete")
    require(manifest.get("protocol_id") == PROTOCOL_ID, "EFRM protocol identity drifted")
    require(manifest.get("method_id") == METHOD_ID, "EFRM method identity drifted")
    require(manifest.get("table_admissible") is False, "public run claims table admission")
    require(
        manifest.get("target_dataset_exposure") is False,
        "public run reports target-dataset pretraining exposure",
    )
    require(
        manifest.get("protected_test_opened") is False,
        "public run reports protected access",
    )
    require(
        manifest.get("runner_config_sha256") == sha256_file(resolved_config),
        "public runner config hash drifted",
    )
    runner_path = resolve_repo_path(manifest["runner_path"])
    require(
        manifest.get("runner_sha256") == sha256_file(runner_path),
        "public runner source hash drifted",
    )
    task = str(manifest["task"])
    outer_fold = int(manifest["outer_fold"])
    seed = int(manifest["seed"])
    surface = load_public_surface(config, task=task)
    fold = surface.folds[outer_fold]
    require(
        manifest.get("public_manifest_sha256") == fold.public_manifest_sha256,
        "public split hash differs from the method-neutral registry",
    )

    feature = dict(manifest["feature_cache"])
    cache_path = resolve_repo_path(feature.pop("path"))
    retained_file_sha256 = str(feature.pop("file_sha256"))
    feature.pop("cache_hit", None)
    cache_manifest_path = cache_path.with_suffix(".json")
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    require(feature == cache_manifest, "run feature identity differs from cache manifest")
    require(
        retained_file_sha256 == sha256_file(cache_path),
        "retained EFRM feature-cache file hash drifted",
    )
    require(
        cache_manifest.get("target_dataset_exposure") is False
        and cache_manifest.get("protected_test_opened") is False,
        "EFRM feature cache crossed target/protected data",
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(checkpoint.get("schema") == CHECKPOINT_SCHEMA, "unexpected EFRM refit checkpoint")
    for field, expected in (
        ("protocol_id", PROTOCOL_ID),
        ("method_id", METHOD_ID),
        ("task", task),
        ("outer_fold", outer_fold),
        ("seed", seed),
    ):
        require(checkpoint.get(field) == expected, f"checkpoint {field} drifted")
    require(
        checkpoint.get("target_dataset_exposure") is False
        and checkpoint.get("protected_test_opened") is False,
        "EFRM refit checkpoint crossed target/protected data",
    )
    require(
        checkpoint.get("runner_config_sha256") == sha256_file(resolved_config),
        "checkpoint runner-config identity drifted",
    )
    require(
        checkpoint.get("runner_sha256") == sha256_file(runner_path),
        "checkpoint runner identity drifted",
    )
    require(
        checkpoint.get("feature_cache_identity_sha256") == stable_hash(cache_manifest),
        "checkpoint feature-cache identity drifted",
    )
    refit_indices = tuple(
        int(value) for value in checkpoint["refit_dataset_indices"].tolist()
    )
    expected_refit = tuple(dict.fromkeys([*fold.train_indices, *fold.validation_indices]))
    if manifest.get("mode") == "public_selection_and_refit":
        require(
            refit_indices == expected_refit,
            "EFRM full public refit membership differs from train+validation",
        )
    require(
        report["public_refit"]["checkpoint_sha256"] == sha256_file(checkpoint_path),
        "public report checkpoint hash drifted",
    )
    require(
        report["public_refit"]["weights_only_reload_match"] is True,
        "public report lacks weights-only replay",
    )
    require(
        int(report["selected_epoch"]) == int(checkpoint["selected_epoch"]),
        "selected/refit epoch drifted",
    )

    with np.load(prediction_path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    expected_arrays = {
        "prediction",
        "target",
        "target_valid_mask",
        "dataset_index",
        "subject",
        "sample_id",
    }
    require(set(arrays) == expected_arrays, "public prediction arrays differ")
    require(
        len(arrays["prediction"]) == int(manifest["selection_validation_sample_count"]),
        "public prediction count drifted",
    )
    require(
        len(set(arrays["sample_id"].astype(str).tolist())) == len(arrays["sample_id"]),
        "public predictions contain duplicate sample identities",
    )
    require(
        bool(np.isfinite(arrays["prediction"]).all()),
        "public predictions contain non-finite values",
    )
    spec = surface.dataset.spec
    if spec.task_type == "classification":
        recomputed = classification_metrics(
            arrays["target"], arrays["prediction"], spec.class_names
        )
        primary_name = "macro_f1"
    else:
        recomputed = regression_metrics(
            arrays["target"],
            arrays["prediction"],
            arrays["target_valid_mask"],
            spec.target_names,
        )
        primary_name = "ccc"
    reported_metrics: Mapping[str, Any] = report["validation_metrics"]
    require(
        close(recomputed[primary_name], reported_metrics[primary_name]),
        f"recomputed public {primary_name} differs from the report",
    )
    require(
        manifest["probe"] == report,
        "manifest/public-selection report payload drifted",
    )
    return {
        "schema": AUDIT_SCHEMA,
        "status": "pass",
        "mode": manifest["mode"],
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "task": task,
        "outer_fold": outer_fold,
        "seed": seed,
        "primary_metric": primary_name,
        "public_validation_primary": float(recomputed[primary_name]),
        "validation_sample_count": len(arrays["prediction"]),
        "selected_epoch": int(checkpoint["selected_epoch"]),
        "feature_cache_sha256": retained_file_sha256,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "table_admissible": False,
        "target_dataset_exposure": False,
        "protected_test_opened": False,
        "run_dir": portable_path(directory),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    config = args.config or (METHOD_ROOT / "configs/downstream_public_v2.yaml")
    report = audit(args.run_dir, config)
    if args.output is not None:
        write_json(resolve_repo_path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
