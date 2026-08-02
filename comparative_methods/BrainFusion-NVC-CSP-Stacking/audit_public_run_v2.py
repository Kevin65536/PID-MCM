#!/usr/bin/env python3
"""Independently audit one retained public BrainFusion development run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import f1_score
import torch


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
ADAPTER_ROOT = METHOD_ROOT / "adapters"
for import_path in (REPO_ROOT, METHOD_ROOT, ADAPTER_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from alignment_data import BrainFusionPublicView, stable_hash
from brainfusion_gpu.pipeline import BrainFusionFoldPipeline
from comparative_methods.audit_public_preflight import sha256_file
from run_public_development_v2 import (
    DEFAULT_CONFIG,
    _fold_membership,
    _materialize,
    diverse_balanced_subset,
    load_runner_config,
    write_json,
)


DEFAULT_OUTPUT = METHOD_ROOT / "evidence/public_development_v2/pilot_audit.json"


def _public_json(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if "protected" in {part.lower() for part in resolved.parts}:
        raise PermissionError(f"refusing protected BrainFusion audit input: {resolved}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("protected_test_opened", False):
        raise PermissionError(f"artifact is invalid or reports protected access: {resolved}")
    return value


def audit_run(
    report_path: Path, *, config_path: Path = DEFAULT_CONFIG, device: str = "cuda:1"
) -> dict[str, Any]:
    report = _public_json(report_path)
    if report.get("schema") != "brainfusion_public_development_run_v2":
        raise ValueError("unexpected BrainFusion public run schema")
    if report.get("status") != "pass" or report.get("table_admissible") is not False:
        raise ValueError("BrainFusion public run is not a passing development-only artifact")
    if report.get("protected_evaluation_authorized") is not False:
        raise PermissionError("BrainFusion public run claims protected authorization")
    config, resolved_config, alignment, _ = load_runner_config(config_path)
    if sha256_file(resolved_config) != report["config_sha256"]:
        raise RuntimeError("BrainFusion runner config hash drifted")
    for name, digest in report["source_file_sha256"].items():
        path = {
            "runner": METHOD_ROOT / "run_public_development_v2.py",
            "alignment_data": METHOD_ROOT / "alignment_data.py",
            "alignment_audit": METHOD_ROOT / "audit_alignment_v2.py",
            "features": ADAPTER_ROOT / "brainfusion_gpu/features.py",
            "stacking": ADAPTER_ROOT / "brainfusion_gpu/stacking.py",
            "pipeline": ADAPTER_ROOT / "brainfusion_gpu/pipeline.py",
        }[name]
        if sha256_file(path) != digest:
            raise RuntimeError(f"BrainFusion run source hash drifted: {name}")

    task = str(report["task"])
    outer_fold = int(report["outer_fold"])
    seed = int(report["seed"])
    inventory, train, validation, manifest_path, manifest_sha256 = _fold_membership(
        alignment, task=task, outer_fold=outer_fold
    )
    if manifest_sha256 != report["public_manifest_sha256"] or str(manifest_path) != report[
        "public_manifest_path"
    ]:
        raise RuntimeError("BrainFusion audited public manifest identity drifted")
    if report["mode"] == "smoke_only":
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
    view = BrainFusionPublicView(inventory)
    train_data = _materialize(view, inventory, train)
    validation_data = _materialize(view, inventory, validation)
    if stable_hash(train_data[4]) != report["train_sample_identity_sha256"]:
        raise RuntimeError("BrainFusion audited training membership drifted")
    if stable_hash(validation_data[4]) != report["validation_sample_identity_sha256"]:
        raise RuntimeError("BrainFusion audited validation membership drifted")
    if set(train_data[4]) & set(validation_data[4]):
        raise RuntimeError("BrainFusion audited public memberships overlap")

    predictions_path = Path(str(report["predictions_path"])).resolve()
    if sha256_file(predictions_path) != report["predictions_sha256"]:
        raise RuntimeError("BrainFusion public prediction artifact hash drifted")
    predictions = _public_json(predictions_path)
    if predictions.get("schema") != "brainfusion_public_predictions_v2":
        raise ValueError("unexpected BrainFusion prediction schema")
    if predictions["sample_ids"] != validation_data[4]:
        raise RuntimeError("BrainFusion prediction sample order drifted")
    target = validation_data[3].numpy()
    if predictions["targets"] != target.tolist():
        raise RuntimeError("BrainFusion retained public targets drifted")

    checkpoint_dir = report_path.resolve().parent / "checkpoint"
    if sha256_file(checkpoint_dir / "manifest.json") != report[
        "checkpoint_manifest_sha256"
    ]:
        raise RuntimeError("BrainFusion checkpoint manifest hash drifted")
    torch_device = torch.device(device)
    pipeline = BrainFusionFoldPipeline.load(checkpoint_dir, device=torch_device)
    tensors = [value.to(torch_device) for value in validation_data[:3]]
    replay_predictions = pipeline.predict(*tensors)
    replay_decisions = pipeline.decision_function(*tensors)
    retained_predictions = np.asarray(predictions["predictions"], dtype=np.int64)
    retained_decisions = np.asarray(predictions["decision_scores"], dtype=np.float64)
    if not np.array_equal(replay_predictions, retained_predictions):
        raise RuntimeError("BrainFusion independent checkpoint prediction replay differed")
    if not np.array_equal(replay_decisions, retained_decisions):
        raise RuntimeError("BrainFusion independent checkpoint score replay differed")
    metric = float(f1_score(target, replay_predictions, average="macro"))
    if not np.isclose(metric, float(report["validation_macro_f1"]), rtol=0.0, atol=1e-15):
        raise RuntimeError("BrainFusion retained public metric drifted")
    fit_state = pipeline.audit_state()
    if fit_state != report["fit_state"]:
        raise RuntimeError("BrainFusion retained fitted-state audit drifted")
    if fit_state["fit_sample_identity_sha256"] != report["train_sample_identity_sha256"]:
        raise RuntimeError("BrainFusion fitted state is not bound to public training membership")
    return {
        "schema": "brainfusion_public_run_artifact_audit_v2",
        "status": "pass",
        "method_id": report["method_id"],
        "task": task,
        "outer_fold": outer_fold,
        "seed": seed,
        "mode": report["mode"],
        "run_report_path": str(report_path.resolve().relative_to(REPO_ROOT)),
        "run_report_sha256": sha256_file(report_path),
        "membership_recomputed": True,
        "targets_recomputed": True,
        "metric_recomputed": True,
        "checkpoint_predictions_recomputed": True,
        "checkpoint_reload_exact": True,
        "train_sample_count": len(train_data[4]),
        "validation_sample_count": len(validation_data[4]),
        "validation_macro_f1": metric,
        "table_admissible": False,
        "protected_evaluation_authorized": False,
        "protected_test_opened": False,
        "audited_at": datetime.now(timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_run(args.report, config_path=args.config, device=args.device)
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
