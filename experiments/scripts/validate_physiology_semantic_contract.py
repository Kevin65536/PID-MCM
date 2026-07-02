#!/usr/bin/env python3
"""Run P1 dry-run/smoke validation for the physiology-semantic data contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    PHYSIOLOGY_SEMANTIC_CACHE_SCHEMA,
    create_configured_multimodal_dataloaders,
    load_experiment_config,
)
from src.utils import write_json, write_yaml  # noqa: E402


DEFAULT_CONFIG = "physiology_semantic_tokenizer/p1_e0_contract_smoke.yaml"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "experiments" / "runs" / "physiology_semantic_tokenizer" / "e0_teacher_validity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("dry-run", "smoke"), default="dry-run")
    parser.add_argument("--output-root", default=str(DEFAULT_RUN_ROOT))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def is_finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def assert_shape(name: str, value: torch.Tensor, expected: list[int]) -> None:
    observed = list(value.shape)
    if observed != expected:
        raise AssertionError(f"{name} shape {observed} != expected {expected}")


def validate_item(item: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    assert item["contract"] == "physiology_semantic_v2"
    assert item["cache_schema_version"] == PHYSIOLOGY_SEMANTIC_CACHE_SCHEMA
    assert_shape("eeg", item["eeg"], list(expected["eeg"]))
    assert_shape("fnirs", item["fnirs"], list(expected["fnirs"]))
    assert_shape("state_mean", item["teacher"]["state_mean"], list(expected["state_mean"]))
    assert_shape("state_var", item["teacher"]["state_var"], list(expected["state_var"]))
    assert_shape(
        "neural_driver_eeg_rate",
        item["teacher"]["neural_driver_eeg_rate"],
        list(expected["neural_driver_eeg_rate"]),
    )

    tensor_fields = {
        "eeg": item["eeg"],
        "fnirs": item["fnirs"],
        "state_mean": item["teacher"]["state_mean"],
        "state_var": item["teacher"]["state_var"],
        "neural_driver_eeg_rate": item["teacher"]["neural_driver_eeg_rate"],
        "neural_driver_var_eeg_rate": item["teacher"]["neural_driver_var_eeg_rate"],
    }
    non_finite = [name for name, tensor in tensor_fields.items() if not is_finite_tensor(tensor)]
    if non_finite:
        raise AssertionError(f"Non-finite target-contract tensors: {non_finite}")
    if torch.any(item["teacher"]["state_var"] < 0):
        raise AssertionError("state_var contains negative values")
    if torch.any(item["teacher"]["neural_driver_var_eeg_rate"] < 0):
        raise AssertionError("neural_driver_var_eeg_rate contains negative values")

    eeg_error = torch.max(
        torch.abs(item["eeg"] - item["decomposition"]["eeg_source"] - item["decomposition"]["eeg_residual"])
    ).item()
    fnirs_error = torch.max(
        torch.abs(
            item["fnirs"]
            - item["decomposition"]["fnirs_source"]
            - item["decomposition"]["fnirs_residual"]
        )
    ).item()
    eeg_magnitude = max(
        1.0,
        float(item["eeg"].abs().max()),
        float(item["decomposition"]["eeg_source"].abs().max()),
        float(item["decomposition"]["eeg_residual"].abs().max()),
    )
    fnirs_magnitude = max(
        1.0,
        float(item["fnirs"].abs().max()),
        float(item["decomposition"]["fnirs_source"].abs().max()),
        float(item["decomposition"]["fnirs_residual"].abs().max()),
    )
    eeg_tolerance = 8.0 * np.finfo(np.float32).eps * eeg_magnitude
    fnirs_tolerance = 8.0 * np.finfo(np.float32).eps * fnirs_magnitude
    if eeg_error > eeg_tolerance or fnirs_error > fnirs_tolerance:
        raise AssertionError(
            "Additive contract error exceeds float32 forward-error bound: "
            f"eeg={eeg_error}/{eeg_tolerance}, fnirs={fnirs_error}/{fnirs_tolerance}"
        )

    teacher_valid = item["teacher"]["teacher_valid_mask"].bool()
    causal_valid = item["teacher"]["causal_valid_mask"].bool()
    if torch.any(causal_valid[:100]):
        raise AssertionError("The first 10 seconds were not invalidated by the crop-boundary causal mask.")
    return {
        "subject_id": int(item["subject_id"]),
        "cache_entry_id": item["cache_entry_id"],
        "eeg_shape": list(item["eeg"].shape),
        "fnirs_shape": list(item["fnirs"].shape),
        "state_shape": list(item["teacher"]["state_mean"].shape),
        "eeg_dtype": str(item["eeg"].dtype),
        "fnirs_dtype": str(item["fnirs"].dtype),
        "state_dtype": str(item["teacher"]["state_mean"].dtype),
        "teacher_valid_fraction": float(teacher_valid.float().mean().item()),
        "cache_valid_fraction": float(item["teacher"]["cache_valid_mask"].float().mean().item()),
        "causal_valid_fraction": float(causal_valid.float().mean().item()),
        "eeg_additive_max_abs_error": float(eeg_error),
        "fnirs_additive_max_abs_error": float(fnirs_error),
        "eeg_additive_tolerance": float(eeg_tolerance),
        "fnirs_additive_tolerance": float(fnirs_tolerance),
        "eeg_unit": item["eeg_unit"],
        "fnirs_units": list(item["fnirs_units"]),
        "eeg_sample_rate_hz": float(item["eeg_sample_rate_hz"]),
        "fnirs_sample_rate_hz": float(item["fnirs_sample_rate_hz"]),
    }


def cache_inventory(dataloaders: Mapping[str, Any]) -> list[dict[str, Any]]:
    paths = sorted(
        {
            entry.cache_path
            for loader in dataloaders.values()
            for entry in loader.dataset.entries
        }
    )
    inventory = []
    for cache_path in paths:
        cache_path = cache_path if cache_path.is_absolute() else (PROJECT_ROOT / cache_path).resolve()
        manifest_path = cache_path.parent / "cache_manifest.json"
        inventory.append(
            {
                "cache_path": str(cache_path.relative_to(PROJECT_ROOT)),
                "cache_sha256": sha256_file(cache_path),
                "manifest_path": str(manifest_path.relative_to(PROJECT_ROOT)),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
    return inventory


def main() -> None:
    args = parse_args()
    started_at = datetime.now().astimezone()
    config = load_experiment_config(args.config)
    dataloaders = create_configured_multimodal_dataloaders(config)
    split_subjects = {
        split: sorted({entry.subject_id for entry in loader.dataset.entries})
        for split, loader in dataloaders.items()
    }
    split_lengths = {split: len(loader.dataset) for split, loader in dataloaders.items()}
    dataset_metadata = {split: loader.dataset.get_gate0_metadata() for split, loader in dataloaders.items()}
    inventory = cache_inventory(dataloaders)

    sample_checks: dict[str, Any] = {}
    if args.mode == "smoke":
        expected = config["validation"]["expected_shapes"]
        sample_checks = {
            split: validate_item(loader.dataset[0], expected)
            for split, loader in dataloaders.items()
        }

    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    run_dir = run_dir / f"{timestamp}_p1_contract_{args.mode.replace('-', '_')}"
    for relative in (
        "checkpoints",
        "metrics",
        "diagnostics",
        "predictions",
        "figures",
        "figure_data",
    ):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)

    decision_protocol = {
        "version": "pst_e0_v1",
        "suite": "e0_teacher_validity",
        "phase": args.mode,
        "primary_endpoint": "held_out_posterior_predictive_error_vs_history_mean_baseline",
        "protected_test_metrics_opened": False,
        "gate_decision": "deferred",
        "reason": "P1 contract validation does not estimate the E0 scientific endpoint.",
    }
    metric_registry = {
        "version": "pst_e0_metrics_v1",
        "metrics": {
            "held_out_posterior_predictive_error_vs_history_mean_baseline": {"role": "primary", "measured": False},
            "additive_max_abs_error": {"role": "deterministic_correctness", "measured": args.mode == "smoke"},
            "teacher_valid_fraction": {"role": "diagnostic", "measured": args.mode == "smoke"},
            "cache_valid_fraction": {"role": "diagnostic", "measured": args.mode == "smoke"},
        },
    }
    evidence_calibration = {
        "version": "pst_e0_calibration_v1",
        "deterministic_tolerances": {
            "additive_max_abs_error": "8 * float32_epsilon * max_abs(input, source, residual, 1)"
        },
        "health_references": {},
        "scientific_effect_calibration": "not_started",
        "protected_data_boundary": "No predictive endpoint or protected-test metric is computed in P1 dry-run/smoke.",
    }

    write_yaml(run_dir / "config.yaml", config)
    write_yaml(run_dir / "resolved_config.yaml", config)
    write_yaml(run_dir / "decision_protocol.yaml", decision_protocol)
    write_json(run_dir / "metric_registry.json", metric_registry)
    write_json(run_dir / "evidence_calibration.json", evidence_calibration)
    write_json(run_dir / "diagnostics" / "data_contract.json", {
        "split_subjects": split_subjects,
        "split_lengths": split_lengths,
        "dataset_metadata": dataset_metadata,
        "sample_checks": sample_checks,
        "cache_inventory": inventory,
    })
    write_json(run_dir / "diagnostics" / "teacher_audit.json", {
        "phase": args.mode,
        "contract_checks_passed": True,
        "scientific_endpoint_measured": False,
        "sample_checks": sample_checks,
    })
    write_json(run_dir / "diagnostics" / "quantizer_health.json", {"status": "not_applicable_in_p1"})
    write_json(run_dir / "diagnostics" / "state_semantics.json", {"status": "not_evaluated_in_p1"})
    write_json(run_dir / "diagnostics" / "information_retention.json", {"status": "not_evaluated_in_p1"})
    (run_dir / "metrics" / "train.jsonl").write_text("", encoding="utf-8")
    (run_dir / "metrics" / "validation.jsonl").write_text("", encoding="utf-8")
    coverage_lines = ["split,subject_id,cache_valid_fraction,causal_valid_fraction,teacher_valid_fraction"]
    coverage_lines.extend(
        ",".join(
            (
                split,
                str(values["subject_id"]),
                str(values["cache_valid_fraction"]),
                str(values["causal_valid_fraction"]),
                str(values["teacher_valid_fraction"]),
            )
        )
        for split, values in sample_checks.items()
    )
    (run_dir / "diagnostics" / "mask_coverage.csv").write_text(
        "\n".join(coverage_lines) + "\n", encoding="utf-8"
    )

    finished_at = datetime.now().astimezone()
    command = " ".join(shlex.quote(value) for value in sys.argv)
    summary = {
        "mode": args.mode,
        "status": f"{args.mode.replace('-', '_')}_passed",
        "gate": "G0_not_evaluated",
        "split_subjects": split_subjects,
        "split_lengths": split_lengths,
        "sample_checks": sample_checks,
    }
    write_json(run_dir / "metrics" / "test_summary.json", summary)
    manifest = {
        "schema_version": "pst_run_manifest_v1",
        "git_commit": git_value("rev-parse", "HEAD"),
        "dirty_worktree": bool(git_value("status", "--porcelain")),
        "cache_schema_version": PHYSIOLOGY_SEMANTIC_CACHE_SCHEMA,
        "dataset_hash": stable_hash(inventory),
        "split_hash": stable_hash(split_subjects),
        "checkpoint_hashes": {},
        "seed": int(config["training"]["seed"]),
        "command": command,
        "start_time": started_at.isoformat(),
        "end_time": finished_at.isoformat(),
        "completion_status": summary["status"],
        "decision_protocol_sha256": sha256_file(run_dir / "decision_protocol.yaml"),
        "metric_registry_sha256": sha256_file(run_dir / "metric_registry.json"),
        "evidence_calibration_sha256": sha256_file(run_dir / "evidence_calibration.json"),
    }
    write_json(run_dir / "manifest.json", manifest)
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                "# P1 physiology-semantic data contract validation",
                "",
                f"- Mode: `{args.mode}`",
                f"- Status: `{summary['status']}`",
                "- Gate decision: `G0_not_evaluated`",
                f"- Split subjects: `{split_subjects}`",
                f"- Split sample counts: `{split_lengths}`",
                "- Scientific endpoint: not measured; E0 short/full formal remains pending.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
