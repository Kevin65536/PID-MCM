#!/usr/bin/env python3
"""Evaluate the registered E1 post-revival retention rule.

This evaluator reads training/validation artifacts only. It deliberately fails
closed when a run opened the protected test split, differs in registered
factors or implementation files, or does not cover the full registered seed
set.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml


MODALITIES = ("eeg", "fnirs")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _check(name: str, passed: bool, observed: Any, required: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "required": required,
    }


def _evaluate_run(
    run_dir: Path,
    registered_factors: dict[str, Any],
    rule: dict[str, Any],
) -> dict[str, Any]:
    manifest = _read_json(run_dir / "manifest.json")
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    snapshot = _read_json(run_dir / "implementation_snapshot.json")
    rows = _read_jsonl(run_dir / "diagnostics" / "quantizer_health.jsonl")
    run_factors = config.get("validation", {}).get("registered_factors", {})
    stop_step = int(registered_factors["revival_stop_after_steps"])
    retained = [
        row
        for row in rows
        if row.get("global_step") is not None and int(row["global_step"]) > stop_step
    ]

    checks = [
        _check(
            "training_complete",
            not rule.get("require_completed_runs", False)
            or manifest.get("status") == "training_complete",
            manifest.get("status"),
            "training_complete",
        ),
        _check(
            "protected_test_closed",
            manifest.get("protected_test_opened")
            == rule.get("protected_test_opened", False),
            manifest.get("protected_test_opened"),
            rule.get("protected_test_opened", False),
        ),
        _check(
            "registered_factors_match",
            not rule.get("require_registered_factors_match", False)
            or run_factors == registered_factors,
            run_factors,
            registered_factors,
        ),
        _check(
            "minimum_retention_epochs",
            len(retained) >= int(rule.get("minimum_retention_epochs", 1)),
            len(retained),
            int(rule.get("minimum_retention_epochs", 1)),
        ),
    ]

    modality_results: dict[str, Any] = {}
    for modality in MODALITIES:
        values = [row.get("validation", {}).get(modality, {}) for row in retained]
        if not values:
            modality_results[modality] = {
                "passed": False,
                "checks": [_check("retention_rows_present", False, 0, ">=1")],
            }
            continue

        revivals = [float(value["total_revivals"]) for value in values]
        effective = [float(value["effective_codes"]) for value in values]
        final = values[-1]
        modality_checks = [
            _check(
                "total_revivals_constant",
                not rule.get("total_revivals_constant_in_retention_window", False)
                or len(set(revivals)) == 1,
                revivals,
                "constant",
            ),
            _check(
                "minimum_effective_codes",
                min(effective)
                >= float(rule["minimum_effective_codes_in_retention_window"][modality]),
                min(effective),
                float(rule["minimum_effective_codes_in_retention_window"][modality]),
            ),
            _check(
                "final_epoch_active_fraction",
                float(final["epoch_active_fraction"])
                >= float(rule["minimum_final_epoch_active_fraction"][modality]),
                float(final["epoch_active_fraction"]),
                float(rule["minimum_final_epoch_active_fraction"][modality]),
            ),
            _check(
                "final_effective_rank",
                int(final["effective_rank"])
                == int(rule["required_final_effective_rank"][modality]),
                int(final["effective_rank"]),
                int(rule["required_final_effective_rank"][modality]),
            ),
            _check(
                "final_nearest_neighbor_cosine",
                float(final["nearest_neighbor_cosine"])
                <= float(rule["maximum_final_nearest_neighbor_cosine"][modality]),
                float(final["nearest_neighbor_cosine"]),
                float(rule["maximum_final_nearest_neighbor_cosine"][modality]),
            ),
            _check(
                "final_quantization_strength",
                math.isclose(
                    float(final["quantization_strength"]),
                    float(rule["required_final_quantization_strength"][modality]),
                    rel_tol=0.0,
                    abs_tol=1.0e-8,
                ),
                float(final["quantization_strength"]),
                float(rule["required_final_quantization_strength"][modality]),
            ),
        ]
        modality_results[modality] = {
            "passed": all(check["passed"] for check in modality_checks),
            "first_retention_step": int(retained[0]["global_step"]),
            "final_retention_step": int(retained[-1]["global_step"]),
            "retention_epochs": len(retained),
            "minimum_effective_codes": min(effective),
            "final_effective_codes": effective[-1],
            "final_active_codes": int(final["epoch_active_codes"]),
            "final_active_fraction": float(final["epoch_active_fraction"]),
            "final_total_revivals": revivals[-1],
            "checks": modality_checks,
        }

    passed = all(check["passed"] for check in checks) and all(
        modality_results[modality]["passed"] for modality in MODALITIES
    )
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "seed": int(manifest["seed"]),
        "passed": passed,
        "checks": checks,
        "modalities": modality_results,
        "implementation_files_sha256": snapshot.get("files_sha256", {}),
    }


def _markdown(decision: dict[str, Any]) -> str:
    lines = [
        "# E1 post-revival retention gate",
        "",
        f"**Decision: {'PASS' if decision['passed'] else 'FAIL'}**",
        "",
        "Training/validation-only decision. The protected test split remained closed.",
        "",
        "| seed | run | pass | retention epochs | EEG min/final effective | fNIRS min/final effective | final active EEG/fNIRS | revivals EEG/fNIRS |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in decision["runs"]:
        eeg = run["modalities"]["eeg"]
        fnirs = run["modalities"]["fnirs"]
        lines.append(
            "| {seed} | {run} | {passed} | {epochs} | {emin:.2f}/{efinal:.2f} | "
            "{fmin:.2f}/{ffinal:.2f} | {eactive}/{factive} | {erev:.0f}/{frev:.0f} |".format(
                seed=run["seed"],
                run=run["run_name"],
                passed="PASS" if run["passed"] else "FAIL",
                epochs=eeg.get("retention_epochs", 0),
                emin=eeg.get("minimum_effective_codes", float("nan")),
                efinal=eeg.get("final_effective_codes", float("nan")),
                fmin=fnirs.get("minimum_effective_codes", float("nan")),
                ffinal=fnirs.get("final_effective_codes", float("nan")),
                eactive=eeg.get("final_active_codes", 0),
                factive=fnirs.get("final_active_codes", 0),
                erev=eeg.get("final_total_revivals", float("nan")),
                frev=fnirs.get("final_total_revivals", float("nan")),
            )
        )
    lines.extend(["", "## Cohort checks", ""])
    for check in decision["cohort_checks"]:
        lines.append(
            f"- {'PASS' if check['passed'] else 'FAIL'} — {check['name']}: "
            f"observed `{check['observed']}`, required `{check['required']}`"
        )
    failures = []
    for run in decision["runs"]:
        for check in run["checks"]:
            if not check["passed"]:
                failures.append(f"{run['run_name']}: {check['name']}")
        for modality in MODALITIES:
            for check in run["modalities"][modality]["checks"]:
                if not check["passed"]:
                    failures.append(f"{run['run_name']} {modality}: {check['name']}")
    if failures:
        lines.extend(["", "## Failed checks", ""])
        lines.extend(f"- {failure}" for failure in failures)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validation = config["validation"]
    calibration = validation["quantizer_health_calibration"]
    is_confirmation = "confirmation_rule" in calibration
    rule = calibration[
        "confirmation_rule" if is_confirmation else "decision_rule"
    ]
    if is_confirmation:
        expected_seeds = [
            int(calibration["calibration_seed"]),
            *[int(seed) for seed in calibration["confirmation_seeds"]],
        ]
    else:
        expected_seeds = [int(config["training"]["seed"])]
    runs = [
        _evaluate_run(path, validation["registered_factors"], rule)
        for path in args.run_dir
    ]
    observed_seeds = [run["seed"] for run in runs]
    snapshots = [run["implementation_files_sha256"] for run in runs]
    cohort_checks = [
        _check(
            "registered_seed_set",
            sorted(observed_seeds) == sorted(expected_seeds),
            sorted(observed_seeds),
            sorted(expected_seeds),
        ),
        _check(
            "implementation_snapshot_match",
            not rule.get("require_implementation_snapshot_match", False)
            or all(snapshot == snapshots[0] for snapshot in snapshots[1:]),
            len({json.dumps(snapshot, sort_keys=True) for snapshot in snapshots}),
            1,
        ),
    ]
    passed = (
        all(run["passed"] for run in runs)
        and all(check["passed"] for check in cohort_checks)
    )
    decision = {
        "schema": "e1_post_revival_retention_gate_v1",
        "passed": passed,
        "evidence_class": calibration["evidence_class"],
        "source": calibration["source"],
        "protected_test_opened": False,
        "registered_rule": rule,
        "expected_seeds": expected_seeds,
        "cohort_checks": cohort_checks,
        "runs": runs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "decision.md").write_text(
        _markdown(decision), encoding="utf-8"
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
