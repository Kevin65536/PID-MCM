#!/usr/bin/env python3
"""Select an E2 semantic weight from training-gradient audits only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import yaml


OBJECTIVES = ("eeg_state", "eeg_prototype", "fnirs_state", "fnirs_prototype")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _shared_gradient_norm(
    objective: Mapping[str, Any],
    modality: str,
) -> float:
    prefixes = (
        f"{modality}_branch.patch_embedding",
        f"{modality}_branch.local_encoder",
        f"{modality}_branch.semantic_head",
    )
    squared = sum(
        float(value) ** 2
        for name, value in objective["parameter_gradient_norms"].items()
        if str(name).startswith(prefixes)
    )
    return math.sqrt(squared)


def summarize_run(
    run_dir: Path,
    *,
    minimum_ratio: float,
    maximum_ratio: float,
) -> dict[str, Any]:
    config_path = run_dir / "resolved_config.yaml"
    audit_path = run_dir / "diagnostics" / "gradient_entry_audit.json"
    manifest_path = run_dir / "manifest.json"
    for path in (config_path, audit_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state_weight = float(config["loss"]["state"]["weight"])
    prototype_weight = float(config["loss"]["prototype"]["weight"])
    if not math.isclose(state_weight, prototype_weight, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"state/prototype weights differ in {run_dir}")

    ratios: dict[str, list[float]] = {name: [] for name in OBJECTIVES}
    for record in audit["records"]:
        objectives = record["objectives"]
        for name in OBJECTIVES:
            objective = objectives[name]
            if objective["status"] != "audited":
                continue
            modality = name.split("_", maxsplit=1)[0]
            reconstruction = objectives[f"{modality}_reconstruction"]
            denominator = _shared_gradient_norm(reconstruction, modality)
            if denominator <= 0.0:
                raise RuntimeError(
                    f"zero shared reconstruction gradient for {modality} in {run_dir}"
                )
            ratios[name].append(
                _shared_gradient_norm(objective, modality) / denominator
            )

    objective_rows = []
    for name in OBJECTIVES:
        values = ratios[name]
        if not values:
            objective_rows.append({
                "objective": name,
                "support": 0,
                "median_ratio": None,
                "minimum_ratio": None,
                "maximum_ratio": None,
                "ratio_pass": False,
            })
            continue
        center = float(median(values))
        objective_rows.append({
            "objective": name,
            "support": len(values),
            "median_ratio": center,
            "minimum_ratio": float(min(values)),
            "maximum_ratio": float(max(values)),
            "ratio_pass": minimum_ratio <= center <= maximum_ratio,
        })
    contracts_passed = bool(audit.get("all_contracts_passed", False))
    candidate_passed = contracts_passed and all(
        row["ratio_pass"] for row in objective_rows
    )
    finite_centers = [
        float(row["median_ratio"])
        for row in objective_rows
        if row["median_ratio"] is not None
    ]
    worst_log10_imbalance = (
        max(abs(math.log10(value)) for value in finite_centers)
        if len(finite_centers) == len(OBJECTIVES)
        else math.inf
    )
    return {
        "run_dir": str(run_dir),
        "weight": state_weight,
        "seed": int(manifest["seed"]),
        "gradient_contracts_passed": contracts_passed,
        "candidate_passed": candidate_passed,
        "worst_log10_imbalance": worst_log10_imbalance,
        "objectives": objective_rows,
        "input_sha256": {
            "resolved_config.yaml": _sha256(config_path),
            "gradient_entry_audit.json": _sha256(audit_path),
            "manifest.json": _sha256(manifest_path),
        },
    }


def analyze(
    run_dirs: Sequence[Path],
    output_dir: Path,
    *,
    minimum_ratio: float,
    maximum_ratio: float,
) -> Path:
    if minimum_ratio <= 0.0 or maximum_ratio < minimum_ratio:
        raise ValueError("invalid gradient-ratio interval")
    candidates = [
        summarize_run(
            run_dir.resolve(),
            minimum_ratio=minimum_ratio,
            maximum_ratio=maximum_ratio,
        )
        for run_dir in run_dirs
    ]
    weights = [candidate["weight"] for candidate in candidates]
    if len(set(weights)) != len(weights):
        raise ValueError(f"duplicate semantic weights: {weights}")
    seeds = {candidate["seed"] for candidate in candidates}
    if len(seeds) != 1:
        raise ValueError(f"calibration runs must use one fixed seed: {sorted(seeds)}")
    admitted = [candidate for candidate in candidates if candidate["candidate_passed"]]
    selected = min(
        admitted,
        key=lambda row: (row["worst_log10_imbalance"], row["weight"]),
        default=None,
    )
    payload = {
        "schema": "physiology_semantic_e2_training_gradient_calibration_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "training_gradient_audit_only",
        "validation_target_decoding_read": False,
        "protected_test_opened": False,
        "seed": next(iter(seeds)),
        "admission_interval": {
            "minimum_batch_median_ratio": minimum_ratio,
            "maximum_batch_median_ratio": maximum_ratio,
        },
        "tie_break": "minimum worst absolute log10 ratio, then smaller weight",
        "candidates": sorted(candidates, key=lambda row: row["weight"]),
        "selected_weight": None if selected is None else selected["weight"],
        "calibration_passed": selected is not None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "gradient_ratios.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "weight", "seed", "objective", "support", "median_ratio",
            "minimum_ratio", "maximum_ratio", "ratio_pass",
            "gradient_contracts_passed", "candidate_passed",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in payload["candidates"]:
            for objective in candidate["objectives"]:
                writer.writerow({
                    "weight": candidate["weight"],
                    "seed": candidate["seed"],
                    **objective,
                    "gradient_contracts_passed": candidate["gradient_contracts_passed"],
                    "candidate_passed": candidate["candidate_passed"],
                })
    lines = [
        "# E2 training-gradient semantic-weight calibration",
        "",
        "Only fixed-seed training gradient audits were read. Validation target "
        "decoding and protected-test data were not accessed.",
        "",
        f"- Admission interval: batch-median shared-gradient ratio "
        f"`[{minimum_ratio:g}, {maximum_ratio:g}]`",
        f"- Selected weight: "
        f"`{payload['selected_weight'] if payload['selected_weight'] is not None else 'none'}`",
        f"- Calibration passed: `{payload['calibration_passed']}`",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-ratio", type=float, default=0.1)
    parser.add_argument("--maximum-ratio", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = analyze(
        [Path(value) for value in args.run],
        Path(args.output_dir).resolve(),
        minimum_ratio=float(args.minimum_ratio),
        maximum_ratio=float(args.maximum_ratio),
    )
    print(json.dumps({"output_dir": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
