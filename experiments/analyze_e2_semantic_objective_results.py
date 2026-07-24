#!/usr/bin/env python3
"""Apply the registered E2 decision rule to frozen development artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "physiology_semantic_e2_development_decision_v1"
ROWS = ("T0", "T1", "T2")
SEMANTIC_ROWS = ("T1", "T2")
MODALITIES = ("eeg", "fnirs")


def _hard_payload(run: Mapping[str, Any], modality: str) -> Mapping[str, Any]:
    return run["representations"][modality]["hard_id"]


def _subject_scores(
    run: Mapping[str, Any],
    modalities: Sequence[str],
    *,
    optional: bool = False,
) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for modality in modalities:
        payload = (
            run["optional_representations"][modality]["hard_id"]
            if optional
            else _hard_payload(run, modality)
        )
        for subject, coordinates in payload["subject_r2"].items():
            values[str(subject)].extend(float(value) for value in coordinates)
    return {
        subject: float(np.mean(coordinates))
        for subject, coordinates in values.items()
    }


def _paired_subject_deltas(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    modalities: Sequence[str],
    *,
    optional: bool = False,
) -> dict[str, float]:
    left = _subject_scores(baseline, modalities, optional=optional)
    right = _subject_scores(candidate, modalities, optional=optional)
    if set(left) != set(right):
        raise RuntimeError("Paired E2 runs do not share validation subjects")
    return {subject: right[subject] - left[subject] for subject in sorted(left)}


def _bootstrap_fixed_seed_mean(
    deltas_by_seed: Mapping[int, Mapping[str, float]],
    *,
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    observed_by_seed = {
        str(seed): float(np.mean(list(deltas.values())))
        for seed, deltas in sorted(deltas_by_seed.items())
    }
    draws = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        seed_draws = []
        for seed in sorted(deltas_by_seed):
            values = np.asarray(list(deltas_by_seed[seed].values()), dtype=np.float64)
            indices = rng.integers(0, len(values), size=len(values))
            seed_draws.append(float(np.mean(values[indices])))
        draws[iteration] = float(np.mean(seed_draws))
    point = float(np.mean(list(observed_by_seed.values())))
    return {
        "subject_mean_delta_by_seed": observed_by_seed,
        "fixed_seed_subject_mean_delta": point,
        "subject_bootstrap_iterations": iterations,
        "subject_bootstrap_ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "positive_seed_count": sum(value > 0.0 for value in observed_by_seed.values()),
        "seed_count": len(observed_by_seed),
    }


def _prototype_summary(path: Path) -> dict[str, Any]:
    pairs = json.loads(path.read_text(encoding="utf-8"))["pairs"]
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    counts: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in pairs:
        key = (str(row["row"]), str(row["modality"]))
        grouped[key].append(float(row["mean_cosine"]))
        counts[key].append(int(row["matched_count"]))
    return {
        row: {
            modality: {
                "mean_matched_cosine": float(np.mean(grouped[(row, modality)])),
                "minimum_matched_cosine": float(np.min(grouped[(row, modality)])),
                "mean_matched_count": float(np.mean(counts[(row, modality)])),
            }
            for modality in MODALITIES
        }
        for row in ROWS
    }


def _quantizer_health(run: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = Path(str(run["run_dir"]))
    final = json.loads(
        (run_dir / "diagnostics" / "quantizer_health.json").read_text(encoding="utf-8")
    )
    history = [
        json.loads(line)
        for line in (
            run_dir / "diagnostics" / "quantizer_health.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    thresholds = {
        "eeg": {
            "minimum_effective_codes": 32.0,
            "minimum_active_fraction": 0.50,
        },
        "fnirs": {
            "minimum_effective_codes": 24.0,
            "minimum_active_fraction": 0.75,
        },
    }
    payload: dict[str, Any] = {}
    all_passed = True
    for modality in MODALITIES:
        retention = [
            float(row["validation"][modality]["total_revivals"])
            for row in history
            if int(row["global_step"]) > 200
        ]
        checks = {
            "retention_total_revivals_constant": bool(
                retention and len(set(retention)) == 1
            ),
            "minimum_effective_codes_pass": (
                float(final[modality]["effective_codes"])
                >= thresholds[modality]["minimum_effective_codes"]
            ),
            "minimum_active_fraction_pass": (
                float(final[modality]["epoch_active_fraction"])
                >= thresholds[modality]["minimum_active_fraction"]
            ),
            "effective_rank_pass": float(final[modality]["effective_rank"]) == 64.0,
            "nearest_neighbor_cosine_pass": (
                float(final[modality]["nearest_neighbor_cosine"]) <= 0.99
            ),
            "quantization_strength_pass": (
                float(final[modality]["quantization_strength"]) == 1.0
            ),
            "final_revival_count_pass": int(final[modality]["revived_codes"]) == 0,
        }
        modality_passed = all(checks.values())
        all_passed &= modality_passed
        payload[modality] = {
            **checks,
            "passed": modality_passed,
            "epoch_active_codes": int(final[modality]["epoch_active_codes"]),
            "effective_codes": float(final[modality]["effective_codes"]),
            "nearest_neighbor_cosine": float(
                final[modality]["nearest_neighbor_cosine"]
            ),
        }
    return {"passed": all_passed, "modalities": payload}


def analyze(
    evaluation_dir: Path,
    output_dir: Path,
    *,
    bootstrap_iterations: int,
    seed: int,
) -> Path:
    decoding_path = evaluation_dir / "state_decoding.json"
    stability_path = evaluation_dir / "prototype_stability.json"
    evaluation_manifest_path = evaluation_dir / "manifest.json"
    decoding = json.loads(decoding_path.read_text(encoding="utf-8"))
    evaluation_manifest = json.loads(
        evaluation_manifest_path.read_text(encoding="utf-8")
    )
    if bool(evaluation_manifest.get("protected_test_opened", True)):
        raise RuntimeError("E2 evaluation unexpectedly opened protected test")
    runs = {
        (str(run["row"]), int(run["seed"])): run
        for run in decoding["runs"]
    }
    expected = {
        (row, seed_value)
        for row in ROWS
        for seed_value in sorted({key[1] for key in runs})
    }
    if set(runs) != expected:
        raise RuntimeError("E2 evaluation does not contain a complete row/seed grid")
    seeds = sorted({key[1] for key in runs})
    rng = np.random.default_rng(seed)

    required_comparisons: dict[str, Any] = {}
    bootstrap_rows: list[dict[str, Any]] = []
    coordinate_rows: list[dict[str, Any]] = []
    for candidate_row in SEMANTIC_ROWS:
        deltas_by_seed: dict[int, dict[str, float]] = {}
        pooled_by_seed: dict[str, float] = {}
        modality_by_seed: dict[str, dict[str, float]] = {}
        null_by_seed: dict[str, dict[str, bool]] = {}
        for seed_value in seeds:
            baseline = runs[("T0", seed_value)]
            candidate = runs[(candidate_row, seed_value)]
            deltas_by_seed[seed_value] = _paired_subject_deltas(
                baseline, candidate, MODALITIES
            )
            modality_delta: dict[str, float] = {}
            null_status: dict[str, bool] = {}
            for modality in MODALITIES:
                left = _hard_payload(baseline, modality)
                right = _hard_payload(candidate, modality)
                modality_delta[modality] = (
                    float(right["mean_r2"]) - float(left["mean_r2"])
                )
                null_status[modality] = bool(
                    right["shuffled_target_null"]["observed_above_q95"]
                )
                for index, coordinate in enumerate(right["coordinate_names"]):
                    coordinate_rows.append({
                        "comparison": f"{candidate_row}-T0",
                        "seed": seed_value,
                        "modality": modality,
                        "coordinate": coordinate,
                        "baseline_r2": float(left["coordinate_r2"][index]),
                        "candidate_r2": float(right["coordinate_r2"][index]),
                        "delta_r2": (
                            float(right["coordinate_r2"][index])
                            - float(left["coordinate_r2"][index])
                        ),
                    })
            modality_by_seed[str(seed_value)] = modality_delta
            null_by_seed[str(seed_value)] = null_status
            pooled_by_seed[str(seed_value)] = float(
                np.mean(list(modality_delta.values()))
            )
        bootstrap = _bootstrap_fixed_seed_mean(
            deltas_by_seed,
            iterations=bootstrap_iterations,
            rng=rng,
        )
        directional_pass = all(value > 0.0 for value in pooled_by_seed.values())
        null_pass = all(
            all(modality.values()) for modality in null_by_seed.values()
        )
        required_comparisons[candidate_row] = {
            "seed_matched_pooled_endpoint_delta": pooled_by_seed,
            "seed_matched_modality_delta": modality_by_seed,
            "hard_token_above_null": null_by_seed,
            "directionally_consistent_improvement": directional_pass,
            "all_required_modalities_above_null": null_pass,
            "paired_subject_bootstrap": bootstrap,
            "required_endpoint_passed": directional_pass and null_pass,
        }
        for seed_value, subject_deltas in deltas_by_seed.items():
            for subject, value in subject_deltas.items():
                bootstrap_rows.append({
                    "comparison": f"{candidate_row}-T0",
                    "seed": seed_value,
                    "subject": subject,
                    "endpoint": "required_hard_token_all_coordinates",
                    "delta_r2": value,
                })

    optional_deltas_by_seed: dict[int, dict[str, float]] = {}
    optional_pooled_by_seed: dict[str, float] = {}
    required_t2_vs_t1: dict[str, float] = {}
    for seed_value in seeds:
        t1 = runs[("T1", seed_value)]
        t2 = runs[("T2", seed_value)]
        optional_deltas_by_seed[seed_value] = _paired_subject_deltas(
            t1, t2, ("eeg",), optional=True
        )
        optional_pooled_by_seed[str(seed_value)] = (
            float(t2["optional_representations"]["eeg"]["hard_id"]["mean_r2"])
            - float(t1["optional_representations"]["eeg"]["hard_id"]["mean_r2"])
        )
        required_t2_vs_t1[str(seed_value)] = float(np.mean([
            float(_hard_payload(t2, modality)["mean_r2"])
            - float(_hard_payload(t1, modality)["mean_r2"])
            for modality in MODALITIES
        ]))
        for subject, value in optional_deltas_by_seed[seed_value].items():
            bootstrap_rows.append({
                "comparison": "T2-T1",
                "seed": seed_value,
                "subject": subject,
                "endpoint": "optional_eeg_s_hard_token",
                "delta_r2": value,
            })
    optional_bootstrap = _bootstrap_fixed_seed_mean(
        optional_deltas_by_seed,
        iterations=bootstrap_iterations,
        rng=rng,
    )
    optional_pass = (
        all(value > 0.0 for value in optional_pooled_by_seed.values())
        and all(value >= 0.0 for value in required_t2_vs_t1.values())
    )

    prototype = _prototype_summary(stability_path)
    prototype_non_decrease = {
        row: {
            modality: (
                prototype[row][modality]["mean_matched_cosine"]
                >= prototype["T0"][modality]["mean_matched_cosine"]
            )
            for modality in MODALITIES
        }
        for row in SEMANTIC_ROWS
    }
    health = {
        f"{row}_seed{seed_value}": _quantizer_health(runs[(row, seed_value)])
        for row in ROWS
        for seed_value in seeds
    }
    all_health_passed = all(value["passed"] for value in health.values())
    gradient_audit = json.loads(
        (evaluation_dir / "gradient_entry_audit.json").read_text(encoding="utf-8")
    )
    gradient_passed = all(
        bool(row.get("audit", {}).get("all_contracts_passed", False))
        for row in gradient_audit["runs"]
    )
    semantic_pass = {
        row: (
            bool(required_comparisons[row]["required_endpoint_passed"])
            and all(prototype_non_decrease[row].values())
            and all_health_passed
            and gradient_passed
        )
        for row in SEMANTIC_ROWS
    }
    selected_row = None
    if semantic_pass["T1"]:
        selected_row = "T2" if semantic_pass["T2"] and optional_pass else "T1"
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_dir": str(evaluation_dir),
        "run_count": len(runs),
        "seeds": seeds,
        "protected_test_opened": False,
        "promotion_eligible": False,
        "primary_endpoint": "heldout_hard_token_registered_signature_mean_r2",
        "required_comparisons": required_comparisons,
        "optional_t2_vs_t1": {
            "seed_matched_optional_delta": optional_pooled_by_seed,
            "seed_matched_required_delta": required_t2_vs_t1,
            "paired_subject_bootstrap": optional_bootstrap,
            "optional_endpoint_passed": optional_pass,
        },
        "prototype_stability": prototype,
        "prototype_non_decrease_vs_t0": prototype_non_decrease,
        "quantizer_health": {
            "source_thresholds": "E1 diverse-farthest v22 retention confirmation",
            "all_runs_passed": all_health_passed,
            "runs": health,
        },
        "gradient_entry_all_runs_passed": gradient_passed,
        "semantic_row_pass": semantic_pass,
        "selected_semantic_row": selected_row,
        "decision": (
            "no_semantic_row_admitted_retain_T0"
            if selected_row is None
            else f"development_semantic_evidence_{selected_row}"
        ),
        "claim_boundary": (
            "E2 development only; no G3 promotion, E6/G2 retention promotion, "
            "or coupling claim"
        ),
        "prototype_parquet_written": bool(
            evaluation_manifest.get("prototype_parquet_written", False)
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "paired_subject_deltas.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["comparison", "seed", "subject", "endpoint", "delta_r2"],
        )
        writer.writeheader()
        writer.writerows(bootstrap_rows)
    with (output_dir / "coordinate_deltas.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "comparison", "seed", "modality", "coordinate",
                "baseline_r2", "candidate_r2", "delta_r2",
            ],
        )
        writer.writeheader()
        writer.writerows(coordinate_rows)
    lines = [
        "# E2 development decision",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Selected semantic row: `{selected_row}`",
        f"- Quantizer health across all runs: `{all_health_passed}`",
        f"- Gradient-entry contracts across all runs: `{gradient_passed}`",
        "- Protected test opened: `False`",
        "- Promotion eligible: `False`",
        "",
        "T1/T2 did not satisfy the registered, seed-consistent required hard-token "
        "endpoint. T2 also did not show seed-consistent optional EEG s improvement.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = analyze(
        Path(args.evaluation_dir).resolve(),
        Path(args.output_dir).resolve(),
        bootstrap_iterations=int(args.bootstrap_iterations),
        seed=int(args.seed),
    )
    print(json.dumps({"output_dir": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
