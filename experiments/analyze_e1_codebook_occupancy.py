#!/usr/bin/env python3
"""Aggregate matched E1 occupancy runs without opening protected test data."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import yaml


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _factor(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def summarize_run(run_dir: Path) -> dict[str, Any]:
    manifest = _read_json(run_dir / "manifest.json")
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    health = _read_json(run_dir / "diagnostics" / "quantizer_health.json")
    health_rows = _read_jsonl(run_dir / "diagnostics" / "quantizer_health.jsonl")
    validation_rows = _read_jsonl(run_dir / "metrics" / "validation.jsonl")
    quantizer = _factor(config, "model", "quantizer", default={})
    reconstruction = _factor(config, "loss", "reconstruction", default={})
    balance = _factor(config, "loss", "codebook_balance", default={})
    training = _factor(config, "training", default={})

    trajectory = []
    for row in health_rows:
        validation = row.get("validation", row.get("health", {}))
        trajectory.append(
            {
                "epoch": row.get("epoch"),
                "global_step": row.get("global_step"),
                "eeg": {
                    key: validation.get("eeg", {}).get(key)
                    for key in (
                        "epoch_active_codes",
                        "effective_codes",
                        "assignment_entropy",
                        "ema_active_fraction",
                        "quantization_strength",
                        "total_revivals",
                    )
                },
                "fnirs": {
                    key: validation.get("fnirs", {}).get(key)
                    for key in (
                        "epoch_active_codes",
                        "effective_codes",
                        "assignment_entropy",
                        "ema_active_fraction",
                        "quantization_strength",
                        "total_revivals",
                    )
                },
            }
        )

    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "status": manifest.get("status"),
        "global_step": manifest.get("global_step"),
        "best_validation": manifest.get("best_validation"),
        "protected_test_opened": manifest.get("protected_test_opened"),
        "seed": manifest.get("seed"),
        "implementation_snapshot": _read_json(run_dir / "implementation_snapshot.json"),
        "factors": {
            "codebook_size": quantizer.get("codebook_size"),
            "assignment": quantizer.get("assignment", "euclidean"),
            "normalize_latents": quantizer.get("normalize_latents", False),
            "kmeans_init": quantizer.get("kmeans_init", False),
            "ema_decay": quantizer.get("decay"),
            "revive_dead_codes": quantizer.get("revive_dead_codes", False),
            "revival_strategy": quantizer.get("revival_strategy", "top_error"),
            "revival_count_prior": quantizer.get("revival_count_prior", "threshold"),
            "revival_stop_after_steps": quantizer.get("revival_stop_after_steps"),
            "reconstruction_semantic_input": reconstruction.get("semantic_input", "expected"),
            "balance_weight": balance.get("weight", 0.0),
            "balance_temperature": balance.get("temperature", 1.0),
            "eeg_balance_temperature": balance.get(
                "eeg_temperature", balance.get("temperature", 1.0)
            ),
            "fnirs_balance_temperature": balance.get(
                "fnirs_temperature", balance.get("temperature", 1.0)
            ),
            "batch_size": training.get("batch_size"),
            "epochs": training.get("epochs"),
            "quantization_warmup": training.get("quantization_warmup", {"enabled": False}),
        },
        "final_validation": {
            "metrics": validation_rows[-1] if validation_rows else {},
            "eeg": health.get("eeg", {}),
            "fnirs": health.get("fnirs", {}),
        },
        "trajectory": trajectory,
    }


def _markdown(runs: list[dict[str, Any]]) -> str:
    lines = [
        "# E1 codebook occupancy comparison",
        "",
        "Training/validation-only evidence; no protected test split was evaluated.",
        "",
        "| run | status | steps | input | balance | geometry | init | batch | revival | EEG active/effective | fNIRS active/effective | best val |",
        "|---|---:|---:|---|---:|---|---|---:|---|---:|---:|---:|",
    ]
    for run in runs:
        factors = run["factors"]
        final = run["final_validation"]
        eeg = final["eeg"]
        fnirs = final["fnirs"]
        geometry = factors["assignment"] + ("+L2" if factors["normalize_latents"] else "")
        lines.append(
            "| {name} | {status} | {steps} | {input} | {balance} | {geometry} | {init} | {batch} | {revival} | {ea}/{ee:.2f} | {fa}/{fe:.2f} | {best:.4f} |".format(
                name=run["run_name"],
                status=run["status"],
                steps=run["global_step"],
                input=factors["reconstruction_semantic_input"],
                balance=(
                    f"{float(factors['balance_weight']):.3g}"
                    f"@T{float(factors['eeg_balance_temperature']):g}/"
                    f"{float(factors['fnirs_balance_temperature']):g}"
                ),
                geometry=geometry,
                init="kmeans" if factors["kmeans_init"] else "random",
                batch=factors["batch_size"],
                revival=(
                    f"{factors['revival_strategy']}/{factors['revival_count_prior']}"
                    + (
                        f"@stop{factors['revival_stop_after_steps']}"
                        if factors["revival_stop_after_steps"] is not None
                        else ""
                    )
                    if factors["revive_dead_codes"]
                    else "off"
                ),
                ea=int(eeg.get("epoch_active_codes", 0)),
                ee=float(eeg.get("effective_codes", 0.0)),
                fa=int(fnirs.get("epoch_active_codes", 0)),
                fe=float(fnirs.get("effective_codes", 0.0)),
                best=float(run["best_validation"]),
            )
        )
    lines.extend(
        [
            "",
            "Active codes count any code selected at least once over the validation epoch; effective codes are exp(assignment entropy) and are the stricter anti-collapse endpoint.",
            "",
        ]
    )
    return "\n".join(lines)


def _multiseed_groups(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        signature = json.dumps(run["factors"], sort_keys=True)
        grouped.setdefault(signature, []).append(run)

    result = []
    for members in grouped.values():
        if len({member["seed"] for member in members}) < 2:
            continue

        def values(modality: str, metric: str) -> list[float]:
            return [
                float(member["final_validation"][modality][metric])
                for member in members
            ]

        def stats(sequence: list[float]) -> dict[str, float]:
            return {
                "mean": statistics.mean(sequence),
                "sample_std": statistics.stdev(sequence),
                "min": min(sequence),
                "max": max(sequence),
            }

        result.append(
            {
                "factors": members[0]["factors"],
                "run_names": [member["run_name"] for member in members],
                "seeds": [member["seed"] for member in members],
                "eeg": {
                    metric: stats(values("eeg", metric))
                    for metric in ("epoch_active_codes", "effective_codes", "total_revivals")
                },
                "fnirs": {
                    metric: stats(values("fnirs", metric))
                    for metric in ("epoch_active_codes", "effective_codes", "total_revivals")
                },
                "best_validation": stats(
                    [float(member["best_validation"]) for member in members]
                ),
            }
        )
    return result


def _multiseed_markdown(groups: list[dict[str, Any]]) -> str:
    if not groups:
        return ""
    lines = [
        "## Multi-seed groups",
        "",
        "| seeds | balance | EEG effective mean ± SD [range] | fNIRS effective mean ± SD [range] | EEG revivals mean [range] | fNIRS revivals mean [range] | val mean ± SD |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        factors = group["factors"]
        eeg_effective = group["eeg"]["effective_codes"]
        fnirs_effective = group["fnirs"]["effective_codes"]
        eeg_revivals = group["eeg"]["total_revivals"]
        fnirs_revivals = group["fnirs"]["total_revivals"]
        validation = group["best_validation"]
        balance = (
            f"{float(factors['balance_weight']):.3g}"
            f"@T{float(factors['eeg_balance_temperature']):g}/"
            f"{float(factors['fnirs_balance_temperature']):g}"
        )
        lines.append(
            "| {seeds} | {balance} | {em:.2f} ± {es:.2f} [{elo:.2f}, {ehi:.2f}] | {fm:.2f} ± {fs:.2f} [{flo:.2f}, {fhi:.2f}] | {erm:.1f} [{erlo:.0f}, {erhi:.0f}] | {frm:.1f} [{frlo:.0f}, {frhi:.0f}] | {vm:.4f} ± {vs:.4f} |".format(
                seeds=", ".join(str(seed) for seed in group["seeds"]),
                balance=balance,
                em=eeg_effective["mean"], es=eeg_effective["sample_std"],
                elo=eeg_effective["min"], ehi=eeg_effective["max"],
                fm=fnirs_effective["mean"], fs=fnirs_effective["sample_std"],
                flo=fnirs_effective["min"], fhi=fnirs_effective["max"],
                erm=eeg_revivals["mean"], erlo=eeg_revivals["min"], erhi=eeg_revivals["max"],
                frm=fnirs_revivals["mean"], frlo=fnirs_revivals["min"], frhi=fnirs_revivals["max"],
                vm=validation["mean"], vs=validation["sample_std"],
            )
        )
    lines.extend(["", "Sample SD is reported descriptively; three seeds do not establish a population-level uncertainty interval.", ""])
    return "\n".join(lines)


def _retention_checks(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    for run in runs:
        stop = run["factors"].get("revival_stop_after_steps")
        if stop is None:
            continue
        retained = [
            row for row in run["trajectory"]
            if row.get("global_step") is not None and row["global_step"] > stop
        ]
        if not retained:
            continue
        modalities = {}
        for modality in ("eeg", "fnirs"):
            effective = [float(row[modality]["effective_codes"]) for row in retained]
            active = [int(row[modality]["epoch_active_codes"]) for row in retained]
            revivals = [float(row[modality]["total_revivals"]) for row in retained]
            modalities[modality] = {
                "start_effective_codes": effective[0],
                "final_effective_codes": effective[-1],
                "minimum_effective_codes": min(effective),
                "start_active_codes": active[0],
                "final_active_codes": active[-1],
                "minimum_active_codes": min(active),
                "start_total_revivals": revivals[0],
                "final_total_revivals": revivals[-1],
                "revival_count_constant": len(set(revivals)) == 1,
            }
        checks.append(
            {
                "run_name": run["run_name"],
                "revival_stop_after_steps": stop,
                "first_retention_step": retained[0]["global_step"],
                "final_retention_step": retained[-1]["global_step"],
                "retention_epochs": len(retained),
                **modalities,
            }
        )
    return checks


def _retention_markdown(checks: list[dict[str, Any]]) -> str:
    if not checks:
        return ""
    lines = [
        "## Post-revival retention checks",
        "",
        "| run | steps | frozen revivals EEG/fNIRS | EEG effective start → final (min) | fNIRS effective start → final (min) | active final EEG/fNIRS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for check in checks:
        eeg = check["eeg"]
        fnirs = check["fnirs"]
        frozen = eeg["revival_count_constant"] and fnirs["revival_count_constant"]
        lines.append(
            "| {run} | {start}–{final} | {er:.0f}/{fr:.0f} ({frozen}) | {es:.2f} → {ef:.2f} ({emin:.2f}) | {fs:.2f} → {ff:.2f} ({fmin:.2f}) | {ea}/{fa} |".format(
                run=check["run_name"],
                start=check["first_retention_step"], final=check["final_retention_step"],
                er=eeg["final_total_revivals"], fr=fnirs["final_total_revivals"],
                frozen="constant" if frozen else "changed",
                es=eeg["start_effective_codes"], ef=eeg["final_effective_codes"],
                emin=eeg["minimum_effective_codes"],
                fs=fnirs["start_effective_codes"], ff=fnirs["final_effective_codes"],
                fmin=fnirs["minimum_effective_codes"],
                ea=eeg["final_active_codes"], fa=fnirs["final_active_codes"],
            )
        )
    lines.extend(["", "Retention starts at the first completed validation epoch after the configured revival stop step.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = [summarize_run(path) for path in args.run_dir]
    if any(run["protected_test_opened"] for run in runs):
        raise RuntimeError("Refusing to aggregate a run that opened protected test data")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    multiseed_groups = _multiseed_groups(runs)
    retention_checks = _retention_checks(runs)
    payload = {
        "schema": "e1_codebook_occupancy_comparison_v2",
        "runs": runs,
        "multiseed_groups": multiseed_groups,
        "retention_checks": retention_checks,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = (
        _markdown(runs)
        + "\n"
        + _multiseed_markdown(multiseed_groups)
        + "\n"
        + _retention_markdown(retention_checks)
    )
    (args.output_dir / "summary.md").write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
