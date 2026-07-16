#!/usr/bin/env python3
"""Post-hoc compromise and identifiability audit for an adaptive SSM run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PARAMETER_BOUNDS = {
    "kas": (0.25, 1.50),
    "kaf": (0.05, 0.90),
    "tau0": (0.60, 5.00),
    "alpha": (0.18, 0.55),
    "e0": (0.20, 0.65),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _driver_compromise(rows: Sequence[Mapping[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (str(row["condition_id"]), str(row["subject"]), int(row["heldout_trial"]), str(row["spatial_mode"]))
        grouped[key][str(row["model"])].append(float(row["shared_driver"]))
    fold_rows = []
    for key, models in sorted(grouped.items()):
        if "adaptive_joint" not in models or "adaptive_eeg_only" not in models:
            continue
        joint = np.asarray(models["adaptive_joint"], dtype=np.float64)
        eeg_only = np.asarray(models["adaptive_eeg_only"], dtype=np.float64)
        fold_rows.append({
            "condition_id": key[0],
            "subject": key[1],
            "heldout_trial": key[2],
            "spatial_mode": key[3],
            "joint_vs_eeg_only_driver_pcc": float(np.corrcoef(joint, eeg_only)[0, 1]),
            "fnirs_driver_shift_normalized_rms": float(np.std(joint - eeg_only) / max(float(np.std(eeg_only)), 1e-12)),
        })
    subject_groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        subject_groups[(row["condition_id"], row["subject"], row["spatial_mode"])].append(row)
    subject_rows = []
    for key, values in sorted(subject_groups.items()):
        subject_rows.append({
            "condition_id": key[0],
            "subject": key[1],
            "spatial_mode": key[2],
            "folds": len(values),
            "joint_vs_eeg_only_driver_pcc": float(np.mean([row["joint_vs_eeg_only_driver_pcc"] for row in values])),
            "fnirs_driver_shift_normalized_rms": float(np.mean([row["fnirs_driver_shift_normalized_rms"] for row in values])),
        })
    return fold_rows, subject_rows


def _local_global_contrast(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    selected = {
        (row["condition_id"], row["subject"], row["model"], row["spatial_mode"]): row
        for row in rows if row["model"] == "adaptive_joint"
    }
    metrics = ("r2", "pcc", "variance_ratio", "eeg_r2", "eeg_pcc")
    output = []
    subjects = sorted({(key[0], key[1], key[2]) for key in selected})
    for condition_id, subject, model in subjects:
        local = selected.get((condition_id, subject, model, "local"))
        global_row = selected.get((condition_id, subject, model, "global"))
        if local is None or global_row is None:
            continue
        row: dict[str, Any] = {"condition_id": condition_id, "subject": subject, "model": model}
        for metric in metrics:
            row[f"local_{metric}"] = float(local[metric])
            row[f"global_{metric}"] = float(global_row[metric])
            row[f"delta_local_minus_global_{metric}"] = float(local[metric]) - float(global_row[metric])
        output.append(row)
    return output


def _parameter_audit(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    output = []
    for parameter, (lower, upper) in PARAMETER_BOUNDS.items():
        values = np.asarray([float(row[parameter]) for row in rows], dtype=np.float64)
        boundary = np.isclose(values, lower, atol=1e-5) | np.isclose(values, upper, atol=1e-5)
        output.append({
            "parameter": parameter,
            "lower_bound": lower,
            "upper_bound": upper,
            "mean": float(np.mean(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "boundary_fits": int(np.count_nonzero(boundary)),
            "total_fits": len(values),
            "boundary_fraction": float(np.mean(boundary)),
        })
    for parameter in ("q_scale", "fnirs_noise_scale"):
        counts = Counter(str(row[parameter]) for row in rows)
        for value, count in sorted(counts.items(), key=lambda item: float(item[0])):
            output.append({
                "parameter": parameter,
                "lower_bound": "",
                "upper_bound": "",
                "mean": "",
                "minimum": value,
                "maximum": value,
                "boundary_fits": count,
                "total_fits": len(rows),
                "boundary_fraction": count / len(rows),
            })
    return output


def run(run_dir: Path) -> None:
    trajectories = _read_csv(run_dir / "trajectories.csv")
    subject_metrics = _read_csv(run_dir / "subject_metrics.csv")
    fit_parameters = _read_csv(run_dir / "fit_parameters.csv")
    compromise_folds, compromise_subjects = _driver_compromise(trajectories)
    contrasts = _local_global_contrast(subject_metrics)
    parameter_audit = _parameter_audit(fit_parameters)
    _write_csv(run_dir / "driver_compromise_fold_metrics.csv", compromise_folds)
    _write_csv(run_dir / "driver_compromise_subject_metrics.csv", compromise_subjects)
    _write_csv(run_dir / "local_global_contrast.csv", contrasts)
    _write_csv(run_dir / "parameter_identifiability_audit.csv", parameter_audit)

    compromise_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in compromise_subjects:
        compromise_groups[(row["condition_id"], row["spatial_mode"])].append(row)
    contrast_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in contrasts:
        contrast_groups[row["condition_id"]].append(row)
    lines = [
        "# Adaptive shared-driver post-hoc audit",
        "",
        "## Joint-versus-EEG-only driver compromise",
        "",
        "| Condition | Spatial | Subjects | Driver PCC | fNIRS-induced shift / EEG-only SD |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for key, values in sorted(compromise_groups.items()):
        lines.append(
            f"| {key[0]} | {key[1]} | {len(values)} | "
            f"{np.mean([row['joint_vs_eeg_only_driver_pcc'] for row in values]):.4f} | "
            f"{np.mean([row['fnirs_driver_shift_normalized_rms'] for row in values]):.4f} |"
        )
    lines.extend([
        "",
        "## Local-minus-global joint contrast",
        "",
        "| Condition | Subjects | delta HbO R2 | delta HbO PCC | delta variance ratio | delta EEG R2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for condition_id, values in sorted(contrast_groups.items()):
        lines.append(
            f"| {condition_id} | {len(values)} | "
            f"{np.mean([row['delta_local_minus_global_r2'] for row in values]):.4f} | "
            f"{np.mean([row['delta_local_minus_global_pcc'] for row in values]):.4f} | "
            f"{np.mean([row['delta_local_minus_global_variance_ratio'] for row in values]):.4f} | "
            f"{np.mean([row['delta_local_minus_global_eeg_r2'] for row in values]):.4f} |"
        )
    lines.extend([
        "",
        "## Identifiability warning",
        "",
        "The bounded optimizer converged numerically for all folds, but frequent physiological-parameter boundary solutions and selection of the smallest fNIRS-noise multiplier show that the fitted parameter values are not independently identifiable. The trajectories may be used as a regularized multimodal compromise candidate; `kas`, `kaf`, `tau0`, `alpha`, and `E0` must not be interpreted as recovered subject physiology.",
        "",
    ])
    (run_dir / "adaptive_posthoc_summary.md").write_text("\n".join(lines), encoding="utf-8")
    artifacts = [
        "driver_compromise_fold_metrics.csv",
        "driver_compromise_subject_metrics.csv",
        "local_global_contrast.csv",
        "parameter_identifiability_audit.csv",
        "adaptive_posthoc_summary.md",
    ]
    manifest = {
        "schema": "adaptive_shared_neural_ssm_posthoc_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "input_hashes": [
            {"path": name, "sha256": _sha256(run_dir / name)}
            for name in ("trajectories.csv", "subject_metrics.csv", "fit_parameters.csv")
        ] + [{"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())}],
        "artifacts": artifacts,
        "claim_boundary": [
            "post-hoc diagnostics, not preregistered endpoints",
            "driver shift quantifies fNIRS influence but not causal neural identification",
            "fitted physiological parameter values are not independently identifiable",
        ],
    }
    (run_dir / "adaptive_posthoc_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    main_manifest_path = run_dir / "manifest.json"
    main_manifest = json.loads(main_manifest_path.read_text(encoding="utf-8"))
    for artifact in [*artifacts, "adaptive_posthoc_manifest.json"]:
        if artifact not in main_manifest["artifacts"]:
            main_manifest["artifacts"].append(artifact)
    main_manifest["posthoc_analyses"] = {
        "adaptive_driver_compromise": {
            "manifest": "adaptive_posthoc_manifest.json",
            "summary": "adaptive_posthoc_summary.md",
        }
    }
    main_manifest_path.write_text(json.dumps(main_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().run_dir.resolve())
