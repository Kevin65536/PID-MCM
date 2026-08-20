#!/usr/bin/env python3
"""Aggregate deterministic SSM observation screen shards and apply the VQ gate."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "ssm_observation_target_screen_multiseed_summary_v1"
ELIGIBLE_MODES = (
    "SSM-SELF",
    "SSM-SELF-XPRED-0.02",
    "SSM-SELF-XPRED-0.05",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = list(rows)
    fields: list[str] = []
    for row in values:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def summarize(run_paths: Sequence[Path], output: Path) -> Path:
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    if len(run_paths) < 2:
        raise ValueError("multiseed summary requires at least two run shards")
    manifests = []
    combined: list[dict[str, str]] = []
    provenance: list[dict[str, str]] = []
    controls: list[dict[str, str]] = []
    for path in run_paths:
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "ssm_observation_target_screen_v1":
            raise ValueError(f"unexpected run schema: {path}")
        if manifest.get("protected_open") is not False:
            raise PermissionError("protected data cannot enter this summary")
        if manifest.get("determinism", {}).get("torch_deterministic_algorithms") is not True:
            raise ValueError("all summary shards must use deterministic algorithms")
        manifests.append(manifest)
        combined.extend(_read_csv(path / "results.csv"))
        provenance.extend(_read_csv(path / "teacher_provenance.csv"))
        controls.extend(_read_csv(path / "provenance_uncertainty_control.csv"))
    input_signatures = {
        tuple((item["path"], item["sha256"]) for item in manifest["inputs"])
        for manifest in manifests
    }
    if len(input_signatures) != 1:
        raise ValueError("run shards were not produced by identical source inputs")
    keys = [(row["task_id"], row["mode"], int(row["seed"])) for row in combined]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate task/mode/seed cells across run shards")
    seeds = sorted(set(seed for _, _, seed in keys))
    if len(seeds) < 3:
        raise ValueError("screen gate requires at least three seeds")
    metric_names = [
        key
        for key in combined[0]
        if key.startswith("selection_") or key == "representation_selection_score"
    ]
    summary_rows: list[dict[str, Any]] = []
    task_modes = sorted(set((task, mode) for task, mode, _ in keys))
    for task_id, mode in task_modes:
        rows = [
            row
            for row in combined
            if row["task_id"] == task_id and row["mode"] == mode
        ]
        if len(rows) != len(seeds):
            raise ValueError(f"incomplete seed matrix for {task_id}/{mode}")
        summary: dict[str, Any] = {
            "schema": SCHEMA,
            "task_id": task_id,
            "mode": mode,
            "seed_count": len(rows),
        }
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=1))
            summary[f"{metric}_min"] = float(np.min(values))
            summary[f"{metric}_max"] = float(np.max(values))
        summary["eeg_representation_pass_all_seeds"] = all(
            float(row["selection_eeg_clean_delta_r2_vs_condition_time_mean"]) > 0.0
            for row in rows
        )
        summary["fnirs_representation_pass_all_seeds"] = all(
            float(row["selection_fnirs_clean_delta_r2_vs_condition_time_mean"]) > 0.0
            for row in rows
        )
        summary["joint_representation_pass_all_seeds"] = bool(
            summary["eeg_representation_pass_all_seeds"]
            and summary["fnirs_representation_pass_all_seeds"]
        )
        summary_rows.append(summary)
    tasks = sorted(set(task for task, _, _ in keys))
    mode_gate: dict[str, bool] = {}
    for mode in ELIGIBLE_MODES:
        cells = [row for row in summary_rows if row["mode"] == mode]
        mode_gate[mode] = bool(
            len(cells) == len(tasks)
            and all(row["joint_representation_pass_all_seeds"] for row in cells)
        )
    vq_gate = any(mode_gate.values())
    control_balanced_accuracy = [
        float(row["subject_equal_balanced_accuracy"]) for row in controls
    ]
    gate = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tasks": tasks,
        "seeds": seeds,
        "eligible_modes": list(ELIGIBLE_MODES),
        "mode_pass": mode_gate,
        "advance_to_independent_k16_vq": bool(vq_gate),
        "decision_rule": (
            "advance only if one non-privileged SSM mode has positive EEG and fNIRS "
            "delta-R2 versus the fit-parameter condition-time mean in every task and seed"
        ),
        "q0_q1_status": "deferred" if not vq_gate else "eligible_not_yet_evidence",
        "protected_open": False,
        "provenance_control_balanced_accuracy_range": [
            float(min(control_balanced_accuracy)),
            float(max(control_balanced_accuracy)),
        ],
        "claim_scope": "deterministic fit-selection architecture/objective QC; not coupling evidence",
        "source_runs": [
            {
                "path": str(path),
                "manifest_sha256": _sha256(path / "manifest.json"),
                "results_sha256": _sha256(path / "results.csv"),
            }
            for path in run_paths
        ],
    }
    key_rows = [
        {
            "task_id": row["task_id"],
            "mode": row["mode"],
            "seed_count": row["seed_count"],
            "eeg_delta_r2_mean": row[
                "selection_eeg_clean_delta_r2_vs_condition_time_mean_mean"
            ],
            "eeg_delta_r2_std": row[
                "selection_eeg_clean_delta_r2_vs_condition_time_mean_std"
            ],
            "fnirs_delta_r2_mean": row[
                "selection_fnirs_clean_delta_r2_vs_condition_time_mean_mean"
            ],
            "fnirs_delta_r2_std": row[
                "selection_fnirs_clean_delta_r2_vs_condition_time_mean_std"
            ],
            "private_macro_f1_mean": row[
                "selection_private_only_subject_equal_macro_f1_mean"
            ],
            "private_plus_shared_macro_f1_mean": row[
                "selection_private_plus_shared_marginal_subject_equal_macro_f1_mean"
            ],
            "private_shared_interaction_macro_f1_mean": row[
                "selection_private_shared_interaction_subject_equal_macro_f1_mean"
            ],
            "interaction_macro_f1_increment_mean": row[
                "selection_interaction_macro_f1_increment_mean"
            ],
            "interaction_macro_f1_increment_std": row[
                "selection_interaction_macro_f1_increment_std"
            ],
            "eeg_pass_all_seeds": row["eeg_representation_pass_all_seeds"],
            "fnirs_pass_all_seeds": row["fnirs_representation_pass_all_seeds"],
            "joint_pass_all_seeds": row["joint_representation_pass_all_seeds"],
        }
        for row in summary_rows
    ]
    output.mkdir(parents=True)
    _write_csv(output / "combined_seed_results.csv", combined)
    _write_csv(output / "key_endpoints.csv", key_rows)
    _write_csv(output / "multiseed_summary.csv", summary_rows)
    _write_csv(output / "teacher_provenance_rows.csv", provenance)
    _write_csv(output / "provenance_uncertainty_control_rows.csv", controls)
    _write_json(output / "vq_stage_gate.json", gate)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        summarize(
            [Path(value).resolve() for value in args.runs],
            Path(args.output).resolve(),
        )
    )
