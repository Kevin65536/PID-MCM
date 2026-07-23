#!/usr/bin/env python3
"""Materialize and optionally execute the registered E2 T0/T1/T2 suite."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _split_sha(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config["data"]["split"], sort_keys=True).encode("utf-8")
    ).hexdigest()


def resolve_row_config(
    base: Mapping[str, Any],
    row: str,
    seed: int,
    *,
    device: str | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base))
    suite = config.pop("e2_suite")
    if row not in suite["rows"]:
        raise KeyError(f"Unknown E2 row: {row}")
    row_cfg = suite["rows"][row]
    config["experiment"]["name"] = f"e2_{row.lower()}_seed{seed}_v1"
    config["training"]["seed"] = int(seed)
    if device is not None:
        config["training"]["device"] = str(device)
    config["loss"]["state"]["weight"] = float(row_cfg["state_weight"])
    config["loss"]["prototype"]["weight"] = float(row_cfg["prototype_weight"])
    for entry in ("local", "prototype"):
        config["loss"]["entry_routing"][entry] = {
            "eeg": list(row_cfg["eeg_coordinates"]),
            "fnirs": list(row_cfg["fnirs_coordinates"]),
        }
    evaluation = config["validation"]["e2_evaluation"]
    evaluation["row"] = row
    objectives = ["eeg_reconstruction", "fnirs_reconstruction", "eeg_balance", "fnirs_balance"]
    if float(row_cfg["state_weight"]) > 0.0:
        objectives.extend(("eeg_state", "fnirs_state"))
    if float(row_cfg["prototype_weight"]) > 0.0:
        objectives.extend(("eeg_prototype", "fnirs_prototype"))
    config["validation"]["gradient_audit"]["objectives"] = objectives
    config["validation"]["e2_row"] = row
    return config


def run(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).resolve()
    base = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    registered = base["e2_suite"]
    rows = list(args.rows or registered["rows"].keys())
    seeds = list(args.seeds or registered["seeds"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else REPO_ROOT / "experiments/runs/physiology_semantic_tokenizer/e2_semantic_objectives" / f"{stamp}_e2_suite_v1"
    )
    config_dir = suite_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    runs = []
    mode = "smoke" if args.smoke else "train"
    for row in rows:
        for seed in seeds:
            resolved = resolve_row_config(base, row, int(seed), device=args.device)
            resolved_path = config_dir / f"{row.lower()}_seed{seed}.yaml"
            resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
            run_dir = suite_dir / "runs" / f"{row.lower()}_seed{seed}"
            command = [
                sys.executable,
                str(REPO_ROOT / "experiments/train_physiology_semantic_tokenizer.py"),
                "--config", str(resolved_path),
                f"--{mode}",
                "--output-dir", str(run_dir),
            ]
            commands.append(command)
            runs.append({"row": row, "seed": int(seed), "config": str(resolved_path), "run_dir": str(run_dir)})
    protocol = {
        "schema": "physiology_semantic_e2_suite_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "primary_endpoint": base["validation"]["primary_endpoint"],
        "rows": rows,
        "seeds": [int(seed) for seed in seeds],
        "split_sha256": _split_sha(base),
        "protected_test_opened": False,
        "promotion_eligible": False,
        "runs": runs,
        "commands": commands,
    }
    _write_json(suite_dir / "suite_manifest.json", protocol)
    (suite_dir / "decision_protocol.yaml").write_text(
        yaml.safe_dump({
            "primary_endpoint": protocol["primary_endpoint"],
            "comparison": "T1 and T2 versus seed-matched T0 on validation subjects",
            "uncertainty": "subject bootstrap with seed consistency",
            "null": "train-target permutation for hard-token probes",
            "protected_test_policy": "never opened by E2 development suite",
            "retention_policy": "E2 cannot promote until the separate E6/G2 gate passes",
        }, sort_keys=False),
        encoding="utf-8",
    )
    _write_json(suite_dir / "metric_registry.json", {
        "primary": base["validation"]["primary_endpoint"],
        "secondary": [
            "continuous_latent_r2", "posterior_r2", "codebook_embedding_r2",
            "prototype_signature_stability", "reconstruction", "quantizer_health",
        ],
        "diagnostic": ["gradient_norm", "gradient_cosine_conflict"],
    })
    _write_json(suite_dir / "evidence_calibration.json", {
        "scope": "training_and_validation_only",
        "target_family": base["data"]["auxiliary_target"]["family"],
        "target_version": base["data"]["auxiliary_target"]["version"],
        "threshold_rule": "directional seed-consistent improvement over T0 and above shuffled-target q95",
        "protected_test_opened": False,
    })
    if args.execute:
        gate_path = REPO_ROOT / str(base["validation"]["target_family_gate_path"])
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if (
            not args.smoke
            and bool(gate.get("requires_e0_channel_aware_revalidation_before_formal_e2", False))
        ):
            raise RuntimeError(
                "Formal E2 execution is blocked by the target-family gate: rebuild and "
                "revalidate the adaptive teacher with current bad-channel masks first"
            )
        for command in commands:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
    print(json.dumps({"suite_dir": str(suite_dir), "run_count": len(runs), "executed": args.execute}, sort_keys=True))
    return suite_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/configs/physiology_semantic_tokenizer/e2_semantic_objective_suite.yaml",
    )
    parser.add_argument("--rows", nargs="+", choices=("T0", "T1", "T2"))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
