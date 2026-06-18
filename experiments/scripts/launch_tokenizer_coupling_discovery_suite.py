#!/usr/bin/env python
"""Launch local coupling and codebook-discovery audits/experiments."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


project_root = Path(__file__).resolve().parents[2]
PREVIOUS_SUITE = project_root / "experiments/runs/tokenizer_coupling_capacity/20260617_185619_codebook_mission_bias_v1"
SUITE_PARENT = project_root / "experiments/runs/tokenizer_coupling_discovery"
MAX_LAG_TOKENS = 5


@dataclass(frozen=True)
class TrainingCondition:
    key: str
    description: str
    gradient_target: str | None = None
    local_coupling: bool = False
    interaction_aux: bool = False
    shared_bottleneck: bool = False


TRAINING_CONDITIONS = (
    TrainingCondition("T0", "continue coupling=0"),
    TrainingCondition("T1", "local residual coupling, gradient_target=none", "none", True),
    TrainingCondition("T2", "local residual coupling, gradient_target=both", "both", True),
    TrainingCondition("T3", "local residual coupling, gradient_target=eeg", "eeg", True),
    TrainingCondition("T4", "local residual coupling, gradient_target=fnirs", "fnirs", True),
    TrainingCondition("T5", "auxiliary EEG-to-fNIRS latent predictor", interaction_aux=True),
    TrainingCondition("T6", "shared-state bottleneck auxiliary", shared_bottleneck=True),
)


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def run_specs() -> list[dict[str, Any]]:
    capacity = PREVIOUS_SUITE / "capacity_sweep"
    specs = []
    for run_dir in sorted(capacity.glob("s2_croce_local_highwl_v2_capacity_k*_coupling0_seed*")):
        checkpoint = run_dir / "checkpoints/best_model.pt"
        if not checkpoint.exists():
            continue
        key = "K256" if "k256" in run_dir.name else ("K128" if "k128" in run_dir.name else ("K64" if "k64" in run_dir.name else "K32"))
        specs.append({
            "key": key,
            "name": run_dir.name,
            "run_dir": run_dir,
            "checkpoint": checkpoint,
            "config": run_dir / "config.yaml",
            "dense_coupling_allowed": key != "K256",
        })
    return specs


def training_base_for(k: str, seed_index: int) -> dict[str, Any]:
    candidates = [
        spec for spec in run_specs()
        if spec["key"] == k and f"seed{20260619 + seed_index}" in spec["name"]
    ]
    if not candidates:
        candidates = [spec for spec in run_specs() if spec["key"] == k]
    if not candidates:
        raise RuntimeError(f"No completed capacity run for {k}")
    return candidates[0]


def training_config(
    suite_name: str,
    condition: TrainingCondition,
    *,
    k: str,
    seed: int,
    seed_index: int,
    smoke: bool,
    device: str,
) -> tuple[str, dict[str, Any]]:
    base = training_base_for(k, seed_index)
    run_name = f"{k.lower()}_{condition.key.lower()}_local_codebook_interaction_seed{seed}"
    cfg: dict[str, Any] = {
        "_base_": str(base["config"]),
        "experiment": {
            "name": run_name,
            "run_group": f"tokenizer_coupling_discovery/{suite_name}/{'smoke' if smoke else 'tokenizer_interventions'}",
            "seed": seed,
            "device": device,
            "description": condition.description,
        },
        "data": {"seed": seed},
        "warm_start": {
            "checkpoint": str(base["checkpoint"]),
            "reset_coupling": True,
            "coupling_reset_std": 0.02,
        },
        "loss": {
            "coupling": {
                "weight": 0.0,
                "max_lag_tokens": MAX_LAG_TOKENS,
                "lag_focus_weight": 0.0,
                "smoothness_weight": 0.0,
                "pair_likelihood_weight": 0.0,
                "lag_evidence_weight": 0.0,
                "effective_smoothness_weight": 0.0,
                "interaction_lag_sparsity_weight": 0.0,
                "local_residual_enabled": False,
                "local_residual_pair_weight": 1.0,
                "local_residual_alpha": 0.5,
                "pair_gradient_target": "none",
            },
            "interaction_aux": {"weight": 0.0, "direction": "eeg_to_fnirs", "stop_gradient": True},
            "shared_state_bottleneck": {"weight": 0.0, "dim": 32},
        },
        "training": {
            "batch_size": 256,
            "learning_rate": 1e-4,
            "min_lr": 2.5e-5,
            "epochs": 2 if smoke else 80,
            "alignment_warmup": {
                "enabled": True,
                "start_epoch": 1,
                "end_epoch": 20,
                "start_scale": 0.0,
                "end_scale": 1.0,
            },
            "validation": {"interval_epochs": 1 if smoke else 2, "start_epoch": 1, "max_batches": 1 if smoke else None},
            "checkpoint": {"save_every": 1 if smoke else 20},
        },
    }
    if smoke:
        cfg["training"]["max_train_batches"] = 2
        cfg["training"]["performance"] = {"tf32": True, "cuda_prefetch": False, "compile": {"enabled": False}}
        cfg["data"]["num_workers"] = 0
        cfg["data"]["dataloader"] = {"persistent_workers": False, "drop_last": False}
    if condition.local_coupling:
        cfg["loss"]["coupling"].update({
            "weight": 0.03,
            "local_residual_enabled": True,
            "pair_gradient_target": condition.gradient_target,
            "effective_smoothness_weight": 0.03,
            "interaction_lag_sparsity_weight": 0.03,
        })
    if condition.interaction_aux:
        cfg["loss"]["interaction_aux"].update({"weight": 0.05, "direction": "eeg_to_fnirs", "stop_gradient": True})
    if condition.shared_bottleneck:
        cfg["loss"]["shared_state_bottleneck"].update({"weight": 0.05, "dim": 32})
    return run_name, cfg


def training_command(config: Path, run_name: str, *, smoke: bool) -> str:
    suffix = " --skip-post-analysis" if smoke else ""
    return (
        "bash experiments/scripts/launch_training_nohup.sh --task source-observation-tokenizer "
        f"--foreground --config {q(config)} --run-name {q(run_name)}{suffix}"
    )


def maybe(command: str, sentinel: Path) -> str:
    return f"if [ ! -e {q(sentinel)} ]; then {command}; fi"


def audit_commands(suite_dir: Path) -> list[str]:
    commands = []
    for spec in run_specs():
        name = spec["name"]
        checkpoint = spec["checkpoint"]
        export_dir = PREVIOUS_SUITE / "token_exports" / name
        position_dir = PREVIOUS_SUITE / "position_audit" / name
        frozen_dir = PREVIOUS_SUITE / "frozen_pairing_audit" / name
        commands.append(maybe(
            f".venv/bin/python experiments/scripts/export_coupling_audit_data.py --checkpoint {q(checkpoint)} "
            f"--output-dir {q(export_dir)} --label {q(name)} --device cuda:0",
            export_dir / "manifest.json",
        ))
        commands.append(maybe(
            f".venv/bin/python experiments/scripts/analyze_coupling_identifiability.py --export-dir {q(export_dir)} "
            f"--output-dir {q(position_dir)} --permutations 500 --bootstraps 1000 --max-lag-tokens {MAX_LAG_TOKENS}",
            position_dir / "summary.json",
        ))
        if spec["key"] == "K256":
            commands.append(maybe(
                f".venv/bin/python experiments/scripts/fit_frozen_coupling_models.py --export-dir {q(export_dir)} "
                f"--output-dir {q(frozen_dir)} --steps 500 --bootstraps 1000 --factorized-only "
                f"--factorized-ranks 16 32 --max-lag-tokens {MAX_LAG_TOKENS}",
                frozen_dir / "frozen_model_results.json",
            ))
        commands.append(
            f".venv/bin/python experiments/scripts/analyze_codebook_geometry.py --checkpoint {q(checkpoint)} "
            f"--output-dir {q(suite_dir / 'codebook_geometry' / name)} --run-name {q(name)}"
        )
        if spec["key"] in {"K64", "K128", "K256"}:
            commands.append(
                f".venv/bin/python experiments/scripts/analyze_information_drop.py --checkpoint {q(checkpoint)} "
                f"--export-dir {q(export_dir)} --output-dir {q(suite_dir / 'information_drop_audit' / name)} "
                f"--run-name {q(name)} --splits val test --max-tokens 50000"
            )
            commands.append(
                f".venv/bin/python experiments/scripts/analyze_local_coupling.py --export-dir {q(export_dir)} "
                f"--output-dir {q(suite_dir / 'local_coupling_audit' / name)} --run-name {q(name)} "
                f"--max-lag-tokens {MAX_LAG_TOKENS}"
            )
    commands.append(f".venv/bin/python experiments/scripts/finalize_tokenizer_coupling_capacity_suite.py --suite-dir {q(PREVIOUS_SUITE)}")
    return commands


def write_queue(path: Path, commands: list[str], done_file: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {q(project_root)}",
            "export PYTHONUNBUFFERED=1",
            *commands,
            f"touch {q(done_file)}",
        ]) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def initialize_suite(suite_dir: Path) -> dict[str, Any]:
    suite_dir.mkdir(parents=True, exist_ok=True)
    suite_name = suite_dir.name
    for name in (
        "codebook_geometry", "information_drop_audit", "local_coupling_audit",
        "tokenizer_interventions", "gated_k128_configs", "queue_logs", "smoke",
    ):
        (suite_dir / name).mkdir(parents=True, exist_ok=True)

    audit_queue = suite_dir / "queue_logs/audit_queue.sh"
    write_queue(audit_queue, audit_commands(suite_dir), suite_dir / "queue_logs/audit.done")

    formal_by_gpu = {0: [], 1: []}
    smoke_by_gpu = {0: [], 1: []}
    configs: dict[str, str] = {}
    for seed_index, seed in enumerate((20260621, 20260622)):
        gpu = seed_index % 2
        device = f"cuda:{gpu}"
        for condition in TRAINING_CONDITIONS:
            run_name, cfg = training_config(
                suite_name, condition, k="K64", seed=seed, seed_index=seed_index,
                smoke=False, device=device,
            )
            path = suite_dir / "tokenizer_interventions/configs" / f"{run_name}.yaml"
            write_yaml(path, cfg)
            configs[str(path.relative_to(suite_dir))] = run_name
            formal_by_gpu[gpu].append(training_command(path, run_name, smoke=False))

            smoke_name, smoke_cfg = training_config(
                suite_name, condition, k="K64", seed=seed, seed_index=seed_index,
                smoke=True, device=device,
            )
            smoke_path = suite_dir / "smoke/configs" / f"{smoke_name}.yaml"
            write_yaml(smoke_path, smoke_cfg)
            smoke_by_gpu[gpu].append(training_command(smoke_path, f"smoke_{smoke_name}", smoke=True))

            gated_name, gated_cfg = training_config(
                suite_name, condition, k="K128", seed=seed, seed_index=seed_index,
                smoke=False, device=device,
            )
            gated_path = suite_dir / "gated_k128_configs" / f"{gated_name}.yaml"
            write_yaml(gated_path, gated_cfg)

    for gpu in (0, 1):
        write_queue(suite_dir / f"queue_logs/smoke_gpu{gpu}.sh", smoke_by_gpu[gpu], suite_dir / f"queue_logs/smoke_gpu{gpu}.done")
        write_queue(suite_dir / f"queue_logs/formal_gpu{gpu}.sh", formal_by_gpu[gpu], suite_dir / f"queue_logs/formal_gpu{gpu}.done")

    manifest = {
        "schema_version": "tokenizer_coupling_discovery_v1",
        "created_at": datetime.now().isoformat(),
        "suite_root": str(suite_dir),
        "git_commit": git_commit(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=project_root, text=True).strip()),
        "previous_capacity_suite": str(PREVIOUS_SUITE),
        "max_lag_tokens": MAX_LAG_TOKENS,
        "training_conditions": [condition.__dict__ for condition in TRAINING_CONDITIONS],
        "default_training_k": "K64",
        "gated_training_k": "K128",
        "queues": {
            "audit": str(audit_queue),
            "smoke_gpu0": str(suite_dir / "queue_logs/smoke_gpu0.sh"),
            "smoke_gpu1": str(suite_dir / "queue_logs/smoke_gpu1.sh"),
            "formal_gpu0": str(suite_dir / "queue_logs/formal_gpu0.sh"),
            "formal_gpu1": str(suite_dir / "queue_logs/formal_gpu1.sh"),
        },
        "formal_configs": configs,
    }
    (suite_dir / "suite_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (suite_dir / "summary.json").write_text(json.dumps({"status": "initialized"}, indent=2) + "\n", encoding="utf-8")
    (suite_dir / "decision.json").write_text(json.dumps({"status": "pending_audits"}, indent=2) + "\n", encoding="utf-8")
    (suite_dir / "report.md").write_text(
        "# Tokenizer Coupling Discovery Suite\n\n"
        "Status: initialized. Run audit queue first, then smoke/formal K64 intervention queues.\n",
        encoding="utf-8",
    )
    return manifest


def run_queue(path: Path, log_path: Path, detached: bool) -> int:
    log = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        ["bash", str(path)],
        cwd=project_root,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=detached,
    )
    log.close()
    if detached:
        return process.pid
    return process.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir")
    parser.add_argument("--mode", choices=("initialize", "audit", "smoke", "launch", "all"), default="initialize")
    args = parser.parse_args()
    suite_dir = Path(args.suite_dir).resolve() if args.suite_dir else (
        SUITE_PARENT / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_local_codebook_interaction_v1"
    )
    if args.mode in {"initialize", "all"}:
        initialize_suite(suite_dir)
    pids: dict[str, int] = {}
    if args.mode in {"audit", "all"}:
        code = run_queue(suite_dir / "queue_logs/audit_queue.sh", suite_dir / "queue_logs/audit_queue.log", detached=False)
        if code:
            raise SystemExit(code)
    if args.mode in {"smoke", "all"}:
        for gpu in (0, 1):
            code = run_queue(suite_dir / f"queue_logs/smoke_gpu{gpu}.sh", suite_dir / f"queue_logs/smoke_gpu{gpu}.log", detached=False)
            if code:
                raise SystemExit(code)
    if args.mode in {"launch", "all"}:
        for gpu in (0, 1):
            pids[f"gpu{gpu}"] = run_queue(
                suite_dir / f"queue_logs/formal_gpu{gpu}.sh",
                suite_dir / f"queue_logs/formal_gpu{gpu}.log",
                detached=True,
            )
        (suite_dir / "queue_logs/pids.json").write_text(json.dumps(pids, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"suite_dir": str(suite_dir), "pids": pids}, indent=2))


if __name__ == "__main__":
    main()
