#!/usr/bin/env python
"""Launch context-conditioned source-token coupling experiments."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


project_root = Path(__file__).resolve().parents[2]
PREVIOUS_SUITE = project_root / "experiments/runs/tokenizer_coupling_capacity/20260617_185619_codebook_mission_bias_v1"
SUITE_PARENT = project_root / "experiments/runs/tokenizer_context_coupling"
MAX_LAG_TOKENS = 5


@dataclass(frozen=True)
class ContextCondition:
    key: str
    description: str
    coupling_weight: float = 0.0
    context_enabled: bool = False
    gradient_target: str = "none"
    context_states: int = 4
    router_type: str = "learned"


CONDITIONS = (
    ContextCondition("C0", "no coupling baseline"),
    ContextCondition(
        "C1",
        "global residual diagnostic, tokenizer frozen",
        coupling_weight=0.01,
        context_enabled=True,
        gradient_target="none",
        context_states=1,
        router_type="uniform",
    ),
    ContextCondition(
        "C2",
        "train context residual/router only, tokenizer frozen",
        coupling_weight=0.01,
        context_enabled=True,
        gradient_target="none",
    ),
    ContextCondition(
        "C3",
        "context residual coupling, gradient_target=fnirs",
        coupling_weight=0.01,
        context_enabled=True,
        gradient_target="fnirs",
    ),
    ContextCondition(
        "C4",
        "context residual coupling, gradient_target=both",
        coupling_weight=0.01,
        context_enabled=True,
        gradient_target="both",
    ),
    ContextCondition(
        "C5",
        "context residual coupling, C=8 confirmatory",
        coupling_weight=0.01,
        context_enabled=True,
        gradient_target="fnirs",
        context_states=8,
    ),
)


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def capacity_specs() -> list[dict[str, Any]]:
    capacity = PREVIOUS_SUITE / "capacity_sweep"
    specs = []
    for run_dir in sorted(capacity.glob("s2_croce_local_highwl_v2_capacity_k*_coupling0_seed*")):
        checkpoint = run_dir / "checkpoints/best_model.pt"
        config = run_dir / "config.yaml"
        if not checkpoint.exists() or not config.exists():
            continue
        key = "K256" if "k256" in run_dir.name else (
            "K128" if "k128" in run_dir.name else ("K64" if "k64" in run_dir.name else "K32")
        )
        specs.append({"key": key, "name": run_dir.name, "checkpoint": checkpoint, "config": config})
    return specs


def training_base_for(k: str, seed_index: int) -> dict[str, Any]:
    seed = 20260619 + seed_index
    candidates = [
        spec for spec in capacity_specs()
        if spec["key"] == k and f"seed{seed}" in spec["name"]
    ]
    if not candidates:
        candidates = [spec for spec in capacity_specs() if spec["key"] == k]
    if not candidates:
        raise RuntimeError(f"No completed capacity run for {k}")
    return candidates[0]


def training_config(
    suite_name: str,
    condition: ContextCondition,
    *,
    k: str,
    seed: int,
    seed_index: int,
    smoke: bool,
    device: str,
    formal_epochs: int = 80,
    checkpoint_save_every: int = 20,
    early_metric: str = "val_loss",
    context_entropy_weight: float = 0.0,
    context_balance_weight: float = 0.1,
) -> tuple[str, dict[str, Any]]:
    base = training_base_for(k, seed_index)
    run_name = f"{k.lower()}_{condition.key.lower()}_context_residual_coupling_seed{seed}"
    cfg: dict[str, Any] = {
        "_base_": str(base["config"]),
        "experiment": {
            "name": run_name,
            "run_group": f"tokenizer_context_coupling/{suite_name}/{'smoke' if smoke else 'tokenizer_interventions'}",
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
                "weight": condition.coupling_weight,
                "max_lag_tokens": MAX_LAG_TOKENS,
                "lag_focus_weight": 0.0,
                "smoothness_weight": 0.0,
                "pair_likelihood_weight": 0.0,
                "lag_evidence_weight": 0.0,
                "effective_smoothness_weight": 0.0,
                "interaction_lag_sparsity_weight": 0.0,
                "local_residual_enabled": False,
                "pair_gradient_target": "none",
                "context_residual_enabled": condition.context_enabled,
                "context_states": condition.context_states,
                "context_rank": 16,
                "context_router_type": condition.router_type,
                "context_pair_weight": 1.0,
                "context_entropy_weight": context_entropy_weight,
                "context_balance_weight": context_balance_weight,
                "context_residual_l1_weight": 0.001,
                "context_gradient_target": condition.gradient_target,
            },
            "interaction_aux": {"weight": 0.0, "direction": "eeg_to_fnirs", "stop_gradient": True},
            "shared_state_bottleneck": {"weight": 0.0, "dim": 32},
        },
        "training": {
            "batch_size": 256,
            "learning_rate": 1e-4,
            "min_lr": 2.5e-5,
            "epochs": 2 if smoke else int(formal_epochs),
            "alignment_warmup": {
                "enabled": True,
                "start_epoch": 1,
                "end_epoch": 20,
                "start_scale": 0.0,
                "end_scale": 1.0,
            },
            "validation": {"interval_epochs": 1 if smoke else 2, "start_epoch": 1, "max_batches": 1 if smoke else None},
            "checkpoint": {"save_every": 1 if smoke else int(checkpoint_save_every)},
            "early_stopping": {
                "enabled": True,
                "patience": 40,
                "metric": early_metric,
                "mode": "min",
                "start_epoch": 1,
            },
        },
    }
    if smoke:
        cfg["training"]["max_train_batches"] = 2
        cfg["training"]["performance"] = {"tf32": True, "cuda_prefetch": False, "compile": {"enabled": False}}
        cfg["data"]["num_workers"] = 0
        cfg["data"]["dataloader"] = {"persistent_workers": False, "drop_last": False}
    if condition.key in {"C1", "C2"}:
        cfg["training"]["trainable_parameter_prefixes"] = [
            "context_coupling_eeg_factors",
            "context_coupling_fnirs_factors",
            "context_coupling_router.",
        ]
    return run_name, cfg


def training_command(config: Path, run_name: str, *, smoke: bool) -> str:
    suffix = " --skip-post-analysis" if smoke else ""
    return (
        "bash experiments/scripts/launch_training_nohup.sh --task source-observation-tokenizer "
        f"--foreground --config {q(config)} --run-name {q(run_name)}{suffix}"
    )


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


def initialize_suite(
    suite_dir: Path,
    *,
    condition_keys: set[str] | None = None,
    formal_epochs: int = 80,
    checkpoint_save_every: int = 20,
    early_metric: str = "val_loss",
    context_entropy_weight: float = 0.0,
    context_balance_weight: float = 0.1,
) -> dict[str, Any]:
    suite_dir.mkdir(parents=True, exist_ok=True)
    suite_name = suite_dir.name
    for name in ("tokenizer_interventions", "gated_k128_configs", "queue_logs", "smoke"):
        (suite_dir / name).mkdir(parents=True, exist_ok=True)

    selected_conditions = tuple(
        condition for condition in CONDITIONS
        if condition_keys is None or condition.key in condition_keys
    )
    if not selected_conditions:
        raise ValueError(f"No context coupling conditions selected from {sorted(condition_keys or [])}")

    formal_by_gpu = {0: [], 1: []}
    smoke_by_gpu = {0: [], 1: []}
    configs: dict[str, str] = {}
    for seed_index, seed in enumerate((20260631, 20260632)):
        gpu = seed_index % 2
        device = f"cuda:{gpu}"
        for condition in selected_conditions:
            run_name, cfg = training_config(
                suite_name,
                condition,
                k="K64",
                seed=seed,
                seed_index=seed_index,
                smoke=False,
                device=device,
                formal_epochs=formal_epochs,
                checkpoint_save_every=checkpoint_save_every,
                early_metric=early_metric,
                context_entropy_weight=context_entropy_weight,
                context_balance_weight=context_balance_weight,
            )
            path = suite_dir / "tokenizer_interventions/configs" / f"{run_name}.yaml"
            write_yaml(path, cfg)
            configs[str(path.relative_to(suite_dir))] = run_name
            if condition.key != "C5":
                formal_by_gpu[gpu].append(training_command(path, run_name, smoke=False))

            smoke_name, smoke_cfg = training_config(
                suite_name,
                condition,
                k="K64",
                seed=seed,
                seed_index=seed_index,
                smoke=True,
                device=device,
                formal_epochs=formal_epochs,
                checkpoint_save_every=checkpoint_save_every,
                early_metric=early_metric,
                context_entropy_weight=context_entropy_weight,
                context_balance_weight=context_balance_weight,
            )
            smoke_path = suite_dir / "smoke/configs" / f"{smoke_name}.yaml"
            write_yaml(smoke_path, smoke_cfg)
            if condition.key != "C5":
                smoke_by_gpu[gpu].append(training_command(smoke_path, f"smoke_{smoke_name}", smoke=True))

            gated_name, gated_cfg = training_config(
                suite_name,
                condition,
                k="K128",
                seed=seed,
                seed_index=seed_index,
                smoke=False,
                device=device,
                formal_epochs=formal_epochs,
                checkpoint_save_every=checkpoint_save_every,
                early_metric=early_metric,
                context_entropy_weight=context_entropy_weight,
                context_balance_weight=context_balance_weight,
            )
            gated_path = suite_dir / "gated_k128_configs" / f"{gated_name}.yaml"
            write_yaml(gated_path, gated_cfg)

    for gpu in (0, 1):
        write_queue(
            suite_dir / f"queue_logs/smoke_gpu{gpu}.sh",
            smoke_by_gpu[gpu],
            suite_dir / f"queue_logs/smoke_gpu{gpu}.done",
        )
        write_queue(
            suite_dir / f"queue_logs/formal_gpu{gpu}.sh",
            formal_by_gpu[gpu],
            suite_dir / f"queue_logs/formal_gpu{gpu}.done",
        )

    manifest = {
        "schema_version": "tokenizer_context_coupling_suite_v1",
        "created_at": datetime.now().isoformat(),
        "suite_root": str(suite_dir),
        "git_commit": git_commit(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=project_root, text=True).strip()),
        "previous_capacity_suite": str(PREVIOUS_SUITE),
        "max_lag_tokens": MAX_LAG_TOKENS,
        "conditions": [condition.__dict__ for condition in selected_conditions],
        "default_training_k": "K64",
        "gated_training_k": "K128",
        "formal_epochs": int(formal_epochs),
        "checkpoint_save_every": int(checkpoint_save_every),
        "early_metric": early_metric,
        "context_entropy_weight": float(context_entropy_weight),
        "context_balance_weight": float(context_balance_weight),
        "queues": {
            "smoke_gpu0": str(suite_dir / "queue_logs/smoke_gpu0.sh"),
            "smoke_gpu1": str(suite_dir / "queue_logs/smoke_gpu1.sh"),
            "formal_gpu0": str(suite_dir / "queue_logs/formal_gpu0.sh"),
            "formal_gpu1": str(suite_dir / "queue_logs/formal_gpu1.sh"),
        },
        "formal_configs": configs,
        "decision_key": {
            "visual_pattern": "best_mi_above_shuffle >= 1.25 * C0",
            "held_out_utility": "best_loso_gain > 0 on test",
            "token_health": "source perplexity >= 0.8 * C0 and primary loss not degraded",
            "leakage_guard": "context/source/subject leakage must be audited before promotion",
        },
    }
    (suite_dir / "suite_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (suite_dir / "summary.json").write_text(json.dumps({"status": "initialized"}, indent=2) + "\n", encoding="utf-8")
    (suite_dir / "decision.json").write_text(json.dumps({"status": "pending_smoke"}, indent=2) + "\n", encoding="utf-8")
    (suite_dir / "report.md").write_text(
        "# Tokenizer Context Coupling Suite\n\n"
        "Status: initialized. Run smoke queues first, then formal K64 C0-C4 queues. "
        "C5 and gated K128 configs are generated for confirmatory use only.\n",
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
    parser.add_argument("--mode", choices=("initialize", "smoke", "launch", "all"), default="initialize")
    parser.add_argument("--conditions", default="C0,C1,C2,C3,C4,C5")
    parser.add_argument("--formal-epochs", type=int, default=80)
    parser.add_argument("--checkpoint-save-every", type=int, default=20)
    parser.add_argument("--early-metric", default="val_loss")
    parser.add_argument("--context-entropy-weight", type=float, default=0.0)
    parser.add_argument("--context-balance-weight", type=float, default=0.1)
    args = parser.parse_args()
    suite_dir = Path(args.suite_dir).resolve() if args.suite_dir else (
        SUITE_PARENT / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_context_residual_coupling_v1"
    )
    condition_keys = {
        item.strip().upper()
        for item in str(args.conditions).split(",")
        if item.strip()
    }
    if args.mode in {"initialize", "all"}:
        initialize_suite(
            suite_dir,
            condition_keys=condition_keys,
            formal_epochs=args.formal_epochs,
            checkpoint_save_every=args.checkpoint_save_every,
            early_metric=args.early_metric,
            context_entropy_weight=args.context_entropy_weight,
            context_balance_weight=args.context_balance_weight,
        )
    pids: dict[str, int] = {}
    if args.mode in {"smoke", "all"}:
        for gpu in (0, 1):
            code = run_queue(
                suite_dir / f"queue_logs/smoke_gpu{gpu}.sh",
                suite_dir / f"queue_logs/smoke_gpu{gpu}.log",
                detached=False,
            )
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
