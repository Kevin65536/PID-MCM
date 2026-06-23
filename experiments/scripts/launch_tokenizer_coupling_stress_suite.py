#!/usr/bin/env python
"""Launch extreme global-coupling stress tests for source tokens."""

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
SUITE_PARENT = project_root / "experiments/runs/tokenizer_coupling_stress"
MAX_LAG_TOKENS = 5


@dataclass(frozen=True)
class StressCondition:
    key: str
    description: str
    coupling_weight: float
    gradient_target: str
    lag_focus_weight: float = 0.0
    joint_entropy_weight: float = 0.0
    pair_likelihood_weight: float = 0.0
    lag_evidence_weight: float = 0.0
    pair_temperature: float = 0.5


CONDITIONS = (
    StressCondition("S0", "no coupling baseline", 0.0, "none"),
    StressCondition(
        "S1",
        "sharp global coupling tensor only, tokenizer gradients blocked",
        1.0,
        "none",
        lag_focus_weight=5.0,
        joint_entropy_weight=5.0,
        pair_likelihood_weight=3.0,
        lag_evidence_weight=2.0,
        pair_temperature=0.35,
    ),
    StressCondition(
        "S2",
        "strong global coupling pressure routed to fNIRS tokens",
        0.5,
        "fnirs",
        lag_focus_weight=5.0,
        joint_entropy_weight=5.0,
        pair_likelihood_weight=3.0,
        lag_evidence_weight=2.0,
        pair_temperature=0.35,
    ),
    StressCondition(
        "S3",
        "ultra-strong global coupling pressure routed to both modalities",
        1.0,
        "both",
        lag_focus_weight=5.0,
        joint_entropy_weight=5.0,
        pair_likelihood_weight=3.0,
        lag_evidence_weight=2.0,
        pair_temperature=0.35,
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
    condition: StressCondition,
    *,
    k: str,
    seed: int,
    seed_index: int,
    smoke: bool,
    device: str,
    formal_epochs: int,
) -> tuple[str, dict[str, Any]]:
    base = training_base_for(k, seed_index)
    run_name = f"{k.lower()}_{condition.key.lower()}_global_coupling_stress_seed{seed}"
    cfg: dict[str, Any] = {
        "_base_": str(base["config"]),
        "experiment": {
            "name": run_name,
            "run_group": f"tokenizer_coupling_stress/{suite_name}/{'smoke' if smoke else 'tokenizer_interventions'}",
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
                "lag_focus_weight": condition.lag_focus_weight,
                "joint_entropy_weight": condition.joint_entropy_weight,
                "smoothness_weight": 0.0,
                "pair_likelihood_weight": condition.pair_likelihood_weight,
                "lag_evidence_weight": condition.lag_evidence_weight,
                "lag_evidence_temperature": 0.15,
                "effective_smoothness_weight": 0.0,
                "interaction_lag_sparsity_weight": 0.0,
                "local_residual_enabled": False,
                "context_residual_enabled": False,
                "pair_gradient_target": condition.gradient_target,
                "pair_temperature": condition.pair_temperature,
                "residualize_fnirs_marginal": False,
            },
            "interaction_aux": {"weight": 0.0, "direction": "eeg_to_fnirs", "stop_gradient": True},
            "shared_state_bottleneck": {"weight": 0.0, "dim": 32},
        },
        "training": {
            "batch_size": 256,
            "learning_rate": 7.5e-5,
            "min_lr": 1.0e-5,
            "epochs": 2 if smoke else int(formal_epochs),
            "alignment_warmup": {
                "enabled": condition.coupling_weight > 0.0,
                "start_epoch": 1,
                "end_epoch": 5 if not smoke else 2,
                "start_scale": 0.0,
                "end_scale": 1.0,
            },
            "validation": {"interval_epochs": 1, "start_epoch": 1, "max_batches": 1 if smoke else None},
            "checkpoint": {"save_every": 1 if smoke else 5},
            "early_stopping": {"enabled": False, "metric": "val_loss", "mode": "min", "start_epoch": 1},
            "gradient_attribution": {
                "enabled": True,
                "max_batches": 1,
                "interval_epochs": 1 if smoke else 5,
            },
        },
    }
    if smoke:
        cfg["training"]["max_train_batches"] = 2
        cfg["training"]["performance"] = {"tf32": True, "cuda_prefetch": False, "compile": {"enabled": False}}
        cfg["data"]["num_workers"] = 0
        cfg["data"]["dataloader"] = {"persistent_workers": False, "drop_last": False}
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
    formal_epochs: int,
    seeds: tuple[int, int],
) -> dict[str, Any]:
    suite_dir.mkdir(parents=True, exist_ok=True)
    suite_name = suite_dir.name
    for name in ("tokenizer_interventions", "queue_logs", "smoke"):
        (suite_dir / name).mkdir(parents=True, exist_ok=True)

    formal_by_gpu = {0: [], 1: []}
    smoke_by_gpu = {0: [], 1: []}
    configs: dict[str, str] = {}
    for seed_index, seed in enumerate(seeds):
        gpu = seed_index % 2
        device = f"cuda:{gpu}"
        for condition in CONDITIONS:
            run_name, cfg = training_config(
                suite_name,
                condition,
                k="K64",
                seed=seed,
                seed_index=seed_index,
                smoke=False,
                device=device,
                formal_epochs=formal_epochs,
            )
            path = suite_dir / "tokenizer_interventions/configs" / f"{run_name}.yaml"
            write_yaml(path, cfg)
            configs[str(path.relative_to(suite_dir))] = run_name
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
            )
            smoke_path = suite_dir / "smoke/configs" / f"{smoke_name}.yaml"
            write_yaml(smoke_path, smoke_cfg)
            smoke_by_gpu[gpu].append(training_command(smoke_path, f"smoke_{smoke_name}", smoke=True))

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
        "schema_version": "tokenizer_coupling_stress_suite_v1",
        "created_at": datetime.now().isoformat(),
        "suite_root": str(suite_dir),
        "git_commit": git_commit(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=project_root, text=True).strip()),
        "previous_capacity_suite": str(PREVIOUS_SUITE),
        "conditions": [condition.__dict__ for condition in CONDITIONS],
        "formal_epochs": int(formal_epochs),
        "seeds": list(seeds),
        "decision_key": {
            "forced_visual_pattern": "Gate3 MI and heatmaps improve strongly under sharp global prior",
            "degradation": "reconstruction/primary validation losses compared against S0 by seed",
            "gradient_conflict": "gradient cosine between reconstruction losses and source_coupling_* components",
        },
        "queues": {
            "smoke_gpu0": str(suite_dir / "queue_logs/smoke_gpu0.sh"),
            "smoke_gpu1": str(suite_dir / "queue_logs/smoke_gpu1.sh"),
            "formal_gpu0": str(suite_dir / "queue_logs/formal_gpu0.sh"),
            "formal_gpu1": str(suite_dir / "queue_logs/formal_gpu1.sh"),
        },
        "formal_configs": configs,
    }
    (suite_dir / "suite_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (suite_dir / "summary.json").write_text(json.dumps({"status": "initialized"}, indent=2) + "\n", encoding="utf-8")
    (suite_dir / "decision.json").write_text(json.dumps({"status": "pending_smoke"}, indent=2) + "\n", encoding="utf-8")
    (suite_dir / "report.md").write_text(
        "# Tokenizer Coupling Stress Suite\n\n"
        "Status: initialized. Run smoke queues first, then formal K64 stress queues.\n",
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
    parser.add_argument("--formal-epochs", type=int, default=20)
    parser.add_argument("--seeds", nargs=2, type=int, default=(20260633, 20260634))
    args = parser.parse_args()
    suite_dir = Path(args.suite_dir).resolve() if args.suite_dir else (
        SUITE_PARENT / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_global_coupling_stress_v1"
    )
    if args.mode in {"initialize", "all"}:
        initialize_suite(suite_dir, formal_epochs=args.formal_epochs, seeds=tuple(args.seeds))
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
