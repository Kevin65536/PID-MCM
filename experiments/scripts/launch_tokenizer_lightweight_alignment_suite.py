#!/usr/bin/env python
"""Launch causal lightweight pre-VQ alignment tokenizer experiments."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUITE_PARENT = PROJECT_ROOT / "experiments/runs/tokenizer_lightweight_alignment"
NEXT_STAGE_SUITE = (
    PROJECT_ROOT
    / "experiments/runs/tokenizer_next_stage/20260620_221822_k128_dim_cognitive_transfer_v1"
)
MAX_LAG_TOKENS = 5


@dataclass(frozen=True)
class AlignmentCondition:
    key: str
    description: str
    interaction_weight: float = 0.0
    interaction_stop_gradient: bool = True
    coupling_weight: float = 0.0
    global_residual_coupling: bool = False
    context_residual_coupling: bool = False
    shared_bottleneck_weight: float = 0.0


CONDITIONS = (
    AlignmentCondition("A0", "no-alignment baseline continuation"),
    AlignmentCondition(
        "A1",
        "detached EEG-to-fNIRS latent predictor",
        interaction_weight=0.05,
        interaction_stop_gradient=True,
    ),
    AlignmentCondition(
        "A2",
        "direct-gradient EEG-to-fNIRS latent predictor",
        interaction_weight=0.05,
        interaction_stop_gradient=False,
    ),
    AlignmentCondition(
        "A3",
        "direct-gradient latent predictor plus weak global residual coupling prior",
        interaction_weight=0.05,
        interaction_stop_gradient=False,
        coupling_weight=0.01,
        global_residual_coupling=True,
    ),
    AlignmentCondition(
        "A4",
        "direct-gradient latent predictor plus context residual coupling prior",
        interaction_weight=0.05,
        interaction_stop_gradient=False,
        coupling_weight=0.01,
        context_residual_coupling=True,
    ),
    AlignmentCondition(
        "A5",
        "shared bottleneck diagnostic",
        shared_bottleneck_weight=0.05,
    ),
)


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def base_specs() -> list[dict[str, Any]]:
    runs = NEXT_STAGE_SUITE / "vector_dim_sweep"
    specs = []
    for run_dir in sorted(runs.glob("k128_dim128_coupling0_seed2026062[34]")):
        checkpoint = run_dir / "checkpoints/best_model.pt"
        config = run_dir / "config.yaml"
        if checkpoint.exists() and config.exists():
            seed = int(run_dir.name.rsplit("seed", 1)[-1])
            specs.append({"seed": seed, "name": run_dir.name, "checkpoint": checkpoint, "config": config})
    if len(specs) < 2:
        raise RuntimeError(f"Expected two K128/dim128 warm-start runs under {runs}")
    return specs


def base_for_seed_index(seed_index: int) -> dict[str, Any]:
    specs = base_specs()
    return specs[seed_index % len(specs)]


def run_name_for(condition: AlignmentCondition, seed: int) -> str:
    return f"k128_dim128_{condition.key.lower()}_causal_alignment_seed{seed}"


def training_config(
    suite_name: str,
    condition: AlignmentCondition,
    *,
    seed: int,
    seed_index: int,
    device: str,
    smoke: bool,
    formal_epochs: int,
) -> tuple[str, dict[str, Any]]:
    base = base_for_seed_index(seed_index)
    run_name = run_name_for(condition, seed)
    cfg: dict[str, Any] = {
        "_base_": str(base["config"]),
        "experiment": {
            "name": run_name,
            "run_group": (
                f"tokenizer_lightweight_alignment/{suite_name}/"
                f"{'smoke' if smoke else 'tokenizer_interventions'}"
            ),
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
                "joint_entropy_weight": 0.0,
                "smoothness_weight": 0.0,
                "pair_likelihood_weight": 1.0 if condition.global_residual_coupling else 0.0,
                "pair_gradient_target": "fnirs",
                "pair_temperature": 0.5,
                "residualize_fnirs_marginal": bool(condition.global_residual_coupling),
                "lag_evidence_weight": 0.0,
                "effective_smoothness_weight": 0.0,
                "interaction_lag_sparsity_weight": 0.0,
                "local_residual_enabled": False,
                "context_residual_enabled": bool(condition.context_residual_coupling),
                "context_states": 4,
                "context_rank": 16,
                "context_router_type": "learned",
                "context_pair_weight": 1.0,
                "context_entropy_weight": 0.01,
                "context_balance_weight": 0.01,
                "context_residual_l1_weight": 0.001,
                "context_gradient_target": "fnirs",
            },
            "interaction_aux": {
                "weight": condition.interaction_weight,
                "direction": "eeg_to_fnirs",
                "stop_gradient": condition.interaction_stop_gradient,
            },
            "shared_state_bottleneck": {
                "weight": condition.shared_bottleneck_weight,
                "dim": 32,
                "stop_gradient": True,
            },
        },
        "training": {
            "batch_size": 256,
            "learning_rate": 7.5e-5,
            "min_lr": 1.0e-5,
            "epochs": 2 if smoke else int(formal_epochs),
            "alignment_warmup": {
                "enabled": condition.coupling_weight > 0.0,
                "start_epoch": 1,
                "end_epoch": 2 if smoke else 8,
                "start_scale": 0.0,
                "end_scale": 1.0,
            },
            "validation": {
                "interval_epochs": 1,
                "start_epoch": 1,
                "max_batches": 2 if smoke else None,
            },
            "checkpoint": {"save_every": 1 if smoke else 5},
            "early_stopping": {
                "enabled": False,
                "metric": "val_primary_loss",
                "mode": "min",
                "start_epoch": 1,
            },
            "gradient_attribution": {
                "enabled": True,
                "max_batches": 1,
                "interval_epochs": 1 if smoke else 5,
            },
        },
    }
    if smoke:
        cfg["training"]["max_train_batches"] = 2
        cfg["training"]["performance"] = {
            "tf32": True,
            "cuda_prefetch": False,
            "compile": {"enabled": False},
        }
        cfg["data"]["num_workers"] = 0
        cfg["data"]["dataloader"] = {"persistent_workers": False, "drop_last": False}
    return run_name, cfg


def training_command(config: Path, run_name: str, *, smoke: bool) -> str:
    suffix = " --skip-post-analysis" if smoke else ""
    return (
        "bash experiments/scripts/launch_training_nohup.sh --task source-observation-tokenizer "
        f"--foreground --config {q(config)} --run-name {q(run_name)}{suffix}"
    )


def maybe(command: str, sentinel: Path) -> str:
    return f"if [ ! -e {q(sentinel)} ]; then {command}; fi"


def audit_commands(suite_dir: Path, device: str) -> list[str]:
    qdir = suite_dir / "queue_logs"
    commands = [
        f"while [ ! -e {q(qdir / 'formal_gpu0.done')} ] || [ ! -e {q(qdir / 'formal_gpu1.done')} ]; do sleep 120; done"
    ]
    for run_dir in sorted((suite_dir / "tokenizer_interventions").glob("k128_dim128_a*_causal_alignment_seed*")):
        label = run_dir.name
        checkpoint = run_dir / "checkpoints/best_model.pt"
        export_dir = suite_dir / "coupling_audit_exports" / label
        position_dir = suite_dir / "position_audit" / label
        geometry_dir = suite_dir / "codebook_geometry" / label
        info_dir = suite_dir / "information_drop_audit" / label
        local_dir = suite_dir / "local_coupling_audit" / label
        token_export = suite_dir / "token_exports" / label
        downstream_dir = suite_dir / "downstream_probe" / label
        commands.extend([
            maybe(
                ".venv/bin/python experiments/scripts/export_coupling_audit_data.py "
                f"--checkpoint {q(checkpoint)} --output-dir {q(export_dir)} --label {q(label)} "
                f"--device {q(device)} --batch-size 512 --num-workers 8 --clear-entry-filters",
                export_dir / "manifest.json",
            ),
            maybe(
                ".venv/bin/python experiments/scripts/analyze_coupling_identifiability.py "
                f"--export-dir {q(export_dir)} --output-dir {q(position_dir)} "
                f"--permutations 500 --bootstraps 1000 --max-lag-tokens {MAX_LAG_TOKENS}",
                position_dir / "summary.json",
            ),
            maybe(
                ".venv/bin/python experiments/scripts/analyze_codebook_geometry.py "
                f"--checkpoint {q(checkpoint)} --output-dir {q(geometry_dir)} --run-name {q(label)}",
                geometry_dir / "summary.json",
            ),
            maybe(
                ".venv/bin/python experiments/scripts/analyze_information_drop.py "
                f"--checkpoint {q(checkpoint)} --export-dir {q(export_dir)} --output-dir {q(info_dir)} "
                f"--run-name {q(label)} --splits val test --max-tokens 50000",
                info_dir / "summary.json",
            ),
            maybe(
                ".venv/bin/python experiments/scripts/analyze_local_coupling.py "
                f"--export-dir {q(export_dir)} --output-dir {q(local_dir)} --run-name {q(label)} "
                f"--max-lag-tokens {MAX_LAG_TOKENS}",
                local_dir / "summary.json",
            ),
            maybe(
                "bash experiments/scripts/launch_training_nohup.sh --task source-observation-token-export --foreground "
                f"--tokenizer-run-dir {q(run_dir)} --checkpoint {q(checkpoint)} "
                f"--output-root {q(suite_dir / 'token_exports')} --run-name {q(label)} "
                f"--splits train,val,test --batch-size 512 --num-workers 8 --clear-entry-filters",
                token_export / "manifest.json",
            ),
            maybe(
                ".venv/bin/python experiments/scripts/run_source_observation_token_downstream_probe.py "
                f"--token-run-dir {q(token_export)} --output-dir {q(downstream_dir)}",
                downstream_dir / "summary.json",
            ),
        ])
    commands.append(f".venv/bin/python experiments/scripts/finalize_tokenizer_lightweight_alignment_suite.py --suite-dir {q(suite_dir)}")
    return commands


def write_queue(path: Path, commands: list[str], done_file: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {q(PROJECT_ROOT)}",
            "export PYTHONUNBUFFERED=1",
            *commands,
            f"touch {q(done_file)}",
        ]) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def write_audit_queue(suite_dir: Path) -> None:
    qdir = suite_dir / "queue_logs"
    write_queue(qdir / "audit_queue.sh", audit_commands(suite_dir, "cuda:0"), qdir / "audit.done")


def initialize_suite(suite_dir: Path, *, formal_epochs: int, seeds: tuple[int, int]) -> dict[str, Any]:
    suite_dir.mkdir(parents=True, exist_ok=True)
    suite_name = suite_dir.name
    for name in (
        "tokenizer_interventions",
        "smoke",
        "queue_logs",
        "coupling_audit_exports",
        "position_audit",
        "codebook_geometry",
        "information_drop_audit",
        "local_coupling_audit",
        "token_exports",
        "downstream_probe",
    ):
        (suite_dir / name).mkdir(parents=True, exist_ok=True)

    formal_by_gpu = {0: [], 1: []}
    smoke_by_gpu = {0: [], 1: []}
    formal_configs: dict[str, str] = {}
    smoke_configs: dict[str, str] = {}
    for seed_index, seed in enumerate(seeds):
        gpu = seed_index % 2
        device = f"cuda:{gpu}"
        for condition in CONDITIONS:
            run_name, cfg = training_config(
                suite_name,
                condition,
                seed=seed,
                seed_index=seed_index,
                device=device,
                smoke=False,
                formal_epochs=formal_epochs,
            )
            config_path = suite_dir / "tokenizer_interventions/configs" / f"{run_name}.yaml"
            write_yaml(config_path, cfg)
            formal_configs[str(config_path.relative_to(suite_dir))] = run_name
            formal_by_gpu[gpu].append(training_command(config_path, run_name, smoke=False))

            smoke_name, smoke_cfg = training_config(
                suite_name,
                condition,
                seed=seed,
                seed_index=seed_index,
                device=device,
                smoke=True,
                formal_epochs=formal_epochs,
            )
            smoke_path = suite_dir / "smoke/configs" / f"{smoke_name}.yaml"
            write_yaml(smoke_path, smoke_cfg)
            smoke_configs[str(smoke_path.relative_to(suite_dir))] = smoke_name
            smoke_by_gpu[gpu].append(training_command(smoke_path, f"smoke_{smoke_name}", smoke=True))

    qdir = suite_dir / "queue_logs"
    for gpu in (0, 1):
        write_queue(qdir / f"smoke_gpu{gpu}.sh", smoke_by_gpu[gpu], qdir / f"smoke_gpu{gpu}.done")
        write_queue(qdir / f"formal_gpu{gpu}.sh", formal_by_gpu[gpu], qdir / f"formal_gpu{gpu}.done")
    write_audit_queue(suite_dir)

    manifest = {
        "schema_version": "tokenizer_lightweight_alignment_suite_v1",
        "created_at": datetime.now().isoformat(),
        "suite_root": str(suite_dir),
        "git_commit": git_commit(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True).strip()),
        "warm_start_suite": str(NEXT_STAGE_SUITE),
        "warm_starts": [
            {key: str(value) for key, value in spec.items()}
            for spec in base_specs()
        ],
        "conditions": [asdict(condition) for condition in CONDITIONS],
        "formal_epochs": int(formal_epochs),
        "seeds": list(seeds),
        "max_lag_tokens": MAX_LAG_TOKENS,
        "decision_key": {
            "information_retention": "hard-token cross-modal CMI/LOSO gain and continuous-to-hard retention vs A0",
            "fine_task_retention": "nback_load_0_vs_2_vs_3 or motor_lmi_vs_rmi balanced accuracy improves by >=0.03 vs A0",
            "coupling_visualization": "Gate3 MI above shuffle improves by >=20% vs A0 with non-diffuse residual pattern",
            "gradient_compatibility": "alignment/coupling vs reconstruction gradient cosine >= -0.2",
            "token_health": "primary/reconstruction degradation <=5%, source perplexity >=80% of A0",
        },
        "queues": {
            "smoke_gpu0": str(qdir / "smoke_gpu0.sh"),
            "smoke_gpu1": str(qdir / "smoke_gpu1.sh"),
            "formal_gpu0": str(qdir / "formal_gpu0.sh"),
            "formal_gpu1": str(qdir / "formal_gpu1.sh"),
            "audit": str(qdir / "audit_queue.sh"),
        },
        "formal_configs": formal_configs,
        "smoke_configs": smoke_configs,
    }
    (suite_dir / "suite_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (suite_dir / "summary.json").write_text(json.dumps({"status": "initialized"}, indent=2) + "\n", encoding="utf-8")
    (suite_dir / "decision.json").write_text(json.dumps({"status": "pending_smoke"}, indent=2) + "\n", encoding="utf-8")
    (suite_dir / "report.md").write_text(
        "# Tokenizer Lightweight Alignment Suite\n\n"
        "Status: initialized. Run smoke queues first, then formal queues, then audit queue.\n",
        encoding="utf-8",
    )
    return manifest


def run_queue(path: Path, log_path: Path, detached: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        ["bash", str(path)],
        cwd=PROJECT_ROOT,
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
    parser.add_argument("--mode", choices=("initialize", "smoke", "launch", "audit", "all"), default="initialize")
    parser.add_argument("--formal-epochs", type=int, default=40)
    parser.add_argument("--seeds", nargs=2, type=int, default=(20260641, 20260642))
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir).resolve() if args.suite_dir else (
        SUITE_PARENT / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_causal_pre_vq_alignment_v1"
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
            pids[f"formal_gpu{gpu}"] = run_queue(
                suite_dir / f"queue_logs/formal_gpu{gpu}.sh",
                suite_dir / f"queue_logs/formal_gpu{gpu}.log",
                detached=True,
            )
    if args.mode in {"audit", "all"}:
        write_audit_queue(suite_dir)
        pids["audit"] = run_queue(
            suite_dir / "queue_logs/audit_queue.sh",
            suite_dir / "queue_logs/audit_queue.log",
            detached=args.mode != "all",
        )
    if pids:
        (suite_dir / "queue_logs/pids.json").write_text(json.dumps(pids, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"suite_dir": str(suite_dir), "pids": pids}, indent=2))


if __name__ == "__main__":
    main()
