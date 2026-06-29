#!/usr/bin/env python
"""Launch lag-aware full cross-modal tokenizer experiments."""

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
SUITE_PARENT = PROJECT_ROOT / "experiments/runs/tokenizer_full_cross_modal_alignment"
WARM_START_ROOT = (
    PROJECT_ROOT
    / "experiments/runs/tokenizer_next_stage/20260620_221822_k128_dim_cognitive_transfer_v1/vector_dim_sweep"
)
MAX_LAG_TOKENS = 5
CONFIRM_KEYS = {"Z0", "Z3", "Z4", "Z5", "Z6"}


@dataclass(frozen=True)
class AlignmentCondition:
    key: str
    description: str
    fusion_mode: str | None = None
    codebook_mode: str = "independent"
    temporal_nce: bool = False
    full_alignment: bool = False


CONDITIONS = (
    AlignmentCondition("Z0", "new-schedule independent-codebook baseline"),
    AlignmentCondition("Z1", "lag-aware temporal contrastive alignment without exchange", temporal_nce=True),
    AlignmentCondition("Z2", "causal full cross-attention without explicit alignment", fusion_mode="causal_cross_attention"),
    AlignmentCondition(
        "Z3", "causal full cross-attention with full self-supervised alignment",
        fusion_mode="causal_cross_attention", temporal_nce=True, full_alignment=True,
    ),
    AlignmentCondition(
        "Z4", "bidirectional full cross-attention with full self-supervised alignment",
        fusion_mode="bidirectional_cross_attention", temporal_nce=True, full_alignment=True,
    ),
    AlignmentCondition(
        "Z5", "causal full alignment with shared joint source vocabulary",
        fusion_mode="causal_cross_attention", codebook_mode="shared_joint", temporal_nce=True, full_alignment=True,
    ),
    AlignmentCondition(
        "Z6", "bidirectional full alignment with shared joint source vocabulary",
        fusion_mode="bidirectional_cross_attention", codebook_mode="shared_joint", temporal_nce=True, full_alignment=True,
    ),
)


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def warm_starts() -> list[dict[str, Any]]:
    specs = []
    for run_dir in sorted(WARM_START_ROOT.glob("k128_dim128_coupling0_seed2026062[34]")):
        checkpoint = run_dir / "checkpoints/best_model.pt"
        config = run_dir / "config.yaml"
        if checkpoint.exists() and config.exists():
            specs.append({"run": run_dir.name, "checkpoint": checkpoint, "config": config})
    if len(specs) != 2:
        raise RuntimeError(f"Expected two K128/dim128 warm starts under {WARM_START_ROOT}, found {len(specs)}")
    return specs


def run_name(condition: AlignmentCondition, seed: int) -> str:
    return f"k128_dim128_{condition.key.lower()}_full_alignment_seed{seed}"


def build_config(
    suite_name: str,
    condition: AlignmentCondition,
    *,
    seed: int,
    seed_index: int,
    device: str,
    stage: str,
    epochs: int,
) -> tuple[str, dict[str, Any]]:
    base = warm_starts()[seed_index % 2]
    name = run_name(condition, seed)
    smoke = stage == "smoke"
    alignment_enabled = condition.temporal_nce or condition.full_alignment
    cfg: dict[str, Any] = {
        "_base_": str(base["config"]),
        "experiment": {
            "name": name,
            "run_group": f"tokenizer_full_cross_modal_alignment/{suite_name}/{stage}",
            "seed": seed,
            "device": device,
            "description": condition.description,
        },
        "data": {"seed": seed},
        "model": {
            "cross_modal_exchange": {"enabled": False},
            "cross_modal_fusion": {
                "enabled": condition.fusion_mode is not None,
                "mode": condition.fusion_mode or "causal_cross_attention",
                "embed_dim": 128,
                "depth": 2,
                "num_heads": 4,
                "max_lag_tokens": MAX_LAG_TOKENS,
                "relative_lag_bias": True,
                "dropout": 0.1,
            },
            "source_codebook": {"mode": condition.codebook_mode},
        },
        "warm_start": {
            "checkpoint": str(base["checkpoint"]),
            "reset_coupling": True,
            "coupling_reset_std": 0.02,
        },
        "loss": {
            "cross_modal_alignment": {
                "temporal_nce_weight": 0.20 if condition.temporal_nce else 0.0,
                "masked_latent_weight": 0.20 if condition.full_alignment else 0.0,
                "soft_code_distillation_weight": 0.05 if condition.full_alignment else 0.0,
                "temperature": 0.10,
                "positive_lag_weights": [0.0, 0.1, 0.4, 0.4, 0.1, 0.0],
                "token_mask_ratio": 0.50,
                "modality_dropout_probability": 0.25,
            },
            "coupling": {
                "weight": 0.0,
                "max_lag_tokens": MAX_LAG_TOKENS,
                "lag_focus_weight": 0.0,
                "joint_entropy_weight": 0.0,
                "smoothness_weight": 0.0,
                "pair_likelihood_weight": 0.0,
                "pair_gradient_target": "fnirs",
                "residualize_fnirs_marginal": True,
                "context_residual_enabled": False,
            },
        },
        "training": {
            "batch_size": 256,
            "learning_rate": 2.0e-5,
            "min_lr": 1.0e-6,
            "epochs": 2 if smoke else int(epochs),
            "alignment_warmup": {
                "enabled": alignment_enabled,
                "start_epoch": 1,
                "ramp_epochs": 2 if smoke else 5,
                "start_scale": 0.0,
            },
            "quantization_warmup": {
                "enabled": True,
                "start_epoch": 1,
                "ramp_epochs": 2 if smoke else 10,
                "start_scale": 0.6779661016949152,
                "end_scale": 1.0,
            },
            "source_target_warmup": {
                "enabled": True, "start_epoch": 1, "ramp_epochs": 1,
                "start_scale": 0.3448275862068966, "end_scale": 0.3448275862068966,
            },
            "observation_target_warmup": {
                "enabled": True, "start_epoch": 1, "ramp_epochs": 1,
                "start_scale": 0.3448275862068966, "end_scale": 0.3448275862068966,
            },
            "optimizer_groups": {
                "enabled": True,
                "new_lr": 1.0e-4,
                "legacy_lr": 2.0e-5,
                "observation_lr": 1.0e-5,
            },
            "staged_unfreeze": {
                "enabled": True,
                "freeze_encoder_epochs": 1 if smoke else 5,
                "freeze_codebook_epochs": 1 if smoke else 5,
            },
            "alignment_gradient_control": {
                "enabled": alignment_enabled,
                "target_ratio": 0.30,
                "min_scale": 0.1,
                "max_scale": 20.0,
                "ema": 0.9,
                "update_interval_steps": 10,
            },
            "validation": {
                "interval_epochs": 1,
                "start_epoch": 1,
                "max_batches": 2 if smoke else None,
                "forced_hard": True,
            },
            "checkpoint": {"save_every": 1 if smoke else 5},
            "early_stopping": {"enabled": False, "metric": "val_primary_loss", "mode": "min", "start_epoch": 1},
            "gradient_attribution": {"enabled": True, "max_batches": 1, "interval_epochs": 1 if smoke else 5},
            "performance": {
                "tf32": True,
                "cuda_prefetch": not smoke,
                "compile": {"enabled": False},
            },
        },
    }
    if smoke:
        cfg["training"]["max_train_batches"] = 2
        cfg["training"]["performance"] = {
            "tf32": True, "cuda_prefetch": False, "compile": {"enabled": False},
        }
        cfg["data"].update({"num_workers": 0, "dataloader": {"persistent_workers": False, "drop_last": False}})
    return name, cfg


def training_command(config: Path, name: str, smoke: bool) -> str:
    suffix = " --skip-post-analysis" if smoke else ""
    return (
        "bash experiments/scripts/launch_training_nohup.sh --task source-observation-tokenizer "
        f"--foreground --config {q(config)} --run-name {q(('smoke_' if smoke else '') + name)}{suffix}"
    )


def maybe(command: str, sentinel: Path) -> str:
    return f"if [ ! -e {q(sentinel)} ]; then {command}; fi"


def audit_commands(suite_dir: Path, run_names: list[str], device: str = "cuda:0") -> list[str]:
    qdir = suite_dir / "queue_logs"
    commands = [
        f"while [ ! -e {q(qdir / 'formal_gpu0.done')} ] || [ ! -e {q(qdir / 'formal_gpu1.done')} ]; do sleep 120; done"
    ]
    for name in sorted(run_names):
        run_dir = suite_dir / "tokenizer_interventions" / name
        for checkpoint_name, suffix in (
            ("best_hard_primary.pt", ""),
            ("best_alignment.pt", "__best_alignment"),
            ("final_model.pt", "__final"),
        ):
            label = f"{name}{suffix}"
            checkpoint = run_dir / "checkpoints" / checkpoint_name
            export_dir = suite_dir / "coupling_audit_exports" / label
            position_dir = suite_dir / "position_audit" / label
            geometry_dir = suite_dir / "codebook_geometry" / label
            info_dir = suite_dir / "information_drop_audit" / label
            local_dir = suite_dir / "local_coupling_audit" / label
            token_dir = suite_dir / "token_exports" / label
            downstream_dir = suite_dir / "downstream_probe" / label
            commands.extend([
                f"if [ -e {q(checkpoint)} ]; then",
                maybe(
                    ".venv/bin/python experiments/scripts/export_coupling_audit_data.py "
                    f"--checkpoint {q(checkpoint)} --output-dir {q(export_dir)} --label {q(label)} "
                    f"--device {q(device)} --batch-size 512 --num-workers 8 --clear-entry-filters",
                    export_dir / "manifest.json",
                ),
                maybe(
                    ".venv/bin/python experiments/scripts/analyze_coupling_identifiability.py "
                    f"--export-dir {q(export_dir)} --output-dir {q(position_dir)} "
                    f"--permutations 200 --bootstraps 500 --max-lag-tokens {MAX_LAG_TOKENS}",
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
                    "--splits train,val,test --batch-size 512 --num-workers 8 --clear-entry-filters",
                    token_dir / "manifest.json",
                ),
                maybe(
                    ".venv/bin/python experiments/scripts/run_source_observation_token_downstream_probe.py "
                    f"--token-run-dir {q(token_dir)} --output-dir {q(downstream_dir)}",
                    downstream_dir / "summary.json",
                ),
                "fi",
            ])
    commands.append(
        f".venv/bin/python experiments/scripts/finalize_tokenizer_full_cross_modal_alignment_suite.py --suite-dir {q(suite_dir)}"
    )
    return commands


def write_queue(path: Path, commands: list[str], done: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([
        "#!/usr/bin/env bash", "set -euo pipefail", f"cd {q(PROJECT_ROOT)}", "export PYTHONUNBUFFERED=1",
        *commands, f"touch {q(done)}",
    ]) + "\n", encoding="utf-8")
    path.chmod(0o755)


def initialize_suite(suite_dir: Path, *, screening_epochs: int, confirm_epochs: int, seeds: tuple[int, int]) -> dict[str, Any]:
    suite_dir.mkdir(parents=True, exist_ok=True)
    for leaf in (
        "smoke", "screening", "tokenizer_interventions", "queue_logs", "coupling_audit_exports",
        "position_audit", "codebook_geometry", "information_drop_audit", "local_coupling_audit",
        "token_exports", "downstream_probe",
    ):
        (suite_dir / leaf).mkdir(parents=True, exist_ok=True)
    queues: dict[str, list[str]] = {f"smoke_gpu{i}": [] for i in range(2)}
    queues.update({f"screen_gpu{i}": [] for i in range(2)})
    queues.update({f"formal_gpu{i}": [] for i in range(2)})
    configs: dict[str, dict[str, str]] = {"smoke": {}, "screening": {}, "confirmatory": {}}

    for index, condition in enumerate(CONDITIONS):
        gpu = index % 2
        name, cfg = build_config(
            suite_dir.name, condition, seed=seeds[0], seed_index=0, device=f"cuda:{gpu}",
            stage="smoke", epochs=2,
        )
        path = suite_dir / "smoke/configs" / f"{name}.yaml"
        write_yaml(path, cfg)
        configs["smoke"][str(path.relative_to(suite_dir))] = name
        queues[f"smoke_gpu{gpu}"].append(training_command(path, name, True))

        name, cfg = build_config(
            suite_dir.name, condition, seed=seeds[0], seed_index=0, device=f"cuda:{gpu}",
            stage="screening", epochs=screening_epochs,
        )
        path = suite_dir / "screening/configs" / f"{name}.yaml"
        write_yaml(path, cfg)
        configs["screening"][str(path.relative_to(suite_dir))] = name
        queues[f"screen_gpu{gpu}"].append(training_command(path, name, False))

    for seed_index, seed in enumerate(seeds):
        gpu = seed_index % 2
        for condition in CONDITIONS:
            if condition.key not in CONFIRM_KEYS:
                continue
            name, cfg = build_config(
                suite_dir.name, condition, seed=seed, seed_index=seed_index, device=f"cuda:{gpu}",
                stage="tokenizer_interventions", epochs=confirm_epochs,
            )
            path = suite_dir / "tokenizer_interventions/configs" / f"{name}.yaml"
            write_yaml(path, cfg)
            configs["confirmatory"][str(path.relative_to(suite_dir))] = name
            queues[f"formal_gpu{gpu}"].append(training_command(path, name, False))

    qdir = suite_dir / "queue_logs"
    for key, commands in queues.items():
        write_queue(qdir / f"{key}.sh", commands, qdir / f"{key}.done")
    write_queue(
        qdir / "audit_queue.sh",
        audit_commands(suite_dir, list(configs["confirmatory"].values())),
        qdir / "audit.done",
    )
    manifest = {
        "schema_version": "tokenizer_full_cross_modal_alignment_suite_v1",
        "created_at": datetime.now().isoformat(),
        "suite_root": str(suite_dir),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True).strip()),
        "conditions": [asdict(condition) for condition in CONDITIONS],
        "seeds": list(seeds),
        "screening_epochs": screening_epochs,
        "confirm_epochs": confirm_epochs,
        "warm_starts": [{key: str(value) for key, value in item.items()} for item in warm_starts()],
        "configs": configs,
        "queues": {key: str(qdir / f"{key}.sh") for key in queues} | {"audit": str(qdir / "audit_queue.sh")},
    }
    (suite_dir / "suite_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (suite_dir / "decision.json").write_text(json.dumps({"status": "pending_smoke"}, indent=2) + "\n", encoding="utf-8")
    (suite_dir / "report.md").write_text("# Full Cross-Modal Alignment Suite\n\nStatus: initialized.\n", encoding="utf-8")
    return manifest


def prepare_z7(suite_dir: Path, winner: str, epochs: int | None = None) -> dict[str, str]:
    winner = winner.upper()
    source = next((item for item in CONDITIONS if item.key == winner), None)
    if source is None or not source.full_alignment:
        raise ValueError("Z7 winner must be one of Z3/Z4/Z5/Z6")
    manifest_path = suite_dir / "suite_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seeds = tuple(int(value) for value in manifest["seeds"])
    epochs = int(epochs or manifest["confirm_epochs"])
    z7 = AlignmentCondition(
        "Z7",
        f"{winner} plus weak residual coupling prior",
        fusion_mode=source.fusion_mode,
        codebook_mode=source.codebook_mode,
        temporal_nce=True,
        full_alignment=True,
    )
    queues = {0: [], 1: []}
    generated: dict[str, str] = {}
    for seed_index, seed_value in enumerate(seeds):
        gpu = seed_index % 2
        name, cfg = build_config(
            suite_dir.name, z7, seed=seed_value, seed_index=seed_index,
            device=f"cuda:{gpu}", stage="tokenizer_interventions", epochs=epochs,
        )
        cfg["loss"]["coupling"].update({
            "weight": 0.01,
            "pair_likelihood_weight": 1.0,
            "pair_temperature": 0.5,
            "pair_gradient_target": "fnirs",
            "residualize_fnirs_marginal": True,
        })
        path = suite_dir / "tokenizer_interventions/configs" / f"{name}.yaml"
        write_yaml(path, cfg)
        generated[str(path.relative_to(suite_dir))] = name
        queues[gpu].append(training_command(path, name, False))
    qdir = suite_dir / "queue_logs"
    for gpu in range(2):
        write_queue(qdir / f"z7_gpu{gpu}.sh", queues[gpu], qdir / f"z7_gpu{gpu}.done")
    manifest.setdefault("configs", {})["z7"] = generated
    manifest["z7_parent"] = winner
    manifest["queues"].update({f"z7_gpu{gpu}": str(qdir / f"z7_gpu{gpu}.sh") for gpu in range(2)})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    all_confirm = list(manifest["configs"]["confirmatory"].values()) + list(generated.values())
    write_queue(qdir / "audit_queue.sh", audit_commands(suite_dir, all_confirm), qdir / "audit.done")
    return generated


def run_queue(path: Path, log_path: Path, detached: bool) -> int:
    log = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        ["bash", str(path)], cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=detached,
    )
    log.close()
    return process.pid if detached else process.wait()


def run_parallel_foreground_queues(suite_dir: Path, prefix: str) -> None:
    processes = []
    logs = []
    for gpu in range(2):
        key = f"{prefix}_gpu{gpu}"
        log = (suite_dir / f"queue_logs/{key}.log").open("a", encoding="utf-8")
        process = subprocess.Popen(
            ["bash", str(suite_dir / f"queue_logs/{key}.sh")],
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        processes.append((key, process))
        logs.append(log)
    try:
        for key, process in processes:
            code = process.wait()
            if code:
                raise RuntimeError(f"{key} failed with exit code {code}")
    finally:
        for log in logs:
            log.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir")
    parser.add_argument(
        "--mode", choices=("initialize", "smoke", "screen", "launch", "prepare-z7", "launch-z7", "audit"),
        default="initialize",
    )
    parser.add_argument("--screening-epochs", type=int, default=20)
    parser.add_argument("--confirm-epochs", type=int, default=40)
    parser.add_argument("--seeds", nargs=2, type=int, default=(20260661, 20260662))
    parser.add_argument("--winner", choices=("Z3", "Z4", "Z5", "Z6"))
    args = parser.parse_args()
    suite_dir = Path(args.suite_dir).resolve() if args.suite_dir else (
        SUITE_PARENT / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_lag_aware_cross_transformer_v1"
    )
    if args.mode == "initialize":
        initialize_suite(
            suite_dir, screening_epochs=args.screening_epochs,
            confirm_epochs=args.confirm_epochs, seeds=tuple(args.seeds),
        )
    if args.mode == "prepare-z7":
        if not args.winner:
            parser.error("--winner is required for --mode prepare-z7")
        generated = prepare_z7(suite_dir, args.winner, args.confirm_epochs)
        print(json.dumps({"suite_dir": str(suite_dir), "z7_configs": generated}, indent=2))
        return
    queue_prefix = {"smoke": "smoke", "screen": "screen", "launch": "formal"}.get(args.mode)
    pids: dict[str, int] = {}
    if args.mode == "smoke":
        run_parallel_foreground_queues(suite_dir, "smoke")
        print(json.dumps({"suite_dir": str(suite_dir), "pids": {}}, indent=2))
        return
    if queue_prefix:
        for gpu in range(2):
            key = f"{queue_prefix}_gpu{gpu}"
            pids[key] = run_queue(
                suite_dir / f"queue_logs/{key}.sh",
                suite_dir / f"queue_logs/{key}.log",
                detached=args.mode != "smoke",
            )
    if args.mode == "audit":
        pids["audit"] = run_queue(
            suite_dir / "queue_logs/audit_queue.sh",
            suite_dir / "queue_logs/audit_queue.log",
            detached=True,
        )
    if args.mode == "launch-z7":
        for gpu in range(2):
            key = f"z7_gpu{gpu}"
            pids[key] = run_queue(
                suite_dir / f"queue_logs/{key}.sh",
                suite_dir / f"queue_logs/{key}.log",
                detached=True,
            )
    if pids:
        (suite_dir / "queue_logs/pids.json").write_text(json.dumps(pids, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"suite_dir": str(suite_dir), "pids": pids}, indent=2))


if __name__ == "__main__":
    main()
