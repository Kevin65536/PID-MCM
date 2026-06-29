#!/usr/bin/env python
"""Launch causal continuous-ceiling and shared-private cross-token experiments."""

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
SUITE_PARENT = PROJECT_ROOT / "experiments/runs/tokenizer_causal_shared_private"
WARM_START_ROOT = (
    PROJECT_ROOT
    / "experiments/runs/tokenizer_next_stage/20260620_221822_k128_dim_cognitive_transfer_v1/vector_dim_sweep"
)
MAX_LAG_TOKENS = 5


@dataclass(frozen=True)
class Condition:
    key: str
    description: str
    apply_to_source: bool
    temporal_nce: bool = False
    cross_token: bool = False
    adaptive_lag: bool = False
    coupling: bool = False
    continuous_ceiling: bool = False


CONDITIONS = (
    Condition("M0", "frozen continuous EEG-to-fNIRS prediction ceiling", False, continuous_ceiling=True),
    Condition("M1", "causal source fusion with fully masked latent prediction only", True),
    Condition("M2", "M1 plus lag-aware temporal NCE", True, temporal_nce=True),
    Condition("M3", "shared-private source tokenizer with dedicated cross token", False, cross_token=True),
    Condition("M4", "M3 plus EEG-conditioned adaptive lag mixture", False, cross_token=True, adaptive_lag=True),
    Condition("M5", "M4 plus weak residual source-token coupling prior", False, cross_token=True, adaptive_lag=True, coupling=True),
)


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def warm_starts() -> list[dict[str, Path | str]]:
    result = []
    for run_dir in sorted(WARM_START_ROOT.glob("k128_dim128_coupling0_seed2026062[34]")):
        checkpoint = run_dir / "checkpoints/best_model.pt"
        config = run_dir / "config.yaml"
        if checkpoint.exists() and config.exists():
            result.append({"run": run_dir.name, "checkpoint": checkpoint, "config": config})
    if len(result) != 2:
        raise RuntimeError(f"Expected two warm starts under {WARM_START_ROOT}, found {len(result)}")
    return result


def run_name(condition: Condition, seed: int) -> str:
    return f"k128_dim128_{condition.key.lower()}_causal_shared_private_seed{seed}"


def build_config(
    suite_name: str,
    condition: Condition,
    *,
    seed: int,
    seed_index: int,
    device: str,
    stage: str,
    epochs: int,
) -> tuple[str, dict[str, Any]]:
    base = warm_starts()[seed_index % len(warm_starts())]
    smoke = stage == "smoke"
    total_epochs = 2 if smoke else int(epochs)
    name = run_name(condition, seed)
    config: dict[str, Any] = {
        "_base_": str(base["config"]),
        "experiment": {
            "name": name,
            "run_group": f"tokenizer_causal_shared_private/{suite_name}/{stage}",
            "seed": seed,
            "device": device,
            "description": condition.description,
        },
        "data": {"seed": seed},
        "model": {
            "cross_modal_exchange": {"enabled": False},
            "cross_modal_fusion": {
                "enabled": True,
                "mode": "causal_cross_attention",
                "embed_dim": 128,
                "depth": 2,
                "num_heads": 4,
                "max_lag_tokens": MAX_LAG_TOKENS,
                "relative_lag_bias": True,
                "dropout": 0.1,
                "apply_to_source": condition.apply_to_source,
                "adaptive_lag_enabled": condition.adaptive_lag,
                "adaptive_lag_temperature": 1.0,
                "adaptive_lag_prior": [0.01, 0.10, 0.39, 0.39, 0.10, 0.01],
            },
            "cross_modal_token": {
                "enabled": condition.cross_token,
                "codebook_size": 64,
                "dim": 48,
                "residual_scale": 0.25,
            },
            "source_codebook": {"mode": "independent"},
        },
        "warm_start": {
            "checkpoint": str(base["checkpoint"]),
            "reset_coupling": True,
            "coupling_reset_std": 0.02,
        },
        "loss": {
            "cross_modal_alignment": {
                "temporal_nce_weight": 0.20 if condition.temporal_nce else 0.0,
                "masked_latent_weight": 0.20 if not condition.cross_token else 0.0,
                "soft_code_distillation_weight": 0.0,
                "paired_margin_weight": 0.20 if not condition.cross_token else 0.0,
                "paired_margin": 0.05,
                "temperature": 0.10,
                "positive_lag_weights": [0.0, 0.1, 0.4, 0.4, 0.1, 0.0],
                "token_mask_ratio": 1.0,
                "modality_dropout_probability": 1.0,
            },
            "cross_modal_token": {
                "latent_weight": 0.20,
                "distillation_weight": 0.05,
                "margin_weight": 0.20,
                "margin": 0.05,
                "balance_weight": 0.01,
            },
            "coupling": {
                "weight": 0.01 if condition.coupling else 0.0,
                "max_lag_tokens": MAX_LAG_TOKENS,
                "lag_focus_weight": 0.0,
                "joint_entropy_weight": 0.0,
                "smoothness_weight": 0.0,
                "pair_likelihood_weight": 1.0 if condition.coupling else 0.0,
                "pair_temperature": 0.5,
                "pair_gradient_target": "fnirs",
                "residualize_fnirs_marginal": True,
                "context_residual_enabled": False,
            },
        },
        "training": {
            "batch_size": 256,
            "learning_rate": 2.0e-5,
            "min_lr": 1.0e-6,
            "epochs": total_epochs,
            "alignment_warmup": {
                "enabled": True,
                "start_epoch": 1,
                "ramp_epochs": 2 if smoke else 5,
                "start_scale": 0.0,
            },
            "quantization_warmup": {
                "enabled": not condition.continuous_ceiling,
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
                "freeze_encoder_epochs": total_epochs if condition.continuous_ceiling else (1 if smoke else 5),
                "freeze_codebook_epochs": total_epochs if condition.continuous_ceiling else (1 if smoke else 5),
            },
            "alignment_gradient_control": {
                "enabled": not condition.continuous_ceiling,
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
            "checkpoint": {
                "save_every": 1 if smoke else 5,
                "alignment_selection": {"min_epoch": 1 if condition.continuous_ceiling else None},
            },
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
            "performance": {"tf32": True, "cuda_prefetch": not smoke, "compile": {"enabled": False}},
        },
    }
    if condition.continuous_ceiling:
        config["training"]["trainable_parameter_prefixes"] = [
            "cross_modal_fusion.", "fnirs_source_mask_token",
        ]
    if smoke:
        config["training"]["max_train_batches"] = 2
        config["training"]["performance"]["cuda_prefetch"] = False
        config["data"].update({
            "num_workers": 0,
            "dataloader": {"persistent_workers": False, "drop_last": False},
        })
    return name, config


def training_command(config_path: Path, name: str, smoke: bool) -> str:
    suffix = " --skip-post-analysis" if smoke else ""
    run_name_value = f"smoke_{name}" if smoke else name
    return (
        "bash experiments/scripts/launch_training_nohup.sh --task source-observation-tokenizer "
        f"--foreground --config {q(config_path)} --run-name {q(run_name_value)}{suffix}"
    )


def write_queue(path: Path, commands: list[str], done_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([
        "#!/usr/bin/env bash", "set -euo pipefail", f"cd {q(PROJECT_ROOT)}", "export PYTHONUNBUFFERED=1",
        *commands, f"touch {q(done_path)}",
    ]) + "\n", encoding="utf-8")
    path.chmod(0o755)


def initialize_suite(
    suite_dir: Path,
    *,
    screening_epochs: int = 20,
    confirm_epochs: int = 40,
    seeds: tuple[int, int] = (20260671, 20260672),
) -> dict[str, Any]:
    for leaf in ("smoke/configs", "screening/configs", "tokenizer_interventions/configs", "queue_logs"):
        (suite_dir / leaf).mkdir(parents=True, exist_ok=True)
    queues = {f"{stage}_gpu{gpu}": [] for stage in ("smoke", "screen", "formal") for gpu in range(2)}
    configs: dict[str, dict[str, str]] = {"smoke": {}, "screening": {}, "confirmatory": {}}

    for index, condition in enumerate(CONDITIONS):
        gpu = index % 2
        for stage, epochs, leaf, queue_stage in (
            ("smoke", 2, "smoke", "smoke"),
            ("screening", screening_epochs, "screening", "screen"),
        ):
            name, config = build_config(
                suite_dir.name, condition, seed=seeds[0], seed_index=0,
                device=f"cuda:{gpu}", stage=stage, epochs=epochs,
            )
            config_path = suite_dir / leaf / "configs" / f"{name}.yaml"
            write_yaml(config_path, config)
            configs[stage][str(config_path.relative_to(suite_dir))] = name
            queues[f"{queue_stage}_gpu{gpu}"].append(training_command(config_path, name, stage == "smoke"))

    for seed_index, seed in enumerate(seeds):
        gpu = seed_index % 2
        for condition in CONDITIONS:
            name, config = build_config(
                suite_dir.name, condition, seed=seed, seed_index=seed_index,
                device=f"cuda:{gpu}", stage="tokenizer_interventions", epochs=confirm_epochs,
            )
            config_path = suite_dir / "tokenizer_interventions/configs" / f"{name}.yaml"
            write_yaml(config_path, config)
            configs["confirmatory"][str(config_path.relative_to(suite_dir))] = name
            queues[f"formal_gpu{gpu}"].append(training_command(config_path, name, False))

    for key, commands in queues.items():
        write_queue(suite_dir / "queue_logs" / f"{key}.sh", commands, suite_dir / "queue_logs" / f"{key}.done")

    manifest = {
        "schema_version": "tokenizer_causal_shared_private_suite_v1",
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
        "queues": {key: str(suite_dir / "queue_logs" / f"{key}.sh") for key in queues},
    }
    (suite_dir / "suite_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (suite_dir / "decision.json").write_text(json.dumps({"status": "pending_smoke"}, indent=2) + "\n", encoding="utf-8")
    return manifest


def run_queues(suite_dir: Path, prefix: str, detached: bool) -> dict[str, int]:
    processes = {}
    handles = []
    for gpu in range(2):
        key = f"{prefix}_gpu{gpu}"
        handle = (suite_dir / "queue_logs" / f"{key}.log").open("a", encoding="utf-8")
        process = subprocess.Popen(
            ["bash", str(suite_dir / "queue_logs" / f"{key}.sh")],
            cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=detached,
        )
        handles.append(handle)
        processes[key] = process
    if not detached:
        try:
            for key, process in processes.items():
                code = process.wait()
                if code:
                    raise RuntimeError(f"{key} failed with exit code {code}")
        finally:
            for handle in handles:
                handle.close()
        return {}
    for handle in handles:
        handle.close()
    return {key: process.pid for key, process in processes.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir")
    parser.add_argument("--mode", choices=("initialize", "smoke", "screen", "launch"), default="initialize")
    parser.add_argument("--screening-epochs", type=int, default=20)
    parser.add_argument("--confirm-epochs", type=int, default=40)
    parser.add_argument("--seeds", nargs=2, type=int, default=(20260671, 20260672))
    args = parser.parse_args()
    suite_dir = Path(args.suite_dir).resolve() if args.suite_dir else (
        SUITE_PARENT / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_causal_shared_private_v1"
    )
    if args.mode == "initialize":
        initialize_suite(
            suite_dir, screening_epochs=args.screening_epochs,
            confirm_epochs=args.confirm_epochs, seeds=tuple(args.seeds),
        )
        print(json.dumps({"suite_dir": str(suite_dir)}, indent=2))
        return
    prefix = {"smoke": "smoke", "screen": "screen", "launch": "formal"}[args.mode]
    pids = run_queues(suite_dir, prefix, detached=args.mode != "smoke")
    if pids:
        (suite_dir / "queue_logs/pids.json").write_text(json.dumps(pids, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"suite_dir": str(suite_dir), "pids": pids}, indent=2))


if __name__ == "__main__":
    main()
