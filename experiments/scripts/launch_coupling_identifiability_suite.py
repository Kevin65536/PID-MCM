#!/usr/bin/env python
"""Create and launch the staged coupling identifiability experiment suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

project_root = Path(__file__).resolve().parents[2]

NEUTRAL_CHECKPOINT = project_root / (
    "experiments/runs/source_observation/croce_local/highwl_v2/"
    "20260611_185811_s2_croce_local_highwl_v2_structural_bs256_280ep_min5e5/"
    "checkpoints/best_model.pt"
)
SHAPED_CHECKPOINT = project_root / (
    "experiments/runs/source_observation/croce_local/highwl_v2/"
    "20260612_232836_s2_croce_local_highwl_v2_coupling_residual_soft_w03_bs256_420ep/"
    "checkpoints/best_model.pt"
)
BASE_CONFIG = project_root / (
    "experiments/configs/source_observation/croce_local/"
    "highwl_v2_coupling_residual_soft_w03_bs256_420ep.yaml"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_manifest_hash() -> str:
    digest = hashlib.sha256()
    root = project_root / "croce_validation/cache/croce_local/highwl_v2"
    for path in sorted(root.rglob("cache_manifest.json")):
        digest.update(str(path.relative_to(project_root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()


def resource_snapshot() -> Dict[str, Any]:
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()
    return {
        "gpus": gpu,
        "cpu_count": os.cpu_count(),
    }


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def intervention_config(
    suite_name: str,
    condition: str,
    seed: int,
    device: str,
    *,
    smoke: bool,
) -> Dict[str, Any]:
    routing = {
        "T0": "none",
        "T1": "none",
        "T2": "both",
        "T3": "eeg",
        "T4": "fnirs",
    }[condition]
    enabled = condition != "T0"
    run_group = f"coupling_design_audit/{suite_name}/{'smoke' if smoke else 'tokenizer_interventions'}"
    config: Dict[str, Any] = {
        "_base_": str(BASE_CONFIG),
        "experiment": {
            "name": f"coupling_identifiability_{condition.lower()}_seed{seed}",
            "run_group": run_group,
            "seed": seed,
            "device": device,
            "description": "Causal coupling-gradient intervention from the common neutral checkpoint.",
        },
        "data": {"seed": seed},
        "warm_start": {
            "checkpoint": str(NEUTRAL_CHECKPOINT),
            "reset_coupling": True,
            "coupling_reset_std": 0.02,
        },
        "training": {
            "batch_size": 256,
            "learning_rate": 1e-4,
            "min_lr": 2.5e-5,
            "epochs": 2 if smoke else 120,
            "alignment_warmup": {
                "enabled": enabled,
                "start_epoch": 1,
                "ramp_epochs": 2 if smoke else 20,
                "start_scale": 0.0,
                "end_scale": 1.0,
            },
            "early_stopping": {"enabled": False},
            "validation": {
                "interval_epochs": 1 if smoke else 2,
                "start_epoch": 1,
                "max_batches": 1 if smoke else None,
            },
            "checkpoint": {"save_every": 1 if smoke else 20},
            "gradient_attribution": {
                "enabled": True,
                "max_batches": 1,
                "interval_epochs": 1 if smoke else 10,
            },
        },
        "loss": {
            "coupling": {
                "weight": 0.03 if enabled else 0.0,
                "lag_focus_weight": 0.0,
                "smoothness_weight": 0.0,
                "lag_evidence_weight": 0.0,
                "pair_likelihood_weight": 0.30 if enabled else 0.0,
                "pair_gradient_target": routing,
                "pair_temperature": 0.5,
                "residualize_fnirs_marginal": True,
            },
        },
    }
    if smoke:
        config["training"]["max_train_batches"] = 1
        config["training"]["performance"] = {
            "tf32": True,
            "cuda_prefetch": False,
            "compile": {"enabled": False},
        }
        config["data"]["num_workers"] = 0
        config["data"]["dataloader"] = {"persistent_workers": False, "drop_last": False}
    return config


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def training_command(config: Path, run_name: str, smoke: bool) -> str:
    suffix = " --skip-post-analysis" if smoke else ""
    return (
        f"bash experiments/scripts/launch_training_nohup.sh --task source-observation-tokenizer "
        f"--foreground --config {q(config)} --run-name {q(run_name)}{suffix}"
    )


def audit_commands(
    suite_dir: Path,
    label: str,
    checkpoint: Path,
    device: str,
    *,
    frozen: bool,
) -> list[str]:
    export_dir = suite_dir / "token_exports" / label
    position_dir = suite_dir / "data_position_audit" / label
    raw_dir = suite_dir / "raw_information_audit" / label
    commands = [
        f".venv/bin/python experiments/scripts/export_coupling_audit_data.py --checkpoint {q(checkpoint)} "
        f"--output-dir {q(export_dir)} --label {q(label)} --device {q(device)}",
        f".venv/bin/python experiments/scripts/analyze_coupling_identifiability.py --export-dir {q(export_dir)} "
        f"--output-dir {q(position_dir)} --permutations 500 --bootstraps 1000",
        f"mkdir -p {q(raw_dir)}",
        f"cp {q(position_dir / 'physiological_information_ladder.json')} {q(raw_dir / 'physiological_information_ladder.json')}",
        f"cp {q(position_dir / 'physiological_information_ladder.csv')} {q(raw_dir / 'physiological_information_ladder.csv')}",
    ]
    if frozen:
        commands.append(
            f".venv/bin/python experiments/scripts/fit_frozen_coupling_models.py --export-dir {q(export_dir)} "
            f"--output-dir {q(suite_dir / 'frozen_calibration' / label)} --steps 500 --bootstraps 1000 "
            f"--position-audit {q(position_dir / 'position_event_lag_audit.json')}"
        )
    return commands


def intervention_commands(suite_dir: Path, suite_name: str, seed: int, device: str) -> list[str]:
    commands = []
    config_dir = suite_dir / "tokenizer_interventions" / "configs"
    run_group = suite_dir / "tokenizer_interventions"
    for condition in ("T0", "T1", "T2", "T3", "T4"):
        config_path = config_dir / f"{condition.lower()}_seed{seed}.yaml"
        run_name = f"{condition.lower()}_seed{seed}"
        commands.append(training_command(config_path, run_name, smoke=False))
        checkpoint = run_group / run_name / "checkpoints/best_model.pt"
        label = f"{condition.lower()}_seed{seed}"
        commands.extend(audit_commands(suite_dir, label, checkpoint, device, frozen=True))
    t5_config = config_dir / f"t5_seed{seed}.yaml"
    commands.append(
        f".venv/bin/python experiments/scripts/prepare_f6_intervention_config.py "
        f"--suite-dir {q(suite_dir)} --seed {seed} --device {q(device)} --output {q(t5_config)}"
    )
    t5_run = f"t5_seed{seed}"
    commands.append(
        f"if [[ -f {q(t5_config)} ]]; then "
        f"{training_command(t5_config, t5_run, smoke=False)}; "
        + "; ".join(audit_commands(
            suite_dir,
            t5_run,
            run_group / t5_run / "checkpoints/best_model.pt",
            device,
            frozen=True,
        ))
        + "; fi"
    )
    return commands


def write_queue(path: Path, commands: list[str], done_file: Path) -> None:
    content = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {q(project_root)}",
        "export PYTHONUNBUFFERED=1",
        *commands,
        f"touch {q(done_file)}",
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")
    path.chmod(0o755)


def initialize_suite(suite_dir: Path) -> Dict[str, Any]:
    for name in (
        "data_position_audit", "raw_information_audit", "token_exports",
        "frozen_calibration", "tokenizer_interventions", "long_validation", "queue_logs",
    ):
        (suite_dir / name).mkdir(parents=True, exist_ok=True)
    suite_name = suite_dir.name
    config_dir = suite_dir / "tokenizer_interventions" / "configs"
    for seed, device in ((20260615, "cuda:0"), (20260616, "cuda:1")):
        for condition in ("T0", "T1", "T2", "T3", "T4"):
            write_yaml(
                config_dir / f"{condition.lower()}_seed{seed}.yaml",
                intervention_config(suite_name, condition, seed, device, smoke=False),
            )
            write_yaml(
                suite_dir / "smoke" / "configs" / f"{condition.lower()}_seed{seed}.yaml",
                intervention_config(suite_name, condition, seed, device, smoke=True),
            )

    wait_for_phase_b = [
        f"while [[ ! -f {q(suite_dir / 'frozen_calibration/neutral/decision.json')} ]]; do sleep 60; done"
    ]
    gpu0_commands = audit_commands(suite_dir, "neutral", NEUTRAL_CHECKPOINT, "cuda:0", frozen=True)
    gpu0_commands += intervention_commands(suite_dir, suite_name, 20260615, "cuda:0")
    gpu1_commands = audit_commands(suite_dir, "shaped", SHAPED_CHECKPOINT, "cuda:1", frozen=False)
    gpu1_commands += wait_for_phase_b
    gpu1_commands += intervention_commands(suite_dir, suite_name, 20260616, "cuda:1")
    write_queue(
        suite_dir / "queue_logs/gpu0_queue.sh", gpu0_commands,
        suite_dir / "queue_logs/gpu0.done",
    )
    write_queue(
        suite_dir / "queue_logs/gpu1_queue.sh", gpu1_commands,
        suite_dir / "queue_logs/gpu1.done",
    )

    manifest = {
        "schema_version": "coupling_identifiability_suite_v1",
        "created_at": datetime.now().isoformat(),
        "suite_root": str(suite_dir),
        "git_commit": git_commit(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=project_root, text=True).strip()),
        "base_config": str(BASE_CONFIG),
        "base_config_sha256": sha256_file(BASE_CONFIG),
        "cache_manifest_sha256": cache_manifest_hash(),
        "checkpoints": {
            "neutral": {"path": str(NEUTRAL_CHECKPOINT), "sha256": sha256_file(NEUTRAL_CHECKPOINT)},
            "shaped": {"path": str(SHAPED_CHECKPOINT), "sha256": sha256_file(SHAPED_CHECKPOINT)},
        },
        "seeds": [20260615, 20260616, 20260617],
        "resources": resource_snapshot(),
        "queues": {
            "gpu0": str(suite_dir / "queue_logs/gpu0_queue.sh"),
            "gpu1": str(suite_dir / "queue_logs/gpu1_queue.sh"),
        },
        "intervention_configs": {
            str(path.relative_to(suite_dir)): sha256_file(path)
            for path in sorted(config_dir.glob("*.yaml"))
        },
        "commands": {
            "export": "experiments/scripts/export_coupling_audit_data.py",
            "position_information_audit": "experiments/scripts/analyze_coupling_identifiability.py",
            "frozen_model_comparison": "experiments/scripts/fit_frozen_coupling_models.py",
            "tokenizer_training": "experiments/scripts/launch_training_nohup.sh --task source-observation-tokenizer",
            "finalization": "experiments/scripts/finalize_coupling_identifiability_suite.py",
        },
    }
    (suite_dir / "suite_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for filename, payload in (
        ("summary.json", {"status": "initialized"}),
        ("decision.json", {"status": "pending"}),
    ):
        (suite_dir / filename).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (suite_dir / "summary.csv").write_text("phase,status\ninitialization,complete\n", encoding="utf-8")
    (suite_dir / "report.md").write_text(
        "# Coupling Identifiability Audit\n\nStatus: initialized. Phase A/B audits precede tokenizer interventions.\n",
        encoding="utf-8",
    )
    return manifest


def run_smoke(suite_dir: Path) -> None:
    processes = []
    for seed, gpu in ((20260615, 0), (20260616, 1)):
        commands = []
        for condition in ("T0", "T1", "T2", "T3", "T4"):
            config = suite_dir / "smoke/configs" / f"{condition.lower()}_seed{seed}.yaml"
            commands.append(training_command(config, f"smoke_{condition.lower()}_seed{seed}", smoke=True))
        log = (suite_dir / f"queue_logs/smoke_gpu{gpu}.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            ["bash", "-lc", "set -euo pipefail; " + "; ".join(commands)],
            cwd=project_root,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, log))
    failures = []
    for process, log in processes:
        return_code = process.wait()
        log.close()
        if return_code:
            failures.append(return_code)
    if failures:
        raise SystemExit(f"Smoke queues failed: {failures}")
    (suite_dir / "queue_logs/smoke.done").touch()


def launch_queues(suite_dir: Path) -> Dict[str, int]:
    pids = {}
    for gpu in (0, 1):
        queue = suite_dir / f"queue_logs/gpu{gpu}_queue.sh"
        log_path = suite_dir / f"queue_logs/gpu{gpu}_queue.log"
        log = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            ["bash", str(queue)],
            cwd=project_root,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pids[f"gpu{gpu}"] = process.pid
        log.close()
    (suite_dir / "queue_logs/pids.json").write_text(json.dumps(pids, indent=2) + "\n", encoding="utf-8")
    return pids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir")
    parser.add_argument("--mode", choices=("initialize", "smoke", "launch", "all"), default="all")
    args = parser.parse_args()
    if args.suite_dir:
        suite_dir = Path(args.suite_dir).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suite_dir = project_root / "experiments/runs/coupling_design_audit" / f"{stamp}_coupling_identifiability_v1"
    if args.mode in {"initialize", "all"}:
        initialize_suite(suite_dir)
    if args.mode in {"smoke", "all"}:
        run_smoke(suite_dir)
    pids = None
    if args.mode in {"launch", "all"}:
        if not (suite_dir / "queue_logs/smoke.done").exists():
            raise RuntimeError("Formal launch requires a completed smoke test")
        pids = launch_queues(suite_dir)
    print(json.dumps({"suite_dir": str(suite_dir), "pids": pids}, indent=2))


if __name__ == "__main__":
    main()
