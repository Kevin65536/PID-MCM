#!/usr/bin/env python
"""Launch the large-codebook and mission-bias coupling capacity suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


project_root = Path(__file__).resolve().parents[2]
BASE_CONFIG = project_root / (
    "experiments/configs/source_observation/croce_local/"
    "highwl_v2_branch_norm_coupling0_lr2e4_compile.yaml"
)
SUITE_PARENT = project_root / "experiments/runs/tokenizer_coupling_capacity"
MAX_LAG_TOKENS = 5


@dataclass(frozen=True)
class CapacitySpec:
    key: str
    source_k: int
    eeg_obs_k: int
    fnirs_obs_k: int
    seeds: tuple[int, ...]
    dense_frozen: bool


CAPACITY_SPECS = (
    CapacitySpec("K32", 32, 64, 64, (20260619, 20260620), True),
    CapacitySpec("K64", 64, 128, 128, (20260619, 20260620), True),
    CapacitySpec("K128", 128, 256, 128, (20260619, 20260620), True),
    CapacitySpec("K256_probe", 256, 256, 128, (20260619,), False),
)


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_manifest_hash() -> str:
    root = project_root / "croce_validation/cache/croce_local/highwl_v2"
    digest = hashlib.sha256()
    for path in sorted(root.rglob("cache_manifest.json")):
        digest.update(str(path.relative_to(project_root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()


def resource_snapshot() -> Dict[str, Any]:
    payload: Dict[str, Any] = {"cpu_count": os.cpu_count()}
    try:
        lines = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip().splitlines()
        payload["gpus"] = lines
    except Exception as exc:  # pragma: no cover - environment dependent
        payload["gpu_error"] = str(exc)
    return payload


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def run_name(spec: CapacitySpec, seed: int) -> str:
    return f"s2_croce_local_highwl_v2_capacity_{spec.key.lower()}_coupling0_seed{seed}"


def capacity_config(
    suite_name: str,
    spec: CapacitySpec,
    seed: int,
    device: str,
    *,
    smoke: bool,
) -> Dict[str, Any]:
    group_leaf = "smoke" if smoke else "capacity_sweep"
    config: Dict[str, Any] = {
        "_base_": str(BASE_CONFIG),
        "experiment": {
            "name": run_name(spec, seed),
            "run_group": f"tokenizer_coupling_capacity/{suite_name}/{group_leaf}",
            "seed": int(seed),
            "device": device,
            "description": (
                "Coupling-free source/observation tokenizer capacity sweep. "
                "Any EEG-fNIRS pairing is spontaneous aligned-data structure, not explicit model exchange."
            ),
        },
        "data": {"seed": int(seed)},
        "model": {
            "source": {"codebook_size": spec.source_k},
            "eeg_observation": {"codebook_size": spec.eeg_obs_k},
            "fnirs_observation": {"codebook_size": spec.fnirs_obs_k},
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
            },
        },
        "training": {
            "batch_size": 256,
            "learning_rate": 2e-4,
            "min_lr": 5e-5,
            "epochs": 1 if smoke else 200,
            "early_stopping": {
                "enabled": not smoke,
                "start_epoch": 61,
                "patience": 80,
                "monitor": "val_loss",
                "mode": "min",
            },
            "validation": {
                "interval_epochs": 1 if smoke else 2,
                "start_epoch": 1,
                "max_batches": 1 if smoke else None,
            },
            "checkpoint": {"save_every": 1 if smoke else 20},
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


def training_command(config: Path, name: str, *, smoke: bool) -> str:
    suffix = " --skip-post-analysis" if smoke else ""
    return (
        "bash experiments/scripts/launch_training_nohup.sh --task source-observation-tokenizer "
        f"--foreground --config {q(config)} --run-name {q(name)}{suffix}"
    )


def audit_commands(suite_dir: Path, spec: CapacitySpec, seed: int, device: str) -> list[str]:
    name = run_name(spec, seed)
    run_dir = suite_dir / "capacity_sweep" / name
    checkpoint = run_dir / "checkpoints/best_model.pt"
    export_dir = suite_dir / "token_exports" / name
    position_dir = suite_dir / "position_audit" / name
    frozen_dir = suite_dir / "frozen_pairing_audit" / name
    commands = [
        f"test -f {q(checkpoint)}",
        f".venv/bin/python experiments/scripts/export_coupling_audit_data.py --checkpoint {q(checkpoint)} "
        f"--output-dir {q(export_dir)} --label {q(name)} --device {q(device)}",
        f".venv/bin/python experiments/scripts/analyze_coupling_identifiability.py --export-dir {q(export_dir)} "
        f"--output-dir {q(position_dir)} --permutations 500 --bootstraps 1000 "
        f"--max-lag-tokens {MAX_LAG_TOKENS}",
    ]
    if spec.dense_frozen:
        commands.append(
            f".venv/bin/python experiments/scripts/fit_frozen_coupling_models.py --export-dir {q(export_dir)} "
            f"--output-dir {q(frozen_dir)} --steps 500 --bootstraps 1000 "
            f"--position-audit {q(position_dir / 'position_event_lag_audit.json')} "
            f"--max-lag-tokens {MAX_LAG_TOKENS}"
        )
    else:
        commands.append(
            f".venv/bin/python experiments/scripts/fit_frozen_coupling_models.py --export-dir {q(export_dir)} "
            f"--output-dir {q(frozen_dir)} --steps 500 --bootstraps 1000 "
            f"--factorized-only --factorized-ranks 16 32 --max-lag-tokens {MAX_LAG_TOKENS}"
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


def dense_tensor_parameter_table() -> Dict[str, Dict[str, int]]:
    return {
        spec.key: {
            "source_k": spec.source_k,
            "dense_single_tensor_parameters": (MAX_LAG_TOKENS + 1) * spec.source_k * spec.source_k,
            "dense_frozen_enabled": int(spec.dense_frozen),
        }
        for spec in CAPACITY_SPECS
    }


def initialize_suite(suite_dir: Path) -> Dict[str, Any]:
    for name in (
        "capacity_sweep", "token_exports", "position_audit", "frozen_pairing_audit",
        "matched_coupling_lift", "queue_logs", "smoke",
    ):
        (suite_dir / name).mkdir(parents=True, exist_ok=True)
    suite_name = suite_dir.name
    configs = []
    smoke_configs = []
    schedule = {0: [], 1: []}
    for spec in CAPACITY_SPECS:
        for index, seed in enumerate(spec.seeds):
            gpu = index % 2
            if spec.key == "K256_probe":
                gpu = 0
            device = f"cuda:{gpu}"
            name = run_name(spec, seed)
            config_path = suite_dir / "capacity_sweep" / "configs" / f"{name}.yaml"
            smoke_config_path = suite_dir / "smoke" / "configs" / f"{name}.yaml"
            write_yaml(config_path, capacity_config(suite_name, spec, seed, device, smoke=False))
            write_yaml(smoke_config_path, capacity_config(suite_name, spec, seed, device, smoke=True))
            configs.append(config_path)
            smoke_configs.append(smoke_config_path)
            schedule[gpu].append((spec, seed, config_path, smoke_config_path, device))

    for gpu, entries in schedule.items():
        commands = []
        smoke_commands = []
        for spec, seed, config_path, smoke_config_path, device in entries:
            name = run_name(spec, seed)
            commands.append(training_command(config_path, name, smoke=False))
            commands.extend(audit_commands(suite_dir, spec, seed, device))
            smoke_commands.append(training_command(smoke_config_path, f"smoke_{name}", smoke=True))
        write_queue(suite_dir / f"queue_logs/gpu{gpu}_queue.sh", commands, suite_dir / f"queue_logs/gpu{gpu}.done")
        write_queue(
            suite_dir / f"queue_logs/smoke_gpu{gpu}_queue.sh",
            smoke_commands,
            suite_dir / f"queue_logs/smoke_gpu{gpu}.done",
        )

    (suite_dir / "matched_coupling_lift" / "README.md").write_text(
        "Phase C is gated by Phase A/B results. Select one capacity by summary criteria before launching "
        "matched coupling lift conditions C0-C5.\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "tokenizer_coupling_capacity_v1",
        "created_at": datetime.now().isoformat(),
        "suite_root": str(suite_dir),
        "git_commit": git_commit(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=project_root, text=True).strip()),
        "base_config": str(BASE_CONFIG),
        "base_config_sha256": sha256_file(BASE_CONFIG),
        "cache_manifest_sha256": cache_manifest_hash(),
        "max_lag_tokens": MAX_LAG_TOKENS,
        "effective_lags": list(range(MAX_LAG_TOKENS + 1)),
        "capacity_specs": [spec.__dict__ for spec in CAPACITY_SPECS],
        "dense_tensor_parameters": dense_tensor_parameter_table(),
        "resources": resource_snapshot(),
        "queues": {
            "gpu0": str(suite_dir / "queue_logs/gpu0_queue.sh"),
            "gpu1": str(suite_dir / "queue_logs/gpu1_queue.sh"),
        },
        "formal_configs": {
            str(path.relative_to(suite_dir)): sha256_file(path)
            for path in sorted(configs)
        },
        "smoke_configs": {
            str(path.relative_to(suite_dir)): sha256_file(path)
            for path in sorted(smoke_configs)
        },
        "phase_c_status": "gated_until_phase_a_b_complete",
    }
    (suite_dir / "suite_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (suite_dir / "summary.json").write_text(json.dumps({"status": "initialized"}, indent=2) + "\n", encoding="utf-8")
    (suite_dir / "decision.json").write_text(json.dumps({"status": "pending_phase_a_b"}, indent=2) + "\n", encoding="utf-8")
    (suite_dir / "summary.csv").write_text("phase,status\ninitialization,complete\n", encoding="utf-8")
    (suite_dir / "report.md").write_text(
        "# Tokenizer Coupling Capacity Suite\n\n"
        "Status: initialized. Phase A trains coupling-free capacity sweep; Phase B audits spontaneous pairing.\n",
        encoding="utf-8",
    )
    return manifest


def run_smoke(suite_dir: Path) -> None:
    failures = []
    for gpu in (0, 1):
        queue = suite_dir / f"queue_logs/smoke_gpu{gpu}_queue.sh"
        log_path = suite_dir / f"queue_logs/smoke_gpu{gpu}_queue.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                ["bash", str(queue)],
                cwd=project_root,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            return_code = process.wait()
        if return_code:
            failures.append({"gpu": gpu, "return_code": return_code, "log": str(log_path)})
    if failures:
        raise SystemExit(json.dumps({"smoke_failures": failures}, indent=2))
    (suite_dir / "queue_logs/smoke.done").touch()


def launch_queues(suite_dir: Path) -> Dict[str, int]:
    if not (suite_dir / "queue_logs/smoke.done").exists():
        raise RuntimeError("Formal launch requires a completed smoke test")
    pids: Dict[str, int] = {}
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
        suite_dir = SUITE_PARENT / f"{stamp}_codebook_mission_bias_v1"
    if args.mode in {"initialize", "all"}:
        initialize_suite(suite_dir)
    if args.mode in {"smoke", "all"}:
        run_smoke(suite_dir)
    pids = None
    if args.mode in {"launch", "all"}:
        pids = launch_queues(suite_dir)
    print(json.dumps({"suite_dir": str(suite_dir), "pids": pids}, indent=2))


if __name__ == "__main__":
    main()
