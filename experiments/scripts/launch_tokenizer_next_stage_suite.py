#!/usr/bin/env python
"""Launch K128 vector-dim and nback/wg transfer tokenizer experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = PROJECT_ROOT / (
    "experiments/configs/source_observation/croce_local/"
    "highwl_v2_branch_norm_coupling0_lr2e4_compile.yaml"
)
SUITE_PARENT = PROJECT_ROOT / "experiments/runs/tokenizer_next_stage"
DIM_SPECS = (96, 128)
DIM_SEEDS = (20260623, 20260624)
COGNITIVE_SMOKE_SEED = 20260625
MAX_LAG_TOKENS = 5


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dim_run_name(dim: int, seed: int) -> str:
    return f"k128_dim{dim}_coupling0_seed{seed}"


def cognitive_run_name(dim: int, seed: int) -> str:
    return f"k128_dim{dim}_cognitive_nback_wg_seed{seed}"


def base_loss_config() -> Dict[str, Any]:
    return {
        "coupling": {
            "weight": 0.0,
            "max_lag_tokens": MAX_LAG_TOKENS,
            "lag_focus_weight": 0.0,
            "smoothness_weight": 0.0,
            "pair_likelihood_weight": 0.0,
            "lag_evidence_weight": 0.0,
            "effective_smoothness_weight": 0.0,
            "interaction_lag_sparsity_weight": 0.0,
        }
    }


def training_config(
    suite_name: str,
    *,
    dim: int,
    seed: int,
    device: str,
    smoke: bool,
    cognitive_only: bool,
) -> Dict[str, Any]:
    leaf = "smoke" if smoke else ("cognitive_tokenizer" if cognitive_only else "vector_dim_sweep")
    name = cognitive_run_name(dim, seed) if cognitive_only else dim_run_name(dim, seed)
    config: Dict[str, Any] = {
        "_base_": str(BASE_CONFIG),
        "experiment": {
            "name": name,
            "run_group": f"tokenizer_next_stage/{suite_name}/{leaf}",
            "seed": int(seed),
            "device": device,
            "description": (
                "K128 nback/wg-only tokenizer transfer test."
                if cognitive_only
                else "K128 source codebook vector-dim sweep with coupling disabled."
            ),
        },
        "data": {"seed": int(seed)},
        "model": {
            "source": {"codebook_size": 128, "codebook_dim": int(dim)},
            "eeg_observation": {"codebook_size": 256},
            "fnirs_observation": {"codebook_size": 128},
        },
        "loss": base_loss_config(),
        "training": {
            "batch_size": 256,
            "learning_rate": 2e-4,
            "min_lr": 5e-5,
            "epochs": 2 if smoke else (160 if cognitive_only else 120),
            "early_stopping": {
                "enabled": not smoke,
                "start_epoch": 41,
                "patience": 60,
                "metric": "val_primary_loss",
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
    if cognitive_only:
        config["data"]["entry_filters"] = {
            "include_source_names": ["simultaneous_cognitive"],
            "include_label_names": ["nback", "wg"],
        }
    if smoke:
        config["training"]["max_train_batches"] = 2
        config["training"]["performance"] = {
            "tf32": True,
            "cuda_prefetch": False,
            "compile": {"enabled": False},
        }
        config["data"]["num_workers"] = 0
        config["data"]["dataloader"] = {"persistent_workers": False, "drop_last": False}
    return config


def training_command(config_path: Path, run_name: str, *, smoke: bool) -> str:
    suffix = " --skip-post-analysis" if smoke else ""
    return (
        "bash experiments/scripts/launch_training_nohup.sh --task source-observation-tokenizer "
        f"--foreground --config {q(config_path)} --run-name {q(run_name)}{suffix}"
    )


def write_queue(path: Path, commands: list[str], done_file: Path) -> None:
    content = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {q(PROJECT_ROOT)}",
        "export PYTHONUNBUFFERED=1",
        *commands,
        f"touch {q(done_file)}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content) + "\n", encoding="utf-8")
    path.chmod(0o755)


def launch_queue(script: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = f"setsid -f nohup bash {q(script)} > {q(log_path)} 2>&1 < /dev/null"
    subprocess.run(cmd, cwd=PROJECT_ROOT, shell=True, check=True)
    return 0


def initialize_suite(suite_dir: Path) -> Dict[str, Any]:
    suite_name = suite_dir.name
    for name in (
        "vector_dim_sweep",
        "cognitive_tokenizer",
        "token_exports",
        "coupling_audit_exports",
        "position_audit",
        "codebook_geometry",
        "information_drop_audit",
        "local_coupling_audit",
        "downstream_probe",
        "queue_logs",
        "smoke",
    ):
        (suite_dir / name).mkdir(parents=True, exist_ok=True)

    smoke_by_gpu = {0: [], 1: []}
    vector_by_gpu = {0: [], 1: []}
    for dim in DIM_SPECS:
        for index, seed in enumerate(DIM_SEEDS):
            gpu = index % 2
            device = f"cuda:{gpu}"
            run_name = dim_run_name(dim, seed)
            config_path = suite_dir / "vector_dim_sweep" / "configs" / f"{run_name}.yaml"
            smoke_config_path = suite_dir / "smoke" / "configs" / f"{run_name}.yaml"
            write_yaml(config_path, training_config(suite_name, dim=dim, seed=seed, device=device, smoke=False, cognitive_only=False))
            write_yaml(smoke_config_path, training_config(suite_name, dim=dim, seed=seed, device=device, smoke=True, cognitive_only=False))
            vector_by_gpu[gpu].append(training_command(config_path, run_name, smoke=False))
            smoke_by_gpu[gpu].append(training_command(smoke_config_path, run_name, smoke=True))

    cognitive_smoke_name = cognitive_run_name(128, COGNITIVE_SMOKE_SEED)
    cognitive_smoke_cfg = suite_dir / "smoke" / "configs" / f"{cognitive_smoke_name}.yaml"
    write_yaml(
        cognitive_smoke_cfg,
        training_config(
            suite_name,
            dim=128,
            seed=COGNITIVE_SMOKE_SEED,
            device="cuda:0",
            smoke=True,
            cognitive_only=True,
        ),
    )
    smoke_by_gpu[0].append(training_command(cognitive_smoke_cfg, cognitive_smoke_name, smoke=True))

    qdir = suite_dir / "queue_logs"
    for gpu in (0, 1):
        write_queue(qdir / f"smoke_gpu{gpu}.sh", smoke_by_gpu[gpu], qdir / f"smoke_gpu{gpu}.done")
        write_queue(qdir / f"vector_gpu{gpu}.sh", vector_by_gpu[gpu], qdir / f"vector_gpu{gpu}.done")

    cognitive_queues: Dict[int, list[str]] = {0: [], 1: []}
    for gpu, seed in enumerate((20260625, 20260626)):
        cognitive_queues[gpu].extend([
            f"while [ ! -e {q(qdir / 'vector_gpu0.done')} ] || [ ! -e {q(qdir / 'vector_gpu1.done')} ]; do sleep 60; done",
            f".venv/bin/python experiments/scripts/finalize_tokenizer_next_stage_suite.py --suite-dir {q(suite_dir)} --write-cognitive-configs",
            f"selected_dim=$(.venv/bin/python - <<'PY'\nimport json\nfrom pathlib import Path\np=Path({str(suite_dir / 'selected_dim.json')!r})\nprint(json.loads(p.read_text())['selected_dim'])\nPY\n)",
            f"config={q(suite_dir / 'cognitive_tokenizer' / 'configs')}/k128_dim${{selected_dim}}_cognitive_nback_wg_seed{seed}.yaml",
            f"run_name=k128_dim${{selected_dim}}_cognitive_nback_wg_seed{seed}",
            "test -f \"$config\"",
            (
                "bash experiments/scripts/launch_training_nohup.sh --task source-observation-tokenizer "
                "--foreground --config \"$config\" --run-name \"$run_name\""
            ),
        ])
        write_queue(qdir / f"cognitive_gpu{gpu}.sh", cognitive_queues[gpu], qdir / f"cognitive_gpu{gpu}.done")

    audit_commands = [
        f"while [ ! -e {q(qdir / 'cognitive_gpu0.done')} ] || [ ! -e {q(qdir / 'cognitive_gpu1.done')} ]; do sleep 120; done",
        f".venv/bin/python experiments/scripts/run_tokenizer_next_stage_audits.py --suite-dir {q(suite_dir)} --device cuda:0",
    ]
    write_queue(qdir / "audit_queue.sh", audit_commands, qdir / "audit.done")

    manifest = {
        "schema_version": "tokenizer_next_stage_suite_v1",
        "suite_dir": str(suite_dir),
        "git_commit": git_commit(),
        "base_config": str(BASE_CONFIG),
        "base_config_sha256": sha256_file(BASE_CONFIG),
        "dim_specs": list(DIM_SPECS),
        "dim_seeds": list(DIM_SEEDS),
        "cognitive_seeds": [20260625, 20260626],
        "max_lag_tokens": MAX_LAG_TOKENS,
        "notes": [
            "Task/source/phase-aware coupling is diagnostic only in this suite.",
            "Tokenizer stage is evaluated as discrete physiological representation, not as final coupling model.",
        ],
    }
    write_json(suite_dir / "suite_manifest.json", manifest)
    write_json(suite_dir / "summary.json", {"status": "initialized"})
    write_json(suite_dir / "decision.json", {"status": "pending"})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-name", default=None)
    parser.add_argument("--suite-dir", default=None)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--launch-formal", action="store_true")
    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = Path(args.suite_dir).resolve() if args.suite_dir else SUITE_PARENT / (
        args.suite_name or f"{timestamp}_k128_dim_cognitive_transfer_v1"
    )
    manifest = initialize_suite(suite_dir)
    if args.launch or args.launch_formal:
        qdir = suite_dir / "queue_logs"
        for gpu in (0, 1):
            launch_queue(qdir / f"smoke_gpu{gpu}.sh", qdir / f"smoke_gpu{gpu}.log")
        if args.launch_formal:
            for name in ("vector_gpu0", "vector_gpu1", "cognitive_gpu0", "cognitive_gpu1", "audit_queue"):
                launch_queue(qdir / f"{name}.sh", qdir / f"{name}.log")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
