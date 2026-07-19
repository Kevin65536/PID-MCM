#!/usr/bin/env python3
"""Multi-fidelity Optuna tuning worker for one physical GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna
import yaml

METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
for path in (REPO_ROOT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sta_net_pytorch.data import STANetUnifiedTaskDataset, get_sta_net_task_spec
from sta_net_pytorch.splits import development_subject_split

RUNG_EPOCHS = (2, 8, 20, 40, 100)
SCHEMA = "sta_net_optuna_tuning_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_split(task: str, cache_root: str, seed: int, study_root: Path) -> Path:
    path = study_root / "splits" / task / "development_cross_subject.json"
    dataset = STANetUnifiedTaskDataset(get_sta_net_task_spec(task), cache_root=cache_root)
    _, _, manifest = development_subject_split(dataset, seed)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("split_sha256") != manifest.get("split_sha256"):
            raise RuntimeError(f"immutable split drift for {task}")
        return path
    write_json(path, manifest)
    return path


def sample_config(trial: optuna.Trial, base: dict[str, Any], task: str) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    config["model"]["dropout"] = trial.suggest_float("dropout", 0.2, 0.6)
    config["loss"]["eeg_aux_weight"] = trial.suggest_categorical(
        "eeg_aux_weight", [0.0, 0.25, 0.5, 1.0]
    )
    config["loss"]["alignment_weight"] = trial.suggest_categorical(
        "alignment_weight", [0.0, 0.1, 0.3, 1.0]
    )
    if task != "refed_regression":
        config["loss"]["class_weighting"] = trial.suggest_categorical(
            "class_weighting", ["none", "inverse_sqrt", "inverse_frequency"]
        )
        config["loss"]["label_smoothing"] = trial.suggest_categorical(
            "label_smoothing", [0.0, 0.05, 0.1]
        )
    training = config["training"]
    training["lr"] = trial.suggest_float("lr", 3e-5, 5e-4, log=True)
    training["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    training["grad_clip_norm"] = trial.suggest_categorical("grad_clip_norm", [0.5, 1.0, 2.0, 5.0])
    training["scheduler"] = trial.suggest_categorical("scheduler", ["constant", "cosine"])
    training["warmup_ratio"] = trial.suggest_categorical("warmup_ratio", [0.0, 0.05, 0.1])
    if task == "refed_regression":
        choices = [4, 8]
    elif task == "visual":
        choices = [16, 24, 32]
    else:
        choices = [16, 32, 64]
    config.setdefault("task_overrides", {}).setdefault(task, {})["batch_size"] = trial.suggest_categorical(
        "batch_size", choices
    )
    training["selection_metric"] = "masked_rmse_scaled" if task == "refed_regression" else "macro_f1"
    training["selection_mode"] = "min" if task == "refed_regression" else "max"
    config["tuning"] = {
        "schema": SCHEMA,
        "trial_number": trial.number,
        "rung_epochs": list(RUNG_EPOCHS),
        "selection_metric": training["selection_metric"],
    }
    return config


def last_validation_metric(run_dir: Path, task: str, epoch: int) -> tuple[float, dict[str, Any]]:
    path = run_dir / "metrics" / "validation_epochs.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    matches = [row for row in rows if int(row["epoch"]) == epoch]
    if not matches:
        raise RuntimeError(f"trial {run_dir} emitted no validation metrics for epoch {epoch}")
    row = matches[-1]
    value = float(row["masked_rmse_scaled"] if task == "refed_regression" else row["macro_f1"])
    return (-value if task == "refed_regression" else value), row


def objective_factory(
    *,
    task: str,
    base_config: dict[str, Any],
    split_path: Path,
    study_root: Path,
    physical_gpu: int,
):
    def objective(trial: optuna.Trial) -> float:
        trial_dir = study_root / "trials" / task / f"trial_{trial.number:05d}"
        trial_dir.mkdir(parents=True, exist_ok=False)
        config = sample_config(trial, base_config, task)
        config_path = trial_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        write_json(trial_dir / "trial_manifest.json", {
            "schema": SCHEMA, "status": "running", "task": task,
            "trial_number": trial.number, "physical_gpu": physical_gpu,
            "created_at": utc_now(), "rung_epochs": list(RUNG_EPOCHS),
            "split_manifest": str(split_path), "parameters": trial.params,
        })
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": str(physical_gpu),
            "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2",
        })
        score = float("-inf")
        rung_rows = []
        for rung_index, epochs in enumerate(RUNG_EPOCHS):
            command = [
                sys.executable, "-u", str(METHOD_ROOT / "train.py"),
                "--config", str(config_path), "--task", task,
                "--device", "cuda:0", "--output-dir", str(trial_dir / "run"),
                "--split-manifest", str(split_path), "--epochs", str(epochs),
            ]
            latest = trial_dir / "run" / "checkpoint_latest.pt"
            if latest.exists():
                command.extend(["--resume", str(latest)])
            with (trial_dir / "process.log").open("a", encoding="utf-8") as log:
                completed = subprocess.run(
                    command, cwd=REPO_ROOT, env=environment, stdout=log,
                    stderr=subprocess.STDOUT, check=False,
                )
            if completed.returncode != 0:
                write_json(trial_dir / "trial_manifest.json", {
                    "schema": SCHEMA, "status": "failed", "task": task,
                    "trial_number": trial.number, "physical_gpu": physical_gpu,
                    "failed_at_epochs": epochs, "returncode": completed.returncode,
                    "failed_at": utc_now(), "parameters": trial.params,
                })
                raise RuntimeError(f"training failed at rung={epochs}; see {trial_dir / 'process.log'}")
            score, metric_row = last_validation_metric(trial_dir / "run", task, epochs)
            rung_rows.append({"rung": rung_index, "epochs": epochs, "score": score, "metrics": metric_row})
            write_json(trial_dir / "rungs.json", {"schema": SCHEMA, "rungs": rung_rows})
            trial.report(score, step=rung_index + 1)
            if epochs < RUNG_EPOCHS[-1] and trial.should_prune():
                write_json(trial_dir / "trial_manifest.json", {
                    "schema": SCHEMA, "status": "pruned", "task": task,
                    "trial_number": trial.number, "physical_gpu": physical_gpu,
                    "pruned_at_epochs": epochs, "score": score, "parameters": trial.params,
                })
                raise optuna.TrialPruned(f"pruned after {epochs} epochs")
        write_json(trial_dir / "trial_manifest.json", {
            "schema": SCHEMA, "status": "completed_100_epochs", "task": task,
            "trial_number": trial.number, "physical_gpu": physical_gpu,
            "completed_at": utc_now(), "score": score, "parameters": trial.params,
        })
        return score
    return objective


def run_task(args: argparse.Namespace, task: str, base: dict[str, Any], root: Path) -> None:
    cache_root = str(base["data"]["cache_root"])
    split_path = prepare_split(task, cache_root, int(base["training"].get("seed", 42)), root)
    storage = f"sqlite:///{(root / 'optuna.sqlite3').resolve()}"
    study = optuna.create_study(
        study_name=f"{args.study_id}__{task}__development_cross_subject",
        storage=storage, direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=int(base["training"].get("seed", 42)), n_startup_trials=6),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=len(RUNG_EPOCHS), reduction_factor=2,
        ),
    )
    study.optimize(
        objective_factory(
            task=task, base_config=base, split_path=split_path,
            study_root=root, physical_gpu=args.physical_gpu,
        ),
        n_trials=args.n_trials, gc_after_trial=True, catch=(RuntimeError,),
    )
    write_json(root / "studies" / f"{task}.json", {
        "schema": SCHEMA, "task": task, "study_name": study.study_name,
        "trial_count": len(study.trials), "best_value": study.best_value,
        "best_params": study.best_params, "updated_at": utc_now(),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--base-config", default=str(METHOD_ROOT / "configs" / "tuning_base.yaml"))
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--n-trials", type=int, default=12)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    base_path = Path(args.base_config).resolve()
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    for task in args.tasks:
        run_task(args, task, base, root)


if __name__ == "__main__":
    main()
