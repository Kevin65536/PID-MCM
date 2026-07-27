#!/usr/bin/env python3
"""Detached sequential runner for the frozen EFRM v2 LODO pretraining stages."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from typing import Any


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
PYTHON = REPO_ROOT / ".venv/bin/python"
TRAIN = METHOD_ROOT / "train_pretrain.py"
PROTOCOL_ID = "efrm_lodo_full_target_fivefold_v2"
DEFAULT_ROOT = METHOD_ROOT / "runs/formal" / PROTOCOL_ID


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def pid_alive(pid: int | None) -> bool:
    return bool(pid and pid > 0 and Path(f"/proc/{pid}").exists())


def completed_run(run_id: str, *, terminal: bool) -> bool:
    run_dir = METHOD_ROOT / "runs/pretraining" / run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = read_json(manifest_path)
    checkpoint = (
        run_dir / "checkpoints/terminal.pt"
        if terminal
        else run_dir / "checkpoints/best.pt"
    )
    return manifest.get("status") == "completed" and checkpoint.is_file()


def select_epoch(run_id: str, destination: Path, job: dict[str, Any]) -> dict[str, Any]:
    run_dir = METHOD_ROOT / "runs/pretraining" / run_id
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics/epochs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError(f"Stage-A run has no completed epochs: {run_id}")
    selected = min(rows, key=lambda row: float(row["validation"]["loss"]))
    epoch_count = int(selected["epoch"]) + 1
    checkpoint = run_dir / "checkpoints/best.pt"
    selection = {
        "schema": "efrm_lodo_stage_a_selection_v2",
        "protocol_id": PROTOCOL_ID,
        "frozen_at": now(),
        "excluded_target_dataset": job["excluded_target_dataset"],
        "selection_run_id": run_id,
        "selection_run_manifest_sha256": file_hash(run_dir / "manifest.json"),
        "selection_config_sha256": job["selection_config_sha256"],
        "lodo_manifest_sha256": job["manifest_sha256"],
        "selection_metric": "total_non_target_validation_loss",
        "selection_mode": "min",
        "completed_epoch_count": len(rows),
        "selected_epoch_zero_based": int(selected["epoch"]),
        "selected_epoch_count": epoch_count,
        "selected_validation_loss": float(selected["validation"]["loss"]),
        "selection_checkpoint": str(checkpoint.resolve()),
        "selection_checkpoint_sha256": file_hash(checkpoint),
        "target_dataset_exposure": False,
    }
    write_json(destination, selection)
    return selection


def run_child(command: list[str], log: Any, state_path: Path, state: dict[str, Any]) -> None:
    state["command"] = command
    state["child_started_at"] = now()
    child = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        close_fds=True,
    )
    state["training_pid"] = child.pid
    write_json(state_path, state)
    exit_code = child.wait()
    state["training_pid"] = None
    state["last_child_exit_code"] = exit_code
    state["child_finished_at"] = now()
    write_json(state_path, state)
    if exit_code != 0:
        raise RuntimeError(
            f"pretraining child failed with exit code {exit_code}: {' '.join(command)}"
        )


def update_protocol_status(
    protocol_root: Path,
    *,
    status: str,
    selection_completed: int,
    final_refit_completed: int,
) -> None:
    path = protocol_root / "status.json"
    payload = read_json(path)
    payload.update(
        {
            "status": status,
            "updated_at": now(),
            "selection_completed": selection_completed,
            "final_refit_completed": final_refit_completed,
            "protected_test_opened": False,
        }
    )
    write_json(path, payload)


def worker(protocol_root: Path, device: str, chunk_size: int) -> int:
    control = protocol_root / "pretraining_queue"
    state_path = control / "state.json"
    log_path = control / "queue.log"
    matrix = read_json(protocol_root / "protocol/pretraining_job_matrix.json")
    jobs = matrix["lodo_jobs"]
    state: dict[str, Any] = {
        "schema": "efrm_lodo_pretraining_queue_v2",
        "protocol_id": PROTOCOL_ID,
        "status": "running",
        "worker_pid": os.getpid(),
        "training_pid": None,
        "device": device,
        "chunk_size": chunk_size,
        "started_at": now(),
        "current_stage": None,
        "current_target_dataset": None,
        "selection_completed": 0,
        "final_refit_completed": 0,
        "protected_test_opened": False,
        "log_path": str(log_path.resolve()),
    }
    write_json(state_path, state)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        try:
            log.write(f"[queue] {now()} worker pid={os.getpid()} device={device}\n")
            selections: dict[str, dict[str, Any]] = {}
            selection_completed = 0
            for job in jobs:
                target = job["excluded_target_dataset"]
                run_id = job["selection_run_id"]
                state.update(
                    {
                        "current_stage": "stage_a_selection",
                        "current_target_dataset": target,
                        "current_run_id": run_id,
                    }
                )
                write_json(state_path, state)
                if not completed_run(run_id, terminal=False):
                    command = [
                        str(PYTHON),
                        "-u",
                        str(TRAIN),
                        "--config",
                        job["selection_config"],
                        "--run-id",
                        run_id,
                        "--device",
                        device,
                        "--chunk-size",
                        str(chunk_size),
                        "--num-workers",
                        "0",
                    ]
                    log.write(f"[queue] {now()} starting Stage A target={target}\n")
                    run_child(command, log, state_path, state)
                selection_path = (
                    protocol_root
                    / "protocol/selections"
                    / f"exclude_{target}.json"
                )
                if selection_path.is_file():
                    selection = read_json(selection_path)
                else:
                    selection = select_epoch(run_id, selection_path, job)
                selections[target] = selection
                selection_completed += 1
                state["selection_completed"] = selection_completed
                write_json(state_path, state)
                update_protocol_status(
                    protocol_root,
                    status="stage_a_selection_running"
                    if selection_completed < len(jobs)
                    else "stage_a_selection_completed",
                    selection_completed=selection_completed,
                    final_refit_completed=0,
                )

            final_completed = 0
            for job in jobs:
                target = job["excluded_target_dataset"]
                run_id = job["final_refit_run_id"]
                selection = selections[target]
                state.update(
                    {
                        "current_stage": "stage_b_final_refit",
                        "current_target_dataset": target,
                        "current_run_id": run_id,
                    }
                )
                write_json(state_path, state)
                if not completed_run(run_id, terminal=True):
                    command = [
                        str(PYTHON),
                        "-u",
                        str(TRAIN),
                        "--config",
                        job["final_refit_config"],
                        "--run-id",
                        run_id,
                        "--device",
                        device,
                        "--epochs",
                        str(selection["selected_epoch_count"]),
                        "--chunk-size",
                        str(chunk_size),
                        "--num-workers",
                        "0",
                    ]
                    log.write(f"[queue] {now()} starting Stage B target={target}\n")
                    run_child(command, log, state_path, state)
                final_run = METHOD_ROOT / "runs/pretraining" / run_id
                terminal = final_run / "checkpoints/terminal.pt"
                refit_manifest = {
                    "schema": "efrm_lodo_stage_b_refit_freeze_v2",
                    "protocol_id": PROTOCOL_ID,
                    "frozen_at": now(),
                    "excluded_target_dataset": target,
                    "final_refit_run_id": run_id,
                    "selected_epoch_count": selection["selected_epoch_count"],
                    "selected_by": str(
                        (
                            protocol_root
                            / "protocol/selections"
                            / f"exclude_{target}.json"
                        ).resolve()
                    ),
                    "final_refit_config_sha256": job[
                        "final_refit_config_sha256"
                    ],
                    "lodo_manifest_sha256": job["manifest_sha256"],
                    "terminal_checkpoint": str(terminal.resolve()),
                    "terminal_checkpoint_sha256": file_hash(terminal),
                    "run_manifest_sha256": file_hash(
                        final_run / "manifest.json"
                    ),
                    "target_dataset_exposure": False,
                }
                write_json(
                    protocol_root
                    / "protocol/final_refits"
                    / f"exclude_{target}.json",
                    refit_manifest,
                )
                final_completed += 1
                state["final_refit_completed"] = final_completed
                write_json(state_path, state)
                update_protocol_status(
                    protocol_root,
                    status="stage_b_final_refit_running"
                    if final_completed < len(jobs)
                    else "lodo_pretraining_completed",
                    selection_completed=selection_completed,
                    final_refit_completed=final_completed,
                )
            state.update(
                {
                    "status": "completed",
                    "finished_at": now(),
                    "current_stage": None,
                    "current_target_dataset": None,
                    "current_run_id": None,
                }
            )
            write_json(state_path, state)
            log.write(f"[queue] {now()} all LODO pretraining jobs completed\n")
            return 0
        except BaseException:
            state.update(
                {
                    "status": "failed",
                    "failed_at": now(),
                    "traceback": traceback.format_exc(),
                }
            )
            write_json(state_path, state)
            log.write(state["traceback"])
            update_protocol_status(
                protocol_root,
                status="lodo_pretraining_failed",
                selection_completed=int(state["selection_completed"]),
                final_refit_completed=int(state["final_refit_completed"]),
            )
            return 1


def start(protocol_root: Path, device: str, chunk_size: int) -> dict[str, Any]:
    control = protocol_root / "pretraining_queue"
    state_path = control / "state.json"
    if state_path.is_file():
        existing = read_json(state_path)
        if existing.get("status") == "running" and pid_alive(
            int(existing.get("worker_pid") or 0)
        ):
            raise RuntimeError(
                f"LODO queue already runs as pid {existing['worker_pid']}"
            )
    control.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON),
        "-u",
        str(Path(__file__).resolve()),
        "_worker",
        "--protocol-root",
        str(protocol_root),
        "--device",
        device,
        "--chunk-size",
        str(chunk_size),
    ]
    with open(os.devnull, "rb") as null_in, open(os.devnull, "ab") as null_out:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdin=null_in,
            stdout=null_out,
            stderr=null_out,
            start_new_session=True,
            close_fds=True,
        )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if state_path.is_file():
            state = read_json(state_path)
            if state.get("worker_pid") == process.pid:
                return state
        if process.poll() is not None:
            break
        time.sleep(0.05)
    raise RuntimeError("detached LODO queue did not initialize")


def status(protocol_root: Path) -> dict[str, Any]:
    state = read_json(protocol_root / "pretraining_queue/state.json")
    state["worker_alive"] = pid_alive(int(state.get("worker_pid") or 0))
    state["training_alive"] = pid_alive(int(state.get("training_pid") or 0))
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "status", "_worker"))
    parser.add_argument("--protocol-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=8)
    args = parser.parse_args()
    protocol_root = Path(args.protocol_root).resolve()
    if args.action == "_worker":
        return worker(protocol_root, args.device, args.chunk_size)
    if args.action == "start":
        result = start(protocol_root, args.device, args.chunk_size)
    else:
        result = status(protocol_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
