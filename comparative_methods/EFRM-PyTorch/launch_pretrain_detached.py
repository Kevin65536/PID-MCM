#!/usr/bin/env python3
"""Launch EFRM pretraining outside the caller's terminal session.

The public launcher starts a small detached supervisor with a new POSIX
session, /dev/null stdin, and file-backed stdout/stderr.  The supervisor keeps
the training PID and exit status auditable even after the invoking shell or a
temporary Codex PTY disappears.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
PYTHON_BIN = REPO_ROOT / ".venv/bin/python"
TRAIN_ENTRYPOINT = METHOD_ROOT / "train_pretrain.py"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
STATE_SCHEMA = "efrm_detached_launcher_state_v1"


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _pid_is_launcher(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        return False
    return b"launch_pretrain_detached.py" in command


def _extract_training_request(arguments: Sequence[str]) -> tuple[str, bool]:
    run_ids: list[str] = []
    resume = False
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value == "--run-id":
            if index + 1 >= len(arguments):
                raise ValueError("--run-id requires a value")
            run_ids.append(arguments[index + 1])
            index += 2
            continue
        if value.startswith("--run-id="):
            run_ids.append(value.split("=", 1)[1])
        elif value == "--resume":
            resume = True
        index += 1
    if len(run_ids) != 1:
        raise ValueError("detached launch requires exactly one --run-id")
    run_id = run_ids[0]
    if not RUN_ID_PATTERN.fullmatch(run_id) or ".." in run_id:
        raise ValueError(f"unsafe run id: {run_id!r}")
    return run_id, resume


def _active_launch(control_dir: Path) -> dict[str, Any] | None:
    for path in (control_dir / "state.json", control_dir / "request.json"):
        payload = _read_json(path)
        if payload is None:
            continue
        status = str(payload.get("status", ""))
        supervisor_pid = payload.get("supervisor_pid")
        if status in {"submitted", "starting", "running"} and _pid_is_launcher(
            int(supervisor_pid) if supervisor_pid is not None else None
        ):
            return payload
    return None


def _update_run_terminal_state(
    run_dir: Path,
    *,
    status: str,
    exit_code: int,
    launcher_state_path: Path,
) -> None:
    terminal_at = _now()
    status_path = run_dir / "status.json"
    status_payload = _read_json(status_path) or {}
    status_payload.update({
        "status": status,
        "terminal_at": terminal_at,
        "exit_code": exit_code,
        "termination_signal": (
            signal.Signals(-exit_code).name if exit_code < 0 else None
        ),
        "launcher_state": str(launcher_state_path.resolve()),
    })
    if run_dir.is_dir():
        _write_json(status_path, status_payload)

    manifest_path = run_dir / "manifest.json"
    manifest_payload = _read_json(manifest_path)
    if manifest_payload is not None:
        manifest_payload["status"] = status
        manifest_payload["launcher_state"] = str(launcher_state_path.resolve())
        if status == "completed":
            manifest_payload.setdefault("completed_at", terminal_at)
        else:
            manifest_payload["failed_at"] = terminal_at
            manifest_payload["exit_code"] = exit_code
        _write_json(manifest_path, manifest_payload)


def _supervise(
    *,
    command: Sequence[str],
    run_id: str,
    run_dir: Path,
    control_dir: Path,
    launch_id: str,
    log_path: Path,
) -> int:
    state_path = control_dir / "state.json"
    state: dict[str, Any] = {
        "schema": STATE_SCHEMA,
        "status": "starting",
        "run_id": run_id,
        "launch_id": launch_id,
        "submitted_at": _now(),
        "supervisor_pid": os.getpid(),
        "session_id": os.getsid(0),
        "process_group_id": os.getpgrp(),
        "training_pid": None,
        "command": list(command),
        "log_path": str(log_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "exit_code": None,
        "termination_signal": None,
    }
    _write_json(state_path, state)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(
            f"[launcher] {state['submitted_at']} detached supervisor "
            f"pid={os.getpid()} sid={os.getsid(0)} pgrp={os.getpgrp()}\n"
        )
        log.write(f"[launcher] command={json.dumps(list(command))}\n")
        try:
            environment = dict(os.environ)
            environment["PYTHONUNBUFFERED"] = "1"
            with Path(os.devnull).open("rb") as null_input:
                child = subprocess.Popen(
                    list(command),
                    cwd=REPO_ROOT,
                    env=environment,
                    stdin=null_input,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    close_fds=True,
                )
            state.update({
                "status": "running",
                "started_at": _now(),
                "training_pid": child.pid,
            })
            _write_json(state_path, state)

            forwarded_signal: int | None = None

            def forward(signum: int, _frame: Any) -> None:
                nonlocal forwarded_signal
                forwarded_signal = signum
                if child.poll() is None:
                    child.send_signal(signum)

            signal.signal(signal.SIGTERM, forward)
            signal.signal(signal.SIGINT, forward)
            exit_code = child.wait()
            terminal_status = "completed" if exit_code == 0 else "failed"
            state.update({
                "status": terminal_status,
                "finished_at": _now(),
                "exit_code": exit_code,
                "termination_signal": (
                    signal.Signals(-exit_code).name if exit_code < 0 else None
                ),
                "forwarded_signal": (
                    signal.Signals(forwarded_signal).name
                    if forwarded_signal is not None else None
                ),
            })
            _update_run_terminal_state(
                run_dir,
                status=terminal_status,
                exit_code=exit_code,
                launcher_state_path=state_path,
            )
            _write_json(state_path, state)
            log.write(
                f"[launcher] {state['finished_at']} status={terminal_status} "
                f"exit_code={exit_code}\n"
            )
            return exit_code
        except BaseException:
            state.update({
                "status": "launcher_failed",
                "finished_at": _now(),
                "exit_code": 125,
                "launcher_traceback": traceback.format_exc(),
            })
            _write_json(state_path, state)
            log.write(state["launcher_traceback"])
            return 125


def launch_detached(
    *,
    command: Sequence[str],
    run_id: str,
    run_dir: Path,
    control_dir: Path,
) -> dict[str, Any]:
    """Start a detached supervisor and return its durable launch request."""

    control_dir.mkdir(parents=True, exist_ok=True)
    with (control_dir / "launch.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        active = _active_launch(control_dir)
        if active is not None:
            raise RuntimeError(
                f"run {run_id!r} already has an active detached launch "
                f"(supervisor pid {active.get('supervisor_pid')})"
            )
        launch_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = control_dir / "logs" / f"{launch_id}.log"
        state_path = control_dir / "state.json"
        supervisor_command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_supervise",
            "--run-id",
            run_id,
            "--run-dir",
            str(run_dir.resolve()),
            "--control-dir",
            str(control_dir.resolve()),
            "--launch-id",
            launch_id,
            "--log-path",
            str(log_path.resolve()),
            "--",
            *command,
        ]
        with (
            Path(os.devnull).open("rb") as null_input,
            Path(os.devnull).open("ab") as null_output,
        ):
            supervisor = subprocess.Popen(
                supervisor_command,
                cwd=REPO_ROOT,
                stdin=null_input,
                stdout=null_output,
                stderr=null_output,
                start_new_session=True,
                close_fds=True,
            )
        request = {
            "schema": STATE_SCHEMA,
            "status": "submitted",
            "run_id": run_id,
            "launch_id": launch_id,
            "submitted_at": _now(),
            "submitter_pid": os.getpid(),
            "supervisor_pid": supervisor.pid,
            "command": list(command),
            "log_path": str(log_path.resolve()),
            "state_path": str(state_path.resolve()),
            "run_dir": str(run_dir.resolve()),
        }
        _write_json(control_dir / "request.json", request)
        _write_json(control_dir / "launches" / f"{launch_id}.json", request)

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        state = _read_json(state_path)
        if state is not None and state.get("launch_id") == launch_id:
            return state
        if supervisor.poll() is not None:
            break
        time.sleep(0.05)
    return request


def _status(run_id: str) -> int:
    if not RUN_ID_PATTERN.fullmatch(run_id) or ".." in run_id:
        raise ValueError(f"unsafe run id: {run_id!r}")
    control_dir = METHOD_ROOT / "runs/launcher" / run_id
    payload = _read_json(control_dir / "state.json") or _read_json(
        control_dir / "request.json"
    )
    if payload is None:
        raise FileNotFoundError(f"no detached-launch state for {run_id}")
    payload["supervisor_alive"] = _pid_is_launcher(payload.get("supervisor_pid"))
    training_pid = payload.get("training_pid")
    payload["training_alive"] = bool(
        training_pid and Path(f"/proc/{int(training_pid)}").exists()
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _public_launch(arguments: Sequence[str]) -> int:
    run_id, resume = _extract_training_request(arguments)
    if not PYTHON_BIN.is_file():
        raise FileNotFoundError(f"missing project Python: {PYTHON_BIN}")
    run_dir = METHOD_ROOT / "runs/pretraining" / run_id
    latest = run_dir / "checkpoints/latest.pt"
    if run_dir.exists() and not resume:
        raise FileExistsError(
            f"run directory already exists; pass --resume only for a compatible "
            f"checkpoint: {run_dir}"
        )
    if resume and not latest.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {latest}")
    command = [str(PYTHON_BIN), "-u", str(TRAIN_ENTRYPOINT), *arguments]
    state = launch_detached(
        command=command,
        run_id=run_id,
        run_dir=run_dir,
        control_dir=METHOD_ROOT / "runs/launcher" / run_id,
    )
    print(json.dumps({
        "run_id": run_id,
        "status": state["status"],
        "supervisor_pid": state["supervisor_pid"],
        "training_pid": state.get("training_pid"),
        "log_path": state["log_path"],
        "state_path": str(
            (METHOD_ROOT / "runs/launcher" / run_id / "state.json").resolve()
        ),
    }, indent=2))
    if state["status"] in {"failed", "launcher_failed"}:
        return int(state.get("exit_code") or 1)
    return 0


def _internal_supervise(arguments: Sequence[str]) -> int:
    if "--" not in arguments:
        raise ValueError("internal supervisor requires -- before the training command")
    boundary = arguments.index("--")
    options, command = list(arguments[:boundary]), list(arguments[boundary + 1:])
    values: dict[str, str] = {}
    index = 0
    while index < len(options):
        key = options[index]
        if not key.startswith("--") or index + 1 >= len(options):
            raise ValueError(f"invalid internal supervisor option: {key}")
        values[key[2:].replace("-", "_")] = options[index + 1]
        index += 2
    required = {"run_id", "run_dir", "control_dir", "launch_id", "log_path"}
    missing = required.difference(values)
    if missing or not command:
        raise ValueError(f"missing internal supervisor values: {sorted(missing)}")
    return _supervise(
        command=command,
        run_id=values["run_id"],
        run_dir=Path(values["run_dir"]),
        control_dir=Path(values["control_dir"]),
        launch_id=values["launch_id"],
        log_path=Path(values["log_path"]),
    )


def main() -> int:
    arguments = sys.argv[1:]
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(
            "Usage:\n"
            "  .venv/bin/python comparative_methods/EFRM-PyTorch/"
            "launch_pretrain_detached.py --run-id RUN_ID [train_pretrain.py args]\n"
            "  .venv/bin/python comparative_methods/EFRM-PyTorch/"
            "launch_pretrain_detached.py status RUN_ID"
        )
        return 0
    if arguments[0] == "_supervise":
        return _internal_supervise(arguments[1:])
    if arguments[0] == "status":
        if len(arguments) != 2:
            raise ValueError("status requires exactly one run id")
        return _status(arguments[1])
    return _public_launch(arguments)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
