"""REVE-owned public data boundary for adapter-alignment v2 auditing."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from comparative_methods.CBraMod.alignment_data import (
    TASK_SPECS,
    PublicInventory,
    RecordGroupedBatchSampler,
    CBraModPublicView,
    data_branch_fingerprints as _shared_data_branch_fingerprints,
    load_public_inventory as _shared_load_public_inventory,
    make_loader,
    resolve_repo_path,
    sample_id,
    stable_hash,
)
from comparative_methods.audit_public_preflight import EXPECTED_REGISTRY_SHA256


METHOD_ROOT = Path(__file__).resolve().parent
CONFIG_SCHEMA = "reve_adapter_alignment_v2"
SUPPORTED_TASKS = (
    "motor_imagery",
    "mental_arithmetic",
    "wg",
    "nback",
    "dsr",
    "visual",
)


def _official_position_names() -> set[str]:
    path = METHOD_ROOT / "checkpoints/reve-positions/config.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing pinned REVE position-bank config: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    names = value.get("position_names", [])
    if len(names) != 543 or len(set(names)) != 543:
        raise ValueError("REVE official position-bank config must contain 543 unique names")
    return {str(name) for name in names}


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = resolve_repo_path(path)
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"expected {CONFIG_SCHEMA} config: {config_path}")
    if value.get("method_id") != "reve" or value.get("mode") != "public_audit_only":
        raise PermissionError("REVE alignment config must remain a public-only REVE audit")
    if value.get("protected_test_default") != "locked":
        raise PermissionError("protected test must default to locked")
    if value.get("registry", {}).get("registry_sha256") != EXPECTED_REGISTRY_SHA256:
        raise ValueError("method-neutral registry fingerprint drifted")
    if int(value["data"]["panel_size"]) != 16:
        raise ValueError("support-matched REVE audit requires a 16-channel panel")
    if float(value["data"]["eeg_sample_rate_hz"]) != 200.0:
        raise ValueError("REVE alignment audit requires canonical 200 Hz EEG")
    if value["adapter"].get("pooling") != "frozen_pretrained_cls_query_attention_pooling":
        raise ValueError("REVE alignment audit requires the frozen pretrained query pooler")

    position_names = _official_position_names()
    for task in SUPPORTED_TASKS:
        task_config = value.get("tasks", {}).get(task)
        if not isinstance(task_config, dict) or task_config.get("supported") is not True:
            raise ValueError(f"supported REVE task is missing from config: {task}")
        panel = tuple(str(name) for name in task_config.get("panel", ()))
        if len(panel) != 16 or len(set(panel)) != 16:
            raise ValueError(f"task {task} must declare 16 unique measured channels")
        missing = sorted(set(panel) - position_names)
        if missing:
            raise ValueError(f"task {task} is absent from the REVE position bank: {missing}")
        if not math.isclose(float(task_config["duration_s"]), TASK_SPECS[task].input_duration_s):
            raise ValueError(f"task duration differs from the canonical contract: {task}")
        if not str(task_config.get("track", "")):
            raise ValueError(f"task {task} must declare its pretraining-exposure track")
    for task in ("motor_imagery", "mental_arithmetic"):
        if value["tasks"][task]["track"] != "open_world_pretrained_with_target_corpus_overlap":
            raise ValueError(f"{task} must retain the known Shin2017A overlap track")
    refed = value.get("tasks", {}).get("refed_regression", {})
    if refed.get("supported") is not False or not refed.get("unsupported_reason_code"):
        raise ValueError("REFED must retain an explicit unsupported disposition")
    return value, config_path


class REVEPublicView(CBraModPublicView):
    """Shared canonical EEG view, after REVE panel coverage is verified."""


def load_public_inventory(
    config: Mapping[str, Any], *, task: str
) -> PublicInventory:
    return _shared_load_public_inventory(config, task=task)


def data_branch_fingerprints(config: Mapping[str, Any]) -> dict[str, str]:
    return _shared_data_branch_fingerprints(config)


__all__ = [
    "SUPPORTED_TASKS",
    "PublicInventory",
    "REVEPublicView",
    "RecordGroupedBatchSampler",
    "data_branch_fingerprints",
    "load_config",
    "load_public_inventory",
    "make_loader",
    "resolve_repo_path",
    "sample_id",
    "stable_hash",
]
