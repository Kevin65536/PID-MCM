#!/usr/bin/env python3
"""Gate-aware full trainer for the physiology-semantic tokenizer."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.factory import create_configured_multimodal_dataloaders
from src.losses.physiology_semantic import PhysiologySemanticLoss
from src.teachers.physical_state_teacher import PhysicalStateTeacher
from src.tokenizers.ema_vector_quantizer import EMAVectorQuantizer
from src.tokenizers.registry import create_tokenizer
import src.tokenizers  # noqa: F401  # active registry side effects


RUN_SCHEMA = "physiology_semantic_training_v2"
E0_SCHEMA = "physiology_semantic_e0_v1"
TARGET_FAMILY_GATE_SCHEMA = "physiology_semantic_target_family_gate_v1"
EEG_COORDINATES = ("r_mean", "r_slope", "r_logvar", "s_mean", "s_slope", "s_logvar")
FNIRS_COORDINATES = (
    "delta_f_mean", "delta_hbo_mean", "delta_hb_mean",
    "delta_f_slope", "delta_hbo_slope", "delta_hb_slope",
    "delta_f_logvar", "delta_hbo_logvar", "delta_hb_logvar",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(dict(payload)), sort_keys=True) + "\n")


def _implementation_snapshot(config_path: Path) -> dict[str, Any]:
    relative_paths = (
        "experiments/train_physiology_semantic_tokenizer.py",
        "src/data/factory.py",
        "src/data/physiology_semantic_local.py",
        "src/data/physiology_semantic_targets.py",
        "src/losses/physiology_semantic.py",
        "src/teachers/physical_state_teacher.py",
        "src/tokenizers/ema_vector_quantizer.py",
        "src/tokenizers/physiology_semantic_tokenizer.py",
    )
    return {
        "schema": "physiology_semantic_implementation_snapshot_v1",
        "git_commit": _git_value("rev-parse", "HEAD"),
        "dirty_worktree": bool(_git_value("status", "--porcelain")),
        "files_sha256": {
            relative: _sha256(REPO_ROOT / relative)
            for relative in relative_paths
        },
        "run_config_sha256": _sha256(config_path),
    }


def _quantizer_reference_tests() -> dict[str, Any]:
    """Run deterministic implementation invariants without touching run data."""

    zero_assignment = EMAVectorQuantizer(codebook_size=2, embedding_dim=2, decay=0.99)
    with torch.no_grad():
        zero_assignment.codebook.copy_(torch.tensor([[0.0, 0.0], [20.0, 20.0]]))
    before = zero_assignment.codebook.detach().clone()
    zero_assignment.train()(torch.tensor([[[2.0, -1.0], [2.0, -1.0]]]))

    reference = EMAVectorQuantizer(codebook_size=4, embedding_dim=3)
    latent = torch.tensor(
        [[[0.1, -0.2, 0.3], [0.4, 0.0, -0.1], [-0.3, 0.2, 0.5]]]
    )
    reference.train()(latent)
    reference.eval()
    expected = reference(latent)
    restored = copy.deepcopy(reference)
    observed = restored(latent)

    checks = {
        "zero_assignment_code_unchanged": bool(torch.equal(zero_assignment.codebook[1], before[1])),
        "first_assignment_matches_centroid": bool(
            torch.allclose(zero_assignment.codebook[0], torch.tensor([2.0, -1.0]))
        ),
        "hard_id_equals_posterior_argmax": bool(
            torch.equal(expected.hard_ids, expected.posterior.argmax(dim=-1))
        ),
        "state_round_trip_exact": bool(
            torch.equal(expected.hard_ids, observed.hard_ids)
            and torch.equal(expected.logits, observed.logits)
            and torch.equal(expected.posterior, observed.posterior)
        ),
    }
    return {
        "schema": "physiology_semantic_quantizer_reference_v1",
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def _update_epoch_health(
    aggregate: dict[str, Any],
    modality: str,
    output: Any,
    valid_mask: torch.Tensor | None,
) -> None:
    quantizer = output.quantizer
    hard_ids = quantizer.hard_ids.detach()
    if valid_mask is None:
        selected = hard_ids.reshape(-1)
    else:
        selected = hard_ids[valid_mask.to(device=hard_ids.device, dtype=torch.bool)]
    codebook_size = int(quantizer.posterior.shape[-1])
    counts = torch.bincount(selected, minlength=codebook_size).cpu()
    state = aggregate.setdefault(
        modality,
        {
            "assignment_counts": torch.zeros(codebook_size, dtype=torch.long),
            "valid_tokens": 0,
            "batches": 0,
            "prototype_drift_sum": 0.0,
            "revived_codes_sum": 0.0,
            "snapshot": {},
        },
    )
    state["assignment_counts"] += counts
    state["valid_tokens"] += int(selected.numel())
    state["batches"] += 1
    state["prototype_drift_sum"] += float(quantizer.health["prototype_drift"].detach())
    state["revived_codes_sum"] += float(quantizer.health["revived_codes"].detach())
    state["snapshot"] = {
        key: float(value.detach()) for key, value in quantizer.health.items()
    }


def _finalize_epoch_health(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for modality, state in aggregate.items():
        counts = state["assignment_counts"].float()
        probabilities = counts / counts.sum().clamp_min(1.0)
        nonzero = probabilities > 0
        entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
        batches = max(int(state["batches"]), 1)
        health = dict(state["snapshot"])
        health.update(
            {
                "assignment_entropy": float(entropy),
                "effective_codes": float(entropy.exp()),
                "epoch_active_codes": int((counts > 0).sum()),
                "epoch_active_fraction": float((counts > 0).float().mean()),
                "valid_tokens": int(state["valid_tokens"]),
                "mean_prototype_drift": float(state["prototype_drift_sum"] / batches),
                "revived_codes": int(state["revived_codes_sum"]),
            }
        )
        result[modality] = health
    return result


def _git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_run_dir(config: Mapping[str, Any]) -> Path:
    experiment = config.get("experiment", {})
    group = experiment.get("run_group", "physiology_semantic_tokenizer/training")
    name = experiment.get("name", "tokenizer_training")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "experiments" / "runs" / str(group) / f"{stamp}_{name}"


def _resolve_device(training: Mapping[str, Any]) -> torch.device:
    requested = str(training.get("device", "auto"))
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _dataset_subjects(dataset: Any) -> set[Any]:
    if hasattr(dataset, "subject_keys"):
        return {str(value) for value in dataset.subject_keys}
    if hasattr(dataset, "entries"):
        return {int(entry.subject_id) for entry in dataset.entries}
    if hasattr(dataset, "sources"):
        subjects: set[int] = set()
        for source in dataset.sources:
            subjects.update(_dataset_subjects(source["dataset"]))
        return subjects
    raise TypeError(f"Cannot audit subjects for dataset type {type(dataset).__name__}")


def _validate_loader_subjects(dataloaders: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    declared = config.get("data", {}).get("split", {})
    for split, key in (("train", "train_subjects"), ("val", "val_subjects"), ("test", "test_subjects")):
        subject_key_name = f"{split}_subject_keys"
        if subject_key_name in declared:
            expected = {str(value) for value in declared.get(subject_key_name, [])}
        else:
            expected = {int(value) for value in declared.get(key, [])}
        observed = _dataset_subjects(dataloaders[split].dataset)
        if observed != expected:
            raise RuntimeError(
                f"{split} cache coverage mismatch: expected subjects {sorted(expected)}, observed {sorted(observed)}"
            )


def _load_training_gate(
    config: Mapping[str, Any],
    *,
    require_pass: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    validation = config.get("validation", {})
    gate_value = validation.get("target_family_gate_path") or validation.get("e0_gate_path")
    if not gate_value:
        if require_pass:
            raise RuntimeError(
                "Teacher-supervised training requires validation.target_family_gate_path "
                "or validation.e0_gate_path; boolean pass flags are not accepted"
            )
        return None, None
    path = Path(gate_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"E0 gate file not found: {path}")
    gate = json.loads(path.read_text(encoding="utf-8"))
    schema = gate.get("schema")
    if schema not in {E0_SCHEMA, TARGET_FAMILY_GATE_SCHEMA}:
        raise ValueError(f"Unsupported E0 gate schema in {path}")
    expected_split = hashlib.sha256(
        json.dumps(config.get("data", {}).get("split", {}), sort_keys=True).encode("utf-8")
    ).hexdigest()
    if gate.get("split_sha256") != expected_split:
        raise ValueError("E0 gate subject split does not match the training configuration")
    data_cfg = config.get("data", {})
    if gate.get("data_contract") != data_cfg.get("contract"):
        raise ValueError("E0 gate data contract does not match the training configuration")
    if schema == E0_SCHEMA:
        if gate.get("gate") != "G0":
            raise ValueError(f"Unsupported E0 gate identity in {path}")
        expected_roots = [source.get("root") for source in data_cfg.get("cache_sources", [])]
        if gate.get("cache_source_roots") != expected_roots:
            raise ValueError("E0 gate cache sources do not match the training configuration")
        passed = bool(gate.get("e0_passed", False))
    else:
        if gate.get("gate") != "E0_OPTIONAL_TARGET_FAMILY_DEVELOPMENT":
            raise ValueError(f"Unsupported target-family gate identity in {path}")
        target_cfg = data_cfg.get("auxiliary_target", {}) or {}
        if gate.get("target_family") != target_cfg.get("family"):
            raise ValueError("Target-family gate does not match the configured family")
        if gate.get("target_version") != target_cfg.get("version"):
            raise ValueError("Target-family gate does not match the configured target version")
        if gate.get("cache_root") != data_cfg.get("cache_root"):
            raise ValueError("Target-family gate measured-cache root mismatch")
        sidecar_path = Path(str(target_cfg.get("root", ""))) / "manifest.json"
        if not sidecar_path.is_absolute():
            sidecar_path = REPO_ROOT / sidecar_path
        if not sidecar_path.is_file():
            raise FileNotFoundError(f"Target-family sidecar manifest not found: {sidecar_path}")
        if gate.get("sidecar_manifest_sha256") != _sha256(sidecar_path):
            raise ValueError("Target-family gate sidecar hash mismatch")
        if bool(gate.get("protected_test_opened", False)):
            raise ValueError("Development target-family gate must keep protected test closed")
        if validation.get("promotion_eligible", False):
            raise ValueError("A development-only target-family gate cannot authorize promotion")
        passed = bool(gate.get("target_family_development_passed", False))
    if require_pass and not passed:
        raise RuntimeError(f"Training gate did not pass: {gate.get('status', 'unknown')}")
    return gate, _sha256(path)


# Backward-compatible import for the existing G0 trainer tests and callers.
_load_e0_gate = _load_training_gate


def _coordinate_mask(names: tuple[str, ...], admitted: Iterable[str] | None) -> torch.Tensor:
    if admitted is None:
        return torch.ones(len(names), dtype=torch.bool)
    admitted = set(admitted)
    return torch.tensor([name in admitted for name in names], dtype=torch.bool)


def _teacher_supervision_requested(config: Mapping[str, Any]) -> bool:
    loss = config.get("loss", {})
    return any(
        float(loss.get(name, {}).get("weight", 0.0)) > 0.0
        for name in ("state", "prototype", "masked_state")
    )


def _loss_from_config(config: Mapping[str, Any], gate: Mapping[str, Any] | None) -> PhysiologySemanticLoss:
    loss = config.get("loss", {})
    reconstruction = loss.get("reconstruction", {})
    balance = loss.get("codebook_balance", {})
    admitted = None if gate is None else gate.get("admissible_coordinates", {})
    eeg_admitted = None if admitted is None else admitted.get("eeg", [])
    fnirs_admitted = None if admitted is None else admitted.get("fnirs", [])
    routing = loss.get("entry_routing", {})

    def entry_masks(
        modality: str,
        names: tuple[str, ...],
        fallback: Iterable[str] | None,
    ) -> dict[str, torch.Tensor]:
        masks: dict[str, torch.Tensor] = {}
        gate_by_entry = {} if gate is None else gate.get("admissible_coordinates_by_entry", {})
        modality_gate = gate_by_entry.get(modality, {}) if isinstance(gate_by_entry, Mapping) else {}
        for entry in ("local", "prototype", "context", "coupling"):
            entry_cfg = routing.get(entry, {}) if isinstance(routing, Mapping) else {}
            configured = entry_cfg.get(modality) if isinstance(entry_cfg, Mapping) else None
            admitted_for_entry = modality_gate.get(entry) if isinstance(modality_gate, Mapping) else None
            selected = configured if configured is not None else admitted_for_entry
            if selected is None:
                selected = fallback
            masks[entry] = _coordinate_mask(names, selected)
        return masks

    return PhysiologySemanticLoss(
        state_weight=loss.get("state", {}).get("weight", 1.0),
        prototype_weight=loss.get("prototype", {}).get("weight", 1.0),
        masked_state_weight=loss.get("masked_state", {}).get("weight", 1.0),
        reconstruction_weight=reconstruction.get("weight", 1.0),
        reconstruction_mode=str(reconstruction.get("mode", "combined")),
        reconstruction_semantic_input=str(reconstruction.get("semantic_input", "expected")),
        vq_weight=loss.get("vq", {}).get("weight", 1.0),
        private_weight=loss.get("private", {}).get("weight", 0.0),
        balance_weight=balance.get("weight", 0.0),
        balance_temperature=balance.get("temperature", 1.0),
        eeg_balance_temperature=balance.get("eeg_temperature"),
        fnirs_balance_temperature=balance.get("fnirs_temperature"),
        eeg_balance_scale=balance.get("eeg_scale", 1.0),
        fnirs_balance_scale=balance.get("fnirs_scale", 1.0),
        eeg_coordinate_mask=_coordinate_mask(EEG_COORDINATES, eeg_admitted),
        fnirs_coordinate_mask=_coordinate_mask(FNIRS_COORDINATES, fnirs_admitted),
        eeg_entry_coordinate_masks=entry_masks("eeg", EEG_COORDINATES, eeg_admitted),
        fnirs_entry_coordinate_masks=entry_masks("fnirs", FNIRS_COORDINATES, fnirs_admitted),
        uncertainty_weighting=bool(loss.get("uncertainty_weighting", True)),
    )


def _gradient_contract(objective: str) -> dict[str, tuple[str, ...]] | None:
    modality, _, loss_name = objective.partition("_")
    if modality not in {"eeg", "fnirs"}:
        return None
    branch = f"{modality}_branch."
    other = "fnirs_branch." if modality == "eeg" else "eeg_branch."
    head_by_loss = {
        "state": "state_head.",
        "prototype": "prototype_state_head.",
        "masked_state": "context.",
        "reconstruction": "decoder.",
    }
    required = [branch + "semantic_head."]
    if loss_name in head_by_loss:
        required.append(branch + head_by_loss[loss_name])
    return {"required_prefixes": tuple(required), "forbidden_prefixes": (other,)}


def _gradient_objective_weight(criterion: PhysiologySemanticLoss, objective: str) -> float:
    modality, _, loss_name = objective.partition("_")
    weight = float(criterion.weights.get(loss_name, 0.0))
    if loss_name == "balance":
        scale = criterion.eeg_balance_scale if modality == "eeg" else criterion.fnirs_balance_scale
        return 0.5 * weight * float(scale)
    return weight


def _audit_objective_gradients(
    model: torch.nn.Module,
    losses: Mapping[str, torch.Tensor],
    criterion: PhysiologySemanticLoss,
    config: Mapping[str, Any],
    *,
    global_step: int,
) -> dict[str, Any]:
    """Measure per-objective reachability and pairwise gradient conflict."""

    objectives = tuple(config.get("objectives", (
        "eeg_state", "fnirs_state", "eeg_prototype", "fnirs_prototype",
        "eeg_reconstruction", "fnirs_reconstruction", "eeg_balance", "fnirs_balance",
    )))
    tolerance = float(config.get("tolerance", 1e-12))
    strict = bool(config.get("strict", True))
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    parameter_names = [name for name, _ in named_parameters]
    parameters = [parameter for _, parameter in named_parameters]
    gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
    rows: dict[str, Any] = {}
    all_passed = True
    for objective in objectives:
        if objective not in losses:
            raise KeyError(f"Gradient-audit objective is absent from loss output: {objective}")
        weight = _gradient_objective_weight(criterion, objective)
        raw = losses[objective]
        if weight <= 0.0:
            rows[objective] = {
                "status": "disabled",
                "configured_weight": weight,
                "raw_loss": float(raw.detach()),
                "contract_passed": True,
            }
            continue
        weighted = raw * weight
        grads = torch.autograd.grad(
            weighted,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        gradients[objective] = grads
        norms = {
            name: float(gradient.detach().float().norm())
            for name, gradient in zip(parameter_names, grads)
            if gradient is not None and float(gradient.detach().float().norm()) > tolerance
        }
        contract = _gradient_contract(objective)
        required_missing: list[str] = []
        forbidden_reached: list[str] = []
        if contract is not None and abs(float(raw.detach())) > tolerance:
            required_missing = [
                prefix for prefix in contract["required_prefixes"]
                if not any(name.startswith(prefix) for name in norms)
            ]
            forbidden_reached = [
                name for name in norms
                if any(name.startswith(prefix) for prefix in contract["forbidden_prefixes"])
            ]
        passed = not required_missing and not forbidden_reached
        all_passed &= passed
        rows[objective] = {
            "status": "audited" if abs(float(raw.detach())) > tolerance else "zero_support",
            "configured_weight": weight,
            "raw_loss": float(raw.detach()),
            "gradient_norm": math.sqrt(sum(value * value for value in norms.values())),
            "reachable_parameters": sorted(norms),
            "parameter_gradient_norms": norms,
            "required_prefixes_missing": required_missing,
            "forbidden_parameters_reached": forbidden_reached,
            "contract_passed": passed,
        }

    cosine: dict[str, float | None] = {}
    active = sorted(gradients)
    for left_index, left in enumerate(active):
        for right in active[left_index + 1:]:
            dot = 0.0
            left_sq = 0.0
            right_sq = 0.0
            for left_grad, right_grad in zip(gradients[left], gradients[right]):
                if left_grad is not None:
                    left_sq += float(left_grad.detach().float().square().sum())
                if right_grad is not None:
                    right_sq += float(right_grad.detach().float().square().sum())
                if left_grad is not None and right_grad is not None:
                    dot += float((left_grad.detach().float() * right_grad.detach().float()).sum())
            denominator = math.sqrt(left_sq * right_sq)
            cosine[f"{left}__{right}"] = None if denominator <= tolerance else dot / denominator
    if strict and not all_passed:
        violations = {
            name: row for name, row in rows.items() if not row.get("contract_passed", True)
        }
        raise RuntimeError(f"Gradient-entry contract failed: {violations}")
    return {
        "schema": "physiology_semantic_gradient_entry_audit_v1",
        "global_step": int(global_step),
        "strict": strict,
        "all_contracts_passed": all_passed,
        "objectives": rows,
        "cosine_conflict": cosine,
    }


def _scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    def scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _quantization_strength_for_epoch(
    epoch: int,
    schedule: Mapping[str, Any] | None,
) -> float:
    schedule = dict(schedule or {})
    if not bool(schedule.get("enabled", False)):
        return 1.0
    start_epoch = int(schedule.get("start_epoch", 1))
    ramp_epochs = max(int(schedule.get("ramp_epochs", 1)), 1)
    start_scale = float(schedule.get("start_scale", 0.0))
    end_scale = float(schedule.get("end_scale", 1.0))
    if not 0.0 <= start_scale <= 1.0 or not 0.0 <= end_scale <= 1.0:
        raise ValueError("Quantization warmup scales must be in [0, 1]")
    if epoch < start_epoch:
        return start_scale
    if ramp_epochs == 1:
        return end_scale
    progress = min(max((epoch - start_epoch) / (ramp_epochs - 1), 0.0), 1.0)
    return start_scale + (end_scale - start_scale) * progress


def _amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader,
    teacher_adapter: PhysicalStateTeacher,
    criterion: PhysiologySemanticLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    grad_clip: float,
    global_step: int,
    max_steps: int | None,
    gradient_audit_config: Mapping[str, Any] | None = None,
    gradient_audit_records: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, float], int, dict[str, Any]]:
    training = optimizer is not None
    model.train(training)
    sums: dict[str, float] = {}
    sample_count = 0
    health_aggregate: dict[str, Any] = {}
    for batch in loader:
        if max_steps is not None and global_step >= max_steps:
            break
        batch = _move_to_device(batch, device)
        batch_size = int(batch["eeg"].shape[0])
        teacher = teacher_adapter(batch["teacher"]) if "teacher" in batch else None
        if teacher is None and criterion.requires_teacher:
            raise RuntimeError("Enabled semantic losses require an auxiliary teacher sidecar")
        token_valid_masks = batch.get("token_valid_mask")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), _amp_context(device, amp_enabled):
            outputs = model(
                batch["eeg"], batch["fnirs"], token_valid_masks=token_valid_masks
            )
            losses = criterion(outputs, teacher, token_valid_masks=token_valid_masks)
        if training:
            audit_cfg = dict(gradient_audit_config or {})
            audit_steps = {int(value) for value in audit_cfg.get("steps", (0,))}
            if bool(audit_cfg.get("enabled", False)) and global_step in audit_steps:
                if gradient_audit_records is None:
                    raise ValueError("Enabled gradient audit requires a record sink")
                gradient_audit_records.append(
                    _audit_objective_gradients(
                        model,
                        losses,
                        criterion,
                        audit_cfg,
                        global_step=global_step,
                    )
                )
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach()) * batch_size
        if "has_auxiliary_target" in batch:
            sums["auxiliary_target_coverage"] = sums.get(
                "auxiliary_target_coverage", 0.0
            ) + float(batch["has_auxiliary_target"].float().sum())
        sample_count += batch_size
        for modality in ("eeg", "fnirs"):
            mask = None if token_valid_masks is None else token_valid_masks.get(modality)
            _update_epoch_health(health_aggregate, modality, outputs[modality], mask)
    if sample_count == 0:
        raise RuntimeError("Epoch consumed zero samples")
    return (
        {key: value / sample_count for key, value in sums.items()},
        global_step,
        _finalize_epoch_health(health_aggregate),
    )


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    config: Mapping[str, Any],
    epoch: int,
    global_step: int,
    best_validation: float,
    epochs_without_improvement: int,
    e0_gate_hash: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": RUN_SCHEMA,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "config": dict(config),
            "epoch": epoch,
            "global_step": global_step,
            "best_validation": best_validation,
            "epochs_without_improvement": epochs_without_improvement,
            "e0_gate_sha256": e0_gate_hash,
        },
        path,
    )


def run(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.seed is not None:
        config.setdefault("training", {})["seed"] = args.seed
    if args.device is not None:
        config.setdefault("training", {})["device"] = args.device
    if args.e0_gate:
        config.setdefault("validation", {})["e0_gate_path"] = args.e0_gate
    if args.smoke_optimizer_steps is not None:
        config.setdefault("training", {})["smoke_optimizer_steps"] = args.smoke_optimizer_steps
    training = config.get("training", {})
    seed = int(training.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    optimizer_requested = bool(args.train or (args.smoke and int(training.get("smoke_optimizer_steps", 0)) > 0))
    teacher_supervision = _teacher_supervision_requested(config)
    gate, gate_hash = _load_training_gate(
        config, require_pass=bool(optimizer_requested and teacher_supervision)
    )
    if (
        args.train
        and gate is not None
        and bool(gate.get("requires_e0_channel_aware_revalidation_before_formal_e2", False))
    ):
        raise RuntimeError(
            "Formal E2 training is blocked until the adaptive target is rebuilt and "
            "revalidated under the current measured bad-channel contract"
        )
    device = _resolve_device(training)
    run_dir = Path(args.output_dir).resolve() if args.output_dir else _default_run_dir(config)
    for relative in ("checkpoints", "metrics", "diagnostics"):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _write_json(run_dir / "implementation_snapshot.json", _implementation_snapshot(config_path))

    validation = config.get("validation", {})
    protocol = {
        "schema": RUN_SCHEMA,
        "phase": validation.get("phase"),
        "primary_endpoint": validation.get("primary_endpoint", "validation_total_loss"),
        "selection_metric": validation.get("selection_metric", "validation total loss"),
        "stopping_rule": "validation early stopping with configured patience",
        "protected_test_policy": "test split is never evaluated by the trainer",
        "promotion_eligible": bool(validation.get("promotion_eligible", False)),
        "registered_factors": validation.get("registered_factors"),
        "e0_gate_sha256": gate_hash,
        "objective": "teacher_supervised" if teacher_supervision else "teacher_free",
    }
    (run_dir / "decision_protocol.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    _write_json(run_dir / "metric_registry.json", {
        "primary": validation.get("primary_endpoint", "validation_total_loss"),
        "training": list(config.get("loss", {})),
        "diagnostic": ["quantizer_health", "learning_rate"],
    })
    _write_json(run_dir / "evidence_calibration.json", {
        "source": "E0 gate" if teacher_supervision else "training-only quantizer pilot",
        "e0_gate_sha256": gate_hash,
        "quantizer_health_calibration": validation.get("quantizer_health_calibration"),
        "protected_test_opened": False,
    })
    reference_tests = _quantizer_reference_tests()
    _write_json(run_dir / "diagnostics" / "quantizer_reference_tests.json", reference_tests)
    if not reference_tests["all_passed"]:
        raise RuntimeError("Deterministic quantizer reference tests failed")
    _write_json(run_dir / "environment.json", {
        "schema": "physiology_semantic_environment_v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "config_sha256": _sha256(config_path),
    })

    dataloaders = create_configured_multimodal_dataloaders(config)
    _validate_loader_subjects(dataloaders, config)
    model = create_tokenizer(config).to(device)
    target_cfg = config.get("data", {}).get("auxiliary_target", {})
    teacher_adapter = PhysicalStateTeacher(
        target_family=str(target_cfg.get("family", "croce_physical_state")),
        target_version=str(target_cfg.get("version", "physiology_semantic_v2")),
    ).to(device)
    criterion = _loss_from_config(config, gate).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("lr", 1e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        betas=tuple(training.get("betas", [0.9, 0.98])),
    )

    epochs = int(training.get("epochs", 1))
    steps_per_epoch = max(len(dataloaders["train"]), 1)
    total_steps = max(epochs * steps_per_epoch, 1)
    smoke_steps = int(training.get("smoke_optimizer_steps", 0))
    max_steps = smoke_steps if args.smoke and smoke_steps > 0 else None
    if max_steps is not None:
        total_steps = max_steps
        epochs = max(1, math.ceil(max_steps / steps_per_epoch))
    scheduler = _scheduler(optimizer, int(training.get("warmup_steps", 0)), total_steps)
    amp_enabled = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 0
    global_step = 0
    best_validation = float("inf")
    epochs_without_improvement = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("schema") != RUN_SCHEMA:
            raise ValueError("Resume checkpoint schema mismatch")
        if checkpoint.get("e0_gate_sha256", "") != (gate_hash or ""):
            raise ValueError("Resume checkpoint E0 gate differs from current gate")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_validation = float(checkpoint["best_validation"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        if max_steps is not None and global_step < max_steps and start_epoch >= epochs:
            epochs = start_epoch + 1

    start_time = time.time()
    status = "dry_run_passed"
    last_health: dict[str, Any] = {}
    gradient_audit_records: list[dict[str, Any]] = []
    gradient_audit_config = validation.get("gradient_audit", {})
    quantization_warmup = training.get("quantization_warmup", {})
    if args.dry_run or (args.smoke and not optimizer_requested):
        quantization_strength = _quantization_strength_for_epoch(
            start_epoch, quantization_warmup
        )
        model.set_quantization_strength(quantization_strength)
        train_metrics, _, last_health = _run_epoch(
            model=model,
            loader=[next(iter(dataloaders["train"]))],
            teacher_adapter=teacher_adapter,
            criterion=criterion,
            device=device,
            optimizer=None,
            scheduler=scheduler,
            scaler=scaler,
            amp_enabled=amp_enabled,
            grad_clip=0.0,
            global_step=0,
            max_steps=None,
            gradient_audit_config=None,
            gradient_audit_records=None,
        )
        _append_jsonl(run_dir / "metrics" / "train.jsonl", {
            "epoch": 0,
            "quantization_strength": quantization_strength,
            **train_metrics,
        })
        _append_jsonl(run_dir / "diagnostics" / "quantizer_health.jsonl", {
            "epoch": 0, "phase": "dry_run", "health": last_health,
        })
        status = "dry_run_passed" if args.dry_run else "smoke_passed_optimizer_not_requested"
    else:
        patience = int(training.get("early_stopping_patience", 10))
        min_delta = float(training.get("early_stopping_min_delta", 0.0))
        grad_clip = float(training.get("grad_clip_norm", 1.0))
        for epoch in range(start_epoch, epochs):
            quantization_strength = _quantization_strength_for_epoch(
                epoch, quantization_warmup
            )
            model.set_quantization_strength(quantization_strength)
            train_metrics, global_step, train_health = _run_epoch(
                model=model,
                loader=dataloaders["train"],
                teacher_adapter=teacher_adapter,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                amp_enabled=amp_enabled,
                grad_clip=grad_clip,
                global_step=global_step,
                max_steps=max_steps,
                gradient_audit_config=gradient_audit_config,
                gradient_audit_records=gradient_audit_records,
            )
            validation_metrics, _, validation_health = _run_epoch(
                model=model,
                loader=dataloaders["val"],
                teacher_adapter=teacher_adapter,
                criterion=criterion,
                device=device,
                optimizer=None,
                scheduler=scheduler,
                scaler=scaler,
                amp_enabled=amp_enabled,
                grad_clip=0.0,
                global_step=global_step,
                max_steps=None,
                gradient_audit_config=None,
                gradient_audit_records=None,
            )
            last_health = validation_health
            _append_jsonl(run_dir / "diagnostics" / "quantizer_health.jsonl", {
                "epoch": epoch,
                "global_step": global_step,
                "train": train_health,
                "validation": validation_health,
            })
            learning_rate = optimizer.param_groups[0]["lr"]
            _append_jsonl(run_dir / "metrics" / "train.jsonl", {
                "epoch": epoch,
                "global_step": global_step,
                "learning_rate": learning_rate,
                "quantization_strength": quantization_strength,
                **train_metrics,
            })
            _append_jsonl(run_dir / "metrics" / "validation.jsonl", {
                "epoch": epoch,
                "global_step": global_step,
                "quantization_strength": quantization_strength,
                **validation_metrics,
            })

            improved = validation_metrics["total"] < best_validation - min_delta
            if improved:
                best_validation = validation_metrics["total"]
                epochs_without_improvement = 0
                _save_checkpoint(
                    run_dir / "checkpoints" / "best.pt",
                    model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    config=config, epoch=epoch, global_step=global_step,
                    best_validation=best_validation,
                    epochs_without_improvement=epochs_without_improvement,
                    e0_gate_hash=gate_hash or "",
                )
            else:
                epochs_without_improvement += 1
            _save_checkpoint(
                run_dir / "checkpoints" / "last.pt",
                model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                config=config, epoch=epoch, global_step=global_step,
                best_validation=best_validation,
                epochs_without_improvement=epochs_without_improvement,
                e0_gate_hash=gate_hash or "",
            )
            if max_steps is not None and global_step >= max_steps:
                break
            if epochs_without_improvement >= patience:
                status = "early_stopped"
                break
        if status != "early_stopped":
            status = "smoke_passed" if args.smoke else "training_complete"

    _write_json(run_dir / "diagnostics" / "quantizer_health.json", last_health)
    _write_json(run_dir / "diagnostics" / "gradient_entry_audit.json", {
        "schema": "physiology_semantic_gradient_entry_audit_collection_v1",
        "enabled": bool(gradient_audit_config.get("enabled", False)),
        "record_count": len(gradient_audit_records),
        "all_contracts_passed": all(
            bool(record.get("all_contracts_passed", False))
            for record in gradient_audit_records
        ) if gradient_audit_records else None,
        "records": gradient_audit_records,
    })
    checkpoint_hashes = {
        path.name: _sha256(path) for path in (run_dir / "checkpoints").glob("*.pt")
    }
    split_hash = hashlib.sha256(
        json.dumps(config.get("data", {}).get("split", {}), sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema": RUN_SCHEMA,
        "status": status,
        "mode": "train" if args.train else ("smoke" if args.smoke else "dry_run"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "dirty_worktree": bool(_git_value("status", "--porcelain")),
        "command": " ".join(sys.argv),
        "seed": seed,
        "device": str(device),
        "e0_gate_sha256": gate_hash,
        "training_gate_schema": None if gate is None else gate.get("schema"),
        "objective": "teacher_supervised" if teacher_supervision else "teacher_free",
        "global_step": global_step,
        "best_validation": None if math.isinf(best_validation) else best_validation,
        "split_sha256": split_hash,
        "loader_class": config.get("data", {}).get("loader_class"),
        "data_contract": config.get("data", {}).get("contract"),
        "dataset_ids": config.get("data", {}).get("dataset_ids"),
        "task_namespaces": config.get("data", {}).get("task_namespaces"),
        "cache_root": config.get("data", {}).get("cache_root"),
        "phase": validation.get("phase"),
        "promotion_eligible": bool(validation.get("promotion_eligible", False)),
        "reconstruction_mode": criterion.reconstruction_mode,
        "reconstruction_semantic_input": criterion.reconstruction_semantic_input,
        "codebook_balance_weight": criterion.weights["balance"],
        "quantization_warmup": quantization_warmup,
        "final_quantization_strength": model.get_quantization_strength(),
        "checkpoint_sha256": checkpoint_hashes,
        "protected_test_opened": False,
        "start_time": datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(run_dir / "manifest.json", manifest)
    summary_lines = [
        "# Physiology-semantic tokenizer run summary",
        "",
        f"- Status: `{status}`",
        f"- Objective: `{'teacher_supervised' if teacher_supervision else 'teacher_free'}`",
        f"- Device: `{device}`",
        f"- Global optimizer steps: `{global_step}`",
        f"- Best validation total loss: `{manifest['best_validation']}`",
        f"- Protected test opened: `False`",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "status": status, "global_step": global_step}, sort_keys=True))
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--train", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--e0-gate", help="Override validation.e0_gate_path with a concrete gate_decision.json")
    parser.add_argument("--smoke-optimizer-steps", type=int, help="Override smoke step budget")
    parser.add_argument("--seed", type=int, help="Override the registered training seed")
    parser.add_argument("--device", help="Override the registered training device")
    parser.add_argument("--output-dir")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
