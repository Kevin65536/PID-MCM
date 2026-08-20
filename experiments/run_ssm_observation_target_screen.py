#!/usr/bin/env python3
"""Run the no-VQ modality-specific SSM observation target screen."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from torch.utils.data import DataLoader, Dataset
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import optimize_lag_conditioned_spvq_architecture as optimized
from experiments import run_lag_conditioned_spvq as reviewed
from src.data.ssm_observation_targets import (
    ObservationChannelSelection,
    ObservationFeatureBatch,
    SSMObservationTeacherBatch,
    apply_ssm_observation_teachers,
    extract_eeg_spatial_band_trajectory,
    extract_fnirs_patch_trajectory,
    fit_observation_channel_selection,
    fit_ssm_observation_teachers,
)
from src.losses.ssm_observation import ssm_observation_objective
from src.metrics.lag_conditioned_downstream import (
    subject_equal_classification_metrics,
)
from src.tokenizers.ssm_observation_shared_private import (
    SSMObservationSharedPrivateModel,
)


SCHEMA = "ssm_observation_target_screen_v1"
REGISTERED_MODES = (
    "NATIVE",
    "SSM-SELF",
    "SSM-JOINT",
    "SSM-SELF-XPRED-0.02",
    "SSM-SELF-XPRED-0.05",
)


def _teacher_mode(mode: str) -> str:
    value = str(mode).upper()
    return "SSM-SELF-XPRED" if value.startswith("SSM-SELF-XPRED-") else value


def _mode_xpred_weight(mode: str) -> float:
    value = str(mode).upper()
    return float(value.rsplit("-", 1)[-1]) if value.startswith("SSM-SELF-XPRED-") else 0.0


def validate_config(config: Mapping[str, Any]) -> None:
    """Fail closed on leakage, direct alignment, VQ, or claim-scope drift."""

    if config.get("experiment", {}).get("schema") != "ssm_observation_target_screen_config_v1":
        raise ValueError("unexpected SSM observation screen config schema")
    source = config["source"]
    if source.get("task_labels_enter_ssm") is not False:
        raise PermissionError("task labels must not enter SSM fitting")
    if source.get("condition_specific_ssm_parameters") is not False:
        raise PermissionError("condition-specific SSM parameters are label leakage")
    modes = tuple(map(str, config["teachers"]["modes"]))
    if modes != REGISTERED_MODES:
        raise ValueError("teacher screen must retain the registered four-mode matrix")
    if config["teachers"].get("joint_role") != "privileged_upper_bound_only":
        raise ValueError("joint teacher must remain a privileged upper bound")
    if int(config["teachers"].get("eeg_target_channel_count", 0)) != 6:
        raise ValueError("EEG teacher must retain six spatial channels")
    if int(config["teachers"].get("fnirs_target_channel_count", 0)) != 2:
        raise ValueError("fNIRS teacher must retain one HbO/HbR pair")
    objective = config["objective"]
    if objective.get("vector_quantization") is not False:
        raise ValueError("continuous target screen must keep VQ disabled")
    if objective.get("direct_latent_alignment") is not False:
        raise ValueError("direct cross-modal latent alignment is forbidden")
    if objective.get("bidirectional_matching") is not False:
        raise ValueError("bidirectional cross-modal matching is forbidden")
    xpred_weights = tuple(map(float, objective["xpred_weight_candidates"]))
    if xpred_weights != (0.02, 0.05):
        raise ValueError("asymmetric XPRED weights must remain fixed at 0.02 and 0.05")
    if tuple(_mode_xpred_weight(mode) for mode in modes[-2:]) != xpred_weights:
        raise ValueError("XPRED mode labels and objective weights differ")
    if config["statistics"].get("coupling_endpoint_claim") is not False:
        raise ValueError("continuous screen cannot claim the coupling endpoint")
    if config["statistics"].get("q0_q1_deferred_until_vq") is not True:
        raise ValueError("q0/q1 must remain deferred until the VQ gate")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = list(rows)
    fields: list[str] = []
    for row in values:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: _jsonable(row.get(key, "")) for key in fields} for row in values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False


def _partition_features(
    partition: Any,
    selection: ObservationChannelSelection,
) -> tuple[ObservationFeatureBatch, ObservationFeatureBatch]:
    eeg_indices = np.asarray(selection.eeg_indices, dtype=np.int64)
    fnirs_indices = np.asarray(selection.fnirs_indices, dtype=np.int64)
    eeg = extract_eeg_spatial_band_trajectory(
        partition.eeg[:, eeg_indices],
        token_valid_mask=partition.eeg_token_mask,
        point_valid_mask=partition.eeg_point_mask,
        channel_valid_mask=partition.eeg_channel_mask[:, eeg_indices],
        channel_names=selection.eeg_channel_names,
        sampling_rate_hz=200.0,
    )
    fnirs = extract_fnirs_patch_trajectory(
        partition.fnirs[:, fnirs_indices],
        token_valid_mask=partition.fnirs_token_mask,
        point_valid_mask=partition.fnirs_point_mask,
        channel_valid_mask=partition.fnirs_channel_mask[:, fnirs_indices],
        channel_names=selection.fnirs_channel_names,
    )
    return eeg, fnirs


@dataclass(frozen=True)
class ScreenPartition:
    prepared: Any
    teacher: SSMObservationTeacherBatch
    mode: str


class ScreenDataset(Dataset):
    def __init__(self, partition: ScreenPartition) -> None:
        self.partition = partition
        self.targets = partition.teacher.targets(_teacher_mode(partition.mode))

    def __len__(self) -> int:
        return len(self.partition.prepared.sample_id)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        source = self.partition.prepared
        output: dict[str, Any] = {
            "index": torch.tensor(index, dtype=torch.long),
            "eeg": torch.from_numpy(source.eeg[index]),
            "fnirs": torch.from_numpy(source.fnirs[index]),
            "eeg_token_valid_mask": torch.from_numpy(source.eeg_token_mask[index]),
            "fnirs_token_valid_mask": torch.from_numpy(source.fnirs_token_mask[index]),
            "target": torch.tensor(int(source.target[index]), dtype=torch.long),
            "sample_id": str(source.sample_id[index]),
            "subject": str(source.subject[index]),
            "condition": str(source.condition[index]),
        }
        for name, value in self.targets.items():
            output[name] = torch.from_numpy(np.asarray(value[index]))
        return output


def _loader(
    partition: ScreenPartition,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        ScreenDataset(partition),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        generator=generator,
        drop_last=False,
    )


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _objective_kwargs(config: Mapping[str, Any], mode: str) -> Mapping[str, float]:
    objective = config["objective"]
    return {
        "clean_weight": float(objective["clean_weight"]),
        "residual_weight": float(objective["residual_weight"]),
        "cross_prediction_weight": _mode_xpred_weight(mode),
        "delta": float(objective["huber_delta"]),
        "epsilon": float(objective["uncertainty_epsilon"]),
        "weight_min": float(objective["uncertainty_weight_min"]),
        "weight_max": float(objective["uncertainty_weight_max"]),
    }


def _model(
    partition: ScreenPartition,
    config: Mapping[str, Any],
) -> SSMObservationSharedPrivateModel:
    model = config["model"]
    source = partition.prepared
    targets = partition.teacher.targets(_teacher_mode(partition.mode))
    return SSMObservationSharedPrivateModel(
        eeg_channels=source.eeg.shape[1],
        fnirs_channels=source.fnirs.shape[1],
        eeg_patch_samples=source.eeg.shape[2] // source.eeg_token_mask.shape[1],
        fnirs_patch_samples=source.fnirs.shape[2] // source.fnirs_token_mask.shape[1],
        num_tokens=source.eeg_token_mask.shape[1],
        eeg_target_dim=targets["eeg_clean_target"].shape[-1],
        fnirs_target_dim=targets["fnirs_clean_target"].shape[-1],
        shared_dim=int(model["shared_dim"]),
        eeg_private_dim=int(model["eeg_private_dim"]),
        fnirs_private_dim=int(model["fnirs_private_dim"]),
        eeg_shared_history_tokens=int(model["eeg_shared_history_tokens"]),
        fnirs_shared_history_tokens=int(model["fnirs_shared_history_tokens"]),
        encoder_depth=int(model["encoder_depth"]),
        encoder_num_heads=int(model["encoder_num_heads"]),
        encoder_feedforward_dim=int(model["encoder_feedforward_dim"]),
        decoder_hidden_dim=int(model["decoder_hidden_dim"]),
        dropout=float(model["dropout"]),
        class_count=len(reviewed.TASK_SPECS[source.role_task_id].class_names)
        if hasattr(source, "role_task_id")
        else 2,
        interaction_rank=int(model["interaction_rank"]),
        allowed_lags=tuple(map(int, model["allowed_lags"])),
        interaction_weight=float(model["interaction_weight"]),
        cross_prediction_lags=(
            tuple(map(int, model["cross_prediction_lags"]))
            if _teacher_mode(partition.mode) == "SSM-SELF-XPRED"
            else None
        ),
    )


def _forward(model: Any, batch: Mapping[str, Any]) -> Mapping[str, Any]:
    return model(
        batch["eeg"],
        batch["fnirs"],
        batch["eeg_token_valid_mask"],
        batch["fnirs_token_valid_mask"],
    )


def _condition_time_means(
    partition: ScreenPartition,
) -> Mapping[str, Mapping[str, np.ndarray]]:
    targets = partition.teacher.targets(_teacher_mode(partition.mode))
    conditions = np.asarray(partition.prepared.condition).astype(str)
    output: dict[str, dict[str, np.ndarray]] = {"eeg": {}, "fnirs": {}}
    for modality in ("eeg", "fnirs"):
        values = targets[f"{modality}_clean_target"]
        mask = targets[f"{modality}_target_valid_mask"]
        for condition in sorted(set(conditions.tolist())):
            selected = conditions == condition
            count = mask[selected].sum(axis=0)
            mean = np.where(mask[selected], values[selected], 0.0).sum(axis=0) / np.maximum(count, 1)
            output[modality][condition] = mean.astype(np.float32)
    return output


def _evaluate(
    model: SSMObservationSharedPrivateModel,
    partition: ScreenPartition,
    *,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    task_id: str,
    baseline: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[Mapping[str, Any], Mapping[str, np.ndarray]]:
    loader = _loader(
        partition,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        seed=seed,
        num_workers=int(config["training"]["num_workers"]),
    )
    collected: dict[str, list[np.ndarray]] = {
        "eeg_clean_prediction": [],
        "fnirs_clean_prediction": [],
        "interaction_only_logits": [],
        "private_only_logits": [],
        "private_plus_shared_marginal_logits": [],
        "private_shared_interaction_logits": [],
    }
    targets: list[np.ndarray] = []
    subjects: list[str] = []
    conditions: list[str] = []
    sample_ids: list[str] = []
    losses = []
    model.eval()
    with torch.no_grad():
        for raw in loader:
            batch = _to_device(raw, device)
            output = _forward(model, batch)
            total, _ = ssm_observation_objective(
                output, batch, **_objective_kwargs(config, partition.mode)
            )
            losses.append(float(total.cpu()) * len(batch["target"]))
            for name in collected:
                collected[name].append(output[name].detach().cpu().numpy())
            targets.append(batch["target"].cpu().numpy())
            subjects.extend(map(str, raw["subject"]))
            conditions.extend(map(str, raw["condition"]))
            sample_ids.extend(map(str, raw["sample_id"]))
    arrays = {name: np.concatenate(values) for name, values in collected.items()}
    arrays.update(
        {
            "target": np.concatenate(targets),
            "subject": np.asarray(subjects, dtype=str),
            "condition": np.asarray(conditions, dtype=str),
            "sample_id": np.asarray(sample_ids, dtype=str),
        }
    )
    teacher_targets = partition.teacher.targets(_teacher_mode(partition.mode))
    metrics: dict[str, Any] = {
        "objective": float(sum(losses) / len(arrays["target"])),
    }
    for modality in ("eeg", "fnirs"):
        truth = teacher_targets[f"{modality}_clean_target"].astype(np.float64)
        prediction = arrays[f"{modality}_clean_prediction"].astype(np.float64)
        mask = teacher_targets[f"{modality}_target_valid_mask"].astype(bool)
        baseline_prediction = np.stack(
            [baseline[modality][condition] for condition in arrays["condition"]]
        ).astype(np.float64)
        model_sse = float(np.square(truth - prediction)[mask].sum())
        baseline_sse = float(np.square(truth - baseline_prediction)[mask].sum())
        metrics[f"{modality}_clean_mse"] = model_sse / int(mask.sum())
        metrics[f"{modality}_clean_delta_r2_vs_condition_time_mean"] = (
            1.0 - model_sse / baseline_sse if baseline_sse > 0.0 else float("nan")
        )
    class_names = reviewed.TASK_SPECS[task_id].class_names
    for head in (
        "private_only",
        "private_plus_shared_marginal",
        "private_shared_interaction",
    ):
        logit_name = f"{head}_logits"
        head_metrics = subject_equal_classification_metrics(
            arrays["target"],
            np.argmax(arrays[logit_name], axis=1),
            arrays["subject"],
            class_names=class_names,
        )
        metrics[f"{head}_subject_equal_macro_f1"] = head_metrics[
            "subject_equal_macro_f1"
        ]
        metrics[f"{head}_subject_equal_balanced_accuracy"] = head_metrics[
            "subject_equal_balanced_accuracy"
        ]
    metrics["interaction_macro_f1_increment"] = (
        metrics["private_shared_interaction_subject_equal_macro_f1"]
        - metrics["private_plus_shared_marginal_subject_equal_macro_f1"]
    )
    metrics["interaction_logit_variance"] = float(
        np.var(arrays["interaction_only_logits"])
    )
    return metrics, arrays


def _provenance_control(
    parameter: ScreenPartition,
    selection: ScreenPartition,
    *,
    task_id: str,
    seed: int,
) -> Mapping[str, Any]:
    def features(value: ScreenPartition) -> np.ndarray:
        target = value.teacher.targets(_teacher_mode(value.mode))
        rows = []
        for modality in ("eeg", "fnirs"):
            std = target[f"{modality}_predictive_std"]
            mask = target[f"{modality}_target_valid_mask"].astype(bool)
            admitted = np.where(mask, std, np.nan)
            rows.extend(
                (
                    np.nanmean(admitted, axis=(1, 2)),
                    np.nanstd(admitted, axis=(1, 2)),
                    1.0 - mask.mean(axis=(1, 2)),
                )
            )
        return np.stack(rows, axis=1)

    train_x = features(parameter)
    test_x = features(selection)
    classifier = LogisticRegression(
        C=1.0,
        max_iter=5000,
        random_state=int(seed),
    ).fit(train_x, parameter.prepared.target)
    prediction = classifier.predict(test_x)
    metrics = subject_equal_classification_metrics(
        selection.prepared.target,
        prediction,
        selection.prepared.subject,
        class_names=reviewed.TASK_SPECS[task_id].class_names,
    )
    return {
        "feature_set": "teacher_uncertainty_missingness_and_constant_provenance_only",
        "subject_equal_macro_f1": metrics["subject_equal_macro_f1"],
        "subject_equal_balanced_accuracy": metrics["subject_equal_balanced_accuracy"],
        "fit_scope": "fit_parameter_only",
    }


def _train_cell(
    task_id: str,
    mode: str,
    seed: int,
    parameter: ScreenPartition,
    selection: ScreenPartition,
    config: Mapping[str, Any],
    *,
    device: torch.device,
    smoke: bool,
    output: Path,
) -> Mapping[str, Any]:
    _set_seed(seed)
    model = _model(parameter, config).to(device)
    # The representation stage is label-free and excludes all task heads.
    assert model.task_head is not None
    for value in model.task_head.parameters():
        value.requires_grad_(False)
    representation_parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        representation_parameters,
        lr=float(config["training"]["learning_rate"]),
        betas=tuple(map(float, config["training"]["betas"])),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    parameter_loader = _loader(
        parameter,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        seed=seed,
        num_workers=int(config["training"]["num_workers"]),
    )
    selection_loader = _loader(
        selection,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        seed=seed,
        num_workers=int(config["training"]["num_workers"]),
    )
    steps = int(
        config["smoke"]["representation_optimizer_steps"]
        if smoke
        else config["training"]["representation_optimizer_steps"]
    )
    interval = min(int(config["training"]["representation_evaluation_interval"]), steps)
    iterator = iter(parameter_loader)
    best_loss = float("inf")
    best_state = None
    history = []
    for step in range(1, steps + 1):
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(parameter_loader)
            raw = next(iterator)
        batch = _to_device(raw, device)
        optimizer.zero_grad(set_to_none=True)
        output_values = _forward(model, batch)
        loss, components = ssm_observation_objective(
            output_values, batch, **_objective_kwargs(config, mode)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            representation_parameters, float(config["training"]["grad_clip_norm"])
        )
        optimizer.step()
        if step % interval == 0 or step == steps:
            model.eval()
            total = 0.0
            clean_selection_total = 0.0
            count = 0
            with torch.no_grad():
                for selection_raw in selection_loader:
                    selection_batch = _to_device(selection_raw, device)
                    selection_output = _forward(model, selection_batch)
                    selection_loss, selection_components = ssm_observation_objective(
                        selection_output,
                        selection_batch,
                        **_objective_kwargs(config, mode),
                    )
                    batch_count = len(selection_batch["target"])
                    total += float(selection_loss.cpu()) * batch_count
                    clean_selection_total += 0.5 * float(
                        (
                            selection_components["eeg_clean"]
                            + selection_components["fnirs_clean"]
                        ).cpu()
                    ) * batch_count
                    count += batch_count
            selection_loss_value = total / count
            selection_score = clean_selection_total / count
            history.append(
                {
                    "stage": "representation",
                    "step": step,
                    "train_objective": float(loss.detach().cpu()),
                    "selection_objective": selection_loss_value,
                    "selection_equal_modality_clean_score": selection_score,
                }
            )
            if selection_score < best_loss:
                best_loss = selection_score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            model.train()
    if best_state is None:
        raise RuntimeError("representation stage produced no fit-selection checkpoint")
    model.load_state_dict(best_state, strict=True)

    # Freeze the selected representation; train only decomposed task heads.
    for value in model.parameters():
        value.requires_grad_(False)
    for value in model.task_head.parameters():
        value.requires_grad_(True)
    head_parameters = list(model.task_head.parameters())
    head_optimizer = torch.optim.AdamW(
        head_parameters,
        lr=float(config["training"]["head_learning_rate"]),
        betas=tuple(map(float, config["training"]["betas"])),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    head_steps = int(
        config["smoke"]["head_optimizer_steps"]
        if smoke
        else config["training"]["head_optimizer_steps"]
    )
    iterator = iter(parameter_loader)
    model.eval()
    model.task_head.train()
    for step in range(1, head_steps + 1):
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(parameter_loader)
            raw = next(iterator)
        batch = _to_device(raw, device)
        head_optimizer.zero_grad(set_to_none=True)
        output_values = _forward(model, batch)
        head_loss = torch.nn.functional.cross_entropy(
            output_values["private_shared_interaction_logits"], batch["target"]
        ) + 0.1 * (
            torch.nn.functional.cross_entropy(
                output_values["private_only_logits"], batch["target"]
            )
            + torch.nn.functional.cross_entropy(
                output_values["private_plus_shared_marginal_logits"], batch["target"]
            )
        )
        head_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            head_parameters, float(config["training"]["grad_clip_norm"])
        )
        head_optimizer.step()
        if step == head_steps:
            history.append(
                {
                    "stage": "task_head",
                    "step": step,
                    "train_objective": float(head_loss.detach().cpu()),
                    "selection_objective": "",
                }
            )
    baseline = _condition_time_means(parameter)
    parameter_metrics, parameter_arrays = _evaluate(
        model,
        parameter,
        config=config,
        device=device,
        seed=seed,
        task_id=task_id,
        baseline=baseline,
    )
    selection_metrics, selection_arrays = _evaluate(
        model,
        selection,
        config=config,
        device=device,
        seed=seed,
        task_id=task_id,
        baseline=baseline,
    )
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "ssm_observation_target_screen_checkpoint_v1",
            "task_id": task_id,
            "mode": mode,
            "seed": seed,
            "xpred_weight": _mode_xpred_weight(mode),
            "representation_selection_score": best_loss,
            "model_state": model.state_dict(),
            "teacher_provenance_id": parameter.teacher.provenance_id,
            "protected_open": False,
        },
        output / "checkpoint.pt",
    )
    _write_csv(output / "history.csv", history)
    np.savez_compressed(
        output / "fit_selection_outputs.npz",
        schema=np.asarray(SCHEMA),
        task_id=np.asarray(task_id),
        mode=np.asarray(mode),
        seed=np.asarray(seed),
        **selection_arrays,
    )
    result = {
        "schema": SCHEMA,
        "task_id": task_id,
        "mode": mode,
        "seed": seed,
        "representation_selection_score": best_loss,
        "fit_parameter": parameter_metrics,
        "fit_selection": selection_metrics,
        "teacher_provenance_id": parameter.teacher.provenance_id,
        "task_labels_entered_teacher": False,
        "vector_quantization": False,
        "xpred_weight": _mode_xpred_weight(mode),
        "protected_open": False,
    }
    _write_json(output / "result.json", result)
    return result


def run(
    config_path: Path,
    output: Path,
    *,
    smoke: bool,
    task_filter: Sequence[str] | None = None,
    mode_filter: Sequence[str] | None = None,
    seed_override: Sequence[int] | None = None,
    representation_steps_override: int | None = None,
    head_steps_override: int | None = None,
) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    config = deepcopy(config)
    if representation_steps_override is not None:
        if int(representation_steps_override) <= 0:
            raise ValueError("representation steps override must be positive")
        config["training"]["representation_optimizer_steps"] = int(
            representation_steps_override
        )
    if head_steps_override is not None:
        if int(head_steps_override) <= 0:
            raise ValueError("head steps override must be positive")
        config["training"]["head_optimizer_steps"] = int(head_steps_override)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    base_path = REPO_ROOT / config["source"]["base_lc_spvq_config"]
    optimization_path = REPO_ROOT / config["source"]["optimization_config"]
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    optimization_config = yaml.safe_load(optimization_path.read_text(encoding="utf-8"))
    device = torch.device(config["training"]["device"])
    seeds = tuple(
        map(
            int,
            seed_override
            if seed_override is not None
            else (
                config["smoke"]["seeds"]
                if smoke
                else config["training"]["seeds"]
            ),
        )
    )
    tasks = tuple(
        map(
            str,
            task_filter
            if task_filter is not None
            else config["source"]["tasks"],
        )
    )
    modes = tuple(
        map(
            str,
            mode_filter
            if mode_filter is not None
            else config["teachers"]["modes"],
        )
    )
    if not seeds or not tasks or not modes:
        raise ValueError("execution filters must retain seeds, tasks, and modes")
    unknown_tasks = set(tasks).difference(map(str, config["source"]["tasks"]))
    unknown_modes = set(modes).difference(REGISTERED_MODES)
    if unknown_tasks or unknown_modes:
        raise ValueError(
            f"execution filter contains unknown tasks/modes: {unknown_tasks}/{unknown_modes}"
        )
    result_rows = []
    leakage_rows = []
    teacher_rows = []
    try:
        shutil.copy2(config_path, staging / "config.yaml")
        for task_id in tasks:
            prepared = optimized.prepare_fit_selection_task(
                optimization_config,
                str(task_id),
                base_config=base_config,
                derangement_seed=int(optimization_config["execution"]["derangement_seed"]),
            )
            # Smoke mode deliberately retains the governed fit partitions and
            # shortens optimizer steps only; teacher fit scope therefore stays
            # identical to the longer screen.
            channel_selection = fit_observation_channel_selection(
                prepared.parameter.eeg,
                prepared.parameter.fnirs,
                eeg_channel_valid_mask=prepared.parameter.eeg_channel_mask,
                fnirs_channel_valid_mask=prepared.parameter.fnirs_channel_mask,
                eeg_channel_names=prepared.parameter.eeg_channel_names,
                fnirs_channel_names=prepared.parameter.fnirs_channel_names,
                fnirs_component_roles=prepared.parameter.fnirs_component_roles,
                eeg_channel_count=int(config["teachers"]["eeg_target_channel_count"]),
            )
            parameter_eeg, parameter_fnirs = _partition_features(
                prepared.parameter, channel_selection
            )
            selection_eeg, selection_fnirs = _partition_features(
                prepared.selection, channel_selection
            )
            fits = fit_ssm_observation_teachers(
                parameter_eeg,
                parameter_fnirs,
                ridge=float(config["teachers"]["ssm_ridge"]),
                max_spectral_radius=float(
                    config["teachers"]["max_spectral_radius"]
                ),
                fit_scope=str(config["source"]["fit_scope"]),
            )
            parameter_teacher = apply_ssm_observation_teachers(
                parameter_eeg, parameter_fnirs, fits
            )
            selection_teacher = apply_ssm_observation_teachers(
                selection_eeg, selection_fnirs, fits
            )
            teacher_rows.append(
                {
                    "task_id": task_id,
                    "provenance_id": fits.provenance_id,
                    "labels_used": fits.labels_used,
                    "parameter_sequences": len(parameter_eeg.values),
                    "selection_sequences": len(selection_eeg.values),
                    "eeg_target_dim": parameter_eeg.values.shape[-1],
                    "fnirs_target_dim": parameter_fnirs.values.shape[-1],
                    "eeg_target_channels": "|".join(channel_selection.eeg_channel_names),
                    "fnirs_target_channels": "|".join(channel_selection.fnirs_channel_names),
                    "eeg_target_indices": "|".join(map(str, channel_selection.eeg_indices)),
                    "fnirs_target_indices": "|".join(map(str, channel_selection.fnirs_indices)),
                    "fit_scope": config["source"]["fit_scope"],
                }
            )
            for mode in modes:
                parameter = ScreenPartition(prepared.parameter, parameter_teacher, str(mode))
                selection = ScreenPartition(prepared.selection, selection_teacher, str(mode))
                leakage = _provenance_control(
                    parameter,
                    selection,
                    task_id=str(task_id),
                    seed=seeds[0],
                )
                leakage_rows.append(
                    {"task_id": task_id, "mode": mode, **leakage}
                )
                for seed in seeds:
                    cell_dir = staging / "cells" / str(task_id) / str(mode) / f"seed_{seed}"
                    result = _train_cell(
                        str(task_id),
                        str(mode),
                        int(seed),
                        parameter,
                        selection,
                        config,
                        device=device,
                        smoke=smoke,
                        output=cell_dir,
                    )
                    row = {
                        "task_id": task_id,
                        "mode": mode,
                        "seed": seed,
                        "xpred_weight": result["xpred_weight"],
                        "representation_selection_score": result[
                            "representation_selection_score"
                        ],
                    }
                    row.update(
                        {
                            f"selection_{key}": value
                            for key, value in result["fit_selection"].items()
                        }
                    )
                    result_rows.append(row)
                    print(
                        f"{task_id} {mode} seed={seed} "
                        f"EEG dR2={row['selection_eeg_clean_delta_r2_vs_condition_time_mean']:.4f} "
                        f"fNIRS dR2={row['selection_fnirs_clean_delta_r2_vs_condition_time_mean']:.4f}",
                        flush=True,
                    )
        _write_csv(staging / "results.csv", result_rows)
        _write_csv(staging / "teacher_provenance.csv", teacher_rows)
        _write_csv(staging / "provenance_uncertainty_control.csv", leakage_rows)
        manifest = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "smoke": bool(smoke),
            "tasks": list(tasks),
            "modes": list(modes),
            "seeds": list(seeds),
            "execution_subset": bool(task_filter is not None or mode_filter is not None or seed_override is not None),
            "representation_steps_override": representation_steps_override,
            "head_steps_override": head_steps_override,
            "cell_count": len(result_rows),
            "architecture_contract": {
                "model_class": "SSMObservationSharedPrivateModel",
                "architecture_name": SSMObservationSharedPrivateModel.architecture_name,
                "vector_quantization": False,
                "quantizer_or_codebook_surface": False,
                "vq_stage": "deferred_until_continuous_gate",
            },
            "teacher_contract": {
                "modality_specific_self_teacher": True,
                "joint_teacher_privileged_upper_bound_only": True,
                "label_arguments_accepted_by_teacher_fit": False,
                "fit_scope": config["source"]["fit_scope"],
                "shared_target": "modality-specific observation trajectory",
                "private_target": "observation minus selected teacher trajectory",
                "uncertainty_weighting": "clipped inverse predictive variance Huber",
            },
            "downstream_contract": {
                "heads": config["statistics"]["downstream_heads"],
                "interaction_has_class_or_lag_bias": False,
                "interaction_pooling": "within-lag then equal supported-lag mean",
                "coupling_endpoint_claim": False,
                "q0_q1_deferred_until_vq": True,
            },
            "protected_open": False,
            "determinism": {
                "torch_deterministic_algorithms": True,
                "cublas_workspace_config": ":4096:8",
                "flash_sdp": False,
                "memory_efficient_sdp": False,
                "math_sdp": True,
            },
            "inputs": [
                {"path": str(config_path), "sha256": _sha256(config_path)},
                {"path": str(base_path), "sha256": _sha256(base_path)},
                {"path": str(optimization_path), "sha256": _sha256(optimization_path)},
                {
                    "path": str(Path(__file__).resolve()),
                    "sha256": _sha256(Path(__file__).resolve()),
                },
                *[
                    {
                        "path": str(REPO_ROOT / relative),
                        "sha256": _sha256(REPO_ROOT / relative),
                    }
                    for relative in (
                        "src/data/ssm_observation_targets.py",
                        "src/inference/modality_observation_ssm.py",
                        "src/losses/ssm_observation.py",
                        "src/tokenizers/ssm_observation_shared_private.py",
                    )
                ],
            ],
            "artifacts": [
                "results.csv",
                "teacher_provenance.csv",
                "provenance_uncertainty_control.csv",
            ],
        }
        _write_json(staging / "manifest.json", manifest)
        os.rename(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/configs/physiology_semantic_tokenizer/ssm_observation_target_screen.yaml",
    )
    parser.add_argument("--output")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--modes", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--representation-steps", type=int)
    parser.add_argument("--head-steps", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    config_path = Path(arguments.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = (
        Path(arguments.output)
        if arguments.output
        else REPO_ROOT
        / loaded["output"]["root"]
        / (
            "20260821_ssm_observation_screen_smoke_v1"
            if arguments.smoke
            else "20260821_ssm_observation_screen_full_v1"
        )
    )
    print(
        run(
            config_path,
            output.resolve(),
            smoke=bool(arguments.smoke),
            task_filter=arguments.tasks,
            mode_filter=arguments.modes,
            seed_override=arguments.seeds,
            representation_steps_override=arguments.representation_steps,
            head_steps_override=arguments.head_steps,
        )
    )
