#!/usr/bin/env python3
"""Zero-retraining coupling/calibration controls for completed LC-SPVQ heads."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import optimize_lag_conditioned_spvq_architecture as optimized
from experiments import run_lag_conditioned_spvq as reviewed
from src.metrics.lag_conditioned_downstream import (
    subject_equal_classification_metrics,
)
from src.tokenizers.lag_conditioned_shared_private_vq import (
    LagAwareContinuousMatchingLoss,
    _masked_lag_balanced_mean,
)


SCHEMA = "lc_spvq_coupling_calibration_audit_v1"


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


def _cross_entropy(logits: np.ndarray, target: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(target, dtype=np.int64)
    shifted = values - values.max(axis=1, keepdims=True)
    log_norm = np.log(np.exp(shifted).sum(axis=1))
    return float(np.mean(log_norm - shifted[np.arange(len(labels)), labels]))


def _fit_private_calibrators(
    logits: np.ndarray, target: np.ndarray
) -> Mapping[str, np.ndarray | float]:
    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(target, dtype=np.int64)
    class_count = values.shape[1]

    def bias_objective(raw: np.ndarray) -> float:
        bias = np.concatenate((raw, np.asarray([0.0])))
        return _cross_entropy(values + bias[None, :], labels)

    bias_result = minimize(
        bias_objective,
        np.zeros(class_count - 1, dtype=np.float64),
        method="BFGS",
        options={"maxiter": 1000, "gtol": 1e-9},
    )
    bias = np.concatenate((bias_result.x, np.asarray([0.0])))
    bias -= bias.mean()

    def temperature_objective(raw: np.ndarray) -> float:
        temperature = float(np.exp(np.clip(raw[0], -5.0, 5.0)))
        intercept = np.concatenate((raw[1:], np.asarray([0.0])))
        intercept -= intercept.mean()
        return _cross_entropy(values / temperature + intercept[None, :], labels)

    temperature_result = minimize(
        temperature_objective,
        np.zeros(class_count, dtype=np.float64),
        method="BFGS",
        options={"maxiter": 2000, "gtol": 1e-9},
    )
    temperature = float(np.exp(np.clip(temperature_result.x[0], -5.0, 5.0)))
    intercept = np.concatenate((temperature_result.x[1:], np.asarray([0.0])))
    intercept -= intercept.mean()
    return {
        "bias": bias,
        "bias_fit_success": bool(bias_result.success),
        "temperature": temperature,
        "temperature_intercept": intercept,
        "temperature_fit_success": bool(temperature_result.success),
    }


def _within_group_permutation(
    subjects: np.ndarray,
    conditions: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    permutation = np.arange(len(subjects))
    for subject in sorted(set(subjects.astype(str).tolist())):
        for condition in sorted(set(conditions[subjects == subject].astype(str).tolist())):
            selected = np.flatnonzero(
                (subjects.astype(str) == subject)
                & (conditions.astype(str) == condition)
            )
            if len(selected) > 1:
                candidate = selected.copy()
                for _ in range(100):
                    rng.shuffle(candidate)
                    if np.all(candidate != selected):
                        break
                permutation[selected] = candidate
    return permutation


def _metric_row(
    *,
    task_id: str,
    candidate_id: str,
    role: str,
    control: str,
    logits: np.ndarray,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    class_names = reviewed.TASK_SPECS[task_id].class_names
    metrics = subject_equal_classification_metrics(
        arrays["target"],
        np.argmax(logits, axis=1),
        arrays["subject"].astype(str),
        class_names=class_names,
    )
    return {
        "schema": SCHEMA,
        "task_id": task_id,
        "candidate_id": candidate_id,
        "role": role,
        "control": control,
        "sample_count": len(arrays["target"]),
        "subject_count": metrics["subject_count"],
        "subject_equal_macro_f1": metrics["subject_equal_macro_f1"],
        "subject_equal_balanced_accuracy": metrics["subject_equal_balanced_accuracy"],
        "cross_entropy_nats": _cross_entropy(logits, arrays["target"]),
    }


def _decomposed_from_export(
    model: Any,
    arrays: Mapping[str, np.ndarray],
    *,
    device: torch.device,
) -> Mapping[str, np.ndarray]:
    eeg = torch.from_numpy(np.asarray(arrays["eeg_posterior"], dtype=np.float32)).to(device)
    fnirs = torch.from_numpy(np.asarray(arrays["fnirs_posterior"], dtype=np.float32)).to(device)
    eeg_mask = torch.from_numpy(np.asarray(arrays["eeg_token_valid_mask"], dtype=bool)).to(device)
    fnirs_mask = torch.from_numpy(np.asarray(arrays["fnirs_token_valid_mask"], dtype=bool)).to(device)
    with torch.no_grad():
        _, pair_mask, components = model.coupling_head(
            eeg,
            fnirs,
            eeg_valid_mask=eeg_mask,
            fnirs_valid_mask=fnirs_mask,
            return_mask=True,
            return_components=True,
        )
        interaction = _masked_lag_balanced_mean(
            components["interaction"], pair_mask, model.allowed_lags
        )
        interaction = interaction - interaction.mean(dim=-1, keepdim=True)
        bias = _masked_lag_balanced_mean(
            components["bias"], pair_mask, model.allowed_lags
        )
        total = _masked_lag_balanced_mean(
            components["interaction"] + components["bias"],
            pair_mask,
            model.allowed_lags,
        )
    return {
        "interaction": interaction.cpu().numpy(),
        "bias": bias.cpu().numpy(),
        "lag_balanced_coupling": total.cpu().numpy(),
    }


def _controls_for_role(
    *,
    task_id: str,
    candidate_id: str,
    role: str,
    arrays: Mapping[str, np.ndarray],
    calibration: Mapping[str, np.ndarray | float],
    shuffle_iterations: int,
    shuffle_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    private = np.asarray(arrays["private_only_logits"], dtype=np.float64)
    coupling = np.asarray(arrays["coupling_only_logits"], dtype=np.float64)
    interaction = np.asarray(arrays["interaction_class_centered_logits"], dtype=np.float64)
    bias_head = np.asarray(arrays["coupling_bias_only_logits"], dtype=np.float64)
    marginal = np.asarray(arrays["shared_marginal_only_logits"], dtype=np.float64)
    controls = {
        "H0_private_only": private,
        "H1_private_plus_train_only_class_bias": private
        + np.asarray(calibration["bias"])[None, :],
        "H1b_private_temperature_intercept": private
        / float(calibration["temperature"])
        + np.asarray(calibration["temperature_intercept"])[None, :],
        "H2_private_plus_head_bias_only": private + bias_head,
        "H3_private_plus_interaction_only": private + interaction,
        "H3b_private_shared_marginal": private + marginal,
        "H3c_private_shared_marginal_interaction": private + marginal + interaction,
        "H4_original_combined": private + coupling,
    }
    rows = [
        _metric_row(
            task_id=task_id,
            candidate_id=candidate_id,
            role=role,
            control=name,
            logits=logits,
            arrays=arrays,
        )
        for name, logits in controls.items()
    ]
    shuffled_rows = []
    for iteration in range(int(shuffle_iterations)):
        permutation = _within_group_permutation(
            arrays["subject"],
            arrays["condition"],
            seed=int(shuffle_seed) + iteration,
        )
        shuffled_rows.append(
            _metric_row(
                task_id=task_id,
                candidate_id=candidate_id,
                role=role,
                control="H2_private_plus_shuffled_coupling",
                logits=private + coupling[permutation],
                arrays=arrays,
            )
        )
    for field in (
        "subject_equal_macro_f1",
        "subject_equal_balanced_accuracy",
        "cross_entropy_nats",
    ):
        values = np.asarray([row[field] for row in shuffled_rows], dtype=float)
        rows.append(
            {
                "schema": SCHEMA,
                "task_id": task_id,
                "candidate_id": candidate_id,
                "role": role,
                "control": f"H2_private_plus_shuffled_coupling_{field}_distribution",
                "shuffle_iterations": int(shuffle_iterations),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        )
    variance = {
        "interaction_class_variance": np.var(interaction, axis=0).tolist(),
        "interaction_total_variance": float(np.var(interaction)),
        "interaction_logit_range": [float(interaction.min()), float(interaction.max())],
        "head_bias_class_variance_across_samples": np.var(bias_head, axis=0).tolist(),
        "coupling_class_variance_across_samples": np.var(coupling, axis=0).tolist(),
    }
    return rows, variance


def _position_only_control(
    partition: Any,
    *,
    lags: Sequence[int],
    device: torch.device,
    steps: int,
    seed: int,
) -> list[dict[str, Any]]:
    # Keep complete groups while bounding the quadratic contrastive matrix.
    subjects = np.asarray(partition.subject).astype(str)
    conditions = np.asarray(partition.condition).astype(str)
    selected: list[int] = []
    for group in sorted(set(zip(subjects.tolist(), conditions.tolist()))):
        indices = np.flatnonzero((subjects == group[0]) & (conditions == group[1]))
        selected.extend(indices.tolist())
        if len(selected) >= 32:
            break
    selected = selected[:32]
    subject = subjects[selected]
    condition = conditions[selected]
    trial_ids = np.asarray(partition.sample_id).astype(str)[selected]
    valid_eeg = torch.from_numpy(partition.eeg_token_mask[selected]).to(device)
    valid_fnirs = torch.from_numpy(partition.fnirs_token_mask[selected]).to(device)
    token_count = valid_eeg.shape[1]
    condition_values = tuple(sorted(set(condition.tolist())))
    condition_index = torch.tensor(
        [condition_values.index(value) for value in condition],
        device=device,
        dtype=torch.long,
    )
    rows = []
    for lag in lags:
        corrected = reviewed.make_same_group_time_negative_mask(
            subject,
            condition,
            token_count=token_count,
            lag=int(lag),
            query_trial_ids=trial_ids,
            target_trial_ids=trial_ids,
            device=device,
        )
        legacy = reviewed.make_same_group_time_negative_mask(
            subject,
            condition,
            token_count=token_count,
            lag=0,
            query_trial_ids=trial_ids,
            target_trial_ids=trial_ids,
            device=device,
        )
        for regime, negative_mask in (
            ("endpoint_aligned", corrected),
            ("legacy_same_position", legacy),
        ):
            random.seed(int(seed) + int(lag))
            torch.manual_seed(int(seed) + int(lag))
            dimension = 16
            query_position = torch.nn.Parameter(
                torch.randn(token_count, dimension, device=device) * 0.05
            )
            target_position = torch.nn.Parameter(
                torch.randn(token_count, dimension, device=device) * 0.05
            )
            condition_embedding = torch.nn.Parameter(
                torch.randn(len(condition_values), dimension, device=device) * 0.05
            )
            optimizer = torch.optim.Adam(
                (query_position, target_position, condition_embedding), lr=0.05
            )
            objective = LagAwareContinuousMatchingLoss(
                positive_lag_weights={int(lag): 1.0},
                temperature=0.07,
                bidirectional=False,
                target_stop_gradient=False,
            ).to(device)
            initial = None
            final = None
            for step in range(int(steps) + 1):
                query = query_position.unsqueeze(0).expand(len(selected), -1, -1)
                target = target_position.unsqueeze(0).expand(len(selected), -1, -1)
                group = condition_embedding[condition_index].unsqueeze(1)
                loss = objective(
                    query + group,
                    target + group,
                    query_valid_mask=valid_eeg,
                    target_valid_mask=valid_fnirs,
                    negative_mask=negative_mask,
                )
                if step == 0:
                    initial = float(loss.detach().cpu())
                if step == int(steps):
                    final = float(loss.detach().cpu())
                    break
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            rows.append(
                {
                    "schema": SCHEMA,
                    "control": "position_condition_only_matching",
                    "regime": regime,
                    "lag_tokens": int(lag),
                    "sample_count": len(selected),
                    "optimizer_steps": int(steps),
                    "initial_loss": initial,
                    "final_loss": final,
                    "loss_reduction": float(initial - final),
                }
            )
    return rows


def run(args: argparse.Namespace) -> Path:
    run_root = Path(args.run_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    output.mkdir(parents=True)
    optimization_config = yaml.safe_load((run_root / "optimization_config.yaml").read_text(encoding="utf-8"))
    base_config = yaml.safe_load((run_root / "base_config.yaml").read_text(encoding="utf-8"))
    device = torch.device(args.device)
    candidate_ids = tuple(args.candidates)
    candidate_lookup = {
        str(value["candidate_id"]): value
        for value in optimization_config["candidates"]
    }
    rows: list[dict[str, Any]] = []
    variance_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    source_inputs = (
        Path(__file__).resolve(),
        REPO_ROOT / "experiments/run_lag_conditioned_spvq.py",
        REPO_ROOT / "src/tokenizers/lag_conditioned_shared_private_vq.py",
        run_root / "optimization_config.yaml",
        run_root / "base_config.yaml",
    )
    inputs = [
        {"path": str(path), "sha256": _sha256(path)} for path in source_inputs
    ]
    for task_index, task_id in enumerate(optimization_config["execution"]["tasks"]):
        prepared = optimized.prepare_fit_selection_task(
            optimization_config,
            str(task_id),
            base_config=base_config,
            derangement_seed=int(optimization_config["execution"]["derangement_seed"]),
        )
        if task_index == 0:
            position_rows.extend(
                _position_only_control(
                    prepared.parameter,
                    lags=(0, 1, 2, 3, 4, 5),
                    device=device,
                    steps=int(args.position_steps),
                    seed=int(args.seed),
                )
            )
        for candidate_id in candidate_ids:
            candidate = candidate_lookup[candidate_id]
            runtime = optimized.candidate_runtime_config(
                base_config,
                optimization_config,
                candidate,
                seed=int(optimization_config["execution"]["seed"]),
            )
            model = reviewed._lc_spvq_model(prepared, runtime).to(device)
            model._lc_spvq_task_id = str(task_id)
            checkpoint_path = (
                run_root
                / "tasks"
                / str(task_id)
                / "candidates"
                / candidate_id
                / "checkpoints"
                / "head_best.pt"
            )
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model.set_quantization_strength(float(checkpoint["quantization_strength"]))
            model.set_posterior_temperature(float(checkpoint["posterior_temperature"]))
            model.eval()
            role_arrays: dict[str, Mapping[str, np.ndarray]] = {}
            for role, partition in (
                ("fit_parameter", prepared.parameter),
                ("fit_selection", prepared.selection),
            ):
                _, arrays, _ = reviewed._evaluate_lc_spvq(
                    model,
                    partition,
                    config=runtime,
                    device=device,
                    seed=int(args.seed),
                )
                role_arrays[role] = arrays
            calibration = _fit_private_calibrators(
                role_arrays["fit_parameter"]["private_only_logits"],
                role_arrays["fit_parameter"]["target"],
            )
            development_path = (
                run_root
                / "development"
                / str(task_id)
                / candidate_id
                / "outputs.npz"
            )
            if development_path.exists():
                with np.load(development_path, allow_pickle=False) as payload:
                    development = {key: payload[key] for key in payload.files}
                decomposed = _decomposed_from_export(
                    model, development, device=device
                )
                development = dict(development)
                development["interaction_class_centered_logits"] = decomposed[
                    "interaction"
                ]
                development["coupling_bias_only_logits"] = decomposed["bias"]
                development["shared_marginal_only_logits"] = development[
                    "shared_marginal_only_logits"
                ]
                role_arrays["development_apply"] = development
                inputs.append(
                    {
                        "path": str(development_path),
                        "sha256": _sha256(development_path),
                    }
                )
            for role, arrays in role_arrays.items():
                role_rows, variance = _controls_for_role(
                    task_id=str(task_id),
                    candidate_id=candidate_id,
                    role=role,
                    arrays=arrays,
                    calibration=calibration,
                    shuffle_iterations=int(args.shuffle_iterations),
                    shuffle_seed=int(args.seed) + 1000 * task_index,
                )
                rows.extend(role_rows)
                variance_rows.append(
                    {
                        "schema": SCHEMA,
                        "task_id": str(task_id),
                        "candidate_id": candidate_id,
                        "role": role,
                        **variance,
                    }
                )
            inputs.append(
                {"path": str(checkpoint_path), "sha256": _sha256(checkpoint_path)}
            )
    scalar_rows = [
        row for row in rows if row.get("subject_equal_macro_f1", "") != ""
    ]
    baselines = {
        (str(row["task_id"]), str(row["candidate_id"]), str(row["role"])): row
        for row in scalar_rows
        if row["control"] == "H0_private_only"
    }
    delta_rows = []
    for row in scalar_rows:
        baseline = baselines[
            (str(row["task_id"]), str(row["candidate_id"]), str(row["role"]))
        ]
        delta_rows.append(
            {
                "schema": SCHEMA,
                "task_id": row["task_id"],
                "candidate_id": row["candidate_id"],
                "role": row["role"],
                "control": row["control"],
                "macro_f1_delta_vs_private": float(row["subject_equal_macro_f1"])
                - float(baseline["subject_equal_macro_f1"]),
                "balanced_accuracy_delta_vs_private": float(
                    row["subject_equal_balanced_accuracy"]
                )
                - float(baseline["subject_equal_balanced_accuracy"]),
                "cross_entropy_delta_vs_private_nats": float(row["cross_entropy_nats"])
                - float(baseline["cross_entropy_nats"]),
            }
        )
    _write_csv(output / "control_metrics.csv", rows)
    _write_csv(output / "control_deltas_vs_private.csv", delta_rows)
    _write_csv(output / "interaction_variance.csv", variance_rows)
    _write_csv(output / "position_only_control.csv", position_rows)
    manifest = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run": str(run_root),
        "candidates": list(candidate_ids),
        "tasks": list(optimization_config["execution"]["tasks"]),
        "shuffle_iterations": int(args.shuffle_iterations),
        "position_optimizer_steps": int(args.position_steps),
        "fit_calibration_scope": "fit_parameter_only",
        "development_role": "existing frozen descriptive exports only",
        "negative_mask_control": {
            "corrected": "E_trial(t) vs F_other_trial(t+lag)",
            "legacy": "E_trial(t) vs F_other_trial(t)",
        },
        "interaction_definition": "lag-balanced, class-centered, class_bias and lag_bias removed",
        "claim_scope": "zero-retraining architecture/calibration QC; not a coupling endpoint",
        "inputs": inputs,
        "artifacts": [
            "control_metrics.csv",
            "control_deltas_vs_private.csv",
            "interaction_variance.csv",
            "position_only_control.csv",
        ],
    }
    _write_json(output / "manifest.json", manifest)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        default="experiments/runs/physiology_semantic_tokenizer/lag_conditioned_spvq_optimization",
    )
    parser.add_argument(
        "--output",
        default="experiments/runs/physiology_semantic_tokenizer/coupling_calibration_audit/20260821_existing_checkpoint_controls_v1",
    )
    parser.add_argument("--candidates", nargs="+", default=("lag05_h23",))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--shuffle-iterations", type=int, default=200)
    parser.add_argument("--position-steps", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
