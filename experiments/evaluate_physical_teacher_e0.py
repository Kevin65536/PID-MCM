#!/usr/bin/env python3
"""Evaluate E0 teacher validity with subject-held-out, null-calibrated evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.factory import create_configured_multimodal_dataloaders
from src.teachers.physical_state_teacher import PhysicalStateTeacher


E0_SCHEMA = "physiology_semantic_e0_v1"
EEG_COORDINATES = ("r_mean", "r_slope", "r_logvar", "s_mean", "s_slope", "s_logvar")
FNIRS_COORDINATES = (
    "delta_f_mean", "delta_hbo_mean", "delta_hb_mean",
    "delta_f_slope", "delta_hbo_slope", "delta_hb_slope",
    "delta_f_logvar", "delta_hbo_logvar", "delta_hb_logvar",
)


@dataclass
class SplitEvidence:
    eeg_features: np.ndarray
    fnirs_features: np.ndarray
    fnirs_highwl_features: np.ndarray
    eeg_target: np.ndarray
    eeg_uncertainty: np.ndarray
    fnirs_target: np.ndarray
    fnirs_uncertainty: np.ndarray
    subjects: np.ndarray
    eeg_teacher_error: np.ndarray
    eeg_zero_error: np.ndarray
    eeg_history_error: np.ndarray
    fnirs_teacher_error: np.ndarray
    fnirs_zero_error: np.ndarray
    fnirs_history_error: np.ndarray
    mask_rows: list[dict[str, Any]]


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _patchify(signal: torch.Tensor, patch_size: int) -> torch.Tensor:
    return signal.unfold(-1, patch_size, patch_size).permute(0, 2, 1, 3).contiguous()


def signal_features(patches: torch.Tensor, spectral_bins: int) -> torch.Tensor:
    """Patch-local statistics and low-frequency log-spectrum features."""
    sample_count = patches.shape[-1]
    time = torch.linspace(-1.0, 1.0, sample_count, device=patches.device, dtype=patches.dtype)
    slope = (patches * time).sum(dim=-1) / time.square().sum().clamp_min(1e-8)
    mean = patches.mean(dim=-1)
    std = patches.std(dim=-1, unbiased=False)
    rms = patches.square().mean(dim=-1).clamp_min(1e-8).sqrt()
    spectrum = torch.fft.rfft(patches, dim=-1).abs().clamp_min(1e-8).log()
    spectrum = spectrum[..., 1 : spectral_bins + 1]
    return torch.cat((mean, std, slope, rms, spectrum.flatten(start_dim=2)), dim=-1)


def subject_bootstrap_ci(
    differences: np.ndarray,
    subjects: np.ndarray,
    *,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    unique = np.unique(subjects)
    per_subject = np.asarray([differences[subjects == subject].mean() for subject in unique])
    observed = float(per_subject.mean())
    if unique.size < 2 or iterations <= 0:
        return observed, observed, observed
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        samples[index] = rng.choice(per_subject, size=per_subject.size, replace=True).mean()
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return observed, float(lower), float(upper)


def subject_signflip_null(
    differences: np.ndarray,
    subjects: np.ndarray,
    *,
    iterations: int,
    quantile: float,
    rng: np.random.Generator,
) -> float:
    unique = np.unique(subjects)
    per_subject = np.asarray([differences[subjects == subject].mean() for subject in unique])
    null = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=per_subject.size)
        null[index] = np.mean(per_subject * signs)
    return float(np.quantile(null, quantile))


def _fit_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    alphas: Sequence[float],
) -> tuple[StandardScaler, Ridge, float, float]:
    scaler = StandardScaler().fit(train_x)
    train_scaled = scaler.transform(train_x)
    val_scaled = scaler.transform(val_x)
    best: tuple[float, Ridge, float] | None = None
    for alpha in alphas:
        model = Ridge(alpha=float(alpha)).fit(train_scaled, train_y)
        mse = float(np.mean(np.square(model.predict(val_scaled) - val_y)))
        if best is None or mse < best[0]:
            best = (mse, model, float(alpha))
    assert best is not None
    baseline_mse = float(np.mean(np.square(val_y - train_y.mean())))
    gain = baseline_mse - best[0]
    return scaler, best[1], best[2], gain


def _permutation_null(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    alpha: float,
    iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    scaler = StandardScaler().fit(train_x)
    train_scaled = scaler.transform(train_x)
    val_scaled = scaler.transform(val_x)
    baseline_mse = float(np.mean(np.square(val_y - train_y.mean())))
    null = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        permuted = rng.permutation(train_y)
        prediction = Ridge(alpha=alpha).fit(train_scaled, permuted).predict(val_scaled)
        null[index] = baseline_mse - np.mean(np.square(prediction - val_y))
    return null


def _collect_split(loader: Iterable[Mapping[str, Any]], adapter: PhysicalStateTeacher) -> SplitEvidence:
    arrays: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "eeg_features", "fnirs_features", "fnirs_highwl_features", "eeg_target",
            "eeg_uncertainty", "fnirs_target", "fnirs_uncertainty", "subjects",
            "eeg_teacher_error", "eeg_zero_error", "eeg_history_error",
            "fnirs_teacher_error", "fnirs_zero_error", "fnirs_history_error",
        )
    }
    mask_rows: list[dict[str, Any]] = []
    for batch in loader:
        teacher = adapter(batch["teacher"])
        eeg_patches = _patchify(batch["eeg"], 400)
        fnirs_patches = _patchify(batch["fnirs"], 20)
        eeg_source = _patchify(batch["decomposition"]["eeg_source"], 400)
        fnirs_source = _patchify(batch["decomposition"]["fnirs_source"], 20)
        valid = teacher.valid_mask

        eeg_feature = signal_features(eeg_patches, spectral_bins=16)
        fnirs_feature = signal_features(fnirs_patches, spectral_bins=10)
        highwl_feature = signal_features(fnirs_patches[:, :, :1], spectral_bins=10)

        eeg_teacher_error = (eeg_source - eeg_patches).square().mean(dim=(-1, -2))
        fnirs_teacher_error = (fnirs_source - fnirs_patches).square().mean(dim=(-1, -2))
        eeg_zero_error = eeg_patches.square().mean(dim=(-1, -2))
        fnirs_zero_error = fnirs_patches.square().mean(dim=(-1, -2))
        eeg_history = torch.zeros_like(eeg_patches)
        fnirs_history = torch.zeros_like(fnirs_patches)
        eeg_history[:, 1:] = eeg_patches[:, :-1]
        fnirs_history[:, 1:] = fnirs_patches[:, :-1]
        eeg_history_error = (eeg_history - eeg_patches).square().mean(dim=(-1, -2))
        fnirs_history_error = (fnirs_history - fnirs_patches).square().mean(dim=(-1, -2))

        subject_matrix = batch["subject_id"].view(-1, 1).expand_as(valid)
        for key, tensor in (
            ("eeg_features", eeg_feature),
            ("fnirs_features", fnirs_feature),
            ("fnirs_highwl_features", highwl_feature),
            ("eeg_target", teacher.eeg_target),
            ("eeg_uncertainty", teacher.eeg_uncertainty),
            ("fnirs_target", teacher.fnirs_target),
            ("fnirs_uncertainty", teacher.fnirs_uncertainty),
            ("subjects", subject_matrix),
            ("eeg_teacher_error", eeg_teacher_error),
            ("eeg_zero_error", eeg_zero_error),
            ("eeg_history_error", eeg_history_error),
            ("fnirs_teacher_error", fnirs_teacher_error),
            ("fnirs_zero_error", fnirs_zero_error),
            ("fnirs_history_error", fnirs_history_error),
        ):
            arrays[key].append(tensor[valid].detach().cpu().numpy())

        eeg_zero_gain = eeg_zero_error - eeg_teacher_error
        fnirs_zero_gain = fnirs_zero_error - fnirs_teacher_error
        for sample_index, (row, subject) in enumerate(zip(valid, batch["subject_id"])):
            mask_rows.append(
                {
                    "subject_id": int(subject),
                    "valid_patches": int(row.sum()),
                    "total_patches": int(row.numel()),
                    "coverage": float(row.float().mean()),
                    "eeg_gain_masked": float(eeg_zero_gain[sample_index][row].mean()) if row.any() else float("nan"),
                    "eeg_gain_mask_ablation": float(eeg_zero_gain[sample_index].mean()),
                    "fnirs_gain_masked": float(fnirs_zero_gain[sample_index][row].mean()) if row.any() else float("nan"),
                    "fnirs_gain_mask_ablation": float(fnirs_zero_gain[sample_index].mean()),
                }
            )

    return SplitEvidence(
        **{key: np.concatenate(value, axis=0) for key, value in arrays.items()},
        mask_rows=mask_rows,
    )


def _predictive_evidence(
    split: SplitEvidence,
    modality: str,
    baseline_name: str,
    bootstrap_iterations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    teacher_error = getattr(split, f"{modality}_teacher_error")
    baseline_error = getattr(split, f"{modality}_{baseline_name}_error")
    gain = baseline_error - teacher_error
    observed, lower, upper = subject_bootstrap_ci(
        gain, split.subjects, iterations=bootstrap_iterations, rng=rng
    )
    return {
        "baseline": baseline_name,
        "teacher_mse": float(teacher_error.mean()),
        "baseline_mse": float(baseline_error.mean()),
        "normalized_mse_gain": observed,
        "bootstrap_ci_95": [lower, upper],
        "subjects": int(np.unique(split.subjects).size),
        "patches": int(split.subjects.size),
    }


def _synthetic_recovery(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=(1000, 5))
    mixing = rng.normal(size=(5, 12))
    observation = state @ mixing + rng.normal(scale=0.1, size=(1000, 12))
    train, test = np.arange(800), np.arange(800, 1000)
    scaler = StandardScaler().fit(observation[train])
    model = Ridge(alpha=1.0).fit(scaler.transform(observation[train]), state[train])
    prediction = model.predict(scaler.transform(observation[test]))
    mse = np.mean(np.square(prediction - state[test]))
    baseline = np.mean(np.square(state[test] - state[train].mean(axis=0)))
    reference = state[:128]
    reference_observation = reference @ mixing
    perturbation_response = []
    for coordinate in range(state.shape[1]):
        perturbed = reference.copy()
        perturbed[:, coordinate] += 1.0
        response = np.linalg.norm((perturbed @ mixing) - reference_observation, axis=1).mean()
        perturbation_response.append(float(response))
    return {
        "mse": float(mse),
        "baseline_mse": float(baseline),
        "gain": float(baseline - mse),
        "coordinate_perturbation_response": perturbation_response,
    }


def _hash_cache_files(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in config.get("data", {}).get("cache_sources", []):
        root = REPO_ROOT / source["root"]
        for path in sorted(root.glob("subject_*/*")):
            if not path.is_file() or (path.name != "cache_manifest.json" and path.suffix != ".npz"):
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            rows.append(
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "sha256": digest.hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
            )
    return rows


def _resource_snapshot() -> dict[str, Any] | None:
    path = REPO_ROOT / ".claude_resources.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_predictive_checks(
    run_dir: Path,
    validation_predictive: Mapping[str, Any],
    test_predictive: Mapping[str, Any],
    observability_rows: Sequence[Mapping[str, Any]],
) -> None:
    figure_dir = run_dir / "figures"
    figure_data_dir = run_dir / "figure_data"
    figure_dir.mkdir(exist_ok=True)
    figure_data_dir.mkdir(exist_ok=True)
    primary = test_predictive or validation_predictive
    split = "test" if test_predictive else "validation"
    coordinate_rows = [row for row in observability_rows if row["split"] == split]
    figure_data = {"split": split, "primary": primary, "coordinates": coordinate_rows}
    _write_json(figure_data_dir / "predictive_check.json", figure_data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    modalities = [name for name in ("eeg", "fnirs") if name in primary]
    gains = [primary[name]["normalized_mse_gain"] for name in modalities]
    lower = [gain - primary[name]["bootstrap_ci_95"][0] for name, gain in zip(modalities, gains)]
    upper = [primary[name]["bootstrap_ci_95"][1] - gain for name, gain in zip(modalities, gains)]
    axes[0].bar(modalities, gains, color=["#2563eb", "#16a34a"], edgecolor="black", linewidth=0.6)
    axes[0].errorbar(modalities, gains, yerr=[lower, upper], fmt="none", color="black", capsize=4)
    axes[0].axhline(0.0, color="#6b7280", linewidth=1)
    axes[0].set_ylabel("Normalized MSE gain")
    axes[0].set_title(f"Posterior predictive check ({split})")
    axes[0].grid(axis="y", alpha=0.25)

    labels = [f"{row['modality']}:{row['coordinate']}" for row in coordinate_rows]
    values = [float(row["mse_gain"]) for row in coordinate_rows]
    colors = ["#16a34a" if row.get("passed", row.get("selected", False)) else "#dc2626" for row in coordinate_rows]
    y = np.arange(len(labels))
    axes[1].barh(y, values, color=colors, edgecolor="black", linewidth=0.4)
    axes[1].set_yticks(y, labels=labels, fontsize=7)
    axes[1].axvline(0.0, color="#6b7280", linewidth=1)
    axes[1].set_xlabel("State observability MSE gain")
    axes[1].set_title(f"Coordinate evidence ({split})")
    axes[1].grid(axis="x", alpha=0.25)
    fig.savefig(figure_dir / "predictive_check.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(figure_dir / "predictive_check.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    e0 = config.get("e0", {})
    seed = int(config.get("training", {}).get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir).resolve() if args.output_dir else (
        REPO_ROOT / "experiments" / "runs" / "physiology_semantic_tokenizer" /
        "e0_teacher_validity" / f"{stamp}_{config['experiment']['name']}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    protocol = {
        "schema": E0_SCHEMA,
        "primary_endpoint": "subject_bootstrap normalized MSE gain over validation-selected zero/history baseline",
        "protected_boundary": {"selection": "validation subjects", "final_evaluation": "test subjects"},
        "coordinate_selection": "validation gain above the configured quantile of label-permutation null",
        "gate_rule": "both observed modalities and at least one coordinate per modality retain positive subject-bootstrap lower confidence bounds on protected test",
        "null_quantile": float(e0.get("null_quantile", 0.95)),
        "bootstrap_iterations": int(e0.get("bootstrap_iterations", 1000)),
        "permutation_iterations": int(e0.get("permutation_iterations", 128)),
        "created_before_test_access": True,
    }
    (run_dir / "decision_protocol.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    _write_json(
        run_dir / "metric_registry.json",
        {
            "schema": E0_SCHEMA,
            "primary": ["eeg_normalized_mse_gain", "fnirs_normalized_mse_gain"],
            "gate": ["coordinate_observability_gain", "subject_bootstrap_ci_95"],
            "diagnostic": ["paired_vs_highwl_gain", "uncertainty_standardized_error", "mask_coverage"],
        },
    )
    _write_json(
        run_dir / "evidence_calibration.json",
        {
            "schema": E0_SCHEMA,
            "status": "frozen_before_test",
            "calibration_split": "validation",
            "threshold_source": "empirical label-permutation null",
            "null_quantile": protocol["null_quantile"],
        },
    )

    dataloaders = create_configured_multimodal_dataloaders(config)
    adapter = PhysicalStateTeacher()
    train = _collect_split(dataloaders["train"], adapter)
    val = _collect_split(dataloaders["val"], adapter)
    declared_split = config.get("data", {}).get("split", {})
    expected_train_subjects = {int(value) for value in declared_split.get("train_subjects", [])}
    expected_val_subjects = {int(value) for value in declared_split.get("val_subjects", [])}
    minimum_subjects = int(e0.get("minimum_subjects_per_eval_split", 3))
    minimum_patches = int(e0.get("minimum_valid_patches_per_eval_split", 100))
    coverage_ok = (
        set(np.unique(train.subjects).astype(int)) == expected_train_subjects
        and set(np.unique(val.subjects).astype(int)) == expected_val_subjects
        and np.unique(val.subjects).size >= minimum_subjects
        and val.subjects.size >= minimum_patches
    )

    calibration_rows = []
    observability_rows = []
    fitted: dict[tuple[str, int, str], tuple[StandardScaler, Ridge]] = {}
    coordinate_pass: dict[str, list[str]] = {"eeg": [], "fnirs": []}
    alphas = tuple(float(value) for value in e0.get("ridge_alphas", [0.1, 1.0, 10.0, 100.0]))
    permutations = int(protocol["permutation_iterations"])
    for modality, names in (("eeg", EEG_COORDINATES), ("fnirs", FNIRS_COORDINATES)):
        train_x = getattr(train, f"{modality}_features")
        val_x = getattr(val, f"{modality}_features")
        train_target = getattr(train, f"{modality}_target")
        val_target = getattr(val, f"{modality}_target")
        for index, name in enumerate(names):
            scaler, model, alpha, gain = _fit_ridge(
                train_x, train_target[:, index], val_x, val_target[:, index], alphas
            )
            null = _permutation_null(
                train_x, train_target[:, index], val_x, val_target[:, index], alpha, permutations, rng
            )
            _, _, _, time_shift_gain = _fit_ridge(
                train_x,
                np.roll(train_target[:, index], max(5, train_target.shape[0] // 10)),
                val_x,
                val_target[:, index],
                [alpha],
            )
            threshold = float(np.quantile(null, protocol["null_quantile"]))
            selected = bool(coverage_ok and gain > threshold)
            if selected:
                coordinate_pass[modality].append(name)
                fitted[(modality, index, "paired")] = (scaler, model)
            observability_rows.append(
                {
                    "split": "validation",
                    "modality": modality,
                    "coordinate": name,
                    "alpha": alpha,
                    "mse_gain": gain,
                    "permutation_threshold": threshold,
                    "time_shift_mse_gain": time_shift_gain,
                    "selected": selected,
                }
            )

            if modality == "fnirs":
                _, _, _, highwl_gain = _fit_ridge(
                    train.fnirs_highwl_features,
                    train_target[:, index],
                    val.fnirs_highwl_features,
                    val_target[:, index],
                    alphas,
                )
                observability_rows[-1]["highwl_mse_gain"] = highwl_gain
                observability_rows[-1]["paired_minus_highwl_gain"] = gain - highwl_gain

    baseline_choice = {}
    validation_predictive = {}
    for modality in ("eeg", "fnirs"):
        zero = _predictive_evidence(val, modality, "zero", int(protocol["bootstrap_iterations"]), rng)
        history = _predictive_evidence(val, modality, "history", int(protocol["bootstrap_iterations"]), rng)
        chosen = "zero" if zero["baseline_mse"] <= history["baseline_mse"] else "history"
        baseline_choice[modality] = chosen
        evidence = zero if chosen == "zero" else history
        difference = getattr(val, f"{modality}_{chosen}_error") - getattr(val, f"{modality}_teacher_error")
        threshold = subject_signflip_null(
            difference,
            val.subjects,
            iterations=int(protocol["permutation_iterations"]),
            quantile=float(protocol["null_quantile"]),
            rng=rng,
        )
        evidence["subject_signflip_threshold"] = threshold
        evidence["above_null"] = bool(evidence["normalized_mse_gain"] > threshold)
        validation_predictive[modality] = evidence

    validation_pass = bool(
        coverage_ok
        and all(validation_predictive[m]["above_null"] for m in ("eeg", "fnirs"))
        and all(coordinate_pass[m] for m in ("eeg", "fnirs"))
    )

    test_predictive: dict[str, Any] = {}
    test_opened = False
    gate_passed = False
    test = None
    if validation_pass and not args.validation_only:
        test_opened = True
        test = _collect_split(dataloaders["test"], adapter)
        expected_test_subjects = {int(value) for value in declared_split.get("test_subjects", [])}
        test_coverage_ok = (
            set(np.unique(test.subjects).astype(int)) == expected_test_subjects
            and np.unique(test.subjects).size >= minimum_subjects
            and test.subjects.size >= minimum_patches
        )
        for modality in ("eeg", "fnirs"):
            test_predictive[modality] = _predictive_evidence(
                test, modality, baseline_choice[modality], int(protocol["bootstrap_iterations"]), rng
            )

        coordinate_test_pass = {"eeg": [], "fnirs": []}
        for modality, names in (("eeg", EEG_COORDINATES), ("fnirs", FNIRS_COORDINATES)):
            test_x = getattr(test, f"{modality}_features")
            test_target = getattr(test, f"{modality}_target")
            test_uncertainty = getattr(test, f"{modality}_uncertainty")
            train_target = getattr(train, f"{modality}_target")
            for index, name in enumerate(names):
                if name not in coordinate_pass[modality]:
                    continue
                scaler, model = fitted[(modality, index, "paired")]
                prediction = model.predict(scaler.transform(test_x))
                baseline_error = np.square(test_target[:, index] - train_target[:, index].mean())
                model_error = np.square(prediction - test_target[:, index])
                gain, lower, upper = subject_bootstrap_ci(
                    baseline_error - model_error,
                    test.subjects,
                    iterations=int(protocol["bootstrap_iterations"]),
                    rng=rng,
                )
                passed = bool(lower > 0)
                if passed:
                    coordinate_test_pass[modality].append(name)
                observability_rows.append(
                    {
                        "split": "test",
                        "modality": modality,
                        "coordinate": name,
                        "alpha": "frozen",
                        "mse_gain": gain,
                        "permutation_threshold": "validation_frozen",
                        "selected": True,
                        "bootstrap_lower": lower,
                        "bootstrap_upper": upper,
                        "passed": passed,
                    }
                )
                standardized = model_error / np.maximum(test_uncertainty[:, index], 1e-8)
                calibration_rows.append(
                    {
                        "modality": modality,
                        "coordinate": name,
                        "mean_standardized_squared_error": float(standardized.mean()),
                        "median_standardized_squared_error": float(np.median(standardized)),
                        "within_90_percent_interval": float(
                            np.mean(np.abs(prediction - test_target[:, index]) <= 1.645 * np.sqrt(test_uncertainty[:, index]))
                        ),
                    }
                )

        gate_passed = bool(
            test_coverage_ok
            and all(test_predictive[m]["bootstrap_ci_95"][0] > 0 for m in ("eeg", "fnirs"))
            and all(coordinate_test_pass[m] for m in ("eeg", "fnirs"))
        )
        coordinate_pass = coordinate_test_pass

    status = "gate_passed" if gate_passed else (
        "gate_blocked_validation" if not validation_pass else "gate_failed_test"
    )
    if args.validation_only and validation_pass:
        status = "validation_passed_test_unopened"
    cache_hashes = _hash_cache_files(config)
    hash_to_paths: dict[str, list[str]] = {}
    for row in cache_hashes:
        if str(row["path"]).endswith(".npz"):
            hash_to_paths.setdefault(str(row["sha256"]), []).append(str(row["path"]))
    duplicate_caches = [paths for paths in hash_to_paths.values() if len(paths) > 1]
    audit = {
        "schema": E0_SCHEMA,
        "status": status,
        "e0_passed": gate_passed,
        "validation_passed": validation_pass,
        "protected_test_opened": test_opened,
        "coverage_ok": coverage_ok,
        "validation_predictive": validation_predictive,
        "test_predictive": test_predictive,
        "admissible_coordinates": coordinate_pass,
        "excluded_coordinates": {
            "eeg": [name for name in EEG_COORDINATES if name not in coordinate_pass["eeg"]],
            "fnirs": [name for name in FNIRS_COORDINATES if name not in coordinate_pass["fnirs"]],
        },
        "synthetic_recovery": _synthetic_recovery(seed),
        "cache_hashes": cache_hashes,
        "duplicate_cache_groups": duplicate_caches,
        "resource_snapshot": _resource_snapshot(),
    }
    _write_json(run_dir / "teacher_audit.json", audit)
    split_sha256 = hashlib.sha256(
        json.dumps(config.get("data", {}).get("split", {}), sort_keys=True).encode("utf-8")
    ).hexdigest()
    _write_json(
        run_dir / "gate_decision.json",
        {
            "schema": E0_SCHEMA,
            "gate": "G0",
            "status": status,
            "e0_passed": gate_passed,
            "admissible_coordinates": coordinate_pass,
            "data_contract": config.get("data", {}).get("contract"),
            "split_sha256": split_sha256,
            "cache_source_roots": [
                source.get("root") for source in config.get("data", {}).get("cache_sources", [])
            ],
            "teacher_audit_sha256": hashlib.sha256((run_dir / "teacher_audit.json").read_bytes()).hexdigest(),
            "decision_protocol_sha256": hashlib.sha256((run_dir / "decision_protocol.yaml").read_bytes()).hexdigest(),
            "metric_registry_sha256": hashlib.sha256((run_dir / "metric_registry.json").read_bytes()).hexdigest(),
            "evidence_calibration_sha256": hashlib.sha256((run_dir / "evidence_calibration.json").read_bytes()).hexdigest(),
        },
    )
    _write_csv(run_dir / "state_observability.csv", observability_rows)
    _write_csv(run_dir / "posterior_calibration.csv", calibration_rows)
    mask_rows = [dict(row, split="train") for row in train.mask_rows]
    mask_rows += [dict(row, split="val") for row in val.mask_rows]
    if test is not None:
        mask_rows += [dict(row, split="test") for row in test.mask_rows]
    _write_csv(run_dir / "mask_coverage.csv", mask_rows)
    _plot_predictive_checks(run_dir, validation_predictive, test_predictive, observability_rows)
    _write_json(
        run_dir / "manifest.json",
        {
            "schema": E0_SCHEMA,
            "status": status,
            "e0_passed": gate_passed,
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip(),
            "dirty_worktree": bool(subprocess.run(
                ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(json.dumps({"run_dir": str(run_dir), "status": status, "e0_passed": gate_passed}, sort_keys=True))
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--validation-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
