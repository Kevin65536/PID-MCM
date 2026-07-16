#!/usr/bin/env python3
"""Run the validation side of the E0-v2 information-transfer contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.e0_v2_measurement_audit import run_measurement_audit
from src.data.factory import create_configured_multimodal_dataloaders
from src.teachers.physical_state_teacher import PhysicalStateTeacher


SCHEMA = "physiology_semantic_e0_v2"
EEG_LOCAL_NAMES = ("r_mean", "r_slope", "s_mean", "s_slope")
FNIRS_LOCAL_NAMES = (
    "delta_f_mean", "delta_hbo_mean", "delta_hb_mean",
    "delta_f_slope", "delta_hbo_slope", "delta_hb_slope",
)
FNIRS_STATE_NAMES = ("delta_f", "delta_hbo", "delta_hb")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
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
    path.write_text(json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames: list[str] = []
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


def _features(patches: torch.Tensor, spectral_bins: int) -> torch.Tensor:
    n = patches.shape[-1]
    time = torch.linspace(-1.0, 1.0, n, dtype=patches.dtype, device=patches.device)
    slope = (patches * time).sum(-1) / time.square().sum().clamp_min(1e-8)
    mean = patches.mean(-1)
    std = patches.std(-1, unbiased=False)
    delta = patches[..., -1] - patches[..., 0]
    spectrum = torch.fft.rfft(patches, dim=-1).abs().clamp_min(1e-8).log()
    spectrum = spectrum[..., 1 : spectral_bins + 1]
    return torch.cat((mean, std, slope, delta, spectrum.flatten(start_dim=2)), dim=-1)


def _normalize_clean(clean: torch.Tensor, offset: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (clean - offset.unsqueeze(-1)) / scale.unsqueeze(-1)


def _correct_legacy_wavelength_clean(clean: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Apply the fixed observation mixing to caches written before the generator fix.

    The old source trace is affine in one chromophore coordinate.  Regressing
    that affine mapping recovers the stored normalization/Jacobian product and
    lets us evaluate the canonical wavelength mixture without rerunning the
    particle filter.  Concentration caches should bypass this function.
    """
    corrected = np.empty_like(clean, dtype=np.float64)
    mixtures = (state[:, 2] + 0.25 * state[:, 3], 0.35 * state[:, 2] + state[:, 3])
    coordinates = (state[:, 2], state[:, 3])
    for channel, (coordinate, mixture) in enumerate(zip(coordinates, mixtures)):
        design = np.column_stack((np.ones(len(coordinate)), coordinate))
        beta, *_ = np.linalg.lstsq(design, clean[:, channel], rcond=None)
        corrected[:, channel] = beta[0] + beta[1] * mixture
    return corrected


@dataclass
class Evidence:
    eeg_features: np.ndarray
    fnirs_features: np.ndarray
    eeg_target: np.ndarray
    fnirs_target: np.ndarray
    eeg_uncertainty: np.ndarray
    fnirs_uncertainty: np.ndarray
    subjects: np.ndarray
    patch_index: np.ndarray
    fnirs_history: np.ndarray
    eeg_history: np.ndarray
    fnirs_level: np.ndarray
    fnirs_innovation: np.ndarray
    physical_rows: list[dict[str, Any]]
    mask_rows: list[dict[str, Any]]
    overlay: dict[str, Any] | None


def _collect(loader: Iterable[Mapping[str, Any]], teacher_adapter: PhysicalStateTeacher, *, overlay: bool) -> Evidence:
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in (
        "eeg_features", "fnirs_features", "eeg_target", "fnirs_target",
        "eeg_uncertainty", "fnirs_uncertainty", "subjects", "patch_index",
        "fnirs_history", "eeg_history", "fnirs_level", "fnirs_innovation",
    )}
    physical_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    overlay_payload: dict[str, Any] | None = None

    for batch in loader:
        teacher = teacher_adapter(batch["teacher"])
        valid = teacher.valid_mask
        eeg_patches = _patchify(batch["eeg"], 400)
        fnirs_patches = _patchify(batch["fnirs"], 20)
        eeg_feature = _features(eeg_patches, 16)
        fnirs_feature = _features(fnirs_patches, 10)
        eeg_target = teacher.eeg_target[..., [0, 1, 3, 4]]
        fnirs_target = teacher.fnirs_target[..., :6]
        eeg_uncertainty = teacher.eeg_uncertainty[..., [0, 1, 3, 4]]
        fnirs_uncertainty = teacher.fnirs_uncertainty[..., :6]

        state = batch["teacher"]["state_mean"].reshape(batch["eeg"].shape[0], 10, 20, 5).mean(dim=2)
        driver = batch["teacher"]["neural_driver_eeg_rate"].reshape(
            batch["eeg"].shape[0], 10, 400, 1
        ).mean(dim=2)
        fnirs_state = state[..., 1:4]
        eeg_state = torch.cat((driver, state[..., 0:1]), dim=-1)
        subject_matrix = batch["subject_id"].view(-1, 1).expand_as(valid)
        patch_matrix = torch.arange(10).view(1, -1).expand_as(valid)

        for key, tensor in (
            ("eeg_features", eeg_feature), ("fnirs_features", fnirs_feature),
            ("eeg_target", eeg_target), ("fnirs_target", fnirs_target),
            ("eeg_uncertainty", eeg_uncertainty), ("fnirs_uncertainty", fnirs_uncertainty),
            ("subjects", subject_matrix), ("patch_index", patch_matrix),
        ):
            arrays[key].append(tensor[valid].detach().cpu().numpy())

        context_mask = teacher.context_valid_mask
        for sample in range(valid.shape[0]):
            for target_index in range(5, 10):
                if not bool(context_mask[sample, target_index]):
                    continue
                # The target patch needs complete causal history, while the
                # five history patches need valid cached physical states.
                if not bool(valid[sample, target_index - 5 : target_index].all()):
                    continue
                arrays["fnirs_history"].append(
                    fnirs_state[sample, target_index - 5 : target_index].reshape(1, -1).cpu().numpy()
                )
                arrays["eeg_history"].append(
                    eeg_state[sample, target_index - 5 : target_index].reshape(1, -1).cpu().numpy()
                )
                level = fnirs_state[sample, target_index]
                arrays["fnirs_level"].append(level.reshape(1, -1).cpu().numpy())
                arrays["fnirs_innovation"].append(
                    (level - fnirs_state[sample, target_index - 1]).reshape(1, -1).cpu().numpy()
                )

        eeg_clean = _normalize_clean(
            batch["teacher"]["eeg_clean_mean"],
            batch["normalization"]["eeg_offset"],
            batch["normalization"]["eeg_scale"],
        )
        fnirs_clean_raw = batch["teacher"]["fnirs_clean_mean"].detach().cpu().numpy().transpose(0, 2, 1)
        state_raw = batch["teacher"]["state_mean"].detach().cpu().numpy()
        corrected_raw = np.stack(
            [_correct_legacy_wavelength_clean(clean, states) for clean, states in zip(fnirs_clean_raw, state_raw)]
        )
        corrected_tensor = torch.from_numpy(corrected_raw.transpose(0, 2, 1)).to(batch["fnirs"].dtype)
        fnirs_clean = _normalize_clean(
            corrected_tensor,
            batch["normalization"]["fnirs_offset"],
            batch["normalization"]["fnirs_scale"],
        )
        eeg_clean_patch = _patchify(eeg_clean, 400)
        fnirs_clean_patch = _patchify(fnirs_clean, 20)
        for sample, subject in enumerate(batch["subject_id"]):
            for modality, observed, clean in (
                ("eeg", eeg_patches[sample], eeg_clean_patch[sample]),
                ("fnirs", fnirs_patches[sample], fnirs_clean_patch[sample]),
            ):
                mse_clean = (observed - clean).square().mean(dim=(-1, -2))
                mse_zero = observed.square().mean(dim=(-1, -2))
                history = torch.zeros_like(observed)
                history[1:] = observed[:-1]
                mse_history = (observed - history).square().mean(dim=(-1, -2))
                for patch in range(10):
                    if bool(valid[sample, patch]):
                        physical_rows.append({
                            "subject": int(subject), "patch": patch, "modality": modality,
                            "mse_clean": float(mse_clean[patch]), "mse_zero": float(mse_zero[patch]),
                            "mse_history": float(mse_history[patch]),
                            "clean_correction": "posthoc canonical wavelength mixing",
                        })
            mask_rows.append({
                "subject": int(subject), "local_valid_patches": int(valid[sample].sum()),
                "context_valid_patches": int(context_mask[sample].sum()),
                "total_patches": int(valid.shape[1]),
                "local_coverage": float(valid[sample].float().mean()),
                "context_coverage": float(context_mask[sample].float().mean()),
            })

        if overlay and overlay_payload is None:
            index = 0
            eeg_time = np.arange(batch["eeg"].shape[-1]) / 200.0
            fnirs_time = np.arange(batch["fnirs"].shape[-1]) / 10.0
            overlay_payload = {
                "subject": int(batch["subject_id"][index]),
                "eeg_time_s": eeg_time.tolist(),
                "fnirs_time_s": fnirs_time.tolist(),
                "eeg_observed_envelope": batch["eeg"][index].abs().mean(0).cpu().tolist(),
                "eeg_clean_envelope": eeg_clean[index].abs().mean(0).cpu().tolist(),
                "fnirs_observed": batch["fnirs"][index].T.cpu().tolist(),
                "fnirs_clean": fnirs_clean[index].T.cpu().tolist(),
                "state_mean": state_raw[index].tolist(),
                "state_time_s": fnirs_time.tolist(),
            }

    combined = {}
    for key, chunks in arrays.items():
        if not chunks:
            width = 0
            combined[key] = np.empty((0, width), dtype=np.float64)
        else:
            combined[key] = np.concatenate(chunks, axis=0)
    return Evidence(**combined, physical_rows=physical_rows, mask_rows=mask_rows, overlay=overlay_payload)


def _fit_coordinate(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    alphas: Sequence[float],
) -> tuple[StandardScaler, Ridge, float, float, np.ndarray]:
    x_scaler = StandardScaler().fit(train_x)
    train_scaled = x_scaler.transform(train_x)
    val_scaled = x_scaler.transform(val_x)
    baseline = float(np.sum(np.square(val_y - np.mean(train_y))))
    best = None
    for alpha in alphas:
        model = Ridge(alpha=float(alpha)).fit(train_scaled, train_y)
        prediction = model.predict(val_scaled)
        r2 = 1.0 - float(np.sum(np.square(val_y - prediction))) / max(baseline, 1e-12)
        if best is None or r2 > best[0]:
            best = (r2, model, float(alpha), prediction)
    assert best is not None
    return x_scaler, best[1], best[2], best[0], best[3]


def _coordinate_audit(
    train: Evidence,
    val: Evidence,
    *,
    alphas: Sequence[float],
    permutations: int,
    quantile: float,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    traces: dict[str, Any] = {}
    for modality, names in (("eeg", EEG_LOCAL_NAMES), ("fnirs", FNIRS_LOCAL_NAMES)):
        train_x = getattr(train, f"{modality}_features")
        val_x = getattr(val, f"{modality}_features")
        train_y = getattr(train, f"{modality}_target")
        val_y = getattr(val, f"{modality}_target")
        uncertainty = getattr(val, f"{modality}_uncertainty")
        for index, name in enumerate(names):
            scaler, model, alpha, r2, prediction = _fit_coordinate(
                train_x, train_y[:, index], val_x, val_y[:, index], alphas
            )
            null = []
            for _ in range(permutations):
                _, _, _, null_r2, _ = _fit_coordinate(
                    train_x, rng.permutation(train_y[:, index]), val_x, val_y[:, index], [alpha]
                )
                null.append(null_r2)
            threshold = float(np.quantile(null, quantile))
            error = prediction - val_y[:, index]
            sigma = np.sqrt(np.maximum(uncertainty[:, index], 1e-12))
            admitted = bool(r2 > max(0.0, threshold))
            rows.append({
                "modality": modality, "coordinate": name, "alpha": alpha, "validation_r2": r2,
                "permutation_q": threshold, "admitted_local_target": admitted,
                "posterior_interval_90_coverage": float(np.mean(np.abs(error) <= 1.645 * sigma)),
                "standardized_rmse": float(np.sqrt(np.mean(np.square(error / sigma)))),
            })
            keep = min(400, len(prediction))
            traces[f"{modality}:{name}"] = {
                "target": val_y[:keep, index].tolist(), "prediction": prediction[:keep].tolist(),
                "subject": val.subjects[:keep].tolist(),
            }
    return rows, traces


def _vocabulary_audit(
    train: Evidence,
    val: Evidence,
    *,
    codebook_size: int,
    random_references: int,
    quantile: float,
    rng: np.random.Generator,
    admitted: Mapping[str, Sequence[int]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    plot_data = {}
    for modality in ("eeg", "fnirs"):
        indices = list(admitted[modality])
        train_y = getattr(train, f"{modality}_target")[:, indices]
        val_y = getattr(val, f"{modality}_target")[:, indices]
        all_names = EEG_LOCAL_NAMES if modality == "eeg" else FNIRS_LOCAL_NAMES
        target_names = [all_names[index] for index in indices]
        scaler = StandardScaler().fit(train_y)
        train_z = scaler.transform(train_y)
        val_z = scaler.transform(val_y)
        if len(train_z) < codebook_size:
            raise ValueError(f"{modality} has {len(train_z)} training patches for K={codebook_size}")
        model = MiniBatchKMeans(
            n_clusters=codebook_size, random_state=int(rng.integers(0, 2**31 - 1)),
            batch_size=min(4096, len(train_z)), n_init=10,
        ).fit(train_z)
        codes = model.predict(val_z)
        reconstruction = model.cluster_centers_[codes]
        coordinate_r2 = 1.0 - np.sum(np.square(val_z - reconstruction), axis=0) / np.maximum(
            np.sum(np.square(val_z), axis=0), 1e-12
        )
        global_r2 = 1.0 - np.sum(np.square(val_z - reconstruction)) / max(np.sum(np.square(val_z)), 1e-12)
        random_r2 = []
        for _ in range(random_references):
            centers = train_z[rng.choice(len(train_z), size=codebook_size, replace=False)]
            distance = np.sum(np.square(val_z[:, None, :] - centers[None, :, :]), axis=-1)
            random_reconstruction = centers[np.argmin(distance, axis=1)]
            random_r2.append(1.0 - np.sum(np.square(val_z - random_reconstruction)) / max(np.sum(np.square(val_z)), 1e-12))
        threshold = float(np.quantile(random_r2, quantile))
        occupancy = np.bincount(codes, minlength=codebook_size)
        pca = PCA(n_components=min(2, train_z.shape[1]), random_state=0).fit(train_z)
        keep = min(3000, len(val_z))
        coordinates = pca.transform(val_z[:keep])
        if coordinates.shape[1] == 1:
            coordinates = np.column_stack((coordinates[:, 0], np.zeros(len(coordinates))))
        rows.append({
            "modality": modality, "codebook_size": codebook_size, "validation_global_r2": float(global_r2),
            "random_reference_q": threshold, "above_random_reference": bool(global_r2 > threshold),
            "active_codes": int(np.sum(occupancy > 0)),
            "perplexity": float(np.exp(-np.sum((occupancy[occupancy > 0] / occupancy.sum()) * np.log(occupancy[occupancy > 0] / occupancy.sum())))),
            "coordinate_r2": coordinate_r2.tolist(),
            "target_coordinates": target_names,
        })
        plot_data[modality] = {
            "pca": coordinates.tolist(), "codes": codes[:keep].tolist(), "occupancy": occupancy.tolist(),
            "coordinate_r2": coordinate_r2.tolist(),
            "target_coordinates": target_names,
        }
    return rows, plot_data


def _synthetic_posterior_calibration(config: Mapping[str, Any]) -> dict[str, Any]:
    module_path = REPO_ROOT / "croce_validation" / "scripts" / "run_local_neighborhood_solver_audit.py"
    spec = importlib.util.spec_from_file_location("e0_v2_solver_audit", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import solver audit from {module_path}")
    audit = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = audit
    spec.loader.exec_module(audit)
    synthetic = config["e0_v2"].get("synthetic_calibration", {})
    args = Namespace(
        duration_s=float(synthetic.get("duration_s", 10.0)), observation_fs=10.0, eeg_fs=200.0,
        integration_dt=0.005, snr_db=float(synthetic.get("snr_db", 5.0)),
        synthetic_eeg_channels=8, synthetic_fnirs_channels=4, eeg_neighbors=4, fnirs_neighbors=3,
        eeg_radius_mm=60.0, fnirs_radius_mm=45.0, eeg_sigma_mm=30.0, fnirs_sigma_mm=22.0,
        eeg_sign_mode="covariance", eeg_unit="uV", fnirs_primary_unit="a.u.", fnirs_secondary_unit="a.u.",
    )
    spatial = audit.SpatialConfig(
        eeg_neighbors=4, fnirs_neighbors=3, eeg_radius_mm=60.0, fnirs_radius_mm=45.0,
        eeg_sigma_mm=30.0, fnirs_sigma_mm=22.0, eeg_sign_mode="covariance",
    )
    bundle = audit.simulate_synthetic_bundle(args, spatial)
    seeds = tuple(int(value) for value in synthetic.get("seeds", [11, 23, 37, 47]))
    split = max(1, len(seeds) // 2)
    filter_config = audit.FilterConfig(
        integration_dt_s=1.0 / bundle.eeg_fs_hz, observation_fs_hz=bundle.fnirs_fs_hz,
        num_particles=int(synthetic.get("num_particles", 128)), resample_fraction=0.5,
        prior_std=np.asarray([0.05, 0.05, 0.05, 0.05, 0.0]),
        state_noise_std=np.asarray([0.02, 0.015, 0.015, 0.015, 0.0]),
        sigma_prop=0.35, sigma_nirs=1.0, seed_list=seeds, time_shift_null_s=2.0,
        run_spatial_null=False, solver_backend="torch_exact", torch_device="cpu",
    )
    results = [audit.run_particle_filter(bundle, filter_config, audit.ModelParams(), seed=seed) for seed in seeds]
    calibration_error = np.concatenate(
        [result["state_estimates"] - bundle.true_states for result in results[:split]], axis=0
    )
    calibration_var = np.concatenate([np.square(result["state_std"]) for result in results[:split]], axis=0)
    variance_scale = np.mean(np.square(calibration_error), axis=0) / np.maximum(np.mean(calibration_var, axis=0), 1e-12)
    variance_scale = np.clip(variance_scale, 1e-6, 1e8)
    rows = []
    evaluation_samples = sum(result["state_estimates"].shape[0] for result in results[split:])
    coverage_tolerance = 1.96 * np.sqrt(0.9 * 0.1 / max(evaluation_samples, 1))
    for index, name in enumerate(("s", "delta_f", "delta_hbo", "delta_hb", "r")):
        error = np.concatenate(
            [result["state_estimates"][:, index] - bundle.true_states[:, index] for result in results[split:]]
        )
        standard = np.concatenate([result["state_std"][:, index] for result in results[split:]])
        unscaled_coverage = float(np.mean(np.abs(error) <= 1.645 * np.maximum(standard, 1e-8)))
        scaled_standard = np.maximum(standard * np.sqrt(variance_scale[index]), 1e-8)
        scaled_coverage = float(np.mean(np.abs(error) <= 1.645 * scaled_standard))
        coverage_error = abs(scaled_coverage - 0.9)
        rows.append({
            "coordinate": name, "variance_scale_from_calibration_seeds": float(variance_scale[index]),
            "unscaled_90_coverage": unscaled_coverage, "scaled_90_coverage": scaled_coverage,
            "scaled_standardized_rmse": float(np.sqrt(np.mean(np.square(error / scaled_standard)))),
            "coverage_abs_error": coverage_error,
            "coverage_tolerance_95": float(coverage_tolerance),
            "calibrated_coverage_pass": bool(coverage_error <= coverage_tolerance),
        })
    unscaled_error = float(np.mean([abs(row["unscaled_90_coverage"] - 0.9) for row in rows]))
    scaled_error = float(np.mean([abs(row["scaled_90_coverage"] - 0.9) for row in rows]))
    return {
        "schema": "e0_v2_synthetic_posterior_calibration",
        "calibration_seeds": list(seeds[:split]), "validation_seeds": list(seeds[split:]),
        "rows": rows, "mean_abs_coverage_error_unscaled": unscaled_error,
        "mean_abs_coverage_error_scaled": scaled_error,
        "calibration_improves_heldout_coverage": bool(scaled_error < unscaled_error),
        "evaluation_samples_per_coordinate": evaluation_samples,
        "posterior_calibration_pass": bool(all(row["calibrated_coverage_pass"] for row in rows)),
    }


def _context_audit(
    train: Evidence,
    val: Evidence,
    *,
    alphas: Sequence[float],
    shuffles: int,
    quantile: float,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    traces = {}
    baseline_x_train = train.fnirs_history
    baseline_x_val = val.fnirs_history
    full_x_train = np.concatenate((train.fnirs_history, train.eeg_history), axis=1)
    full_x_val = np.concatenate((val.fnirs_history, val.eeg_history), axis=1)
    for target_name in ("fnirs_level", "fnirs_innovation"):
        train_y = getattr(train, target_name)
        val_y = getattr(val, target_name)
        baseline_predictions = []
        full_predictions = []
        for coordinate in range(train_y.shape[1]):
            _, _, _, baseline_r2, baseline_prediction = _fit_coordinate(
                baseline_x_train, train_y[:, coordinate], baseline_x_val, val_y[:, coordinate], alphas
            )
            _, _, _, full_r2, full_prediction = _fit_coordinate(
                full_x_train, train_y[:, coordinate], full_x_val, val_y[:, coordinate], alphas
            )
            baseline_predictions.append(baseline_prediction)
            full_predictions.append(full_prediction)
            rows.append({
                "target": target_name, "coordinate": FNIRS_STATE_NAMES[coordinate],
                "fnirs_history_r2": baseline_r2, "plus_eeg_history_r2": full_r2,
                "incremental_r2": full_r2 - baseline_r2,
            })
        baseline_prediction = np.column_stack(baseline_predictions)
        full_prediction = np.column_stack(full_predictions)
        residual_base = val_y - baseline_prediction
        residual_full = val_y - full_prediction
        cov_base = np.cov(residual_base, rowvar=False) + np.eye(val_y.shape[1]) * 1e-8
        cov_full = np.cov(residual_full, rowvar=False) + np.eye(val_y.shape[1]) * 1e-8
        gain = 0.5 * float(np.linalg.slogdet(cov_base)[1] - np.linalg.slogdet(cov_full)[1])
        null = []
        for _ in range(shuffles):
            shuffled_train = np.concatenate((train.fnirs_history, train.eeg_history[rng.permutation(len(train.eeg_history))]), axis=1)
            shuffled_predictions = []
            for coordinate in range(train_y.shape[1]):
                _, _, _, _, prediction = _fit_coordinate(
                    shuffled_train, train_y[:, coordinate], full_x_val, val_y[:, coordinate], [10.0]
                )
                shuffled_predictions.append(prediction)
            residual = val_y - np.column_stack(shuffled_predictions)
            covariance = np.cov(residual, rowvar=False) + np.eye(val_y.shape[1]) * 1e-8
            null.append(0.5 * float(np.linalg.slogdet(cov_base)[1] - np.linalg.slogdet(covariance)[1]))
        threshold = float(np.quantile(null, quantile))
        rows.append({
            "target": target_name, "coordinate": "joint_logdet",
            "conditional_information_nats": gain, "shuffled_eeg_q": threshold,
            "above_shuffled_eeg": bool(gain > max(0.0, threshold)),
        })
        keep = min(400, len(val_y))
        traces[target_name] = {
            "target": val_y[:keep].tolist(), "fnirs_history_prediction": baseline_prediction[:keep].tolist(),
            "plus_eeg_history_prediction": full_prediction[:keep].tolist(),
        }
    return rows, traces


def _physical_audit(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for modality in ("eeg", "fnirs"):
        selected = [row for row in rows if row["modality"] == modality]
        clean = np.asarray([row["mse_clean"] for row in selected])
        zero = np.asarray([row["mse_zero"] for row in selected])
        history = np.asarray([row["mse_history"] for row in selected])
        baseline_name, baseline = ("zero", zero) if zero.mean() <= history.mean() else ("history", history)
        gains = baseline - clean
        subjects = np.asarray([row["subject"] for row in selected])
        per_subject = [float(gains[subjects == subject].mean()) for subject in np.unique(subjects)]
        output.append({
            "modality": modality, "baseline": baseline_name, "clean_mse": float(clean.mean()),
            "baseline_mse": float(baseline.mean()), "mean_gain": float(np.mean(per_subject)),
            "positive_subject_fraction": float(np.mean(np.asarray(per_subject) > 0)),
            "subjects": len(per_subject),
        })
    return output


def _hash_inputs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in config["data"].get("cache_sources", []):
        root = REPO_ROOT / source["root"]
        for path in sorted(list(root.glob("subject_*/cache_manifest.json")) + list(root.glob("subject_*/*.npz"))):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append({"path": str(path.relative_to(REPO_ROOT)), "sha256": digest, "size_bytes": path.stat().st_size})
    return rows


def run(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    e0 = config["e0_v2"]
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
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir()
    (run_dir / "figure_data").mkdir()
    (run_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    protocol = {
        "schema": SCHEMA,
        "created_before_protected_test_access": True,
        "protected_test_default": "closed",
        "measurement_gate": "finite auditable transforms plus exact crop-position invariance; scale comparison requires visual review",
        "local_target_gate": "validation R2 exceeds zero and the frozen label-permutation quantile",
        "vocabulary_gate": "K=128 target geometry reconstruction exceeds random-centroid quantile",
        "coupling_gate": "continuous conditional log-determinant gain exceeds zero and shuffled-EEG quantile",
        "physical_observation_gate": "clean prediction improves the selected zero/history baseline",
        "visual_gate": "all registered figures reviewed against checklist before protected test can open",
        "null_quantile": float(e0["null_quantile"]),
    }
    (run_dir / "decision_protocol.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    _write_json(run_dir / "metric_registry.json", {
        "schema": SCHEMA,
        "primary": ["local_target_validation_r2", "physical_observation_gain", "continuous_conditional_information_nats"],
        "gate": ["measurement_contract", "finite_vocabulary_transmissibility", "visual_review"],
        "diagnostic": ["posterior_interval_coverage", "canonical_scale_ratio", "mask_coverage", "coordinate_incremental_r2"],
    })

    measurement = run_measurement_audit(config, run_dir / "figure_data")
    _write_csv(run_dir / "unit_scale_audit.csv", measurement["summary_rows"])
    posterior_calibration = _synthetic_posterior_calibration(config)
    _write_csv(run_dir / "posterior_calibration.csv", posterior_calibration["rows"])
    _write_json(run_dir / "figure_data" / "posterior_calibration.json", posterior_calibration)
    dataloaders = create_configured_multimodal_dataloaders(config)
    adapter = PhysicalStateTeacher()
    train = _collect(dataloaders["train"], adapter, overlay=False)
    val = _collect(dataloaders["val"], adapter, overlay=True)
    alphas = [float(value) for value in e0["ridge_alphas"]]
    coordinate_rows, observability_traces = _coordinate_audit(
        train, val, alphas=alphas, permutations=int(e0["permutation_iterations"]),
        quantile=float(e0["null_quantile"]), rng=rng,
    )
    admitted_indices = {
        modality: [
            index for index, name in enumerate(EEG_LOCAL_NAMES if modality == "eeg" else FNIRS_LOCAL_NAMES)
            if any(
                row["modality"] == modality and row["coordinate"] == name and row["admitted_local_target"]
                for row in coordinate_rows
            )
        ]
        for modality in ("eeg", "fnirs")
    }
    vocabulary_rows, vocabulary_plot = _vocabulary_audit(
        train, val, codebook_size=int(e0["codebook_size"]),
        random_references=int(e0["random_reference_iterations"]),
        quantile=float(e0["null_quantile"]), rng=rng,
        admitted=admitted_indices,
    )
    context_rows, context_traces = _context_audit(
        train, val, alphas=alphas, shuffles=int(e0["context_shuffle_iterations"]),
        quantile=float(e0["null_quantile"]), rng=rng,
    )
    physical_rows = _physical_audit(val.physical_rows)

    _write_csv(run_dir / "local_target_observability.csv", coordinate_rows)
    _write_csv(run_dir / "vocabulary_transmissibility.csv", vocabulary_rows)
    _write_csv(run_dir / "continuous_coupling_upper_bound.csv", context_rows)
    _write_csv(run_dir / "physical_observation_checks.csv", physical_rows)
    _write_csv(run_dir / "teacher_mask_coverage.csv", val.mask_rows)
    _write_csv(run_dir / "mask_coverage.csv", val.mask_rows)
    _write_json(run_dir / "target_contract.json", {
        "schema": SCHEMA,
        "local_state_projection": {
            "eeg": [EEG_LOCAL_NAMES[index] for index in admitted_indices["eeg"]],
            "fnirs": [FNIRS_LOCAL_NAMES[index] for index in admitted_indices["fnirs"]],
        },
        "context_transition_target": ["fnirs_level", "fnirs_innovation"],
        "posterior_covariance_role": "uncertainty weighting after synthetic variance calibration",
        "physical_observation_mean_role": "semantic decoder target; validation currently evaluated separately",
        "local_validity": "cache_valid_mask",
        "context_validity": "cache_valid_mask AND causal_valid_mask AND tokenizer context_valid_mask",
        "protected_test_status": "closed",
    })
    _write_json(run_dir / "figure_data" / "target_observability.json", {
        "rows": coordinate_rows, "traces": observability_traces,
    })
    _write_json(run_dir / "figure_data" / "vocabulary_transmissibility.json", {
        "rows": vocabulary_rows, "plot": vocabulary_plot,
    })
    _write_json(run_dir / "figure_data" / "continuous_coupling_upper_bound.json", {
        "rows": context_rows, "traces": context_traces,
    })
    _write_json(run_dir / "figure_data" / "physical_teacher_overlay.json", val.overlay or {})
    _write_json(run_dir / "figure_data" / "physical_observation_checks.json", {"rows": physical_rows})
    _write_json(run_dir / "cache_manifest_hashes.json", {"files": _hash_inputs(config)})
    _write_json(run_dir / "evidence_calibration.json", {
        "schema": SCHEMA,
        "label_permutation_iterations": int(e0["permutation_iterations"]),
        "random_centroid_iterations": int(e0["random_reference_iterations"]),
        "context_shuffle_iterations": int(e0["context_shuffle_iterations"]),
        "null_quantile": float(e0["null_quantile"]),
        "synthetic_posterior_calibration": posterior_calibration,
        "protected_test_used": False,
    })

    local_pass = all(
        any(row["modality"] == modality and row["admitted_local_target"] for row in coordinate_rows)
        for modality in ("eeg", "fnirs")
    )
    vocabulary_pass = all(row["above_random_reference"] for row in vocabulary_rows)
    coupling_pass = any(
        row.get("coordinate") == "joint_logdet" and row.get("above_shuffled_eeg") for row in context_rows
    )
    physical_pass = all(row["mean_gain"] > 0 for row in physical_rows)
    uncertainty_pass = bool(posterior_calibration["posterior_calibration_pass"])
    measurement_pass = bool(
        measurement["crop_position_invariance_max_abs_delta"] <= 1e-12
        and all(row["finite_fraction"] == 1.0 for row in measurement["summary_rows"])
    )
    machine_validation_pass = bool(
        measurement_pass and local_pass and vocabulary_pass and coupling_pass and physical_pass and uncertainty_pass
    )
    summary = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "validation": {
            "measurement_contract_pass": measurement_pass,
            "local_target_observability_pass": local_pass,
            "finite_vocabulary_transmissibility_pass": vocabulary_pass,
            "continuous_coupling_upper_bound_pass": coupling_pass,
            "physical_observation_pass": physical_pass,
            "posterior_uncertainty_calibration_pass": uncertainty_pass,
            "machine_validation_pass": machine_validation_pass,
            "visual_review": "pending",
        },
        "protected_test": {"opened": False, "reason": "requires machine validation and completed visual review"},
        "measurement": {
            "canonical_scale_max_ratio": measurement["validation_canonical_scale_max_ratio"],
            "crop_position_max_abs_delta": measurement["crop_position_invariance_max_abs_delta"],
        },
        "local_targets": coordinate_rows,
        "vocabulary": vocabulary_rows,
        "continuous_coupling": context_rows,
        "physical_observation": physical_rows,
        "posterior_calibration": posterior_calibration,
        "sample_counts": {
            "train_local_patches": int(len(train.subjects)), "validation_local_patches": int(len(val.subjects)),
            "train_context_rows": int(len(train.fnirs_level)), "validation_context_rows": int(len(val.fnirs_level)),
        },
    }
    _write_json(run_dir / "summary.json", summary)

    # Untracked research checkouts outside this experiment do not change the
    # executable provenance.  Track committed-file modifications separately;
    # cache/config inputs are already covered by explicit hashes.
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip())
    manifest = {
        "schema": SCHEMA,
        "command": " ".join(sys.argv),
        "start_policy": "validation-only; protected test closed",
        "completion_status": "validation_complete",
        "git_commit": summary["git_commit"],
        "dirty_worktree": dirty,
        "seed": seed,
        "config": "config.yaml",
        "cache_hash_inventory": "cache_manifest_hashes.json",
        "decision_protocol": "decision_protocol.yaml",
        "metric_registry": "metric_registry.json",
        "evidence_calibration": "evidence_calibration.json",
    }
    _write_json(run_dir / "manifest.json", manifest)
    resource_path = REPO_ROOT / ".claude_resources.json"
    if resource_path.is_file():
        (run_dir / "environment_resources.json").write_bytes(resource_path.read_bytes())

    visualizer = REPO_ROOT / "experiments" / "scripts" / "visualize_e0_v2_audit.py"
    subprocess.run([sys.executable, str(visualizer), "--run-dir", str(run_dir)], cwd=REPO_ROOT, check=True)
    print(run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
