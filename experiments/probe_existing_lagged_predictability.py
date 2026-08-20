#!/usr/bin/env python3
"""Estimate offline delayed EEG/fNIRS representation association.

This is deliberately a post-selection development analysis, not a fresh
fit-to-held-out or causal/future probe.  The legacy continuous shared/private
``best.pt`` was selected using development validation loss (subjects 19--23),
and the full run discarded fit latents.  Therefore this evaluator derives fit
latents by an eval-only forward pass over fit ``source_samples.npz`` with the
declared frozen checkpoint; it never fits on validation latents.  Validation
shared latents are read from ``validation_predictions.npz``.  Native patch
features are extracted from canonical robust-SD signals (validation raw
observations are reconstructed with fit-only normalization statistics).

Lag convention
--------------
``lag_seconds`` is target time minus source time.  Thus a positive lag aligns
EEG token ``t`` to the later fNIRS token ``t + lag_tokens`` for an offline
delayed association, while a negative lag aligns a later EEG token to an
earlier fNIRS token.  Negative lags are never silently clipped or wrapped; the
output carries both ``negative_lag`` and ``lag_direction``.  Because the saved
encoder has bidirectional full-window context and the checkpoint was selected
on these development subjects, no row is a causal/future claim.

The evaluator uses no ``target``, ``eeg_driver`` or ``fnirs_driver`` arrays;
``target_mask`` is used only as a validity mask.  It writes a new output
directory atomically, never writes into the input continuous shared/private
run, and refuses ``protected_open=true``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

try:  # Torch is only needed for the real full-run path.
    import torch
except Exception:  # pragma: no cover - fixture-only environments may omit torch.
    torch = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.physiological_patch_features import (  # noqa: E402
    extract_eeg_patch_features,
    extract_fnirs_patch_features,
)


SCHEMA = "existing_lagged_predictability_probe_v1"
CELL_SCHEMA = "existing_lagged_predictability_cell_v1"
SOURCE_SCHEMA = "continuous_shared_private_source_v1"
VALIDATION_SCHEMA = "continuous_shared_private_predictions_v1"
CLAIM_STATUS = "post_selection_development_offline_delayed_association"
CHECKPOINT_SELECTION_STATUS = "legacy_best_pt_selected_on_development_validation_subjects19_23"
EVALUATOR_TEMPORAL_MODE = "offline_same_window"
FORBIDDEN_MODEL_FIELDS = ("target", "eeg_driver", "fnirs_driver")
LAG_SECONDS = (-4, -2, 0, 2, 4, 6, 8, 10)
LAG_TOKENS = tuple(value // 2 for value in LAG_SECONDS)
# Descriptive aliases used by downstream audit code.
LAG_BANK_SECONDS = LAG_SECONDS
LAG_BANK_TOKENS = LAG_TOKENS
TOKEN_STEP_SECONDS = 2
REPRESENTATIONS = (
    "eeg_shared_to_fnirs_shared",
    "eeg_native_patch_to_fnirs_native_patch",
)
CONDITIONS = (
    "matched",
    "deranged_same_subject_same_condition_nonidentity",
    "within_trial_circular_shift",
)
NULL_POLICIES = {
    "matched": "matched_same_trial",
    "deranged_same_subject_same_condition_nonidentity": "same_subject_same_condition_nonidentity_trial_donor",
}


def _null_policy(condition: str, circular_shift_tokens: int) -> str:
    if condition in NULL_POLICIES:
        return NULL_POLICIES[condition]
    return f"within_trial_circular_shift_tokens_{int(circular_shift_tokens)}"


# ---------------------------------------------------------------------------
# Small serialization and validation helpers


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
    rows = list(rows)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: _jsonable(row.get(key, "")) for key in fields} for row in rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _subject_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Subject-equal lag summary; token pairs never enter this aggregation."""
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    keys = ("task_id", "dataset_id", "seed", "representation", "condition", "lag_tokens", "lag_seconds")
    for row in rows:
        groups.setdefault(tuple(row.get(key, "") for key in keys), []).append(row)
    output: list[dict[str, Any]] = []
    for values, grouped in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        delta = np.asarray([float(row["delta_r2"]) for row in grouped], dtype=float)
        prediction = np.asarray([float(row["proper_prediction_score"]) for row in grouped], dtype=float)
        finite_delta = np.isfinite(delta)
        finite_prediction = np.isfinite(prediction)
        output.append(
            {
                "schema": SCHEMA,
                **dict(zip(keys, values, strict=True)),
                "negative_lag": bool(int(values[-2]) < 0),
                "claim_status": ";".join(sorted({str(row.get("claim_status", CLAIM_STATUS)) for row in grouped})),
                "checkpoint_selection_status": ";".join(sorted({str(row.get("checkpoint_selection_status", CHECKPOINT_SELECTION_STATUS)) for row in grouped})),
                "selection_data": "fit_only_fixed_probe_alpha",
                "fresh_fit_held_out": False,
                "causal_future_claim": False,
                "evaluator_temporal_mode": EVALUATOR_TEMPORAL_MODE,
                "null_policy": ";".join(sorted({str(row.get("null_policy", "")) for row in grouped})),
                "pair_mask_sha256": ";".join(sorted({str(row.get("pair_mask_sha256", "")) for row in grouped})),
                "evaluation_unit": "subject_equal_mean",
                "subject_count": len(grouped),
                "supported_subject_count": int(finite_delta.sum()),
                "positive_subject_count": int(np.sum(delta[finite_delta] > 0)),
                "mean_delta_r2": float(np.mean(delta[finite_delta])) if finite_delta.any() else float("nan"),
                "mean_prediction_score": float(np.mean(prediction[finite_prediction])) if finite_prediction.any() else float("nan"),
                "status": "ok" if finite_delta.any() else "skipped",
            }
        )
    return output


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "manifest.json" and path.parent == root:
            continue
        output.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return output


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the independent probe contract before opening any data."""

    source = config.get("source", {})
    protected = source.get("protected_open", config.get("protected_open"))
    if protected is not False:
        raise PermissionError("existing lag probe requires protected_open=false")
    lag = config.get("lag_bank_seconds", config.get("lags_seconds"))
    if tuple(int(value) for value in (lag or ())) != LAG_SECONDS:
        raise ValueError(f"lag bank must be exactly {LAG_SECONDS}")
    if int(config.get("token_step_seconds", TOKEN_STEP_SECONDS)) != TOKEN_STEP_SECONDS:
        raise ValueError("token step must be exactly 2 seconds")
    governance = config.get("governance", {})
    if governance:
        if governance.get("causal_future_claim") is not False:
            raise ValueError("governance forbids causal/future claims")
        if governance.get("fresh_fit_held_out") is not False:
            raise ValueError("governance forbids fresh fit-to-held-out wording")
        if tuple(governance.get("forbidden_model_fields_not_used", ())) != FORBIDDEN_MODEL_FIELDS:
            raise ValueError("governance forbidden model fields differ from the no-driver contract")
        if governance.get("evaluator_temporal_mode") != EVALUATOR_TEMPORAL_MODE:
            raise ValueError("governance evaluator temporal mode must be offline_same_window")
    probe = config.get("probe", {})
    if str(probe.get("method", "ridge")).lower() != "ridge":
        raise ValueError("the first implementation supports ridge only")
    if float(probe.get("alpha", 1.0)) <= 0:
        raise ValueError("ridge alpha must be positive")
    if int(probe.get("components", 1)) <= 0:
        raise ValueError("ridge components must be positive")
    conditions = tuple(config.get("evaluation_conditions", CONDITIONS))
    if conditions != CONDITIONS:
        raise ValueError(f"evaluation conditions must be exactly {CONDITIONS}")
    shift = int(config.get("nulls", {}).get("within_trial_circular_shift_tokens", 1))
    if shift == 0:
        raise ValueError("within-trial circular shift must be nonzero")
    feature = config.get("features", {})
    for modality in ("eeg", "fnirs"):
        rate = float(feature.get("sample_rate_hz", {}).get(modality, 0.0))
        patch = int(feature.get("patch_samples", {}).get(modality, 0))
        if rate <= 0 or patch <= 0:
            raise ValueError(f"positive {modality} sample rate and patch size are required")


# ---------------------------------------------------------------------------
# Canonical arrays and synthetic fixture support


@dataclass
class ArrayBundle:
    """One fit or development split with token and feature masks retained."""

    eeg_shared: np.ndarray
    fnirs_shared: np.ndarray
    eeg_native: np.ndarray
    fnirs_native: np.ndarray
    eeg_shared_mask: np.ndarray
    fnirs_shared_mask: np.ndarray
    eeg_native_mask: np.ndarray
    fnirs_native_mask: np.ndarray
    subject: np.ndarray
    condition: np.ndarray
    sample_id: np.ndarray
    role: str
    target_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in (
            "eeg_shared",
            "fnirs_shared",
            "eeg_native",
            "fnirs_native",
            "eeg_shared_mask",
            "fnirs_shared_mask",
            "eeg_native_mask",
            "fnirs_native_mask",
            "subject",
            "condition",
            "sample_id",
        ):
            setattr(self, name, np.asarray(getattr(self, name)))
        if self.target_mask is None:
            self.target_mask = np.asarray(self.eeg_shared_mask, dtype=bool) & np.asarray(self.fnirs_shared_mask, dtype=bool)
        else:
            self.target_mask = np.asarray(self.target_mask, dtype=bool)
        n = len(self.subject)
        if any(len(getattr(self, name)) != n for name in (
            "eeg_shared",
            "fnirs_shared",
            "eeg_native",
            "fnirs_native",
            "eeg_shared_mask",
            "fnirs_shared_mask",
            "eeg_native_mask",
            "fnirs_native_mask",
            "target_mask",
            "condition",
            "sample_id",
        )):
            raise ValueError(f"{self.role} arrays do not have a common trial axis")
        if self.eeg_shared.ndim != 3 or self.fnirs_shared.ndim != 3:
            raise ValueError("shared arrays must be [trial, token, dimension]")
        if self.eeg_native.ndim != 3 or self.fnirs_native.ndim != 3:
            raise ValueError("native arrays must be [trial, token, feature]")
        token_count = self.eeg_shared.shape[1]
        if self.fnirs_shared.shape[1] != token_count or self.eeg_native.shape[1] != token_count or self.fnirs_native.shape[1] != token_count:
            raise ValueError("all representations must use the same token count")
        for name in ("eeg_shared_mask", "fnirs_shared_mask"):
            mask = getattr(self, name)
            if mask.shape != (n, token_count):
                raise ValueError(f"{name} must be [trial, token]")
        for name, dim in (("eeg_native_mask", self.eeg_native.shape[-1]), ("fnirs_native_mask", self.fnirs_native.shape[-1])):
            mask = getattr(self, name)
            if mask.shape != (n, token_count, dim):
                raise ValueError(f"{name} must be [trial, token, feature]")
        if self.target_mask.shape != (n, token_count):
            raise ValueError("target_mask must be [trial, token]")
        self.subject = self.subject.astype(str)
        self.condition = self.condition.astype(str)
        self.sample_id = self.sample_id.astype(str)
        if len(set(self.sample_id.tolist())) != n:
            raise ValueError(f"{self.role} sample_id values must be unique")
        self.eeg_shared_mask = self.eeg_shared_mask.astype(bool)
        self.fnirs_shared_mask = self.fnirs_shared_mask.astype(bool)
        self.eeg_native_mask = self.eeg_native_mask.astype(bool)
        self.fnirs_native_mask = self.fnirs_native_mask.astype(bool)
        self.target_mask = self.target_mask.astype(bool)

    @property
    def token_count(self) -> int:
        return int(self.eeg_shared.shape[1])

    @property
    def count(self) -> int:
        return int(len(self.subject))


def _token_mask(mask: np.ndarray, count: int, tokens: int) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    # The continuous runner stores SSM target validity at [trial, token, point].
    # A latent/native token is admitted only when all target points are valid.
    if value.ndim == 3 and value.shape[:2] == (count, tokens):
        value = value.all(axis=-1)
    if value.shape == (count, tokens):
        return value
    raise ValueError(f"token mask must have shape {(count, tokens)}, got {value.shape}")


def _feature_mask(values: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    values = np.asarray(values)
    finite = np.isfinite(values)
    if mask is None:
        return finite
    value = np.asarray(mask, dtype=bool)
    if value.shape != values.shape:
        raise ValueError(f"feature mask shape {value.shape} differs from values {values.shape}")
    return value & finite


def make_synthetic_fixture(seed: int = 20260822) -> dict[str, ArrayBundle]:
    """Return a deterministic fixture with valid same-subject/condition groups.

    The fixture intentionally has a one-token positive relationship so that a
    smoke run exercises the negative-lag and later-token offline-association
    code paths without
    touching a checkpoint or real data.
    """

    rng = np.random.default_rng(seed)
    n_fit = 16
    n_validation = 16
    tokens = 10
    shared_x_dim, shared_y_dim = 6, 4
    native_x_dim, native_y_dim = 8, 5

    def make_split(count: int, role: str, offset: int) -> ArrayBundle:
        subjects = np.asarray([f"S{index // 8 + 1}" for index in range(count)], dtype=str)
        conditions = np.asarray([f"condition_{(index // 2) % 4}" for index in range(count)], dtype=str)
        sample_ids = np.asarray([f"{role}-{offset + index:04d}" for index in range(count)], dtype=str)
        eeg_shared = rng.normal(size=(count, tokens, shared_x_dim)).astype(np.float32)
        fnirs_shared = rng.normal(scale=0.05, size=(count, tokens, shared_y_dim)).astype(np.float32)
        eeg_native = rng.normal(size=(count, tokens, native_x_dim)).astype(np.float32)
        fnirs_native = rng.normal(scale=0.05, size=(count, tokens, native_y_dim)).astype(np.float32)
        # Future fNIRS carries a low-dimensional linear copy of current EEG.
        fnirs_shared[:, 1:, :] += 0.75 * eeg_shared[:, :-1, :shared_y_dim]
        fnirs_native[:, 1:, :] += 0.65 * eeg_native[:, :-1, :native_y_dim]
        shared_mask = np.ones((count, tokens), dtype=bool)
        native_x_mask = np.ones((count, tokens, native_x_dim), dtype=bool)
        native_y_mask = np.ones((count, tokens, native_y_dim), dtype=bool)
        if role == "development_validation":
            shared_mask[0, -1] = False
            native_x_mask[0, -1] = False
            native_y_mask[0, -1] = False
        return ArrayBundle(
            eeg_shared=eeg_shared,
            fnirs_shared=fnirs_shared,
            eeg_native=eeg_native,
            fnirs_native=fnirs_native,
            eeg_shared_mask=shared_mask.copy(),
            fnirs_shared_mask=shared_mask.copy(),
            eeg_native_mask=native_x_mask,
            fnirs_native_mask=native_y_mask,
            subject=subjects,
            condition=conditions,
            sample_id=sample_ids,
            role=role,
        )

    return {
        "fit": make_split(n_fit, "fit", 0),
        "validation": make_split(n_validation, "development_validation", 1000),
    }


def _npz_mapping(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _fixture_arrays(path: Path) -> dict[str, np.ndarray]:
    """Load either one combined fixture NPZ or fit/validation NPZ files."""

    if path.is_file():
        return _npz_mapping(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    combined = path / "fixture.npz"
    if combined.exists():
        return _npz_mapping(combined)
    candidates = ((path / "fit_export.npz", "fit"), (path / "validation_export.npz", "validation"))
    merged: dict[str, np.ndarray] = {}
    found = False
    for candidate, prefix in candidates:
        if candidate.exists():
            found = True
            merged.update({f"{prefix}_{key}": value for key, value in _npz_mapping(candidate).items()})
    if found:
        return merged
    raise FileNotFoundError(
        f"fixture {path} must be fixture.npz or contain fit_export.npz and validation_export.npz"
    )


def _bundle_from_mapping(mapping: Mapping[str, np.ndarray], prefix: str, role: str) -> ArrayBundle:
    def get(name: str, required: bool = True) -> np.ndarray | None:
        key = f"{prefix}_{name}"
        if key in mapping:
            return np.asarray(mapping[key])
        if required:
            raise KeyError(f"fixture lacks {key}")
        return None

    eeg_shared = get("eeg_shared")
    fnirs_shared = get("fnirs_shared")
    eeg_native = get("eeg_native", required=False)
    fnirs_native = get("fnirs_native", required=False)
    if eeg_native is None or fnirs_native is None:
        raise KeyError("fixture must provide precomputed eeg_native and fnirs_native arrays")
    count, tokens = eeg_shared.shape[:2]
    eeg_shared_mask = get("eeg_shared_mask", required=False)
    fnirs_shared_mask = get("fnirs_shared_mask", required=False)
    eeg_native_mask = get("eeg_native_mask", required=False)
    fnirs_native_mask = get("fnirs_native_mask", required=False)
    target_mask = get("target_mask", required=False)
    if eeg_shared_mask is None:
        eeg_shared_mask = np.isfinite(eeg_shared).all(axis=-1)
    if fnirs_shared_mask is None:
        fnirs_shared_mask = np.isfinite(fnirs_shared).all(axis=-1)
    if eeg_native_mask is None:
        eeg_native_mask = np.isfinite(eeg_native)
    if fnirs_native_mask is None:
        fnirs_native_mask = np.isfinite(fnirs_native)
    subject = get("subject")
    condition = get("condition")
    sample_id = get("sample_id")
    return ArrayBundle(
        eeg_shared=eeg_shared,
        fnirs_shared=fnirs_shared,
        eeg_native=eeg_native,
        fnirs_native=fnirs_native,
        eeg_shared_mask=_token_mask(eeg_shared_mask, count, tokens),
        fnirs_shared_mask=_token_mask(fnirs_shared_mask, count, tokens),
        eeg_native_mask=_feature_mask(eeg_native, eeg_native_mask),
        fnirs_native_mask=_feature_mask(fnirs_native, fnirs_native_mask),
        subject=subject,
        condition=condition,
        sample_id=sample_id,
        role=role,
        target_mask=(
            _token_mask(target_mask, count, tokens)
            if target_mask is not None
            else np.asarray(eeg_shared_mask, dtype=bool) & np.asarray(fnirs_shared_mask, dtype=bool)
        ),
    )


def load_fixture(path: Path) -> dict[str, ArrayBundle]:
    mapping = _fixture_arrays(path)
    if "protected_open" in mapping and bool(np.asarray(mapping["protected_open"]).item()) is not False:
        raise PermissionError("synthetic fixture declares protected_open=true")
    return {
        "fit": _bundle_from_mapping(mapping, "fit", "fit"),
        "validation": _bundle_from_mapping(mapping, "validation", "development_validation"),
    }


# ---------------------------------------------------------------------------
# Existing full-run ingestion (checkpoint forward pass only, never training)


def _normalise_signal(signal: np.ndarray, mask: np.ndarray, mean: Sequence[float], scale: Sequence[float], patch: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    channel_mean = np.asarray(mean, dtype=np.float32)[None, :, None]
    channel_scale = np.asarray(scale, dtype=np.float32)[None, :, None]
    point = np.repeat(np.asarray(mask, dtype=bool), patch, axis=1)[:, None, :]
    result = np.where(point, (signal - channel_mean) / channel_scale, 0.0)
    if not np.all(np.isfinite(result)):
        raise ValueError("train-normalized signal contains non-finite values")
    return result.astype(np.float32)


def _denormalise_signal(signal: np.ndarray, mask: np.ndarray, mean: Sequence[float], scale: Sequence[float], patch: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    channel_mean = np.asarray(mean, dtype=np.float32)[None, :, None]
    channel_scale = np.asarray(scale, dtype=np.float32)[None, :, None]
    point = np.repeat(np.asarray(mask, dtype=bool), patch, axis=1)[:, None, :]
    result = np.where(point, signal * channel_scale + channel_mean, 0.0)
    if not np.all(np.isfinite(result)):
        raise ValueError("reconstructed canonical signal contains non-finite values")
    return result.astype(np.float32)


def _extract_native(
    eeg: np.ndarray,
    fnirs: np.ndarray,
    eeg_mask: np.ndarray,
    fnirs_mask: np.ndarray,
    feature_config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    rates = feature_config["sample_rate_hz"]
    patches = feature_config["patch_samples"]
    eeg_batch = extract_eeg_patch_features(
        eeg,
        sample_rate_hz=float(rates["eeg"]),
        patch_size=int(patches["eeg"]),
        valid_mask=eeg_mask,
    )
    fnirs_batch = extract_fnirs_patch_features(
        fnirs,
        sample_rate_hz=float(rates["fnirs"]),
        patch_size=int(patches["fnirs"]),
        valid_mask=fnirs_mask,
    )
    eeg_values = eeg_batch.flatten_channels().astype(np.float32)
    fnirs_values = fnirs_batch.flatten_channels().astype(np.float32)
    eeg_valid = eeg_batch.feature_valid_mask.reshape(eeg_values.shape).astype(bool)
    fnirs_valid = fnirs_batch.feature_valid_mask.reshape(fnirs_values.shape).astype(bool)
    manifest = {
        "eeg": eeg_batch.manifest.to_dict(),
        "fnirs": fnirs_batch.manifest.to_dict(),
    }
    return eeg_values, fnirs_values, eeg_valid, fnirs_valid, manifest


def _forward_shared(
    model_config: Mapping[str, Any],
    checkpoint: Path,
    eeg: np.ndarray,
    fnirs: np.ndarray,
    eeg_mask: np.ndarray,
    fnirs_mask: np.ndarray,
    batch_size: int,
    *,
    expected_task_id: str | None = None,
    expected_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if torch is None:
        raise RuntimeError("torch is required to read a real continuous shared/private checkpoint")
    from src.tokenizers.continuous_shared_private import ContinuousSharedPrivateModel

    kwargs = dict(model_config)
    kwargs.pop("type", None)
    model = ContinuousSharedPrivateModel(**kwargs).to(torch.device("cpu"))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "continuous_shared_private_checkpoint_v1":
        raise ValueError("checkpoint schema is not continuous_shared_private_checkpoint_v1")
    if expected_task_id is not None and str(payload.get("task_id")) != str(expected_task_id):
        raise ValueError("checkpoint task_id does not match its cell")
    if expected_seed is not None and int(payload.get("seed")) != int(expected_seed):
        raise ValueError("checkpoint seed does not match its cell")
    if payload.get("protected_open") is not False:
        raise PermissionError("checkpoint does not declare protected_open=false")
    if payload.get("vector_quantization") is not False:
        raise ValueError("existing checkpoint is not the no-VQ continuous model")
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    eeg_parts: list[np.ndarray] = []
    fnirs_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(eeg), max(1, int(batch_size))):
            stop = min(start + max(1, int(batch_size)), len(eeg))
            result = model(
                torch.from_numpy(eeg[start:stop]),
                torch.from_numpy(fnirs[start:stop]),
                torch.from_numpy(eeg_mask[start:stop]),
                torch.from_numpy(fnirs_mask[start:stop]),
            )
            eeg_parts.append(result["eeg_shared"].float().cpu().numpy())
            fnirs_parts.append(result["fnirs_shared"].float().cpu().numpy())
    del model
    return np.concatenate(eeg_parts, axis=0), np.concatenate(fnirs_parts, axis=0)


def _limit_rows(bundle: ArrayBundle, limit: int | None, *, preserve_groups: bool = False) -> ArrayBundle:
    if limit is None or bundle.count <= limit:
        return bundle
    if preserve_groups:
        selected: list[int] = []
        for subject in sorted(set(bundle.subject.tolist())):
            for condition in sorted(set(bundle.condition[bundle.subject == subject].tolist())):
                group = np.flatnonzero((bundle.subject == subject) & (bundle.condition == condition))
                selected.extend(group[:2].tolist())
        indices = np.asarray(sorted(set(selected))[:limit], dtype=int)
    else:
        indices = np.arange(limit, dtype=int)
    return ArrayBundle(
        **{
            name: np.asarray(getattr(bundle, name))[indices]
            for name in (
                "eeg_shared",
                "fnirs_shared",
                "eeg_native",
                "fnirs_native",
                "eeg_shared_mask",
                "fnirs_shared_mask",
                "eeg_native_mask",
                "fnirs_native_mask",
                "target_mask",
                "subject",
                "condition",
                "sample_id",
            )
        },
        role=bundle.role,
    )


def _subject_number(subject: str) -> int | None:
    match = re.search(r"(?:subject_|VP)(\d+)$", str(subject))
    return int(match.group(1)) if match else None


def _load_real_cell(
    source_run: Path,
    task_id: str,
    seed_dir: Path,
    source_config: Mapping[str, Any],
    probe_config: Mapping[str, Any],
    *,
    smoke: bool,
) -> tuple[ArrayBundle, ArrayBundle, dict[str, Any]]:
    cell_manifest_path = seed_dir / "manifest.json"
    if not cell_manifest_path.exists():
        raise FileNotFoundError(cell_manifest_path)
    cell_manifest = json.loads(cell_manifest_path.read_text(encoding="utf-8"))
    if cell_manifest.get("schema") != "continuous_shared_private_cell_v1":
        raise ValueError(f"cell manifest schema mismatch: {seed_dir}")
    if cell_manifest.get("protected_open") is not False:
        raise PermissionError("validation cell opened a protected cohort")
    if str(cell_manifest.get("task_id")) != str(task_id):
        raise ValueError("cell manifest task_id does not match its path")
    parsed_seed = int(seed_dir.name.split("_", 1)[1])
    if int(cell_manifest.get("seed", parsed_seed)) != parsed_seed:
        raise ValueError("cell manifest seed does not match its path")
    _verify_artifacts(
        seed_dir,
        cell_manifest,
        ("best.pt", "train_statistics.json", "validation_predictions.npz"),
        f"cell {seed_dir}",
    )
    source_path = source_run / "source_samples.npz"
    source_fields = {
        "schema", "eeg", "fnirs", "eeg_mask", "fnirs_mask", "target_mask",
        "sample_id", "subject", "role", "condition", "task_id",
    }
    with np.load(source_path, allow_pickle=False) as source:
        if str(source["schema"].item()) != SOURCE_SCHEMA:
            raise ValueError("source_samples.npz schema mismatch")
        source_arrays = {key: np.asarray(source[key]) for key in source_fields if key in source.files}
    missing_source_fields = source_fields.difference(source_arrays)
    if missing_source_fields:
        raise KeyError(f"source_samples.npz lacks required fields: {sorted(missing_source_fields)}")
    source_task = source_arrays["task_id"].astype(str) == str(task_id)
    fit_rows = source_task & (source_arrays["role"].astype(str) == "fit")
    if not fit_rows.any():
        raise RuntimeError(f"no fit source rows for {task_id}")

    validation_path = seed_dir / "validation_predictions.npz"
    required = {
        "sample_id", "subject", "condition", "eeg_mask", "fnirs_mask",
        "eeg_shared", "fnirs_shared", "eeg_observed", "fnirs_observed",
    }
    validation_fields = required | {"schema", "target_mask"}
    with np.load(validation_path, allow_pickle=False) as validation:
        if str(validation["schema"].item()) != VALIDATION_SCHEMA:
            raise ValueError(f"validation schema mismatch: {validation_path}")
        validation_arrays = {key: np.asarray(validation[key]) for key in validation_fields if key in validation.files}
    if not required.issubset(validation_arrays):
        raise KeyError(f"validation export lacks {sorted(required - set(validation_arrays))}")
    # The full NPZ also contains target/driver predictions, but this evaluator
    # deliberately never loads or consumes them; target_mask is validity only.

    statistics = json.loads((seed_dir / "train_statistics.json").read_text(encoding="utf-8"))
    feature_config = probe_config["features"]
    source_model_config = source_config["model"]
    eeg_raw = source_arrays["eeg"][fit_rows].astype(np.float32)
    fnirs_raw = source_arrays["fnirs"][fit_rows].astype(np.float32)
    eeg_mask = source_arrays["eeg_mask"][fit_rows].astype(bool)
    fnirs_mask = source_arrays["fnirs_mask"][fit_rows].astype(bool)
    normalisation = statistics["normalization"]
    eeg_norm = _normalise_signal(
        eeg_raw, eeg_mask, normalisation["eeg"]["mean"], normalisation["eeg"]["scale"], int(feature_config["patch_samples"]["eeg"])
    )
    fnirs_norm = _normalise_signal(
        fnirs_raw, fnirs_mask, normalisation["fnirs"]["mean"], normalisation["fnirs"]["scale"], int(feature_config["patch_samples"]["fnirs"])
    )
    fit_eeg_shared, fit_fnirs_shared = _forward_shared(
        source_model_config,
        seed_dir / "best.pt",
        eeg_norm,
        fnirs_norm,
        eeg_mask,
        fnirs_mask,
        int(probe_config.get("inference_batch_size", 32)),
        expected_task_id=task_id,
        expected_seed=int(cell_manifest.get("seed", parsed_seed)),
    )
    # Native features use the canonical robust-SD source signals.  The model
    # still receives the train-only z-scored copies above.
    fit_eeg_native, fit_fnirs_native, fit_eeg_native_mask, fit_fnirs_native_mask, feature_manifest = _extract_native(
        eeg_raw, fnirs_raw, eeg_mask, fnirs_mask, feature_config
    )
    fit_bundle = ArrayBundle(
        eeg_shared=fit_eeg_shared,
        fnirs_shared=fit_fnirs_shared,
        eeg_native=fit_eeg_native,
        fnirs_native=fit_fnirs_native,
        eeg_shared_mask=eeg_mask,
        fnirs_shared_mask=fnirs_mask,
        eeg_native_mask=fit_eeg_native_mask,
        fnirs_native_mask=fit_fnirs_native_mask,
        subject=source_arrays["subject"][fit_rows],
        condition=source_arrays["condition"][fit_rows],
        sample_id=source_arrays["sample_id"][fit_rows],
        role="fit",
        target_mask=_token_mask(source_arrays["target_mask"][fit_rows], int(fit_rows.sum()), eeg_mask.shape[1]),
    )
    fit_numbers = {
        number for number in (_subject_number(value) for value in fit_bundle.subject)
        if number is not None
    }
    if fit_numbers and any(number >= 19 for number in fit_numbers):
        raise ValueError("fit source unexpectedly contains post-selection development subjects")
    validation_eeg_mask = validation_arrays["eeg_mask"].astype(bool)
    validation_fnirs_mask = validation_arrays["fnirs_mask"].astype(bool)
    validation_eeg_raw = _denormalise_signal(
        validation_arrays["eeg_observed"],
        validation_eeg_mask,
        normalisation["eeg"]["mean"],
        normalisation["eeg"]["scale"],
        int(feature_config["patch_samples"]["eeg"]),
    )
    validation_fnirs_raw = _denormalise_signal(
        validation_arrays["fnirs_observed"],
        validation_fnirs_mask,
        normalisation["fnirs"]["mean"],
        normalisation["fnirs"]["scale"],
        int(feature_config["patch_samples"]["fnirs"]),
    )
    val_eeg_native, val_fnirs_native, val_eeg_native_mask, val_fnirs_native_mask, _ = _extract_native(
        validation_eeg_raw,
        validation_fnirs_raw,
        validation_eeg_mask,
        validation_fnirs_mask,
        feature_config,
    )
    validation_target_mask = _token_mask(
        validation_arrays.get("target_mask", validation_arrays["eeg_mask"] & validation_arrays["fnirs_mask"]).astype(bool),
        len(validation_arrays["subject"]),
        validation_arrays["eeg_mask"].shape[1],
    )
    validation_bundle = ArrayBundle(
        eeg_shared=validation_arrays["eeg_shared"].astype(np.float32),
        fnirs_shared=validation_arrays["fnirs_shared"].astype(np.float32),
        eeg_native=val_eeg_native,
        fnirs_native=val_fnirs_native,
        eeg_shared_mask=validation_arrays["eeg_mask"].astype(bool),
        fnirs_shared_mask=validation_arrays["fnirs_mask"].astype(bool),
        eeg_native_mask=val_eeg_native_mask,
        fnirs_native_mask=val_fnirs_native_mask,
        subject=validation_arrays["subject"].astype(str),
        condition=validation_arrays["condition"].astype(str),
        sample_id=validation_arrays["sample_id"].astype(str),
        role="development_validation",
        target_mask=validation_target_mask,
    )
    validation_numbers = {
        number for number in (_subject_number(value) for value in validation_bundle.subject)
        if number is not None
    }
    if validation_numbers and not validation_numbers.issubset({19, 20, 21, 22, 23}):
        raise ValueError("legacy post-selection development validation must be subjects 19--23")
    if smoke:
        smoke_cfg = probe_config.get("smoke", {})
        fit_bundle = _limit_rows(fit_bundle, int(smoke_cfg.get("max_fit_trials", 16)))
        validation_bundle = _limit_rows(
            validation_bundle,
            int(smoke_cfg.get("max_validation_trials", 16)),
            preserve_groups=True,
        )
    metadata = {
        "task_id": task_id,
        "dataset_id": str(cell_manifest.get("dataset_id", "")),
        "seed": int(cell_manifest.get("seed", seed_dir.name.rsplit("_", 1)[-1])),
        "source_cell": str(seed_dir),
        "cell_manifest_sha256": _sha256(seed_dir / "manifest.json"),
        "validation_prediction_sha256": _sha256(validation_path),
        "train_statistics_sha256": _sha256(seed_dir / "train_statistics.json"),
        "checkpoint_sha256": _sha256(seed_dir / "best.pt"),
        "fit_subjects": sorted(set(fit_bundle.subject.tolist())),
        "validation_subjects": sorted(set(validation_bundle.subject.tolist())),
        "fit_source_schema": SOURCE_SCHEMA,
        "validation_schema": VALIDATION_SCHEMA,
        "feature_manifest": feature_manifest,
    }
    return fit_bundle, validation_bundle, metadata


def _discover_real_cells(source_run: Path, probe_config: Mapping[str, Any], *, smoke: bool) -> list[tuple[str, int, Path]]:
    cells: list[tuple[str, int, Path]] = []
    for validation_path in sorted(source_run.glob("cells/*/seed_*/validation_predictions.npz")):
        task_id = validation_path.parent.parent.name
        seed_text = validation_path.parent.name
        try:
            seed = int(seed_text.split("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"cannot parse seed directory {validation_path.parent}") from exc
        cells.append((task_id, seed, validation_path.parent))
    if not cells:
        raise FileNotFoundError(f"no validation_predictions.npz cells found under {source_run}")
    if smoke:
        smoke_cfg = probe_config.get("smoke", {})
        requested = int(smoke_cfg.get("cells", 1))
        cells = cells[: max(1, requested)]
    return cells


# ---------------------------------------------------------------------------
# Lag pairing, derangement, and ridge estimator


def make_derangement(
    subjects: Sequence[str], conditions: Sequence[str], sample_ids: Sequence[str], seed: int = 0
) -> np.ndarray:
    """Deterministically cycle each subject/condition group without identity."""

    subjects = np.asarray(subjects).astype(str)
    conditions = np.asarray(conditions).astype(str)
    sample_ids = np.asarray(sample_ids).astype(str)
    if not (len(subjects) == len(conditions) == len(sample_ids)):
        raise ValueError("derangement metadata lengths differ")
    donor = np.full(len(subjects), -1, dtype=int)
    groups: dict[tuple[str, str], list[int]] = {}
    for index, key in enumerate(zip(subjects, conditions, strict=True)):
        groups.setdefault((str(key[0]), str(key[1])), []).append(index)
    for key, indices in groups.items():
        if len(indices) < 2:
            raise ValueError(f"derangement group has fewer than two validation trials: {key}")
        ranked = sorted(
            indices,
            key=lambda index: hashlib.sha256(f"{seed}|{sample_ids[index]}".encode()).hexdigest(),
        )
        for position, target in enumerate(ranked):
            donor[target] = ranked[(position + 1) % len(ranked)]
    if np.any(donor < 0) or np.any(donor == np.arange(len(donor))):
        raise RuntimeError("non-identity derangement construction failed")
    if np.any(subjects != subjects[donor]) or np.any(conditions != conditions[donor]):
        raise RuntimeError("derangement changed subject or condition")
    return donor


@dataclass(frozen=True)
class LaggedPairs:
    """Public, schema-light lag pair result for standalone fixture tests."""

    source: np.ndarray
    target: np.ndarray
    sample_index: np.ndarray
    source_token: np.ndarray
    target_token: np.ndarray
    pair_mask: np.ndarray

    @property
    def valid_pair_mask(self) -> np.ndarray:
        return self.pair_mask


@dataclass
class PairData:
    x: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    sample_id: np.ndarray
    source_trial_index: np.ndarray
    source_token_index: np.ndarray
    target_token_index: np.ndarray
    grid_mask: np.ndarray
    grid_target_index: np.ndarray


def _pair_data(
    source: np.ndarray,
    target: np.ndarray,
    source_token_mask: np.ndarray,
    target_token_mask: np.ndarray,
    source_feature_mask: np.ndarray,
    target_feature_mask: np.ndarray,
    subjects: np.ndarray,
    sample_ids: np.ndarray,
    lag_tokens: int,
    *,
    source_trial_indices: np.ndarray | None = None,
    circular_shift_tokens: int = 0,
) -> PairData:
    source = np.asarray(source)
    target = np.asarray(target)
    source_token_mask = np.asarray(source_token_mask, dtype=bool)
    target_token_mask = np.asarray(target_token_mask, dtype=bool)
    source_feature_mask = np.asarray(source_feature_mask, dtype=bool)
    target_feature_mask = np.asarray(target_feature_mask, dtype=bool)
    n, tokens, _ = source.shape
    if target.shape[0] != n or target.shape[1] != tokens:
        raise ValueError("source and target trial/token axes differ")
    if circular_shift_tokens:
        source = np.roll(source, int(circular_shift_tokens), axis=1)
        source_token_mask = np.roll(source_token_mask, int(circular_shift_tokens), axis=1)
        source_feature_mask = np.roll(source_feature_mask, int(circular_shift_tokens), axis=1)
    if lag_tokens >= 0:
        source_positions = np.arange(0, tokens - lag_tokens, dtype=int)
        target_positions = source_positions + int(lag_tokens)
    else:
        source_positions = np.arange(-int(lag_tokens), tokens, dtype=int)
        target_positions = source_positions + int(lag_tokens)
    grid_mask = np.zeros((n, tokens), dtype=bool)
    grid_target_index = np.full((n, tokens), -1, dtype=np.int16)
    if len(source_positions):
        valid = (
            source_token_mask[:, source_positions]
            & target_token_mask[:, target_positions]
            & source_feature_mask[:, source_positions].all(axis=-1)
            & target_feature_mask[:, target_positions].all(axis=-1)
            & np.isfinite(source[:, source_positions]).all(axis=-1)
            & np.isfinite(target[:, target_positions]).all(axis=-1)
        )
        grid_mask[:, source_positions] = valid
        grid_target_index[:, source_positions] = target_positions
        trial_index = np.repeat(np.arange(n)[:, None], len(source_positions), axis=1)
        source_index = np.broadcast_to(source_positions[None, :], trial_index.shape)
        target_index = np.broadcast_to(target_positions[None, :], trial_index.shape)
        admitted = valid
        x = source[:, source_positions][admitted]
        y = target[:, target_positions][admitted]
        subject = np.broadcast_to(np.asarray(subjects)[:, None], trial_index.shape)[admitted]
        sample_id = np.broadcast_to(np.asarray(sample_ids)[:, None], trial_index.shape)[admitted]
        source_trial_index = trial_index[admitted]
        source_token_index = source_index[admitted]
        target_token_index = target_index[admitted]
    else:
        x = np.empty((0, source.shape[-1]), dtype=source.dtype)
        y = np.empty((0, target.shape[-1]), dtype=target.dtype)
        subject = np.empty((0,), dtype=str)
        sample_id = np.empty((0,), dtype=str)
        source_trial_index = np.empty((0,), dtype=int)
        source_token_index = np.empty((0,), dtype=int)
        target_token_index = np.empty((0,), dtype=int)
    return PairData(
        x=np.asarray(x),
        y=np.asarray(y),
        subject=np.asarray(subject).astype(str),
        sample_id=np.asarray(sample_id).astype(str),
        source_trial_index=source_trial_index.astype(int),
        source_token_index=source_token_index.astype(int),
        target_token_index=target_token_index.astype(int),
        grid_mask=grid_mask,
        grid_target_index=grid_target_index,
    )


def build_lagged_pairs(
    source_latent: np.ndarray,
    target: np.ndarray,
    source_valid_mask: np.ndarray,
    target_valid_mask: np.ndarray,
    lag_tokens: int,
) -> LaggedPairs:
    """Build finite, mask-intersected pairs using the registered lag sign.

    This public wrapper intentionally uses no metadata or fitting, making it
    suitable for synthetic sign/mask tests.  Feature-level masks are admitted
    only when all coordinates of a token are valid, the same policy used by the
    native feature probe.
    """

    source = np.asarray(source_latent)
    target = np.asarray(target)
    if source.ndim != 3 or target.ndim != 3:
        raise ValueError("source_latent and target must be [trial, token, dimension]")
    if source.shape[:2] != target.shape[:2]:
        raise ValueError("source and target leading trial/token shapes differ")
    n, tokens = source.shape[:2]

    def feature_mask(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        candidate = np.asarray(mask, dtype=bool)
        if candidate.shape == values.shape[:2]:
            candidate = np.broadcast_to(candidate[..., None], values.shape)
        elif candidate.shape != values.shape:
            raise ValueError(f"valid mask shape {candidate.shape} differs from values {values.shape}")
        return candidate & np.isfinite(values)

    source_mask = feature_mask(source, source_valid_mask)
    target_mask = feature_mask(target, target_valid_mask)
    subjects = np.asarray([str(index) for index in range(n)])
    sample_ids = np.asarray([f"sample-{index}" for index in range(n)])
    pairs = _pair_data(
        source,
        target,
        source_mask.all(axis=-1),
        target_mask.all(axis=-1),
        source_mask,
        target_mask,
        subjects,
        sample_ids,
        int(lag_tokens),
    )
    return LaggedPairs(
        source=pairs.x,
        target=pairs.y,
        sample_index=pairs.source_trial_index,
        source_token=pairs.source_token_index,
        target_token=pairs.target_token_index,
        pair_mask=pairs.grid_mask,
    )


@dataclass
class RidgeModel:
    x_mean: np.ndarray
    x_scale: np.ndarray
    basis: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    weights: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        centered = (x - self.x_mean) / self.x_scale
        reduced = centered @ self.basis
        predicted = reduced @ self.weights
        return (predicted * self.y_scale + self.y_mean).astype(np.float32)


def fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    *,
    components: int = 32,
    alpha: float = 1.0,
) -> RidgeModel:
    """Fit train-only PCA-ridge with finite, explicit standardization."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("ridge inputs must be [pair, feature] with a common pair axis")
    if x.shape[0] < 2:
        raise ValueError("ridge requires at least two fit pairs")
    if float(alpha) <= 0:
        raise ValueError("ridge alpha must be positive")
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    if int(finite.sum()) < 2:
        raise ValueError("ridge has fewer than two finite fit pairs")
    x = x[finite]
    y = y[finite]
    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0)
    x_scale = np.where(x_scale > 1e-8, x_scale, 1.0)
    x_scaled = (x - x_mean) / x_scale
    _, singular, vt = np.linalg.svd(x_scaled, full_matrices=False)
    rank = int(np.sum(singular > 1e-10))
    component_count = min(int(components), rank, vt.shape[0], vt.shape[1])
    if component_count:
        basis = vt[:component_count].T
        reduced = x_scaled @ basis
    else:
        basis = np.zeros((x.shape[1], 0), dtype=np.float64)
        reduced = np.zeros((x.shape[0], 0), dtype=np.float64)
    y_mean = y.mean(axis=0)
    y_scale = y.std(axis=0)
    y_scale = np.where(y_scale > 1e-8, y_scale, 1.0)
    y_scaled = (y - y_mean) / y_scale
    if component_count:
        system = reduced.T @ reduced + float(alpha) * np.eye(component_count)
        weights = np.linalg.solve(system, reduced.T @ y_scaled)
    else:
        weights = np.zeros((0, y.shape[1]), dtype=np.float64)
    return RidgeModel(
        x_mean=x_mean,
        x_scale=x_scale,
        basis=basis,
        y_mean=y_mean,
        y_scale=y_scale,
        weights=weights,
    )


def ridge_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    components: int = 32,
    alpha: float = 1.0,
) -> np.ndarray:
    return fit_ridge(train_x, train_y, components=components, alpha=alpha).predict(test_x)


# ---------------------------------------------------------------------------
# Subject-level metrics and one-cell evaluation


def _empty_metric() -> dict[str, float]:
    return {
        "proper_prediction_score": float("nan"),
        "prediction_score": float("nan"),
        "prediction_r2": float("nan"),
        "baseline_prediction_score": float("nan"),
        "delta_r2": float("nan"),
        "model_sse": float("nan"),
        "baseline_sse": float("nan"),
        "total_sse": float("nan"),
    }


def _metric(y: np.ndarray, prediction: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    model_sse = float(np.square(y - prediction).sum())
    baseline_sse = float(np.square(y - baseline).sum())
    target_center = y.mean(axis=0, keepdims=True) if len(y) else np.zeros((1, y.shape[-1]))
    total_sse = float(np.square(y - target_center).sum())
    prediction_score = float(1.0 - model_sse / total_sse) if total_sse > 1e-12 else float("nan")
    baseline_score = float(1.0 - baseline_sse / total_sse) if total_sse > 1e-12 else float("nan")
    delta_r2 = float(1.0 - model_sse / baseline_sse) if baseline_sse > 1e-12 else float("nan")
    return {
        "proper_prediction_score": prediction_score,
        "prediction_score": prediction_score,
        "prediction_r2": prediction_score,
        "baseline_prediction_score": baseline_score,
        "delta_r2": delta_r2,
        "model_sse": model_sse,
        "baseline_sse": baseline_sse,
        "total_sse": total_sse,
    }


def _lag_metadata(
    lag_tokens: int,
    *,
    claim_status: str = CLAIM_STATUS,
    checkpoint_selection_status: str = CHECKPOINT_SELECTION_STATUS,
) -> dict[str, Any]:
    lag_seconds = int(lag_tokens) * TOKEN_STEP_SECONDS
    if lag_seconds < 0:
        direction = "negative_lag_later_eeg_to_earlier_fnirs_offline_association"
    elif lag_seconds > 0:
        direction = "positive_lag_later_fnirs_offline_association"
    else:
        direction = "zero_lag_contemporaneous_offline_association"
    return {
        "lag_tokens": int(lag_tokens),
        "lag_seconds": lag_seconds,
        "negative_lag": bool(lag_seconds < 0),
        "lag_direction": direction,
        "claim_status": claim_status,
        "checkpoint_selection_status": checkpoint_selection_status,
        "causal_future_claim": False,
        "evaluator_temporal_mode": EVALUATOR_TEMPORAL_MODE,
    }


def _representation_arrays(bundle: ArrayBundle, representation: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_token_mask = bundle.fnirs_shared_mask & bundle.target_mask
    if representation == "eeg_shared_to_fnirs_shared":
        return bundle.eeg_shared, bundle.fnirs_shared, bundle.eeg_shared_mask[..., None], target_token_mask[..., None]
    if representation == "eeg_native_patch_to_fnirs_native_patch":
        target_feature_mask = bundle.fnirs_native_mask & bundle.target_mask[..., None]
        return bundle.eeg_native, bundle.fnirs_native, bundle.eeg_native_mask, target_feature_mask
    raise ValueError(f"unknown representation {representation}")


def evaluate_cell(
    task_id: str,
    seed: int,
    fit_bundle: ArrayBundle,
    validation_bundle: ArrayBundle,
    *,
    dataset_id: str = "",
    claim_status: str = CLAIM_STATUS,
    checkpoint_selection_status: str = CHECKPOINT_SELECTION_STATUS,
    alpha: float,
    components: int,
    circular_shift_tokens: int,
    donor_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    """Fit matched-only ridge per lag and return subject rows plus arrays."""

    fit_subject_count = len(set(fit_bundle.subject.tolist()))
    if fit_subject_count < 2:
        raise ValueError("at least two fit subjects are required for subject-level probe evaluation")
    overlap = set(fit_bundle.sample_id.tolist()).intersection(validation_bundle.sample_id.tolist())
    if overlap:
        raise ValueError(f"fit/validation sample_id overlap: {sorted(overlap)[:3]}")
    donor = make_derangement(
        validation_bundle.subject, validation_bundle.condition, validation_bundle.sample_id, seed=donor_seed
    )
    all_metrics: list[dict[str, Any]] = []
    advantage_rows: list[dict[str, Any]] = []
    source_token_grid = np.full((len(LAG_TOKENS), validation_bundle.token_count), -1, dtype=np.int16)
    for lag_index, lag_tokens in enumerate(LAG_TOKENS):
        if lag_tokens >= 0:
            positions = np.arange(0, validation_bundle.token_count - lag_tokens, dtype=int)
        else:
            positions = np.arange(-int(lag_tokens), validation_bundle.token_count, dtype=int)
        source_token_grid[lag_index, positions] = positions
    saved: dict[str, np.ndarray] = {
        "sample_id": validation_bundle.sample_id,
        "subject": validation_bundle.subject,
        "condition": validation_bundle.condition,
        "donor_index": donor,
        "donor_sample_id": validation_bundle.sample_id[donor],
        "fit_sample_id": fit_bundle.sample_id,
        "fit_subject": fit_bundle.subject,
        "fit_condition": fit_bundle.condition,
        "fit_eeg_shared_mask": fit_bundle.eeg_shared_mask,
        "fit_fnirs_shared_mask": fit_bundle.fnirs_shared_mask,
        "fit_eeg_native_mask": fit_bundle.eeg_native_mask,
        "fit_fnirs_native_mask": fit_bundle.fnirs_native_mask,
        "fit_target_mask": fit_bundle.target_mask,
        "eeg_shared_source": validation_bundle.eeg_shared,
        "eeg_native_source": validation_bundle.eeg_native,
        "fnirs_shared_target": validation_bundle.fnirs_shared,
        "fnirs_native_target": validation_bundle.fnirs_native,
        "eeg_shared_mask": validation_bundle.eeg_shared_mask,
        "fnirs_shared_mask": validation_bundle.fnirs_shared_mask,
        "eeg_native_mask": validation_bundle.eeg_native_mask,
        "fnirs_native_mask": validation_bundle.fnirs_native_mask,
        "target_mask": validation_bundle.target_mask,
        "lag_seconds": np.asarray(LAG_SECONDS, dtype=np.int16),
        "lag_tokens": np.asarray(LAG_TOKENS, dtype=np.int16),
        "source_token_index": source_token_grid,
        "condition_names": np.asarray(CONDITIONS),
        "negative_lag": np.asarray([lag < 0 for lag in LAG_SECONDS], dtype=bool),
        "claim_status": np.asarray(claim_status),
        "checkpoint_selection_status": np.asarray(checkpoint_selection_status),
        "evaluator_temporal_mode": np.asarray(EVALUATOR_TEMPORAL_MODE),
        "causal_future_claim": np.asarray(False),
        "forbidden_model_fields_not_used": np.asarray(FORBIDDEN_MODEL_FIELDS),
    }
    for representation in REPRESENTATIONS:
        fit_x, fit_y, fit_x_token, fit_y_token = _representation_arrays(fit_bundle, representation)
        val_x, val_y, val_x_token, val_y_token = _representation_arrays(validation_bundle, representation)
        representation_slug = "shared" if representation.startswith("eeg_shared") else "native"
        prediction_grid: np.ndarray | None = None
        grid_mask = np.zeros((len(CONDITIONS), len(LAG_TOKENS), validation_bundle.count, validation_bundle.token_count), dtype=bool)
        grid_target_index = np.full(grid_mask.shape, -1, dtype=np.int16)
        for lag_index, lag_tokens in enumerate(LAG_TOKENS):
            fit_pairs = _pair_data(
                fit_x,
                fit_y,
                fit_x_token.any(axis=-1),
                fit_y_token.any(axis=-1),
                fit_x_token,
                fit_y_token,
                fit_bundle.subject,
                fit_bundle.sample_id,
                lag_tokens,
            )
            fit_status = "ok"
            fit_skip_reason = ""
            model: RidgeModel | None
            fit_baseline: np.ndarray | None
            if len(fit_pairs.x) < 2:
                model = None
                fit_baseline = None
                fit_status = "skipped"
                fit_skip_reason = "no_supported_lag_pairs"
            else:
                model = fit_ridge(fit_pairs.x, fit_pairs.y, components=components, alpha=alpha)
                fit_baseline = fit_pairs.y.mean(axis=0, keepdims=True)
            condition_pairs: dict[str, PairData] = {}
            for condition_index, condition in enumerate(CONDITIONS):
                if condition == "matched":
                    source_indices = np.arange(validation_bundle.count, dtype=int)
                    shift = 0
                elif condition == "deranged_same_subject_same_condition_nonidentity":
                    source_indices = donor
                    shift = 0
                else:
                    source_indices = np.arange(validation_bundle.count, dtype=int)
                    shift = int(circular_shift_tokens)
                # The target remains the current validation trial in all conditions.
                pair = _pair_data(
                    val_x[source_indices],
                    val_y,
                    val_x_token[source_indices].any(axis=-1),
                    val_y_token.any(axis=-1),
                    val_x_token[source_indices],
                    val_y_token,
                    validation_bundle.subject,
                    validation_bundle.sample_id,
                    lag_tokens,
                    source_trial_indices=source_indices,
                    circular_shift_tokens=shift,
                )
                condition_pairs[condition] = pair
                pair_mask_sha256 = hashlib.sha256(
                    np.ascontiguousarray(pair.grid_mask).view(np.uint8)
                ).hexdigest()
                null_policy = _null_policy(condition, circular_shift_tokens)
                if prediction_grid is None:
                    prediction_grid = np.full(
                        (len(CONDITIONS), len(LAG_TOKENS), validation_bundle.count, validation_bundle.token_count, val_y.shape[-1]),
                        np.nan,
                        dtype=np.float32,
                    )
                predictions = (
                    model.predict(pair.x)
                    if model is not None and len(pair.x)
                    else np.empty((0, val_y.shape[-1]), dtype=np.float32)
                )
                if len(pair.x) and model is not None:
                    prediction_grid[condition_index, lag_index, pair.source_trial_index, pair.source_token_index] = predictions
                grid_mask[condition_index, lag_index] = pair.grid_mask
                grid_target_index[condition_index, lag_index] = pair.grid_target_index
                for subject in sorted(set(validation_bundle.subject.tolist())):
                    selected = pair.subject == subject
                    if fit_status != "ok":
                        metric = _empty_metric()
                        status = fit_status
                        skipped_reason = fit_skip_reason
                    elif not selected.any():
                        metric = _empty_metric()
                        status = "skipped"
                        skipped_reason = "no_supported_validation_pairs"
                    else:
                        assert fit_baseline is not None
                        metric = _metric(
                            pair.y[selected],
                            predictions[selected],
                            np.broadcast_to(fit_baseline, pair.y[selected].shape),
                        )
                        status = "ok"
                        skipped_reason = ""
                    row: dict[str, Any] = {
                        "schema": SCHEMA,
                        "task_id": task_id,
                        "dataset_id": dataset_id,
                        "seed": int(seed),
                        "subject": subject,
                        "representation": representation,
                        "condition": condition,
                        "evaluation_unit": "subject",
                        "evaluator_temporal_mode": EVALUATOR_TEMPORAL_MODE,
                        "checkpoint_selection_status": checkpoint_selection_status,
                        "selection_data": "fit_only_fixed_probe_alpha",
                        "fresh_fit_held_out": False,
                        "causal_future_claim": False,
                        "token_or_window_as_replicate": False,
                        "components": int(components),
                        "alpha": float(alpha),
                        "status": status,
                        "skipped_reason": skipped_reason,
                        "null_policy": null_policy,
                        "pair_mask_sha256": pair_mask_sha256,
                        "fit_pair_count": int(len(fit_pairs.x)),
                        "fit_subject_count": int(len(set(fit_bundle.subject.tolist()))),
                        "supported_pairs": int(selected.sum()),
                        "supported_trials": int(len(set(pair.sample_id[selected].tolist()))) if selected.any() else 0,
                        "supported_subject_count": 1 if selected.any() else 0,
                        **_lag_metadata(
                            lag_tokens,
                            claim_status=claim_status,
                            checkpoint_selection_status=checkpoint_selection_status,
                        ),
                        **metric,
                    }
                    all_metrics.append(row)
            # Matched advantage is always computed at the same lag and subject.
            for subject in sorted(set(validation_bundle.subject.tolist())):
                matched = next(row for row in all_metrics[::-1] if row["task_id"] == task_id and row["seed"] == int(seed) and row["representation"] == representation and row["condition"] == "matched" and row["subject"] == subject and row["lag_tokens"] == int(lag_tokens))
                for null_condition in CONDITIONS[1:]:
                    null = next(row for row in all_metrics[::-1] if row["task_id"] == task_id and row["seed"] == int(seed) and row["representation"] == representation and row["condition"] == null_condition and row["subject"] == subject and row["lag_tokens"] == int(lag_tokens))
                    matched_score = float(matched["delta_r2"])
                    null_score = float(null["delta_r2"])
                    advantage_rows.append(
                        {
                            "schema": SCHEMA,
                            "task_id": task_id,
                            "dataset_id": dataset_id,
                            "seed": int(seed),
                            "subject": subject,
                            "representation": representation,
                            "comparison": f"matched_minus_{null_condition}",
                            **_lag_metadata(
                            lag_tokens,
                            claim_status=claim_status,
                            checkpoint_selection_status=checkpoint_selection_status,
                        ),
                            "matched_delta_r2": matched_score,
                            "null_delta_r2": null_score,
                            "matched_pair_mask_sha256": matched["pair_mask_sha256"],
                            "null_pair_mask_sha256": null["pair_mask_sha256"],
                            "null_policy": null["null_policy"],
                            "matched_prediction_score": float(matched["proper_prediction_score"]),
                            "null_prediction_score": float(null["proper_prediction_score"]),
                            "matched_advantage_prediction_score": float(matched["proper_prediction_score"]) - float(null["proper_prediction_score"]) if np.isfinite(float(matched["proper_prediction_score"])) and np.isfinite(float(null["proper_prediction_score"])) else float("nan"),
                            "matched_advantage": matched_score - null_score if np.isfinite(matched_score) and np.isfinite(null_score) else float("nan"),
                            "status": "ok" if matched["status"] == "ok" and null["status"] == "ok" else "skipped",
                            "skipped_reason": "" if matched["status"] == "ok" and null["status"] == "ok" else "no_supported_lag_pairs",
                            "evaluation_unit": "subject",
                            "evaluator_temporal_mode": EVALUATOR_TEMPORAL_MODE,
                            "checkpoint_selection_status": checkpoint_selection_status,
                            "fresh_fit_held_out": False,
                            "causal_future_claim": False,
                        }
                    )
        if prediction_grid is not None:
            saved[f"{representation_slug}_prediction"] = prediction_grid
            saved[f"{representation_slug}_pair_mask"] = grid_mask
            saved[f"{representation_slug}_pair_mask_sha256"] = np.asarray(
                hashlib.sha256(np.ascontiguousarray(grid_mask).view(np.uint8)).hexdigest()
            )
            saved[f"{representation_slug}_pair_count"] = grid_mask.sum(axis=(-1, -2)).astype(np.int64)
            saved[f"{representation_slug}_target_token_index"] = grid_target_index
    return all_metrics, advantage_rows, saved


# ---------------------------------------------------------------------------
# Run orchestration and command line interface


def _verify_artifacts(root: Path, manifest: Mapping[str, Any], required: Sequence[str], label: str) -> None:
    entries = {str(item.get("path")): item for item in manifest.get("artifacts", [])}
    for relative in required:
        path = root / relative
        entry = entries.get(relative)
        if not path.exists() or entry is None:
            raise FileNotFoundError(f"{label} lacks declared artifact {relative}")
        expected = str(entry.get("sha256", ""))
        actual = _sha256(path)
        if expected != actual:
            raise RuntimeError(f"{label} artifact hash mismatch: {relative}")


def _load_source_manifest(source_run: Path) -> dict[str, Any]:
    manifest_path = source_run / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "continuous_shared_private_suite_v1":
        raise ValueError("source run is not the continuous shared/private full-run export")
    if manifest.get("status") != "completed" or manifest.get("failed_cell_count") != 0:
        raise RuntimeError("source continuous run is incomplete or contains failed cells")
    if manifest.get("protected_open") is not False:
        raise PermissionError("source run manifest does not declare protected_open=false")
    _verify_artifacts(source_run, manifest, ("config.yaml", "source_samples.npz"), "source run")
    return manifest


def _write_saved_npz(path: Path, saved: Mapping[str, np.ndarray]) -> None:
    np.savez_compressed(path, schema=np.asarray(SCHEMA), **saved)


def run(args: argparse.Namespace) -> Path:
    config_path = _resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    fixture_path = _resolve(args.fixture) if args.fixture is not None else None
    real_smoke = bool(getattr(args, "real_smoke", False))
    if bool(args.smoke) and real_smoke:
        raise ValueError("choose only one of --smoke and --real-smoke")
    source_value = args.source_run or config.get("source", {}).get("continuous_run")
    source_run = _resolve(source_value) if source_value is not None else None
    if fixture_path is None and (not args.smoke or real_smoke):
        if source_run is None:
            raise ValueError("a continuous_run source is required unless --fixture or --smoke is used")
        source_manifest = _load_source_manifest(source_run)
    else:
        source_manifest = None
    output = _resolve(args.output_dir) if args.output_dir is not None else _resolve(config["output"]["root"]) / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + config["experiment"]["name"]
    )
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    all_metrics: list[dict[str, Any]] = []
    all_advantage: list[dict[str, Any]] = []
    saved_cells: list[dict[str, Any]] = []
    feature_manifests: list[dict[str, Any]] = []
    try:
        shutil.copy2(config_path, staging / "config.yaml")
        if fixture_path is not None:
            bundles = load_fixture(fixture_path)
            metrics, advantage, saved = evaluate_cell(
                "synthetic_fixture", int(config.get("probe", {}).get("seed", 0)), bundles["fit"], bundles["validation"],
                claim_status="synthetic_fixture_offline_delayed_association",
                checkpoint_selection_status="synthetic_fixture_no_checkpoint",
                alpha=float(config["probe"]["alpha"]), components=int(config["probe"]["components"]),
                circular_shift_tokens=int(config.get("nulls", {}).get("within_trial_circular_shift_tokens", 1)), donor_seed=int(config.get("probe", {}).get("seed", 0)),
            )
            all_metrics.extend(metrics)
            all_advantage.extend(advantage)
            saved_cells.append(saved)
        elif args.smoke and not real_smoke and args.fixture is None:
            bundles = make_synthetic_fixture(seed=int(config.get("probe", {}).get("seed", 20260822)))
            metrics, advantage, saved = evaluate_cell(
                "synthetic_smoke", int(config.get("probe", {}).get("seed", 0)), bundles["fit"], bundles["validation"],
                claim_status="synthetic_fixture_offline_delayed_association",
                checkpoint_selection_status="synthetic_fixture_no_checkpoint",
                alpha=float(config["probe"]["alpha"]), components=int(config["probe"]["components"]),
                circular_shift_tokens=int(config.get("nulls", {}).get("within_trial_circular_shift_tokens", 1)), donor_seed=int(config.get("probe", {}).get("seed", 0)),
            )
            all_metrics.extend(metrics)
            all_advantage.extend(advantage)
            saved_cells.append(saved)
        else:
            source_config = yaml.safe_load((source_run / "config.yaml").read_text(encoding="utf-8"))
            for task_id, seed, seed_dir in _discover_real_cells(
                source_run, config, smoke=real_smoke
            ):
                fit_bundle, validation_bundle, metadata = _load_real_cell(
                    source_run, task_id, seed_dir, source_config, config,
                    smoke=real_smoke
                )
                metrics, advantage, saved = evaluate_cell(
                    task_id,
                    seed,
                    fit_bundle,
                    validation_bundle,
                    dataset_id=str(metadata.get("dataset_id", "")),
                    alpha=float(config["probe"]["alpha"]),
                    components=int(config["probe"]["components"]),
                    circular_shift_tokens=int(config.get("nulls", {}).get("within_trial_circular_shift_tokens", 1)),
                    donor_seed=seed,
                )
                all_metrics.extend(metrics)
                all_advantage.extend(advantage)
                saved_cells.append(saved)
                feature_manifests.append(metadata)
        _write_csv(staging / "lag_probe_subject_metrics.csv", all_metrics)
        _write_csv(staging / "lag_probe_subject_summary.csv", _subject_summary(all_metrics))
        _write_csv(staging / "matched_advantage_subject_metrics.csv", all_advantage)
        # Cells can differ in target latent dimension; write one NPZ per cell to
        # avoid padding or silently changing feature coordinates.
        prediction_dir = staging / "prediction_cells"
        prediction_dir.mkdir()
        for index, saved in enumerate(saved_cells):
            _write_saved_npz(prediction_dir / f"cell_{index:04d}.npz", saved)
        input_records: list[dict[str, Any]] = [
            {"path": str(config_path.relative_to(REPO_ROOT)) if config_path.is_relative_to(REPO_ROOT) else str(config_path), "sha256": _sha256(config_path), "source_kind": "configuration"},
            {"path": str(Path(__file__).resolve().relative_to(REPO_ROOT)), "sha256": _sha256(Path(__file__).resolve()), "source_kind": "runtime_module"},
            {"path": "src/analysis/physiological_patch_features.py", "sha256": _sha256(REPO_ROOT / "src/analysis/physiological_patch_features.py"), "source_kind": "runtime_module"},
            {"path": "src/tokenizers/continuous_shared_private.py", "sha256": _sha256(REPO_ROOT / "src/tokenizers/continuous_shared_private.py"), "source_kind": "runtime_module"},
        ]
        if source_manifest is not None:
            input_records.extend(
                [
                    {"path": str((source_run / "manifest.json").relative_to(REPO_ROOT)), "sha256": _sha256(source_run / "manifest.json")},
                    {"path": str((source_run / "source_samples.npz").relative_to(REPO_ROOT)), "sha256": _sha256(source_run / "source_samples.npz")},
                ]
            )
        for item in feature_manifests:
            for name, hash_key in (("validation_predictions.npz", "validation_prediction_sha256"), ("manifest.json", "cell_manifest_sha256"), ("best.pt", "checkpoint_sha256"), ("train_statistics.json", "train_statistics_sha256")):
                path = Path(item["source_cell"]) / name
                input_records.append({"path": str(path), "sha256": item.get(hash_key)})
        if fixture_path is not None:
            fixture_hash_paths = (
                [fixture_path]
                if fixture_path.is_file()
                else [fixture_path / name for name in ("fixture.npz", "fit_export.npz", "validation_export.npz") if (fixture_path / name).exists()]
            )
            input_records.extend({"path": str(path), "sha256": _sha256(path)} for path in fixture_hash_paths)
        synthetic_mode = fixture_path is not None or (bool(args.smoke) and not real_smoke)
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "completed",
            "mode": (
                "fixture"
                if fixture_path is not None
                else ("real_export_smoke" if real_smoke else ("smoke_synthetic" if args.smoke else "full_export"))
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment": config["experiment"],
            "protected_open": False,
            "protected_loader_constructed": False,
            "encoder_retrained": False,
            "probe_method": "ridge",
            "selection_data": "fit_only_fixed_probe_alpha",
            "probe_selection_data": "fit_only",
            "checkpoint_selection_status": CHECKPOINT_SELECTION_STATUS if source_manifest is not None else "synthetic_fixture_no_checkpoint",
            "checkpoint_selection_subjects": {
                "single_trial": ["subject_19", "subject_20", "subject_21", "subject_22", "subject_23"],
                "simultaneous": ["VP019", "VP020", "VP021", "VP022", "VP023"],
            } if source_manifest is not None else {},
            "checkpoint_selection_subject_range": "19-23" if source_manifest is not None else None,
            "token_temporal_scope": "bidirectional_full_window",
            "evaluator_temporal_mode": EVALUATOR_TEMPORAL_MODE,
            "claim_status": CLAIM_STATUS if source_manifest is not None else "synthetic_fixture_offline_delayed_association",
            "fresh_fit_held_out": False,
            "causal_future_claim": False,
            "fit_policy": (
                "synthetic_matched_fit_rows_only"
                if synthetic_mode
                else "matched_fit_rows_only_from_fit_source"
            ),
            "fit_latent_policy": (
                "synthetic arrays; no checkpoint or encoder"
                if synthetic_mode
                else "saved_checkpoint_eval_only_forward_pass_for_fit_source; no optimizer or encoder retraining"
            ),
            "fit_latent_export_present": bool(synthetic_mode),
            "fit_latent_source": (
                "fixture fit_eeg_shared/fit_fnirs_shared arrays"
                if fixture_path is not None
                else (
                    "deterministic synthetic generated arrays"
                    if synthetic_mode
                    else "source_samples.npz plus declared cell best.pt checkpoint"
                )
            ),
            "inputs": input_records,
            "evaluation_unit": "subject",
            "token_or_window_as_biological_replicate": False,
            "lag_bank_seconds": list(LAG_SECONDS),
            "lag_bank_tokens": list(LAG_TOKENS),
            "token_step_seconds": TOKEN_STEP_SECONDS,
            "lag_sign_convention": "lag_seconds=target_fnirs_time-source_eeg_time; positive means a later fNIRS token in offline association, not future/causal prediction",
            "negative_lag_explicit": True,
            "evaluation_conditions": list(CONDITIONS),
            "null_registration": {
                "matched": "matched_same_trial",
                "deranged_same_subject_same_condition_nonidentity": "same_subject_same_condition_nonidentity_trial_donor",
                "within_trial_circular_shift": f"within_trial_circular_shift_tokens_{int(config.get('nulls', {}).get('within_trial_circular_shift_tokens', 1))}",
            },
            "derangement_nonoverlap_verified": bool(synthetic_mode),
            "derangement_window_overlap_policy": (
                "synthetic fixture; measured-window overlap is not applicable"
                if synthetic_mode
                else "legacy exports omit event timestamps; nonidentity is verified but window nonoverlap is not"
            ),
            "forbidden_model_fields_not_used": list(FORBIDDEN_MODEL_FIELDS),
            "target_mask_use": "validity_only; no target values or target/driver predictions",
            "source_representations": ["eeg_shared", "eeg_native_patch_features"],
            "target_representations": ["fnirs_shared", "fnirs_native_patch_features"],
            "target_type": "representation_to_representation_association",
            "not_raw_fnirs_prediction": True,
            "shared_latent_teacher_supervision_caveat": True,
            "lag_unit": "seconds",
            "matched_advantage": "matched delta_r2 minus each null-condition delta_r2, subject-wise",
            "baseline_policy": "train-fit matched target mean; delta_r2=1-model_sse/baseline_sse within subject",
            "mask_policy": "source_and_shifted_target_token_masks plus all feature coordinates and all target points",
            "source_run": (
                str(source_run.relative_to(REPO_ROOT))
                if source_run.is_relative_to(REPO_ROOT)
                else str(source_run)
            ) if source_manifest is not None else None,
            "source_manifest_sha256": _sha256(source_run / "manifest.json") if source_manifest is not None else None,
            "source_source_schema": SOURCE_SCHEMA,
            "source_prediction_schema": VALIDATION_SCHEMA,
            "fixture": str(fixture_path) if fixture_path is not None else None,
            "cell_count": len(saved_cells),
            "metrics_row_count": len(all_metrics),
            "summary_row_count": len(_subject_summary(all_metrics)),
            "advantage_row_count": len(all_advantage),
            "feature_extraction": "src.analysis.physiological_patch_features; canonical robust-SD signals, with validation de-normalized using fit-only statistics",
            "feature_manifests": feature_manifests,
            "fit_subjects_by_cell": {
                f"{item.get('task_id')}|seed={item.get('seed')}": item.get("fit_subjects", [])
                for item in feature_manifests
            },
            "validation_subjects_by_cell": {
                f"{item.get('task_id')}|seed={item.get('seed')}": item.get("validation_subjects", [])
                for item in feature_manifests
            },
            "source_prediction_hashes": [
                {
                    "task_id": item.get("task_id"),
                    "dataset_id": item.get("dataset_id"),
                    "seed": item.get("seed"),
                    "path": str(Path(item["source_cell"]) / "validation_predictions.npz"),
                    "sha256": item.get("validation_prediction_sha256"),
                }
                for item in feature_manifests
            ],
            "limitations": (
                [
                    "Synthetic known-lag smoke data validate implementation wiring only and provide no empirical physiology evidence.",
                    "No legacy checkpoint, measured subject, or protected cohort is opened in synthetic mode.",
                    "Token pairs are used for ridge fitting, but reported metrics remain subject-pooled.",
                ]
                if synthetic_mode
                else [
                    "The legacy best.pt checkpoint was selected on development validation loss for subjects 19--23; reported 19--23 results are post-selection development, not fresh fit-to-held-out evidence.",
                    "Fit latents are a manifest-bound derived export from source_samples.npz and the frozen checkpoint; the original full run did not persist train_collected latents.",
                    "The saved continuous encoder has bidirectional full-window token context; positive lag denotes later-token offline delayed association, not causal/future prediction.",
                    "The probe uses no target values, eeg_driver, or fnirs_driver predictions; target_mask is validity-only. Shared latent targets remain representation associations influenced by shared-driver teacher supervision, not raw fNIRS prediction.",
                    "Native patch features use canonical robust-SD signals; validation raw observations are reconstructed with fit-only normalization statistics and are not physical voltage/concentration units.",
                    "Token pairs are used for ridge fitting, but all reported metrics and advantages are subject-pooled and never treat tokens/windows as biological replicates.",
                    "The legacy exports omit event timestamps, so the deranged null verifies same-subject/same-condition nonidentity but cannot verify that two 20 s source windows do not overlap; no nonoverlap claim is made for this legacy diagnostic.",
                    "Every lag/condition records shifted pair masks, support counts, mask hashes, and null policy; no lag is selected from validation.",
                    "Development validation is the only reported evaluation split; no protected cohort is opened.",
                ]
            ),
            "artifacts": [],
        }
        manifest["artifacts"] = _artifact_inventory(staging)
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, output)
        return output
    except Exception:
        print(f"failed staging retained at {staging}", file=sys.stderr)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "experiments/configs/physiology_semantic_tokenizer/probe_existing_lagged_predictability.yaml",
    )
    parser.add_argument("--source-run", type=Path, help="override the full continuous shared/private export")
    parser.add_argument("--fixture", type=Path, help="combined fixture.npz or fixture directory")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true", help="run deterministic synthetic smoke data when no fixture is supplied")
    parser.add_argument(
        "--real-smoke",
        action="store_true",
        help="evaluate one real legacy cell with configured row caps",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
