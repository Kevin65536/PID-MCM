"""Statistics used by the coupling identifiability experiment suite."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


EPS = 1e-12


def occupancy_weighted_gauge(
    residual_logits: torch.Tensor,
    eeg_occupancy: torch.Tensor,
) -> torch.Tensor:
    """Remove EEG-independent fNIRS column bias from each lag slice."""
    if residual_logits.ndim != 3:
        raise ValueError("residual_logits must have shape [lag, eeg, fnirs]")
    if eeg_occupancy.shape != residual_logits.shape[:2]:
        raise ValueError("eeg_occupancy must have shape [lag, eeg]")
    weights = eeg_occupancy.clamp_min(0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    column_bias = torch.einsum("le,lef->lf", weights, residual_logits)
    return residual_logits - column_bias[:, None, :]


def effective_conditional_probabilities(
    residual_logits: torch.Tensor,
    fnirs_prior: torch.Tensor,
    eeg_occupancy: torch.Tensor,
) -> torch.Tensor:
    """Return q_lag(f|e) proportional to p0_lag(f) exp(R_lag,e,f)."""
    if fnirs_prior.shape != (residual_logits.shape[0], residual_logits.shape[2]):
        raise ValueError("fnirs_prior must have shape [lag, fnirs]")
    gauged = occupancy_weighted_gauge(residual_logits, eeg_occupancy)
    return F.softmax(gauged + fnirs_prior.clamp_min(1e-8).log()[:, None, :], dim=-1)


def conditional_probabilities_from_counts(
    counts: np.ndarray,
    *,
    alpha: float = 0.5,
    prior: np.ndarray | None = None,
) -> np.ndarray:
    """Dirichlet-smoothed P(fNIRS|EEG, lag) from [lag,eeg,fnirs] counts."""
    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("counts must have shape [lag, eeg, fnirs]")
    if prior is None:
        prior_values = np.ones((values.shape[0], values.shape[2]), dtype=np.float64)
        prior_values /= values.shape[2]
    else:
        prior_values = np.asarray(prior, dtype=np.float64)
        prior_values /= np.maximum(prior_values.sum(axis=-1, keepdims=True), EPS)
    smoothed = values + float(alpha) * prior_values[:, None, :]
    return smoothed / np.maximum(smoothed.sum(axis=-1, keepdims=True), EPS)


@dataclass(frozen=True)
class LagPairTable:
    eeg: np.ndarray
    fnirs: np.ndarray
    lag: np.ndarray
    sample: np.ndarray
    eeg_position: np.ndarray
    fnirs_position: np.ndarray


def build_lag_pair_table(
    eeg_tokens: np.ndarray,
    fnirs_tokens: np.ndarray,
    *,
    n_lags: int | None = None,
) -> LagPairTable:
    """Expand aligned token windows into a lag-balanced pair table."""
    eeg = np.asarray(eeg_tokens, dtype=np.int64)
    fnirs = np.asarray(fnirs_tokens, dtype=np.int64)
    if eeg.shape != fnirs.shape or eeg.ndim != 2:
        raise ValueError("eeg_tokens and fnirs_tokens must share shape [sample, token]")
    token_count = eeg.shape[1]
    n_lags = token_count if n_lags is None else min(int(n_lags), token_count)
    fields: Dict[str, list[np.ndarray]] = {
        "eeg": [], "fnirs": [], "lag": [], "sample": [],
        "eeg_position": [], "fnirs_position": [],
    }
    sample_index = np.arange(eeg.shape[0], dtype=np.int64)
    for lag in range(n_lags):
        valid = token_count - lag
        fields["eeg"].append(eeg[:, :valid].reshape(-1))
        fields["fnirs"].append(fnirs[:, lag:].reshape(-1))
        fields["lag"].append(np.full(eeg.shape[0] * valid, lag, dtype=np.int64))
        fields["sample"].append(np.repeat(sample_index, valid))
        fields["eeg_position"].append(np.tile(np.arange(valid, dtype=np.int64), eeg.shape[0]))
        fields["fnirs_position"].append(np.tile(np.arange(lag, token_count, dtype=np.int64), eeg.shape[0]))
    return LagPairTable(**{key: np.concatenate(value) for key, value in fields.items()})


def _joint_counts(eeg: np.ndarray, fnirs: np.ndarray, k_eeg: int, k_fnirs: int) -> np.ndarray:
    flat = np.asarray(eeg, dtype=np.int64) * k_fnirs + np.asarray(fnirs, dtype=np.int64)
    return np.bincount(flat, minlength=k_eeg * k_fnirs).reshape(k_eeg, k_fnirs).astype(np.float64)


def mutual_information_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    joint = counts / total
    expected = joint.sum(axis=1, keepdims=True) * joint.sum(axis=0, keepdims=True)
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / np.maximum(expected[mask], EPS))))


def lag_mutual_information(
    eeg_tokens: np.ndarray,
    fnirs_tokens: np.ndarray,
    *,
    k_eeg: int,
    k_fnirs: int,
    n_lags: int | None = None,
) -> np.ndarray:
    table = build_lag_pair_table(eeg_tokens, fnirs_tokens, n_lags=n_lags)
    lag_count = int(table.lag.max()) + 1 if table.lag.size else 0
    return np.asarray([
        mutual_information_from_counts(_joint_counts(
            table.eeg[table.lag == lag], table.fnirs[table.lag == lag], k_eeg, k_fnirs,
        ))
        for lag in range(lag_count)
    ])


def subject_block_bootstrap_gain(
    per_subject_model_nll: Mapping[int, float],
    per_subject_baseline_nll: Mapping[int, float],
    *,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, float]:
    subjects = sorted(set(per_subject_model_nll) & set(per_subject_baseline_nll))
    if not subjects:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    gains = np.asarray([
        per_subject_baseline_nll[subject] - per_subject_model_nll[subject]
        for subject in subjects
    ], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = gains[rng.integers(0, len(gains), size=(int(n_bootstrap), len(gains)))].mean(axis=1)
    return {
        "mean": float(gains.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
    }


def patch_features(signal: np.ndarray, *, sample_rate_hz: float, patch_size: int, eeg: bool) -> np.ndarray:
    """Extract fixed physiological patch descriptors from [B,C,T] signals."""
    values = np.asarray(signal, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] % patch_size:
        raise ValueError("signal must have shape [B,C,T] with T divisible by patch_size")
    patches = values.reshape(values.shape[0], values.shape[1], -1, patch_size).transpose(0, 2, 1, 3)
    mean = patches.mean(axis=-1)
    std = patches.std(axis=-1)
    rms = np.sqrt(np.mean(np.square(patches), axis=-1))
    slope_axis = np.linspace(-1.0, 1.0, patch_size, dtype=np.float32)
    slope_axis /= np.sum(slope_axis * slope_axis)
    slope = np.einsum("btcp,p->btc", patches, slope_axis)
    if not eeg:
        endpoint_delta = patches[..., -1] - patches[..., 0]
        return np.concatenate([mean, std, slope, endpoint_delta], axis=-1).astype(np.float32)

    spectrum = np.abs(np.fft.rfft(patches, axis=-1)) ** 2
    frequencies = np.fft.rfftfreq(patch_size, d=1.0 / float(sample_rate_hz))
    bands = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0))
    band_features = []
    for low, high in bands:
        mask = (frequencies >= low) & (frequencies < high)
        power = spectrum[..., mask].mean(axis=-1) if mask.any() else np.zeros_like(mean)
        band_features.append(np.log(np.maximum(power, 1e-8)))
    return np.concatenate([mean, std, rms, slope, *band_features], axis=-1).astype(np.float32)


def patch_features_torch(
    signal: torch.Tensor,
    *,
    sample_rate_hz: float,
    patch_size: int,
    eeg: bool,
) -> torch.Tensor:
    """GPU-friendly equivalent of :func:`patch_features` for [B,C,T] tensors."""
    if signal.ndim != 3 or signal.shape[-1] % patch_size:
        raise ValueError("signal must have shape [B,C,T] with T divisible by patch_size")
    values = signal.float()
    patches = values.reshape(values.shape[0], values.shape[1], -1, patch_size).permute(0, 2, 1, 3)
    mean = patches.mean(dim=-1)
    std = patches.std(dim=-1, correction=0)
    rms = patches.square().mean(dim=-1).sqrt()
    slope_axis = torch.linspace(-1.0, 1.0, patch_size, device=values.device, dtype=values.dtype)
    slope_axis = slope_axis / slope_axis.square().sum()
    slope = torch.einsum("btcp,p->btc", patches, slope_axis)
    if not eeg:
        endpoint_delta = patches[..., -1] - patches[..., 0]
        return torch.cat([mean, std, slope, endpoint_delta], dim=-1)

    spectrum = torch.fft.rfft(patches, dim=-1).abs().square()
    frequencies = torch.fft.rfftfreq(
        patch_size,
        d=1.0 / float(sample_rate_hz),
        device=values.device,
    )
    band_features = []
    for low, high in ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0)):
        mask = (frequencies >= low) & (frequencies < high)
        power = spectrum[..., mask].mean(dim=-1) if bool(mask.any()) else torch.zeros_like(mean)
        band_features.append(power.clamp_min(1e-8).log())
    return torch.cat([mean, std, rms, slope, *band_features], dim=-1)


def load_export_split(export_dir: str | Path, split: str) -> Dict[str, np.ndarray]:
    """Load either a legacy monolithic export or a streaming sharded export."""
    root = Path(export_dir)
    legacy_path = root / f"{split}.npz"
    if legacy_path.exists():
        with np.load(legacy_path, allow_pickle=False) as payload:
            return {key: payload[key] for key in payload.files}

    manifest_path = root / f"{split}.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks: Dict[str, list[np.ndarray]] = {}
    for relative_path in manifest["shards"]:
        with np.load(root / relative_path, allow_pickle=False) as payload:
            for key in payload.files:
                chunks.setdefault(key, []).append(payload[key])
    return {key: np.concatenate(values, axis=0) for key, values in chunks.items()}


def loso_ridge_scores(
    x: np.ndarray,
    y: np.ndarray,
    subjects: Sequence[int],
    *,
    alpha: float = 10.0,
) -> Dict[str, object]:
    """Subject-macro LOSO ridge R2 with train-only standardization."""
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    subject_values = np.asarray(subjects)
    scores: Dict[int, float] = {}
    for subject in np.unique(subject_values):
        test = subject_values == subject
        train = ~test
        if train.sum() < 2 or test.sum() < 1:
            continue
        x_scaler = StandardScaler().fit(x_values[train])
        y_scaler = StandardScaler().fit(y_values[train])
        model = Ridge(alpha=float(alpha)).fit(
            x_scaler.transform(x_values[train]), y_scaler.transform(y_values[train]),
        )
        prediction = y_scaler.inverse_transform(model.predict(x_scaler.transform(x_values[test])))
        scores[int(subject)] = float(r2_score(y_values[test], prediction, multioutput="variance_weighted"))
    values = np.asarray(list(scores.values()), dtype=np.float64)
    return {
        "subject_scores": scores,
        "mean": float(values.mean()) if values.size else float("nan"),
        "median": float(np.median(values)) if values.size else float("nan"),
    }


def gaussian_conditional_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    nuisance: np.ndarray,
    *,
    ridge: float = 1e-4,
) -> float:
    """Gaussian CMI proxy after linear residualization against nuisance."""
    x_values = StandardScaler().fit_transform(np.asarray(x, dtype=np.float64))
    y_values = StandardScaler().fit_transform(np.asarray(y, dtype=np.float64))
    nuisance_values = np.asarray(nuisance)
    if nuisance_values.ndim == 1:
        nuisance_values = nuisance_values[:, None]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    design = encoder.fit_transform(nuisance_values)
    design = np.concatenate([np.ones((design.shape[0], 1)), design], axis=1)
    beta_x = np.linalg.lstsq(design, x_values, rcond=None)[0]
    beta_y = np.linalg.lstsq(design, y_values, rcond=None)[0]
    x_res = x_values - design @ beta_x
    y_res = y_values - design @ beta_y
    covariance = np.cov(np.concatenate([x_res, y_res], axis=1), rowvar=False)
    dx = x_res.shape[1]
    cxx = covariance[:dx, :dx] + ridge * np.eye(dx)
    cyy = covariance[dx:, dx:] + ridge * np.eye(y_res.shape[1])
    cxy = covariance[:dx, dx:]
    whitening_x = np.linalg.inv(np.linalg.cholesky(cxx))
    whitening_y = np.linalg.inv(np.linalg.cholesky(cyy))
    canonical = np.linalg.svd(whitening_x @ cxy @ whitening_y.T, compute_uv=False)
    canonical = np.clip(canonical, 0.0, 1.0 - 1e-8)
    return float(-0.5 * np.log1p(-np.square(canonical)).sum())
