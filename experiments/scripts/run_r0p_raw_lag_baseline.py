#!/usr/bin/env python3
"""Preregistered raw EEG-fNIRS lag association baseline.

This is a measurement-layer, same-trial offline association analysis.  It is
not a token analysis, a directed prediction test, or causal evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import rankdata


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.shared_driver_dataset import SharedDriverWindowDataset  # noqa: E402


BANDS = ("theta", "alpha", "beta")
CHROMOPHORES = ("HbO", "HbR")
BAND_EDGES_HZ = ((4.0, 7.0), (8.0, 13.0), (14.0, 30.0))
LAG_TOKENS = np.asarray([1, 2, 3, 4, 5], dtype=np.int64)
LAG_SECONDS = LAG_TOKENS.astype(np.float64) * 2.0
PATCH_COUNT = 10
EEG_PATCH_SAMPLES = 400
FNIRS_PATCH_SAMPLES = 20
TRAIN_SUBJECTS = tuple(f"subject_{value:02d}" for value in range(1, 19))
VALIDATION_SUBJECTS = tuple(f"subject_{value:02d}" for value in range(19, 24))
PROTECTED_SUBJECTS = frozenset(f"subject_{value:02d}" for value in range(24, 30))
EXPECTED_PREREGISTRY_ID = "R0P-RAW-LAG-BASELINE-20260728"
CLAIM_CEILING = "offline_raw_association_only"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_subject(value: str) -> str:
    text = str(value)
    if "|" in text:
        text = text.rsplit("|", 1)[-1]
    if text.startswith("subject_"):
        return text
    if text.isdigit():
        return f"subject_{int(text):02d}"
    raise ValueError(f"Unsupported subject identifier: {value!r}")


def assert_development_subjects(subjects: Iterable[str]) -> tuple[str, ...]:
    """Fail before constructing a loader if a protected subject is requested."""

    canonical = tuple(_canonical_subject(value) for value in subjects)
    forbidden = sorted(set(canonical).intersection(PROTECTED_SUBJECTS))
    if forbidden:
        raise PermissionError(
            "R0-P raw lag baseline refuses protected subjects before array "
            f"dereference: {forbidden}"
        )
    allowed = set(TRAIN_SUBJECTS + VALIDATION_SUBJECTS)
    unknown = sorted(set(canonical).difference(allowed))
    if unknown:
        raise ValueError(f"Subjects are outside the registered development split: {unknown}")
    return canonical


def load_preregistry(path: str | Path) -> dict[str, Any]:
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    if registry.get("registry_id") != EXPECTED_PREREGISTRY_ID:
        raise ValueError("Unexpected R0-P preregistry identifier")
    if registry.get("scope", {}).get("claim_ceiling") != CLAIM_CEILING:
        raise ValueError("R0-P claim ceiling must remain offline_raw_association_only")
    feature = registry.get("feature_contract", {})
    observed_edges = tuple(
        tuple(float(value) for value in feature["eeg_bands_hz"][name])
        for name in BANDS
    )
    if observed_edges != BAND_EDGES_HZ:
        raise ValueError("Registered EEG bands differ from the frozen implementation")
    lag_tokens = tuple(int(value) for value in registry["lag_contract"]["lags_tokens"])
    if lag_tokens != tuple(LAG_TOKENS.tolist()):
        raise ValueError("Registered lag family differs from the frozen implementation")
    primary = registry["primary_estimand"]
    if primary.get("feature_pair") != ["alpha", "HbO"]:
        raise ValueError("Primary feature pair is not the registered alpha-HbO pair")
    if primary.get("registered_direction") != "negative Spearman association":
        raise ValueError("Primary direction differs from the preregistered direction")
    if bool(registry["data_contract"].get("protected_array_dereference_allowed", True)):
        raise ValueError("Protected array access must remain disabled")
    return registry


def validate_bundle_before_array_load(bundle_root: str | Path) -> dict[str, Any]:
    """Validate the development boundary using manifests only."""

    root = Path(bundle_root)
    bundle_manifest_path = root / "manifest.json"
    raw_manifest_path = root / "raw_view_registry" / "manifest.json"
    if not bundle_manifest_path.is_file() or not raw_manifest_path.is_file():
        raise FileNotFoundError("R1-P bundle/raw-view manifest is missing")
    bundle = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    if bundle.get("schema") != "shared_driver_r1p_population_frozen_bundle_v1":
        raise ValueError("Raw lag baseline requires the R1-P population-frozen bundle")
    if bundle.get("teacher_scope") != "population_frozen":
        raise ValueError("R1-P teacher scope must be population_frozen")
    for manifest in (bundle, raw):
        if bool(manifest.get("protected_open", False)):
            raise PermissionError("Protected split is open")
        if bool(manifest.get("protected_test_included", False)):
            raise PermissionError("Development artifact contains protected samples")
    if raw.get("selected_fnirs_channels") != ["FC3FC5_HbO", "FC3FC5_HbR"]:
        raise ValueError("Frozen fNIRS raw view must be ordered [HbO,HbR]")
    if len(raw.get("selected_eeg_channels", [])) != 6:
        raise ValueError("Frozen EEG raw view must contain exactly six channels")
    return {
        "bundle_manifest": bundle,
        "raw_manifest": raw,
        "bundle_manifest_sha256": _sha256(bundle_manifest_path),
        "raw_manifest_sha256": _sha256(raw_manifest_path),
    }


@dataclass(frozen=True)
class TrialFeatures:
    sample_key: str
    subject: str
    session: str
    condition: str
    eeg: np.ndarray  # [patch, band]
    fnirs: np.ndarray  # [patch, chromophore]


@dataclass(frozen=True)
class PreparedCell:
    subject: str
    session: str
    condition: str
    sample_keys: tuple[str, ...]
    eeg: np.ndarray  # [trial, patch, band]
    fnirs: np.ndarray  # [trial, patch, chromophore]
    eeg_rank_unit: np.ndarray
    fnirs_rank_unit: np.ndarray


def extract_patch_features(
    eeg: np.ndarray,
    fnirs: np.ndarray,
    *,
    eeg_valid_patches: np.ndarray | None = None,
    fnirs_valid_patches: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the frozen 2-second measurement feature contract."""

    eeg = np.asarray(eeg, dtype=np.float64)
    fnirs = np.asarray(fnirs, dtype=np.float64)
    if eeg.shape != (6, PATCH_COUNT * EEG_PATCH_SAMPLES):
        raise ValueError(f"Expected EEG [6,4000], got {eeg.shape}")
    if fnirs.shape != (2, PATCH_COUNT * FNIRS_PATCH_SAMPLES):
        raise ValueError(f"Expected fNIRS [2,200], got {fnirs.shape}")
    eeg_valid = (
        np.ones(PATCH_COUNT, dtype=bool)
        if eeg_valid_patches is None
        else np.asarray(eeg_valid_patches, dtype=bool)
    )
    fnirs_valid = (
        np.ones(PATCH_COUNT, dtype=bool)
        if fnirs_valid_patches is None
        else np.asarray(fnirs_valid_patches, dtype=bool)
    )
    if eeg_valid.shape != (PATCH_COUNT,) or fnirs_valid.shape != (PATCH_COUNT,):
        raise ValueError("Patch-valid masks must have shape [10]")

    patches = eeg.reshape(6, PATCH_COUNT, EEG_PATCH_SAMPLES)
    patches = patches - patches.mean(axis=-1, keepdims=True)
    hann = np.hanning(EEG_PATCH_SAMPLES + 1)[:-1]
    spectrum = np.abs(np.fft.rfft(patches * hann, axis=-1)) ** 2
    frequencies = np.fft.rfftfreq(EEG_PATCH_SAMPLES, d=1.0 / 200.0)
    eeg_features = np.empty((PATCH_COUNT, len(BANDS)), dtype=np.float64)
    for band_index, (low_hz, high_hz) in enumerate(BAND_EDGES_HZ):
        selected = (frequencies >= low_hz) & (frequencies <= high_hz)
        channel_log_power = np.log(
            np.maximum(spectrum[..., selected].sum(axis=-1), 1e-12)
        )
        eeg_features[:, band_index] = channel_log_power.mean(axis=0)
    fnirs_features = fnirs.reshape(2, PATCH_COUNT, FNIRS_PATCH_SAMPLES).mean(axis=-1).T
    eeg_features[~eeg_valid] = np.nan
    fnirs_features[~fnirs_valid] = np.nan
    return eeg_features, fnirs_features


def _rank_unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(
            "Formal vectorized permutation requires finite features in every "
            "registered trial/patch; report coverage before proceeding"
        )
    ranked = rankdata(values, axis=0, method="average")
    centered = ranked - ranked.mean(axis=0, keepdims=True)
    norm = np.sqrt(np.sum(centered * centered, axis=0, keepdims=True))
    unit = np.divide(
        centered,
        norm,
        out=np.full_like(centered, np.nan),
        where=norm > 0,
    )
    return unit


def prepare_cells(
    trials: Sequence[TrialFeatures],
    *,
    expected_trials: int = 10,
) -> list[PreparedCell]:
    grouped: dict[tuple[str, str, str], list[TrialFeatures]] = defaultdict(list)
    for trial in trials:
        if _canonical_subject(trial.subject) in PROTECTED_SUBJECTS:
            raise PermissionError("Protected trial reached cell preparation")
        grouped[(trial.subject, trial.session, trial.condition)].append(trial)
    cells: list[PreparedCell] = []
    for (subject, session, condition), values in sorted(grouped.items()):
        values = sorted(values, key=lambda item: item.sample_key)
        if len(values) != expected_trials:
            raise ValueError(
                f"Expected {expected_trials} trials in {(subject, session, condition)}, "
                f"got {len(values)}"
            )
        eeg = np.stack([value.eeg for value in values])
        fnirs = np.stack([value.fnirs for value in values])
        cells.append(
            PreparedCell(
                subject=subject,
                session=session,
                condition=condition,
                sample_keys=tuple(value.sample_key for value in values),
                eeg=eeg,
                fnirs=fnirs,
                eeg_rank_unit=_rank_unit(eeg),
                fnirs_rank_unit=_rank_unit(fnirs),
            )
        )
    if not cells:
        raise ValueError("No trial cells were prepared")
    return cells


def _fisher_z(correlation: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(correlation, -1.0 + 1e-7, 1.0 - 1e-7))


def compute_observed(
    cells: Sequence[PreparedCell],
) -> tuple[list[dict[str, Any]], np.ndarray, tuple[str, ...]]:
    """Return cell rows and subject-equal [subject,lag,band,chrom] estimates."""

    subjects = tuple(sorted({cell.subject for cell in cells}))
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    sums = np.zeros((len(subjects), len(LAG_TOKENS), 3, 2), dtype=np.float64)
    counts = np.zeros_like(sums, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for cell in cells:
        correlations = np.einsum(
            "nsb,ntc->stbc",
            cell.eeg_rank_unit,
            cell.fnirs_rank_unit,
            optimize=True,
        )
        for lag_index, lag in enumerate(LAG_TOKENS):
            for source_patch in range(PATCH_COUNT - int(lag)):
                z_values = _fisher_z(correlations[source_patch, source_patch + lag])
                finite = np.isfinite(z_values)
                index = subject_index[cell.subject]
                sums[index, lag_index][finite] += z_values[finite]
                counts[index, lag_index][finite] += 1
                for band_index, band in enumerate(BANDS):
                    for chrom_index, chromophore in enumerate(CHROMOPHORES):
                        value = z_values[band_index, chrom_index]
                        rows.append(
                            {
                                "subject": cell.subject,
                                "session": cell.session,
                                "condition": cell.condition,
                                "source_patch": source_patch,
                                "target_patch": source_patch + int(lag),
                                "lag_tokens": int(lag),
                                "lag_seconds": int(lag * 2),
                                "band": band,
                                "chromophore": chromophore,
                                "n_trials": len(cell.sample_keys),
                                "fisher_z": float(value) if np.isfinite(value) else math.nan,
                            }
                        )
    subject_values = np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan),
        where=counts > 0,
    )
    return rows, subject_values, subjects


def registered_auc(curves: np.ndarray) -> np.ndarray:
    """Sign-aligned alpha-HbO AUC, normalized by the registered 8-s width."""

    values = -np.asarray(curves)[..., 1, 0]
    return np.trapezoid(values, x=LAG_SECONDS, axis=-1) / (
        LAG_SECONDS[-1] - LAG_SECONDS[0]
    )


def permute_fnirs_trials(
    cell: PreparedCell, permutation: Sequence[int]
) -> PreparedCell:
    """Synthetic/audit helper: preserve full fNIRS trials but change pairing."""

    order = np.asarray(permutation, dtype=np.int64)
    if sorted(order.tolist()) != list(range(len(cell.sample_keys))):
        raise ValueError("permutation must reorder every trial exactly once")
    fnirs = cell.fnirs[order].copy()
    return PreparedCell(
        subject=cell.subject,
        session=cell.session,
        condition=cell.condition,
        sample_keys=cell.sample_keys,
        eeg=cell.eeg.copy(),
        fnirs=fnirs,
        eeg_rank_unit=cell.eeg_rank_unit.copy(),
        fnirs_rank_unit=cell.fnirs_rank_unit[order].copy(),
    )


def permutation_null(
    cells: Sequence[PreparedCell],
    *,
    permutations: int,
    seed: int,
    batch_size: int = 250,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Within-cell trial-pairing null for split-level subject-equal estimates."""

    if permutations < 1 or batch_size < 1:
        raise ValueError("permutations and batch_size must be positive")
    subjects = tuple(sorted({cell.subject for cell in cells}))
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    member_null = np.empty((permutations, len(LAG_TOKENS), 3, 2), dtype=np.float32)
    primary_null = np.empty(permutations, dtype=np.float32)
    max_abs_null = np.empty(permutations, dtype=np.float32)
    rng = np.random.default_rng(seed)
    written = 0
    while written < permutations:
        current = min(batch_size, permutations - written)
        sums = np.zeros(
            (current, len(subjects), len(LAG_TOKENS), 3, 2),
            dtype=np.float64,
        )
        counts = np.zeros_like(sums, dtype=np.int32)
        for cell in cells:
            n_trials = len(cell.sample_keys)
            orders = np.argsort(rng.random((current, n_trials)), axis=1)
            permuted_fnirs = cell.fnirs_rank_unit[orders]
            correlations = np.einsum(
                "nsb,pntc->pstbc",
                cell.eeg_rank_unit,
                permuted_fnirs,
                optimize=True,
            )
            subject_slot = subject_index[cell.subject]
            for lag_index, lag in enumerate(LAG_TOKENS):
                diagonal = np.stack(
                    [
                        correlations[:, source, source + int(lag)]
                        for source in range(PATCH_COUNT - int(lag))
                    ],
                    axis=1,
                )
                z_values = _fisher_z(diagonal)
                finite = np.isfinite(z_values)
                sums[:, subject_slot, lag_index] += np.where(
                    finite, z_values, 0.0
                ).sum(axis=1)
                counts[:, subject_slot, lag_index] += finite.sum(axis=1)
        subject_values = np.divide(
            sums,
            counts,
            out=np.full_like(sums, np.nan),
            where=counts > 0,
        )
        split_values = np.nanmean(subject_values, axis=1)
        member_null[written : written + current] = split_values.astype(np.float32)
        primary_null[written : written + current] = registered_auc(split_values).astype(
            np.float32
        )
        max_abs_null[written : written + current] = np.nanmax(
            np.abs(split_values), axis=(1, 2, 3)
        ).astype(np.float32)
        written += current
    return member_null, primary_null, max_abs_null


def permutation_pvalues(
    observed_members: np.ndarray,
    member_null: np.ndarray,
    max_abs_null: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(observed_members, dtype=np.float64)
    null = np.asarray(member_null, dtype=np.float64)
    max_null = np.asarray(max_abs_null, dtype=np.float64)
    if null.shape[1:] != observed.shape:
        raise ValueError("Member null and observed family shapes differ")
    denominator = null.shape[0] + 1.0
    unadjusted = (
        1.0 + np.sum(np.abs(null) >= np.abs(observed)[None, ...], axis=0)
    ) / denominator
    adjusted = (
        1.0
        + np.sum(
            max_null[:, None, None, None] >= np.abs(observed)[None, ...],
            axis=0,
        )
    ) / denominator
    return unadjusted, adjusted


def bootstrap_subjects(
    subject_values: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(subject_values, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError("subject_values must be [subject,lag,band,chromophore]")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.shape[0], size=(iterations, values.shape[0]))
    curves = np.nanmean(values[indices], axis=1)
    aucs = registered_auc(curves)
    return curves, aucs


def _subject_keys(subjects: Sequence[str]) -> tuple[str, ...]:
    canonical = assert_development_subjects(subjects)
    return tuple(f"eeg_fnirs_single_trial|{value}" for value in canonical)


def load_trial_features(
    *,
    cache_root: str | Path,
    raw_registry_root: str | Path,
    subjects: Sequence[str],
    eeg_signal_branch: str,
    window_duration_s: float,
    window_offset_s: float,
) -> list[TrialFeatures]:
    """Read measured arrays only after the split allowlist has passed."""

    subject_keys = _subject_keys(subjects)
    dataset = SharedDriverWindowDataset(
        cache_root=str(cache_root),
        raw_view_registry_root=str(raw_registry_root),
        trajectory_sidecar_root=None,
        require_trajectory_target=False,
        restrict_to_registered_views=True,
        dataset_ids=("eeg_fnirs_single_trial",),
        subject_keys=subject_keys,
        window_duration_s=float(window_duration_s),
        window_offset_s=float(window_offset_s),
        eeg_signal_branch=str(eeg_signal_branch),
    )
    trials: list[TrialFeatures] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if sample["subject"] not in set(subjects):
            raise RuntimeError("Loader returned a subject outside the explicit allowlist")
        eeg_features, fnirs_features = extract_patch_features(
            sample["eeg"].numpy(),
            sample["fnirs"].numpy(),
            eeg_valid_patches=sample["token_valid_mask"]["eeg"].numpy(),
            fnirs_valid_patches=sample["token_valid_mask"]["fnirs"].numpy(),
        )
        trials.append(
            TrialFeatures(
                sample_key=str(sample["target_sample_key"]),
                subject=str(sample["subject"]),
                session=str(sample["record_id"]),
                condition=str(sample["condition"]),
                eeg=eeg_features,
                fnirs=fnirs_features,
            )
        )
    return trials


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _coverage_rows(cells: Sequence[PreparedCell], split: str) -> list[dict[str, Any]]:
    return [
        {
            "split": split,
            "subject": cell.subject,
            "session": cell.session,
            "condition": cell.condition,
            "n_trials": len(cell.sample_keys),
            "finite_eeg_features": int(np.isfinite(cell.eeg).sum()),
            "total_eeg_features": int(cell.eeg.size),
            "finite_fnirs_features": int(np.isfinite(cell.fnirs).sum()),
            "total_fnirs_features": int(cell.fnirs.size),
        }
        for cell in cells
    ]


def _subject_rows(
    split: str, subjects: Sequence[str], values: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    subject_aucs = registered_auc(values)
    for subject_index, subject in enumerate(subjects):
        for lag_index, lag in enumerate(LAG_TOKENS):
            for band_index, band in enumerate(BANDS):
                for chrom_index, chromophore in enumerate(CHROMOPHORES):
                    rows.append(
                        {
                            "split": split,
                            "subject": subject,
                            "lag_tokens": int(lag),
                            "lag_seconds": int(lag * 2),
                            "band": band,
                            "chromophore": chromophore,
                            "fisher_z": float(
                                values[
                                    subject_index, lag_index, band_index, chrom_index
                                ]
                            ),
                            "registered_primary_auc": (
                                float(subject_aucs[subject_index])
                                if lag_index == 0
                                and band_index == 1
                                and chrom_index == 0
                                else ""
                            ),
                        }
                    )
    return rows


def _split_rows(
    split: str,
    observed: np.ndarray,
    subject_values: np.ndarray,
    boot_curves: np.ndarray,
    unadjusted: np.ndarray,
    adjusted: np.ndarray,
) -> list[dict[str, Any]]:
    lower, upper = np.nanpercentile(boot_curves, [2.5, 97.5], axis=0)
    rows: list[dict[str, Any]] = []
    for lag_index, lag in enumerate(LAG_TOKENS):
        for band_index, band in enumerate(BANDS):
            for chrom_index, chromophore in enumerate(CHROMOPHORES):
                rows.append(
                    {
                        "split": split,
                        "lag_tokens": int(lag),
                        "lag_seconds": int(lag * 2),
                        "band": band,
                        "chromophore": chromophore,
                        "fisher_z": float(observed[lag_index, band_index, chrom_index]),
                        "positive_subject_count": int(
                            np.sum(
                                subject_values[:, lag_index, band_index, chrom_index]
                                > 0.0
                            )
                        ),
                        "subject_count": int(subject_values.shape[0]),
                        "ci95_low": float(lower[lag_index, band_index, chrom_index]),
                        "ci95_high": float(upper[lag_index, band_index, chrom_index]),
                        "permutation_p_two_sided_unadjusted": float(
                            unadjusted[lag_index, band_index, chrom_index]
                        ),
                        "permutation_p_maxstat_fwer": float(
                            adjusted[lag_index, band_index, chrom_index]
                        ),
                        "analysis_status": "diagnostic_not_primary",
                    }
                )
    return rows


def create_figure(
    output_stem: Path,
    split_payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    colors = {"train": "#0072B2", "validation": "#D55E00"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    ax_curve, ax_train, ax_validation, ax_subject = axes.flat

    for split in ("train", "validation"):
        payload = split_payloads[split]
        observed = -payload["observed"][:, 1, 0]
        boot = -payload["boot_curves"][:, :, 1, 0]
        lower, upper = np.nanpercentile(boot, [2.5, 97.5], axis=0)
        ax_curve.plot(
            LAG_SECONDS,
            observed,
            marker="o",
            color=colors[split],
            label=f"{split} (n={len(payload['subjects'])})",
        )
        ax_curve.fill_between(
            LAG_SECONDS, lower, upper, color=colors[split], alpha=0.18, linewidth=0
        )
    ax_curve.axhline(0.0, color="#666666", linewidth=0.8, linestyle=":")
    ax_curve.set_xlabel("EEG→fNIRS lag (s)")
    ax_curve.set_ylabel("Sign-aligned Fisher z\n(−alpha–HbO)")
    ax_curve.legend(frameon=False, fontsize=7)
    ax_curve.set_title("Registered lag curve")

    for split, axis in (("train", ax_train), ("validation", ax_validation)):
        matrix = np.nanmean(split_payloads[split]["observed"], axis=0).T
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-0.12, vmax=0.12, aspect="auto")
        axis.set_xticks(range(3), BANDS)
        axis.set_yticks(range(2), CHROMOPHORES)
        axis.set_xlabel("EEG band")
        axis.set_title(f"{split.capitalize()}: mean over lags")
        for row in range(2):
            for column in range(3):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Fisher z")

    rng = np.random.default_rng(17)
    for split_index, split in enumerate(("train", "validation")):
        aucs = registered_auc(split_payloads[split]["subject_values"])
        x = split_index + rng.uniform(-0.08, 0.08, size=len(aucs))
        ax_subject.scatter(
            x,
            aucs,
            s=18,
            alpha=0.7,
            color=colors[split],
            edgecolor="white",
            linewidth=0.3,
        )
        mean = float(np.nanmean(aucs))
        ci = np.nanpercentile(split_payloads[split]["boot_aucs"], [2.5, 97.5])
        ax_subject.errorbar(
            split_index,
            mean,
            yerr=[[mean - ci[0]], [ci[1] - mean]],
            fmt="D",
            color="black",
            markersize=4,
            capsize=3,
            linewidth=1,
        )
    ax_subject.axhline(0.0, color="#666666", linewidth=0.8, linestyle=":")
    ax_subject.set_xticks([0, 1], ["Train", "Validation"])
    ax_subject.set_ylabel("Registered alpha–HbO AUC")
    ax_subject.set_title("Subjects are the biological units")

    for label, axis in zip("ABCD", axes.flat):
        axis.text(
            -0.16,
            1.08,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=7)
        axis.xaxis.label.set_size(8)
        axis.yaxis.label.set_size(8)
        axis.title.set_size(8)
    fig.suptitle(
        "Raw EEG–fNIRS trial-pairing lag baseline (offline association only)",
        fontsize=10,
    )
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["experiment"].get("claim_ceiling") != CLAIM_CEILING:
        raise ValueError("Configuration claim ceiling is not frozen")
    return config


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def run(config_path: str | Path, output_dir: str | Path) -> Path:
    config_path = _resolve(config_path)
    config = _load_config(config_path)
    preregistry_path = _resolve(config["experiment"]["preregistry"])
    preregistry = load_preregistry(preregistry_path)
    bundle_root = _resolve(config["data"]["r1p_bundle"])
    provenance = validate_bundle_before_array_load(bundle_root)

    inference = config["inference"]
    if int(inference["permutations"]) != int(
        preregistry["null_contract"]["formal_permutations"]
    ):
        raise ValueError("Formal permutation count differs from preregistration")
    if int(inference["bootstrap_iterations"]) != int(
        preregistry["uncertainty_and_reporting"]["subject_bootstrap_iterations"]
    ):
        raise ValueError("Bootstrap count differs from preregistration")
    output = _resolve(output_dir)
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True)

    payloads: dict[str, dict[str, Any]] = {}
    all_cell_rows: list[dict[str, Any]] = []
    all_subject_rows: list[dict[str, Any]] = []
    all_split_rows: list[dict[str, Any]] = []
    all_coverage_rows: list[dict[str, Any]] = []
    null_arrays: dict[str, np.ndarray] = {}
    split_specs = (
        ("train", TRAIN_SUBJECTS, 0),
        ("validation", VALIDATION_SUBJECTS, 1),
    )
    for split, subjects, seed_offset in split_specs:
        trials = load_trial_features(
            cache_root=_resolve(config["data"]["cache_root"]),
            raw_registry_root=bundle_root / "raw_view_registry",
            subjects=subjects,
            eeg_signal_branch=config["data"]["eeg_signal_branch"],
            window_duration_s=float(config["data"]["window_duration_s"]),
            window_offset_s=float(config["data"]["window_offset_s"]),
        )
        cells = prepare_cells(trials)
        cell_rows, subject_values, observed_subjects = compute_observed(cells)
        observed = np.nanmean(subject_values, axis=0)
        member_null, primary_null, max_abs_null = permutation_null(
            cells,
            permutations=int(inference["permutations"]),
            seed=int(inference["permutation_seed"]) + seed_offset,
            batch_size=int(inference["permutation_batch_size"]),
        )
        unadjusted, adjusted = permutation_pvalues(
            observed, member_null, max_abs_null
        )
        boot_curves, boot_aucs = bootstrap_subjects(
            subject_values,
            iterations=int(inference["bootstrap_iterations"]),
            seed=int(inference["bootstrap_seed"]) + seed_offset,
        )
        for row in cell_rows:
            row["split"] = split
        all_cell_rows.extend(cell_rows)
        all_subject_rows.extend(_subject_rows(split, observed_subjects, subject_values))
        all_split_rows.extend(
            _split_rows(
                split,
                observed,
                subject_values,
                boot_curves,
                unadjusted,
                adjusted,
            )
        )
        all_coverage_rows.extend(_coverage_rows(cells, split))
        null_arrays[f"{split}_member_null"] = member_null
        null_arrays[f"{split}_primary_auc_null"] = primary_null
        null_arrays[f"{split}_max_abs_fisher_z_null"] = max_abs_null
        observed_primary = float(registered_auc(observed[None, ...])[0])
        primary_p = float(
            (1.0 + np.sum(primary_null >= observed_primary))
            / (len(primary_null) + 1.0)
        )
        payloads[split] = {
            "subjects": observed_subjects,
            "subject_values": subject_values,
            "observed": observed,
            "boot_curves": boot_curves,
            "boot_aucs": boot_aucs,
            "primary_auc": observed_primary,
            "primary_ci95": np.nanpercentile(boot_aucs, [2.5, 97.5]),
            "primary_permutation_p_one_sided": primary_p,
        }

    _write_tsv(output / "cell_level_fisher_z.tsv", all_cell_rows)
    _write_tsv(output / "subject_level_lag_estimates.tsv", all_subject_rows)
    _write_tsv(output / "split_level_diagnostics.tsv", all_split_rows)
    _write_tsv(output / "data_coverage.tsv", all_coverage_rows)
    np.savez_compressed(output / "permutation_nulls.npz", **null_arrays)
    create_figure(output / "raw_lag_baseline", payloads)

    summary = {
        "schema": "r0p_raw_lag_baseline_summary_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "claim_ceiling": CLAIM_CEILING,
        "interpretation": (
            "Offline raw measurement association only; not token coupling, "
            "directed prediction, or causal evidence."
        ),
        "preregistry": {
            "id": preregistry["registry_id"],
            "path": str(preregistry_path),
            "sha256": _sha256(preregistry_path),
            "status": preregistry["status"],
        },
        "source": {
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
            **{
                key: value
                for key, value in provenance.items()
                if key.endswith("_sha256")
            },
        },
        "protected": {
            "open": False,
            "included": False,
            "array_dereference_count": 0,
            "subjects": sorted(PROTECTED_SUBJECTS),
        },
        "inference": {
            "permutations": int(inference["permutations"]),
            "bootstrap_iterations": int(inference["bootstrap_iterations"]),
            "maxstat_family_size": 30,
            "primary": "negative alpha-HbO normalized lag AUC",
        },
        "results": {
            split: {
                "subject_count": len(payloads[split]["subjects"]),
                "subjects": list(payloads[split]["subjects"]),
                "primary_auc": payloads[split]["primary_auc"],
                "primary_ci95": payloads[split]["primary_ci95"].tolist(),
                "primary_positive_subject_count": int(
                    np.sum(registered_auc(payloads[split]["subject_values"]) > 0.0)
                ),
                "primary_permutation_p_one_sided": payloads[split][
                    "primary_permutation_p_one_sided"
                ],
            }
            for split in ("train", "validation")
        },
        "outputs": {
            "cell_table": "cell_level_fisher_z.tsv",
            "subject_table": "subject_level_lag_estimates.tsv",
            "split_table": "split_level_diagnostics.tsv",
            "coverage_table": "data_coverage.tsv",
            "nulls": "permutation_nulls.npz",
            "figure_pdf": "raw_lag_baseline.pdf",
            "figure_svg": "raw_lag_baseline.svg",
            "figure_png": "raw_lag_baseline.png",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/configs/physiology_semantic_tokenizer/r0p_raw_lag_baseline.yaml",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = run(args.config, args.output_dir)
    print(output)


if __name__ == "__main__":
    main()
