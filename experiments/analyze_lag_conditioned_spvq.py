#!/usr/bin/env python3
"""Analyze LC-SPVQ token exports without refitting representation encoders.

The analysis is boundary-safe and fit-parameter scoped: token display orders and
q0/q1 categorical probes are fitted only on ``fit_parameter`` exports. Reused
subjects 19--23 are labelled post-selection development throughout. Independent
EEG/fNIRS codebooks retain modality-specific IDs; no same-ID interpretation or
cross-variant code alignment is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.lagged_token_coupling import (
    apply_token_order,
    build_lagged_categorical_rows,
    evaluate_q0_q1_by_subject,
    fit_q0_q1,
    fit_train_only_token_order,
    matched_minus_deranged_expected_residual_log_lift,
    soft_cooccurrence_tensor,
    subject_block_bootstrap,
    top_pair_jaccard,
    top_pair_stability,
)

SCHEMA = "lag_conditioned_spvq_analysis_v1"
ROLES = ("fit_parameter", "fit_selection", "development_apply")
LAGS = (0, 1, 2, 3, 4, 5)
LAG_SECONDS = tuple(2 * value for value in LAGS)
CLAIM_STATUS = "exploratory_reused_post_selection_development_smoke"
PROBE_TARGET_LABEL_SMOOTHING = 0.05
PROBE_L2 = 1.0
PROBE_MAX_ITER = 5000
PROBE_TOL = 1e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    names: list[str] = []
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows({name: _jsonable(row.get(name, "")) for name in names} for row in rows)


def _inventory(root: Path) -> list[dict[str, Any]]:
    output = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == root / "manifest.json":
            continue
        output.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return output


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    schema = str(values.get("schema", np.asarray("")).item())
    if schema not in {"lc_spvq_token_exports_v2", "lc_spvq_token_exports_v3"}:
        raise ValueError(f"{path} is not an LC-SPVQ token export")
    if bool(values["protected_open"].item()):
        raise PermissionError("token export opens a protected cohort")
    if bool(values["development_is_new_independent_holdout"].item()):
        raise ValueError("reused development cannot be labelled independent")
    if not bool(values.get("derangement_nonoverlap_verified", np.asarray(False)).item()):
        raise ValueError("token export lacks verified nonoverlapping derangements")
    expected_negative = (
        "same_subject_condition_nonidentity_same_token_time"
        if schema == "lc_spvq_token_exports_v2"
        else "same_subject_condition_nonidentity_lag_endpoint_aligned"
    )
    observed_negative = str(
        values.get("registered_hard_negative_policy", np.asarray("")).item()
    )
    if observed_negative != expected_negative:
        raise ValueError("token export hard-negative policy differs from the contract")
    return values


def _role(values: Mapping[str, np.ndarray], role: str) -> dict[str, np.ndarray]:
    prefix = role + "__"
    output = {
        name[len(prefix) :]: np.asarray(value)
        for name, value in values.items()
        if name.startswith(prefix)
    }
    required = {
        "eeg_posterior",
        "fnirs_posterior",
        "eeg_token_valid_mask",
        "fnirs_token_valid_mask",
        "subject",
        "condition",
        "record_id",
        "eeg_event_time_ms",
        "fnirs_event_time_ms",
        "donor_index",
        "sample_id",
    }
    if not required.issubset(output):
        raise KeyError(f"{role} export lacks {sorted(required - set(output))}")
    eeg = output["eeg_posterior"]
    fnirs = output["fnirs_posterior"]
    if eeg.ndim != 3 or fnirs.ndim != 3 or eeg.shape[:2] != fnirs.shape[:2]:
        raise ValueError(f"{role} posterior shapes differ")
    if eeg.shape[-1] != 16 or fnirs.shape[-1] != 16:
        raise ValueError("primary analysis requires independent K16 codebooks")
    sample_id = output["sample_id"].astype(str)
    if len(set(sample_id.tolist())) != len(eeg):
        raise ValueError(f"{role} sample IDs are not unique")
    donor_raw = np.asarray(output["donor_index"])
    if donor_raw.shape != (len(eeg),) or not np.issubdtype(
        donor_raw.dtype, np.integer
    ):
        raise ValueError(f"{role} donor index must be an integer vector")
    donor = donor_raw.astype(np.int64, copy=False)
    if np.any(donor < 0) or np.any(donor >= len(eeg)):
        raise ValueError(f"{role} donor index is invalid")
    if np.unique(donor).size != len(eeg):
        raise ValueError(f"{role} donor index is not a permutation")
    if np.any(donor == np.arange(len(donor))):
        raise ValueError(f"{role} donor index contains identity pairs")
    if np.any(output["subject"].astype(str) != output["subject"].astype(str)[donor]):
        raise ValueError(f"{role} donor changed subject")
    if np.any(output["condition"].astype(str) != output["condition"].astype(str)[donor]):
        raise ValueError(f"{role} donor changed condition")
    record = output["record_id"].astype(str)
    eeg_time = output["eeg_event_time_ms"].astype(np.float64)
    fnirs_time = output["fnirs_event_time_ms"].astype(np.float64)
    if eeg_time.shape != (len(eeg),) or fnirs_time.shape != (len(eeg),):
        raise ValueError(f"{role} donor timing vectors are invalid")
    if np.any(~np.isfinite(eeg_time)) or np.any(~np.isfinite(fnirs_time)):
        raise ValueError(f"{role} donor timing contains non-finite values")
    same_record = record == record[donor]
    overlap = (np.abs(eeg_time - eeg_time[donor]) < 20_000.0) | (
        np.abs(fnirs_time - fnirs_time[donor]) < 20_000.0
    )
    if np.any(same_record & overlap):
        raise ValueError(f"{role} donor windows overlap")
    return output


def _validate_paired_variant_exports(
    loaded: Mapping[str, Mapping[str, np.ndarray]],
    role_data: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> None:
    if set(loaded) != {"M1", "N1"}:
        raise ValueError("paired analysis requires exactly M1 and N1 exports")
    for label in ("M1", "N1"):
        declared = str(np.asarray(loaded[label]["variant"]).item())
        if declared != label:
            raise ValueError(f"{label} input declares variant {declared!r}")
    for field in (
        "schema",
        "task_id",
        "seed",
        "development_is_new_independent_holdout",
        "derangement_nonoverlap_verified",
        "registered_hard_negative_policy",
    ):
        if field not in loaded["M1"] or field not in loaded["N1"]:
            raise KeyError(f"paired exports lack top-level {field}")
        if not np.array_equal(loaded["M1"][field], loaded["N1"][field]):
            raise ValueError(f"M1/N1 paired top-level mismatch for {field}")
    paired_fields = (
        "sample_id",
        "subject",
        "condition",
        "record_id",
        "eeg_event_time_ms",
        "fnirs_event_time_ms",
        "target",
        "donor_index",
        "eeg_token_valid_mask",
        "fnirs_token_valid_mask",
    )
    for role_name in ROLES:
        m1 = role_data["M1"][role_name]
        n1 = role_data["N1"][role_name]
        for field in paired_fields:
            if field not in m1 or field not in n1:
                raise KeyError(f"paired {role_name} export lacks {field}")
            if not np.array_equal(m1[field], n1[field]):
                raise ValueError(
                    f"M1/N1 paired export mismatch for {role_name}.{field}"
                )


def _coupling(
    role: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eeg = role["eeg_posterior"]
    fnirs = role["fnirs_posterior"]
    eeg_mask = role["eeg_token_valid_mask"].astype(bool)
    fnirs_mask = role["fnirs_token_valid_mask"].astype(bool)
    donor = role["donor_index"].astype(np.int64)
    matched = soft_cooccurrence_tensor(
        eeg,
        fnirs,
        lags=LAGS,
        eeg_valid_mask=eeg_mask,
        fnirs_valid_mask=fnirs_mask,
    )
    deranged = soft_cooccurrence_tensor(
        eeg,
        fnirs[donor],
        lags=LAGS,
        eeg_valid_mask=eeg_mask,
        fnirs_valid_mask=fnirs_mask[donor],
    )
    residual = matched_minus_deranged_expected_residual_log_lift(
        matched, deranged[None, ...], alpha=0.5
    )
    subject_residual = []
    for subject in sorted(set(role["subject"].astype(str).tolist())):
        selected = role["subject"].astype(str) == subject
        subject_matched = soft_cooccurrence_tensor(
            eeg[selected],
            fnirs[selected],
            lags=LAGS,
            eeg_valid_mask=eeg_mask[selected],
            fnirs_valid_mask=fnirs_mask[selected],
        )
        subject_deranged = soft_cooccurrence_tensor(
            eeg[selected],
            fnirs[donor][selected],
            lags=LAGS,
            eeg_valid_mask=eeg_mask[selected],
            fnirs_valid_mask=fnirs_mask[donor][selected],
        )
        subject_residual.append(
            matched_minus_deranged_expected_residual_log_lift(
                subject_matched, subject_deranged[None, ...], alpha=0.5
            )
        )
    return matched, deranged, residual, np.asarray(subject_residual)


def _condition_ids(
    role: Mapping[str, np.ndarray], condition_order: Sequence[str]
) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(condition_order)}
    conditions = role["condition"].astype(str)
    unknown = set(conditions.tolist()).difference(lookup)
    if unknown:
        raise ValueError(f"evaluation contains unknown conditions: {sorted(unknown)}")
    return np.asarray([lookup[value] for value in conditions], dtype=np.int64)


def _smooth_probe_train_targets(targets: np.ndarray) -> np.ndarray:
    """Apply fixed train-only label smoothing so absent classes have finite fits."""
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 16:
        raise ValueError("probe targets must have shape [rows,16]")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("probe targets must be finite and non-negative")
    row_sum = values.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0.0):
        raise ValueError("probe target rows must have positive mass")
    normalized = values / row_sum
    return (
        (1.0 - PROBE_TARGET_LABEL_SMOOTHING) * normalized
        + PROBE_TARGET_LABEL_SMOOTHING / values.shape[1]
    )


def _proper_score_rows(
    label: str,
    role_data: Mapping[str, Mapping[str, np.ndarray]],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    fit = role_data["fit_parameter"]
    condition_order = tuple(sorted(set(fit["condition"].astype(str).tolist())))
    output = []
    for lag in LAGS:
        fit_rows = build_lagged_categorical_rows(
            fit["eeg_posterior"],
            fit["fnirs_posterior"],
            lag=lag,
            subject_ids=fit["subject"].astype(str),
            condition_ids=_condition_ids(fit, condition_order),
            condition_count=len(condition_order),
            eeg_valid_mask=fit["eeg_token_valid_mask"],
            fnirs_valid_mask=fit["fnirs_token_valid_mask"],
            fnirs_history_steps=1,
        )
        models = fit_q0_q1(
            _smooth_probe_train_targets(fit_rows.fnirs_target),
            q0_design_train=fit_rows.q0_design,
            eeg_posterior_train=fit_rows.eeg_posterior,
            n_classes=16,
            l2=PROBE_L2,
            max_iter=PROBE_MAX_ITER,
            tol=PROBE_TOL,
        )
        if not models.q0.converged or not models.q1.converged:
            raise RuntimeError(
                f"q0/q1 categorical fit did not converge for {label}, lag={lag}: "
                f"q0={models.q0.converged}, q1={models.q1.converged}"
            )
        for role_name in ("fit_selection", "development_apply"):
            role = role_data[role_name]
            rows = build_lagged_categorical_rows(
                role["eeg_posterior"],
                role["fnirs_posterior"],
                lag=lag,
                subject_ids=role["subject"].astype(str),
                condition_ids=_condition_ids(role, condition_order),
                condition_count=len(condition_order),
                eeg_valid_mask=role["eeg_token_valid_mask"],
                fnirs_valid_mask=role["fnirs_token_valid_mask"],
                fnirs_history_steps=1,
            )
            scores = evaluate_q0_q1_by_subject(models, rows)
            gains = np.asarray(
                [row["log_loss_gain_nats"] for row in scores["subject_rows"]],
                dtype=float,
            )
            bootstrap = subject_block_bootstrap(
                gains,
                iterations=bootstrap_iterations,
                seed=bootstrap_seed + lag,
            )
            for subject_row in scores["subject_rows"]:
                output.append(
                    {
                        "schema": SCHEMA,
                        "variant": label,
                        "role": role_name,
                        "lag_tokens": lag,
                        "lag_seconds": 2 * lag,
                        "subject": subject_row["subject"],
                        "row_count": subject_row["row_count"],
                        "q0_fit_converged": models.q0.converged,
                        "q1_fit_converged": models.q1.converged,
                        "q0_fit_iterations": models.q0.iterations,
                        "q1_fit_iterations": models.q1.iterations,
                        "train_target_label_smoothing": PROBE_TARGET_LABEL_SMOOTHING,
                        "probe_l2": PROBE_L2,
                        "q0_log_loss_nats": subject_row["q0_log_loss_nats"],
                        "q1_log_loss_nats": subject_row["q1_log_loss_nats"],
                        "log_loss_gain_nats": subject_row["log_loss_gain_nats"],
                        "q0_brier_score": subject_row["q0_brier_score"],
                        "q1_brier_score": subject_row["q1_brier_score"],
                        "brier_gain": subject_row["brier_gain"],
                        "subject_equal_gain": scores[
                            "subject_equal_log_loss_gain_nats"
                        ],
                        "bootstrap_ci_lower": bootstrap["ci_lower"],
                        "bootstrap_ci_upper": bootstrap["ci_upper"],
                        "bootstrap_probability_positive": bootstrap[
                            "bootstrap_probability_positive"
                        ],
                        "bootstrap_subject_count": bootstrap["subject_count"],
                        "claim_status": CLAIM_STATUS,
                        "causal_future_claim": False,
                    }
                )
    return output


def _paired_m1_n1_rows(
    proper_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    """Pair M1 and N1 proper-score gains by held-out subject and fixed lag."""
    indexed: dict[tuple[str, str, int, str], Mapping[str, Any]] = {}
    for row in proper_rows:
        key = (
            str(row["variant"]),
            str(row["role"]),
            int(row["lag_tokens"]),
            str(row["subject"]),
        )
        if key in indexed:
            raise ValueError(f"duplicate proper-score row: {key}")
        indexed[key] = row
    output: list[dict[str, Any]] = []
    for role_name in ("fit_selection", "development_apply"):
        for lag in LAGS:
            m1_subjects = {
                key[3]
                for key in indexed
                if key[:3] == ("M1", role_name, lag)
            }
            n1_subjects = {
                key[3]
                for key in indexed
                if key[:3] == ("N1", role_name, lag)
            }
            if m1_subjects != n1_subjects:
                raise ValueError(
                    f"M1/N1 subjects differ for {role_name}, lag={lag}: "
                    f"{sorted(m1_subjects)} versus {sorted(n1_subjects)}"
                )
            deltas = np.asarray(
                [
                    float(indexed[("M1", role_name, lag, subject)]["log_loss_gain_nats"])
                    - float(indexed[("N1", role_name, lag, subject)]["log_loss_gain_nats"])
                    for subject in sorted(m1_subjects)
                ],
                dtype=float,
            )
            bootstrap = subject_block_bootstrap(
                deltas,
                iterations=bootstrap_iterations,
                seed=bootstrap_seed + 100 + lag,
            )
            for subject in sorted(m1_subjects):
                m1 = indexed[("M1", role_name, lag, subject)]
                n1 = indexed[("N1", role_name, lag, subject)]
                output.append(
                    {
                        "schema": SCHEMA,
                        "role": role_name,
                        "lag_tokens": lag,
                        "lag_seconds": 2 * lag,
                        "subject": subject,
                        "m1_log_loss_gain_nats": m1["log_loss_gain_nats"],
                        "n1_log_loss_gain_nats": n1["log_loss_gain_nats"],
                        "m1_minus_n1_log_loss_gain_nats": float(m1["log_loss_gain_nats"])
                        - float(n1["log_loss_gain_nats"]),
                        "m1_brier_gain": m1["brier_gain"],
                        "n1_brier_gain": n1["brier_gain"],
                        "m1_minus_n1_brier_gain": float(m1["brier_gain"])
                        - float(n1["brier_gain"]),
                        "subject_equal_m1_minus_n1_log_loss_gain": float(deltas.mean()),
                        "bootstrap_ci_lower": bootstrap["ci_lower"],
                        "bootstrap_ci_upper": bootstrap["ci_upper"],
                        "bootstrap_probability_positive": bootstrap[
                            "bootstrap_probability_positive"
                        ],
                        "bootstrap_subject_count": bootstrap["subject_count"],
                        "claim_status": CLAIM_STATUS,
                        "causal_future_claim": False,
                    }
                )
    return output


def _plot_residual(
    ordered: np.ndarray,
    *,
    label: str,
    task_id: str,
    subject_count: int,
    output_base: Path,
    color_limit: float,
) -> None:
    maximum = max(float(color_limit), 1e-8)
    norm = mpl.colors.TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)
    cmap = mpl.colormaps["RdBu_r"].with_extremes(bad="#777777")
    with mpl.rc_context(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    ):
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(7.2, 5.2),
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        image = None
        for axis, values, seconds in zip(axes.flat, ordered, LAG_SECONDS, strict=True):
            image = axis.imshow(
                values,
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                origin="lower",
                aspect="equal",
            )
            axis.set_title(f"lag = {seconds} s")
            axis.set_xticks((0, 5, 10, 15))
            axis.set_yticks((0, 5, 10, 15))
        for axis in axes[-1]:
            axis.set_xlabel("fNIRS code (fit-order index)")
        for axis in axes[:, 0]:
            axis.set_ylabel("EEG code (fit-order index)")
        fig.suptitle(
            f"{task_id} · {label} · development residual log-lift\n"
            f"SMOKE QC, n={subject_count} subject(s); no inferential claim",
            fontsize=10,
        )
        assert image is not None
        fig.colorbar(
            image,
            ax=axes,
            label="matched − deranged conditional log-lift (natural log)",
            shrink=0.88,
        )
        fig.savefig(output_base.with_suffix(".png"), dpi=240, metadata={"Software": "Matplotlib"})
        fig.savefig(output_base.with_suffix(".pdf"), metadata={"Creator": "LC-SPVQ analysis"})
        plt.close(fig)


def analyze(
    exports: Mapping[str, Path],
    output: Path,
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> Path:
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    loaded = {label: _load(path) for label, path in exports.items()}
    task_ids = {str(values["task_id"].item()) for values in loaded.values()}
    if len(task_ids) != 1:
        raise ValueError("all compared exports must belong to one task")
    task_id = next(iter(task_ids))
    paired_role_data = {
        label: {role: _role(values, role) for role in ROLES}
        for label, values in loaded.items()
    }
    _validate_paired_variant_exports(loaded, paired_role_data)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    coupling_rows = []
    proper_rows = []
    arrays: dict[str, np.ndarray] = {}
    plots: list[tuple[str, np.ndarray, int]] = []
    try:
        for label in loaded:
            role_data = paired_role_data[label]
            fit_matched, fit_deranged, fit_residual, fit_subject_residual = _coupling(
                role_data["fit_parameter"]
            )
            order = fit_train_only_token_order(fit_residual, method="svd")
            arrays[f"{label}__row_order"] = np.asarray(order.row_order, dtype=np.int64)
            arrays[f"{label}__column_order"] = np.asarray(order.column_order, dtype=np.int64)
            fit_ordered = apply_token_order(fit_residual, order)
            arrays[f"{label}__fit_parameter__residual_ordered"] = fit_ordered
            role_residuals = {"fit_parameter": fit_residual}
            role_subject_residuals = {"fit_parameter": fit_subject_residual}
            for role_name in ("fit_selection", "development_apply"):
                matched, deranged, residual, subject_residual = _coupling(
                    role_data[role_name]
                )
                role_residuals[role_name] = residual
                role_subject_residuals[role_name] = subject_residual
                arrays[f"{label}__{role_name}__matched"] = matched
                arrays[f"{label}__{role_name}__deranged"] = deranged
                arrays[f"{label}__{role_name}__residual"] = residual
                arrays[f"{label}__{role_name}__residual_ordered"] = apply_token_order(
                    residual, order
                )
            arrays[f"{label}__fit_parameter__matched"] = fit_matched
            arrays[f"{label}__fit_parameter__deranged"] = fit_deranged
            arrays[f"{label}__fit_parameter__residual"] = fit_residual
            for role_name in ROLES:
                role = role_data[role_name]
                residual = role_residuals[role_name]
                subject_residual = role_subject_residuals[role_name]
                stability = top_pair_stability(subject_residual, top_k=10)
                for lag_index, lag in enumerate(LAGS):
                    positive = np.maximum(residual[lag_index], 0.0)
                    positive_total = float(positive.sum())
                    top10 = float(
                        np.sort(positive.reshape(-1))[::-1][:10].sum()
                        / positive_total
                    ) if positive_total > 0.0 else 0.0
                    coupling_rows.append(
                        {
                            "schema": SCHEMA,
                            "variant": label,
                            "role": role_name,
                            "lag_tokens": lag,
                            "lag_seconds": 2 * lag,
                            "subject_count": len(set(role["subject"].astype(str))),
                            "pair_support": float(
                                arrays[f"{label}__{role_name}__matched"][lag_index].sum()
                            ),
                            "residual_peak": float(residual[lag_index].max()),
                            "residual_trough": float(residual[lag_index].min()),
                            "positive_residual_top10_concentration": top10,
                            "subject_top10_jaccard": float(
                                stability["per_lag_jaccard"][lag_index]
                            ),
                            "claim_status": CLAIM_STATUS,
                            "causal_future_claim": False,
                        }
                    )
            coupling_rows.append(
                {
                    "schema": SCHEMA,
                    "variant": label,
                    "role": "fit_to_development_stability",
                    "lag_tokens": "all",
                    "lag_seconds": "all",
                    "subject_count": len(
                        set(role_data["development_apply"]["subject"].astype(str))
                    ),
                    "fit_development_top10_jaccard": top_pair_jaccard(
                        fit_residual,
                        role_residuals["development_apply"],
                        top_k=10,
                    ),
                    "claim_status": CLAIM_STATUS,
                    "causal_future_claim": False,
                }
            )
            proper_rows.extend(
                _proper_score_rows(
                    label,
                    role_data,
                    bootstrap_iterations=bootstrap_iterations,
                    bootstrap_seed=bootstrap_seed,
                )
            )
            development_ordered = arrays[
                f"{label}__development_apply__residual_ordered"
            ]
            subject_count = len(
                set(role_data["development_apply"]["subject"].astype(str))
            )
            plots.append((label, development_ordered, subject_count))
        global_color_limit = max(
            float(np.max(np.abs(values))) for _, values, _ in plots
        )
        for label, development_ordered, subject_count in plots:
            _plot_residual(
                development_ordered,
                label=label,
                task_id=task_id,
                subject_count=subject_count,
                output_base=staging / f"{label}_development_residual_log_lift",
                color_limit=global_color_limit,
            )
        np.savez_compressed(
            staging / "coupling_arrays.npz",
            schema=np.asarray(SCHEMA),
            task_id=np.asarray(task_id),
            **arrays,
        )
        _write_csv(staging / "coupling_summary.csv", coupling_rows)
        _write_csv(staging / "q0_q1_subject_scores.csv", proper_rows)
        paired_rows = _paired_m1_n1_rows(
            proper_rows,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
        )
        _write_csv(staging / "m1_vs_n1_paired_proper_scores.csv", paired_rows)
        (staging / "FIGURE_DESCRIPTION.md").write_text(
            "# Figure description\n\n"
            "Each six-panel QC heatmap shows the measured-smoke development "
            "matched-minus-deranged conditional log-lift for lags 0–10 s. Rows "
            "are independent EEG codes and columns independent fNIRS codes. "
            "Orders are fitted separately for each variant from fit-parameter "
            "residual maps and then held fixed; cell positions must not be compared "
            "across variants. Red indicates positive residual log-lift and blue "
            "negative; all panels and both variants share one zero-centered scale. The development "
            "smoke contains one subject and is post-selection, so the figure is "
            "a software/QC artifact, not physiological evidence. Underlying "
            "values are in coupling_arrays.npz and coupling_summary.csv; paired "
            "M1-minus-N1 proper scores are in m1_vs_n1_paired_proper_scores.csv.\n",
            encoding="utf-8",
        )
        manifest = {
            "schema": SCHEMA,
            "status": "completed",
            "mode": "measured_smoke_qc",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task_id": task_id,
            "variants": list(exports),
            "lags_tokens": list(LAGS),
            "lags_seconds": list(LAG_SECONDS),
            "codebook_contract": "independent EEG/fNIRS K16; IDs are modality-specific and never aligned",
            "display_order_fit_scope": "fit_parameter only",
            "q0_fit_scope": "fit_parameter only",
            "q0_q1_fit_convergence": f"required; max_iter={PROBE_MAX_ITER}, tol={PROBE_TOL}",
            "q0_q1_probe_l2": PROBE_L2,
            "q0_q1_train_target_label_smoothing": PROBE_TARGET_LABEL_SMOOTHING,
            "q0_q1_evaluation_target_smoothing": 0.0,
            "analysis_script_sha256": _sha256(Path(__file__).resolve()),
            "q0_controls": ["one-step fNIRS history", "target token time", "task condition"],
            "q1_addition": "EEG posterior at source token",
            "primary_proper_score": "categorical log loss in nats; gain=q0-q1",
            "secondary_proper_score": "multiclass Brier; gain=q0-q1",
            "m1_vs_n1_primary_comparison": "paired subject M1 gain minus N1 gain at each fixed lag",
            "derangement": "same-subject same-condition nonidentity registered donor",
            "development_is_new_independent_holdout": False,
            "claim_status": CLAIM_STATUS,
            "causal_future_claim": False,
            "protected_open": any(
                bool(values["protected_open"].item()) for values in loaded.values()
            ),
            "paired_export_identity_verified": True,
            "derangement_nonoverlap_verified": all(
                bool(values["derangement_nonoverlap_verified"].item())
                for values in loaded.values()
            ),
            "biological_unit": "subject",
            "bootstrap_iterations": int(bootstrap_iterations),
            "bootstrap_seed": int(bootstrap_seed),
            "figure_scope": "provisional general QC; no journal compliance claim",
            "figure_color_scale": {
                "type": "shared symmetric diverging across variants and lags",
                "limit": float(global_color_limit),
                "center": 0.0,
            },
            "inputs": [
                *[
                    {
                        "label": label,
                        "path": str(path),
                        "sha256": _sha256(path),
                        "source_kind": "token_export",
                    }
                    for label, path in exports.items()
                ],
                {
                    "label": "analysis_script",
                    "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                    "sha256": _sha256(Path(__file__).resolve()),
                    "source_kind": "runtime_module",
                },
                {
                    "label": "coupling_runtime",
                    "path": "src/analysis/lagged_token_coupling.py",
                    "sha256": _sha256(
                        REPO_ROOT / "src/analysis/lagged_token_coupling.py"
                    ),
                    "source_kind": "runtime_module",
                },
            ],
            "limitations": [
                "Representations come from three-step pretrain/head and two-step VQ smoke schedules; they are not scientific endpoint estimates.",
                "Development subjects are reused post-selection and contain one smoke subject.",
                "Bootstrap intervals with one smoke subject are degenerate and exist only to exercise code paths.",
                "No protected cohort was opened.",
            ],
            "artifacts": [],
        }
        manifest["artifacts"] = _inventory(staging)
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, output)
        return output
    except Exception:
        print(f"failed staging retained at {staging}")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-export", type=Path, required=True)
    parser.add_argument("--n1-export", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=100)
    parser.add_argument("--bootstrap-seed", type=int, default=20260823)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        analyze(
            {"M1": args.m1_export.resolve(), "N1": args.n1_export.resolve()},
            args.output_dir.resolve(),
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=args.bootstrap_seed,
        )
    )
