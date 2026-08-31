#!/usr/bin/env python3
"""Aggregate existing EFRM pre-training metrics into a checkpoint trajectory.

This module is intentionally read-only with respect to training runs.  It
consumes the durable ``metrics/epochs.jsonl``/``analysis_metrics.json`` files
and the small checkpoint-level JSON summaries already exported by the EFRM
audit.  It does *not* load model checkpoints or similarity matrices, and it
never interpolates an alignment value between epochs.  Missing alignment
fields are represented as empty cells in the tidy table and are called out in
the report/manifest.

The command-line entry point writes a compact, review-ready bundle consisting
of a tidy CSV, JSON summary, PNG/PDF figures, a provenance manifest,
and a Markdown report.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCHEMA = "efrm_checkpoint_trajectory_analysis_v1"
OUTPUT_SCHEMA = "efrm_checkpoint_trajectory_tidy_v1"
ALIGNMENT_FIELDS = (
    "alignment_pair_count",
    "positive_cosine_mean",
    "negative_cosine_mean",
    "positive_minus_negative_cosine",
    "positive_vs_all_negative_auc",
    "positive_minus_hardest_negative_mean",
    "identity_pair_permutation_p_one_sided",
    "eeg_to_fnirs_mrr",
    "fnirs_to_eeg_mrr",
    "eeg_to_fnirs_top1",
    "fnirs_to_eeg_top1",
    "eeg_to_fnirs_top5",
    "fnirs_to_eeg_top5",
    "eeg_embedding_effective_rank",
    "fnirs_embedding_effective_rank",
    "eeg_embedding_first_axis_energy_fraction",
    "fnirs_embedding_first_axis_energy_fraction",
    "eeg_embedding_off_diagonal_cosine_mean",
    "fnirs_embedding_off_diagonal_cosine_mean",
)
GEOMETRY_FIELDS = (
    "eeg_embedding_effective_rank",
    "fnirs_embedding_effective_rank",
    "eeg_embedding_first_axis_energy_fraction",
    "fnirs_embedding_first_axis_energy_fraction",
    "eeg_embedding_off_diagonal_cosine_mean",
    "fnirs_embedding_off_diagonal_cosine_mean",
)
TRAINING_FIELDS = (
    "train_loss",
    "validation_loss",
    "train_eeg_reconstruction_loss",
    "validation_eeg_reconstruction_loss",
    "train_fnirs_reconstruction_loss",
    "validation_fnirs_reconstruction_loss",
    "train_clip_alignment_loss",
    "validation_clip_alignment_loss",
    "train_pair_count",
    "validation_pair_count",
    "learning_rate",
    "seconds",
)
TIDY_FIELDS = (
    "run_id",
    "run_scope",
    "lodo_stage",
    "excluded_target_dataset",
    "run_state",
    "record_kind",
    "epoch",
    "epoch_1based",
    "checkpoint_label",
    "checkpoint_epoch",
    "checkpoint_epoch_known",
    "checkpoint_epoch_source",
    "source_relpath",
    "metric_status",
    "reconstruction_observed",
    "clip_observed",
    "alignment_observed",
    "geometry_observed",
    "alignment_metric_basis",
    "alignment_evidence_scope_kind",
    "alignment_evidence_scope_representative_full_validation",
    *TRAINING_FIELDS,
    *ALIGNMENT_FIELDS,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def _finite_or_none(value: Any) -> float | int | str | None:
    """Return scalars suitable for CSV/JSON, preserving only observed values."""

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    return None


def _nested(mapping: Mapping[str, Any] | None, *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _float(value: Any) -> float | None:
    value = _finite_or_none(value)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    value = _finite_or_none(value)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _run_scope(run_id: str, manifest: Mapping[str, Any]) -> str:
    """Classify runs for plotting/reporting without changing inclusion rules."""

    lower = run_id.lower()
    stage = str(manifest.get("lodo_stage", ""))
    protocol = str(manifest.get("protocol_id", ""))
    # Smoke/architecture probes are kept separate even when their names also
    # contain ``paper_mi`` or a protocol token.
    if "smoke" in lower or "architecture" in lower:
        return "smoke"
    if "stage_a" in lower or stage == "selection":
        return "formal_stage_a"
    if "stage_b" in lower or stage == "final_refit":
        return "formal_stage_b"
    if "source_seed42" in lower or "source_boundary" in lower:
        return "source_reference"
    if "paper_mi" in lower:
        return "paper_reference"
    if "sync_dev" in lower:
        return "development"
    if protocol.startswith("efrm_"):
        return "efrm_other"
    return "unknown"


def _is_default_included(run_id: str, run_dir: Path) -> bool:
    """Keep formal Stage-A + source/reference runs; exclude Stage-B/smoke by default.

    The function is deliberately name/artifact based.  A caller can pass
    ``include_all=True`` to include every run with an epoch log.
    """

    scope = _run_scope(run_id, _read_json(run_dir / "manifest.json") if (run_dir / "manifest.json").is_file() else {})
    if scope in {"formal_stage_a", "source_reference", "paper_reference", "development"}:
        return True
    return False


def _run_metadata(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    manifest = _read_json(run_dir / "manifest.json") if (run_dir / "manifest.json").is_file() else {}
    status = _read_json(run_dir / "status.json") if (run_dir / "status.json").is_file() else {}
    analysis_path = run_dir / "analysis/analysis_metrics.json"
    analysis = _read_json(analysis_path) if analysis_path.is_file() else None
    return manifest, status, analysis


def _alignment_mapping(alignment: Mapping[str, Any]) -> dict[str, float | int | None]:
    """Flatten exactly the scalar alignment fields present in a JSON object."""

    result: dict[str, float | int | None] = {field: None for field in ALIGNMENT_FIELDS}
    values: dict[str, Any] = {
        "alignment_pair_count": alignment.get("pair_count"),
        "positive_cosine_mean": alignment.get("positive_cosine_mean"),
        "negative_cosine_mean": alignment.get("negative_cosine_mean"),
        "positive_minus_negative_cosine": alignment.get("positive_minus_negative_cosine"),
        "positive_vs_all_negative_auc": alignment.get("positive_vs_all_negative_auc"),
        "positive_minus_hardest_negative_mean": alignment.get("positive_minus_hardest_negative_mean"),
        "identity_pair_permutation_p_one_sided": alignment.get("identity_pair_permutation_p_one_sided"),
        "eeg_to_fnirs_mrr": _nested(alignment, "eeg_to_fnirs", "mrr"),
        "fnirs_to_eeg_mrr": _nested(alignment, "fnirs_to_eeg", "mrr"),
        "eeg_to_fnirs_top1": _nested(alignment, "eeg_to_fnirs", "top1"),
        "fnirs_to_eeg_top1": _nested(alignment, "fnirs_to_eeg", "top1"),
        "eeg_to_fnirs_top5": _nested(alignment, "eeg_to_fnirs", "top5"),
        "fnirs_to_eeg_top5": _nested(alignment, "fnirs_to_eeg", "top5"),
        "eeg_embedding_effective_rank": _nested(alignment, "eeg_embedding_geometry", "effective_rank"),
        "fnirs_embedding_effective_rank": _nested(alignment, "fnirs_embedding_geometry", "effective_rank"),
        "eeg_embedding_first_axis_energy_fraction": _nested(
            alignment, "eeg_embedding_geometry", "first_axis_energy_fraction"
        ),
        "fnirs_embedding_first_axis_energy_fraction": _nested(
            alignment, "fnirs_embedding_geometry", "first_axis_energy_fraction"
        ),
        "eeg_embedding_off_diagonal_cosine_mean": _nested(
            alignment, "eeg_embedding_geometry", "off_diagonal_cosine_mean"
        ),
        "fnirs_embedding_off_diagonal_cosine_mean": _nested(
            alignment, "fnirs_embedding_geometry", "off_diagonal_cosine_mean"
        ),
    }
    for key, value in values.items():
        if key == "alignment_pair_count":
            result[key] = _int(value)
        else:
            result[key] = _float(value)
    return result


def _training_mapping(row: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten nested train/validation metrics; absent keys stay missing."""

    train = row.get("train") if isinstance(row.get("train"), Mapping) else {}
    validation = row.get("validation") if isinstance(row.get("validation"), Mapping) else {}
    values: dict[str, Any] = {
        "train_loss": train.get("loss"),
        "validation_loss": validation.get("loss"),
        "train_eeg_reconstruction_loss": train.get("eeg_reconstruction_loss"),
        "validation_eeg_reconstruction_loss": validation.get("eeg_reconstruction_loss"),
        "train_fnirs_reconstruction_loss": train.get("fnirs_reconstruction_loss"),
        "validation_fnirs_reconstruction_loss": validation.get("fnirs_reconstruction_loss"),
        "train_clip_alignment_loss": train.get("clip_alignment_loss"),
        "validation_clip_alignment_loss": validation.get("clip_alignment_loss"),
        "train_pair_count": train.get("pair_count"),
        "validation_pair_count": validation.get("pair_count"),
        "learning_rate": row.get("learning_rate"),
        "seconds": row.get("seconds"),
    }
    result: dict[str, Any] = {}
    for key, value in values.items():
        result[key] = _int(value) if key.endswith("pair_count") else _float(value)
    return result


def _empty_row() -> dict[str, Any]:
    return {field: None for field in TIDY_FIELDS}


def _epoch_row(
    run_id: str,
    scope: str,
    manifest: Mapping[str, Any],
    run_state: str,
    source_relpath: str,
    epoch_record: Mapping[str, Any],
) -> dict[str, Any]:
    row = _empty_row()
    row.update(
        {
            "run_id": run_id,
            "run_scope": scope,
            "lodo_stage": manifest.get("lodo_stage"),
            "excluded_target_dataset": manifest.get("excluded_target_dataset"),
            "run_state": run_state,
            "record_kind": "epoch_metric",
            "epoch": _int(epoch_record.get("epoch")),
            "epoch_1based": (_int(epoch_record.get("epoch")) + 1) if _int(epoch_record.get("epoch")) is not None else None,
            "checkpoint_label": None,
            "checkpoint_epoch_known": False,
            "source_relpath": source_relpath,
            "metric_status": "observed_epoch_metrics",
            "reconstruction_observed": any(
                _training_mapping(epoch_record).get(name) is not None
                for name in (
                    "validation_eeg_reconstruction_loss",
                    "validation_fnirs_reconstruction_loss",
                )
            ),
            "clip_observed": _training_mapping(epoch_record).get("validation_clip_alignment_loss") is not None,
            "alignment_observed": False,
            "geometry_observed": False,
            "alignment_metric_basis": None,
            "alignment_evidence_scope_kind": None,
            "alignment_evidence_scope_representative_full_validation": None,
        }
    )
    row.update(_training_mapping(epoch_record))
    # Some future exporters may include alignment scalars inline.  We only use
    # them if the nested object is actually present in this row.
    inline_alignment = epoch_record.get("alignment")
    if isinstance(inline_alignment, Mapping):
        row.update(_alignment_mapping(inline_alignment))
        row["alignment_observed"] = any(row.get(name) is not None for name in ALIGNMENT_FIELDS)
        row["geometry_observed"] = any(row.get(name) is not None for name in GEOMETRY_FIELDS)
        row["alignment_metric_basis"] = "inline_export_unspecified"
        row["metric_status"] = "observed_epoch_and_alignment_metrics"
    return row


def _checkpoint_epoch_info(
    label: str,
    analysis: Mapping[str, Any] | None,
) -> tuple[int | None, bool, str | None]:
    """Use only explicit/inferred metadata, never file order or interpolation."""

    audit = analysis.get("audit", {}) if isinstance(analysis, Mapping) else {}
    checkpoint = audit.get("checkpoint", {}) if isinstance(audit, Mapping) else {}
    inferred = _int(checkpoint.get("checkpoint_epoch_inferred_from_log"))
    if inferred is not None:
        return inferred, True, "analysis.audit.checkpoint_epoch_inferred_from_log"
    # The analysis audit names the objective-selected epoch, but it does not
    # identify the separate best_alignment JSON checkpoint.  It is therefore
    # safe to associate this only with ``best`` and to leave all other labels
    # unknown.
    if label == "best":
        best_epoch = _int(audit.get("best_epoch"))
        if best_epoch is not None:
            return best_epoch, True, "analysis.audit.best_epoch"
    return None, False, None


def _checkpoint_label(path: Path) -> str:
    name = path.name
    if name.startswith("best_alignment"):
        return "best_alignment"
    if name.startswith("latest"):
        return "latest"
    if name.startswith("best"):
        return "best"
    return "checkpoint_summary"


def _checkpoint_row(
    run_id: str,
    scope: str,
    manifest: Mapping[str, Any],
    run_state: str,
    source_relpath: str,
    checkpoint_label: str,
    checkpoint_metrics: Mapping[str, Any],
    analysis: Mapping[str, Any] | None,
) -> dict[str, Any]:
    row = _empty_row()
    epoch, known, source = _checkpoint_epoch_info(checkpoint_label, analysis)
    alignment = checkpoint_metrics.get("alignment") if isinstance(checkpoint_metrics.get("alignment"), Mapping) else checkpoint_metrics
    row.update(
        {
            "run_id": run_id,
            "run_scope": scope,
            "lodo_stage": manifest.get("lodo_stage"),
            "excluded_target_dataset": manifest.get("excluded_target_dataset"),
            "run_state": run_state,
            "record_kind": "checkpoint_metric",
            "checkpoint_label": checkpoint_label,
            "checkpoint_epoch": epoch,
            "checkpoint_epoch_known": known,
            "checkpoint_epoch_source": source,
            "source_relpath": source_relpath,
            "metric_status": "observed_checkpoint_alignment_metrics",
            "reconstruction_observed": False,
            "clip_observed": False,
            "alignment_observed": False,
            "geometry_observed": False,
            "alignment_metric_basis": "row_weighted_duplicate_unaware_existing_export",
            "alignment_evidence_scope_kind": _nested(alignment, "evidence_scope", "kind") if isinstance(alignment, Mapping) else None,
            "alignment_evidence_scope_representative_full_validation": _nested(
                alignment, "evidence_scope", "representative_of_full_validation"
            ) if isinstance(alignment, Mapping) else None,
        }
    )
    if isinstance(alignment, Mapping):
        row.update(_alignment_mapping(alignment))
    row["alignment_observed"] = any(row.get(name) is not None for name in ALIGNMENT_FIELDS)
    row["geometry_observed"] = any(row.get(name) is not None for name in GEOMETRY_FIELDS)
    if not row["alignment_observed"]:
        row["metric_status"] = "checkpoint_alignment_fields_missing"
    return row


def _analysis_export_row(
    run_id: str,
    scope: str,
    manifest: Mapping[str, Any],
    run_state: str,
    analysis_path: Path,
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    """Represent alignment embedded in analysis_metrics.json without inventing a checkpoint.

    Existing analyses commonly evaluate ``latest`` or a selected exported
    validation evidence file, but the JSON does not always expose which model
    checkpoint produced it.  The row is therefore labelled ``analysis_export``
    and its epoch remains missing unless the audit explicitly records one.
    """

    row = _checkpoint_row(
        run_id,
        scope,
        manifest,
        run_state,
        str(analysis_path.relative_to(analysis_path.parents[2])),
        "analysis_export",
        analysis,
        analysis,
    )
    row["metric_status"] = (
        "observed_analysis_export_alignment_metrics"
        if row["alignment_observed"]
        else "analysis_export_alignment_fields_missing"
    )
    return row


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def load_run_rows(run_dir: str | Path, source_root: str | Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load one run into tidy rows and an inventory record.

    This is exposed for tests and for small downstream audit scripts.  It only
    reads JSON/CSV summaries; model ``.pt`` files and evidence ``.npz`` files
    are never opened.
    """

    run = Path(run_dir).resolve()
    root = Path(source_root).resolve() if source_root else run.parent
    manifest, status, analysis = _run_metadata(run)
    run_id = str(manifest.get("run_id", run.name))
    scope = _run_scope(run_id, manifest)
    run_state = str(manifest.get("status", status.get("status", "unknown")))
    epoch_path = run / "metrics/epochs.jsonl"
    if not epoch_path.is_file():
        raise FileNotFoundError(epoch_path)
    epochs = _read_jsonl(epoch_path)
    rows = [
        _epoch_row(
            run_id,
            scope,
            manifest,
            run_state,
            _safe_relative(epoch_path, root),
            value,
        )
        for value in epochs
    ]

    alignment_sources: list[str] = []
    analysis_path = run / "analysis/analysis_metrics.json"
    if analysis is not None and isinstance(analysis.get("alignment"), Mapping):
        rows.append(
            _analysis_export_row(
                run_id,
                scope,
                manifest,
                run_state,
                analysis_path,
                analysis,
            )
        )
        alignment_sources.append(_safe_relative(analysis_path, root))

    checkpoint_dir = run / "analysis/checkpoints"
    checkpoint_paths = sorted(checkpoint_dir.glob("*_full_validation_metrics.json")) if checkpoint_dir.is_dir() else []
    # Also accept a generic checkpoint metrics summary emitted by future runs.
    if checkpoint_dir.is_dir():
        checkpoint_paths.extend(
            path
            for path in sorted(checkpoint_dir.glob("*.json"))
            if path not in checkpoint_paths and path.name.endswith("_metrics.json")
        )
    seen: set[Path] = set()
    for checkpoint_path in checkpoint_paths:
        if checkpoint_path in seen:
            continue
        seen.add(checkpoint_path)
        metrics = _read_json(checkpoint_path)
        label = _checkpoint_label(checkpoint_path)
        rows.append(
            _checkpoint_row(
                run_id,
                scope,
                manifest,
                run_state,
                _safe_relative(checkpoint_path, root),
                label,
                metrics,
                analysis,
            )
        )
        if any(value is not None for value in _alignment_mapping(metrics.get("alignment", metrics) if isinstance(metrics, Mapping) else {} ).values()):
            alignment_sources.append(_safe_relative(checkpoint_path, root))

    def availability(fields: Sequence[str]) -> bool:
        return any(row.get(field) is not None for row in rows for field in fields)

    inventory = {
        "run_id": run_id,
        "run_dir": _safe_relative(run, root),
        "run_scope": scope,
        "lodo_stage": manifest.get("lodo_stage"),
        "excluded_target_dataset": manifest.get("excluded_target_dataset"),
        "manifest_status": manifest.get("status"),
        "status_file_status": status.get("status"),
        "run_state": run_state,
        "epoch_count": len(epochs),
        "epoch_ids": [_int(item.get("epoch")) for item in epochs],
        "analysis_metrics_present": analysis is not None,
        "alignment_source_count": len(alignment_sources),
        "alignment_sources": alignment_sources,
        "field_availability": {
            "reconstruction": availability(("validation_eeg_reconstruction_loss", "validation_fnirs_reconstruction_loss")),
            "clip": availability(("validation_clip_alignment_loss",)),
            "alignment": availability(ALIGNMENT_FIELDS),
            "geometry": availability(GEOMETRY_FIELDS),
        },
        "source_hashes": {
            _safe_relative(epoch_path, root): _sha256(epoch_path),
            **({
                _safe_relative(analysis_path, root): _sha256(analysis_path)
            } if analysis_path.is_file() else {}),
            **{
                _safe_relative(path, root): _sha256(path)
                for path in checkpoint_paths
                if path.is_file()
            },
        },
    }
    return rows, inventory


def discover_runs(
    pretraining_root: str | Path,
    *,
    include_all: bool = False,
    run_filters: Sequence[str] | None = None,
) -> list[Path]:
    root = Path(pretraining_root).resolve()
    patterns = tuple(run_filters or ())
    result: list[Path] = []
    for run in sorted(path for path in root.iterdir() if path.is_dir()):
        if not (run / "metrics/epochs.jsonl").is_file():
            continue
        manifest = _read_json(run / "manifest.json") if (run / "manifest.json").is_file() else {}
        run_id = str(manifest.get("run_id", run.name))
        if patterns and not any(re.search(pattern, run_id) for pattern in patterns):
            continue
        if not include_all and not _is_default_included(run_id, run):
            continue
        result.append(run)
    return result


def _write_tidy_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIDY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in TIDY_FIELDS})


def _short_label(run_id: str, max_length: int = 32) -> str:
    if len(run_id) <= max_length:
        return run_id
    return run_id[: max_length - 3] + "..."


def _alignment_short_label(run_id: str) -> str:
    """Short, stable labels for checkpoint figures."""

    if run_id in FORMAL_STAGE_A_LABELS:
        return FORMAL_STAGE_A_LABELS[run_id]
    lower = run_id.lower()
    if "sync_dev" in lower:
        return "sync-dev"
    if "paper_mi_trial_mixed_clip_only" in lower:
        return "paper-trial-clip"
    if "paper_mi_trial_mixed" in lower:
        return "paper-trial"
    if "paper_mi_clip_only" in lower:
        return "paper-clip"
    if "paper_mi_diag" in lower:
        return "paper-diag"
    if "source_seed42" in lower:
        return "source"
    return _short_label(run_id, 18)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save_figure(figure: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_trajectory(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    run_ids = sorted({str(row["run_id"]) for row in rows if row.get("record_kind") == "epoch_metric"})
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(run_ids), 1)))
    color_by_run = dict(zip(run_ids, colors))
    figure, axes = plt.subplots(3, 1, figsize=(11.6, 11.0), constrained_layout=True)
    for run_id in run_ids:
        values = [row for row in rows if row.get("run_id") == run_id and row.get("record_kind") == "epoch_metric"]
        values = sorted(values, key=lambda row: row.get("epoch") if row.get("epoch") is not None else -1)
        x = np.asarray([row["epoch_1based"] for row in values], dtype=float)
        color = color_by_run[run_id]
        eeg = np.asarray([np.nan if row.get("validation_eeg_reconstruction_loss") is None else row["validation_eeg_reconstruction_loss"] for row in values])
        fnirs = np.asarray([np.nan if row.get("validation_fnirs_reconstruction_loss") is None else row["validation_fnirs_reconstruction_loss"] for row in values])
        clip = np.asarray([np.nan if row.get("validation_clip_alignment_loss") is None else row["validation_clip_alignment_loss"] for row in values])
        axes[0].plot(x, eeg, color=color, lw=1.2, label=_short_label(run_id))
        axes[1].plot(x, fnirs, color=color, lw=1.2, label=_short_label(run_id))
        axes[2].plot(x, clip, color=color, lw=1.2, label=_short_label(run_id))
    for axis, ylabel, title in (
        (axes[0], "validation EEG reconstruction loss", "Observed epoch trajectory: EEG reconstruction"),
        (axes[1], "validation fNIRS reconstruction loss", "Observed epoch trajectory: fNIRS reconstruction"),
        (axes[2], "validation CLIP alignment loss", "Observed epoch trajectory: CLIP loss"),
    ):
        axis.set(xlabel="completed epoch (1-based)", ylabel=ylabel, title=title)
        axis.grid(alpha=0.2)
        axis.legend(loc="best", ncol=2, frameon=False)
    figure.suptitle("EFRM P0 checkpoint trajectory (raw logged values only)", fontsize=12)
    _save_figure(figure, output_dir / "trajectory_all_runs")


FORMAL_STAGE_A_LABELS = {
    "efrm_lodo_full_target_fivefold_v2__exclude_eeg_fnirs_single_trial__stage_a_seed42": "exclude-ST",
    "efrm_lodo_full_target_fivefold_v2__exclude_refed__stage_a_seed42": "exclude-REFED",
    "efrm_lodo_full_target_fivefold_v2__exclude_simultaneous_eeg_nirs__stage_a_seed42": "exclude-Sim",
    "efrm_lodo_full_target_fivefold_v2__exclude_visual_cognitive_motivation__stage_a_seed42": "exclude-Visual",
}


def _plot_formal_stage_a_trajectory(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    """Legible primary plot for the four formal LODO Stage-A runs.

    The all-run figure is retained as a supplement, but the primary figure
    avoids an unreadable ten-entry legend and uses both colour and line style
    redundantly.  Missing alignment metrics are not plotted here because no
    epoch-wise alignment observations exist in the source logs.
    """

    run_ids = [run_id for run_id in FORMAL_STAGE_A_LABELS if any(
        row.get("run_id") == run_id and row.get("record_kind") == "epoch_metric"
        for row in rows
    )]
    figure, axes = plt.subplots(3, 1, figsize=(10.8, 9.4), constrained_layout=True)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    styles = ("-", "--", "-.", ":")
    for run_id, color, style in zip(run_ids, colors, styles):
        values = sorted(
            [row for row in rows if row.get("run_id") == run_id and row.get("record_kind") == "epoch_metric"],
            key=lambda row: row.get("epoch") if row.get("epoch") is not None else -1,
        )
        x = np.asarray([row["epoch_1based"] for row in values], dtype=float)
        label = FORMAL_STAGE_A_LABELS[run_id]
        for axis, key in (
            (axes[0], "validation_eeg_reconstruction_loss"),
            (axes[1], "validation_fnirs_reconstruction_loss"),
            (axes[2], "validation_clip_alignment_loss"),
        ):
            y = np.asarray([
                np.nan if row.get(key) is None else row[key]
                for row in values
            ], dtype=float)
            axis.plot(x, y, color=color, linestyle=style, lw=1.6, label=label)
    for axis, ylabel, title in (
        (axes[0], "validation EEG reconstruction loss", "Formal LODO Stage-A: EEG reconstruction"),
        (axes[1], "validation fNIRS reconstruction loss", "Formal LODO Stage-A: fNIRS reconstruction"),
        (axes[2], "validation CLIP alignment loss", "Formal LODO Stage-A: CLIP loss"),
    ):
        axis.set(xlabel="completed epoch (1-based)", ylabel=ylabel, title=title)
        axis.grid(alpha=0.2)
        axis.legend(ncol=2, loc="best", frameon=False)
    figure.suptitle(
        "EFRM P0 primary trajectory — four formal Stage-A runs (logged values only)",
        fontsize=12,
    )
    _save_figure(figure, output_dir / "trajectory")


def _plot_alignment(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    observed = [
        row
        for row in rows
        if row.get("record_kind") == "checkpoint_metric" and row.get("alignment_observed")
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.6), constrained_layout=True)
    if observed:
        labels = [f"{_alignment_short_label(str(row['run_id']))}\n{row.get('checkpoint_label')}" for row in observed]
        x = np.arange(len(observed))
        auc = [row.get("positive_vs_all_negative_auc") for row in observed]
        axes[0, 0].bar(x, [np.nan if value is None else value for value in auc], color="#0072B2")
        axes[0, 0].axhline(0.5, ls="--", lw=1, color="#666666", label="AUC chance")
        axes[0, 0].set_xticks(x, labels, rotation=70, ha="right")
        axes[0, 0].set_ylabel("positive-vs-negative AUC")
        axes[0, 0].set_title("Checkpoint alignment separation")
        axes[0, 0].legend(frameon=False)
        eeg_rank = [row.get("eeg_embedding_effective_rank") for row in observed]
        fnirs_rank = [row.get("fnirs_embedding_effective_rank") for row in observed]
        axes[0, 1].plot(x, [np.nan if value is None else value for value in eeg_rank], "o-", label="EEG", color="#0072B2")
        axes[0, 1].plot(x, [np.nan if value is None else value for value in fnirs_rank], "s--", label="fNIRS", color="#009E73")
        axes[0, 1].set_xticks(x, labels, rotation=70, ha="right")
        axes[0, 1].set_ylabel("centered effective rank")
        axes[0, 1].set_title("Embedding geometry at exported checkpoints")
        axes[0, 1].legend(frameon=False)
        eeg_axis = [row.get("eeg_embedding_first_axis_energy_fraction") for row in observed]
        fnirs_axis = [row.get("fnirs_embedding_first_axis_energy_fraction") for row in observed]
        axes[1, 0].plot(x, [np.nan if value is None else value for value in eeg_axis], "o-", label="EEG", color="#0072B2")
        axes[1, 0].plot(x, [np.nan if value is None else value for value in fnirs_axis], "s--", label="fNIRS", color="#009E73")
        axes[1, 0].set_xticks(x, labels, rotation=70, ha="right")
        axes[1, 0].set_ylim(0, 1.02)
        axes[1, 0].set_ylabel("first-axis energy fraction")
        axes[1, 0].set_title("Axis concentration (observed only)")
        axes[1, 0].legend(frameon=False)
        eeg_mrr = [row.get("eeg_to_fnirs_mrr") for row in observed]
        fnirs_mrr = [row.get("fnirs_to_eeg_mrr") for row in observed]
        axes[1, 1].plot(x, [np.nan if value is None else value for value in eeg_mrr], "o-", label="EEG→fNIRS", color="#0072B2")
        axes[1, 1].plot(x, [np.nan if value is None else value for value in fnirs_mrr], "s--", label="fNIRS→EEG", color="#E69F00")
        axes[1, 1].set_xticks(x, labels, rotation=70, ha="right")
        axes[1, 1].set_ylabel("MRR")
        axes[1, 1].set_title("Bidirectional retrieval (exported checkpoints)")
        axes[1, 1].legend(frameon=False)
    else:
        for axis in axes.flat:
            axis.text(0.5, 0.5, "No checkpoint-level alignment JSON exported", ha="center", va="center")
            axis.set_axis_off()
    figure.suptitle("EFRM P0 alignment/geometry diagnostics; missing epochs are not imputed", fontsize=12)
    _save_figure(figure, output_dir / "alignment_geometry")


def _summary_by_run(rows: Sequence[Mapping[str, Any]], inventories: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for inventory in inventories:
        run_id = str(inventory["run_id"])
        epoch_rows = [row for row in rows if row.get("run_id") == run_id and row.get("record_kind") == "epoch_metric"]
        checkpoint_rows = [row for row in rows if row.get("run_id") == run_id and row.get("record_kind") != "epoch_metric"]
        result.append(
            {
                "run_id": run_id,
                "run_scope": inventory.get("run_scope"),
                "epoch_count": len(epoch_rows),
                "first_epoch": epoch_rows[0].get("epoch") if epoch_rows else None,
                "last_epoch": epoch_rows[-1].get("epoch") if epoch_rows else None,
                "validation_eeg_reconstruction_first": epoch_rows[0].get("validation_eeg_reconstruction_loss") if epoch_rows else None,
                "validation_eeg_reconstruction_last": epoch_rows[-1].get("validation_eeg_reconstruction_loss") if epoch_rows else None,
                "validation_fnirs_reconstruction_first": epoch_rows[0].get("validation_fnirs_reconstruction_loss") if epoch_rows else None,
                "validation_fnirs_reconstruction_last": epoch_rows[-1].get("validation_fnirs_reconstruction_loss") if epoch_rows else None,
                "validation_clip_first": epoch_rows[0].get("validation_clip_alignment_loss") if epoch_rows else None,
                "validation_clip_last": epoch_rows[-1].get("validation_clip_alignment_loss") if epoch_rows else None,
                "alignment_export_rows": len([row for row in checkpoint_rows if row.get("alignment_observed")]),
                "alignment_checkpoint_labels": [row.get("checkpoint_label") for row in checkpoint_rows if row.get("alignment_observed")],
                "alignment_epoch_known_rows": len([row for row in checkpoint_rows if row.get("alignment_observed") and row.get("checkpoint_epoch_known")]),
                "alignment_is_missing_for_epochs": not any(row.get("alignment_observed") for row in epoch_rows),
            }
        )
    return result


def _markdown_report(
    source_root: Path,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    inventories: Sequence[Mapping[str, Any]],
    summary: Sequence[Mapping[str, Any]],
) -> str:
    formal = [item for item in summary if item.get("run_scope") == "formal_stage_a"]
    alignment_rows = [row for row in rows if row.get("alignment_observed")]
    lines = [
        "# EFRM P0 checkpoint trajectory analysis",
        "",
        f"- Schema: `{SCHEMA}`",
        f"- Generated: `{_now()}`",
        "- Scope: read-only public train/validation artifacts; protected test was not opened.",
        f"- Source root: `{source_root}`",
        f"- Included runs: {len(inventories)}; tidy rows: {len(rows)}.",
        "",
        "## What was measured",
        "",
        "Epoch rows are copied from each run's durable `metrics/epochs.jsonl` (or equivalent exported table when a future run provides one). The observed fields are validation/train reconstruction, CLIP loss, pair count, learning rate, and epoch duration. Alignment/AUC/MRR/margin/effective-rank/first-axis fields are added only from an existing alignment summary JSON; no similarity matrix was loaded and no per-epoch alignment value was interpolated.",
        "",
        "## Run inventory",
        "",
        "| Run | Scope | Epochs | Alignment export rows | Known alignment epochs | Alignment fields available | Evidence scope |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for item in summary:
        inventory = next(value for value in inventories if value["run_id"] == item["run_id"])
        evidence_kinds = sorted({
            str(row.get("alignment_evidence_scope_kind"))
            for row in rows
            if row.get("run_id") == item["run_id"]
            and row.get("alignment_evidence_scope_kind") is not None
        })
        evidence_text = ", ".join(evidence_kinds) if evidence_kinds else "missing"
        lines.append(
            f"| `{item['run_id']}` | `{item.get('run_scope')}` | {item.get('epoch_count')} | "
            f"{item.get('alignment_export_rows')} | {item.get('alignment_epoch_known_rows')} | "
            f"{bool(inventory.get('field_availability', {}).get('alignment'))} | {evidence_text} |"
        )
    lines += [
        "",
        "## Observed formal Stage-A trajectory",
        "",
        "The formal Stage-A runs provide direct evidence that the reconstruction branches improve while the scalar CLIP term moves little. This is an observed optimization pattern, not evidence that the cross-modal alignment mechanism succeeded.",
        "",
        "| Run | Validation EEG reconstruction change | Validation fNIRS reconstruction change | Validation CLIP change | Exported AUC | Exported EEG→fNIRS MRR | Exported fNIRS→EEG MRR | EEG rank / first axis | fNIRS rank / first axis |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in formal:
        def change(first_key: str, last_key: str) -> str:
            first, last = item.get(first_key), item.get(last_key)
            if first is None or last is None or first == 0:
                return "missing"
            return f"{(float(last) - float(first)) / float(first):+.3%}"

        candidates = [row for row in alignment_rows if row.get("run_id") == item["run_id"]]
        # Prefer analysis_export, then latest, then best_alignment; this is a
        # presentation choice, not a replacement of any row in the tidy table.
        preferred = next((row for label in ("analysis_export", "latest", "best_alignment", "best") for row in candidates if row.get("checkpoint_label") == label), None)
        if preferred is None:
            auc = mrr_e = mrr_f = rank_e = axis_e = rank_f = axis_f = "missing"
        else:
            auc = "missing" if preferred.get("positive_vs_all_negative_auc") is None else f"{preferred['positive_vs_all_negative_auc']:.4f}"
            mrr_e = "missing" if preferred.get("eeg_to_fnirs_mrr") is None else f"{preferred['eeg_to_fnirs_mrr']:.4f}"
            mrr_f = "missing" if preferred.get("fnirs_to_eeg_mrr") is None else f"{preferred['fnirs_to_eeg_mrr']:.4f}"
            rank_e = "missing" if preferred.get("eeg_embedding_effective_rank") is None else f"{preferred['eeg_embedding_effective_rank']:.3f}"
            axis_e = "missing" if preferred.get("eeg_embedding_first_axis_energy_fraction") is None else f"{preferred['eeg_embedding_first_axis_energy_fraction']:.3f}"
            rank_f = "missing" if preferred.get("fnirs_embedding_effective_rank") is None else f"{preferred['fnirs_embedding_effective_rank']:.3f}"
            axis_f = "missing" if preferred.get("fnirs_embedding_first_axis_energy_fraction") is None else f"{preferred['fnirs_embedding_first_axis_energy_fraction']:.3f}"
        lines.append(
            f"| `{item['run_id']}` | {change('validation_eeg_reconstruction_first', 'validation_eeg_reconstruction_last')} | "
            f"{change('validation_fnirs_reconstruction_first', 'validation_fnirs_reconstruction_last')} | "
            f"{change('validation_clip_first', 'validation_clip_last')} | {auc} | {mrr_e} | {mrr_f} | {rank_e} / {axis_e} | {rank_f} / {axis_f} |"
        )
    lines += [
        "",
        "## Missingness and interpretation boundaries",
        "",
        "- There is no valid epoch-wise AUC/MRR/margin/effective-rank trajectory in the current artifacts. The alignment fields in `metrics/epochs.jsonl` are absent, so the analysis leaves those cells missing rather than copying a final checkpoint value across epochs.",
        "- Reconstruction and CLIP losses are shown on their native scales. Different runs use different dataset mixtures, pair counts, and objective composition; absolute loss levels must not be used to rank runs across panels. The primary figure only makes within-run temporal changes easy to read.",
        "- `analysis_export` rows are intentionally not relabelled as `best` or `latest` unless the source audit explicitly identifies the checkpoint epoch. Existing `analysis_metrics.json` files can contain alignment evidence generated from an exported validation file without an unambiguous checkpoint epoch.",
        "- Checkpoint JSON summaries are retained as separate rows (`best_alignment`, `latest`, etc.). If several summaries exist, they are not averaged or ordered as a time trajectory.",
        "- Existing full-validation alignment exports are labelled `row_weighted_duplicate_unaware_existing_export`: the upstream InventoryDiverseBatchSampler may repeat samples from smaller datasets when constructing a balanced epoch, and the existing diagonal-only positive mask does not deduplicate repeated samples. These values are retained for provenance and must not be mixed with any future deduplicated re-analysis.",
        "- A low CLIP-loss change is not interpreted in isolation: the faithful fixed logit multiplier compresses the CE scale. AUC, retrieval, hardest-negative margin, and geometry are the decisive observed alignment diagnostics where exported.",
        "",
        "## Reproducibility",
        "",
        "The manifest records source hashes for every consumed JSON/JSONL summary. Model `.pt` checkpoints, alignment `.npz` files, and protected-test artifacts were not opened. Re-run the script with the same source root to regenerate the bundle.",
        "",
        "## Files",
        "",
        "- `efrm_checkpoint_trajectory_tidy.csv` — one row per observed epoch/checkpoint summary.",
        "- `trajectory.{png,pdf}` — primary legible figure for the four formal Stage-A runs (short labels; colour and line-style redundancy).",
        "- `trajectory_all_runs.{png,pdf}` — supplementary all-selected-run figure.",
        "- `alignment_geometry.{png,pdf}` — checkpoint-level AUC/MRR and geometry diagnostics.",
        "- `manifest.json` — provenance, field availability, and source hashes.",
        "- `summary.json` — machine-readable run-level summary.",
        "",
    ]
    return "\n".join(lines)


def run_analysis(
    pretraining_root: str | Path,
    output_dir: str | Path,
    *,
    include_all: bool = False,
    run_filters: Sequence[str] | None = None,
) -> dict[str, Any]:
    source_root = Path(pretraining_root).resolve()
    output = Path(output_dir).resolve()
    _configure_matplotlib()
    runs = discover_runs(source_root, include_all=include_all, run_filters=run_filters)
    if not runs:
        raise RuntimeError(f"no runs with metrics/epochs.jsonl found under {source_root}")
    all_rows: list[dict[str, Any]] = []
    inventories: list[dict[str, Any]] = []
    for run in runs:
        rows, inventory = load_run_rows(run, source_root)
        all_rows.extend(rows)
        inventories.append(inventory)
    all_rows.sort(
        key=lambda row: (
            str(row.get("run_id")),
            0 if row.get("record_kind") == "epoch_metric" else 1,
            row.get("epoch") if row.get("epoch") is not None else math.inf,
            str(row.get("checkpoint_label")),
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    tidy_path = output / "efrm_checkpoint_trajectory_tidy.csv"
    _write_tidy_csv(tidy_path, all_rows)
    summary = _summary_by_run(all_rows, inventories)
    summary_payload = {
        "schema": OUTPUT_SCHEMA,
        "generated_at": _now(),
        "source_root": str(source_root),
        "run_count": len(inventories),
        "row_count": len(all_rows),
        "field_columns": list(TIDY_FIELDS),
        "runs": summary,
        "alignment_epoch_interpolation": False,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    _plot_trajectory(all_rows, output)
    _plot_formal_stage_a_trajectory(all_rows, output)
    _plot_alignment(all_rows, output)
    report_path = output / "REPORT.md"
    report_path.write_text(_markdown_report(source_root, output, all_rows, inventories, summary), encoding="utf-8")
    manifest = {
        "schema": SCHEMA,
        "generated_at": _now(),
        "source_root": str(source_root),
        "output_dir": str(output),
        "selection": {
            "include_all": include_all,
            "run_filters": list(run_filters or []),
            "default_rule": "formal_stage_a, source_reference, paper_reference, and development runs; smoke/Stage-B excluded unless --include-all",
        },
        "run_count": len(inventories),
        "row_count": len(all_rows),
        "alignment_epoch_interpolation": False,
        "alignment_metric_basis": "row_weighted_duplicate_unaware_existing_export",
        "model_checkpoints_loaded": False,
        "similarity_matrices_loaded": False,
        "protected_test_opened": False,
        "runs": inventories,
        "outputs": {},
        "missingness_policy": "Only fields physically present in source JSON/JSONL are populated; missing alignment values remain blank/None.",
    }
    # Add output hashes only after every file has been written.
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest["outputs"][path.name] = _sha256(path)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    method_root = Path(__file__).resolve().parents[1]
    repo_root = method_root.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretraining-root",
        type=Path,
        default=method_root / "runs/pretraining",
        help="directory containing EFRM pretraining run directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "comparative_methods/runs/performance_analysis/20260816_p0/efrm_trajectory",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="include every run with metrics/epochs.jsonl, including smoke and Stage-B runs",
    )
    parser.add_argument(
        "--run-filter",
        action="append",
        default=[],
        help="regular expression applied to run_id; may be repeated",
    )
    args = parser.parse_args(argv)
    manifest = run_analysis(
        args.pretraining_root,
        args.output_dir,
        include_all=args.include_all,
        run_filters=args.run_filter,
    )
    print(json.dumps({"output_dir": manifest["output_dir"], "run_count": manifest["run_count"], "row_count": manifest["row_count"]}, indent=2))


if __name__ == "__main__":
    main()
