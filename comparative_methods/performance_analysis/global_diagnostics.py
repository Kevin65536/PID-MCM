#!/usr/bin/env python3
"""Frozen, descriptive diagnostics for the protected comparison campaign.

This module deliberately consumes only the sealed aggregate/cells.csv and its
companion aggregate.json.  It does not tune a method, open prediction files,
or perform a significance test.  The generated figures and tables are
therefore *frozen descriptive/post-hoc* artifacts; folds are displayed as
evaluation units and are never treated as independent subjects.

The command line entry point is::

    .venv/bin/python comparative_methods/performance_analysis/global_diagnostics.py

Use ``--aggregate-dir`` and ``--output-dir`` to point to an alternate sealed
aggregate or a temporary output directory (the latter is useful for tests).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGGREGATE_DIR = (
    REPO_ROOT
    / "comparative_methods"
    / "runs"
    / "protected_campaign"
    / "joint-comparison-protected-20260813-v3-single-gpu"
    / "joint-comparison-protected-20260813-v3-single-gpu"
    / "aggregate"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "comparative_methods"
    / "runs"
    / "performance_analysis"
    / "20260816_p0"
    / "global_diagnostics"
)

CLASSIFICATION_TASK_ORDER = (
    "dsr",
    "mental_arithmetic",
    "motor_imagery",
    "nback",
    "visual",
    "wg",
)
METHOD_ORDER = (
    "biot",
    "cbramod",
    "efrm_sync_200_10_variable_channel_v1",
    "normwear_eeg_fnirs_adapted",
    "reve",
    "brainfusion_nvc_csp_stacking_reimplementation",
)
METHOD_LABELS = {
    "biot": "BIOT",
    "cbramod": "CBraMod",
    "efrm_sync_200_10_variable_channel_v1": "EFRM",
    "normwear_eeg_fnirs_adapted": "NormWear",
    "reve": "REVE",
    "brainfusion_nvc_csp_stacking_reimplementation": "BrainFusion",
}
TASK_LABELS = {
    "dsr": "DSR",
    "mental_arithmetic": "Mental arithmetic",
    "motor_imagery": "Motor imagery",
    "nback": "n-back",
    "refed_regression": "REFED",
    "visual": "Visual",
    "wg": "Word generation",
}
METRIC_LABELS = {
    "macro_f1": "Macro-F1",
    "native_coordinate_masked_ccc": "Masked CCC",
}

STATUS_CODES = {
    "TABLE_READY": "ready",
    "TABLE_READY_WITH_NOTE": "note",
    "OVERLAP_TRACK_ONLY": "overlap",
    "REJECTED_VALUE": "rejected",
    "FAILURE_RESULT": "failure",
    "INVALID_VALUE": "invalid",
    "UNSUPPORTED": "NA",
}

# ColorBrewer BrBG is a blue/green-to-orange/brown diverging palette.  It is
# used instead of a red/green palette so the signed delta remains readable for
# common color-vision deficiencies.  The categorical bars use Okabe-Ito-like
# blue/orange/green colors.
BAR_COLORS = ("#0072B2", "#E69F00", "#009E73")


@dataclass(frozen=True)
class AggregateRecord:
    """One row from aggregate/cells.csv plus frozen fold values."""

    method_id: str
    method_slug: str
    task: str
    track: str
    metric: str
    value: float | None
    fold_sample_sd: float | None
    b0: float | None
    minimum_admissible: float | None
    preferred_target: float | None
    numeric_acceptance: str
    terminal: str
    fold_values: tuple[float, ...]

    @property
    def delta_b0(self) -> float | None:
        if self.value is None or self.b0 is None:
            return None
        return float(self.value - self.b0)

    @property
    def supported(self) -> bool:
        return self.value is not None and self.b0 is not None


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_value(value: Any) -> Any:
    """Convert numpy/scalar values to JSON-safe values."""

    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("method_id", "")),
        str(row.get("task", "")),
        str(row.get("track", "")),
        str(row.get("metric", "")),
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"cells.csv is empty: {path}")
    required = {
        "method_id",
        "task",
        "track",
        "metric",
        "value",
        "fold_sample_sd",
        "B0",
        "minimum_admissible",
        "preferred_target",
        "numeric_acceptance",
        "terminal",
    }
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"cells.csv missing columns: {sorted(missing)}")
    return rows


def _fold_values_from_payload(
    row: Mapping[str, Any], aggregate_payload: Mapping[str, Any]
) -> tuple[float, ...]:
    """Read fold-level values without touching prediction artifacts.

    ``fold_rows`` is preferred because it records the outer-fold index and
    seed-mean provenance.  Older aggregate payloads may only contain a
    ``fold_values`` list in ``cells``; that form remains supported.
    """

    fold_rows = aggregate_payload.get("fold_rows", [])
    if isinstance(fold_rows, list):
        matching = [
            r
            for r in fold_rows
            if isinstance(r, Mapping) and _key(r) == _key(row)
        ]
        matching.sort(key=lambda r: _int_or_none(r.get("outer_fold")) or 0)
        values = tuple(
            value
            for value in (_float_or_none(r.get("seed_mean")) for r in matching)
            if value is not None
        )
        if values:
            return values

    cells = aggregate_payload.get("cells", [])
    if isinstance(cells, list):
        for candidate in cells:
            if isinstance(candidate, Mapping) and _key(candidate) == _key(row):
                raw_values = candidate.get("fold_values", [])
                if isinstance(raw_values, list):
                    return tuple(
                        value
                        for value in (_float_or_none(v) for v in raw_values)
                        if value is not None
                    )
    return ()


def load_records(
    cells_csv: Path,
    aggregate_json: Path | None = None,
) -> tuple[list[AggregateRecord], dict[str, Any]]:
    """Load sealed aggregate rows and attach their frozen fold summaries."""

    aggregate_json = aggregate_json or cells_csv.with_name("aggregate.json")
    csv_rows = _read_csv(cells_csv)
    payload = _read_json(aggregate_json)
    records: list[AggregateRecord] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in csv_rows:
        key = _key(row)
        if key in seen:
            raise ValueError(f"duplicate aggregate cell: {key}")
        seen.add(key)
        method_id = str(row["method_id"])
        method_slug = str(row.get("method_slug") or method_id)
        numeric_acceptance = str(row.get("numeric_acceptance") or "")
        records.append(
            AggregateRecord(
                method_id=method_id,
                method_slug=method_slug,
                task=str(row["task"]),
                track=str(row["track"]),
                metric=str(row["metric"]),
                value=_float_or_none(row.get("value")),
                fold_sample_sd=_float_or_none(row.get("fold_sample_sd")),
                b0=_float_or_none(row.get("B0")),
                minimum_admissible=_float_or_none(row.get("minimum_admissible")),
                preferred_target=_float_or_none(row.get("preferred_target")),
                numeric_acceptance=numeric_acceptance,
                terminal=str(row.get("terminal") or numeric_acceptance),
                fold_values=_fold_values_from_payload(row, payload),
            )
        )
    if len(seen) != len(csv_rows):  # defensive; the duplicate check is primary
        raise ValueError("aggregate cell identity is not unique")
    return records, payload


def _record_sort_key(record: AggregateRecord) -> tuple[int, int, str, str]:
    method_rank = METHOD_ORDER.index(record.method_id) if record.method_id in METHOD_ORDER else len(METHOD_ORDER)
    task_rank = (
        CLASSIFICATION_TASK_ORDER.index(record.task)
        if record.task in CLASSIFICATION_TASK_ORDER
        else len(CLASSIFICATION_TASK_ORDER)
    )
    return method_rank, task_rank, record.metric, record.task


def _ordered_methods(records: Iterable[AggregateRecord]) -> list[str]:
    values = {record.method_id for record in records}
    return sorted(values, key=lambda value: (METHOD_ORDER.index(value) if value in METHOD_ORDER else len(METHOD_ORDER), value))


def _ordered_tasks(records: Iterable[AggregateRecord], metric: str | None = None) -> list[str]:
    values = {record.task for record in records if metric is None or record.metric == metric}
    return sorted(
        values,
        key=lambda value: (
            CLASSIFICATION_TASK_ORDER.index(value)
            if value in CLASSIFICATION_TASK_ORDER
            else len(CLASSIFICATION_TASK_ORDER),
            value,
        ),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output = {}
            for field in fieldnames:
                value = row.get(field)
                if value is None:
                    output[field] = ""
                elif isinstance(value, bool):
                    output[field] = "true" if value else "false"
                else:
                    output[field] = value
            writer.writerow(output)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_value)
        handle.write("\n")


def build_task_method_rows(records: Sequence[AggregateRecord]) -> list[dict[str, Any]]:
    """Return a long, explicit task×method table, including unsupported cells."""

    rows = []
    for record in sorted(records, key=_record_sort_key):
        rows.append(
            {
                "method_id": record.method_id,
                "method_slug": record.method_slug,
                "method_label": METHOD_LABELS.get(record.method_id, record.method_slug),
                "task": record.task,
                "task_label": TASK_LABELS.get(record.task, record.task),
                "track": record.track,
                "metric": record.metric,
                "metric_label": METRIC_LABELS.get(record.metric, record.metric),
                "value": record.value,
                "B0": record.b0,
                "value_minus_B0": record.delta_b0,
                "fold_sample_sd": record.fold_sample_sd,
                "n_folds": len(record.fold_values),
                "minimum_admissible": record.minimum_admissible,
                "preferred_target": record.preferred_target,
                "numeric_acceptance": record.numeric_acceptance,
                "terminal": record.terminal,
                "supported": record.supported,
                "missing_cell": not record.supported,
                "protected_use": "frozen descriptive/post-hoc",
            }
        )
    return rows


def build_fold_rows(
    records: Sequence[AggregateRecord],
    aggregate_payload: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Flatten fold values; retain one explicit missing row for unsupported cells.

    When the sealed aggregate payload is available, preserve its outer-fold
    index, seed-level SD, job IDs, and campaign disposition.  The fallback
    path keeps this helper useful for small synthetic tables and older payloads
    that only contain a ``fold_values`` list.
    """

    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=_record_sort_key):
        payload_fold_rows = []
        if aggregate_payload is not None:
            raw_fold_rows = aggregate_payload.get("fold_rows", [])
            if isinstance(raw_fold_rows, list):
                payload_fold_rows = [
                    row
                    for row in raw_fold_rows
                    if isinstance(row, Mapping) and _key(row) == _key(record.__dict__)
                ]
                payload_fold_rows.sort(key=lambda row: _int_or_none(row.get("outer_fold")) or 0)
        if payload_fold_rows:
            for payload_row in payload_fold_rows:
                fold_index = _int_or_none(payload_row.get("outer_fold"))
                fold_value = _float_or_none(payload_row.get("seed_mean"))
                if fold_index is None or fold_value is None:
                    continue
                b0 = _float_or_none(payload_row.get("B0_seed_mean"))
                if b0 is None:
                    b0 = record.b0
                rows.append(
                    {
                        "method_id": record.method_id,
                        "method_label": METHOD_LABELS.get(record.method_id, record.method_slug),
                        "task": record.task,
                        "task_label": TASK_LABELS.get(record.task, record.task),
                        "track": record.track,
                        "metric": record.metric,
                        "outer_fold": fold_index,
                        "fold_value": fold_value,
                        "B0": b0,
                        "fold_value_minus_B0": None if b0 is None else fold_value - b0,
                        "seed_sample_sd": _float_or_none(payload_row.get("seed_sample_sd")),
                        "cell_value": record.value,
                        "terminal": record.terminal,
                        "missing_cell": False,
                        "job_ids": ";".join(str(v) for v in payload_row.get("job_ids", [])),
                        "campaign_disposition": payload_row.get("campaign_disposition", ""),
                        "unit_note": "outer fold seed-mean; descriptive, not independent subject",
                    }
                )
            if any(
                row["method_id"] == record.method_id
                and row["task"] == record.task
                and row["metric"] == record.metric
                for row in rows
            ):
                continue
        if not record.fold_values:
            rows.append(
                {
                    "method_id": record.method_id,
                    "method_label": METHOD_LABELS.get(record.method_id, record.method_slug),
                    "task": record.task,
                    "task_label": TASK_LABELS.get(record.task, record.task),
                    "track": record.track,
                    "metric": record.metric,
                    "outer_fold": "",
                    "fold_value": None,
                    "B0": record.b0,
                    "fold_value_minus_B0": None,
                    "seed_sample_sd": None,
                    "cell_value": record.value,
                    "terminal": record.terminal,
                    "missing_cell": True,
                    "job_ids": "",
                    "campaign_disposition": "",
                    "unit_note": "unsupported cell; no fold value",
                }
            )
            continue
        for fold_index, fold_value in enumerate(record.fold_values):
            rows.append(
                {
                    "method_id": record.method_id,
                    "method_label": METHOD_LABELS.get(record.method_id, record.method_slug),
                    "task": record.task,
                    "task_label": TASK_LABELS.get(record.task, record.task),
                    "track": record.track,
                    "metric": record.metric,
                    "outer_fold": fold_index,
                    "fold_value": fold_value,
                    "B0": record.b0,
                    "fold_value_minus_B0": (
                        None if record.b0 is None else fold_value - record.b0
                    ),
                    "seed_sample_sd": None,
                    "cell_value": record.value,
                    "terminal": record.terminal,
                    "missing_cell": False,
                    "job_ids": "",
                    "campaign_disposition": "",
                    "unit_note": "outer fold seed-mean; descriptive, not independent subject",
                }
            )
    return rows


def _group_supported(
    records: Sequence[AggregateRecord],
    *,
    metric: str,
    group_field: str,
) -> list[dict[str, Any]]:
    groups: dict[str, list[AggregateRecord]] = defaultdict(list)
    for record in records:
        if record.metric != metric or not record.supported:
            continue
        groups[str(getattr(record, group_field))].append(record)
    output = []
    if group_field == "method_id":
        group_order = _ordered_methods(values for rows in groups.values() for values in rows)
    else:
        group_order = _ordered_tasks(
            [row for rows in groups.values() for row in rows], metric=metric
        )
    for group in group_order:
        values = groups[group]
        metric_values = [float(row.value) for row in values if row.value is not None]
        deltas = [float(row.delta_b0) for row in values if row.delta_b0 is not None]
        output.append(
            {
                "metric": metric,
                "group": group,
                "group_label": (
                    METHOD_LABELS.get(group, group)
                    if group_field == "method_id"
                    else TASK_LABELS.get(group, group)
                ),
                "n_cells": len(values),
                "mean_value": float(np.mean(metric_values)),
                "sd_value": float(np.std(metric_values, ddof=1)) if len(metric_values) > 1 else 0.0,
                "mean_value_minus_B0": float(np.mean(deltas)),
                "sd_value_minus_B0": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                "unit_note": "cell-level descriptive summary; no p-value",
            }
        )
    return output


def _fold_summary(
    fold_rows: Sequence[Mapping[str, Any]], metric: str
) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        if row.get("metric") != metric or row.get("missing_cell"):
            continue
        fold = _int_or_none(row.get("outer_fold"))
        value = _float_or_none(row.get("fold_value"))
        delta = _float_or_none(row.get("fold_value_minus_B0"))
        if fold is None or value is None:
            continue
        grouped[fold].append({"value": value, "delta": delta})
    output = []
    for fold, values in sorted(grouped.items()):
        raw = [float(row["value"]) for row in values]
        delta = [float(row["delta"]) for row in values if row["delta"] is not None]
        output.append(
            {
                "metric": metric,
                "outer_fold": fold,
                "n_cells": len(raw),
                "mean_value": float(np.mean(raw)),
                "sd_value": float(np.std(raw, ddof=1)) if len(raw) > 1 else 0.0,
                "mean_value_minus_B0": float(np.mean(delta)) if delta else None,
                "sd_value_minus_B0": float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0,
                "unit_note": "outer fold summary across cells; folds are not independent subjects",
            }
        )
    return output


def _balanced_matrix(
    records: Sequence[AggregateRecord], metric: str
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    supported = [r for r in records if r.metric == metric and r.supported]
    tasks = _ordered_tasks(supported, metric=metric)
    by_method: dict[str, dict[str, AggregateRecord]] = defaultdict(dict)
    for record in supported:
        by_method[record.method_id][record.task] = record
    methods = [
        method
        for method in _ordered_methods(supported)
        if all(task in by_method[method] for task in tasks)
    ]
    values = np.asarray(
        [[by_method[method][task].value for task in tasks] for method in methods],
        dtype=float,
    )
    deltas = np.asarray(
        [[by_method[method][task].delta_b0 for task in tasks] for method in methods],
        dtype=float,
    )
    return methods, tasks, values, deltas


def _variance_components(matrix: np.ndarray) -> dict[str, float | int | None]:
    """Two-way descriptive SS partition for a complete method×task panel."""

    if matrix.ndim != 2 or matrix.size == 0:
        return {
            "n_methods": 0,
            "n_tasks": 0,
            "ss_total": None,
            "ss_method": None,
            "ss_task": None,
            "ss_residual": None,
            "method_pct": None,
            "task_pct": None,
            "residual_pct": None,
        }
    n_methods, n_tasks = matrix.shape
    grand = float(matrix.mean())
    method_means = matrix.mean(axis=1)
    task_means = matrix.mean(axis=0)
    ss_method = float(n_tasks * np.square(method_means - grand).sum())
    ss_task = float(n_methods * np.square(task_means - grand).sum())
    residual = matrix - method_means[:, None] - task_means[None, :] + grand
    ss_residual = float(np.square(residual).sum())
    ss_total = float(np.square(matrix - grand).sum())
    denominator = ss_method + ss_task + ss_residual
    if denominator <= 0:
        percentages = (0.0, 0.0, 0.0)
    else:
        percentages = tuple(
            float(component / denominator)
            for component in (ss_method, ss_task, ss_residual)
        )
    return {
        "n_methods": n_methods,
        "n_tasks": n_tasks,
        "ss_total": ss_total,
        "ss_method": ss_method,
        "ss_task": ss_task,
        "ss_residual": ss_residual,
        "method_pct": percentages[0],
        "task_pct": percentages[1],
        "residual_pct": percentages[2],
        "unit_note": "descriptive complete-case SS partition; not an inferential ANOVA",
    }


def build_descriptive_summary(
    records: Sequence[AggregateRecord],
    fold_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = sorted({record.metric for record in records})
    output: dict[str, Any] = {
        "protected_data_policy": "frozen descriptive/post-hoc; no tuning; folds are not independent subjects",
        "method_summaries": {},
        "task_summaries": {},
        "fold_summaries": {},
        "variance_components": {},
    }
    for metric in metrics:
        methods, tasks, values, deltas = _balanced_matrix(records, metric)
        output["method_summaries"][metric] = _group_supported(
            records, metric=metric, group_field="method_id"
        )
        output["task_summaries"][metric] = _group_supported(
            records, metric=metric, group_field="task"
        )
        output["fold_summaries"][metric] = _fold_summary(fold_rows, metric)
        output["variance_components"][metric] = {
            "methods": methods,
            "tasks": tasks,
            "raw_value": _variance_components(values),
            "value_minus_B0": _variance_components(deltas),
        }
    return output


def _summary_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level, key in (
        ("method", "method_summaries"),
        ("task", "task_summaries"),
        ("fold", "fold_summaries"),
    ):
        for metric, values in summary.get(key, {}).items():
            for row in values:
                output = dict(row)
                output["level"] = level
                output["metric"] = metric
                rows.append(output)
    for metric, components in summary.get("variance_components", {}).items():
        for panel_name in ("raw_value", "value_minus_B0"):
            panel = components.get(panel_name, {})
            for component, pct_key in (
                ("method", "method_pct"),
                ("task", "task_pct"),
                ("residual", "residual_pct"),
            ):
                rows.append(
                    {
                        "level": "variance_component",
                        "metric": metric,
                        "group": component,
                        "panel": panel_name,
                        "n_methods": panel.get("n_methods"),
                        "n_tasks": panel.get("n_tasks"),
                        "ss_total": panel.get("ss_total"),
                        "ss_method": panel.get("ss_method"),
                        "ss_task": panel.get("ss_task"),
                        "ss_residual": panel.get("ss_residual"),
                        "percentage": panel.get(pct_key),
                        "unit_note": panel.get("unit_note"),
                    }
                )
    return rows


def _matrix_rows(
    records: Sequence[AggregateRecord], metric: str, field: str
) -> list[dict[str, Any]]:
    methods = _ordered_methods(records)
    tasks = _ordered_tasks(records, metric=metric)
    lookup = {(r.task, r.method_id): r for r in records if r.metric == metric}
    rows = []
    for task in tasks:
        row: dict[str, Any] = {"task": task, "task_label": TASK_LABELS.get(task, task)}
        for method in methods:
            record = lookup.get((task, method))
            if record is None:
                row[method] = None
            elif field == "value_minus_B0":
                row[method] = record.delta_b0
            else:
                row[method] = getattr(record, field)
        rows.append(row)
    return rows


def _safe_values(records: Sequence[AggregateRecord], metric: str) -> list[float]:
    return [
        float(record.delta_b0)
        for record in records
        if record.metric == metric and record.delta_b0 is not None
    ]


def _metric_stem(metric: str) -> str:
    return "macro_f1" if metric == "macro_f1" else metric


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> tuple[Path, Path]:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def _status_code(status: str) -> str:
    return STATUS_CODES.get(status, status or "NA")


def plot_value_minus_b0_heatmap(
    records: Sequence[AggregateRecord],
    output_dir: Path,
    *,
    metric: str = "macro_f1",
    stem: str | None = None,
) -> dict[str, Any]:
    """Plot a signed delta heatmap with explicit gray missing cells."""

    metric_records = [r for r in records if r.metric == metric]
    methods = _ordered_methods(metric_records)
    tasks = _ordered_tasks(metric_records, metric=metric)
    lookup = {(record.task, record.method_id): record for record in metric_records}
    matrix = np.full((len(tasks), len(methods)), np.nan, dtype=float)
    for i, task in enumerate(tasks):
        for j, method in enumerate(methods):
            record = lookup.get((task, method))
            if record is not None and record.delta_b0 is not None:
                matrix[i, j] = record.delta_b0

    max_abs = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
    max_abs = max(max_abs, 0.01)
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("BrBG").copy()
    cmap.set_bad("#BDBDBD")
    fig, ax = plt.subplots(
        figsize=(max(9.0, len(methods) * 1.55), max(5.7, len(tasks) * 0.82))
    )
    image = ax.imshow(masked, cmap=cmap, norm=Normalize(vmin=-max_abs, vmax=max_abs), aspect="auto")
    ax.set_xticks(
        range(len(methods)),
        [METHOD_LABELS.get(m, m) for m in methods],
        rotation=32,
        ha="right",
        fontsize=10,
    )
    ax.set_yticks(range(len(tasks)), [TASK_LABELS.get(t, t) for t in tasks], fontsize=10)
    ax.set_xlabel("Method")
    ax.set_ylabel("Task")
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} − B0 | frozen aggregate", fontsize=13, pad=10)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.04)
    colorbar.set_label("value − B0")
    for i, task in enumerate(tasks):
        for j, method in enumerate(methods):
            record = lookup.get((task, method))
            if record is None or record.delta_b0 is None:
                label = "NA"
                color = "#202020"
            else:
                label = f"{record.delta_b0:+.3f}\n{_status_code(record.terminal)}"
                color = "white" if abs(record.delta_b0) > 0.55 * max_abs else "#202020"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)
    # Keep the policy and interpretation note in the manifest so it
    # cannot collide with rotated method labels in the raster figure.
    fig.subplots_adjust(left=0.17, right=0.90, top=0.90, bottom=0.24)
    if stem is None:
        stem = "value_minus_b0_heatmap" if metric == "macro_f1" else f"value_minus_b0_{_metric_stem(metric)}_heatmap"
    png, pdf = _save_figure(fig, output_dir, stem)
    finite = matrix[np.isfinite(matrix)]
    positive = int(np.sum(finite > 0))
    negative = int(np.sum(finite < 0))
    missing = int(np.sum(~np.isfinite(matrix)))
    alt = (
        f"Signed {METRIC_LABELS.get(metric, metric)} minus B0 heatmap for "
        f"{len(tasks)} tasks and {len(methods)} methods. "
        f"{positive} supported cells are above B0 and {negative} are below B0; "
        f"{missing} unsupported or missing cells are gray and labeled NA. "
        "Each cell is descriptive and no inferential comparison is encoded."
    )
    return {
        "figure_id": stem,
        "metric": metric,
        "png": png.name,
        "pdf": pdf.name,
        "alt_text": alt,
        "missing_cells": missing,
        "positive_cells": positive,
        "negative_cells": negative,
        "color_palette": "BrBG diverging; gray masked missing cells",
    }


def plot_descriptive_decomposition(
    records: Sequence[AggregateRecord],
    fold_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    output_dir: Path,
    *,
    metric: str = "macro_f1",
) -> dict[str, Any]:
    """Plot fold, method, task and complete-panel descriptive summaries."""

    method_rows = summary.get("method_summaries", {}).get(metric, [])
    task_rows = summary.get("task_summaries", {}).get(metric, [])
    fold_summary = summary.get("fold_summaries", {}).get(metric, [])
    variance = summary.get("variance_components", {}).get(metric, {})
    raw_components = variance.get("raw_value", {})
    delta_components = variance.get("value_minus_B0", {})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))
    fig.suptitle(
        f"{METRIC_LABELS.get(metric, metric)}: frozen aggregate descriptive decomposition",
        fontsize=13,
        y=0.965,
    )

    # Fold panel: mean and cell-level spread.  This is a visualization of
    # heterogeneity, not a confidence interval or an independent sample test.
    ax = axes[0, 0]
    if fold_summary:
        x = np.asarray([int(row["outer_fold"]) for row in fold_summary])
        means = np.asarray([float(row["mean_value_minus_B0"]) for row in fold_summary])
        spread = np.asarray([float(row["sd_value_minus_B0"]) for row in fold_summary])
        ax.errorbar(x, means, yerr=spread, fmt="o-", color=BAR_COLORS[0], capsize=4, lw=1.5)
        ax.set_xticks(x)
    ax.axhline(0.0, color="#777777", lw=0.8, ls="--")
    ax.set_title("Outer-fold mean(value − B0)", fontsize=11)
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("value − B0")
    ax.grid(axis="y", color="#dddddd", lw=0.7)

    # Method panel.
    ax = axes[0, 1]
    if method_rows:
        x = np.arange(len(method_rows))
        means = [row["mean_value_minus_B0"] for row in method_rows]
        spread = [row["sd_value_minus_B0"] for row in method_rows]
        ax.errorbar(x, means, yerr=spread, fmt="o", color=BAR_COLORS[1], capsize=4, lw=1.5)
        ax.set_xticks(
            x,
            [row["group_label"] for row in method_rows],
            rotation=25,
            ha="right",
            fontsize=9,
        )
    ax.axhline(0.0, color="#777777", lw=0.8, ls="--")
    ax.set_title("Method-level mean(value − B0)", fontsize=11)
    ax.set_ylabel("value − B0")
    ax.grid(axis="y", color="#dddddd", lw=0.7)

    # Task panel.
    ax = axes[1, 0]
    if task_rows:
        x = np.arange(len(task_rows))
        means = [row["mean_value_minus_B0"] for row in task_rows]
        spread = [row["sd_value_minus_B0"] for row in task_rows]
        ax.errorbar(x, means, yerr=spread, fmt="o", color=BAR_COLORS[2], capsize=4, lw=1.5)
        ax.set_xticks(
            x,
            [row["group_label"] for row in task_rows],
            rotation=25,
            ha="right",
            fontsize=9,
        )
    ax.axhline(0.0, color="#777777", lw=0.8, ls="--")
    ax.set_title("Task-level mean(value − B0)", fontsize=11)
    ax.set_ylabel("value − B0")
    ax.grid(axis="y", color="#dddddd", lw=0.7)

    # Variance panel: raw and B0-centered complete-case proportions.
    ax = axes[1, 1]
    labels = ["method", "task", "residual"]
    positions = np.arange(len(labels))
    raw = [float(raw_components.get(f"{label}_pct", 0.0)) for label in labels]
    delta = [float(delta_components.get(f"{label}_pct", 0.0)) for label in labels]
    width = 0.36
    ax.bar(positions - width / 2, raw, width, color=BAR_COLORS[0], label="raw value")
    ax.bar(positions + width / 2, delta, width, color=BAR_COLORS[1], label="value − B0")
    for xpos, value in zip(positions - width / 2, raw):
        ax.text(xpos, value + 0.015, f"{100 * value:.1f}%", ha="center", va="bottom", fontsize=8)
    for xpos, value in zip(positions + width / 2, delta):
        ax.text(xpos, value + 0.015, f"{100 * value:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(positions, ["method", "task", "residual"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Descriptive SS proportion")
    ax.set_title("Complete-case SS proportions", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color="#dddddd", lw=0.7)
    # Caption-level policy and inferential caveats stay in the manifest rather
    # than being drawn over a panel baseline.
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.18, hspace=0.58, wspace=0.28)
    png, pdf = _save_figure(fig, output_dir, "descriptive_decomposition")
    alt = (
        f"Four-panel descriptive decomposition of frozen {METRIC_LABELS.get(metric, metric)}. "
        "The first three panels show B0-centered outer-fold, method, and task means with "
        "descriptive cell spread. The final panel shows raw versus B0-centered complete-case "
        "sum-of-squares proportions. Error bars are not confidence intervals; folds are not "
        "treated as independent subjects, and the decomposition is not an inferential ANOVA."
    )
    return {
        "figure_id": "descriptive_decomposition",
        "metric": metric,
        "png": png.name,
        "pdf": pdf.name,
        "alt_text": alt,
        "color_palette": "Okabe-Ito-like blue/orange/green accents",
        "complete_case_methods": raw_components.get("n_methods"),
        "complete_case_tasks": raw_components.get("n_tasks"),
    }


def run_diagnostics(
    *,
    aggregate_dir: Path = DEFAULT_AGGREGATE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Generate all P0 global diagnostics and return a compact manifest."""

    aggregate_dir = Path(aggregate_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cells_csv = aggregate_dir / "cells.csv"
    aggregate_json = aggregate_dir / "aggregate.json"
    records, aggregate_payload = load_records(cells_csv, aggregate_json)
    fold_rows = build_fold_rows(records, aggregate_payload)
    task_rows = build_task_method_rows(records)

    table_fields = [
        "method_id",
        "method_slug",
        "method_label",
        "task",
        "task_label",
        "track",
        "metric",
        "metric_label",
        "value",
        "B0",
        "value_minus_B0",
        "fold_sample_sd",
        "n_folds",
        "minimum_admissible",
        "preferred_target",
        "numeric_acceptance",
        "terminal",
        "supported",
        "missing_cell",
        "protected_use",
    ]
    _write_csv(output_dir / "task_method_table.csv", task_rows, table_fields)
    _write_csv(
        output_dir / "fold_level.csv",
        fold_rows,
        [
            "method_id",
            "method_label",
            "task",
            "task_label",
            "track",
            "metric",
            "outer_fold",
            "fold_value",
            "B0",
            "fold_value_minus_B0",
            "seed_sample_sd",
            "cell_value",
            "terminal",
            "missing_cell",
            "job_ids",
            "campaign_disposition",
            "unit_note",
        ],
    )

    for metric in sorted({record.metric for record in records}):
        matrix_fields = ["task", "task_label"] + _ordered_methods(
            [record for record in records if record.metric == metric]
        )
        _write_csv(
            output_dir / f"task_method_value_{_metric_stem(metric)}.csv",
            _matrix_rows(records, metric, "value"),
            matrix_fields,
        )
        _write_csv(
            output_dir / f"task_method_delta_b0_{_metric_stem(metric)}.csv",
            _matrix_rows(records, metric, "value_minus_B0"),
            matrix_fields,
        )

    summary = build_descriptive_summary(records, fold_rows)
    _write_json(output_dir / "descriptive_decomposition.json", summary)
    summary_fields = sorted(
        {
            field
            for row in _summary_rows(summary)
            for field in row.keys()
        }
        | {"level", "metric"}
    )
    _write_csv(output_dir / "descriptive_decomposition.csv", _summary_rows(summary), summary_fields)

    figures: list[dict[str, Any]] = []
    for metric in sorted({record.metric for record in records}):
        figures.append(plot_value_minus_b0_heatmap(records, output_dir, metric=metric))
    if "macro_f1" in {record.metric for record in records}:
        figures.append(plot_descriptive_decomposition(records, fold_rows, summary, output_dir, metric="macro_f1"))
    result = {
        "schema": "global_diagnostics_v2",
        "analysis_id": "20260816_p0_global_diagnostics",
        "created_by": "comparative_methods/performance_analysis/global_diagnostics.py",
        "protected_data_policy": "frozen descriptive/post-hoc; no tuning; folds are not independent subjects",
        "source": {
            "cells_csv": str(cells_csv),
            "cells_csv_sha256": _sha256_file(cells_csv),
            "aggregate_json": str(aggregate_json),
            "aggregate_json_sha256": _sha256_file(aggregate_json),
            "campaign_id": aggregate_payload.get("campaign_id"),
            "aggregate_schema": aggregate_payload.get("schema"),
        },
        "n_cells": len(records),
        "n_supported_cells": int(sum(record.supported for record in records)),
        "n_missing_or_unsupported_cells": int(sum(not record.supported for record in records)),
        "n_fold_rows": len([row for row in fold_rows if not row.get("missing_cell")]),
        "figures": figures,
        "outputs": {
            "task_method_table": "task_method_table.csv",
            "fold_level": "fold_level.csv",
            "descriptive_decomposition_csv": "descriptive_decomposition.csv",
            "descriptive_decomposition_json": "descriptive_decomposition.json",
        },
        "method_order": _ordered_methods(records),
        "task_order_by_metric": {
            metric: _ordered_tasks(records, metric=metric)
            for metric in sorted({record.metric for record in records})
        },
    }
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate-dir",
        type=Path,
        default=DEFAULT_AGGREGATE_DIR,
        help="sealed directory containing cells.csv and aggregate.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for machine-readable tables and figures",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = run_diagnostics(aggregate_dir=args.aggregate_dir, output_dir=args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "figures": manifest["figures"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI smoke
    raise SystemExit(main())
