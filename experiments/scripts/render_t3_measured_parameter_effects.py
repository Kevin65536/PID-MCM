#!/usr/bin/env python3
"""Post-hoc visual audit of the measured T3a Balloon parameter fit.

The renderer only consumes a completed, public measured run.  It reloads the
same fit-cohort gauge and checks its calibration before touching any output.
The protected 24--29 split is rejected before the physiology loader is called.
All curves are in the run's standardized EEG/HbO/HbR observation coordinates;
they are not absolute concentrations or recovered physiological ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_t3_measured_reconstruction_null import (  # noqa: E402
    OBS_NAMES,
    PARAMETER_NAMES,
    PreparedTrial,
    _center_mask,
    _fit_models,
    _mode_input,
    _parameter_values,
    _prepare_measured_series,
    _replace_parameter_values,
    _split_stage_trials,
    _trial_observation,
    load_config,
)
from src.inference.t3a_balloon_robust_ssm import BalloonParameters, smooth_balloon  # noqa: E402
from src.visualization.token_physiology_plots import save_figure_atomic  # noqa: E402


SOURCE_SCHEMA = "t3_measured_reconstruction_null_v1"
PUBLIC_SUBJECTS = tuple(f"subject_{i:02d}" for i in range(1, 24))
FIT_SUBJECTS = tuple(f"subject_{i:02d}" for i in range(1, 19))
VALIDATION_SUBJECTS = tuple(f"subject_{i:02d}" for i in range(19, 24))
PROTECTED_SUBJECTS = tuple(f"subject_{i:02d}" for i in range(24, 30))
TARGET_COLUMNS = {"EEG": 0, "HbO": 1, "HbR": 2}
STAGE_IDS = (
    "M0_fixed",
    "M1_kappa",
    "M5_plus_E0_diagnostic",
)
PARAMETER_LABELS = {
    "beta": "β 神经血管增益",
    "kappa": "κ 舒张信号衰减",
    "tau": "τ 血液通过时间",
    "gamma": "γ 血流反馈",
    "alpha": "α 流出-容积指数",
    "E0": "E0 静息氧提取",
}
STAGE_LABELS = {
    "M0_fixed": "M0 全固定",
    "M1_kappa": "M1 仅 κ",
    "M5_plus_E0_diagnostic": "M5 六参数（诊断）",
}
CJK_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP", "AR PL UMing CN", "DejaVu Sans"],
    "axes.unicode_minus": True,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}
STAGE_COLORS = {"M0_fixed": "#555555", "M1_kappa": "#0072B2", "M5_plus_E0_diagnostic": "#D55E00"}
STAGE_STYLES = {"M0_fixed": (0, (3, 2)), "M1_kappa": "-", "M5_plus_E0_diagnostic": (0, (7, 2))}
TRADEOFF_COLORS = ("#555555", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6A3D9A", "#E69F00", "#56B4E9")
TRADEOFF_MARKERS = ("o", "s", "^", "v", "D", "P", "X", "<")
OAT_COLORS = {"low": "#0072B2", "prior": "#009E73", "high": "#D55E00"}
OAT_STYLES = {"low": (0, (3, 2)), "prior": "-", "high": (0, (7, 2))}
OAT_LABELS = {"low": "下界", "prior": "先验均值", "high": "上界"}


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    temporary.replace(path)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return "" if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _require_source(source: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    if not source.is_dir():
        raise ValueError(f"source run does not exist: {source}")
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("source run manifest.json is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SOURCE_SCHEMA or manifest.get("analysis_kind") != "staged_subject_parameter_fit":
        raise ValueError("source run is not the registered staged measured fit")
    if manifest.get("completion_status") != "complete":
        raise ValueError("source run must have completion_status=complete")
    if manifest.get("protected_data_opened") is not False or manifest.get("protected_data_enabled") is not False:
        raise ValueError("source run indicates protected data access")
    if tuple(manifest.get("closed_protected_subjects", ())) != PROTECTED_SUBJECTS:
        raise ValueError("source manifest closed protected subjects do not match 24--29")
    if tuple(manifest.get("fit_subjects", ())) != FIT_SUBJECTS or tuple(manifest.get("validation_subjects", ())) != VALIDATION_SUBJECTS:
        raise ValueError("source manifest public subject split mismatch")
    config_path = source / "resolved_config.yaml"
    if not config_path.is_file():
        raise ValueError("source run resolved_config.yaml is required")
    config = load_config(config_path)
    condition = config["data"]["conditions"][0]
    if tuple(map(str, condition["protected_subjects"])) != PROTECTED_SUBJECTS:
        raise ValueError("resolved config protected subjects do not match 24--29")
    for filename in ("calibration.json", "subject_stage_parameters.csv", "subject_stage_metrics.csv", "subject_stage_geometry.csv", "subject_final_parameters.csv"):
        if not (source / filename).is_file():
            raise ValueError(f"source run file is missing: {filename}")
    for filename in ("subject_stage_parameters.csv", "subject_stage_metrics.csv", "subject_stage_geometry.csv", "subject_final_parameters.csv"):
        rows = _read_csv(source / filename)
        if "subject" in (rows[0] if rows else {}):
            bad = sorted({row["subject"] for row in rows if row.get("subject") not in PUBLIC_SUBJECTS})
            if bad:
                raise ValueError(f"source file contains protected/non-public subjects: {filename}: {bad}")
    return manifest, config, config_path


def _numeric_calibration(value: Any, path: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            output.update(_numeric_calibration(item, f"{path}.{key}" if path else str(key)))
    elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        if np.isfinite(float(value)):
            output[path] = float(value)
    return output


def _check_calibration(source: Path, recalculated: Mapping[str, Any]) -> dict[str, float]:
    recorded = json.loads((source / "calibration.json").read_text(encoding="utf-8"))
    expected = _numeric_calibration(recalculated)
    actual = _numeric_calibration(recorded)
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise ValueError(f"source calibration is missing recomputed fields: {missing}")
    mismatches = {
        key: (expected[key], actual[key])
        for key in expected
        if not math.isclose(expected[key], actual[key], rel_tol=1e-10, abs_tol=1e-12)
    }
    if mismatches:
        raise ValueError(f"source calibration mismatch: {mismatches}")
    return actual


def _parameter_rows(source: Path) -> list[dict[str, Any]]:
    rows = []
    for row in _read_csv(source / "subject_stage_parameters.csv"):
        row = dict(row)
        row["estimate"] = float(row["estimate"])
        row["hard_lower"] = float(row["hard_lower"])
        row["hard_upper"] = float(row["hard_upper"])
        row["prior_mean"] = float(row["prior_mean"])
        row["prior_sd"] = float(row["prior_sd"])
        row["is_free"] = row["is_free"].lower() == "true"
        rows.append(row)
    return rows


def _final_parameter_rows(source: Path) -> list[dict[str, Any]]:
    rows = []
    for row in _read_csv(source / "subject_final_parameters.csv"):
        row = dict(row)
        row["estimate"] = float(row["estimate"])
        row["hard_lower"] = float(row["hard_lower"])
        row["hard_upper"] = float(row["hard_upper"])
        rows.append(row)
    return rows


def _stage_vectors(base: BalloonParameters, rows: Sequence[Mapping[str, Any]], stage_id: str) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    fit_rows = [row for row in rows if row["split"] == "fit" and row["stage"] == stage_id and row["fit_scope"] == "eight_trials_fit_two_trials_internal_holdout"]
    if not fit_rows:
        raise ValueError(f"no internal-fit parameter rows for {stage_id}")
    values = _parameter_values(base)
    by_subject: dict[str, dict[str, float]] = defaultdict(dict)
    for row in fit_rows:
        by_subject[str(row["subject"])][str(row["parameter"])] = float(row["estimate"])
    free = sorted({str(row["parameter"]) for row in fit_rows if row["is_free"]})
    for name in free:
        estimates = [mapping[name] for mapping in by_subject.values() if name in mapping and np.isfinite(mapping[name])]
        if not estimates:
            raise ValueError(f"no finite estimates for {stage_id}:{name}")
        values[name] = float(np.median(estimates))
    return values, by_subject


def _raw_feature(item: PreparedTrial | np.ndarray) -> np.ndarray:
    values = _trial_observation(item) if isinstance(item, PreparedTrial) else np.asarray(item, dtype=np.float64)
    pieces: list[float] = []
    for column in range(3):
        signal = values[:, column]
        pieces.extend((float(np.mean(signal)), float(np.std(signal)), float(np.quantile(signal, 0.25)), float(np.quantile(signal, 0.75)), float(np.min(signal)), float(np.max(signal))))
    return np.asarray(pieces, dtype=np.float64)


def _medoid(items: Sequence[PreparedTrial], label: str) -> tuple[PreparedTrial, str]:
    if not items:
        raise ValueError(f"cannot select {label} medoid from an empty set")
    matrix = np.vstack([_raw_feature(item) for item in items])
    center = np.median(matrix, axis=0)
    scale = 1.4826 * np.median(np.abs(matrix - center), axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-9)] = 1.0
    z = (matrix - center) / scale
    distance = np.sqrt(np.sum(z * z, axis=1))
    best = min(range(len(items)), key=lambda i: (float(distance[i]), items[i].trial.subject, int(items[i].trial.event_index)))
    item = items[best]
    return item, f"minimum robust Euclidean distance over per-channel mean/std/Q25/Q75/min/max; distance={distance[best]:.6g}"


def _fit_holdout_items(fit_series: Sequence[PreparedTrial], config: Mapping[str, Any]) -> list[PreparedTrial]:
    fit_subjects = tuple(map(str, config["data"]["conditions"][0]["fit_subjects"]))
    positions = tuple(map(int, config["ssm"]["t3a"]["parameter_fit"]["heldout_trial_positions"]))
    # Reuse the owning splitter to validate trial ordering and held-out support.
    _, heldout = _split_stage_trials(fit_series, fit_subjects, positions)
    event_keys = {(subject, event) for subject in fit_subjects for event, _ in heldout[subject]}
    return [item for item in fit_series if (item.trial.subject, int(item.trial.event_index)) in event_keys]


def _fit_metric_medoid(source: Path, candidates: Sequence[PreparedTrial]) -> tuple[PreparedTrial, str]:
    """Select the internal holdout medoid from recorded model metrics.

    The score vector is deliberately small and fixed: M0/M5 NLL, NRMSE and
    PCC, each averaged over HbO/HbR.  It is a selection description, not a new
    fit and never reads outside the public 01--18 internal holdout rows.
    """
    metric_rows = _read_csv(source / "subject_stage_metrics.csv")
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in metric_rows:
        if row.get("mode") == "center_masked_fnirs" and row.get("target") in {"HbO", "HbR"} and row.get("stage") in {"M0_fixed", "M5_plus_E0_diagnostic"}:
            grouped[(row["subject"], int(row["event_index"]), row["stage"])].append(row)
    features: list[np.ndarray] = []
    usable: list[PreparedTrial] = []
    for item in candidates:
        values: list[float] = []
        valid = True
        for stage in ("M0_fixed", "M5_plus_E0_diagnostic"):
            rows = grouped.get((item.trial.subject, int(item.trial.event_index), stage), [])
            if len(rows) != 2:
                valid = False
                break
            for field in ("gaussian_negative_log_score", "nrmse", "pcc"):
                field_values = [float(row[field]) for row in rows]
                values.append(float(np.nanmean(field_values)))
        if valid and np.all(np.isfinite(values)):
            usable.append(item)
            features.append(np.asarray(values, dtype=np.float64))
    if not usable:
        raise ValueError("no fit holdout rows have complete M0/M5 HbO/HbR metrics")
    matrix = np.vstack(features)
    center = np.median(matrix, axis=0)
    scale = 1.4826 * np.median(np.abs(matrix - center), axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-9)] = 1.0
    distance = np.sqrt(np.sum(((matrix - center) / scale) ** 2, axis=1))
    best = min(range(len(usable)), key=lambda i: (float(distance[i]), usable[i].trial.subject, int(usable[i].trial.event_index)))
    item = usable[best]
    return item, f"minimum robust Euclidean distance over M0/M5 heldout HbO/HbR (NLL, NRMSE, PCC); distance={distance[best]:.6g}"


def _selection(validation: Sequence[PreparedTrial], fit_series: Sequence[PreparedTrial], config: Mapping[str, Any], source: Path) -> tuple[list[dict[str, Any]], dict[str, PreparedTrial]]:
    validation_medoid, validation_rule = _medoid(validation, "validation")
    fit_candidates = _fit_holdout_items(fit_series, config)
    fit_medoid, fit_rule = _fit_metric_medoid(source, fit_candidates)
    selected = {"validation_raw_feature_medoid": validation_medoid, "fit_holdout_metric_medoid": fit_medoid}
    rows = [
        {
            "role": "validation_raw_feature_medoid",
            "split": "validation",
            "subject": validation_medoid.trial.subject,
            "event_index": int(validation_medoid.trial.event_index),
            "criterion": validation_rule,
            "source_run": str(source),
            "data_note": "raw prepared EEG driver plus selected HbO/HbR; no model score used",
        },
        {
            "role": "fit_holdout_metric_medoid",
            "split": "fit_internal_holdout",
            "subject": fit_medoid.trial.subject,
            "event_index": int(fit_medoid.trial.event_index),
            "criterion": fit_rule,
            "source_run": str(source),
            "data_note": "two configured held-out trial positions per fit subject; model-score medoid is post-hoc descriptive; no protected data",
        },
    ]
    return rows, selected


def _center_time(item: PreparedTrial, config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    fs = float(config["analysis"]["sampling_hz"])
    data = config["data"]
    time = np.arange(len(item.hbo), dtype=np.float64) / fs + float(data["window_offset_s"])
    center_cfg = config["analysis"]["center_mask"]
    center = _center_mask(len(item.hbo), fs, float(center_cfg["relative_start_s"]), float(center_cfg["duration_s"]))
    return time, center


def _predict(item: PreparedTrial, parameters: BalloonParameters, spec: Any, balloon_config: Any, mode: str, center: np.ndarray) -> dict[str, Any]:
    values = _mode_input(item, mode, center)
    result = smooth_balloon(values, parameters=parameters, observation_spec=spec, config=balloon_config, observation_mask=np.isfinite(values))
    return {
        "input": values,
        "estimate": np.asarray(result.observation_mean, dtype=np.float64),
        "predictive_std": np.sqrt(np.maximum(np.asarray(result.total_variance, dtype=np.float64), 0.0)),
        "nll": -float(result.predictive_log_likelihood),
    }


def _curve_rows(item: PreparedTrial, time: np.ndarray, center: np.ndarray, curve_id: str, mode: str, target: str, label: str, prediction: Mapping[str, Any], parameter_values: Mapping[str, float], stage: str | None = None, parameter: str | None = None, level: str | None = None) -> list[dict[str, Any]]:
    observed = _trial_observation(item)
    values = np.asarray(prediction["input"], dtype=np.float64)
    estimate = np.asarray(prediction["estimate"], dtype=np.float64)
    predictive_std = np.asarray(prediction["predictive_std"], dtype=np.float64)
    column = TARGET_COLUMNS[target]
    rows: list[dict[str, Any]] = []
    for index, time_s in enumerate(time):
        rows.append({
            "curve_id": curve_id,
            "split": "validation" if item.trial.subject in VALIDATION_SUBJECTS else "fit_internal_holdout",
            "subject": item.trial.subject,
            "event_index": int(item.trial.event_index),
            "mode": mode,
            "target": target,
            "time_s": float(time_s),
            "center_mask": bool(center[index]),
            "observed_standardized": float(observed[index, column]),
            "input_standardized": float(values[index, column]) if np.isfinite(values[index, column]) else None,
            "reconstructed_standardized": float(estimate[index, column]),
            "predictive_sd_standardized": float(predictive_std[index, column]) if np.isfinite(predictive_std[index, column]) else None,
            "stage": stage or "",
            "stage_label": label,
            "parameter": parameter or "",
            "level": level or "",
            "beta": float(parameter_values["beta"]),
            "kappa": float(parameter_values["kappa"]),
            "tau": float(parameter_values["tau"]),
            "gamma": float(parameter_values["gamma"]),
            "alpha": float(parameter_values["alpha"]),
            "E0": float(parameter_values["E0"]),
        })
    return rows


def _shade(axis: Any, time: np.ndarray, center: np.ndarray) -> None:
    where = np.flatnonzero(center)
    if len(where):
        axis.axvspan(float(time[where[0]]), float(time[where[-1]]), color="#E69F00", alpha=0.14, label="中心段：该模态目标被遮挡")


def _plot_medoid_curves(selected: Mapping[str, PreparedTrial], vectors: Mapping[str, Mapping[str, float]], subject_vectors: Mapping[str, Mapping[str, Mapping[str, float]]], base: BalloonParameters, spec: Any, balloon_config: Any, config: Mapping[str, Any], output: Path, dpi: int, curve_rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, squeeze=False, layout="constrained")
    for row, (role, item) in enumerate(selected.items()):
        time, center = _center_time(item, config)
        for column, target in enumerate(OBS_NAMES):
            mode = "center_masked_eeg" if target == "EEG" else "center_masked_fnirs"
            axis = axes[row, column]
            observed = _trial_observation(item)[:, TARGET_COLUMNS[target]]
            axis.plot(time, observed, color="#111111", linewidth=1.0, label="实测观测（非干净真值）")
            for stage in STAGE_IDS:
                parameter_values = vectors[stage]
                if item.trial.subject in FIT_SUBJECTS:
                    parameter_values = subject_vectors.get(stage, {}).get(item.trial.subject, parameter_values)
                parameters = _replace_parameter_values(base, parameter_values)
                prediction = _predict(item, parameters, spec, balloon_config, mode, center)
                curve_rows.extend(_curve_rows(item, time, center, f"medoid_{row}_{stage}_{target}", mode, target, STAGE_LABELS[stage], prediction, parameter_values, stage=stage))
                axis.plot(time, prediction["estimate"][:, TARGET_COLUMNS[target]], color=STAGE_COLORS[stage], linestyle=STAGE_STYLES[stage], linewidth=1.3, label=STAGE_LABELS[stage])
            _shade(axis, time, center)
            axis.grid(alpha=0.2)
            if row == 0:
                axis.set_title(target + "（标准化坐标）")
            if column == 0:
                role_label = "冻结验证 raw medoid" if role == "validation_raw_feature_medoid" else "内部留出 metric medoid"
                axis.set_ylabel(f"{role_label}\n{item.trial.subject} 事件 {int(item.trial.event_index)}")
            if row == 1:
                axis.set_xlabel("事件相对时间（秒）")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=7)
    fig.suptitle("实测 medoid：M0、M1 κ、M5 在中心遮挡下的 T3a 曲线")
    save_figure_atomic(fig, output / "figures/01_medoid_validation_fit_holdout", formats=("png",), dpi=dpi)
    plt.close(fig)


def _plot_oat(item: PreparedTrial, vectors: Mapping[str, float], base: BalloonParameters, spec: Any, balloon_config: Any, config: Mapping[str, Any], output: Path, dpi: int, curve_rows: list[dict[str, Any]], parameter_specs: Mapping[str, Any]) -> None:
    fig, axes = plt.subplots(3, 6, figsize=(18, 8.5), sharex=True, squeeze=False, layout="constrained")
    time, center = _center_time(item, config)
    observed = _trial_observation(item)
    for parameter_index, name in enumerate(PARAMETER_NAMES):
        p_spec = parameter_specs[name]
        low, high = (float(value) for value in p_spec["bounds"])
        prior = float(p_spec["prior_mean"])
        for row, target in enumerate(OBS_NAMES):
            mode = "center_masked_eeg" if target == "EEG" else "center_masked_fnirs"
            axis = axes[row, parameter_index]
            axis.plot(time, observed[:, row], color="#111111", linewidth=0.8, label="实测")
            for level, value in (("low", low), ("prior", prior), ("high", high)):
                conditional = dict(vectors)
                conditional[name] = value
                parameters = _replace_parameter_values(base, conditional)
                prediction = _predict(item, parameters, spec, balloon_config, mode, center)
                curve_rows.extend(_curve_rows(item, time, center, f"oat_{name}_{level}_{target}", mode, target, f"条件 OAT {name}={value:g}", prediction, conditional, parameter=name, level=level))
                axis.plot(time, prediction["estimate"][:, row], color=OAT_COLORS[level], linestyle=OAT_STYLES[level], linewidth=1.0, label=f"{OAT_LABELS[level]}: {value:g}")
            _shade(axis, time, center)
            axis.grid(alpha=0.2)
            if row == 0:
                axis.set_title(f"{PARAMETER_LABELS[name]}\n{low:g} / {prior:g} / {high:g}")
            if parameter_index == 0:
                axis.set_ylabel(target + "\n标准化坐标")
            if row == 2:
                axis.set_xlabel("秒")
    handles = [mpl.lines.Line2D([], [], color="#111111", linewidth=0.8, label="实测"), *[mpl.lines.Line2D([], [], color=OAT_COLORS[level], linestyle=OAT_STYLES[level], label=OAT_LABELS[level]) for level in ("low", "prior", "high")]]
    axes[0, 0].legend(handles=handles, frameon=False, ncol=4, fontsize=7)
    fig.suptitle("validation raw-feature medoid 的六参数条件 OAT 敏感度（其余参数固定为 M0；仅表示局部条件响应）")
    save_figure_atomic(fig, output / "figures/02_oat_sensitivity_validation_medoid", formats=("png",), dpi=dpi)
    plt.close(fig)


def _metric_tradeoff(source: Path, parameter_rows: Sequence[Mapping[str, Any]], stage_ids: Sequence[str]) -> list[dict[str, Any]]:
    metrics = _read_csv(source / "subject_stage_metrics.csv")
    selected = [row for row in metrics if row.get("mode") == "center_masked_fnirs" and row.get("target") in {"HbO", "HbR"}]
    label_by_stage = {str(row["stage"]): str(row.get("stage_label_zh", row["stage"])) for row in selected}
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        grouped[(row["subject"], row["stage"])].append(row)
    baseline: dict[str, tuple[float, float]] = {}
    for subject in FIT_SUBJECTS:
        rows = grouped.get((subject, "M0_fixed"), [])
        if rows:
            baseline[subject] = (float(np.mean([float(r["gaussian_negative_log_score"]) for r in rows])), float(np.mean([float(r["nrmse"]) for r in rows])))
    boundary: dict[str, float] = {}
    for stage in stage_ids:
        stage_rows = [r for r in parameter_rows if r["split"] == "fit" and r["fit_scope"] == "eight_trials_fit_two_trials_internal_holdout" and r["stage"] == stage and r["is_free"]]
        boundary[stage] = float(np.mean([str(r["boundary_status"]) == "BOUNDARY" for r in stage_rows])) if stage_rows else 0.0
    geometry = _read_csv(source / "subject_stage_geometry.csv")
    nonid: dict[str, float] = {}
    for stage in stage_ids:
        rows = [r for r in geometry if r["split"] == "fit" and r["stage"] == stage]
        nonid[stage] = float(np.mean([str(r["stage_status"]) not in {"FIXED", "IDENTIFIABLE"} for r in rows])) if rows else 0.0
    output: list[dict[str, Any]] = []
    for stage in stage_ids:
        for subject in FIT_SUBJECTS:
            rows = grouped.get((subject, stage), [])
            if not rows or subject not in baseline:
                continue
            nll = float(np.mean([float(r["gaussian_negative_log_score"]) for r in rows]))
            nrmse = float(np.mean([float(r["nrmse"]) for r in rows]))
            output.append({"stage": stage, "stage_label": label_by_stage.get(stage, stage), "subject": subject, "heldout_nll": nll, "heldout_nrmse": nrmse, "delta_nll_vs_M0": nll - baseline[subject][0], "delta_nrmse_vs_M0": nrmse - baseline[subject][1], "free_parameter_boundary_fraction": boundary[stage], "nonidentifiable_subject_fraction": nonid[stage]})
    return output


def _plot_tradeoff(rows: Sequence[Mapping[str, Any]], stage_ids: Sequence[str], output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    color_by_stage = {stage: TRADEOFF_COLORS[index % len(TRADEOFF_COLORS)] for index, stage in enumerate(stage_ids)}
    marker_by_stage = {stage: TRADEOFF_MARKERS[index % len(TRADEOFF_MARKERS)] for index, stage in enumerate(stage_ids)}
    label_by_stage = {stage: next((str(row["stage_label"]) for row in rows if row["stage"] == stage), stage) for stage in stage_ids}
    for stage in stage_ids:
        subset = [row for row in rows if row["stage"] == stage]
        axes[0].scatter([float(row["delta_nll_vs_M0"]) for row in subset], [float(row["delta_nrmse_vs_M0"]) for row in subset], color=color_by_stage[stage], marker=marker_by_stage[stage], label=label_by_stage[stage], s=24, alpha=0.8)
    axes[0].axhline(0, color="#999999", linewidth=0.7); axes[0].axvline(0, color="#999999", linewidth=0.7)
    axes[0].set_xlabel("留出 fNIRS NLL 相对 M0（负值=改善）"); axes[0].set_ylabel("留出 fNIRS NRMSE 相对 M0（负值=改善）")
    axes[0].grid(alpha=0.2); axes[0].legend(frameon=False, fontsize=7)
    stage_values = []
    for stage in stage_ids:
        subset = [row for row in rows if row["stage"] == stage]
        if subset:
            stage_values.append((stage, float(subset[0]["free_parameter_boundary_fraction"]), float(subset[0]["nonidentifiable_subject_fraction"])))
    x = np.arange(len(stage_values)); width = 0.36
    axes[1].bar(x - width / 2, [v[1] for v in stage_values], width, color="#D55E00", label="自由参数命中边界")
    axes[1].bar(x + width / 2, [v[2] for v in stage_values], width, color="#0072B2", label="被试阶段不可辨识")
    axes[1].set_xticks(x, [label_by_stage[v[0]] for v in stage_values], rotation=28, ha="right"); axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("比例"); axes[1].grid(axis="y", alpha=0.2); axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle("逐级释放的留出收益与边界/不可辨识代价（01–18 内部留出）")
    save_figure_atomic(fig, output / "figures/03_stage_tradeoff", formats=("png",), dpi=dpi)
    plt.close(fig)


def _plot_m5_bounds(source: Path, output: Path, dpi: int) -> list[dict[str, Any]]:
    rows = [row for row in _final_parameter_rows(source) if row["stage"] == "M5_plus_E0_diagnostic"]
    figure_rows: list[dict[str, Any]] = []
    fig, axis = plt.subplots(figsize=(11, 5.3), layout="constrained")
    rng = np.random.default_rng(20260902)
    for index, name in enumerate(PARAMETER_NAMES):
        subset = [row for row in rows if row["parameter"] == name]
        for split, marker, color in (("fit", "o", "#0072B2"), ("validation", "s", "#D55E00")):
            values = [row for row in subset if row["split"] == split]
            jitter = rng.uniform(-0.11, 0.11, len(values))
            for offset, row in zip(jitter, values):
                normalized = (float(row["estimate"]) - float(row["hard_lower"])) / (float(row["hard_upper"]) - float(row["hard_lower"]))
                axis.scatter(index + offset, normalized, color=color, marker=marker, s=28, alpha=0.8)
                figure_rows.append({"stage": row["stage"], "split": split, "subject": row["subject"], "parameter": name, "estimate": float(row["estimate"]), "lower": float(row["hard_lower"]), "upper": float(row["hard_upper"]), "normalized_position": normalized, "boundary_status": row.get("boundary_status", "")})
        if subset:
            prior_position = (float(subset[0]["prior_mean"]) - float(subset[0]["hard_lower"])) / (float(subset[0]["hard_upper"]) - float(subset[0]["hard_lower"]))
            axis.scatter(index, prior_position, color="#111111", marker="D", s=34, zorder=4)
    axis.axhspan(-0.02, 0.02, color="#D55E00", alpha=0.08); axis.axhspan(0.98, 1.02, color="#D55E00", alpha=0.08)
    axis.axhline(0, color="#D55E00", linewidth=0.8, linestyle="--"); axis.axhline(1, color="#D55E00", linewidth=0.8, linestyle="--")
    axis.set_xticks(range(len(PARAMETER_NAMES)), [PARAMETER_LABELS[name] for name in PARAMETER_NAMES], rotation=18, ha="right"); axis.set_ylim(-0.08, 1.08); axis.set_ylabel("归一化参数位置：0=下界，1=上界"); axis.grid(axis="y", alpha=0.2)
    axis.legend(handles=[mpl.lines.Line2D([], [], color="#0072B2", marker="o", linestyle="None", label="01–18 fit"), mpl.lines.Line2D([], [], color="#D55E00", marker="s", linestyle="None", label="19–23 validation 后置描述"), mpl.lines.Line2D([], [], color="#111111", marker="D", linestyle="None", label="先验均值")], frameon=False)
    fig.suptitle("M5 六参数边界占用：后置描述性拟合（标准化观测坐标；不代表真实生理量）")
    save_figure_atomic(fig, output / "figures/04_m5_parameter_bounds", formats=("png",), dpi=dpi)
    plt.close(fig)
    return figure_rows


def _kappa_profiles(fit_series: Sequence[PreparedTrial], subject_rows: Mapping[str, Mapping[str, float]], base: BalloonParameters, spec: Any, balloon_config: Any, config: Mapping[str, Any], output: Path, dpi: int) -> list[dict[str, Any]]:
    p_spec = config["ssm"]["t3a"]["parameter_fit"]["parameters"]["kappa"]
    lower, upper = (float(value) for value in p_spec["bounds"])
    prior, prior_sd = float(p_spec["prior_mean"]), float(p_spec["prior_sd"])
    grid = np.linspace(lower, upper, 21)
    targets = {"subject_05": "下界", "subject_14": "内部最小", "subject_06": "上界"}
    fit_subjects = tuple(map(str, config["data"]["conditions"][0]["fit_subjects"]))
    heldout_positions = tuple(map(int, config["ssm"]["t3a"]["parameter_fit"]["heldout_trial_positions"]))
    train_observations, _ = _split_stage_trials(fit_series, fit_subjects, heldout_positions)
    by_subject: dict[str, list[np.ndarray]] = {subject: [observation for _, observation in train_observations[subject]] for subject in fit_subjects}
    rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True, layout="constrained")
    for axis, subject in zip(axes, targets):
        likelihood: list[float] = []
        prior_total: list[float] = []
        for kappa in grid:
            values = dict(_parameter_values(base)); values.update(subject_rows.get(subject, {})); values["kappa"] = float(kappa)
            parameters = _replace_parameter_values(base, values)
            nll = 0.0
            for vals in by_subject[subject]:
                result = smooth_balloon(vals, parameters=parameters, observation_spec=spec, config=balloon_config, observation_mask=np.isfinite(vals))
                nll += -float(result.predictive_log_likelihood)
            likelihood.append(nll)
            prior_total.append(nll + 0.5 * ((float(kappa) - prior) / prior_sd) ** 2)
        likelihood = np.asarray(likelihood); prior_total = np.asarray(prior_total)
        likelihood -= float(np.min(likelihood)); prior_total -= float(np.min(prior_total))
        estimate = float(subject_rows.get(subject, {}).get("kappa", prior))
        axis.plot(grid, likelihood, color="#0072B2", linestyle="-", linewidth=1.3, label="仅 likelihood")
        axis.plot(grid, prior_total, color="#D55E00", linestyle=(0, (5, 2)), linewidth=1.3, label="+ 一次先验罚项")
        axis.axvline(estimate, color="#222222", linestyle=(0, (2, 2)), linewidth=1.0, label=f"拟合 κ={estimate:.3g}")
        axis.axvline(prior, color="#777777", linestyle=":", linewidth=0.9, label="先验均值")
        axis.set_title(f"{subject}: {targets[subject]}"); axis.set_xlabel("κ（每秒）"); axis.grid(alpha=0.2)
        for kappa, nll, penalized in zip(grid, likelihood, prior_total):
            rows.append({"subject": subject, "status": targets[subject], "kappa": float(kappa), "likelihood_nll_relative_min": float(nll), "nll_plus_prior_relative_min": float(penalized), "fitted_kappa": estimate, "prior_mean": prior, "prior_sd": prior_sd, "fit_trial_count": len(by_subject[subject])})
        print(f"完成 κ conditional profile: {subject}", flush=True)
    axes[0].set_ylabel("相对最小值的 NLL（8 个 fit trial）")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("κ 条件剖面：边界命中并不等于确定的生理极值")
    save_figure_atomic(fig, output / "figures/05_kappa_conditional_profiles", formats=("png",), dpi=dpi)
    plt.close(fig)
    return rows


def _summary(source: Path, selection: Sequence[Mapping[str, Any]], tradeoff: Sequence[Mapping[str, Any]], bounds: Sequence[Mapping[str, Any]], profiles: Sequence[Mapping[str, Any]], parameter_rows: Sequence[Mapping[str, Any]], calibration: Mapping[str, float]) -> str:
    boundary_lines = []
    for name in PARAMETER_NAMES:
        subset = [row for row in bounds if row["parameter"] == name]
        rate = float(np.mean([float(row["normalized_position"]) <= 0.01 or float(row["normalized_position"]) >= 0.99 for row in subset])) if subset else float("nan")
        boundary_lines.append(f"- {name}: M5 归一化位置命中边界率 {rate:.1%}（01–23 后置描述性拟合）")
    trade_lines = []
    stage_order = list(dict.fromkeys(str(row["stage"]) for row in tradeoff))
    for stage in stage_order:
        subset = [row for row in tradeoff if row["stage"] == stage]
        if subset:
            trade_lines.append(f"- {stage}: 留出 fNIRS ΔNLL 中位数 {np.median([float(r['delta_nll_vs_M0']) for r in subset]):.3g}；ΔNRMSE 中位数 {np.median([float(r['delta_nrmse_vs_M0']) for r in subset]):.3g}；自由参数边界率 {float(subset[0]['free_parameter_boundary_fraction']):.1%}；阶段不可辨识率 {float(subset[0]['nonidentifiable_subject_fraction']):.1%}")
    staged_m5 = [row for row in parameter_rows if row["split"] == "fit" and row["fit_scope"] == "eight_trials_fit_two_trials_internal_holdout" and row["stage"] == "M5_plus_E0_diagnostic" and row["is_free"]]
    staged_boundary = [row for row in staged_m5 if row["boundary_status"] == "BOUNDARY"]
    staged_boundary_lines = []
    for name in PARAMETER_NAMES:
        subset = [row for row in staged_m5 if row["parameter"] == name]
        lower = sum(math.isclose(float(row["estimate"]), float(row["hard_lower"]), rel_tol=1e-8, abs_tol=1e-10) for row in subset)
        upper = sum(math.isclose(float(row["estimate"]), float(row["hard_upper"]), rel_tol=1e-8, abs_tol=1e-10) for row in subset)
        staged_boundary_lines.append(f"- {name}: {sum(row['boundary_status'] == 'BOUNDARY' for row in subset)}/{len(subset)} 触边（下界 {lower}，上界 {upper}）")
    profile_lines = []
    for subject in ("subject_05", "subject_14", "subject_06"):
        subset = [row for row in profiles if row["subject"] == subject]
        if subset:
            likelihood_min = min(subset, key=lambda row: float(row["likelihood_nll_relative_min"]))
            posterior_min = min(subset, key=lambda row: float(row["nll_plus_prior_relative_min"]))
            profile_lines.append(f"- {subject}: likelihood-only 网格最小 κ={float(likelihood_min['kappa']):.3g}/s；加一次先验后 κ={float(posterior_min['kappa']):.3g}/s；原拟合 κ={float(subset[0]['fitted_kappa']):.3g}/s")
    geometry = [row for row in _read_csv(source / "subject_stage_geometry.csv") if row.get("split") == "fit" and row.get("stage") == "M5_plus_E0_diagnostic"]
    return "\n".join([
        "# T3a 实测参数影响可视化（后处理）",
        "",
        f"来源 run：`{source}`；本次仅读取公开 01–23，24–29 未开放。",
        "",
        "观测坐标是 fit-cohort gauge 下的 EEG 10 Hz 功率代理、HbO、HbR 标准化坐标；曲线阴影为中心段遮挡，实测线不是干净生理真值。",
        "",
        "## 窗口选择",
        "",
        *[f"- {row['role']}: {row['subject']} event {row['event_index']}；{row['criterion']}" for row in selection],
        "",
        "## 逐级释放与边界代价",
        "",
        *trade_lines,
        "",
        "## M5 边界占用",
        "",
        f"01–18 的 8-trial M5 stage fit：{len(staged_boundary)}/{len(staged_m5)} 个自由参数行触边；{sum(row.get('stage_status') == 'BOUNDARY' for row in geometry)}/{len(geometry)} 个被试阶段标为 BOUNDARY。优化成功不等于参数可辨识。",
        "",
        *staged_boundary_lines,
        "",
        "01–23 全 trial 后置描述性 refit：",
        "",
        *boundary_lines,
        "",
        "## κ 条件目标剖面",
        "",
        *profile_lines,
        "",
        "## 固定 gauge 的尺度",
        "",
        f"- EEG driver SD={calibration['fit_driver_sd']:.4g}，模型 driver target SD={calibration['driver_target_sd']:.4g}，固定 EEG loading={calibration['eeg_loading']:.4g}。",
        f"- fNIRS pooled SD={calibration['fnirs_pooled_std']:.4g}，固定 P0={calibration['t3a_P0_gauge']:.4g}、Q0={calibration['t3a_Q0_gauge']:.4g}；观测噪声尺度 EEG/HbO/HbR={calibration['observation_scale_EEG']:.4g}/{calibration['observation_scale_HbO']:.4g}/{calibration['observation_scale_HbR']:.4g}。",
        "",
        "解释：六状态中每时刻只有 r、p、q 直接进入三条观测；β/κ/γ 共用 s→f 途径，τ/α 共同改变血容量响应，E0 只通过 q/HbR 进入。20 秒单条件、短中心遮挡、固定 P0/Q0/EEG loading 与 fit-cohort 标准化使这些方向高度相关。平滑器还可用潜状态和过程噪声吸收误差；先验每被试只计一次，而似然累积约数千个观测点，所以硬边界成为有效正则化，优化在平坦/斜谷上停在角点。M5 的 fNIRS 分数改善因此可能只是参数补偿，不能解释为真实个体血管或氧代谢极值。κ 剖面图把这一点显示为 likelihood-only 与加先验曲线的条件剖面。",
        "",
        "限制：内部 held-out trial 与其余 fit trial 共享用 01–18 全 fit cohort 估计的通道选择、EEG adapter 和 calibration，因此不是完全 cross-fit；这里的分数只作探索性比较。",
        "",
        "## 输出约定",
        "",
        "PNG only；长表 CSV 保留 observed/input/reconstructed/predictive SD、mask、stage 与参数值。所有写入先到临时文件再原子替换；旧 run 不覆盖。",
        "",
    ])


def render(source: Path, output: Path, dpi: int) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    if source == output or source in output.parents:
        raise ValueError("output directory must not be inside the immutable source run")
    _, config, _ = _require_source(source)
    _, fit_series, validation, _, _, _, _ = _prepare_measured_series(config)
    bundle, recalculated = _fit_models(fit_series, config, fit_comparison_models=False)
    calibration = _check_calibration(source, recalculated)
    output.mkdir(parents=True)
    (output / "figures").mkdir()
    with mpl.rc_context(CJK_STYLE):
        base, observation_spec, balloon_config = bundle.t3a
        rows = _parameter_rows(source)
        vectors: dict[str, dict[str, float]] = {}
        subject_vectors: dict[str, dict[str, dict[str, float]]] = {}
        for stage in STAGE_IDS:
            vectors[stage], subject_vectors[stage] = _stage_vectors(base, rows, stage)
        selection_rows, selected = _selection(validation, fit_series, config, source)
        curve_rows: list[dict[str, Any]] = []
        _plot_medoid_curves({"validation_raw_feature_medoid": selected["validation_raw_feature_medoid"], "fit_holdout_metric_medoid": selected["fit_holdout_metric_medoid"]}, vectors, subject_vectors, base, observation_spec, balloon_config, config, output, dpi, curve_rows)
        _plot_oat(selected["validation_raw_feature_medoid"], _parameter_values(base), base, observation_spec, balloon_config, config, output, dpi, curve_rows, config["ssm"]["t3a"]["parameter_fit"]["parameters"])
        all_stage_ids = tuple(str(stage["id"]) for stage in config["ssm"]["t3a"]["parameter_fit"]["stages"])
        tradeoff = _metric_tradeoff(source, rows, all_stage_ids)
        _plot_tradeoff(tradeoff, all_stage_ids, output, dpi)
        bounds = _plot_m5_bounds(source, output, dpi)
        profiles = _kappa_profiles(fit_series, subject_vectors["M1_kappa"], base, observation_spec, balloon_config, config, output, dpi)
    _atomic_csv(output / "selected_curves.csv", curve_rows)
    _atomic_csv(output / "stage_tradeoff.csv", tradeoff)
    _atomic_csv(output / "m5_parameter_positions.csv", bounds)
    _atomic_csv(output / "kappa_profiles.csv", profiles)
    _atomic_csv(output / "selection.csv", selection_rows)
    _atomic_text(output / "summary.md", _summary(source, selection_rows, tradeoff, bounds, profiles, rows, calibration))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dpi", default=180, type=int)
    args = parser.parse_args(argv)
    if int(args.dpi) <= 0:
        parser.error("--dpi must be positive")
    render(args.source_run.resolve(), args.output_dir.resolve(), int(args.dpi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
