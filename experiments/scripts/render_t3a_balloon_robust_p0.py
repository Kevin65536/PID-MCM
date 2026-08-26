#!/usr/bin/env python3
"""Render the synthetic-only T3a Balloon P0 diagnostic figures.

The renderer is deliberately post-hoc: it consumes the P0 CSV/JSON exports,
never re-fits a model, and writes a small fixed set of Chinese-labelled PNG
figures.  The files are treated as long tables where possible, while the
trajectory/state columns used by the P0 runner are also accepted directly.

The optional spatial panel is only produced when finite channel geometry is
present in an input table.  Sensor-coordinate associations are not a brain
activation map and are labelled accordingly.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

try:
    from scipy.stats import t as student_t
except Exception:  # pragma: no cover - scipy is a project dependency
    student_t = None

try:
    from src.visualization.token_physiology_plots import save_figure_atomic
except ModuleNotFoundError:  # direct invocation from outside the repository
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.visualization.token_physiology_plots import save_figure_atomic


MODEL_LABELS = {
    "T3a-balloon-robust": "T3a Balloon-robust",
    "T0-native": "T0 持续性基线",
    "T1-self": "T1 独立 LDS",
    "T2b-adaptive-legacy": "T2b 自适应旧基线",
    "t3a_balloon_robust": "T3a Balloon-robust",
    "t3a": "T3a Balloon-robust",
    "joint": "联合观测",
    "eeg_only": "仅 EEG",
    "fnirs_only": "仅 fNIRS",
}
MODEL_COLORS = {
    "T3a-balloon-robust": "#D55E00",
    "T0-native": "#777777",
    "T1-self": "#0072B2",
    "T2b-adaptive-legacy": "#CC79A7",
    "t3a_balloon_robust": "#D55E00",
    "t3a": "#D55E00",
    "joint": "#0072B2",
    "eeg_only": "#009E73",
    "fnirs_only": "#CC79A7",
}
NULL_LABELS = {
    "independent": "独立驱动",
    "pairing": "错配配对",
    "time_shift": "时间平移",
}
STATE_SPECS = (
    ("r", "共享神经驱动 r(t)", "s^-2", 0.0),
    ("s", "血管舒张信号 s", "s^-1", 0.0),
    ("f", "归一化流入 f", "无量纲", 1.0),
    ("v", "归一化血容量 v", "无量纲", 1.0),
    ("p", "归一化总血红蛋白 p", "无量纲", 1.0),
    ("q", "归一化脱氧血红蛋白 q", "无量纲", 1.0),
)
SIGNAL_SPECS = (
    ("eeg", "EEG 观测代理", "EEG"),
    ("hbo", "HbO", "HbO"),
    ("hbr", "HbR", "HbR"),
)
SIGNAL_COLORS = {"eeg": "#0072B2", "hbo": "#D55E00", "hbr": "#009E73"}
OBS_COLOR = "#333333"
CORRUPTED_COLOR = "#888888"
MISSING_COLOR = "#777777"
GRID_COLOR = "#D9D9D9"
CJK_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Sans CJK TC",
        "AR PL UMing CN",
        "DejaVu Sans",
    ],
    "axes.unicode_minus": True,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}

MANDATORY_STEMS = (
    "观测真值后验_轨迹",
    "生理状态真值后验_r_s_f_v_p_q",
    "噪声创新_伪影分离",
    "不确定性分解_观测状态",
    "校准_PIT_覆盖率",
    "参数恢复_多起点_profile",
    "null_严重度性能",
)

# ``save_figure_atomic`` writes this text alongside each PNG as a small,
# machine-readable alt-text sidecar.  Keep the descriptions factual: they
# state what is plotted and the relevant uncertainty semantics, without
# implying source-localized brain activity.
ALT_TEXTS = {
    "观测真值后验_轨迹": "三类观测的污染输入、合成干净真值、后验均值及 Student-t 95% 后验预测带；黄色区域标出注入伪影掩码。",
    "生理状态真值后验_r_s_f_v_p_q": "T3a 的共享神经驱动 r 与血管状态 s、f、v、p、q 的合成真值、后验均值和条件高斯区间。",
    "噪声创新_伪影分离": "观测与后验残差创新、注入伪影和可选干扰状态的时间序列；创新不自动等同于已分离噪声。",
    "不确定性分解_观测状态": "EEG/HbO/HbR 观测和 r/s/f/v/p/q 状态的偶然、条件状态后验（epistemic*）与总方差时间序列。",
    "校准_PIT_覆盖率": "Student-t 预测的 PIT、区间经验覆盖率、CRPS/NLL 等 proper score 及不确定性风险指标。",
    "参数恢复_多起点_profile": "参数真值、估计误差、边界接触状态和固定另一参数的目标函数切片；边界主导参数不作生理解释。",
    "null_严重度性能": "受控伪影严重度下的性能曲线、重复实验四分位距与实际导出的零假设对照分布。",
    "通道空间关联_边界": "带真实 channel_id、几何坐标和关联值的通道级散点；该图不是全脑活动热图或源定位。",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"缺少 T3a P0 输出：{path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"T3a P0 输出为空：{path}")
    return rows


def _read_csv_optional(path: Path) -> list[dict[str, str]]:
    """Read an auxiliary table when present; absent profile data stays NA."""

    return _read_csv(path) if path.exists() else []


def _read_gates(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"缺少 T3a P0 门状态：{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("gates.json 必须是 JSON 对象")
    return value


def _float(row: Mapping[str, Any], *keys: str, default: float = np.nan) -> float:
    for key in keys:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(result):
            return result
    return default


def _text(row: Mapping[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def _bool(row: Mapping[str, Any], *keys: str, default: bool = True) -> bool:
    for key in keys:
        value = row.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}
    return default


def _id_tuple(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        _text(row, "replicate_id", "replicate", "subject", default="replicate_00"),
        _text(row, "scenario_id", "scenario", "condition_id", default="scenario_00"),
        _text(row, "model_id", "model", default="t3a_balloon_robust"),
    )


def _time_key(value: float) -> float:
    return round(float(value), 9)


def _group_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_id_tuple(row)].append(dict(row))
    for values in groups.values():
        values.sort(key=lambda row: _float(row, "time_s", default=0.0))
    return dict(groups)


def _preferred_sample(
    groups: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    *,
    prefer_corruption: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Select one replicate/scenario pair and retain every available model."""

    if not groups:
        return {}
    pairs = sorted({(key[0], key[1]) for key in groups})
    corrupted = [
        pair for pair in pairs
        if pair[1] not in {"clean", "missing_eeg", "missing_fnirs"}
        and not pair[1].startswith("null_")
    ]
    candidates = corrupted if prefer_corruption and corrupted else pairs

    def support(pair: tuple[str, str]) -> int:
        target_rows = [
            row
            for key, values in groups.items()
            if key[:2] == pair and _model_sort_key(key[2])[0] == 0
            for row in values
        ]
        if not target_rows:
            target_rows = [row for key, values in groups.items() if key[:2] == pair for row in values]
        fields = (
            "posterior_mean",
            "trajectory_mean",
            "state_mean",
            "total_variance",
            "epistemic_variance",
            "value",
        )
        return sum(any(np.isfinite(_float(row, field)) for field in fields) for row in target_rows)

    pair = max(candidates, key=support)
    selected = {
        key[2]: [dict(row) for row in values]
        for key, values in sorted(groups.items())
        if key[:2] == pair
    }
    return selected


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " "))


def _model_color(model: str) -> str:
    return MODEL_COLORS.get(model, "#CC79A7")


def _model_sort_key(model: str) -> tuple[int, str]:
    """Place the T3a candidate first while retaining deterministic ordering."""

    return (0 if model in {"T3a-balloon-robust", "t3a_balloon_robust", "t3a"} else 1, model)


def _component_label(component: str) -> str:
    return {
        "eeg": "EEG",
        "hbo": "HbO",
        "hbr": "HbR",
        "all": "全部观测",
        "r": "共享驱动 r",
    }.get(component, component)


def _normalise_component(value: str) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "eeg_proxy": "eeg",
        "eeg_observation": "eeg",
        "hbo_state": "hbo",
        "hbr_state": "hbr",
        "hb_o": "hbo",
        "hb_r": "hbr",
        "shared_driver": "r",
        "driver": "r",
        "vasoactive": "s",
    }
    return aliases.get(text, text.removeprefix("state_").removesuffix("_state"))


def _series(row: Mapping[str, Any], *keys: str) -> float:
    return _float(row, *keys)


def _array(rows: Sequence[Mapping[str, Any]], *keys: str) -> np.ndarray:
    return np.asarray([_series(row, *keys) for row in rows], dtype=float)


def _valid_array(
    rows: Sequence[Mapping[str, Any]],
    *keys: str,
    default: bool = True,
) -> np.ndarray:
    return np.asarray([_bool(row, *keys, default=default) for row in rows], dtype=bool)


def _artifact_mask(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            _bool(row, "artifact_mask", "corruption_mask", "mask", default=False)
            for row in rows
        ],
        dtype=bool,
    )


def _style_axes(axes: Iterable[Any]) -> None:
    for axis in axes:
        axis.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)


def _legend_if_handles(axis: Any, **kwargs: Any) -> None:
    """Avoid an empty-legend warning when a model has no supported rows."""

    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(**kwargs)


def _save(fig: Any, output_dir: Path, stem: str) -> Path:
    artifacts = save_figure_atomic(
        fig,
        output_dir / stem,
        formats="png",
        dpi=300,
        alt_text=ALT_TEXTS.get(stem, stem),
        write_manifest=False,
    )
    plt.close(fig)
    return artifacts.figure_paths[0]


def _shade_mask(axis: Any, time: np.ndarray, mask: np.ndarray) -> None:
    mask = np.asarray(mask, dtype=bool)
    if len(time) != len(mask) or not np.any(mask):
        return
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    stops = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    for start, stop in zip(starts, stops, strict=True):
        axis.axvspan(time[start], time[stop], color="#F2C14E", alpha=0.16, linewidth=0)


def _student_dof(rows: Sequence[Mapping[str, Any]], gates: Mapping[str, Any]) -> float:
    candidates: list[Any] = []
    for row in rows:
        candidates.extend(row.get(key) for key in ("student_t_dof", "student_t_df", "dof", "nu"))
    candidates.extend(gates.get(key) for key in ("student_t_dof", "student_t_df", "dof", "nu"))
    for value in candidates:
        try:
            dof = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(dof) and dof > 0:
            return dof
    return float("nan")


def _student_critical(dof: float) -> float:
    if np.isfinite(dof) and student_t is not None and dof > 2.0:
        try:
            # Exported variance is the Student-t variance, not its scale².
            return float(student_t.ppf(0.975, dof) * np.sqrt((dof - 2.0) / dof))
        except Exception:
            pass
    return 1.96


def _uncertainty_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[tuple[str, str, str], float, str], dict[str, Any]]:
    lookup: dict[tuple[tuple[str, str, str], float, str], dict[str, Any]] = {}
    for row in rows:
        component = _normalise_component(_text(row, "component", "target", "state_name"))
        key = (_id_tuple(row), _time_key(_float(row, "time_s")), component)
        lookup[key] = dict(row)
    return lookup


def _uncertainty_value(
    lookup: Mapping[tuple[tuple[str, str, str], float, str], Mapping[str, Any]],
    identity: tuple[str, str, str],
    time: float,
    component: str,
    kind: str,
) -> float:
    row = lookup.get((identity, _time_key(time), _normalise_component(component)))
    if row is None:
        return float("nan")
    direct = _float(row, f"{kind}_variance")
    if np.isfinite(direct):
        return direct
    standard = _float(row, f"{kind}_std")
    return float(standard * standard) if np.isfinite(standard) else float("nan")


def _wide_trajectory_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pivot the P0 long trajectory table into one row per time point."""

    if not rows:
        return []
    has_coordinate = any(
        _text(row, "coordinate", "signal", "target", default="")
        and ("clean_truth" in row or "corrupted_observation" in row or "posterior_mean" in row)
        for row in rows
    )
    if not has_coordinate:
        return [dict(row) for row in rows]
    grouped: dict[tuple[tuple[str, str, str], float], dict[str, Any]] = {}
    for row in rows:
        identity = _id_tuple(row)
        time = _float(row, "time_s")
        key = (identity, _time_key(time))
        current = grouped.setdefault(key, {
            "replicate_id": identity[0],
            "scenario_id": identity[1],
            "model_id": identity[2],
            "time_s": time,
        })
        component = _normalise_component(_text(row, "coordinate", "signal", "target"))
        prefix = {"eeg": "eeg", "hbo": "hbo", "hbr": "hbr"}.get(component)
        if prefix is None:
            continue
        current[f"{prefix}_clean"] = _float(row, "clean_truth", "truth")
        current[f"{prefix}_obs"] = _float(row, "corrupted_observation", "observation", "observed")
        current[f"{prefix}_mean"] = _float(row, "posterior_mean", "trajectory_mean", "mean")
        current[f"artifact_{prefix}"] = _float(row, "artifact", f"artifact_{prefix}")
        current[f"{prefix}_valid"] = _bool(row, "valid", "observation_valid", default=True)
        for variance_kind in ("aleatoric", "epistemic", "total"):
            value = _float(row, f"{variance_kind}_variance")
            if np.isfinite(value):
                current[f"{prefix}_{variance_kind}_variance"] = value
            standard = _float(row, f"{variance_kind}_std")
            if np.isfinite(standard):
                current[f"{prefix}_{variance_kind}_std"] = standard
        if _bool(row, "artifact_mask", "corruption_mask", "mask", default=False):
            current["artifact_mask"] = True
    output = list(grouped.values())
    output.sort(key=lambda row: _float(row, "time_s", default=0.0))
    return output


def _plot_band(
    axis: Any,
    time: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    valid: np.ndarray,
    *,
    color: str,
    label: str,
    critical: float,
    band_label: str = "95% 后验带",
) -> None:
    mean = np.asarray(mean, dtype=float).copy()
    std = np.asarray(std, dtype=float).copy()
    valid = np.asarray(valid, dtype=bool) & np.isfinite(mean)
    mean[~valid] = np.nan
    std[~(valid & np.isfinite(std) & (std >= 0))] = np.nan
    if np.any(np.isfinite(std)):
        axis.fill_between(
            time,
            mean - critical * std,
            mean + critical * std,
            color=color,
            alpha=0.14,
            linewidth=0,
            label=band_label,
        )
    axis.plot(time, mean, color=color, linewidth=1.25, label=label)


def _plot_trajectories(
    trajectories: Sequence[Mapping[str, Any]],
    uncertainty: Sequence[Mapping[str, Any]],
    output_dir: Path,
    gates: Mapping[str, Any],
) -> Path:
    selected_raw = _preferred_sample(_group_rows(trajectories), prefer_corruption=True)
    selected = {model: _wide_trajectory_rows(rows) for model, rows in selected_raw.items()}
    models = sorted(selected, key=_model_sort_key)
    models = models or [""]
    lookup = _uncertainty_lookup(uncertainty)
    dof = _student_dof(uncertainty, gates)
    critical = _student_critical(dof)
    fig, axes = plt.subplots(3, len(models), figsize=(6.2 * len(models), 10.0), squeeze=False, sharex="col", constrained_layout=True)
    for column, model in enumerate(models):
        rows = selected.get(model, [])
        time = _array(rows, "time_s")
        identity = _id_tuple(rows[0]) if rows else ("", "", model)
        mask = _artifact_mask(rows)
        for row_index, (prefix, title, label) in enumerate(SIGNAL_SPECS):
            axis = axes[row_index, column]
            observed = _array(rows, f"{prefix}_obs", f"{prefix}_corrupted", f"{prefix}_observation")
            clean = _array(rows, f"{prefix}_clean", f"{prefix}_truth", f"clean_{prefix}")
            mean = _array(rows, f"{prefix}_mean", f"{prefix}_posterior", f"{prefix}_estimate")
            valid = _valid_array(rows, f"{prefix}_valid", "valid", default=True)
            std = np.asarray([
                np.sqrt(max(
                    _uncertainty_value(lookup, identity, current_time, prefix, "total")
                    if np.isfinite(_uncertainty_value(lookup, identity, current_time, prefix, "total"))
                    else _float(rows[index], f"{prefix}_total_variance", default=np.nan),
                    0.0,
                ))
                for index, current_time in enumerate(time)
            ])
            if np.any(np.isfinite(clean)):
                axis.plot(time, np.where(valid, clean, np.nan), color=OBS_COLOR, linewidth=1.0, label="干净真值")
            if np.any(np.isfinite(observed)):
                axis.plot(time, np.where(valid, observed, np.nan), color=CORRUPTED_COLOR, linewidth=0.9, label="污染观测（模型输入）")
            if np.any(np.isfinite(mean)):
                _plot_band(
                    axis,
                    time,
                    mean,
                    std,
                    valid,
                    color=_model_color(model),
                    label="后验均值",
                    critical=critical,
                    band_label=f"95% 后验预测带（Student-t，{('ν=' + format(dof, 'g')) if np.isfinite(dof) else 'ν 未记录'}）",
                )
            _shade_mask(axis, time, mask)
            axis.set_ylabel(label)
            axis.set_title(_model_label(model) if row_index == 0 else "")
            _legend_if_handles(axis, loc="best", fontsize=7)
        axes[-1, column].set_xlabel("时间（秒）")
    _style_axes(axes.flat)
    dof_text = f"ν={dof:g}" if np.isfinite(dof) else "ν 未记录"
    scenario = _id_tuple(next(iter(next(iter(selected_raw.values()))), {}))[1] if selected_raw else "未记录"
    fig.suptitle(
        f"观测、干净真值与后验轨迹（Student-t，{dof_text}）\n"
        f"场景：{scenario}；灰线为污染输入，黑线为合成干净真值；黄色区域仅表示注入伪影掩码",
        fontsize=13,
    )
    return _save(fig, output_dir, "观测真值后验_轨迹")


def _state_rows_by_model(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    output: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        model = _id_tuple(row)[2]
        state_name = _text(row, "state_name", "state", default="")
        if state_name:
            output[model][_normalise_component(state_name)].append(dict(row))
            continue
        for name, _label, _unit, _rest in STATE_SPECS:
            truth = _float(row, f"{name}_true", f"{name}_truth")
            mean = _float(row, f"{name}_mean", f"{name}_posterior")
            if np.isfinite(truth) or np.isfinite(mean):
                generated = dict(row)
                generated.update({"state_name": name, "truth": truth, "posterior_mean": mean})
                output[model][name].append(generated)
    for states in output.values():
        for values in states.values():
            values.sort(key=lambda row: _float(row, "time_s", default=0.0))
    return {model: dict(states) for model, states in output.items()}


def _plot_states(
    states: Sequence[Mapping[str, Any]],
    uncertainty: Sequence[Mapping[str, Any]],
    output_dir: Path,
    gates: Mapping[str, Any],
) -> Path:
    groups = _group_rows(states)
    selected = _preferred_sample(groups)
    first_rows = next(iter(selected.values()), [])
    pair = _id_tuple(first_rows[0])[:2] if first_rows else ("", "")
    models = sorted(selected, key=_model_sort_key)
    state_by_model = _state_rows_by_model([row for values in selected.values() for row in values])
    models = models or sorted(state_by_model) or [""]
    lookup = _uncertainty_lookup(uncertainty)
    # State variance is the conditional EKF/Laplace Gaussian approximation;
    # only observation predictive bands use the Student-t critical factor.
    critical = 1.96
    fig, axes = plt.subplots(6, len(models), figsize=(6.2 * len(models), 16.5), squeeze=False, sharex="col", constrained_layout=True)
    for column, model in enumerate(models):
        identity = (pair[0], pair[1], model)
        for row_index, (name, label, unit, rest) in enumerate(STATE_SPECS):
            axis = axes[row_index, column]
            rows = state_by_model.get(model, {}).get(name, [])
            time = _array(rows, "time_s")
            truth = _array(rows, "truth", "state_truth", f"{name}_true")
            mean = _array(rows, "posterior_mean", "mean", "state_mean", f"{name}_mean")
            valid = _valid_array(rows, "state_valid", "valid", default=True)
            std = np.asarray([
                np.sqrt(max(
                    _float(rows[index], "state_variance", "variance")
                    if np.isfinite(_float(rows[index], "state_variance", "variance"))
                    else (
                        _uncertainty_value(lookup, identity, current_time, name, "total")
                        if np.isfinite(_uncertainty_value(lookup, identity, current_time, name, "total"))
                        else _uncertainty_value(lookup, identity, current_time, name, "epistemic")
                    ),
                    0.0,
                ))
                for index, current_time in enumerate(time)
            ])
            if np.any(np.isfinite(truth)):
                axis.plot(time, truth, color=OBS_COLOR, linewidth=1.0, label="状态真值")
            if np.any(np.isfinite(mean)):
                _plot_band(
                    axis,
                    time,
                    mean,
                    std,
                    valid,
                    color=_model_color(model),
                    label="后验均值",
                    critical=critical,
                    band_label="95% 条件高斯区间",
                )
            else:
                axis.text(
                    0.5,
                    0.5,
                    "该模型未输出此生理状态",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    color=MISSING_COLOR,
                )
            if np.isfinite(rest):
                axis.axhline(rest, color="#777777", linestyle=":", linewidth=0.8, label="静息参考" if row_index == 0 else None)
            axis.set_ylabel(f"{label}\n({unit})")
            axis.set_title(_model_label(model) if row_index == 0 else "")
            _legend_if_handles(axis, loc="best", fontsize=7)
        axes[-1, column].set_xlabel("时间（秒）")
    _style_axes(axes.flat)
    fig.suptitle(
        "T3a 生理状态：r/s/f/v/p/q 真值与后验\n"
        "r 为操作性有效驱动；p/q 为归一化血管室坐标，不是 HbO/HbR 浓度",
        fontsize=13,
    )
    return _save(fig, output_dir, "生理状态真值后验_r_s_f_v_p_q")


def _plot_noise_innovation(
    trajectories: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> Path:
    selected_raw = _preferred_sample(_group_rows(trajectories), prefer_corruption=True)
    selected = {model: _wide_trajectory_rows(rows) for model, rows in selected_raw.items()}
    models = sorted(selected, key=_model_sort_key) or [""]
    fig, axes = plt.subplots(3, len(models), figsize=(6.2 * len(models), 10.0), squeeze=False, sharex="col", constrained_layout=True)
    for column, model in enumerate(models):
        rows = selected.get(model, [])
        time = _array(rows, "time_s")
        mask = _artifact_mask(rows)
        for row_index, (prefix, title, label) in enumerate(SIGNAL_SPECS):
            axis = axes[row_index, column]
            observed = _array(rows, f"{prefix}_obs", f"{prefix}_corrupted", f"{prefix}_observation")
            posterior = _array(rows, f"{prefix}_mean", f"{prefix}_posterior", f"{prefix}_estimate")
            artifact = _array(rows, f"artifact_{prefix}", f"{prefix}_artifact", "artifact")
            nuisance = _array(rows, f"nuisance_{prefix}", f"{prefix}_nuisance", f"{prefix}_nuisance_mean")
            innovation = _array(rows, f"innovation_{prefix}", f"{prefix}_innovation")
            if not np.any(np.isfinite(innovation)):
                innovation = observed - posterior
            if np.any(np.isfinite(artifact)):
                axis.plot(time, artifact, color="#CC79A7", linewidth=1.0, label="注入伪影/干扰真值")
            if np.any(np.isfinite(nuisance)):
                axis.plot(time, nuisance, color="#009E73", linewidth=1.0, label="后验干扰状态")
            if np.any(np.isfinite(innovation)):
                axis.plot(time, innovation, color="#D55E00", linewidth=1.0, label="创新（观测−后验）")
            if not any(np.any(np.isfinite(values)) for values in (artifact, nuisance, innovation)):
                axis.text(0.5, 0.5, "该模型未输出可用诊断", transform=axis.transAxes, ha="center", va="center", color=MISSING_COLOR)
            axis.axhline(0.0, color="#555555", linewidth=0.7)
            _shade_mask(axis, time, mask)
            axis.set_ylabel(f"{label}\n残差量纲")
            axis.set_title(_model_label(model) if row_index == 0 else "")
            _legend_if_handles(axis, loc="best", fontsize=7)
        axes[-1, column].set_xlabel("时间（秒）")
    _style_axes(axes.flat)
    scenario = _id_tuple(next(iter(next(iter(selected_raw.values()))), {}))[1] if selected_raw else "未记录"
    fig.suptitle(
        "噪声、伪影与创新诊断\n"
        f"场景：{scenario}；创新仅是观测−后验残差；只有在 T-P3 支持后才可称为已分离噪声",
        fontsize=13,
    )
    return _save(fig, output_dir, "噪声创新_伪影分离")


def _uncertainty_component_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        component = _normalise_component(_text(row, "component", "target", "state_name"))
        if component:
            output[component].append(dict(row))
    for values in output.values():
        values.sort(key=lambda row: _float(row, "time_s", default=0.0))
    return dict(output)


def _plot_uncertainty(
    uncertainty: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]],
    output_dir: Path,
    gates: Mapping[str, Any],
) -> Path:
    groups = _group_rows(uncertainty)
    selected = _preferred_sample(groups, prefer_corruption=True)
    # One model keeps component colours readable; model comparisons belong in
    # the metrics panel and do not require nine overlaid lines here.
    model = sorted(selected, key=_model_sort_key)[0] if selected else ""
    selected_pair = _id_tuple(next(iter(next(iter(selected.values()))), {}))[:2] if selected else ("", "")
    identity = (selected_pair[0], selected_pair[1], model)
    component_rows = _uncertainty_component_rows(selected.get(model, []))
    # The P0 runner may export state variance in states.csv before adding
    # state components to uncertainty.csv.  Keep that conditional posterior
    # visible, but do not invent missing aleatoric/total components.
    state_groups = _group_rows(states)
    state_key = (identity[0], identity[1], model)
    for name in (item[0] for item in STATE_SPECS):
        if name in component_rows or state_key not in state_groups:
            continue
        generated: list[dict[str, Any]] = []
        for row in state_groups[state_key]:
            if _normalise_component(_text(row, "state_name", "state")) != name:
                continue
            variance = _float(row, "state_variance", "variance")
            generated.append({
                "time_s": _float(row, "time_s"),
                "component": name,
                "epistemic_variance": variance,
                "aleatoric_variance": np.nan,
                "total_variance": np.nan,
                "status": "conditional_state_only",
            })
        if generated:
            component_rows[name] = generated
    components = [name for name, _label, _unit, _rest in STATE_SPECS]
    components[:0] = ["eeg", "hbo", "hbr"]
    fig, axes = plt.subplots(3, 3, figsize=(15.0, 12.0), squeeze=False, constrained_layout=True)
    identity_errors: list[float] = []
    for axis, component in zip(axes.flat, components, strict=True):
        rows = component_rows.get(component, [])
        time = _array(rows, "time_s")
        values: dict[str, np.ndarray] = {}
        for kind in ("aleatoric", "epistemic", "total"):
            values[kind] = np.asarray([
                _float(row, f"{kind}_variance", default=np.nan)
                if np.isfinite(_float(row, f"{kind}_variance", default=np.nan))
                else np.square(_float(row, f"{kind}_std", default=np.nan))
                for row in rows
            ])
        finite_identity = np.isfinite(values["aleatoric"]) & np.isfinite(values["epistemic"]) & np.isfinite(values["total"])
        if np.any(finite_identity):
            identity_errors.extend(np.abs(values["total"][finite_identity] - values["aleatoric"][finite_identity] - values["epistemic"][finite_identity]).tolist())
        plotted = False
        for kind, label, color in (
            ("aleatoric", "偶然不确定性（aleatoric）", "#0072B2"),
            ("epistemic", "条件状态后验方差（epistemic*）", "#009E73"),
            ("total", "总方差", "#D55E00"),
        ):
            if np.any(np.isfinite(values[kind])):
                axis.plot(time, values[kind], color=color, linewidth=1.1, label=label)
                plotted = True
        if not plotted:
            axis.text(0.5, 0.5, "未提供该分量", transform=axis.transAxes, ha="center", va="center", color=MISSING_COLOR)
        axis.set_title({"eeg": "EEG 观测代理", "hbo": "HbO", "hbr": "HbR", **{name: label for name, label, _unit, _rest in STATE_SPECS}}.get(component, component))
        axis.set_ylabel("方差（原坐标单位²）")
        _legend_if_handles(axis, loc="best", fontsize=7)
    _style_axes(axes.flat)
    max_error = max(identity_errors, default=0.0)
    identity_label = f"max|总−偶然−认知|={max_error:.2e}" if identity_errors else "方差恒等式未能检查"
    fig.suptitle(
        f"观测与状态不确定性分解（{_model_label(model)}）\n"
        f"{identity_label}；Student-t 观测噪声；epistemic* 为条件 latent-state 后验，"
        "不含 κ/τ/模型不确定性，缺少参数分量显示 NA",
        fontsize=13,
    )
    return _save(fig, output_dir, "不确定性分解_观测状态")


def _metric_name(row: Mapping[str, Any]) -> str:
    return _text(row, "metric", "metric_name", "measure", default="metric")


def _plot_calibration(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    gates: Mapping[str, Any],
) -> Path:
    dof = _student_dof(rows, gates)
    pit_bins: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    pit_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    coverages: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    proper: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    risk: dict[tuple[str, str], list[float]] = defaultdict(list)
    # The evaluator currently exports one row containing named calibration
    # fields (``empirical_coverage``, ``pit``, ``crps``, ...), while older
    # fixtures use a ``metric,value`` pair.  Keep both contracts readable.
    pit_means: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        model = _text(row, "model_id", "model", default="t3a_balloon_robust")
        component = _normalise_component(_text(row, "component", "target", default="all"))
        key = (model, component)
        explicit_value = _float(row, "value", "estimate")
        entries: list[tuple[str, float, bool]] = []
        if np.isfinite(explicit_value):
            entries.append((_metric_name(row).lower(), explicit_value, False))
        else:
            # Named fields are aggregate scalars.  In particular, ``pit`` is
            # a mean PIT in the current runner, so it must not be rendered as
            # an artificial PIT histogram.
            for metric_name, field_name, mean_pit in (
                ("coverage", "empirical_coverage", False),
                ("mean_pit", "pit", True),
                ("crps", "crps", False),
                ("nll", "nll", False),
                ("uncertainty_risk_spearman", "uncertainty_risk_spearman", False),
            ):
                value = _float(row, field_name)
                if np.isfinite(value):
                    entries.append((metric_name, value, mean_pit))
        for metric, value, mean_pit in entries:
            match = re.search(r"(?:pit|hist)[^0-9]*([0-9]{1,2})$", metric)
            if match:
                pit_bins[key][int(match.group(1))] = value
            elif metric in {"pit", "pit_value"} or metric.startswith("pit_value"):
                pit_values[key].append(value)
            elif mean_pit or metric in {"mean_pit", "pit_mean"}:
                pit_means[key].append(value)
            if "coverage" in metric:
                nominal = _float(row, "nominal_level", "nominal", "level")
                if not np.isfinite(nominal):
                    match = re.search(r"(50|80|95)", metric)
                    nominal = float(match.group(1)) / 100.0 if match else np.nan
                if np.isfinite(nominal):
                    coverages[key].append((nominal, value))
            if any(token in metric for token in ("crps", "nll", "log_score", "proper_score")):
                proper[key].append((metric, value))
            if "risk" in metric or "spearman" in metric:
                risk[key].append(value)
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.0), constrained_layout=True)
    centers = np.linspace(0.05, 0.95, 10)
    for key, bins in sorted(pit_bins.items()):
        if bins:
            values = [bins.get(index, np.nan) for index in range(10)]
            axes[0, 0].plot(centers, values, marker="o", linewidth=1.1, label=f"{_model_label(key[0])} · {_component_label(key[1])}")
    for key, values in sorted(pit_values.items()):
        if values:
            histogram, _ = np.histogram(values, bins=np.linspace(0.0, 1.0, 11))
            axes[0, 0].plot(centers, histogram / max(len(values), 1), marker="o", linewidth=1.1, label=f"{_model_label(key[0])} · {_component_label(key[1])}")
    for key, values in sorted(pit_means.items()):
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            mean = float(np.mean(finite))
            axes[0, 0].axvline(
                mean,
                color=_model_color(key[0]),
                linestyle="-.",
                linewidth=1.2,
                label=f"{_model_label(key[0])} · {_component_label(key[1])} · 均值 PIT（非分布）",
            )
    axes[0, 0].axhline(0.1, color="#555555", linestyle="--", linewidth=0.9, label="均匀分布期望")
    axes[0, 0].set(title="PIT 分布/导出均值（Student-t）", xlabel="PIT 分箱中心或均值位置", ylabel="箱内比例（均值线不代表分布）")
    for key, values in sorted(coverages.items()):
        values = sorted(values)
        axes[0, 1].plot([value[0] for value in values], [value[1] for value in values], marker="o", linewidth=1.1, label=f"{_model_label(key[0])} · {_component_label(key[1])}")
    axes[0, 1].plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=0.9, label="理想校准")
    axes[0, 1].set(title="区间覆盖率校准", xlabel="名义覆盖率", ylabel="经验覆盖率")
    proper_names = sorted({metric for values in proper.values() for metric, _value in values})
    for key, values in sorted(proper.items()):
        grouped: dict[str, list[float]] = defaultdict(list)
        for metric, value in values:
            grouped[metric].append(value)
        for index, metric in enumerate(proper_names):
            metric_values = grouped.get(metric, [])
            if metric_values:
                axes[1, 0].scatter(
                    np.full(len(metric_values), index),
                    metric_values,
                    color=_model_color(key[0]),
                    alpha=0.75,
                    label=f"{_model_label(key[0])} · {_component_label(key[1])}" if index == 0 else None,
                )
    axes[1, 0].set_xticks(
        range(len(proper_names)),
        [{"crps": "CRPS", "nll": "负对数似然"}.get(name, name) for name in proper_names],
    )
    axes[1, 0].set(title="恰当评分（越低越好）", xlabel="指标类别", ylabel="评分值")
    for key, values in sorted(risk.items()):
        axes[1, 1].scatter(np.arange(len(values)), values, color=_model_color(key[0]), alpha=0.75, label=f"{_model_label(key[0])} · {_component_label(key[1])}")
    axes[1, 1].axhline(0.0, color="#555555", linestyle="--", linewidth=0.8)
    axes[1, 1].set(title="不确定性—风险秩相关", xlabel="记录序号", ylabel="Spearman / 风险指标")
    for axis in axes.flat:
        if not axis.lines and not axis.collections:
            axis.text(0.5, 0.5, "calibration.csv 未提供该指标", transform=axis.transAxes, ha="center", va="center", color=MISSING_COLOR)
        _legend_if_handles(axis, loc="best", fontsize=7)
    _style_axes(axes.flat)
    dof_text = f"ν={dof:g}" if np.isfinite(dof) else "ν 未记录"
    fig.suptitle(f"Student-t 预测校准：PIT、覆盖率与 proper score（{dof_text}）", fontsize=13)
    return _save(fig, output_dir, "校准_PIT_覆盖率")


def _parameter_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        name = _text(row, "parameter", "parameter_name", default="")
        if name:
            output[name].append(dict(row))
    return dict(output)


def _plot_parameters(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    profile_rows: Sequence[Mapping[str, Any]] = (),
) -> Path:
    grouped = _parameter_rows(rows)
    profile_grouped = _parameter_rows(profile_rows)
    names = sorted(grouped)
    fig = plt.figure(figsize=(16.0, max(7.0, 2.0 * max(len(names), 1))), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.65))
    recovery_axis = fig.add_subplot(grid[0, 0])
    profile_grid = grid[0, 1].subgridspec(max(1, len(names)), 1)
    if not names:
        recovery_axis.text(0.5, 0.5, "parameter_recovery.csv 未提供参数", transform=recovery_axis.transAxes, ha="center", va="center")
    for index, name in enumerate(names):
        values = grouped[name]
        truth = np.asarray([_float(row, "true", "true_value", "truth", "parameter_truth") for row in values], dtype=float)
        estimate = np.asarray([_float(row, "estimate", "posterior_mean", "parameter_estimate") for row in values], dtype=float)
        sd = np.asarray([_float(row, "sd", "std", "parameter_sd") for row in values], dtype=float)
        lower = np.asarray([_float(row, "lower", "lower_bound", "prior_lower") for row in values], dtype=float)
        upper = np.asarray([_float(row, "upper", "upper_bound", "prior_upper") for row in values], dtype=float)
        finite_bounds = np.isfinite(lower) & np.isfinite(upper) & (upper > lower)
        scale_low = float(np.nanmin(lower[finite_bounds])) if np.any(finite_bounds) else np.nan
        scale_high = float(np.nanmax(upper[finite_bounds])) if np.any(finite_bounds) else np.nan
        if np.isfinite(scale_low) and np.isfinite(scale_high) and scale_high > scale_low:
            transform = lambda array: (array - scale_low) / (scale_high - scale_low)
            xlabel = "参数值（共享先验范围归一化）"
            lower_marker, upper_marker = 0.0, 1.0
        else:
            transform = lambda array: array
            xlabel = "参数值（原单位）"
            lower_marker = upper_marker = np.nan
        finite_estimate = np.isfinite(estimate)
        if np.any(finite_estimate):
            positions = np.full(np.count_nonzero(finite_estimate), index, dtype=float)
            recovery_axis.errorbar(
                transform(estimate[finite_estimate]),
                positions,
                xerr=transform(estimate[finite_estimate] + np.nan_to_num(sd[finite_estimate], nan=0.0)) - transform(estimate[finite_estimate]),
                fmt="o",
                color="#0072B2",
                alpha=0.8,
                label="估计 ± SD" if index == 0 else None,
            )
        finite_truth = np.isfinite(truth)
        if np.any(finite_truth):
            recovery_axis.scatter(transform(truth[finite_truth]), np.full(np.count_nonzero(finite_truth), index), marker="x", color="#D55E00", s=42, label="真值" if index == 0 else None)
        if np.isfinite(lower_marker):
            recovery_axis.axvline(lower_marker, color="#777777", linestyle=":", linewidth=0.7)
            recovery_axis.axvline(upper_marker, color="#777777", linestyle=":", linewidth=0.7)
        statuses = sorted({_text(row, "identifiability_status", "status", default="未声明") for row in values})
        recovery_axis.text(1.02, index, ";".join(statuses), transform=recovery_axis.get_yaxis_transform(), va="center", fontsize=7)
        profile_axis = fig.add_subplot(profile_grid[index, 0])
        profile_source = profile_grouped.get(name, values)
        profile_points = [
            (_float(row, "grid_value", "profile_value", "parameter_value"), _float(row, "delta_objective", "profile_delta", "profile_objective"))
            for row in profile_source
        ]
        profile_points = [(x, y) for x, y in profile_points if np.isfinite(x) and np.isfinite(y)]
        if profile_points:
            profile_points.sort()
            profile_axis.plot([point[0] for point in profile_points], [point[1] for point in profile_points], marker="o", linewidth=1.0, color="#0072B2")
            profile_axis.axhline(0.0, color="#555555", linestyle="--", linewidth=0.8)
        else:
            profile_axis.text(0.5, 0.5, "无 profile 数据", transform=profile_axis.transAxes, ha="center", va="center", color=MISSING_COLOR)
        true_value = _float(values[0], "true", "true_value", "truth", "parameter_truth")
        lower_value = _float(values[0], "lower", "lower_bound", "prior_lower")
        upper_value = _float(values[0], "upper", "upper_bound", "prior_upper")
        if np.isfinite(true_value):
            profile_axis.axvline(true_value, color="#D55E00", linestyle="-", linewidth=0.9, label="真值")
        if np.isfinite(lower_value):
            profile_axis.axvline(lower_value, color="#777777", linestyle=":", linewidth=0.7)
        if np.isfinite(upper_value):
            profile_axis.axvline(upper_value, color="#777777", linestyle=":", linewidth=0.7)
        profile_axis.set_ylabel(name)
        profile_axis.grid(True, color=GRID_COLOR, linewidth=0.6)
    recovery_axis.set_yticks(np.arange(len(names)), names)
    recovery_axis.set_xlabel(xlabel if names else "参数值")
    recovery_axis.set_title("真值/估计/边界")
    _legend_if_handles(recovery_axis, loc="best", fontsize=8)
    recovery_axis.grid(axis="x", color=GRID_COLOR, linewidth=0.6)
    fig.suptitle("参数恢复、边界接触与目标函数切片\n切片固定另一参数；边界/先验主导的参数不得获得生理解释", fontsize=13)
    return _save(fig, output_dir, "参数恢复_多起点_profile")


def _first_metric(rows: Sequence[Mapping[str, Any]], preferred: Sequence[str]) -> str:
    names = {_metric_name(row) for row in rows if np.isfinite(_float(row, "value"))}
    lowered = {name.lower(): name for name in names}
    for candidate in preferred:
        for lower, original in lowered.items():
            if candidate in lower:
                return original
    return sorted(names)[0] if names else "metric"


def _plot_null_severity(
    metrics: Sequence[Mapping[str, Any]],
    null_metrics: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> Path:
    stress = [row for row in metrics if np.isfinite(_float(row, "severity"))]
    selected_metrics: list[str] = []
    for candidate in ("artifact_attenuation", "truth_nrmse", "off_artifact_distortion"):
        for row in stress:
            metric = _metric_name(row)
            if candidate in metric.lower() and metric not in selected_metrics:
                selected_metrics.append(metric)
                break
    if not selected_metrics:
        selected_metrics = [_first_metric(stress, ("recovery", "attenuation", "nrmse"))]
    selected_metrics = selected_metrics[:3]
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.0), constrained_layout=True)
    for panel, metric in enumerate(selected_metrics):
        axis = axes.flat[panel]
        candidates = [row for row in stress if _metric_name(row) == metric]
        models = sorted({_text(row, "model_id", "model", default="t3a_balloon_robust") for row in candidates})
        plotted_model = False
        for model in models:
            by_severity: dict[float, list[float]] = defaultdict(list)
            for row in candidates:
                if _text(row, "model_id", "model", default="t3a_balloon_robust") != model:
                    continue
                severity = _float(row, "severity")
                value = _float(row, "value")
                if np.isfinite(severity) and np.isfinite(value):
                    by_severity[severity].append(value)
            xs = sorted(by_severity)
            if not xs:
                continue
            medians = [float(np.median(by_severity[x])) for x in xs]
            lows = [float(np.quantile(by_severity[x], 0.25)) for x in xs]
            highs = [float(np.quantile(by_severity[x], 0.75)) for x in xs]
            axis.plot(xs, medians, marker="o", color=_model_color(model), linewidth=1.1, label=_model_label(model))
            axis.fill_between(xs, lows, highs, color=_model_color(model), alpha=0.14, linewidth=0, label="重复实验 IQR")
            plotted_model = True
        metric_label = {
            "artifact_attenuation": "伪影衰减率",
            "truth_nrmse": "相对真值归一化误差",
            "off_artifact_distortion": "伪影区外失真",
        }.get(metric, metric)
        axis.set_title(f"严重度：{metric_label}")
        axis.set_xlabel("伪影严重度")
        axis.set_ylabel("指标值")
        if plotted_model:
            _legend_if_handles(axis, loc="best", fontsize=7)
        else:
            axis.text(0.5, 0.5, "metrics.csv 未提供该严重度指标", transform=axis.transAxes, ha="center", va="center", color=MISSING_COLOR)
    for panel in range(len(selected_metrics), 3):
        axis = axes.flat[panel]
        axis.set_title("严重度指标不可用")
        axis.text(0.5, 0.5, "metrics.csv 未提供更多严重度指标", transform=axis.transAxes, ha="center", va="center", color=MISSING_COLOR)
    null_axis = axes.flat[3]
    exported_null_types = sorted({_text(row, "null_type", "null_id", default="null") for row in null_metrics if _text(row, "null_type", "null_id", default="")})
    null_metric = _first_metric(null_metrics, ("recovery", "association", "nrmse", "pcc"))
    null_rows = [row for row in null_metrics if _metric_name(row) == null_metric]
    null_types = sorted({_text(row, "null_type", "null_id", default="null") for row in null_rows})
    for index, null_type in enumerate(null_types):
        values = np.asarray([_float(row, "value") for row in null_rows if _text(row, "null_type", "null_id", default="null") == null_type], dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.zeros(1)
        null_axis.scatter(
            np.full(len(values), index) + jitter,
            values,
            alpha=0.75,
            s=24,
            label=NULL_LABELS.get(null_type, null_type),
        )
        threshold = np.asarray([_float(row, "threshold") for row in null_rows if _text(row, "null_type", "null_id", default="null") == null_type], dtype=float)
        threshold = threshold[np.isfinite(threshold)]
        if len(threshold):
            null_axis.plot([index - 0.2, index + 0.2], [float(np.median(threshold))] * 2, color="#D55E00", linewidth=1.2)
    if not null_rows:
        message = "null_metrics.csv 未提供可用数值"
        if exported_null_types:
            message += "\n已导出类型：" + "、".join(NULL_LABELS.get(value, value) for value in exported_null_types)
        null_axis.text(0.5, 0.5, message, transform=null_axis.transAxes, ha="center", va="center", color=MISSING_COLOR)
    null_axis.axhline(0.0, color="#555555", linestyle="--", linewidth=0.8)
    null_metric_label = {
        "cross_modal_driver_correlation": "跨模态驱动相关",
    }.get(null_metric, null_metric)
    null_axis.set(title=f"零假设分布：{null_metric_label}", xlabel="零假设类型", ylabel="指标值")
    null_axis.set_xticks(
        range(len(null_types)),
        [NULL_LABELS.get(value, value) for value in null_types],
        rotation=18,
        ha="right",
    )
    _legend_if_handles(null_axis, loc="best", fontsize=7)
    _style_axes(axes.flat)
    if exported_null_types:
        null_summary = "实际导出的零假设：" + "、".join(NULL_LABELS.get(value, value) for value in exported_null_types)
    else:
        null_summary = "未导出可用零假设类型"
    fig.suptitle(
        "受控伪影严重度与零假设对照\n"
        f"点为重复实验；{null_summary}；零假设结果仅作边界对照，不把它当作活动证据",
        fontsize=13,
    )
    return _save(fig, output_dir, "null_严重度性能")


def _geometry_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        x = _float(row, "x", "coord_x", "position_x")
        y = _float(row, "y", "coord_y", "position_y")
        z = _float(row, "z", "coord_z", "position_z")
        # A coordinate alone is not a spatial result.  Require the explicit
        # channel identifier and a finite, exported association/weight; never
        # fill an absent association with zero (that would create a fake map).
        channel = _text(row, "channel_id", default="")
        value = _float(
            row,
            "association",
            "correlation",
            "association_value",
            "source_weight_recovered",
            "recovered_weight",
        )
        if channel and np.isfinite(x) and np.isfinite(y) and np.isfinite(value):
            output.append({**dict(row), "x": x, "y": y, "z": z, "channel": channel, "value": value})
    return output


def _plot_spatial_if_available(
    trajectories: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> Path | None:
    values = _geometry_rows([*trajectories, *states])
    if not values:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5), constrained_layout=True)
    finite_value = np.asarray([row["value"] for row in values], dtype=float)
    finite_value = finite_value[np.isfinite(finite_value)]
    if not finite_value.size:  # defensive; _geometry_rows already filters this
        return None
    limit = max(float(np.nanmax(np.abs(finite_value))), 1e-12)
    norm = mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    markers = {"EEG": "o", "HbO": "^", "HbR": "v"}
    for axis, ykey, ylabel in ((axes[0], "y", "y（空间坐标）"), (axes[1], "z", "z（空间坐标）")):
        for role in sorted({ _text(row, "role", "modality", default="通道") for row in values}):
            subset = [row for row in values if _text(row, "role", "modality", default="通道") == role]
            x = np.asarray([row["x"] for row in subset], dtype=float)
            y = np.asarray([row[ykey] for row in subset], dtype=float)
            c = np.asarray([row["value"] for row in subset], dtype=float)
            axis.scatter(x, y, c=c, cmap="RdBu_r", norm=norm, s=44, marker=markers.get(role, "o"), alpha=0.82, label=role)
        axis.set_xlabel("x（空间坐标）")
        axis.set_ylabel(ylabel)
        _legend_if_handles(axis, loc="best", fontsize=8)
        axis.grid(True, color=GRID_COLOR, linewidth=0.6)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), ax=axes, label="通道关联/恢复权重")
    fig.suptitle(
        "传感器空间关联（有几何字段时）\n"
        "这是通道级空间支持，不是全脑活动热图，也不是源定位结果",
        fontsize=13,
    )
    return _save(fig, output_dir, "通道空间关联_边界")


def _prepare_output_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"拒绝覆盖已有 T3a 可视化文件：{path}")
        if any(path.iterdir()):
            raise FileExistsError(f"拒绝覆盖已有 T3a 可视化目录：{path}")
    path.mkdir(parents=True, exist_ok=True)


def _render(run_dir: Path, output_dir: Path | None = None) -> Path:
    trajectories = _read_csv(run_dir / "trajectories.csv")
    states = _read_csv(run_dir / "states.csv")
    uncertainty = _read_csv(run_dir / "uncertainty.csv")
    metrics = _read_csv(run_dir / "metrics.csv")
    parameter_recovery = _read_csv(run_dir / "parameter_recovery.csv")
    profile_likelihood = _read_csv_optional(run_dir / "profile_likelihood.csv")
    calibration = _read_csv(run_dir / "calibration.csv")
    null_metrics = _read_csv(run_dir / "null_metrics.csv")
    gates = _read_gates(run_dir / "gates.json")
    if output_dir is None:
        output_dir = run_dir / "figures" / "t3a_p0"
    output_dir = Path(output_dir)
    _prepare_output_dir(output_dir)
    outputs = [
        _plot_trajectories(trajectories, uncertainty, output_dir, gates),
        _plot_states(states, uncertainty, output_dir, gates),
        _plot_noise_innovation(trajectories, output_dir),
        _plot_uncertainty(uncertainty, states, output_dir, gates),
        _plot_calibration(calibration, output_dir, gates),
        _plot_parameters(parameter_recovery, output_dir, profile_likelihood),
        _plot_null_severity(metrics, null_metrics, output_dir),
    ]
    spatial = _plot_spatial_if_available(trajectories, states, output_dir)
    if spatial is not None:
        outputs.append(spatial)
    gates_summary = gates.get("gates", gates)
    lines = [
        "# T3a Balloon-robust P0 中文可视化",
        "",
        f"输入目录：`{run_dir}`",
        "",
        "## 输出",
        "",
        *[f"- `{path.name}`" for path in outputs],
        "",
        "## 证据边界",
        "",
        "- 观测图区分干净真值、污染输入和后验均值；重建误差不是主要资格指标。",
        "- 状态图中的 `r` 是 operational effective forcing，`s/f/v/p/q` 只按声明的方程和单位解释。",
        "- `p/q` 是归一化血管室坐标；HbO/HbR 必须由显式 forward map 得到，不能互换命名。",
        "- 创新是观测减后验残差；在 T-P3 未支持前不能自动称为已分离噪声。",
        "- 不确定性图区分 aleatoric、epistemic 和 total variance；固定参数协方差不等于 epistemic。",
        "- 校准图按 Student-t 预测分布解释；PIT/coverage 使用导出的 calibration.csv，不重新拟合。",
        "- 参数边界接触或先验主导只能标记为不可辨识，不能获得生理学标签。",
        "- severity 与 null 图保留重复实验和 null 类型；不把 null 结果当作活动证据。",
        "- 空间图（若生成）仅在同时存在 channel_id、有限几何坐标和真实关联值时绘制；它是通道坐标上的关联/权重散点，不是全脑活动热图，也不是源定位。",
        "",
        "## 门状态",
        "",
        f"{json.dumps(gates_summary, ensure_ascii=False, indent=2)}",
    ]
    if spatial is None:
        lines.insert(
            lines.index("## 门状态"),
            "- 未发现同时具备 channel_id、有限几何坐标和真实关联值的字段，空间图不可用；未生成伪热图。",
        )
    summary_path = output_dir / "可视化说明.md"
    if summary_path.exists():
        raise FileExistsError(f"拒绝覆盖已有可视化说明：{summary_path}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_dir


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    fields = fields or ["value"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_self_check() -> None:
    """Render a tiny complete P0 fixture and assert all evidence boundaries."""

    with tempfile.TemporaryDirectory(prefix="t3a_balloon_p0_visual_check_") as temp:
        run_dir = Path(temp)
        rng = np.random.default_rng(17)
        times = np.arange(32, dtype=float) / 10.0 - 1.6
        trajectory_rows: list[dict[str, Any]] = []
        state_rows: list[dict[str, Any]] = []
        uncertainty_rows: list[dict[str, Any]] = []
        model = "t3a_balloon_robust"
        for index, time in enumerate(times):
            clean = {
                "eeg": np.sin(time),
                "hbo": 0.4 * np.cos(time),
                "hbr": -0.2 * np.sin(time),
            }
            artifact = 0.8 if 10 <= index < 15 else 0.0
            row = {
                "replicate_id": "replicate_00",
                "scenario_id": "severity_1",
                "model_id": model,
                "time_s": time,
                "artifact_mask": 10 <= index < 15,
            }
            for prefix, value in clean.items():
                row.update({
                    f"{prefix}_clean": value,
                    f"{prefix}_obs": value + artifact,
                    f"{prefix}_mean": value + 0.1 * artifact,
                    f"artifact_{prefix}": artifact,
                    f"{prefix}_valid": True,
                })
            trajectory_rows.append(row)
            state_values = {
                "r": np.sin(time),
                "s": 0.2 * np.cos(time),
                "f": 1.0 + 0.05 * np.sin(time),
                "v": 1.0 + 0.03 * np.cos(time),
                "p": 1.0 + 0.02 * np.sin(time),
                "q": 1.0 - 0.015 * np.sin(time),
            }
            for name, value in state_values.items():
                state_rows.append({
                    "replicate_id": "replicate_00",
                    "scenario_id": "severity_1",
                    "model_id": model,
                    "time_s": time,
                    "state_name": name,
                    "truth": value,
                    "posterior_mean": value + rng.normal(scale=0.01),
                    "state_variance": 0.01,
                    "state_valid": True,
                    "unit": next(unit for state, _label, unit, _rest in STATE_SPECS if state == name),
                })
            for component in ("eeg", "hbo", "hbr", "r", "s", "f", "v", "p", "q"):
                uncertainty_rows.append({
                    "replicate_id": "replicate_00",
                    "scenario_id": "severity_1",
                    "model_id": model,
                    "time_s": time,
                    "component": component,
                    "aleatoric_variance": 0.01,
                    "epistemic_variance": 0.002,
                    "total_variance": 0.012,
                    "status": "ok",
                    "student_t_dof": 4,
                })
        metric_rows = []
        for severity in (0.0, 0.5, 1.0, 2.0):
            for metric, value in (("artifact_attenuation", 0.8 - 0.1 * severity), ("r_recovery", 0.9 - 0.1 * severity), ("off_mask_nrmse", 0.1 + 0.02 * severity)):
                metric_rows.append({"model_id": model, "scenario_id": f"severity_{severity:g}", "replicate_id": "replicate_00", "stress_case": "spike", "severity": severity, "component": "r", "metric": metric, "value": value, "support_n": 32, "status": "ok"})
        parameter_rows = []
        for name, truth, lower, upper in (("kappa", 0.7, 0.2, 1.5), ("gamma", 0.4, 0.1, 1.0), ("tau", 2.0, 0.5, 5.0), ("alpha", 0.32, 0.2, 0.6), ("E0", 0.4, 0.2, 0.7)):
            for grid_value in np.linspace(lower, upper, 5):
                parameter_rows.append({"model_id": model, "replicate_id": "replicate_00", "parameter": name, "true": truth, "estimate": truth + 0.02, "sd": 0.03, "lower": lower, "upper": upper, "grid_value": grid_value, "delta_objective": (grid_value - truth) ** 2, "identifiability_status": "identified"})
        calibration_rows = [{"model_id": model, "scenario_id": "severity_1", "component": "r", "metric": f"pit_bin_{index:02d}", "value": 0.1, "support_n": 32, "student_t_dof": 4} for index in range(10)]
        calibration_rows.extend({"model_id": model, "scenario_id": "severity_1", "component": "r", "metric": "coverage", "value": level, "nominal_level": level, "support_n": 32, "student_t_dof": 4} for level in (0.5, 0.8, 0.95))
        calibration_rows.extend(({"model_id": model, "scenario_id": "severity_1", "component": "r", "metric": metric, "value": value, "support_n": 32, "student_t_dof": 4} for metric, value in (("crps", 0.1), ("nll", 0.3), ("uncertainty_risk_spearman", 0.5))))
        null_rows = [{"model_id": model, "null_id": null_type, "null_type": null_type, "replicate_id": f"replicate_{index:02d}", "component": "r", "metric": "r_recovery", "value": value, "threshold": 0.5, "status": "fail"} for null_type, value in (("independent_pairing", 0.1), ("time_shift", 0.05), ("spatial_permutation", 0.08)) for index in range(3)]
        _write_csv(run_dir / "trajectories.csv", trajectory_rows)
        _write_csv(run_dir / "states.csv", state_rows)
        _write_csv(run_dir / "uncertainty.csv", uncertainty_rows)
        _write_csv(run_dir / "metrics.csv", metric_rows)
        _write_csv(run_dir / "parameter_recovery.csv", parameter_rows)
        _write_csv(run_dir / "calibration.csv", calibration_rows)
        _write_csv(run_dir / "null_metrics.csv", null_rows)
        (run_dir / "gates.json").write_text(json.dumps({"student_t_dof": 4, "gates": {"T-P0": "PASS", "T-P1": "PASS", "T-P2": "PENDING", "T-P3": "PENDING"}}, ensure_ascii=False), encoding="utf-8")
        output = _render(run_dir)
        for stem in MANDATORY_STEMS:
            assert (output / f"{stem}.png").exists(), stem
            assert (output / f"{stem}.alt.txt").exists(), stem
        assert not list(output.glob("*.manifest.json"))
        assert not (output / "通道空间关联_边界.png").exists()
        explanation = (output / "可视化说明.md").read_text(encoding="utf-8")
        assert "不是全脑活动热图" in explanation
        assert "空间图不可用" in explanation
        assert "Student-t" in explanation
        try:
            _render(run_dir, output)
        except FileExistsError:
            pass
        else:  # pragma: no cover - protects the no-overwrite contract
            raise AssertionError("renderer unexpectedly overwrote an existing directory")
    print("T3a P0 visualization self-check passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="completed synthetic-only T3a P0 run directory")
    parser.add_argument("--output-dir", type=Path, help="new figure directory; non-empty directories are rejected")
    parser.add_argument("--self-check", action="store_true", help="render a temporary synthetic fixture and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_check:
        with mpl.rc_context(CJK_STYLE):
            _synthetic_self_check()
        return
    if args.run_dir is None:
        raise SystemExit("需要 --run-dir，或使用 --self-check")
    with mpl.rc_context(CJK_STYLE):
        output = _render(args.run_dir.resolve(), output_dir=args.output_dir)
    print(output)


if __name__ == "__main__":
    main()
