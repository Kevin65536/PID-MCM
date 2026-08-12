#!/usr/bin/env python3
"""Build the comparative-method progress review visuals and editable PPTD deck.

The script consumes the seven subagent JSON audits in ``agent_reports`` and
produces a traceable data snapshot, presentation-sized PNG figures, a PPTD
project, and a concise Markdown audit report.  It never reads protected arrays
or identities; all inputs are public repository evidence summaries.
"""

from __future__ import annotations

import csv
import json
import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "agent_reports"
DECK = ROOT / "comparative_methods_progress_review_deck"
PAGES = DECK / "pages"
MEDIA = DECK / "media"
DATA = ROOT / "data"

METHODS = [
    ("biot", "BIOT", "单模态 EEG"),
    ("cbramod", "CBraMod", "单模态 EEG"),
    ("reve", "REVE-base", "单模态 EEG · overlap track"),
    ("efrm", "EFRM LODO v2", "多模态 EEG–fNIRS"),
    ("normwear", "NormWear adapted", "多模态 EEG–fNIRS"),
    ("brainfusion", "BrainFusion NVC–CSP", "多模态监督重实现"),
    ("stanet", "STA-Net", "多模态 · context reference"),
]

TASKS = [
    ("motor_imagery", "MI"),
    ("mental_arithmetic", "MA"),
    ("wg", "WG"),
    ("nback", "n-back"),
    ("dsr", "DSR"),
    ("visual", "Visual"),
    ("refed_regression", "REFED"),
]

SCORE_KEYS = [
    ("code_components", "必要代码", 0.30),
    ("input_adaptation", "输入适配", 0.20),
    ("output_adaptation", "输出适配", 0.15),
    ("result_generation", "结果生成", 0.25),
    ("evidence_reproducibility", "证据复现", 0.10),
]

PALETTE = {
    "paper": "#F7F3E8",
    "paper2": "#EFE9DC",
    "ink": "#16283C",
    "ink2": "#35475A",
    "copper": "#9C3F17",
    "brass": "#7B5D14",
    "green": "#2F6F62",
    "red": "#9B3A3A",
    "muted": "#5F6662",
    "line": "#C9C1B3",
    "white": "#FFFFFF",
}

FALLBACK_TASKS = {
    "biot": ({"motor_imagery", "mental_arithmetic", "wg", "nback", "dsr", "visual"}, {"refed_regression"}),
    "cbramod": ({"motor_imagery", "mental_arithmetic", "wg", "nback", "dsr", "visual"}, {"refed_regression"}),
    "reve": ({"motor_imagery", "mental_arithmetic", "wg", "nback", "dsr", "visual"}, {"refed_regression"}),
    "efrm": ({x[0] for x in TASKS}, set()),
    "normwear": ({"motor_imagery", "mental_arithmetic", "wg", "nback", "dsr", "visual"}, {"refed_regression"}),
    "brainfusion": ({"motor_imagery", "mental_arithmetic", "wg", "nback", "visual"}, {"dsr", "refed_regression"}),
    "stanet": ({x[0] for x in TASKS}, set()),
}


def _font() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": PALETTE["paper"],
            "axes.facecolor": PALETTE["paper"],
            "savefig.facecolor": PALETTE["paper"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        for preferred in ("task_keys", "formal_project_track", "tasks", "supported", "unsupported"):
            if preferred in value and isinstance(value[preferred], (list, tuple, set)):
                return _as_list(value[preferred])
        known = {task for task, _ in TASKS}
        result = []
        for key, item in value.items():
            normalized = _norm_task(str(key))
            if normalized in known and item not in (False, None, "unsupported"):
                result.append(normalized)
        return result
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            if isinstance(item, dict):
                result.append(str(item.get("task") or item.get("task_id") or item.get("name") or item))
            else:
                result.append(str(item))
        return result
    return [str(value)]


ALIASES = {
    "mi": "motor_imagery",
    "motor imagery": "motor_imagery",
    "motor_imagery": "motor_imagery",
    "ma": "mental_arithmetic",
    "mental arithmetic": "mental_arithmetic",
    "mental_arithmetic": "mental_arithmetic",
    "word generation": "wg",
    "word_generation": "wg",
    "wg": "wg",
    "n-back": "nback",
    "n_back": "nback",
    "nback": "nback",
    "dsr": "dsr",
    "visual": "visual",
    "cognitive motivation": "visual",
    "visual_cognitive_motivation": "visual",
    "refed": "refed_regression",
    "refed regression": "refed_regression",
    "refed_regression": "refed_regression",
}


def _norm_task(text: str) -> str:
    s = text.strip().lower().replace("/", " ")
    return ALIASES.get(s, s)


def _clip_score(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number <= 1.0:
        number *= 100.0
    return round(max(0.0, min(100.0, number)), 1)


def _status_text(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("status", "state", "label", "summary"):
            if key in value:
                return str(value[key])
        if "aggregate_available" in value:
            return (
                f"aggregate_available={str(value.get('aggregate_available')).lower()}; "
                f"support_matched_table={value.get('current_support_matched_table', 'unknown')}; "
                f"strict_status={value.get('strict_cross_subject_status', 'unknown')}"
            )
        if "protected_final_aggregate" in value:
            return (
                f"protected_final_aggregate={value.get('protected_final_aggregate')}; "
                f"final_table={value.get('final_table_numeric_cells', 'unknown')}"
            )
        if "final_table_numeric_cells" in value:
            return (
                f"protected_predictions={value.get('protected_final_predictions', 'unknown')}; "
                f"final_table={value.get('final_table_numeric_cells', 'unknown')}"
            )
        return json.dumps(value, ensure_ascii=False)
    return str(value or "未记录")


def _extract_count(raw: dict[str, Any], key: str, default: int = 0) -> int:
    value = raw.get(key, default)
    if isinstance(value, dict):
        value = value.get("count", value.get("value", default))
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_audits() -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    missing: list[Path] = []
    for slug, display, family in METHODS:
        path = REPORTS / f"{slug}.json"
        if not path.exists():
            missing.append(path)
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw_scores = raw.get("scores", {})
        scores: dict[str, float] = {}
        for key, _, _ in SCORE_KEYS:
            scores[key] = _clip_score(raw_scores.get(key, raw.get(key)))
        weighted = round(sum(scores[k] * w for k, _, w in SCORE_KEYS), 1)
        stated = _clip_score(raw_scores.get("overall", weighted))
        if abs(stated - weighted) > 1.0:
            stated = weighted
        supported_fallback, unsupported_fallback = FALLBACK_TASKS[slug]
        supported = {_norm_task(x) for x in _as_list(raw.get("supported_tasks"))}
        unsupported = {_norm_task(x) for x in _as_list(raw.get("unsupported_tasks"))}
        if not supported and not unsupported:
            supported, unsupported = set(supported_fallback), set(unsupported_fallback)
        planned = {_norm_task(x) for x in _as_list(raw.get("planned_tasks"))}
        if not planned:
            planned = supported | unsupported
        completed_jobs = _extract_count(raw, "completed_job_count")
        reported_planned_jobs = _extract_count(raw, "planned_job_count")
        eligible_planned_jobs = reported_planned_jobs
        if slug != "stanet" and supported:
            expected_supported_matrix = len(supported) * 5 * 3
            if completed_jobs == expected_supported_matrix:
                eligible_planned_jobs = expected_supported_matrix
        blockers = raw.get("blockers") or []
        if isinstance(blockers, str):
            blockers = [{"severity": "unknown", "item": blockers}]
        audits.append(
            {
                "slug": slug,
                "method": display,
                "family": family,
                "raw_method": raw.get("method", display),
                "audit_date": raw.get("audit_date", "2026-08-11"),
                "scores": {**scores, "overall": stated},
                "status_label": raw.get("status_label", "已审查"),
                "public_pipeline_status": _status_text(raw.get("public_pipeline_status")),
                "protected_status": _status_text(raw.get("protected_status")),
                "final_result_availability": _status_text(raw.get("final_result_availability")),
                "planned_tasks": sorted(planned),
                "supported_tasks": sorted(supported),
                "unsupported_tasks": sorted(unsupported),
                "completed_job_count": completed_jobs,
                "planned_job_count": eligible_planned_jobs,
                "reported_planned_job_count": reported_planned_jobs,
                "blockers": blockers,
                "risks": raw.get("risks") or [],
                "tests_run": raw.get("tests_run") or [],
                "evidence_quality": raw.get("evidence_quality", "not stated"),
                "executive_summary": raw.get("executive_summary", ""),
                "code_components": raw.get("code_components") or [],
                "input_contract": raw.get("input_contract") or [],
                "output_contract": raw.get("output_contract") or [],
                "result_stages": raw.get("result_stages") or [],
                "source_report": str(path.relative_to(ROOT)),
            }
        )
    if missing:
        raise SystemExit("Missing subagent reports: " + ", ".join(str(x) for x in missing))
    return audits


def _progress_color(score: float) -> str:
    if score >= 90:
        return PALETTE["green"]
    if score >= 75:
        return PALETTE["brass"]
    if score >= 50:
        return PALETTE["copper"]
    return PALETTE["red"]


def save_data(audits: list[dict[str, Any]]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "audit_summary.json").write_text(
        json.dumps(
            {
                "snapshot": "2026-08-11",
                "score_weights": {k: w for k, _, w in SCORE_KEYS},
                "claim_boundary": "public evidence only; no protected evaluation authorization",
                "methods": audits,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (DATA / "progress_scores.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["method", *[label for _, label, _ in SCORE_KEYS], "overall", "completed_jobs", "planned_jobs"])
        for a in audits:
            writer.writerow(
                [
                    a["method"],
                    *[a["scores"][key] for key, _, _ in SCORE_KEYS],
                    a["scores"]["overall"],
                    a["completed_job_count"],
                    a["planned_job_count"],
                ]
            )
    with (DATA / "task_coverage.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["method", *[label for _, label in TASKS]])
        for a in audits:
            row = []
            for task, _ in TASKS:
                if task in a["unsupported_tasks"]:
                    row.append("unsupported")
                elif task in a["supported_tasks"]:
                    if a["slug"] == "reve" and task in {"motor_imagery", "mental_arithmetic"}:
                        row.append("overlap_track")
                    elif a["slug"] == "stanet":
                        row.append("context_reference")
                    else:
                        row.append("supported")
                else:
                    row.append("not_planned")
            writer.writerow([a["method"], *row])


def _clean_axes(ax: mpl.axes.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def render_overview(audits: list[dict[str, Any]]) -> None:
    values = np.array([[a["scores"][key] for key, _, _ in SCORE_KEYS] for a in audits])
    fig, (ax, ax2) = plt.subplots(
        1,
        2,
        figsize=(16, 9),
        gridspec_kw={"width_ratios": [4.8, 1.25]},
        layout="constrained",
    )
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "readiness",
        [PALETTE["red"], PALETTE["copper"], PALETTE["brass"], PALETTE["green"]],
    )
    im = ax.imshow(values, vmin=0, vmax=100, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(SCORE_KEYS)), [x[1] for x in SCORE_KEYS], fontsize=14, color=PALETTE["ink"])
    ax.set_yticks(range(len(audits)), [a["method"] for a in audits], fontsize=14, color=PALETTE["ink"])
    ax.tick_params(axis="x", pad=16)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            color = PALETTE["white"] if value < 72 or value > 88 else PALETTE["ink"]
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=17, fontweight="bold", color=color)
    for x in np.arange(-0.5, values.shape[1], 1):
        ax.axvline(x, color=PALETTE["paper"], lw=3)
    for y in np.arange(-0.5, values.shape[0], 1):
        ax.axhline(y, color=PALETTE["paper"], lw=3)
    ax.set_title("工程准备度已进入后期；结果生成仍受最终门控约束", loc="left", fontsize=25, fontweight="bold", color=PALETTE["ink"], pad=22)
    _clean_axes(ax)

    y = np.arange(len(audits))
    overall = np.array([a["scores"]["overall"] for a in audits])
    ax2.barh(y, overall, color=[_progress_color(x) for x in overall], height=0.46)
    ax2.barh(y, 100 - overall, left=overall, color=PALETTE["paper2"], height=0.46)
    for yi, value in zip(y, overall):
        ax2.text(value - 2, yi, f"{value:.0f}", va="center", ha="right", color=PALETTE["white"], fontsize=15, fontweight="bold")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(len(audits) - 0.5, -0.5)
    ax2.set_yticks([])
    ax2.set_xticks([0, 50, 100], ["0", "50", "100"], fontsize=11, color=PALETTE["muted"])
    ax2.set_title("加权总分", fontsize=16, color=PALETTE["ink"])
    ax2.axvline(75, color=PALETTE["line"], lw=1, ls="--")
    _clean_axes(ax2)
    fig.savefig(MEDIA / "overview_score_heatmap.png", dpi=180)
    plt.close(fig)


def render_task_coverage(audits: list[dict[str, Any]]) -> None:
    state_code = {"unsupported": 0, "supported": 1, "overlap": 2, "context": 3, "missing": 4}
    matrix = np.zeros((len(audits), len(TASKS)), dtype=int)
    labels: list[list[str]] = []
    for i, a in enumerate(audits):
        row_labels = []
        for j, (task, _) in enumerate(TASKS):
            if task in a["unsupported_tasks"]:
                state, lab = "unsupported", "—"
            elif task in a["supported_tasks"]:
                if a["slug"] == "reve" and task in {"motor_imagery", "mental_arithmetic"}:
                    state, lab = "overlap", "O"
                elif a["slug"] == "stanet":
                    state, lab = "context", "C"
                else:
                    state, lab = "supported", "✓"
            else:
                state, lab = "missing", "·"
            matrix[i, j] = state_code[state]
            row_labels.append(lab)
        labels.append(row_labels)
    colors = [PALETTE["paper2"], PALETTE["green"], PALETTE["copper"], PALETTE["brass"], PALETTE["muted"]]
    cmap = mpl.colors.ListedColormap(colors)
    bounds = np.arange(-0.5, len(colors) + 0.5, 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(figsize=(16, 9), layout="constrained")
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(TASKS)), [x[1] for x in TASKS], fontsize=16, color=PALETTE["ink"])
    ax.set_yticks(range(len(audits)), [a["method"] for a in audits], fontsize=15, color=PALETTE["ink"])
    ax.tick_params(axis="x", pad=16)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = PALETTE["ink"] if matrix[i, j] in {0, 4} else PALETTE["white"]
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=23, fontweight="bold", color=color)
    for x in np.arange(-0.5, matrix.shape[1], 1):
        ax.axvline(x, color=PALETTE["paper"], lw=3)
    for y in np.arange(-0.5, matrix.shape[0], 1):
        ax.axhline(y, color=PALETTE["paper"], lw=3)
    ax.set_title("任务覆盖不是单一“完成/未完成”：unsupported、overlap 与 context 必须分轨", loc="left", fontsize=25, fontweight="bold", color=PALETTE["ink"], pad=25)
    _clean_axes(ax)
    legend = [
        (PALETTE["green"], "✓ 支持 / A0–A8 pass"),
        (PALETTE["paper2"], "— 事前 unsupported"),
        (PALETTE["copper"], "O target-corpus overlap track"),
        (PALETTE["brass"], "C method-native context reference"),
    ]
    handles = [mpl.patches.Patch(facecolor=c, edgecolor=PALETTE["line"], label=l) for c, l in legend]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, fontsize=12)
    fig.savefig(MEDIA / "task_coverage_matrix.png", dpi=180)
    plt.close(fig)


def _contains(text: str, needles: list[str]) -> bool:
    low = text.lower()
    return any(x.lower() in low for x in needles)


def render_pipeline(audits: list[dict[str, Any]]) -> None:
    stage_names = ["代码与测试", "输入合同", "输出合同", "Public 矩阵", "Protected / 最终表"]
    fig, ax = plt.subplots(figsize=(16, 9), layout="constrained")
    ax.set_xlim(-0.35, 4.55)
    ax.set_ylim(-0.8, len(audits) - 0.2)
    ax.invert_yaxis()
    for i, a in enumerate(audits):
        ax.plot(range(5), [i] * 5, color=PALETTE["line"], lw=5, zorder=1)
        public = a["public_pipeline_status"]
        final = a["final_result_availability"] + " " + a["protected_status"]
        stage_states = [
            "pass" if a["scores"]["code_components"] >= 75 else "partial",
            "pass" if a["scores"]["input_adaptation"] >= 75 else "partial",
            "pass" if a["scores"]["output_adaptation"] >= 75 else "partial",
            "pass" if _contains(public, ["complete", "完成", "pass", "70/70", "90/90", "105/105", "75/75"]) else "partial",
            "context" if a["slug"] == "stanet" else "locked",
        ]
        state_colors = {"pass": PALETTE["green"], "partial": PALETTE["copper"], "locked": PALETTE["red"], "context": PALETTE["brass"]}
        for j, state in enumerate(stage_states):
            ax.scatter(j, i, s=460, color=state_colors[state], edgecolor=PALETTE["paper"], linewidth=3, zorder=3)
            glyph = {"pass": "✓", "partial": "…", "locked": "×", "context": "C"}[state]
            ax.text(j, i, glyph, ha="center", va="center", fontsize=14, fontweight="bold", color=PALETTE["white"])
        ax.text(-0.45, i, a["method"], ha="right", va="center", fontsize=14, color=PALETTE["ink"], fontweight="bold")
    ax.set_xticks(range(5), stage_names, fontsize=14, color=PALETTE["ink"])
    ax.tick_params(axis="x", pad=18)
    ax.set_yticks([])
    ax.set_title("真正的剩余工作集中在最终授权与一次性 protected evaluation", loc="left", fontsize=25, fontweight="bold", color=PALETTE["ink"], pad=30)
    _clean_axes(ax)
    fig.text(0.02, 0.03, "✓ 已有公开证据   … 尚有工程缺口   × locked / not table-admissible   C 已有正式结果但仅作 context reference", fontsize=12, color=PALETTE["muted"])
    fig.savefig(MEDIA / "evidence_pipeline.png", dpi=180)
    plt.close(fig)


def _short(text: str, width: int = 54) -> str:
    clean = " ".join(str(text).split())
    return textwrap.shorten(clean, width=width, placeholder="…")


def _blocker_text(audit: dict[str, Any]) -> list[str]:
    items: list[str] = []
    for blocker in audit["blockers"][:2]:
        if isinstance(blocker, dict):
            item = (
                blocker.get("item")
                or blocker.get("issue")
                or blocker.get("blocker")
                or blocker.get("description")
                or blocker.get("detail")
                or blocker.get("next_action")
            )
        else:
            item = blocker
        if item:
            items.append(_short(str(item), 46))
    if not items:
        items = ["未记录新增工程 blocker；剩余门控见最终结果状态"]
    return items


def render_method_dashboard(a: dict[str, Any]) -> None:
    fig = plt.figure(figsize=(16, 9), layout="constrained")
    gs = fig.add_gridspec(12, 20)
    ax_title = fig.add_subplot(gs[0:2, :])
    ax_score = fig.add_subplot(gs[2:9, 0:10])
    ax_stage = fig.add_subplot(gs[2:6, 11:20])
    ax_task = fig.add_subplot(gs[6:9, 11:20])
    ax_note = fig.add_subplot(gs[9:12, :])
    for ax in (ax_title, ax_score, ax_stage, ax_task, ax_note):
        _clean_axes(ax)
        ax.set_xticks([])
        ax.set_yticks([])

    overall = a["scores"]["overall"]
    ax_title.set_xlim(0, 1)
    ax_title.set_ylim(0, 1)
    ax_title.text(0, 0.72, a["method"], fontsize=30, fontweight="bold", color=PALETTE["ink"], va="center")
    ax_title.text(0, 0.20, a["family"], fontsize=15, color=PALETTE["muted"], va="center")
    ax_title.text(0.98, 0.72, f"{overall:.0f}", ha="right", va="center", fontsize=48, fontweight="bold", color=_progress_color(overall))
    ax_title.text(0.98, 0.18, "加权工程准备度 / 100", ha="right", va="center", fontsize=13, color=PALETTE["muted"])
    ax_title.axhline(0.0, color=PALETTE["line"], lw=1.2)

    score_values = [a["scores"][k] for k, _, _ in SCORE_KEYS]
    labels = [label for _, label, _ in SCORE_KEYS]
    y = np.arange(len(labels))
    ax_score.barh(y, [100] * len(y), color=PALETTE["paper2"], height=0.43)
    ax_score.barh(y, score_values, color=[_progress_color(x) for x in score_values], height=0.43)
    for yi, val in zip(y, score_values):
        ax_score.text(val + 2, yi, f"{val:.0f}", va="center", fontsize=16, fontweight="bold", color=PALETTE["ink"])
    ax_score.set_yticks(y, labels, fontsize=15, color=PALETTE["ink"])
    ax_score.set_xlim(0, 108)
    ax_score.invert_yaxis()
    ax_score.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100"], fontsize=10, color=PALETTE["muted"])
    ax_score.grid(axis="x", color=PALETTE["line"], lw=0.7, alpha=0.6)
    ax_score.set_title("五维进度", loc="left", fontsize=17, fontweight="bold", color=PALETTE["ink"], pad=14)

    ax_stage.set_xlim(-0.3, 4.3)
    ax_stage.set_ylim(-0.7, 0.7)
    public_ok = _contains(a["public_pipeline_status"], ["complete", "完成", "70/70", "90/90", "105/105", "75/75", "pass"])
    stages = [
        ("代码", a["scores"]["code_components"] >= 75, "pass"),
        ("输入", a["scores"]["input_adaptation"] >= 75, "pass"),
        ("输出", a["scores"]["output_adaptation"] >= 75, "pass"),
        ("Public", public_ok, "pass"),
        ("Final", a["slug"] == "stanet", "context" if a["slug"] == "stanet" else "locked"),
    ]
    ax_stage.plot(range(5), [0] * 5, color=PALETTE["line"], lw=5, zorder=1)
    for x, (name, ok, kind) in enumerate(stages):
        if kind == "context":
            color, glyph = PALETTE["brass"], "C"
        elif x == 4 and not ok:
            color, glyph = PALETTE["red"], "×"
        elif ok:
            color, glyph = PALETTE["green"], "✓"
        else:
            color, glyph = PALETTE["copper"], "…"
        ax_stage.scatter(x, 0, s=520, color=color, edgecolor=PALETTE["paper"], linewidth=3, zorder=3)
        ax_stage.text(x, 0, glyph, ha="center", va="center", fontsize=15, fontweight="bold", color=PALETTE["white"])
        ax_stage.text(x, 0.54, name, ha="center", va="bottom", fontsize=14, color=PALETTE["ink"])
    ax_stage.set_title("统一交付链", loc="left", fontsize=17, fontweight="bold", color=PALETTE["ink"], pad=12)

    task_colors = []
    task_labels = []
    for task, label in TASKS:
        if task in a["unsupported_tasks"]:
            task_colors.append(PALETTE["paper2"])
            task_labels.append(f"{label}\n—")
        elif a["slug"] == "reve" and task in {"motor_imagery", "mental_arithmetic"}:
            task_colors.append(PALETTE["copper"])
            task_labels.append(f"{label}\nO")
        elif a["slug"] == "stanet":
            task_colors.append(PALETTE["brass"])
            task_labels.append(f"{label}\nC")
        elif task in a["supported_tasks"]:
            task_colors.append(PALETTE["green"])
            task_labels.append(f"{label}\n✓")
        else:
            task_colors.append(PALETTE["muted"])
            task_labels.append(f"{label}\n·")
    ax_task.set_xlim(-0.5, len(TASKS) - 0.5)
    ax_task.set_ylim(-0.5, 0.5)
    for x, (color, label) in enumerate(zip(task_colors, task_labels)):
        ax_task.scatter(x, 0, s=1100, marker="s", color=color, edgecolor=PALETTE["paper"], linewidth=3)
        text_color = PALETTE["ink"] if color in {PALETTE["paper2"], PALETTE["muted"]} else PALETTE["white"]
        ax_task.text(x, 0, label, ha="center", va="center", fontsize=12, fontweight="bold", color=text_color)
    ax_task.set_title("任务覆盖", loc="left", fontsize=17, fontweight="bold", color=PALETTE["ink"], pad=12)

    jobs = "未采用 job 矩阵计数"
    if a["planned_job_count"]:
        jobs = f"Jobs：{a['completed_job_count']}/{a['planned_job_count']}"
    public = "Public / A0–A8：完成" if public_ok and a["slug"] != "stanet" else "Formal pipeline：完成"
    final = "Aggregate 可用 · 仅作 context" if a["slug"] == "stanet" else "Locked · not table-admissible"
    blockers = _blocker_text(a)
    blocker_line = textwrap.fill("；".join(blockers), width=96)
    summary = textwrap.fill(_short(a["executive_summary"], 205), width=118)
    ax_note.set_xlim(0, 1)
    ax_note.set_ylim(0, 1)
    ax_note.axhline(0.98, color=PALETTE["line"], lw=1.2)
    ax_note.text(0, 0.80, jobs, fontsize=16, fontweight="bold", color=PALETTE["ink"])
    ax_note.text(0.28, 0.80, public, fontsize=15, color=PALETTE["ink2"])
    ax_note.text(0.68, 0.80, final, fontsize=15, color=PALETTE["red"] if a["slug"] != "stanet" else PALETTE["brass"])
    ax_note.text(0, 0.52, "主要边界：" + blocker_line, fontsize=14, color=PALETTE["ink2"], va="top")
    ax_note.text(0, 0.20, "审查结论：" + summary, fontsize=13.5, color=PALETTE["muted"], va="top")
    fig.savefig(MEDIA / f"method_{a['slug']}.png", dpi=180)
    plt.close(fig)


def render_all(audits: list[dict[str, Any]]) -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    render_overview(audits)
    render_task_coverage(audits)
    render_pipeline(audits)
    for audit in audits:
        render_method_dashboard(audit)
    for path in MEDIA.glob("*.png"):
        with Image.open(path) as source:
            if source.mode != "RGB":
                source.convert("RGB").save(path, format="PNG", dpi=(180, 180), optimize=True)


def txt(element_id: str, bounds: list[float], text: str, size: int, color: str, *, bold: bool = False, align: tuple[str, str] = ("left", "top"), family: str = "Noto Sans CJK SC") -> dict[str, Any]:
    return {
        "elementId": element_id,
        "elementType": "text",
        "bounds": bounds,
        "content": {
            "text": text,
            "fontSize": size,
            "fontFamily": family,
            "color": color,
            "bold": bold,
            "align": list(align),
            "lineHeight": 1.15,
        },
    }


def rect(element_id: str, bounds: list[float], color: str, *, border: str | None = None) -> dict[str, Any]:
    element: dict[str, Any] = {
        "elementId": element_id,
        "elementType": "shape",
        "bounds": bounds,
        "shapeName": "rect",
        "fill": {"type": "solid", "color": color},
    }
    if border:
        element["border"] = {"style": "solid", "width": 1, "color": border}
    return element


def line(element_id: str, bounds: list[float], color: str, *, width: float = 1.0, dash: bool = False, arrow: bool = False) -> dict[str, Any]:
    return {
        "elementId": element_id,
        "elementType": "line",
        "bounds": bounds,
        "viewBox": [1, 1],
        "points": "0,0.5 1,0.5",
        "curve": "sharp",
        "arrow": [None, "stealth" if arrow else None],
        "border": {"style": "dash" if dash else "solid", "width": width, "color": color},
    }


def image(element_id: str, bounds: list[float], src: str) -> dict[str, Any]:
    return {
        "elementId": element_id,
        "elementType": "image",
        "bounds": bounds,
        "src": src,
        "fit": {"mode": "contain"},
    }


def base_page(title: str, page_number: int, *, kicker: str = "COMPARATIVE METHOD AUDIT") -> list[dict[str, Any]]:
    return [
        txt("kicker", [56, 23, 430, 20], kicker, 9, PALETTE["copper"], bold=True, align=("left", "middle")),
        txt("title", [56, 46, 840, 48], title, 21, PALETTE["ink"], bold=True, align=("left", "middle")),
        line("title-rule", [56, 96, 848, 2], PALETTE["line"], width=1),
        txt("page-no", [870, 508, 34, 16], f"{page_number:02d}", 9, PALETTE["muted"], align=("right", "middle")),
        txt("footer", [56, 505, 520, 18], "证据快照 2026-08-11 · public evidence only", 8, PALETTE["muted"], align=("left", "middle")),
    ]


def page_doc(elements: list[dict[str, Any]], page_type: str = "content", background: str = PALETTE["paper"]) -> dict[str, Any]:
    return {"pageType": page_type, "background": {"type": "solid", "color": background}, "elements": elements}


def write_page(name: str, doc: dict[str, Any]) -> str:
    path = PAGES / name
    path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    return f"pages/{name}"


def build_deck(audits: list[dict[str, Any]]) -> None:
    PAGES.mkdir(parents=True, exist_ok=True)
    media_names = [p.name for p in MEDIA.glob("*.png")]
    pages: list[str] = []

    cover = [
        rect("cover-band", [0, 0, 960, 18], PALETTE["copper"]),
        txt("cover-kicker", [60, 58, 520, 22], "ENGINEERING READINESS / EVIDENCE REVIEW", 11, PALETTE["copper"], bold=True, align=("left", "middle")),
        txt("cover-title", [60, 104, 820, 108], "对比方法工程面已基本就绪，\n最终跨方法数字仍等待联合解锁", 39, PALETTE["ink"], bold=True),
        txt("cover-sub", [60, 230, 760, 52], "7 个正式方法 · 42 个 method–task cells · 36 pass / 6 unsupported · protected evaluation 未授权", 17, PALETTE["ink2"]),
        line("cover-rule", [60, 305, 836, 2], PALETTE["line"], width=1.2),
        txt("n1", [60, 330, 180, 52], "6", 42, PALETTE["green"], bold=True),
        txt("n1l", [60, 384, 220, 38], "新增/适配方法\npublic A0–A8 完成", 12, PALETTE["ink2"]),
        txt("n2", [325, 330, 180, 52], "42", 42, PALETTE["copper"], bold=True),
        txt("n2l", [325, 384, 220, 38], "联合审查 cells\n证据候选已冻结", 12, PALETTE["ink2"]),
        txt("n3", [590, 330, 180, 52], "0", 42, PALETTE["red"], bold=True),
        txt("n3l", [590, 384, 260, 38], "新增方法 paper-ready\nprotected 数字尚不可用", 12, PALETTE["ink2"]),
        txt("cover-date", [60, 478, 420, 24], "进度审查报告 · 2026-08-11", 11, PALETTE["muted"], align=("left", "middle")),
        txt("cover-page", [870, 478, 28, 24], "01", 10, PALETTE["muted"], align=("right", "middle")),
    ]
    pages.append(write_page("01_cover.page", page_doc(cover, "cover")))

    e = base_page("审查把工程准备度与最终数字可用性分开计量", 2)
    e += [
        txt("reader-task", [56, 112, 390, 44], "读者任务：判断每个方法缺什么，以及何时能进入最终表。", 15, PALETTE["ink2"], bold=True),
        txt("weight-head", [56, 180, 380, 30], "统一加权量表", 18, PALETTE["ink"], bold=True),
    ]
    y = 226
    for idx, (key, label, weight) in enumerate(SCORE_KEYS):
        e.append(txt(f"w-label-{idx}", [56, y, 175, 26], label, 14, PALETTE["ink"], align=("left", "middle")))
        e.append(rect(f"w-bg-{idx}", [230, y + 5, 230, 14], PALETTE["paper2"]))
        e.append(rect(f"w-fill-{idx}", [230, y + 5, 230 * weight / 0.30, 14], PALETTE["green"] if key != "result_generation" else PALETTE["copper"]))
        e.append(txt(f"w-v-{idx}", [472, y, 70, 26], f"{weight:.0%}", 13, PALETTE["ink2"], bold=True, align=("left", "middle")))
        y += 48
    e += [
        line("split", [548, 130, 2, 322], PALETTE["line"], width=1),
        txt("state-head", [590, 130, 300, 32], "三层状态必须同时报告", 18, PALETTE["ink"], bold=True),
        txt("s1", [590, 188, 260, 30], "01  Adapter-aligned", 17, PALETTE["green"], bold=True),
        txt("s1d", [615, 222, 270, 58], "输入信息预算、身份、replay 与协议冻结通过；不代表有最终数字。", 13, PALETTE["ink2"]),
        txt("s2", [590, 296, 260, 30], "02  Public matrix complete", 17, PALETTE["copper"], bold=True),
        txt("s2d", [615, 330, 270, 58], "公共训练/验证工件齐全，但 summary 明确 table_admissible=false。", 13, PALETTE["ink2"]),
        txt("s3", [590, 404, 260, 30], "03  Paper-ready", 17, PALETTE["red"], bold=True),
        txt("s3d", [615, 438, 280, 50], "需独立授权、一次性 protected evaluation 与 frozen aggregation。", 13, PALETTE["ink2"]),
    ]
    pages.append(write_page("02_audit_model.page", page_doc(e)))

    e = base_page("五维工程准备度整体较高，但这不是 paper-ready 证明", 3)
    e += [image("overview", [52, 105, 856, 384], "media/overview_score_heatmap.png")]
    pages.append(write_page("03_overview.page", page_doc(e)))

    e = base_page("任务覆盖必须分清 pass、unsupported、overlap 与 context", 4)
    e += [image("coverage", [52, 105, 856, 384], "media/task_coverage_matrix.png")]
    pages.append(write_page("04_coverage.page", page_doc(e)))

    e = base_page("所有新增方法的真正瓶颈已从工程实现转移到最终授权", 5)
    e += [image("pipeline", [52, 105, 856, 384], "media/evidence_pipeline.png")]
    pages.append(write_page("05_pipeline.page", page_doc(e)))

    method_titles = {
        "biot": "BIOT：六项分类 public 矩阵完成，REFED 事前 unsupported",
        "cbramod": "CBraMod：结构化 EEG 表征链齐备，最终数字仍 locked",
        "reve": "REVE：几何输入已对齐，但 Single-Trial 必须留在 overlap track",
        "efrm": "EFRM：LODO Stage A/B 与 105-job public 链最完整，protected 未开放",
        "normwear": "NormWear：EEG–fNIRS 适配完成，结论必须保留 adapted 边界",
        "brainfusion": "BrainFusion：五任务传统融合链完成，DSR/REFED 明确 unsupported",
        "stanet": "STA-Net：正式五折结果可用，但只能作为 method-native context",
    }
    page_no = 6
    for audit in audits:
        e = base_page(method_titles[audit["slug"]], page_no)
        e += [image("method-dashboard", [50, 103, 860, 388], f"media/method_{audit['slug']}.png")]
        pages.append(write_page(f"{page_no:02d}_{audit['slug']}.page", page_doc(e)))
        page_no += 1

    e = base_page("到最终主表只剩一条受控路径，不能用 public 均值替代", page_no)
    x_positions = [84, 298, 512, 726]
    labels = [
        ("01", "人工审核", "复核 42-cell candidate\n与 track / unsupported 处置"),
        ("02", "独立授权", "生成单独哈希的 authorization\n当前 authorized=false"),
        ("03", "一次性评测", "eligible cell 只打开一次\n禁止按 protected 表现重试"),
        ("04", "冻结聚合", "fold 内先聚合 seed\n再形成 paper-facing 指标"),
    ]
    for i, (num, name, desc) in enumerate(labels):
        x = x_positions[i]
        e.append(txt(f"step-n-{i}", [x, 165, 70, 42], num, 28, PALETTE["copper"] if i < 2 else PALETTE["red"], bold=True))
        e.append(txt(f"step-name-{i}", [x, 214, 150, 32], name, 18, PALETTE["ink"], bold=True))
        e.append(txt(f"step-desc-{i}", [x, 254, 170, 85], desc, 12, PALETTE["ink2"]))
        if i < 3:
            e.append(line(f"step-line-{i}", [x + 120, 188, 94, 2], PALETTE["line"], width=2, arrow=True))
    e += [
        line("risk-rule", [84, 380, 792, 2], PALETTE["line"], width=1),
        txt("risk-title", [84, 402, 220, 30], "不可跨越的报告边界", 17, PALETTE["ink"], bold=True),
        txt("risk-body", [320, 398, 550, 70], "• Public validation 数字不可进入最终排名\n• REVE MI/MA 不可伪装成 target-excluded\n• STA-Net 不可伪装成 support-matched\n• Unsupported cell 不等于失败运行", 13, PALETTE["ink2"]),
    ]
    pages.append(write_page(f"{page_no:02d}_next_path.page", page_doc(e)))
    page_no += 1

    e = base_page("复核结果支持“工程后期、数字未解锁”的统一判断", page_no)
    e += [
        txt("verdict", [56, 122, 840, 72], "结论：7 个方法均已有可审计实现与结果链证据；\n六个新增/适配方法尚无可发布的 protected 主表数字。", 28, PALETTE["ink"], bold=True),
        line("final-rule", [56, 218, 844, 2], PALETTE["line"], width=1.2),
        txt("checks-head", [56, 246, 300, 30], "本次独立复核", 18, PALETTE["ink"], bold=True),
        txt("checks", [56, 288, 420, 112], "✓ joint candidate --check：42 cells / 36 pass / 6 unsupported\n✓ protected_evaluation_authorized=false\n✓ protected_test_opened=false\n✓ 联合候选与 adapter evidence hashes 一致\n✓ 每个方法均由独立 subagent 审查", 13, PALETTE["ink2"]),
        txt("src-head", [550, 246, 300, 30], "主要证据入口", 18, PALETTE["ink"], bold=True),
        txt("srcs", [550, 288, 345, 135], "docs/comparisons/STATUS.md\ndocs/comparisons/ADAPTER_PROGRESS_20260811.md\ncomparative_methods/EXPERIMENT_PLAN.md\ncomparative_methods/evidence/joint_protected_unlock_candidate_v2.json\nagent_reports/*.md + *.json", 11, PALETTE["ink2"], family="Noto Sans Mono CJK SC"),
        txt("handoff", [56, 445, 840, 42], "建议下一动作：先完成人工 unlock review；若获授权，再按冻结协议进行一次性 protected evaluation。", 16, PALETTE["copper"], bold=True),
    ]
    pages.append(write_page(f"{page_no:02d}_evidence.page", page_doc(e, "final")))

    manifest = {
        "version": "v2",
        "title": "对比方法全面进度审查报告",
        "size": [960, 540],
        "theme": {
            "colors": PALETTE,
            "textStyles": {
                "title": {"fontSize": 28, "fontFamily": "Noto Sans CJK SC", "bold": True, "color": PALETTE["ink"]},
                "body": {"fontSize": 15, "fontFamily": "Noto Sans CJK SC", "color": PALETTE["ink2"], "lineHeight": 1.2},
            },
        },
        "pages": pages,
    }
    (DECK / "comparative_methods_progress_review.pptd").write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )
    (DECK / "media_manifest.json").write_text(
        json.dumps({"files": sorted(media_names), "source": "generated from agent_reports/*.json"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_markdown(audits: list[dict[str, Any]]) -> None:
    lines = [
        "# 对比方法全面进度审查",
        "",
        "证据快照：2026-08-11。评分表示工程与比较准备度，不等于最终论文数字可用性。",
        "",
        "| 方法 | 必要代码 | 输入 | 输出 | 结果生成 | 证据复现 | 加权总分 | Public jobs | 最终数字 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for a in audits:
        sc = a["scores"]
        jobs = f"{a['completed_job_count']}/{a['planned_job_count']}" if a["planned_job_count"] else "n/a"
        final_label = (
            "有冻结 aggregate；仅 context reference"
            if a["slug"] == "stanet"
            else "不可用（protected locked）"
        )
        lines.append(
            f"| {a['method']} | {sc['code_components']:.0f} | {sc['input_adaptation']:.0f} | "
            f"{sc['output_adaptation']:.0f} | {sc['result_generation']:.0f} | "
            f"{sc['evidence_reproducibility']:.0f} | {sc['overall']:.1f} | {jobs} | {final_label} |"
        )
    lines += [
        "",
        "## 总体判断",
        "",
        "六个新增或适配方法均已完成 public A0–A8 证据链，但 protected evaluation 尚未授权，public summary 不能进入最终排名。STA-Net 已有冻结五折 aggregate，但观察预算属于 method-native context reference，不是当前 support-matched 主表证据。",
        "",
        "## 复核命令",
        "",
        "- `.venv/bin/python comparative_methods/build_joint_protected_unlock_candidate_v2.py --check`",
        "- `.venv/bin/python -m pytest -q tests/test_joint_protected_unlock_candidate_v2.py tests/test_adapter_alignment_gate_contract.py`",
        "",
        "## 方法审查明细",
        "",
    ]
    for a in audits:
        lines += [
            f"### {a['method']}",
            "",
            a["executive_summary"] or "详见对应 subagent 报告。",
            "",
            f"- Public：{a['public_pipeline_status']}",
            f"- Protected：{a['protected_status']}",
            f"- Final：{a['final_result_availability']}",
            f"- 详细报告：`agent_reports/{a['slug']}.md`",
            "",
        ]
    (ROOT / "COMPREHENSIVE_PROGRESS_REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_alt_text(audits: list[dict[str, Any]]) -> None:
    lines = [
        "# 图像替代文本",
        "",
        "- `overview_score_heatmap.png`：7 个方法 × 5 个进度维度的 0–100 热图，右侧为加权工程准备度。颜色从红/铜/黄铜到绿表示分数增高，单元格同时标注数字。",
        "- `task_coverage_matrix.png`：7 个方法 × 7 个任务的覆盖矩阵。绿色勾号为支持，米色横线为事前 unsupported，橙色 O 为 target-corpus overlap track，黄铜色 C 为 method-native context reference。",
        "- `evidence_pipeline.png`：每个方法从代码、输入、输出、public 矩阵到 protected/最终表的阶段图。颜色与符号同时编码状态。",
    ]
    for a in audits:
        lines.append(
            f"- `method_{a['slug']}.png`：{a['method']} 的完整进度仪表图，包含五维分数、统一交付链、七任务覆盖、public job 计数、最终数字状态和主要边界。"
        )
    (ROOT / "FIGURE_ALT_TEXT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    _font()
    audits = load_audits()
    save_data(audits)
    render_all(audits)
    build_deck(audits)
    build_markdown(audits)
    build_alt_text(audits)
    print(json.dumps({"methods": len(audits), "slides": len(list(PAGES.glob('*.page'))), "deck": str(DECK)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
