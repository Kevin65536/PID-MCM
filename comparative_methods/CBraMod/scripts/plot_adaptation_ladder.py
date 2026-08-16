#!/usr/bin/env python3
"""Create an auditable figure set for the CBraMod public pilot.

The source ``report.json`` is treated as immutable input.  The script writes
two matched figures (epoch trajectory and final-score dot plot), a CSV data
table, alt text, and a provenance manifest.  It never selects the best epoch:
the displayed final point is always the fixed-budget final epoch.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = REPO_ROOT / "runs/performance_analysis/20260816_p0/cbramod_ladder/report.json"
DEFAULT_OUTPUT = REPO_ROOT / "runs/performance_analysis/20260816_p0/cbramod_ladder"

CAPACITY_STYLE: dict[str, dict[str, Any]] = {
    "frozen_linear": {"label": "Frozen linear", "color": "#0072B2", "linestyle": "-", "marker": "o"},
    "frozen_mlp": {"label": "Frozen MLP", "color": "#E69F00", "linestyle": "--", "marker": "s"},
    "last_block_linear": {"label": "Last block + linear", "color": "#009E73", "linestyle": "-.", "marker": "^"},
    "full_finetune_linear": {"label": "Full fine-tune + linear", "color": "#D55E00", "linestyle": ":", "marker": "D"},
    "random_linear": {"label": "Random linear", "color": "#CC79A7", "linestyle": (0, (5, 2)), "marker": "X"},
    "random_mlp": {"label": "Random MLP", "color": "#56B4E9", "linestyle": (0, (3, 1, 1, 1)), "marker": "P"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def load_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"pilot report is missing: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "completed" or report.get("protected_test_opened") is not False:
        raise ValueError("figure input must be a completed public-only pilot report")
    capacities = tuple(str(row["capacity"]) for row in report.get("results", []))
    if capacities != tuple(CAPACITY_STYLE):
        raise ValueError(f"report capacity order differs from declared order: {capacities}")
    if report.get("claim_boundary") != "public_development_pilot_not_protected_or_final_table_evidence":
        raise ValueError("figure input has an unexpected claim boundary")
    return report


def trajectory_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in report["results"]:
        capacity = str(result["capacity"])
        history = result["report"].get("history", [])
        if not history:
            raise ValueError(f"capacity has no epoch history: {capacity}")
        for item in history:
            metric = item["validation"]["macro_f1"]
            rows.append({"capacity": capacity, "epoch": int(item["epoch"]), "validation_macro_f1": float(metric), "is_final": int(item["epoch"]) == len(history)})
    return rows


def final_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in report["results"]:
        history = result["report"].get("history", [])
        rows.append({
            "capacity": str(result["capacity"]),
            "epoch": int(history[-1]["epoch"]),
            "final_validation_macro_f1": float(result["validation_metrics"]["macro_f1"]),
            "accuracy": float(result["validation_metrics"]["accuracy"]),
        })
    return rows


def write_data_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("capacity", "epoch", "validation_macro_f1", "is_final")
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row[column]) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def title_suffix(report: Mapping[str, Any]) -> str:
    return (
        f"{report['task']} / outer fold {report['outer_fold']} · "
        f"64/class train + 64/class validation · fixed 5 epochs · no validation selection"
    )


def add_header(fig: plt.Figure, report: Mapping[str, Any], *, final: bool) -> None:
    main = "CBraMod adaptation ladder: final fixed-budget scores" if final else "CBraMod adaptation ladder: epoch trajectory"
    final_note = "Epoch 5 shown for every capacity" if final else "Open markers = final fixed-budget epoch"
    fig.suptitle(main, fontsize=13, y=0.975)
    fig.text(
        0.5,
        0.925,
        f"single-fold public pilot · {report['task']} / outer fold {report['outer_fold']}",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#444444",
    )
    fig.text(
        0.5,
        0.895,
        "64/class train + 64/class validation · fixed 5 epochs",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#444444",
    )
    fig.text(
        0.5,
        0.865,
        f"{final_note} · validation not used for selection",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#444444",
    )


def style_axes(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0], [0],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label=style["label"],
        )
        for style in CAPACITY_STYLE.values()
    ]


def make_trajectory(report: Mapping[str, Any], rows: list[dict[str, Any]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.4, 5.3), layout=None)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.28, top=0.79)
    add_header(fig, report, final=False)
    for capacity, style in CAPACITY_STYLE.items():
        subset = [row for row in rows if row["capacity"] == capacity]
        epochs = np.asarray([row["epoch"] for row in subset], dtype=int)
        values = np.asarray([row["validation_macro_f1"] for row in subset], dtype=float)
        ax.plot(
            epochs,
            values,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=5.5,
            linewidth=1.6,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=style["label"],
        )
        # An open marker denotes the fixed-budget final epoch.  It is not a
        # best-epoch highlight and no point is selected from validation.
        ax.scatter(
            [epochs[-1]],
            [values[-1]],
            s=68,
            marker=style["marker"],
            facecolors="white",
            edgecolors=style["color"],
            linewidths=1.5,
            zorder=5,
        )
    ax.axhline(1 / 3, color="#555555", linestyle=(0, (2, 2)), linewidth=1.0, label="Majority-class macro-F1 = 0.333")
    ax.set(xlabel="Fixed training epoch", ylabel="Validation macro-F1")
    ax.set_xticks(sorted({row["epoch"] for row in rows}))
    ax.set_xlim(0.8, max(row["epoch"] for row in rows) + 0.2)
    ax.set_ylim(0.30, 0.66)
    style_axes(ax)
    ax.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False, fontsize=8.5, columnspacing=1.2, handlelength=2.5)
    return fig


def make_final_dotplot(report: Mapping[str, Any], rows: list[dict[str, Any]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.4, 5.3), layout=None)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.38, top=0.79)
    add_header(fig, report, final=True)
    x = np.arange(len(rows), dtype=float)
    for position, row in zip(x, rows, strict=True):
        style = CAPACITY_STYLE[row["capacity"]]
        ax.scatter(
            position,
            row["final_validation_macro_f1"],
            s=80,
            color=style["color"],
            marker=style["marker"],
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        ax.scatter(
            position,
            row["final_validation_macro_f1"],
            s=112,
            facecolors="none",
            edgecolors=style["color"],
            marker=style["marker"],
            linewidths=1.4,
            zorder=5,
        )
        ax.annotate(
            f"{row['final_validation_macro_f1']:.3f}",
            (position, row["final_validation_macro_f1"]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#222222",
        )
    ax.axhline(1 / 3, color="#555555", linestyle=(0, (2, 2)), linewidth=1.0)
    ax.text(
        0.98,
        1 / 3 + 0.006,
        "majority-class macro-F1 = 0.333",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    ax.set(xlabel="Adaptation capacity", ylabel="Final validation macro-F1")
    ax.set_xticks(x, [CAPACITY_STYLE[row["capacity"]]["label"].replace(" + ", "\n+ ") for row in rows], rotation=0)
    ax.set_ylim(0.30, 0.66)
    style_axes(ax)
    ax.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3, frameon=False, fontsize=8.5, columnspacing=1.2, handlelength=2.5)
    return fig


def alt_text(report: Mapping[str, Any], final: list[dict[str, Any]]) -> str:
    scores = "; ".join(f"{CAPACITY_STYLE[row['capacity']]['label']} {row['final_validation_macro_f1']:.3f}" for row in final)
    return (
        f"Two-panel figure for the CBraMod {report['task']} outer-fold {report['outer_fold']} public pilot. "
        "The left panel shows validation macro-F1 over five fixed training epochs for six adaptation capacities; "
        "color, line style, and marker shape redundantly identify each capacity, and open markers denote the fixed final epoch. "
        "The right panel shows the corresponding final epoch scores as labeled dots. "
        f"Final macro-F1 scores are: {scores}. "
        "The dashed horizontal reference is the majority-class macro-F1 of 0.333. "
        "This is a single-fold public pilot; validation was recorded descriptively and was not used for model or epoch selection."
    )


def export_figure(fig: plt.Figure, path: Path) -> dict[str, Any]:
    png = path.with_suffix(".png")
    pdf = path.with_suffix(".pdf")
    if png.exists() or pdf.exists():
        raise FileExistsError(f"refusing to overwrite figure outputs for {path}")
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return {
        "png": {"path": portable(png), "sha256": sha256_file(png), "bytes": png.stat().st_size, "dpi": 300},
        "pdf": {"path": portable(pdf), "sha256": sha256_file(pdf), "bytes": pdf.stat().st_size},
    }


def run(report_path: Path, output_dir: Path) -> dict[str, Any]:
    report = load_report(report_path.resolve())
    output_dir = output_dir.resolve()
    if "protected" in {part.lower() for part in output_dir.parts}:
        raise PermissionError(f"refusing protected figure output path: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories = trajectory_rows(report)
    finals = final_rows(report)
    data_path = output_dir / "adaptation_ladder_figure_data.csv"
    if data_path.exists():
        raise FileExistsError(f"refusing to overwrite figure data: {data_path}")
    write_data_csv(data_path, trajectories)
    alt_path = output_dir / "adaptation_ladder_alt_text.md"
    if alt_path.exists():
        raise FileExistsError(f"refusing to overwrite alt text: {alt_path}")
    alt_path.write_text("# Alt text\n\n" + alt_text(report, finals) + "\n", encoding="utf-8")
    trajectory_files = export_figure(make_trajectory(report, trajectories), output_dir / "adaptation_ladder_epoch_trajectory")
    final_files = export_figure(make_final_dotplot(report, finals), output_dir / "adaptation_ladder_final_scores")
    manifest = {
        "schema": "cbramod_adaptation_ladder_figure_manifest_v1",
        "status": "complete",
        "created_at": utc_now(),
        "figure_type": "static_scientific_diagnostic",
        "audience": "manuscript/supplement diagnostic; publisher requirements pending",
        "source_report": {"path": portable(report_path), "sha256": sha256_file(report_path)},
        "source_contract": {
            "task": report["task"],
            "outer_fold": report["outer_fold"],
            "train_sample_count": report["train_sample_count"],
            "validation_sample_count": report["validation_sample_count"],
            "fixed_budget": report["fixed_budget"],
            "validation_used_for_selection": False,
            "protected_test_opened": report["protected_test_opened"],
            "claim_boundary": report["claim_boundary"],
        },
        "transformations": [
            "read epoch-level validation macro-F1 from report.json",
            "plot final fixed-budget epoch; no epoch or capacity selection",
            "use majority-class macro-F1=1/3 as a declared reference line",
        ],
        "palette": {
            "name": "Okabe-Ito-inspired redundant qualitative palette",
            "color_not_sufficient": True,
            "redundant_encodings": ["line_style", "marker_shape", "legend_label"],
            "styles": CAPACITY_STYLE,
        },
        "data_table": {"path": portable(data_path), "sha256": sha256_file(data_path)},
        "alt_text": {"path": portable(alt_path), "sha256": sha256_file(alt_path), "text": alt_text(report, finals)},
        "figures": {
            "epoch_trajectory": trajectory_files,
            "final_scores": final_files,
        },
        "render": {
            "matplotlib": matplotlib.__version__,
            "python": platform.python_version(),
            "backend": matplotlib.get_backend(),
            "figsize_inches": {"epoch_trajectory": [7.4, 5.3], "final_scores": [7.4, 5.3]},
            "font_family": matplotlib.rcParams.get("font.family"),
            "pdf_fonttype": matplotlib.rcParams.get("pdf.fonttype"),
            "background": "opaque white",
        },
        "manual_review": {
            "layout_engine": "manual_subplots_adjust",
            "overlap_check": "manual_view_image_completed",
            "reviewed_files": [
                "adaptation_ladder_epoch_trajectory.png",
                "adaptation_ladder_final_scores.png",
                "adaptation_ladder_epoch_trajectory.pdf",
                "adaptation_ladder_final_scores.pdf",
            ],
            "notes": "No legend, annotation, or axis-label overlap observed at rendered PNG size; open final markers are not best-epoch highlights.",
        },
        "protected_test_opened": False,
    }
    manifest_path = output_dir / "figure_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite figure manifest: {manifest_path}")
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args.report, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
