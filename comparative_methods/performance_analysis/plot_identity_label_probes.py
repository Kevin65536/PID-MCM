#!/usr/bin/env python3
"""Plot the public motor-imagery identity/label probe summary.

The input is the tidy ``metrics.csv`` emitted by
``identity_label_probes.py``.  The plot deliberately keeps task/session
probes (strict subject-group holdout) separate from the closed-set subject-ID
row-split diagnostic.  Chance is encoded as short local segments so that the
different subject-class counts (23 versus 29) are not silently treated as one
common baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ("biot", "cbramod", "reve", "normwear")
METHOD_LABELS = {"biot": "BIOT", "cbramod": "CBraMod", "reve": "REVE", "normwear": "NormWear"}
PROBE_ORDER = ("task", "session", "subject_closed_set")
PROBE_LABELS = {
    "task": "Task label\n(strict subject holdout)",
    "session": "Session label\n(strict subject holdout)",
    "subject_closed_set": "Subject ID\n(closed-set row split)",
}
PROBE_CHANCE = {"task": 0.5, "session": 1.0 / 3.0}
COLORS = {"biot": "#0072B2", "cbramod": "#E69F00", "reve": "#009E73", "normwear": "#D55E00"}
MARKERS = {"biot": "o", "cbramod": "s", "reve": "^", "normwear": "D"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _subject_chance(metrics: Sequence[Mapping[str, str]], method: str) -> float:
    counts = [int(row["n_test_subjects"]) for row in metrics if row["method"] == method and row["probe"] == "subject_closed_set"]
    # n_test_subjects is not the number of target classes.  The cache's target
    # class count is included in capabilities, but each subject has equal rows;
    # derive the class count from the rows' train/test subject union when only
    # metrics.csv is available.
    # The MI public caches have 23 subject classes for BIOT/CBraMod/REVE and 29
    # for NormWear.  Infer from n_train/n_test proportions is not valid, so the
    # caller supplies the audited class count from the manifest if available.
    del counts
    return 1.0 / (29.0 if method == "normwear" else 23.0)


def aggregate(metrics: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    """Aggregate macro-F1 by method/probe; SD is the CV/repeat spread."""

    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in metrics:
        if row.get("task") != "motor_imagery":
            continue
        if row.get("probe") not in PROBE_ORDER:
            continue
        groups[(row["method"], row["probe"])].append(float(row["macro_f1"]))
    result: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        for probe in PROBE_ORDER:
            values = np.asarray(groups.get((method, probe), []), dtype=float)
            if values.size == 0:
                continue
            result.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "probe": probe,
                    "probe_label": PROBE_LABELS[probe].replace("\n", " "),
                    "metric": "macro_f1",
                    "n": int(values.size),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "chance": PROBE_CHANCE.get(probe, 1.0 / (29.0 if method == "normwear" else 23.0)),
                    "uncertainty": "SD across GroupKFold folds or repeated row splits; descriptive, not an independent-subject CI",
                }
            )
    return result


def make_figure(summary: Sequence[Mapping[str, Any]]) -> mpl.figure.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    # Explicit margins leave room for the figure title, legend, and rotated
    # method labels at the intended single-column print size.
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.24, top=0.78, wspace=0.16)
    by_key = {(str(row["method"]), str(row["probe"])): row for row in summary}
    x = np.arange(len(METHOD_ORDER), dtype=float)
    for panel, probe in zip(axes, PROBE_ORDER):
        panel.set_title(PROBE_LABELS[probe], pad=5)
        panel.set_ylim(0.0, 1.0)
        panel.set_xlim(-0.55, len(METHOD_ORDER) - 0.45)
        panel.set_xticks(x, [METHOD_LABELS[m] for m in METHOD_ORDER], rotation=20, ha="right")
        panel.set_yticks(np.linspace(0, 1, 6))
        panel.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        panel.set_axisbelow(True)
        panel.spines["top"].set_visible(False)
        panel.spines["right"].set_visible(False)
        for index, method in enumerate(METHOD_ORDER):
            row = by_key.get((method, probe))
            if row is None:
                panel.text(index, 0.5, "NA", ha="center", va="center", color="#666666")
                continue
            panel.errorbar(
                index,
                float(row["mean"]),
                yerr=float(row["sd"]),
                fmt=MARKERS[method],
                markersize=5.5,
                markerfacecolor=COLORS[method],
                markeredgecolor="#222222",
                markeredgewidth=0.45,
                ecolor=COLORS[method],
                elinewidth=1.1,
                capsize=2.5,
                zorder=3,
                label=METHOD_LABELS[method],
            )
            chance = float(row["chance"])
            panel.plot(
                [index - 0.25, index + 0.25],
                [chance, chance],
                color="#555555",
                linestyle=(0, (2, 2)),
                linewidth=1.0,
                zorder=2,
            )
    axes[0].set_ylabel("Macro-F1")
    handles = [
        mpl.lines.Line2D(
            [],
            [],
            color=COLORS[m],
            marker=MARKERS[m],
            linestyle="None",
            markerfacecolor=COLORS[m],
            markeredgecolor="#222222",
            markeredgewidth=0.45,
            markersize=5.5,
            label=METHOD_LABELS[m],
        )
        for m in METHOD_ORDER
    ]
    handles.append(
        mpl.lines.Line2D(
            [],
            [],
            color="#555555",
            linestyle=(0, (2, 2)),
            linewidth=1.0,
            label="local chance",
        )
    )
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.94), columnspacing=1.8)
    fig.suptitle("Public motor-imagery representation probes", y=0.995, fontsize=13, fontweight="bold")
    return fig


def _write_manifest(output: Path, source_csv: Path, summary: Sequence[Mapping[str, Any]]) -> None:
    manifest = {
        "schema": "identity_label_probe_figure_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_data": str(source_csv),
        "source_transformations": [
            "retain task=motor_imagery",
            "retain probe in task/session/subject_closed_set",
            "aggregate macro-F1 by method and probe",
            "error bars are SD across existing folds/repeated row splits",
            "chance for task=1/2; session=1/3; subject_closed_set=1/K with K=23 for BIOT/CBraMod/REVE and K=29 for NormWear",
        ],
        "methods": list(METHOD_ORDER),
        "probes": list(PROBE_ORDER),
        "encoding": {
            "position": "mean macro-F1",
            "error_bar": "descriptive SD",
            "color_and_marker": "redundant method encoding",
            "chance": "short dotted local segments",
        },
        "export": {
            "figure_size_in": [13.0, 4.5],
            "raster_dpi": 600,
            "formats": ["png", "pdf"],
            "bbox_inches": "tight",
            "background": "opaque white",
        },
        "summary_rows": list(summary),
        "protected_data": "not used; source metrics are public-only feature-cache probes",
        "caveat": "subject_closed_set is a row-split upper-bound identity-retention diagnostic and does not exclude record/session overlap",
    }
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def run(source_csv: Path, output: Path) -> list[dict[str, Any]]:
    metrics = _read_csv(source_csv)
    summary = aggregate(metrics)
    output.mkdir(parents=True, exist_ok=True)
    source_out = output / "identity_probe_macro_f1_source.csv"
    with source_out.open("w", encoding="utf-8", newline="") as handle:
        fields = ["method", "method_label", "probe", "probe_label", "metric", "n", "mean", "sd", "chance", "uncertainty"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    fig = make_figure(summary)
    fig.savefig(output / "identity_probe_macro_f1.png", dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output / "identity_probe_macro_f1.pdf", facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    _write_manifest(output, source_csv, summary)
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = run(args.source_csv, args.output)
    print(json.dumps({"rows": len(summary), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
