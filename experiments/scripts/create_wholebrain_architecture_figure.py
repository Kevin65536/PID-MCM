#!/usr/bin/env python
"""Create the whole-brain source/observation downstream architecture figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_box(ax, xy, width, height, text, color, fontsize=9):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.2,
        edgecolor="#333333",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#111111",
        wrap=True,
    )
    return box


def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.2,
            color="#333333",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(ax, (0.03, 0.64), 0.18, 0.18, "Grouped event sample\n[A anchors x 4 branches x T tokens]", "#D8EAF7")
    add_box(ax, (0.03, 0.36), 0.18, 0.18, "Channel metadata\nanchor mask + dataset id\nexclusive EEG owner map", "#E8E8E8")
    add_box(ax, (0.27, 0.64), 0.18, 0.18, "Token embeddings\nbranch + modality + time + anchor", "#F6E3B4")
    add_box(ax, (0.50, 0.72), 0.18, 0.13, "Temporal encoder\nwithin each anchor-branch", "#CDECCF")
    add_box(ax, (0.50, 0.53), 0.18, 0.13, "Branch fusion\nsource/observation and EEG/fNIRS", "#CDECCF")
    add_box(ax, (0.72, 0.63), 0.20, 0.16, "Spatial encoder\nmasked anchor attention\nownership-aware bias", "#D9CEF2")
    add_box(ax, (0.72, 0.36), 0.20, 0.13, "Pretraining heads\nspan MLM + cross-branch InfoNCE", "#F3D0C7")
    add_box(ax, (0.72, 0.15), 0.20, 0.13, "Fine-tuning heads\nnback / WG / cognitive / MI / MA", "#F3D0C7")

    arrow(ax, (0.21, 0.73), (0.27, 0.73))
    arrow(ax, (0.21, 0.45), (0.72, 0.67))
    arrow(ax, (0.45, 0.73), (0.50, 0.79))
    arrow(ax, (0.59, 0.72), (0.59, 0.66))
    arrow(ax, (0.68, 0.595), (0.72, 0.68))
    arrow(ax, (0.82, 0.63), (0.82, 0.49))
    arrow(ax, (0.82, 0.63), (0.82, 0.28))

    ax.text(
        0.03,
        0.93,
        "Whole-brain source/observation token foundation model",
        fontsize=14,
        fontweight="bold",
        ha="left",
    )
    ax.text(
        0.03,
        0.89,
        "One sample is a complete event window, not a single fNIRS anchor. Missing anchors are masked; datasets share a global anchor vocabulary.",
        fontsize=9,
        ha="left",
    )
    ax.text(
        0.03,
        0.05,
        "Current token exports already contain local EEG-neighborhood tokens. Strict non-overlap EEG ownership requires a later raw/cache-level token re-export.",
        fontsize=8,
        ha="left",
        color="#444444",
    )

    fig.savefig(output, dpi=300, bbox_inches="tight")
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output, bbox_inches="tight")
    plt.close(fig)
    print(output)
    print(pdf_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
