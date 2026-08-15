#!/usr/bin/env python3
"""Render the evidence-backed experiment plan as accessible SVG and PNG."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "docs/figures/experiment_plan_status.json"
DEFAULT_SVG = REPO_ROOT / "docs/figures/experiment_plan.svg"
DEFAULT_PNG = REPO_ROOT / "docs/figures/experiment_plan.png"
DEFAULT_ALT = REPO_ROOT / "docs/figures/experiment_plan.alt.txt"
DEFAULT_MANIFEST = REPO_ROOT / "docs/figures/experiment_plan.manifest.json"

FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

STYLE = {
    "completed": {
        "fill": "#E7F4EE",
        "edge": "#007F5F",
        "symbol": "✓",
        "hatch": None,
        "linestyle": "-",
    },
    "negative": {
        "fill": "#FCEBE3",
        "edge": "#D55E00",
        "symbol": "×",
        "hatch": None,
        "linestyle": "-",
    },
    "failed": {
        "fill": "#F9E1DD",
        "edge": "#B33A3A",
        "symbol": "×",
        "hatch": "xx",
        "linestyle": "-",
    },
    "exploratory": {
        "fill": "#FFF4D6",
        "edge": "#B77900",
        "symbol": "●",
        "hatch": None,
        "linestyle": "-",
    },
    "undetermined": {
        "fill": "#FFF1E8",
        "edge": "#CC79A7",
        "symbol": "?",
        "hatch": "..",
        "linestyle": "-",
    },
    "running": {
        "fill": "#E4F0FA",
        "edge": "#0072B2",
        "symbol": "▶",
        "hatch": None,
        "linestyle": "-",
    },
    "next": {
        "fill": "#F4EAF2",
        "edge": "#9B4F96",
        "symbol": "◇",
        "hatch": None,
        "linestyle": "--",
    },
    "conditional": {
        "fill": "#F1F3F5",
        "edge": "#667085",
        "symbol": "△",
        "hatch": None,
        "linestyle": "--",
    },
    "blocked": {
        "fill": "#E5E7EB",
        "edge": "#343A40",
        "symbol": "⊘",
        "hatch": "///",
        "linestyle": "--",
    },
    "decision_stop": {
        "fill": "#F8D7DA",
        "edge": "#A61B1B",
        "symbol": "!",
        "hatch": "xx",
        "linestyle": "-",
    },
}

LEGEND = [
    ("completed", "已完成 / 通过"),
    ("negative", "完成 / 阴性"),
    ("failed", "资格门失败"),
    ("exploratory", "完成 / 探索"),
    ("running", "正在运行"),
    ("next", "下一步"),
    ("conditional", "条件后运行"),
    ("blocked", "阻断 / 未授权"),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _style_key(node: dict[str, Any]) -> str:
    status = node["status"]
    outcome = node.get("outcome", "")
    if status in {"failed", "undetermined", "running", "next", "conditional", "blocked", "decision_stop"}:
        return status
    if outcome == "negative" or outcome == "no_semantic_row_admitted_retain_T0":
        return "negative"
    if outcome == "exploratory":
        return "exploratory"
    return "completed"


def _comparison_status(source: dict[str, Any]) -> dict[str, Any]:
    try:
        status = source["live_status"]["comparison_campaign"]
    except KeyError as exc:
        raise ValueError("Missing live_status.comparison_campaign") from exc
    required = {
        "campaign_id",
        "completed_job_count",
        "expected_job_count",
        "failed_job_count",
        "protected_test_opened",
        "unblind_authorized",
        "aggregate_created_at",
        "cell_count",
        "terminal_counts",
    }
    missing = required - status.keys()
    if missing:
        raise ValueError(f"Incomplete comparison campaign status: {sorted(missing)}")
    if (
        status["completed_job_count"] != status["expected_job_count"]
        or status["expected_job_count"] != 540
        or status["failed_job_count"] != 0
        or status["protected_test_opened"] is not True
        or status["unblind_authorized"] is not True
        or status["cell_count"] != 42
        or sum(status["terminal_counts"].values()) != 42
    ):
        raise ValueError("Comparison campaign terminal status is inconsistent")
    return status


def _node_detail(source: dict[str, Any], node: dict[str, Any]) -> str:
    return node["detail"]


def _validate(source: dict[str, Any]) -> None:
    if source.get("schema") != "experiment_plan_source_v1":
        raise ValueError("Unsupported experiment-plan source schema")
    nodes = source.get("nodes", [])
    node_ids = [node["id"] for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Duplicate node IDs")
    known = set(node_ids)
    lanes = {lane["id"] for lane in source.get("lanes", [])}
    for node in nodes:
        if node["lane"] not in lanes:
            raise ValueError(f"Unknown lane for {node['id']}")
        if _style_key(node) not in STYLE:
            raise ValueError(f"Unknown status/style for {node['id']}")
        _node_detail(source, node)
    for edge in source.get("edges", []):
        if edge["from"] not in known or edge["to"] not in known:
            raise ValueError(f"Unknown edge endpoint: {edge}")
    expected_decision = {
        "promotion_eligible": False,
        "next_action": "do_not_enter_r2_p",
        "protected_subjects_24_29": "closed",
    }
    if source.get("decision") != expected_decision:
        raise ValueError("Main-method decision metadata changed unexpectedly")
    _comparison_status(source)


def _install_font() -> str:
    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        return font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    return "DejaVu Sans"


def _edge_points(
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[tuple[float, float], tuple[float, float]]:
    dx = right["x"] - left["x"]
    if abs(dx) >= 0.5:
        direction = 1 if dx > 0 else -1
        start = (left["x"] + direction * left["width"] / 2, left["y"])
        end = (right["x"] - direction * right["width"] / 2, right["y"])
    else:
        direction = 1 if right["y"] > left["y"] else -1
        start = (left["x"], left["y"] + direction * 0.42)
        end = (right["x"], right["y"] - direction * 0.42)
    return start, end


def _draw(source: dict[str, Any], svg_path: Path, png_path: Path) -> None:
    font_name = _install_font()
    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.size": 8.5,
            "svg.hashsalt": "pid-mcm-experiment-plan-v1",
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
            "hatch.linewidth": 0.7,
        }
    )

    fig, ax = plt.subplots(figsize=(24, 10.5), constrained_layout=False)
    fig.patch.set_facecolor("#FFFFFF")
    fig.subplots_adjust(left=0.015, right=0.995, bottom=0.02, top=0.99)
    ax.set_xlim(0, 30.7)
    ax.set_ylim(0, 13.25)
    ax.axis("off")

    lane_colors = ["#F8FAFC", "#FFFFFF", "#F8FAFC", "#FFFFFF"]
    for lane, fill in zip(source["lanes"], lane_colors):
        height = lane["ymax"] - lane["ymin"]
        band = FancyBboxPatch(
            (0.2, lane["ymin"]),
            30.25,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fill,
            edgecolor="#D0D5DD",
            linewidth=0.8,
            zorder=0,
        )
        ax.add_patch(band)
        ax.text(
            0.48,
            lane["ymax"] - 0.24,
            lane["label"],
            ha="left",
            va="top",
            fontsize=9.3,
            fontweight="bold",
            color="#344054",
            zorder=1,
        )

    ax.text(
        0.35,
        12.94,
        source["title"],
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="#101828",
    )
    ax.text(
        0.37,
        12.49,
        source["subtitle"],
        ha="left",
        va="top",
        fontsize=9.3,
        color="#475467",
    )
    ax.text(
        30.35,
        12.90,
        "当前主线：promotion_eligible = false  ·  protected 24–29 = closed",
        ha="right",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        color="#A61B1B",
    )

    legend_y = 11.82
    legend_xs = [0.75, 4.48, 8.20, 11.93, 15.66, 19.39, 23.12, 26.85]
    for x, (key, label) in zip(legend_xs, LEGEND):
        style = STYLE[key]
        swatch = FancyBboxPatch(
            (x, legend_y - 0.18),
            0.50,
            0.32,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=style["fill"],
            edgecolor=style["edge"],
            linewidth=1.3,
            linestyle=style["linestyle"],
            hatch=style["hatch"],
        )
        ax.add_patch(swatch)
        ax.text(
            x + 0.64,
            legend_y - 0.01,
            label,
            ha="left",
            va="center",
            fontsize=8.2,
            color="#344054",
        )

    node_by_id = {node["id"]: node for node in source["nodes"]}

    for index, edge in enumerate(source["edges"]):
        start, end = _edge_points(node_by_id[edge["from"]], node_by_id[edge["to"]])
        kind = edge.get("kind", "normal")
        edge_color = "#98A2B3"
        linestyle = "-"
        linewidth = 1.15
        if kind == "gate":
            edge_color = "#B33A3A"
            linewidth = 1.45
        elif kind == "blocked":
            edge_color = "#667085"
            linestyle = "--"
        elif kind == "conditional":
            edge_color = "#7B8190"
            linestyle = "--"
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=linewidth,
            linestyle=linestyle,
            color=edge_color,
            connectionstyle=f"arc3,rad={edge.get('curve', 0.0)}",
            shrinkA=0,
            shrinkB=0,
            zorder=2,
        )
        arrow.set_gid(f"edge-{index}")
        ax.add_patch(arrow)

    for node in source["nodes"]:
        key = _style_key(node)
        style = STYLE[key]
        height = 0.86
        patch = FancyBboxPatch(
            (node["x"] - node["width"] / 2, node["y"] - height / 2),
            node["width"],
            height,
            boxstyle="round,pad=0.035,rounding_size=0.09",
            facecolor=style["fill"],
            edgecolor=style["edge"],
            linewidth=2.0 if key in {"failed", "running", "decision_stop"} else 1.45,
            linestyle=style["linestyle"],
            hatch=style["hatch"],
            zorder=4,
        )
        patch.set_gid(f"stage-{node['id']}")
        ax.add_patch(patch)

        symbol = style["symbol"]
        symbol_size = 10.2
        ax.text(
            node["x"] - node["width"] / 2 + 0.12,
            node["y"] + 0.26,
            symbol,
            ha="left",
            va="center",
            fontsize=symbol_size,
            fontweight="bold",
            color=style["edge"],
            zorder=5,
        )
        label = ax.text(
            node["x"],
            node["y"] + 0.16,
            node["label"],
            ha="center",
            va="center",
            fontsize=8.45,
            fontweight="bold",
            color="#101828",
            zorder=5,
        )
        label.set_gid(f"label-{node['id']}")
        ax.text(
            node["x"],
            node["y"] - 0.19,
            _node_detail(source, node),
            ha="center",
            va="center",
            fontsize=7.35,
            linespacing=1.05,
            color="#475467",
            zorder=5,
        )

    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        svg_path,
        format="svg",
        bbox_inches=None,
        facecolor=fig.get_facecolor(),
        metadata={
            "Date": source["snapshot_at"],
            "Title": source["title"],
            "Description": source["subtitle"],
        },
    )
    fig.savefig(
        png_path,
        format="png",
        dpi=300,
        bbox_inches=None,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def _augment_svg(source: dict[str, Any], svg_path: Path) -> None:
    comparison = _comparison_status(source)
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    tree = ET.parse(svg_path)
    root = tree.getroot()
    root.set("role", "img")
    root.set("aria-labelledby", "experiment-plan-title experiment-plan-desc")

    title = ET.Element(f"{{{SVG_NS}}}title", {"id": "experiment-plan-title"})
    title.text = source["title"]
    desc = ET.Element(f"{{{SVG_NS}}}desc", {"id": "experiment-plan-desc"})
    desc.text = (
        "Four-lane experiment plan. The main tokenizer completed E0 through E2 "
        "and the R-series prerequisites but stopped after R1-P and R2-D failed. "
        "Future R2-P through R7 are blocked. The T0 Atlas Core run and STA-Net "
        "formal benchmark are complete. The joint comparison campaign completed "
        f"{comparison['completed_job_count']} of {comparison['expected_job_count']} "
        "protected jobs with zero failures, followed by dual-signature unblinding, "
        "aggregation, and terminal assignment for all 42 cells. Atlas Statistical "
        "tier and Croce Synthetic Phase 1 remain next."
    )

    metadata_payload = {
        "schema": "experiment-plan-v1",
        "source_schema": source["schema"],
        "snapshot_at": source["snapshot_at"],
        "timezone": source["timezone"],
        "decision": source["decision"],
        "authoritative_sources": source["authoritative_sources"],
        "live_status": source["live_status"],
    }
    metadata = ET.Element(f"{{{SVG_NS}}}metadata")
    metadata.text = json.dumps(
        metadata_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    root.insert(0, metadata)
    root.insert(0, desc)
    root.insert(0, title)

    for node in source["nodes"]:
        group = root.find(f".//*[@id='stage-{node['id']}']")
        label = root.find(f".//*[@id='label-{node['id']}']")
        if group is None or label is None:
            raise RuntimeError(f"Missing SVG group for node {node['id']}")
        group.set("role", "group")
        group.set("aria-labelledby", f"label-{node['id']}")
        group.set("data-stage", node["id"])
        group.set("data-status", node["status"])
        group.set("data-outcome", node.get("outcome", ""))
        group.set("data-authorized", str(bool(node["authorized"])).lower())

    for index, edge in enumerate(source["edges"]):
        group = root.find(f".//*[@id='edge-{index}']")
        if group is None:
            raise RuntimeError(f"Missing SVG group for edge {index}")
        group.set("data-from", edge["from"])
        group.set("data-to", edge["to"])
        group.set("data-kind", edge.get("kind", "normal"))

    ET.indent(tree, space="  ")
    tree.write(svg_path, encoding="utf-8", xml_declaration=True)


def _write_sidecars(
    source: dict[str, Any],
    source_path: Path,
    svg_path: Path,
    png_path: Path,
    alt_path: Path,
    manifest_path: Path,
) -> None:
    comparison = _comparison_status(source)
    terminals = comparison["terminal_counts"]
    alt_text = (
        "EEG–fNIRS full experiment plan, status frozen "
        f"{source['snapshot_at']}. The main-method lane shows the unified data "
        "contract and E0/E1 complete, E2 complete without an admitted semantic "
        "row, R0-P negative, R1-D exploratory, R1-P and R2-D failed, and D1B "
        "undetermined. A stop decision blocks R2-P through R7 and keeps subjects "
        "24–29 closed. The Token Atlas lane shows T0 Core complete, Statistical "
        "tier next, Full coupling-null conditional, and a new-VQ Atlas blocked. "
        "The comparison lane shows STA-Net 70/70 complete as a context reference; "
        f"the joint campaign completed {comparison['completed_job_count']}/"
        f"{comparison['expected_job_count']} protected jobs with zero failures; "
        "dual-signature unblinding and aggregation complete; and all 42 cells "
        f"assigned {terminals['TABLE_READY_WITH_NOTE']} ready-with-note, "
        f"{terminals['REJECTED_VALUE']} rejected, "
        f"{terminals['OVERLAP_TRACK_ONLY']} overlap-only, and "
        f"{terminals['UNSUPPORTED']} unsupported terminals. The "
        "Croce lane shows legacy diagnostics complete, redesigned Synthetic "
        "Phase 1 next, Real Phase 2 conditional, and main-method requalification "
        "requiring a new independent contract."
    )
    alt_path.write_text(alt_text + "\n", encoding="utf-8")

    manifest = {
        "schema": "experiment_plan_figure_manifest_v1",
        "created_from": str(source_path.relative_to(REPO_ROOT)),
        "snapshot_at": source["snapshot_at"],
        "source_sha256": _sha256(source_path),
        "outputs": {
            str(svg_path.relative_to(REPO_ROOT)): {
                "bytes": svg_path.stat().st_size,
                "sha256": _sha256(svg_path),
                "format": "svg",
            },
            str(png_path.relative_to(REPO_ROOT)): {
                "bytes": png_path.stat().st_size,
                "sha256": _sha256(png_path),
                "format": "png",
                "dpi": 300,
            },
            str(alt_path.relative_to(REPO_ROOT)): {
                "bytes": alt_path.stat().st_size,
                "sha256": _sha256(alt_path),
                "format": "plain_text_alt",
            },
        },
        "software": {
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
            "font": str(FONT_PATH) if FONT_PATH.exists() else "DejaVu Sans",
        },
        "accessibility": {
            "svg_role": "img",
            "svg_title": True,
            "svg_description": True,
            "status_redundancy": "color + symbol + border/hatch",
        },
        "authoritative_sources": source["authoritative_sources"],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--svg", type=Path, default=DEFAULT_SVG)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--alt", type=Path, default=DEFAULT_ALT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    source_path = args.source.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    _validate(source)
    _draw(source, args.svg.resolve(), args.png.resolve())
    _augment_svg(source, args.svg.resolve())
    _write_sidecars(
        source,
        source_path,
        args.svg.resolve(),
        args.png.resolve(),
        args.alt.resolve(),
        args.manifest.resolve(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
