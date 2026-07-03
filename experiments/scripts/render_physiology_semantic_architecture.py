#!/usr/bin/env python3
"""Render the maintained physiology-semantic architecture specification as SVG."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = REPO_ROOT / "docs/physiology_semantic_tokenizer/architecture/physiology_semantic_architecture.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs/physiology_semantic_tokenizer/figures/physiology_semantic_architecture.svg"

STATUS_STYLE = {
    "implemented": ("#e8f1ff", "#2563eb", "#173b6c"),
    "interface": ("#e8faf1", "#15965a", "#14532d"),
    "training": ("#f2ecff", "#7c3aed", "#3b0764"),
    "guarded": ("#fff7d6", "#ca8a04", "#713f12"),
    "output": ("#e6f8f6", "#0f8f83", "#134e4a"),
    "blocked": ("#f3f4f6", "#6b7280", "#374151"),
    "planned": ("#ecfdf5", "#16a34a", "#14532d"),
}

CHANGE_STYLE = {
    "add": ("#16a34a", "ADD"),
    "modify": ("#dc2626", "MODIFY"),
    "remove": ("#991b1b", "REMOVE"),
}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate(spec: Mapping[str, Any], changes: Mapping[str, Any] | None = None) -> None:
    if spec.get("schema") != "physiology_semantic_architecture_v1":
        raise ValueError("Unsupported architecture specification schema")
    nodes = list(spec.get("nodes", []))
    node_ids = [node["id"] for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Architecture node ids must be unique")
    known = set(node_ids)
    for edge in spec.get("edges", []):
        if edge["from"] not in known or edge["to"] not in known:
            raise ValueError(f"Unknown edge endpoint: {edge}")
    if changes is None:
        return
    if changes.get("schema") != "physiology_semantic_architecture_changes_v1":
        raise ValueError("Unsupported change-overlay schema")
    add_nodes = list(changes.get("add_nodes", []))
    added_ids = {node["id"] for node in add_nodes}
    if known.intersection(added_ids):
        raise ValueError("Added node ids must not replace baseline nodes")
    known |= added_ids
    for change in changes.get("changes", []):
        if change["node_id"] not in known:
            raise ValueError(f"Change references unknown node {change['node_id']!r}")
        if change["kind"] not in CHANGE_STYLE:
            raise ValueError(f"Unsupported change kind {change['kind']!r}")
    for edge in changes.get("add_edges", []):
        if edge["from"] not in known or edge["to"] not in known:
            raise ValueError(f"Unknown added-edge endpoint: {edge}")


def _node_anchor(node: Mapping[str, Any], side: str) -> tuple[float, float]:
    x, y = float(node["x"]), float(node["y"])
    width, height = float(node["width"]), float(node["height"])
    if side == "left":
        return x, y + height / 2
    if side == "right":
        return x + width, y + height / 2
    if side == "top":
        return x + width / 2, y
    return x + width / 2, y + height


def _edge_path(source: Mapping[str, Any], target: Mapping[str, Any]) -> str:
    source_center_x = float(source["x"]) + float(source["width"]) / 2
    target_center_x = float(target["x"]) + float(target["width"]) / 2
    source_center_y = float(source["y"]) + float(source["height"]) / 2
    target_center_y = float(target["y"]) + float(target["height"]) / 2
    if abs(target_center_x - source_center_x) >= abs(target_center_y - source_center_y):
        if target_center_x >= source_center_x:
            start = _node_anchor(source, "right")
            end = _node_anchor(target, "left")
        else:
            start = _node_anchor(source, "left")
            end = _node_anchor(target, "right")
        middle = (start[0] + end[0]) / 2
        return f"M {start[0]:.1f} {start[1]:.1f} C {middle:.1f} {start[1]:.1f}, {middle:.1f} {end[1]:.1f}, {end[0]:.1f} {end[1]:.1f}"
    if target_center_y >= source_center_y:
        start = _node_anchor(source, "bottom")
        end = _node_anchor(target, "top")
    else:
        start = _node_anchor(source, "top")
        end = _node_anchor(target, "bottom")
    middle = (start[1] + end[1]) / 2
    return f"M {start[0]:.1f} {start[1]:.1f} C {start[0]:.1f} {middle:.1f}, {end[0]:.1f} {middle:.1f}, {end[0]:.1f} {end[1]:.1f}"


def _render_edge(edge: Mapping[str, Any], node_map: Mapping[str, Mapping[str, Any]], added: bool = False) -> str:
    style = str(edge.get("style", "implemented"))
    color = {
        "implemented": "#64748b",
        "training": "#7c3aed",
        "guarded": "#ca8a04",
        "blocked": "#6b7280",
    }.get(style, "#64748b")
    dash = " stroke-dasharray=\"7 6\"" if style in {"training", "blocked"} or added else ""
    marker = "arrow-add" if added else "arrow"
    path = _edge_path(node_map[edge["from"]], node_map[edge["to"]])
    label = ""
    if edge.get("label"):
        source = node_map[edge["from"]]
        target = node_map[edge["to"]]
        x = (float(source["x"]) + float(source["width"]) / 2 + float(target["x"]) + float(target["width"]) / 2) / 2
        y = (float(source["y"]) + float(source["height"]) / 2 + float(target["y"]) + float(target["height"]) / 2) / 2 - 5
        label = f'<text class="edge-label" x="{x:.1f}" y="{y:.1f}">{escape(str(edge["label"]))}</text>'
    return f'<g class="edge edge-{escape(style)}"><path d="{path}" fill="none" stroke="{color}" stroke-width="2" marker-end="url(#{marker})"{dash}/>{label}</g>'


def _render_node(node: Mapping[str, Any], change: Mapping[str, Any] | None = None) -> str:
    status = str(node.get("status", "implemented"))
    fill, stroke, text_color = STATUS_STYLE[status]
    x, y = float(node["x"]), float(node["y"])
    width, height = float(node["width"]), float(node["height"])
    title_y = y + 25
    details = list(node.get("details", []))
    detail_lines = "".join(
        f'<tspan x="{x + 12:.1f}" dy="16">{escape(str(line))}</tspan>' for line in details
    )
    overlay = ""
    change_attrs = ""
    if change is not None:
        change_color, badge = CHANGE_STYLE[change["kind"]]
        overlay = (
            f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="{width + 10:.1f}" height="{height + 10:.1f}" '
            f'rx="13" fill="none" stroke="{change_color}" stroke-width="4" stroke-dasharray="9 5"/>'
            f'<rect x="{x + width - 66:.1f}" y="{y - 15:.1f}" width="66" height="22" rx="11" fill="{change_color}"/>'
            f'<text x="{x + width - 33:.1f}" y="{y:.1f}" class="change-badge">{badge}</text>'
        )
        if change["kind"] == "remove":
            overlay += (
                f'<path d="M {x + 8:.1f} {y + 8:.1f} L {x + width - 8:.1f} {y + height - 8:.1f} M {x + width - 8:.1f} {y + 8:.1f} L {x + 8:.1f} {y + height - 8:.1f}" '
                f'stroke="{change_color}" stroke-width="4"/>'
            )
        change_attrs = f' data-change-kind="{escape(str(change["kind"]))}"'
    description = "; ".join([str(node["label"]), *map(str, details)])
    return (
        f'<g id="node-{escape(str(node["id"]))}" class="node status-{escape(status)}" data-status="{escape(status)}"{change_attrs}>'
        f'<title>{escape(str(node["label"]))}</title><desc>{escape(description)}</desc>'
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        f'<text x="{x + 12:.1f}" y="{title_y:.1f}" class="node-title" fill="{text_color}">{escape(str(node["label"]))}</text>'
        f'<text x="{x + 12:.1f}" y="{title_y + 8:.1f}" class="node-detail" fill="{text_color}">{detail_lines}</text>'
        f'{overlay}</g>'
    )


def render_svg(spec: Mapping[str, Any], changes: Mapping[str, Any] | None = None) -> str:
    _validate(spec, changes)
    width, height = int(spec["width"]), int(spec["height"])
    nodes = [dict(node) for node in spec["nodes"]]
    edges = [dict(edge) for edge in spec["edges"]]
    change_map: Dict[str, Mapping[str, Any]] = {}
    added_ids: set[str] = set()
    if changes is not None:
        nodes.extend(dict(node) for node in changes.get("add_nodes", []))
        edges.extend(dict(edge, _added=True) for edge in changes.get("add_edges", []))
        change_map = {change["node_id"]: change for change in changes.get("changes", [])}
        added_ids = {node["id"] for node in changes.get("add_nodes", [])}
        for node_id in added_ids:
            change_map.setdefault(node_id, {"node_id": node_id, "kind": "add", "note": "Added by plan"})
    node_map = {node["id"]: node for node in nodes}

    metadata = {
        "schema": spec["schema"],
        "change_plan": None if changes is None else changes.get("plan_id"),
        "source": str(DEFAULT_SPEC.relative_to(REPO_ROOT)),
    }
    section_svg = "".join(
        f'<g id="section-{escape(str(section["id"]))}"><rect x="{section["x"]}" y="{section["y"]}" width="{section["width"]}" height="{section["height"]}" rx="16" class="section-box"/>'
        f'<text x="{section["x"] + 16}" y="{section["y"] + 30}" class="section-title">{escape(str(section["label"]))}</text></g>'
        for section in spec["sections"]
    )
    edge_svg = "".join(_render_edge(edge, node_map, bool(edge.get("_added"))) for edge in edges)
    node_svg = "".join(_render_node(node, change_map.get(node["id"])) for node in nodes)
    notes = ""
    if changes is not None:
        rows = []
        for index, change in enumerate(changes.get("changes", [])):
            color, badge = CHANGE_STYLE[change["kind"]]
            rows.append(
                f'<circle cx="55" cy="{985 + index * 20}" r="5" fill="{color}"/><text x="68" y="{990 + index * 20}" class="change-note">{badge}: {escape(str(change.get("note", change["node_id"])))}</text>'
            )
        notes = "".join(rows)

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="svg-title svg-desc">
  <title id="svg-title">{escape(str(spec["title"]))}</title>
  <desc id="svg-desc">{escape(str(spec["subtitle"]))}</desc>
  <metadata>{escape(json.dumps(metadata, sort_keys=True))}</metadata>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/></marker>
    <marker id="arrow-add" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#16a34a"/></marker>
  </defs>
  <style>
    text {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .page-title {{ font-size: 28px; font-weight: 700; fill: #0f172a; }}
    .page-subtitle {{ font-size: 15px; fill: #475569; }}
    .section-box {{ fill: #f8fafc; stroke: #cbd5e1; stroke-width: 1.5; }}
    .section-title {{ font-size: 17px; font-weight: 700; fill: #334155; }}
    .node-title {{ font-size: 14px; font-weight: 700; }}
    .node-detail {{ font-size: 11.5px; }}
    .edge-label {{ font-size: 10px; text-anchor: middle; fill: #475569; paint-order: stroke; stroke: #ffffff; stroke-width: 4px; }}
    .change-badge {{ font-size: 10px; font-weight: 800; text-anchor: middle; fill: #ffffff; }}
    .change-note {{ font-size: 11px; fill: #334155; }}
    .legend-label {{ font-size: 11px; fill: #334155; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="30" y="46" class="page-title">{escape(str(spec["title"]))}</text>
  <text x="30" y="73" class="page-subtitle">{escape(str(spec["subtitle"]))}</text>
  {section_svg}
  <g id="edges">{edge_svg}</g>
  <g id="nodes">{node_svg}</g>
  <g id="legend" transform="translate(30, 970)">
    <text x="0" y="0" class="section-title">Status</text>
    <rect x="75" y="-15" width="18" height="18" rx="4" fill="#e8f1ff" stroke="#2563eb"/><text x="100" y="0" class="legend-label">implemented</text>
    <rect x="205" y="-15" width="18" height="18" rx="4" fill="#f2ecff" stroke="#7c3aed"/><text x="230" y="0" class="legend-label">training-only</text>
    <rect x="345" y="-15" width="18" height="18" rx="4" fill="#fff7d6" stroke="#ca8a04"/><text x="370" y="0" class="legend-label">guarded</text>
    <rect x="455" y="-15" width="18" height="18" rx="4" fill="#f3f4f6" stroke="#6b7280"/><text x="480" y="0" class="legend-label">blocked</text>
    <rect x="555" y="-15" width="18" height="18" rx="4" fill="#e6f8f6" stroke="#0f8f83"/><text x="580" y="0" class="legend-label">export / interface</text>
  </g>
  <g id="change-notes">{notes}</g>
</svg>
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--changes", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Fail when output differs from a fresh render")
    args = parser.parse_args()

    spec = load_json(args.spec)
    changes = load_json(args.changes) if args.changes else None
    output = args.output.resolve()
    if changes is not None and output == DEFAULT_OUTPUT.resolve():
        raise ValueError("A plan overlay must use a plan-specific output and cannot overwrite the current architecture SVG")
    rendered = render_svg(spec, changes)
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"Architecture SVG is stale: {output}")
        print(f"Architecture SVG is current: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
