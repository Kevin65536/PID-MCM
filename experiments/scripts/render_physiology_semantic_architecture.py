#!/usr/bin/env python3
"""Render the maintained physiology-semantic architecture specification as SVG.

The renderer deliberately separates four concepts that the v1 diagram collapsed:
functional role, runtime scope, implementation state, and scientific evidence.
Plan overlays are composed as proposed after-state views without mutating the
canonical specification.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import textwrap
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = REPO_ROOT / "docs/physiology_semantic_tokenizer/architecture/physiology_semantic_architecture.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs/physiology_semantic_tokenizer/figures/physiology_semantic_architecture.svg"

SPEC_SCHEMAS = {"physiology_semantic_architecture_v1", "physiology_semantic_architecture_v2"}
OVERLAY_SCHEMAS = {
    "physiology_semantic_architecture_changes_v1",
    "physiology_semantic_architecture_changes_v2",
}

ROLE_STYLE = {
    "data": ("#EAF3FA", "#0072B2", "#123B56"),
    "encoder": ("#EDF4FB", "#3B78A8", "#173B55"),
    "latent": ("#F3EEFA", "#8A5FB0", "#432A5C"),
    "quantizer": ("#EAF7F2", "#009E73", "#124C3D"),
    "objective": ("#FFF5DF", "#D68B00", "#5B3A00"),
    "lifecycle": ("#F1F4F6", "#59636E", "#29313A"),
    "interface": ("#EAF7F7", "#008A91", "#16474B"),
    "evaluator": ("#FCEEF2", "#C44569", "#68243A"),
    "teacher": ("#F3EEFA", "#7A54A3", "#432A5C"),
}

EDGE_STYLE = {
    "data": ("#526272", "", "solid data flow"),
    "training": ("#7A54A3", "9 6", "training-only supervision"),
    "gradient": ("#A45C00", "3 5", "gradient path"),
    "control": ("#59636E", "12 5 3 5", "lifecycle control"),
    "evaluation": ("#C44569", "5 5", "frozen evaluation"),
    "guarded": ("#A45C00", "12 6", "guarded transition"),
    "blocked": ("#6B7280", "3 6", "blocked transition"),
    "removed": ("#9F1239", "4 5", "removed relationship"),
}

CHANGE_STYLE = {
    "add": ("#15803D", "A"),
    "modify": ("#1D4ED8", "M"),
    "remove": ("#9F1239", "R"),
}

VALID_ROLE = set(ROLE_STYLE)
VALID_SCOPE = {"inference", "training_only", "export", "evaluation", "governance"}
VALID_IMPLEMENTATION = {"implemented", "planned", "removed"}
VALID_EVIDENCE = {"admitted", "guarded", "blocked", "n_a"}
VALID_SIDES = {"left", "right", "top", "bottom"}
REPLACE_FIELDS = {"label", "details", "role", "scope", "implementation", "evidence"}
LAYOUT_FIELDS = {"x", "y", "width", "height"}
EDGE_REPLACE_FIELDS = {"label", "style", "route", "label_at"}
LEGACY_LAYOUT_SCALE = 1.2
LEGACY_MIN_NODE_HEIGHT = 96

LEGACY_STATUS = {
    "implemented": ("encoder", "inference", "implemented", "n_a"),
    "interface": ("interface", "inference", "implemented", "n_a"),
    "training": ("objective", "training_only", "implemented", "n_a"),
    "guarded": ("lifecycle", "training_only", "implemented", "guarded"),
    "output": ("interface", "export", "implemented", "n_a"),
    "blocked": ("evaluator", "evaluation", "planned", "blocked"),
    "planned": ("lifecycle", "training_only", "planned", "n_a"),
}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scale_geometry(item: MutableMapping[str, Any], scale: float, *, fit_text: bool = False) -> None:
    for field in ("x", "y", "width", "height"):
        if field in item:
            item[field] = round(float(item[field]) * scale, 1)
    if fit_text and "height" in item:
        item["height"] = max(float(item["height"]), LEGACY_MIN_NODE_HEIGHT)


def _prepare_inputs(
    spec: Mapping[str, Any], changes: Mapping[str, Any] | None
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Expand legacy geometry for legibility without changing its source JSON."""

    prepared_spec = copy.deepcopy(dict(spec))
    prepared_changes = copy.deepcopy(dict(changes)) if changes is not None else None
    if prepared_spec.get("schema") != "physiology_semantic_architecture_v1":
        return prepared_spec, prepared_changes

    scale = LEGACY_LAYOUT_SCALE
    prepared_spec["width"] = int(round(float(prepared_spec["width"]) * scale))
    prepared_spec["height"] = int(round(float(prepared_spec["height"]) * scale + 60))
    prepared_spec["footer_y"] = prepared_spec["height"] - 82
    for section in prepared_spec.get("sections", []):
        _scale_geometry(section, scale)
    for node in prepared_spec.get("nodes", []):
        _scale_geometry(node, scale, fit_text=True)

    if prepared_changes is None:
        return prepared_spec, None
    for node in prepared_changes.get("add_nodes", []):
        _scale_geometry(node, scale, fit_text=True)
    for change in prepared_changes.get("changes", []):
        layout = change.get("layout")
        if layout:
            _scale_geometry(layout, scale, fit_text="height" in layout)
    return prepared_spec, prepared_changes


def _normalize_node(node: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(node)
    if "role" not in normalized:
        role, scope, implementation, evidence = LEGACY_STATUS[str(normalized.get("status", "implemented"))]
        normalized.update(role=role, scope=scope, implementation=implementation, evidence=evidence)
    normalized.setdefault("scope", "inference")
    normalized.setdefault("implementation", "implemented")
    normalized.setdefault("evidence", "n_a")
    normalized.setdefault("details", [])
    return normalized


def _edge_id(edge: Mapping[str, Any]) -> str:
    explicit = edge.get("id")
    if explicit:
        return str(explicit)
    return f'{edge["from"]}--{edge["to"]}'


def _validate_node(node: Mapping[str, Any], width: int, content_bottom: int) -> None:
    missing = {"id", "label", "x", "y", "width", "height"}.difference(node)
    if missing:
        raise ValueError(f"Node is missing fields {sorted(missing)}: {node}")
    normalized = _normalize_node(node)
    if normalized["role"] not in VALID_ROLE:
        raise ValueError(f"Unsupported role {normalized['role']!r}")
    if normalized["scope"] not in VALID_SCOPE:
        raise ValueError(f"Unsupported scope {normalized['scope']!r}")
    if normalized["implementation"] not in VALID_IMPLEMENTATION:
        raise ValueError(f"Unsupported implementation {normalized['implementation']!r}")
    if normalized["evidence"] not in VALID_EVIDENCE:
        raise ValueError(f"Unsupported evidence {normalized['evidence']!r}")
    x, y = float(node["x"]), float(node["y"])
    node_width, node_height = float(node["width"]), float(node["height"])
    if min(x, y) < 0 or min(node_width, node_height) <= 0:
        raise ValueError(f"Invalid node geometry: {node['id']}")
    if x + node_width > width or y + node_height > content_bottom:
        raise ValueError(f"Node lies outside content area: {node['id']}")
    required_height = 62 + 17 * len(normalized.get("details", []))
    if node_height < required_height:
        raise ValueError(
            f"Node {node['id']!r} is too short for {len(normalized.get('details', []))} detail lines "
            f"({node_height:g} < {required_height:g})"
        )


def _validate_route(route: Mapping[str, Any], edge_id: str) -> None:
    if route.get("kind", "orthogonal") != "orthogonal":
        raise ValueError(f"Only orthogonal routing is supported: {edge_id}")
    for field in ("from_side", "to_side"):
        if field in route and route[field] not in VALID_SIDES:
            raise ValueError(f"Invalid {field} for edge {edge_id}")
    for point in route.get("via", []):
        if not isinstance(point, Sequence) or len(point) != 2:
            raise ValueError(f"Invalid waypoint for edge {edge_id}: {point}")


def _validate_node_overlaps(nodes: Sequence[Mapping[str, Any]]) -> None:
    for index, first in enumerate(nodes):
        if first.get("implementation") == "removed":
            continue
        ax1, ay1 = float(first["x"]), float(first["y"])
        ax2, ay2 = ax1 + float(first["width"]), ay1 + float(first["height"])
        for second in nodes[index + 1 :]:
            if second.get("implementation") == "removed":
                continue
            bx1, by1 = float(second["x"]), float(second["y"])
            bx2, by2 = bx1 + float(second["width"]), by1 + float(second["height"])
            if min(ax2, bx2) > max(ax1, bx1) and min(ay2, by2) > max(ay1, by1):
                raise ValueError(f"Architecture nodes overlap: {first['id']} and {second['id']}")


def _validate(spec: Mapping[str, Any], changes: Mapping[str, Any] | None = None) -> None:
    if spec.get("schema") not in SPEC_SCHEMAS:
        raise ValueError("Unsupported architecture specification schema")
    width = int(spec["width"])
    content_bottom = int(spec.get("footer_y", spec["height"] - 80))
    nodes = [_normalize_node(node) for node in spec.get("nodes", [])]
    for node in nodes:
        _validate_node(node, width, content_bottom)
    node_ids = [str(node["id"]) for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Architecture node ids must be unique")
    known = set(node_ids)
    canonical_node_map = {str(node["id"]): node for node in nodes}
    edge_ids: list[str] = []
    for edge in spec.get("edges", []):
        edge_id = _edge_id(edge)
        edge_ids.append(edge_id)
        if edge["from"] not in known or edge["to"] not in known:
            raise ValueError(f"Unknown edge endpoint: {edge}")
        if edge.get("style", "data") not in EDGE_STYLE:
            raise ValueError(f"Unsupported edge style: {edge}")
        _validate_route(edge.get("route", {}), edge_id)
    if len(edge_ids) != len(set(edge_ids)):
        raise ValueError("Architecture edge ids must be unique; specify explicit ids for parallel edges")
    if changes is None:
        return
    if changes.get("schema") not in OVERLAY_SCHEMAS:
        raise ValueError("Unsupported change-overlay schema")
    add_nodes = [_normalize_node(node) for node in changes.get("add_nodes", [])]
    added_ids = [str(node["id"]) for node in add_nodes]
    if len(added_ids) != len(set(added_ids)):
        raise ValueError("Added node ids must be unique")
    if known.intersection(added_ids):
        raise ValueError("Added node ids must not replace baseline nodes")
    for node in add_nodes:
        _validate_node(node, width, content_bottom)
    known |= set(added_ids)
    for change in changes.get("changes", []):
        if change["node_id"] not in known:
            raise ValueError(f"Change references unknown node {change['node_id']!r}")
        if change["kind"] not in CHANGE_STYLE:
            raise ValueError(f"Unsupported change kind {change['kind']!r}")
        if not str(change.get("note", "")).strip():
            raise ValueError(f"Change note is required for {change['node_id']}")
        unknown_replace = set(change.get("replace", {})).difference(REPLACE_FIELDS)
        unknown_layout = set(change.get("layout", {})).difference(LAYOUT_FIELDS)
        if unknown_replace or unknown_layout:
            raise ValueError(f"Unsupported node replacement fields: {sorted(unknown_replace | unknown_layout)}")
        candidate = dict(canonical_node_map[change["node_id"]])
        candidate.update(change.get("replace", {}))
        candidate.update(change.get("layout", {}))
        if change["kind"] == "remove":
            candidate["implementation"] = "removed"
        _validate_node(candidate, width, content_bottom)
    known_edges = set(edge_ids)
    for change in changes.get("edge_changes", []):
        if change["edge_id"] not in known_edges:
            raise ValueError(f"Change references unknown edge {change['edge_id']!r}")
        if change["kind"] not in CHANGE_STYLE:
            raise ValueError(f"Unsupported edge change kind {change['kind']!r}")
        if not str(change.get("note", "")).strip():
            raise ValueError(f"Edge change note is required for {change['edge_id']}")
        unknown_edge_replace = set(change.get("replace", {})).difference(EDGE_REPLACE_FIELDS)
        if unknown_edge_replace:
            raise ValueError(f"Unsupported edge replacement fields: {sorted(unknown_edge_replace)}")
        if change.get("replace", {}).get("style", "data") not in EDGE_STYLE:
            raise ValueError(f"Unsupported replacement edge style for {change['edge_id']}")
        if "route" in change.get("replace", {}):
            _validate_route(change["replace"]["route"], change["edge_id"])
    added_edge_ids: list[str] = []
    for edge in changes.get("add_edges", []):
        if edge["from"] not in known or edge["to"] not in known:
            raise ValueError(f"Unknown added-edge endpoint: {edge}")
        edge_id = _edge_id(edge)
        added_edge_ids.append(edge_id)
        if edge_id in known_edges:
            raise ValueError(f"Added edge id replaces baseline edge: {edge_id}")
        if edge.get("style", "data") not in EDGE_STYLE:
            raise ValueError(f"Unsupported added-edge style: {edge}")
        _validate_route(edge.get("route", {}), edge_id)
    if len(added_edge_ids) != len(set(added_edge_ids)):
        raise ValueError("Added edge ids must be unique")


def _change_records(changes: Mapping[str, Any]) -> tuple[list[dict[str, str]], dict[tuple[str, str], str]]:
    records: list[dict[str, str]] = []
    references: dict[tuple[str, str], str] = {}
    counts = {kind: 0 for kind in CHANGE_STYLE}

    def add(kind: str, object_type: str, object_id: str, note: str) -> None:
        counts[kind] += 1
        reference = f"{CHANGE_STYLE[kind][1]}{counts[kind]}"
        records.append(
            {"kind": kind, "object_type": object_type, "object_id": object_id, "note": note, "ref": reference}
        )
        references[(object_type, object_id)] = reference

    for change in changes.get("changes", []):
        add(str(change["kind"]), "node", str(change["node_id"]), str(change["note"]))
    for node in changes.get("add_nodes", []):
        add("add", "node", str(node["id"]), str(node.get("note", f"Add {node['label']}")))
    for change in changes.get("edge_changes", []):
        add(str(change["kind"]), "edge", str(change["edge_id"]), str(change["note"]))
    for edge in changes.get("add_edges", []):
        edge_id = _edge_id(edge)
        add("add", "edge", edge_id, str(edge.get("note", f"Add {edge['from']} to {edge['to']} flow")))
    return records, references


def _compose_view(
    spec: Mapping[str, Any], changes: Mapping[str, Any] | None
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    view = copy.deepcopy(dict(spec))
    view["nodes"] = [_normalize_node(node) for node in view["nodes"]]
    view["edges"] = [dict(edge) for edge in view["edges"]]
    if changes is None:
        return view, []

    records, references = _change_records(changes)
    node_map: dict[str, MutableMapping[str, Any]] = {node["id"]: node for node in view["nodes"]}
    for change in changes.get("changes", []):
        node = node_map[change["node_id"]]
        node["_canonical"] = {
            key: node[key] for key in ("label", "role", "scope", "implementation", "evidence")
        }
        node.update(copy.deepcopy(change.get("replace", {})))
        node.update(copy.deepcopy(change.get("layout", {})))
        if change["kind"] == "remove":
            node["implementation"] = "removed"
        node["_change_kind"] = change["kind"]
        node["_change_note"] = change["note"]
        node["_change_ref"] = references[("node", change["node_id"])]

    for raw_node in changes.get("add_nodes", []):
        node = _normalize_node(raw_node)
        node["_change_kind"] = "add"
        node["_change_note"] = raw_node.get("note", f"Add {raw_node['label']}")
        node["_change_ref"] = references[("node", raw_node["id"])]
        view["nodes"].append(node)
        node_map[node["id"]] = node

    edge_map = {_edge_id(edge): edge for edge in view["edges"]}
    for change in changes.get("edge_changes", []):
        edge = edge_map[change["edge_id"]]
        edge.update(copy.deepcopy(change.get("replace", {})))
        edge["_change_kind"] = change["kind"]
        edge["_change_note"] = change["note"]
        edge["_change_ref"] = references[("edge", change["edge_id"])]
        if change["kind"] == "remove":
            edge["style"] = "removed"
    for raw_edge in changes.get("add_edges", []):
        edge = dict(raw_edge)
        edge["_change_kind"] = "add"
        edge["_change_note"] = raw_edge.get("note", f"Add {raw_edge['from']} to {raw_edge['to']} flow")
        edge["_change_ref"] = references[("edge", _edge_id(raw_edge))]
        view["edges"].append(edge)

    view["title"] = changes.get("title", f"Plan · {changes['plan_id'].replace('_', ' ')}")
    view["subtitle"] = changes.get("subtitle", changes.get("summary", "Proposed architecture after-state"))
    if "banner" in changes:
        view["banner"] = changes["banner"]
    view["_plan_id"] = changes["plan_id"]
    view["_plan_summary"] = changes.get("summary", "")
    return view, records


def _node_anchor(node: Mapping[str, Any], side: str, fraction: float = 0.5) -> tuple[float, float]:
    x, y = float(node["x"]), float(node["y"])
    width, height = float(node["width"]), float(node["height"])
    fraction = max(0.05, min(0.95, fraction))
    if side == "left":
        return x, y + height * fraction
    if side == "right":
        return x + width, y + height * fraction
    if side == "top":
        return x + width * fraction, y
    return x + width * fraction, y + height


def _route_points(edge: Mapping[str, Any], node_map: Mapping[str, Mapping[str, Any]]) -> list[tuple[float, float]]:
    source = node_map[edge["from"]]
    target = node_map[edge["to"]]
    route = edge.get("route", {})
    source_x = float(source["x"]) + float(source["width"]) / 2
    target_x = float(target["x"]) + float(target["width"]) / 2
    source_y = float(source["y"]) + float(source["height"]) / 2
    target_y = float(target["y"]) + float(target["height"]) / 2
    if "from_side" in route:
        from_side = route["from_side"]
    else:
        from_side = "right" if abs(target_x - source_x) >= abs(target_y - source_y) and target_x >= source_x else "bottom"
    if "to_side" in route:
        to_side = route["to_side"]
    else:
        to_side = "left" if from_side in {"right", "left"} else "top"
    start = _node_anchor(source, from_side, float(route.get("from_fraction", 0.5)))
    end = _node_anchor(target, to_side, float(route.get("to_fraction", 0.5)))
    via = [(float(point[0]), float(point[1])) for point in route.get("via", [])]
    if via:
        points = [start, *via, end]
    elif from_side in {"left", "right"} and to_side in {"left", "right"}:
        midpoint = (start[0] + end[0]) / 2
        points = [start, (midpoint, start[1]), (midpoint, end[1]), end]
    elif from_side in {"top", "bottom"} and to_side in {"top", "bottom"}:
        midpoint = (start[1] + end[1]) / 2
        points = [start, (start[0], midpoint), (end[0], midpoint), end]
    else:
        points = [start, (end[0], start[1]), end]
    compact: list[tuple[float, float]] = []
    for point in points:
        if compact and point == compact[-1]:
            continue
        if len(compact) >= 2:
            a, b = compact[-2], compact[-1]
            if (a[0] == b[0] == point[0]) or (a[1] == b[1] == point[1]):
                compact[-1] = point
                continue
        compact.append(point)
    return compact


def _label_position(points: Sequence[tuple[float, float]], label_at: Mapping[str, Any] | None) -> tuple[float, float]:
    if label_at:
        segment = max(0, min(int(label_at.get("segment", 0)), len(points) - 2))
        position = float(label_at.get("position", 0.5))
    else:
        lengths = [abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in zip(points, points[1:])]
        segment = max(range(len(lengths)), key=lengths.__getitem__)
        position = 0.5
    start, end = points[segment], points[segment + 1]
    x = start[0] + (end[0] - start[0]) * position
    y = start[1] + (end[1] - start[1]) * position
    if label_at:
        x += float(label_at.get("dx", 0))
        y += float(label_at.get("dy", -8))
    else:
        y -= 8
    return x, y


def _safe_xml_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value)


def _render_edge(edge: Mapping[str, Any], node_map: Mapping[str, Mapping[str, Any]]) -> str:
    edge_id = _edge_id(edge)
    safe_id = _safe_xml_id(edge_id)
    style = str(edge.get("style", "data"))
    color, dash_pattern, meaning = EDGE_STYLE[style]
    points = _route_points(edge, node_map)
    path = " ".join(
        [f"M {points[0][0]:.1f} {points[0][1]:.1f}"]
        + [f"L {point[0]:.1f} {point[1]:.1f}" for point in points[1:]]
    )
    dash = f' stroke-dasharray="{dash_pattern}"' if dash_pattern else ""
    change_kind = edge.get("_change_kind")
    change_attrs = ""
    halo = ""
    if change_kind:
        change_attrs = f' data-change-kind="{escape(str(change_kind))}" data-change-ref="{escape(str(edge["_change_ref"]))}"'
        change_color = CHANGE_STYLE[str(change_kind)][0]
        halo = f'<path d="{path}" fill="none" stroke="{change_color}" stroke-width="7" opacity="0.14"/>'
    label_text = str(edge.get("label", ""))
    if edge.get("_change_ref"):
        label_text = f'{edge["_change_ref"]} · {label_text}' if label_text else str(edge["_change_ref"])
    label = ""
    if label_text:
        x, y = _label_position(points, edge.get("label_at"))
        label_width = max(36, min(260, 8 + len(label_text) * 6.6))
        label = (
            f'<g class="edge-label" transform="translate({x:.1f},{y:.1f})">'
            f'<rect x="{-label_width / 2:.1f}" y="-13" width="{label_width:.1f}" height="20" rx="7"/>'
            f'<text x="0" y="1">{escape(label_text)}</text></g>'
        )
    description = str(edge.get("_change_note", edge.get("label", meaning)))
    return (
        f'<g id="edge-{safe_id}" class="edge edge-{escape(style)}" data-edge-style="{escape(style)}"{change_attrs} '
        f'role="group" aria-labelledby="edge-{safe_id}-title edge-{safe_id}-desc">'
        f'<title id="edge-{safe_id}-title">{escape(str(edge["from"]))} to {escape(str(edge["to"]))}</title>'
        f'<desc id="edge-{safe_id}-desc">{escape(description)}</desc>{halo}'
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linejoin="round" '
        f'marker-end="url(#arrow-{escape(style)})"{dash}/>{label}</g>'
    )


def _render_node(node: Mapping[str, Any]) -> str:
    role = str(node["role"])
    scope = str(node["scope"])
    implementation = str(node["implementation"])
    evidence = str(node["evidence"])
    fill, accent, text_color = ROLE_STYLE[role]
    x, y = float(node["x"]), float(node["y"])
    width, height = float(node["width"]), float(node["height"])
    border_dash = ' stroke-dasharray="8 5"' if implementation == "planned" else ""
    opacity = "0.45" if implementation == "removed" else "1"
    details = list(node.get("details", []))
    detail_lines = "".join(
        f'<tspan x="{x + 18:.1f}" dy="17">{escape(str(line))}</tspan>' for line in details
    )
    evidence_badge = ""
    if evidence != "n_a":
        badge_fill = {"admitted": "#166534", "guarded": "#92400E", "blocked": "#4B5563"}[evidence]
        badge_width = 18 + len(evidence.upper()) * 7
        evidence_badge = (
            f'<rect x="{x + width - badge_width - 12:.1f}" y="{y + height - 27:.1f}" width="{badge_width:.1f}" height="19" rx="9.5" fill="{badge_fill}"/>'
            f'<text x="{x + width - badge_width / 2 - 12:.1f}" y="{y + height - 13:.1f}" class="micro-badge">{evidence.upper()}</text>'
        )
    implementation_badge = ""
    if implementation != "implemented":
        badge_fill = "#15803D" if implementation == "planned" else "#9F1239"
        badge_width = 18 + len(implementation.upper()) * 7
        implementation_badge = (
            f'<rect x="{x + 12:.1f}" y="{y + height - 27:.1f}" width="{badge_width:.1f}" height="19" rx="9.5" fill="{badge_fill}"/>'
            f'<text x="{x + 12 + badge_width / 2:.1f}" y="{y + height - 13:.1f}" class="micro-badge">{implementation.upper()}</text>'
        )
    else:
        implementation_badge = (
            f'<text x="{x + 16:.1f}" y="{y + height - 12:.1f}" class="scope-label">{escape(scope.replace("_", " ").upper())}</text>'
        )
    change_kind = node.get("_change_kind")
    change_overlay = ""
    change_attrs = ""
    if change_kind:
        change_color = CHANGE_STYLE[str(change_kind)][0]
        reference = str(node["_change_ref"])
        change_attrs = (
            f' data-change-kind="{escape(str(change_kind))}" data-change-ref="{escape(reference)}"'
        )
        change_overlay = (
            f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="{width + 10:.1f}" height="{height + 10:.1f}" rx="15" '
            f'fill="none" stroke="{change_color}" stroke-width="3" stroke-dasharray="8 5"/>'
            f'<rect x="{x + width - 45:.1f}" y="{y - 15:.1f}" width="45" height="24" rx="12" fill="{change_color}"/>'
            f'<text x="{x + width - 22.5:.1f}" y="{y + 1:.1f}" class="change-badge">{escape(reference)}</text>'
        )
    if implementation == "removed":
        change_overlay += (
            f'<path d="M {x + 10:.1f} {y + 10:.1f} L {x + width - 10:.1f} {y + height - 10:.1f} '
            f'M {x + width - 10:.1f} {y + 10:.1f} L {x + 10:.1f} {y + height - 10:.1f}" '
            f'stroke="#9F1239" stroke-width="4"/>'
        )
    canonical = node.get("_canonical", {})
    canonical_attrs = ""
    if canonical:
        canonical_attrs = (
            f' data-canonical-status="{escape(str(canonical["implementation"]))}"'
            f' data-canonical-evidence="{escape(str(canonical["evidence"]))}"'
        )
    description_parts = [str(node["label"]), *map(str, details), f"scope {scope}", f"implementation {implementation}"]
    if evidence != "n_a":
        description_parts.append(f"evidence {evidence}")
    if node.get("_change_note"):
        description_parts.append(str(node["_change_note"]))
    node_id = _safe_xml_id(str(node["id"]))
    return (
        f'<g id="node-{node_id}" class="node role-{escape(role)}" data-role="{escape(role)}" data-scope="{escape(scope)}" '
        f'data-implementation="{escape(implementation)}" data-evidence="{escape(evidence)}" data-status="{escape(evidence if evidence != "n_a" else implementation)}"'
        f'{canonical_attrs}{change_attrs} opacity="{opacity}" role="group" '
        f'aria-labelledby="node-{node_id}-title node-{node_id}-desc">'
        f'<title id="node-{node_id}-title">{escape(str(node["label"]))}</title>'
        f'<desc id="node-{node_id}-desc">{escape("; ".join(description_parts))}</desc>'
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="11" fill="{fill}" stroke="{accent}" stroke-width="2.2"{border_dash}/>'
        f'<rect x="{x:.1f}" y="{y:.1f}" width="7" height="{height:.1f}" rx="3.5" fill="{accent}"/>'
        f'<text x="{x + 18:.1f}" y="{y + 29:.1f}" class="node-title" fill="{text_color}">{escape(str(node["label"]))}</text>'
        f'<text x="{x + 18:.1f}" y="{y + 42:.1f}" class="node-detail" fill="{text_color}">{detail_lines}</text>'
        f'{implementation_badge}{evidence_badge}{change_overlay}</g>'
    )


def _render_sections(sections: Iterable[Mapping[str, Any]]) -> str:
    rendered = []
    for section in sections:
        description = str(section.get("description", ""))
        rendered.append(
            f'<g id="section-{escape(str(section["id"]))}">'
            f'<rect x="{section["x"]}" y="{section["y"]}" width="{section["width"]}" height="{section["height"]}" rx="18" class="section-box"/>'
            f'<text x="{section["x"] + 18}" y="{section["y"] + 31}" class="section-title">{escape(str(section["label"]))}</text>'
            + (
                f'<text x="{section["x"] + 18}" y="{section["y"] + 52}" class="section-description">{escape(description)}</text>'
                if description
                else ""
            )
            + "</g>"
        )
    return "".join(rendered)


def _render_legend(y: int) -> str:
    roles = ["data", "teacher", "encoder", "latent", "quantizer", "objective", "interface", "evaluator"]
    role_items = []
    x = 190
    for role in roles:
        fill, accent, _ = ROLE_STYLE[role]
        role_items.append(
            f'<rect x="{x}" y="{y - 15}" width="18" height="18" rx="4" fill="{fill}" stroke="{accent}"/>'
            f'<text x="{x + 25}" y="{y}" class="legend-label">{role}</text>'
        )
        x += 105 if role not in {"quantizer", "objective"} else 125
    return (
        f'<g id="legend"><text x="40" y="{y}" class="legend-title">Functional role</text>{"".join(role_items)}'
        f'<text x="40" y="{y + 35}" class="legend-title">Line / state</text>'
        f'<path d="M 190 {y + 30} L 235 {y + 30}" stroke="#526272" stroke-width="2.2" marker-end="url(#arrow-data)"/>'
        f'<text x="245" y="{y + 35}" class="legend-label">inference/data</text>'
        f'<path d="M 365 {y + 30} L 410 {y + 30}" stroke="#7A54A3" stroke-width="2.2" stroke-dasharray="9 6" marker-end="url(#arrow-training)"/>'
        f'<text x="420" y="{y + 35}" class="legend-label">training only</text>'
        f'<rect x="550" y="{y + 19}" width="46" height="20" rx="10" fill="#166534"/><text x="573" y="{y + 34}" class="micro-badge">ADMITTED</text>'
        f'<rect x="675" y="{y + 19}" width="46" height="20" rx="10" fill="#92400E"/><text x="698" y="{y + 34}" class="micro-badge">GUARDED</text>'
        f'<rect x="800" y="{y + 19}" width="46" height="20" rx="10" fill="#4B5563"/><text x="823" y="{y + 34}" class="micro-badge">BLOCKED</text>'
        f'<rect x="925" y="{y + 18}" width="74" height="23" rx="7" fill="none" stroke="#15803D" stroke-width="2" stroke-dasharray="7 4"/><text x="1008" y="{y + 35}" class="legend-label">planned</text>'
        f'</g>'
    )


def _wrap_callout(note: str, width: int = 96) -> list[str]:
    return textwrap.wrap(note, width=width, break_long_words=False, break_on_hyphens=False) or [""]


def _render_callouts(records: Sequence[Mapping[str, str]], start_y: int, width: int) -> tuple[str, int]:
    if not records:
        return "", start_y
    columns = 2
    column_width = (width - 120) / columns
    column_x = [40, 80 + column_width]
    per_column = (len(records) + columns - 1) // columns
    rendered = [
        f'<g id="plan-callouts"><rect x="30" y="{start_y}" width="{width - 60}" height="__HEIGHT__" rx="18" class="callout-panel"/>'
        f'<text x="50" y="{start_y + 34}" class="callout-title">Plan delta · proposed after-state</text>'
    ]
    bottoms = []
    for column in range(columns):
        y = start_y + 64
        for record in records[column * per_column : (column + 1) * per_column]:
            lines = _wrap_callout(str(record["note"]), 88)
            row_height = 34 + 17 * len(lines)
            color = CHANGE_STYLE[str(record["kind"])][0]
            callout_id = _safe_xml_id(f'{record["object_type"]}-{record["object_id"]}')
            rendered.append(
                f'<g id="callout-{callout_id}" role="group" aria-labelledby="callout-{callout_id}-title callout-{callout_id}-desc">'
                f'<title id="callout-{callout_id}-title">{escape(str(record["ref"]))}: {escape(str(record["object_id"]))}</title>'
                f'<desc id="callout-{callout_id}-desc">{escape(str(record["note"]))}</desc>'
                f'<rect x="{column_x[column]:.1f}" y="{y:.1f}" width="42" height="24" rx="12" fill="{color}"/>'
                f'<text x="{column_x[column] + 21:.1f}" y="{y + 16:.1f}" class="change-badge">{escape(str(record["ref"]))}</text>'
                f'<text x="{column_x[column] + 54:.1f}" y="{y + 15:.1f}" class="callout-object">{escape(str(record["object_type"]))} · {escape(str(record["object_id"]))}</text>'
                f'<text x="{column_x[column] + 54:.1f}" y="{y + 35:.1f}" class="callout-note">'
                + "".join(
                    f'<tspan x="{column_x[column] + 54:.1f}" dy="{0 if index == 0 else 17}">{escape(line)}</tspan>'
                    for index, line in enumerate(lines)
                )
                + "</text></g>"
            )
            y += row_height
        bottoms.append(y)
    panel_height = max(bottoms) - start_y + 18
    rendered[0] = rendered[0].replace("__HEIGHT__", f"{panel_height:.1f}")
    rendered.append("</g>")
    return "".join(rendered), int(start_y + panel_height)


def render_svg(
    spec: Mapping[str, Any],
    changes: Mapping[str, Any] | None = None,
    *,
    spec_source: str | None = None,
    changes_source: str | None = None,
) -> str:
    """Render a deterministic SVG from a canonical spec and optional overlay."""

    prepared_spec, prepared_changes = _prepare_inputs(spec, changes)
    _validate(prepared_spec, prepared_changes)
    view, records = _compose_view(prepared_spec, prepared_changes)
    width = int(view["width"])
    base_height = int(view["height"])
    footer_y = int(view.get("footer_y", base_height - 80))
    callout_svg = ""
    canvas_height = base_height
    if records:
        callout_svg, callout_bottom = _render_callouts(records, base_height + 20, width)
        canvas_height = callout_bottom + 24
    callout_block = f"  {callout_svg}\n" if callout_svg else ""
    nodes = [dict(node) for node in view["nodes"]]
    for node in nodes:
        _validate_node(node, width, footer_y)
    _validate_node_overlaps(nodes)
    node_map = {str(node["id"]): node for node in nodes}
    metadata = {
        "schema": view["schema"],
        "change_plan": view.get("_plan_id"),
        "spec_source": spec_source or str(view.get("source", "in-memory")),
        "changes_source": changes_source,
    }
    section_svg = _render_sections(view["sections"])
    edge_svg = "".join(_render_edge(edge, node_map) for edge in view["edges"])
    node_svg = "".join(_render_node(node) for node in nodes)
    banner = str(view.get("banner", ""))
    banner_svg = ""
    if banner:
        banner_svg = (
            f'<g id="evidence-boundary"><rect x="40" y="96" width="{width - 80}" height="43" rx="10" class="banner-box"/>'
            f'<text x="58" y="123" class="banner-text">{escape(banner)}</text></g>'
        )
    banner_block = f"  {banner_svg}\n" if banner_svg else ""
    marker_defs = "".join(
        f'<marker id="arrow-{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{style[0]}"/></marker>'
        for name, style in EDGE_STYLE.items()
    )
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{canvas_height}" viewBox="0 0 {width} {canvas_height}" role="img" aria-labelledby="svg-title svg-desc">
  <title id="svg-title">{escape(str(view["title"]))}</title>
  <desc id="svg-desc">{escape(str(view["subtitle"]))}</desc>
  <metadata>{escape(json.dumps(metadata, sort_keys=True))}</metadata>
  <defs>{marker_defs}</defs>
  <style>
    text {{ font-family: "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", Inter, ui-sans-serif, system-ui, sans-serif; }}
    .page-title {{ font-size: 29px; font-weight: 750; fill: #13202B; }}
    .page-subtitle {{ font-size: 15px; fill: #4B5B68; }}
    .banner-box {{ fill: #FFF8E8; stroke: #D68B00; stroke-width: 1.4; }}
    .banner-text {{ font-size: 13.5px; font-weight: 650; fill: #5B3A00; }}
    .section-box {{ fill: #FAFBFC; stroke: #C9D2D9; stroke-width: 1.4; }}
    .section-title {{ font-size: 17px; font-weight: 750; fill: #2D3A45; }}
    .section-description {{ font-size: 12px; fill: #667784; }}
    .node-title {{ font-size: 15px; font-weight: 750; }}
    .node-detail {{ font-size: 12.5px; }}
    .scope-label {{ font-size: 9.5px; font-weight: 750; fill: #53616D; letter-spacing: .35px; }}
    .micro-badge {{ font-size: 8.5px; font-weight: 800; text-anchor: middle; fill: #FFFFFF; }}
    .edge-label rect {{ fill: #FFFFFF; stroke: #CDD5DB; stroke-width: 1; opacity: .96; }}
    .edge-label text {{ font-size: 10.5px; text-anchor: middle; fill: #34424E; }}
    .change-badge {{ font-size: 10px; font-weight: 850; text-anchor: middle; fill: #FFFFFF; }}
    .legend-title {{ font-size: 12px; font-weight: 750; fill: #34424E; }}
    .legend-label {{ font-size: 10.5px; fill: #465661; }}
    .callout-panel {{ fill: #F8FAFC; stroke: #BFCAD2; stroke-width: 1.4; }}
    .callout-title {{ font-size: 17px; font-weight: 750; fill: #2D3A45; }}
    .callout-object {{ font-size: 12px; font-weight: 750; fill: #2D3A45; }}
    .callout-note {{ font-size: 11.5px; fill: #53616D; }}
  </style>
  <rect width="100%" height="100%" fill="#FFFFFF"/>
  <text x="40" y="49" class="page-title">{escape(str(view["title"]))}</text>
  <text x="40" y="77" class="page-subtitle">{escape(str(view["subtitle"]))}</text>
{banner_block}  {section_svg}
  <g id="edges">{edge_svg}</g>
  <g id="nodes">{node_svg}</g>
  {_render_legend(footer_y)}
{callout_block}</svg>
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
    try:
        spec_source = str(args.spec.resolve().relative_to(REPO_ROOT))
    except ValueError:
        spec_source = str(args.spec.resolve())
    changes_source = None
    if args.changes:
        try:
            changes_source = str(args.changes.resolve().relative_to(REPO_ROOT))
        except ValueError:
            changes_source = str(args.changes.resolve())
    rendered = render_svg(spec, changes, spec_source=spec_source, changes_source=changes_source)
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
