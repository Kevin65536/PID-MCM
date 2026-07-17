import copy
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from experiments.scripts.render_physiology_semantic_architecture import render_svg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / "docs/physiology_semantic_tokenizer/architecture/physiology_semantic_architecture.json"
SVG_PATH = PROJECT_ROOT / "docs/physiology_semantic_tokenizer/figures/physiology_semantic_architecture.svg"
SPEC_SOURCE = "docs/physiology_semantic_tokenizer/architecture/physiology_semantic_architecture.json"
REGISTERED_PLANS = [
    "measurement_first_input_contract_plan",
    "shared_state_reconstruction_bound_plan",
]


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _spec():
    return _load(SPEC_PATH)


def _xml(svg: str):
    return ET.fromstring(svg)


def test_current_and_existing_plan_svgs_are_deterministic_and_not_stale():
    spec = _spec()
    assert SVG_PATH.read_text(encoding="utf-8") == render_svg(spec, spec_source=SPEC_SOURCE)
    for plan_id in REGISTERED_PLANS:
        changes_path = PROJECT_ROOT / f"docs/physiology_semantic_tokenizer/architecture/{plan_id}.json"
        svg_path = PROJECT_ROOT / f"docs/physiology_semantic_tokenizer/figures/plans/{plan_id}.svg"
        assert svg_path.read_text(encoding="utf-8") == render_svg(
            spec,
            _load(changes_path),
            spec_source=SPEC_SOURCE,
            changes_source=f"docs/physiology_semantic_tokenizer/architecture/{plan_id}.json",
        )


def test_current_svg_preserves_baseline_runtime_content_and_exposes_visual_axes():
    root = _xml(SVG_PATH.read_text(encoding="utf-8"))
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert root.attrib["role"] == "img"
    assert root.find("svg:title", namespace).text == "Physiology-Semantic Tokenizer — Current Runtime Architecture"
    assert "E0-v2 validation blocked" in root.find("svg:desc", namespace).text

    required = {
        "loader",
        "teacher_adapter",
        "eeg_quantizer",
        "fnirs_quantizer",
        "eeg_context",
        "fnirs_context",
        "trainer_gate",
        "export",
        "consumers",
        "p6_coupling",
    }
    xml_ids = {element.attrib.get("id") for element in root.iter()}
    assert {f"node-{node_id}" for node_id in required}.issubset(xml_ids)
    assert root.find(".//*[@id='node-trainer_gate']").attrib["data-status"] == "guarded"
    assert root.find(".//*[@id='node-p6_coupling']").attrib["data-status"] == "blocked"


def test_legacy_geometry_is_expanded_in_memory_without_mutating_design_source():
    spec = _spec()
    before = copy.deepcopy(spec)
    root = _xml(render_svg(spec))
    assert spec == before
    assert int(root.attrib["width"]) > int(spec["width"])
    assert int(root.attrib["height"]) > int(spec["height"])


def test_generic_v2_overlay_composes_after_state_without_updating_any_plan_file():
    changes = {
        "schema": "physiology_semantic_architecture_changes_v2",
        "plan_id": "rendering_test_only",
        "title": "Proposed After-State · Rendering Test",
        "subtitle": "in-memory test overlay",
        "summary": "test",
        "changes": [
            {
                "node_id": "eeg_quantizer",
                "kind": "modify",
                "note": "exercise visual replacement",
                "replace": {"label": "Proposed vocabulary", "implementation": "planned"},
            }
        ],
        "add_nodes": [],
        "edge_changes": [],
        "add_edges": [],
    }
    spec = _spec()
    before = copy.deepcopy(spec)
    root = _xml(render_svg(spec, changes))
    assert spec == before
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert root.find("svg:title", namespace).text == "Proposed After-State · Rendering Test"
    node = root.find(".//*[@id='node-eeg_quantizer']")
    assert node.attrib["data-change-kind"] == "modify"
    assert node.attrib["data-canonical-status"] == "implemented"
    assert node.attrib["data-implementation"] == "planned"


def test_edges_have_stable_accessible_ids_and_orthogonal_paths():
    root = _xml(SVG_PATH.read_text(encoding="utf-8"))
    edge_groups = [
        element
        for element in root.iter()
        if element.tag.endswith("g") and element.attrib.get("id", "").startswith("edge-")
    ]
    edge_ids = [element.attrib["id"] for element in edge_groups]
    assert edge_groups
    assert len(edge_ids) == len(set(edge_ids))
    for group in edge_groups:
        assert group.attrib["aria-labelledby"]
        path = next(child for child in group if child.tag.endswith("path") and child.attrib.get("marker-end"))
        assert " L " in path.attrib["d"]
        assert " C " not in path.attrib["d"]


def test_existing_v1_plan_gets_dynamic_callouts_without_source_changes():
    plan_id = "measurement_first_input_contract_plan"
    changes_path = PROJECT_ROOT / f"docs/physiology_semantic_tokenizer/architecture/{plan_id}.json"
    changes = _load(changes_path)
    before = copy.deepcopy(changes)
    root = _xml(render_svg(_spec(), changes))
    assert changes == before
    assert root.find(".//*[@id='node-optional_target_bank']").attrib["data-change-kind"] == "add"
    assert root.find(".//*[@id='callout-node-optional_target_bank']") is not None
    assert int(root.attrib["height"]) > int(_spec()["height"])


def test_validation_rejects_bad_replacement_unknown_endpoint_and_duplicate_ids():
    bad_replace = {
        "schema": "physiology_semantic_architecture_changes_v2",
        "plan_id": "bad",
        "summary": "bad",
        "changes": [
            {
                "node_id": "eeg_quantizer",
                "kind": "modify",
                "note": "bad",
                "replace": {"id": "replacement_is_forbidden"},
            }
        ],
    }
    with pytest.raises(ValueError, match="Unsupported node replacement fields"):
        render_svg(_spec(), bad_replace)

    bad_endpoint = copy.deepcopy(_spec())
    bad_endpoint["edges"][0]["to"] = "missing"
    with pytest.raises(ValueError, match="Unknown edge endpoint"):
        render_svg(bad_endpoint)

    duplicate = copy.deepcopy(_spec())
    duplicate["nodes"][1]["id"] = duplicate["nodes"][0]["id"]
    with pytest.raises(ValueError, match="node ids must be unique"):
        render_svg(duplicate)
