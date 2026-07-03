import json
from pathlib import Path
from xml.etree import ElementTree as ET

from experiments.scripts.render_physiology_semantic_architecture import render_svg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / "docs/physiology_semantic_tokenizer/architecture/physiology_semantic_architecture.json"
SVG_PATH = PROJECT_ROOT / "docs/physiology_semantic_tokenizer/figures/physiology_semantic_architecture.svg"


def _spec():
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def test_current_svg_is_deterministic_and_not_stale():
    assert SVG_PATH.read_text(encoding="utf-8") == render_svg(_spec())


def test_current_svg_is_accessible_and_covers_runtime_boundaries():
    root = ET.fromstring(SVG_PATH.read_text(encoding="utf-8"))
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert root.attrib["role"] == "img"
    assert root.find("svg:title", namespace).text
    assert root.find("svg:desc", namespace).text

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


def test_change_overlay_marks_modify_remove_and_added_nodes():
    changes = {
        "schema": "physiology_semantic_architecture_changes_v1",
        "plan_id": "test_plan",
        "summary": "test",
        "changes": [
            {"node_id": "eeg_quantizer", "kind": "modify", "note": "change quantizer"},
            {"node_id": "p6_coupling", "kind": "remove", "note": "remove blocked node"},
        ],
        "add_nodes": [
            {
                "id": "future_probe",
                "label": "Future probe",
                "details": ["plan-only component"],
                "x": 1450,
                "y": 830,
                "width": 100,
                "height": 60,
                "status": "planned",
            }
        ],
        "add_edges": [{"from": "export", "to": "future_probe"}],
    }
    root = ET.fromstring(render_svg(_spec(), changes))
    assert root.find(".//*[@id='node-eeg_quantizer']").attrib["data-change-kind"] == "modify"
    assert root.find(".//*[@id='node-p6_coupling']").attrib["data-change-kind"] == "remove"
    assert root.find(".//*[@id='node-future_probe']").attrib["data-change-kind"] == "add"
