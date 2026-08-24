import copy
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from experiments.scripts.render_physiology_semantic_architecture import render_svg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / "docs/physiology_semantic_tokenizer/architecture/physiology_semantic_architecture.json"
DRAWIO_PATH = PROJECT_ROOT / (
    "docs/physiology_semantic_tokenizer/architecture/physiology_semantic_architecture.drawio"
)
SVG_PATH = PROJECT_ROOT / "docs/physiology_semantic_tokenizer/figures/physiology_semantic_architecture.svg"
RUNTIME_OVERVIEW_DRAWIO_PATH = PROJECT_ROOT / (
    "docs/physiology_semantic_tokenizer/architecture/physiology_semantic_runtime_overview.drawio"
)
RUNTIME_OVERVIEW_SVG_PATH = PROJECT_ROOT / (
    "docs/physiology_semantic_tokenizer/figures/physiology_semantic_runtime_overview.svg"
)
SPEC_SOURCE = "docs/physiology_semantic_tokenizer/architecture/physiology_semantic_architecture.json"
EXPLORATION_SPEC_PATH = PROJECT_ROOT / (
    "docs/physiology_semantic_tokenizer/architecture/observation_source_exploration_v2.json"
)
EXPLORATION_DRAWIO_PATH = PROJECT_ROOT / (
    "docs/physiology_semantic_tokenizer/architecture/observation_source_exploration_v2.drawio"
)
EXPLORATION_SVG_PATH = PROJECT_ROOT / (
    "docs/physiology_semantic_tokenizer/figures/plans/observation_source_exploration_v2.svg"
)
EXPLORATION_ALT_PATH = PROJECT_ROOT / (
    "docs/physiology_semantic_tokenizer/figures/plans/observation_source_exploration_v2.alt.txt"
)
EXPLORATION_SPEC_SOURCE = (
    "docs/physiology_semantic_tokenizer/architecture/observation_source_exploration_v2.json"
)
REGISTERED_PLANS = [
    "measurement_first_input_contract_plan",
    "physical_teacher_gradient_entry_plan",
    "shared_driver_semantic_return_plan",
    "shared_state_reconstruction_bound_plan",
]


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _spec():
    return _load(SPEC_PATH)


def _exploration_spec():
    return _load(EXPLORATION_SPEC_PATH)


def _xml(svg: str):
    return ET.fromstring(svg)


def _drawio_cells(root):
    return {
        cell.attrib["id"]: (cell.attrib.get("value", ""), cell.attrib.get("style", ""))
        for cell in root.iter("mxCell")
        if "id" in cell.attrib
    }


def test_existing_plan_svgs_are_deterministic_and_not_stale():
    spec = _spec()
    for plan_id in REGISTERED_PLANS:
        changes_path = PROJECT_ROOT / f"docs/physiology_semantic_tokenizer/architecture/{plan_id}.json"
        svg_path = PROJECT_ROOT / f"docs/physiology_semantic_tokenizer/figures/plans/{plan_id}.svg"
        assert svg_path.read_text(encoding="utf-8") == render_svg(
            spec,
            _load(changes_path),
            spec_source=SPEC_SOURCE,
            changes_source=f"docs/physiology_semantic_tokenizer/architecture/{plan_id}.json",
        )


def test_runtime_renderer_preserves_baseline_content_and_exposes_visual_axes():
    root = _xml(render_svg(_spec(), spec_source=SPEC_SOURCE))
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert root.attrib["role"] == "img"
    assert root.find("svg:title", namespace).text == "Physiology-Semantic Tokenizer — Current Runtime Architecture"
    assert "Implemented E2-compatible runtime" in root.find("svg:desc", namespace).text
    assert "fixed K=128 health passed" in root.find("svg:desc", namespace).text
    assert "no E2 teacher semantic row admitted" in root.find("svg:desc", namespace).text

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
    root = _xml(render_svg(_spec(), spec_source=SPEC_SOURCE))
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


def test_registered_plan_labels_keep_historical_roles_visible():
    expected = {
        "measurement_first_input_contract_plan": (
            "Merged Historical Overlay · Measurement-First Input Contract",
            "MERGED HISTORICAL OVERLAY",
        ),
        "physical_teacher_gradient_entry_plan": (
            "Superseded Historical Plan · Coupling-Aware Foundation Pipeline",
            "SUPERSEDED HISTORICAL PLAN",
        ),
        "shared_driver_semantic_return_plan": (
            "Historical Pre-Gate Plan · Shared-Driver Semantic VQ",
            "HISTORICAL PRE-GATE PLAN",
        ),
        "shared_state_reconstruction_bound_plan": (
            "Diagnostic-Only Historical Overlay · Shared-State Reconstruction Bound",
            "DIAGNOSTIC-ONLY HISTORICAL OVERLAY",
        ),
    }
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    for plan_id, (title, banner_prefix) in expected.items():
        changes_path = PROJECT_ROOT / f"docs/physiology_semantic_tokenizer/architecture/{plan_id}.json"
        changes = _load(changes_path)
        root = _xml(render_svg(_spec(), changes))
        assert root.find("svg:title", namespace).text == title
        boundary = root.find(".//*[@id='evidence-boundary']")
        assert boundary is not None
        assert "".join(boundary.itertext()).startswith(banner_prefix)


def test_physical_teacher_plan_keeps_preservation_discovery_and_certificate_distinct():
    plan_id = "physical_teacher_gradient_entry_plan"
    changes = _load(
        PROJECT_ROOT / f"docs/physiology_semantic_tokenizer/architecture/{plan_id}.json"
    )
    root = _xml(render_svg(_spec(), changes))
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert root.find("svg:title", namespace).text == (
        "Superseded Historical Plan · Coupling-Aware Foundation Pipeline"
    )
    assert root.find(".//*[@id='evidence-boundary']") is not None
    assert root.find(".//*[@id='node-coupling_shaper']").attrib["data-implementation"] == "planned"
    assert root.find(".//*[@id='node-p6_coupling']").attrib["data-evidence"] == "blocked"
    assert root.find(".//*[@id='edge-shaper--eeg-gradient']").attrib["data-edge-style"] == (
        "gradient"
    )
    assert root.find(".//*[@id='edge-foundation--certificate']").attrib[
        "data-edge-style"
    ] == "evaluation"


def test_shared_driver_return_plan_keeps_independent_k128_and_scoped_external_evaluation():
    plan_id = "shared_driver_semantic_return_plan"
    changes = _load(
        PROJECT_ROOT / f"docs/physiology_semantic_tokenizer/architecture/{plan_id}.json"
    )
    root = _xml(render_svg(_spec(), changes))
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert root.find("svg:title", namespace).text == (
        "Historical Pre-Gate Plan · Shared-Driver Semantic VQ"
    )
    assert root.find(".//*[@id='node-eeg_quantizer']").attrib["data-implementation"] == (
        "implemented"
    )
    assert root.find(".//*[@id='node-fnirs_quantizer']").attrib["data-implementation"] == (
        "implemented"
    )
    assert root.find(".//*[@id='node-eeg_context']").attrib["data-implementation"] == (
        "planned"
    )
    assert root.find(".//*[@id='node-semantic_losses']").attrib["data-implementation"] == (
        "planned"
    )
    assert root.find(".//*[@id='node-eeg_residual']").attrib["data-implementation"] == (
        "removed"
    )
    assert root.find(".//*[@id='node-p6_coupling']").attrib["data-evidence"] == "blocked"
    assert root.find(".//*[@id='node-consumers-title']").text == "R6A development evaluator"
    assert root.find(".//*[@id='node-p6_coupling-title']").text == "R6B prospective cutoff"
    assert root.find(".//*[@id='edge-export--p6_coupling']").attrib[
        "data-edge-style"
    ] == "evaluation"


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


@pytest.mark.parametrize(
    ("drawio_path", "svg_path"),
    [
        (DRAWIO_PATH, SVG_PATH),
        (RUNTIME_OVERVIEW_DRAWIO_PATH, RUNTIME_OVERVIEW_SVG_PATH),
        (EXPLORATION_DRAWIO_PATH, EXPLORATION_SVG_PATH),
    ],
)
def test_drawio_owned_svgs_embed_current_source_and_shared_style(drawio_path, svg_path):
    source_root = ET.parse(drawio_path).getroot()
    svg_root = _xml(svg_path.read_text(encoding="utf-8"))
    embedded_root = ET.fromstring(svg_root.attrib["content"])
    assert _drawio_cells(embedded_root) == _drawio_cells(source_root)
    assert svg_root.attrib["role"] == "img"
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert svg_root.find("svg:title", namespace) is not None
    assert svg_root.find("svg:desc", namespace) is not None
    styles = " ".join(style for _, style in _drawio_cells(source_root).values())
    assert "fontFamily=Helvetica" in styles
    assert "rounded=1" in styles
    assert "shadow=1" not in styles
    values = " ".join(value for value, _ in _drawio_cells(source_root).values())
    assert "PID-MCM" not in values
    assert "planned target" not in values.lower()
    assert "private" not in values.lower()
    assert "innovation path" not in values.lower()
    if drawio_path == DRAWIO_PATH:
        assert "Optional contribution probe" in values
        assert "architecture remains revisable" in values


def test_renderer_uses_the_shared_drawio_visual_language():
    svg = render_svg(_spec(), spec_source=SPEC_SOURCE)
    assert 'font-family: Helvetica' in svg
    assert '.banner-box { fill: #F1F5F9; stroke: #CBD5E1;' in svg
    assert '.section-box { fill: #FBFCFE; stroke: #B9C9D8;' in svg
    assert 'rx="16"' in svg
    assert '#E9F2FF' in svg
    assert '#FFF4D6' in svg
    assert '#F1ECFA' in svg
    assert '#E5F6EF' in svg


def test_v2_exploration_visual_and_alt_text_are_distinct_from_runtime():
    exploration = _exploration_spec()
    assert exploration["schema"] == "physiology_semantic_architecture_v2"
    assert EXPLORATION_DRAWIO_PATH.exists()
    assert EXPLORATION_SVG_PATH != SVG_PATH
    alt = EXPLORATION_ALT_PATH.read_text(encoding="utf-8")
    assert alt.startswith("Exploratory observation–source design note, not runtime:")
    assert "No measured values are shown." in alt


def test_v2_exploration_keeps_paths_independent_without_freezing_decomposition():
    exploration = _exploration_spec()
    edges = {edge["id"]: edge for edge in exploration["edges"]}
    required_edges = {
        "measured-x-e--target-adapter-e",
        "measured-x-f--target-adapter-f",
        "target-adapter-e--self-teacher-e",
        "target-adapter-f--self-teacher-f",
        "croce-joint-candidate--source-inference-e",
        "croce-joint-candidate--source-inference-f",
        "self-teacher-e--source-inference-e",
        "self-teacher-e--observation-inference-e",
        "self-teacher-f--source-inference-f",
        "self-teacher-f--observation-inference-f",
        "fine-codebook-e--coarse-aggregator-e",
        "fine-codebook-f--coarse-aggregator-f",
        "coarse-aggregator-e--endpoint-aligned-g-tau",
        "coarse-aggregator-f--endpoint-aligned-g-tau",
        "endpoint-aligned-g-tau--conditional-probe",
        "observation-e--observation-objective",
        "observation-f--observation-objective",
        "observation-objective--conditional-probe",
        "endpoint-aligned-g-tau--evaluation-preregistration",
        "preregister--heldout-proper-score-null",
    }
    assert required_edges.issubset(edges)
    assert len(edges) == len(exploration["edges"])
    assert len(edges) == len({edge["id"] for edge in exploration["edges"]})
    assert all(edge.get("route") for edge in exploration["edges"])
    assert edges["measured-x-e--croce-joint-candidate"]["style"] == "guarded"
    assert edges["measured-x-f--croce-joint-candidate"]["style"] == "guarded"
    assert edges["croce-joint-candidate--source-inference-e"]["style"] == "guarded"
    assert edges["croce-joint-candidate--source-inference-f"]["style"] == "guarded"
    assert "no inference input" in edges[
        "croce-joint-candidate--source-inference-e"
    ]["label"]
    assert edges["preregister--heldout-proper-score-null"]["style"] == "evaluation"
    assert not any(
        edge["from"] == "conditional_contribution_probe"
        and edge["to"] == "evaluation_preregistration"
        for edge in exploration["edges"]
    )
    node_ids = {node["id"] for node in exploration["nodes"]}
    assert {
        "measured_x_e",
        "measured_x_f",
        "target_adapter_e",
        "target_adapter_f",
        "self_teacher_e",
        "self_teacher_f",
        "croce_joint_candidate",
        "source_inference_e",
        "observation_inference_e",
        "source_inference_f",
        "observation_inference_f",
        "fine_codebook_e",
        "coarse_aggregator_e",
        "fine_codebook_f",
        "coarse_aggregator_f",
        "endpoint_aligned_g_tau",
        "observation_objective",
        "conditional_contribution_probe",
        "evaluation_preregistration",
        "heldout_proper_score_null",
    }.issubset(node_ids)


def test_v2_exploration_status_axes_and_accessibility_are_explicit():
    exploration = _exploration_spec()
    assert all(node["implementation"] == "planned" for node in exploration["nodes"])
    croce = next(
        node for node in exploration["nodes"] if node["id"] == "croce_joint_candidate"
    )
    assert croce["scope"] == "training_only"
    assert croce["role"] == "teacher"
    probe = next(
        node
        for node in exploration["nodes"]
        if node["id"] == "conditional_contribution_probe"
    )
    assert probe["evidence"] == "n_a"
    assert "may be replaced or removed" in " ".join(probe["details"])
    assert exploration["title"] == "Observation–Source Candidate Exploration"
    root = _xml(render_svg(exploration, spec_source=EXPLORATION_SPEC_SOURCE))
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert root.attrib["role"] == "img"
    assert "Replaceable candidates" in root.find("svg:desc", namespace).text
    boundary = root.find(".//*[@id='evidence-boundary']")
    assert boundary is not None
    assert "no method identity or decomposition is frozen" in "".join(boundary.itertext())
    node_groups = [
        element
        for element in root.iter()
        if element.tag.endswith("g") and element.attrib.get("id", "").startswith("node-")
    ]
    assert len(node_groups) == len(exploration["nodes"])
    for group in node_groups:
        assert group.attrib["aria-labelledby"]
        assert group.find("svg:title", namespace) is not None
        assert group.find("svg:desc", namespace) is not None
        assert group.attrib["data-implementation"] == "planned"
        assert group.attrib["data-evidence"] in {"guarded", "n_a"}
    edge_groups = [
        element
        for element in root.iter()
        if element.tag.endswith("g") and element.attrib.get("id", "").startswith("edge-")
    ]
    assert len(edge_groups) == len(exploration["edges"])
    assert len({group.attrib["id"] for group in edge_groups}) == len(edge_groups)
    for group in edge_groups:
        assert group.attrib["aria-labelledby"]
        assert group.find("svg:title", namespace) is not None
        assert group.find("svg:desc", namespace) is not None
        path = next(
            child
            for child in group
            if child.tag.endswith("path") and child.attrib.get("marker-end")
        )
        assert " L " in path.attrib["d"]
        assert " C " not in path.attrib["d"]
