import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = PROJECT_ROOT / "docs/figures"
SVG_PATH = FIGURE_ROOT / "experiment_plan.svg"
PNG_PATH = FIGURE_ROOT / "experiment_plan.png"
SOURCE_PATH = FIGURE_ROOT / "experiment_plan_status.json"
ALT_PATH = FIGURE_ROOT / "experiment_plan.alt.txt"
MANIFEST_PATH = FIGURE_ROOT / "experiment_plan.manifest.json"
SVG_NS = {"svg": "http://www.w3.org/2000/svg"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_experiment_plan_svg_preserves_scientific_status_and_accessibility():
    root = ET.parse(SVG_PATH).getroot()
    metadata_element = root.find("svg:metadata", SVG_NS)
    assert metadata_element is not None
    metadata = json.loads(metadata_element.text)

    assert root.attrib["role"] == "img"
    for label_id in root.attrib["aria-labelledby"].split():
        element = root.find(f".//*[@id='{label_id}']")
        assert element is not None
        assert "".join(element.itertext()).strip()

    assert metadata["schema"] == "experiment-plan-v1"
    assert metadata["decision"] == {
        "promotion_eligible": False,
        "next_action": "do_not_enter_r2_p",
        "protected_subjects_24_29": "closed",
    }
    assert {
        "docs/physiology_semantic_tokenizer/06_EXPERIMENT_LOG.md",
        "docs/physiology_semantic_tokenizer/analysis/20260728_R_SERIES_EXPERIMENT_REPORT.md",
    } <= set(metadata["authoritative_sources"])

    nodes = {
        node.attrib["data-stage"]: node
        for node in root.findall(".//*[@data-stage]")
    }
    required = {
        "E0",
        "E1",
        "E2",
        "R0-P",
        "R1-D",
        "R1-P",
        "R2-D",
        "D1B",
        "R2-P",
        "R3",
        "R4",
        "R5",
        "R6A",
        "R6B",
        "R7",
    }
    assert required <= nodes.keys()

    assert nodes["E2"].attrib["data-status"] == "completed"
    assert (
        nodes["E2"].attrib["data-outcome"]
        == "no_semantic_row_admitted_retain_T0"
    )
    assert nodes["R0-P"].attrib["data-status"] == "completed"
    assert nodes["R0-P"].attrib["data-outcome"] == "negative"
    assert nodes["R1-D"].attrib["data-outcome"] == "exploratory"
    assert nodes["R1-P"].attrib["data-status"] == "failed"
    assert nodes["R2-D"].attrib["data-status"] == "failed"
    assert nodes["D1B"].attrib["data-status"] == "undetermined"
    assert nodes["R2-P"].attrib["data-status"] == "blocked"

    for stage in ("R2-P", "R3", "R4", "R5", "R6A", "R6B", "R7"):
        assert nodes[stage].attrib["data-authorized"] == "false"
        assert nodes[stage].attrib["data-status"] == "blocked"

    for node in nodes.values():
        assert node.attrib["role"] == "group"
        label_id = node.attrib["aria-labelledby"]
        label = root.find(f".//*[@id='{label_id}']")
        assert label is not None
        assert "".join(label.itertext()).strip()

    edges = {
        (edge.attrib["data-from"], edge.attrib["data-to"])
        for edge in root.findall(".//*[@data-from][@data-to]")
    }
    assert {
        ("R1-P", "STOP"),
        ("R2-D", "STOP"),
        ("STOP", "R2-P"),
    } <= edges


def test_experiment_plan_outputs_match_source_manifest():
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    campaign = source["live_status"]["comparison_campaign"]

    assert source["schema"] == "experiment_plan_source_v1"
    assert manifest["schema"] == "experiment_plan_figure_manifest_v1"
    assert manifest["source_sha256"] == _sha256(SOURCE_PATH)
    assert manifest["snapshot_at"] == source["snapshot_at"]

    for path in (SVG_PATH, PNG_PATH, ALT_PATH):
        relative = str(path.relative_to(PROJECT_ROOT))
        assert manifest["outputs"][relative]["sha256"] == _sha256(path)
        assert manifest["outputs"][relative]["bytes"] == path.stat().st_size

    with Image.open(PNG_PATH) as image:
        assert image.info["dpi"][0] >= 299
        assert image.info["dpi"][1] >= 299
        assert image.width >= 7000
        assert image.height >= 3000

    alt_text = ALT_PATH.read_text(encoding="utf-8")
    for phrase in (
        "R1-P and R2-D failed",
        "blocks R2-P through R7",
        (
            "joint campaign completed "
            f"{campaign['completed_job_count']}/"
            f"{campaign['expected_job_count']} protected jobs with zero failures"
        ),
        "22 ready-with-note, 12 rejected, 2 overlap-only, and 6 unsupported terminals",
    ):
        assert phrase in alt_text

    root = ET.parse(SVG_PATH).getroot()
    metadata = json.loads(root.find("svg:metadata", SVG_NS).text)
    assert metadata["live_status"] == source["live_status"]

    nodes = {
        node.attrib["data-stage"]: node
        for node in root.findall(".//*[@data-stage]")
    }
    assert campaign["protected_test_opened"] is True
    assert campaign["unblind_authorized"] is True
    assert campaign["cell_count"] == 42
    assert sum(campaign["terminal_counts"].values()) == campaign["cell_count"]

    for stage in (
        "MATRIX",
        "PUBLIC",
        "CAMPAIGN",
        "UNBLIND",
        "ACCEPTANCE",
        "FINAL-TABLE",
    ):
        assert nodes[stage].attrib["data-status"] == "completed"
        assert nodes[stage].attrib["data-authorized"] == "true"
