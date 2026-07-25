import json
from pathlib import Path
from xml.etree import ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SVG_PATH = PROJECT_ROOT / "docs/figures/project_evolution_map.svg"


def test_project_evolution_map_records_architecture_return_and_current_snapshot():
    root = ET.parse(SVG_PATH).getroot()
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    metadata = json.loads(root.find("svg:metadata", namespace).text)

    assert root.attrib["role"] == "img"
    assert root.attrib["data-snapshot"] == "main@6d6c648"
    assert metadata["reachable_commits"] == 319
    assert metadata["main_commits"] == 310
    assert metadata["semantic_nodes"] == 38
    assert metadata["causal_edges"] == 21

    for node_id in ("node-d8", "node-t8", "node-m12", "node-m13"):
        node = root.find(f".//*[@id='{node_id}']")
        assert node is not None
        labelled_by = node.attrib["aria-labelledby"].split()
        assert all(root.find(f".//*[@id='{label_id}']") is not None for label_id in labelled_by)

    for edge_id in ("edge-f20", "edge-f21"):
        assert root.find(f".//*[@id='{edge_id}']") is not None
