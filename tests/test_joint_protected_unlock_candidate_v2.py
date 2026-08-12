from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "comparative_methods/build_joint_protected_unlock_candidate_v2.py"
CANDIDATE = REPO_ROOT / "comparative_methods/evidence/joint_protected_unlock_candidate_v2.json"


def _module():
    spec = importlib.util.spec_from_file_location("joint_unlock_candidate_v2", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_is_complete_and_strictly_non_authorizing() -> None:
    candidate = _module().build_candidate()
    assert candidate["status"] == "ready_for_human_unlock_review"
    assert candidate["authorization"] == {
        "human_review_status": "pending",
        "protected_evaluation_authorized": False,
        "protected_test_opened": False,
        "target_dataset_exposure": False,
        "table_admissible": False,
    }
    assert candidate["global_alignment_audit"]["status"] == "pass"
    assert candidate["global_alignment_audit"]["cell_count"] == 42
    assert candidate["global_alignment_audit"]["pass_cell_count"] == 36
    assert candidate["global_alignment_audit"]["unsupported_cell_count"] == 6
    assert sum(cell["adapter_eligible_for_unlock_review"] for cell in candidate["cells"]) == 36
    assert all(cell["cell_status"] in {"pass", "unsupported"} for cell in candidate["cells"])


def test_retained_candidate_exactly_matches_builder() -> None:
    retained = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    assert retained == _module().build_candidate()
