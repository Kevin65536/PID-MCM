import copy
from pathlib import Path

import pytest

from src.utils.project_state import (
    DEFAULT_REGISTRY,
    ProjectStateError,
    current_records,
    current_snapshot,
    load_registry,
    render_agent_summary,
    render_readme_block,
    render_status_markdown,
    validate_registry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _registry():
    return load_registry(DEFAULT_REGISTRY)


def _current_by_entity(registry):
    return {record["entity"]: record for record in current_records(registry)}


def test_current_registry_is_lightweight_and_views_are_readable():
    registry = _registry()
    validate_registry(registry, repo_root=PROJECT_ROOT)

    status = render_status_markdown(registry, repo_root=PROJECT_ROOT)
    assert "| Item | Status | Conclusion | Evidence | Next | Updated |" in status
    assert "## Evidence registry" not in status
    assert "主方法实验日志" in status
    assert "依赖：" in status

    readme = render_readme_block(registry)
    assert "### Next steps" in readme
    next_section = readme.split("### Next steps", 1)[1].split("\n\n", 1)[0]
    assert 1 <= len([line for line in next_section.splitlines() if line.startswith("- ")]) <= 3

    snapshot = current_snapshot(registry)
    assert snapshot["status_axes"] == ["execution", "scientific_verdict"]
    assert {source["id"] for source in snapshot["evidence"]} == {
        evidence_id
        for record in snapshot["records"]
        for evidence_id in record["evidence_ids"]
    }


def test_execution_and_scientific_verdict_remain_independent():
    current = _current_by_entity(_registry())

    assert current["main.r1p"]["execution"] == "completed"
    assert current["main.r1p"]["scientific_verdict"] == "rejected"

    assert current["main.d1b"]["execution"] == "failed"
    assert current["main.d1b"]["scientific_verdict"] == "inconclusive"

    assert current["main.future_vq"]["execution"] == "blocked"
    assert current["main.future_vq"]["scientific_verdict"] == "unreviewed"

    assert current["comparison.campaign"]["execution"] == "completed"
    assert current["comparison.campaign"]["scientific_verdict"] == "mixed"
    assert current["comparison.efrm"]["execution"] == "completed"
    assert current["comparison.efrm"]["scientific_verdict"] == "mixed"


def test_comparison_method_totals_match_the_campaign_aggregate():
    current = _current_by_entity(_registry())
    method_entities = (
        "comparison.biot",
        "comparison.cbramod",
        "comparison.reve",
        "comparison.efrm",
        "comparison.normwear",
        "comparison.brainfusion",
    )
    campaign = current["comparison.campaign"]

    method_job_total = sum(
        current[entity]["progress"]["total"] for entity in method_entities
    )
    assert method_job_total == campaign["progress"]["total"] == 540
    for outcome, expected in campaign["outcome_counts"].items():
        assert (
            sum(
                current[entity].get("outcome_counts", {}).get(outcome, 0)
                for entity in method_entities
            )
            == expected
        )


def test_registry_does_not_scan_authorization_names_recursively():
    registry = copy.deepcopy(_registry())
    campaign = next(
        item for item in registry["records"] if item["entity"] == "comparison.campaign"
    )
    campaign["outcome_counts"]["authorization_status"] = 1

    validate_registry(registry, repo_root=PROJECT_ROOT)


def test_registry_rejects_scientific_verdict_before_execution():
    registry = copy.deepcopy(_registry())
    record = next(item for item in registry["records"] if item["entity"] == "atlas.statistical")
    record["scientific_verdict"] = "qualified"

    with pytest.raises(ProjectStateError, match="planned work cannot"):
        validate_registry(registry, repo_root=PROJECT_ROOT)


def test_registry_rejects_mixed_verdict_without_mixed_outcomes():
    registry = copy.deepcopy(_registry())
    campaign = next(
        item for item in registry["records"] if item["entity"] == "comparison.campaign"
    )
    campaign["outcome_counts"] = {"table_ready_with_note": 42}

    with pytest.raises(ProjectStateError, match="two non-zero outcome counts"):
        validate_registry(registry, repo_root=PROJECT_ROOT)


def test_registry_accepts_a_date_only_snapshot_timestamp():
    registry = copy.deepcopy(_registry())
    registry["updated_at"] = "2026-08-16"

    validate_registry(registry, repo_root=PROJECT_ROOT)


def test_evidence_drift_is_only_an_audit_failure():
    registry = copy.deepcopy(_registry())
    registry["evidence"][0]["sha256"] = "0" * 64

    validate_registry(registry, repo_root=PROJECT_ROOT)
    with pytest.raises(ProjectStateError, match="evidence drift"):
        validate_registry(registry, repo_root=PROJECT_ROOT, audit=True)


def test_current_record_can_be_updated_in_place_without_supersedes():
    registry = copy.deepcopy(_registry())
    record = next(item for item in registry["records"] if item["entity"] == "atlas.statistical")
    record["summary"] = "状态摘要可在当前记录中直接更新。"
    record.pop("supersedes")
    record.pop("depends_on")

    validate_registry(registry, repo_root=PROJECT_ROOT)


def test_effective_snapshot_timestamp_follows_a_current_record_update():
    registry = copy.deepcopy(_registry())
    record = next(item for item in registry["records"] if item["entity"] == "atlas.statistical")
    record["updated_at"] = "2026-09-01"

    validate_registry(registry, repo_root=PROJECT_ROOT)
    assert current_snapshot(registry)["updated_at"] == "2026-09-01"
    assert "updated_at=2026-09-01" in render_agent_summary(registry)
    assert "_Registry snapshot: `2026-09-01`" in render_status_markdown(
        registry, repo_root=PROJECT_ROOT
    )


def test_superseding_record_replaces_exactly_one_current_state():
    registry = copy.deepcopy(_registry())
    previous = next(
        item for item in registry["records"] if item["entity"] == "atlas.statistical"
    )
    replacement = copy.deepcopy(previous)
    replacement.update(
        {
            "state_id": "atlas.statistical@2026-08-17.reviewed",
            "supersedes": [previous["state_id"]],
            "updated_at": "2026-08-17T09:00:00+08:00",
        }
    )
    registry["records"].append(replacement)
    registry["updated_at"] = replacement["updated_at"]

    validate_registry(registry, repo_root=PROJECT_ROOT)
    current = _current_by_entity(registry)
    assert current["atlas.statistical"]["state_id"] == replacement["state_id"]


def test_optional_supersedes_cannot_replace_a_newer_snapshot_with_an_older_one():
    registry = copy.deepcopy(_registry())
    previous = next(
        item for item in registry["records"] if item["entity"] == "atlas.statistical"
    )
    replacement = copy.deepcopy(previous)
    replacement.update(
        {
            "state_id": "atlas.statistical@2020-01-01.backfill",
            "supersedes": [previous["state_id"]],
            "updated_at": "2020-01-01",
        }
    )
    registry["records"].append(replacement)

    with pytest.raises(ProjectStateError, match="newer than superseded"):
        validate_registry(registry, repo_root=PROJECT_ROOT)


def test_registry_rejects_supersedes_cycle_and_missing_current_state():
    registry = copy.deepcopy(_registry())
    first = next(item for item in registry["records"] if item["entity"] == "atlas.full")
    second = copy.deepcopy(first)
    second.update(
        {
            "state_id": "atlas.full@2026-08-17.revision",
            "supersedes": [first["state_id"]],
            "updated_at": "2026-08-17T09:00:00+08:00",
        }
    )
    first["supersedes"] = [second["state_id"]]
    registry["records"].append(second)
    registry["updated_at"] = second["updated_at"]

    with pytest.raises(ProjectStateError, match="newer than superseded|cycle|current state"):
        validate_registry(registry, repo_root=PROJECT_ROOT)


def test_agent_summary_has_only_the_two_status_axes():
    summary = render_agent_summary(_registry())

    assert "status_axes=execution,scientific_verdict" in summary
    assert "main.r1p" in summary
    assert "authorization" not in summary
    assert "authorized" not in summary
