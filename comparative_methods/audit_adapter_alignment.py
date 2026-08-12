#!/usr/bin/env python3
"""Validate adapter-alignment v2 contracts and per-cell evidence artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = Path(__file__).with_name("adapter_alignment_gate_contract_v2.yaml")


class AlignmentAuditError(ValueError):
    """Raised when a contract or evidence artifact violates the v2 schema."""


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AlignmentAuditError(f"missing file: {path}")
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AlignmentAuditError(f"expected a mapping in {path}")
    return value


def _require_fields(value: Mapping[str, Any], fields: Sequence[str], *, context: str) -> None:
    missing = sorted(set(fields) - set(value))
    if missing:
        raise AlignmentAuditError(f"{context} is missing fields: {missing}")


def validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if contract.get("schema") != "adapter_alignment_gate_contract_v2":
        raise AlignmentAuditError("unexpected alignment contract schema")
    if contract.get("contract_version") != 2:
        raise AlignmentAuditError("alignment contract version must be 2")
    gate_ids = [str(gate.get("id")) for gate in contract.get("gates", [])]
    if gate_ids != [f"A{index}" for index in range(9)]:
        raise AlignmentAuditError(f"expected ordered A0-A8 gates, got {gate_ids}")
    if contract.get("authority", {}).get("protected_test_default") != "locked":
        raise AlignmentAuditError("protected_test_default must be locked")
    profiles = contract.get("alignment_profiles")
    tasks = contract.get("task_contracts")
    scopes = contract.get("evidence_scopes")
    if not all(isinstance(value, dict) and value for value in (profiles, tasks, scopes)):
        raise AlignmentAuditError("profiles, tasks, and evidence scopes must be non-empty mappings")
    for profile_name, profile in profiles.items():
        if "direct_ranking_allowed" not in profile or "exact_equal_fields" not in profile:
            raise AlignmentAuditError(f"alignment profile {profile_name!r} is incomplete")
    schema = contract.get("cell_evidence_schema", {})
    if schema.get("schema") != "adapter_alignment_cell_evidence_v2":
        raise AlignmentAuditError("cell evidence schema declaration is missing")
    return {
        "schema": contract["schema"],
        "contract_version": contract["contract_version"],
        "gate_ids": gate_ids,
        "task_count": len(tasks),
        "profile_count": len(profiles),
    }


def _scope_allows_gate(
    contract: Mapping[str, Any], evidence_scope: str, minimum_scope: str
) -> bool:
    scopes = contract["evidence_scopes"]
    if evidence_scope not in scopes:
        raise AlignmentAuditError(f"unknown evidence_scope {evidence_scope!r}")
    if minimum_scope == "public_complete":
        return bool(scopes[evidence_scope].get("permits_full_coverage_claim", False))
    return int(scopes[evidence_scope]["rank"]) >= int(scopes[minimum_scope]["rank"])


def validate_cell(
    cell: Mapping[str, Any], contract: Mapping[str, Any], *, source: str
) -> dict[str, Any]:
    cell_schema = contract["cell_evidence_schema"]
    _require_fields(cell, cell_schema["required_top_level_fields"], context=source)
    if cell["schema"] != cell_schema["schema"]:
        raise AlignmentAuditError(f"{source} has unexpected cell schema {cell['schema']!r}")
    task_id = str(cell["task_id"])
    method_id = str(cell["method_id"])
    profile_name = str(cell["alignment_profile"])
    evidence_scope = str(cell["evidence_scope"])
    cell_status = str(cell["cell_status"])
    if task_id not in contract["task_contracts"]:
        raise AlignmentAuditError(f"{source} has unknown task_id {task_id!r}")
    known_methods = set(contract["current_repository_assessment"]["method_readiness"])
    if method_id not in known_methods:
        raise AlignmentAuditError(f"{source} has unknown method_id {method_id!r}")
    if profile_name not in contract["alignment_profiles"]:
        raise AlignmentAuditError(f"{source} has unknown alignment_profile {profile_name!r}")
    if cell_status not in contract["status_values"]:
        raise AlignmentAuditError(f"{source} has unknown cell_status {cell_status!r}")
    if evidence_scope not in contract["evidence_scopes"]:
        raise AlignmentAuditError(f"{source} has unknown evidence_scope {evidence_scope!r}")

    comparison_fields = cell["comparison_fields"]
    if not isinstance(comparison_fields, dict):
        raise AlignmentAuditError(f"{source}.comparison_fields must be a mapping")
    if not isinstance(cell["adapter_identity"], dict) or not cell["adapter_identity"]:
        raise AlignmentAuditError(f"{source}.adapter_identity must be a non-empty mapping")
    profile = contract["alignment_profiles"][profile_name]
    _require_fields(
        comparison_fields,
        profile["exact_equal_fields"],
        context=f"{source}.comparison_fields",
    )
    expected_dataset = contract["task_contracts"][task_id]["dataset_id"]
    if comparison_fields["dataset_id"] != expected_dataset:
        raise AlignmentAuditError(
            f"{source} dataset_id {comparison_fields['dataset_id']!r} does not match "
            f"task contract {expected_dataset!r}"
        )
    if comparison_fields["task_id"] != task_id:
        raise AlignmentAuditError(f"{source} comparison_fields.task_id does not match task_id")

    gate_status = cell["gate_status"]
    if not isinstance(gate_status, dict):
        raise AlignmentAuditError(f"{source}.gate_status must be a mapping")
    gate_ids = [str(gate["id"]) for gate in contract["gates"]]
    if set(gate_status) != set(gate_ids):
        raise AlignmentAuditError(f"{source}.gate_status must contain exactly {gate_ids}")
    for gate in contract["gates"]:
        gate_id = str(gate["id"])
        status = str(gate_status[gate_id])
        if status not in contract["status_values"]:
            raise AlignmentAuditError(f"{source}.{gate_id} has unknown status {status!r}")
        if status == "pass" and not _scope_allows_gate(
            contract, evidence_scope, str(gate["minimum_evidence_scope"])
        ):
            raise AlignmentAuditError(
                f"{source}.{gate_id}=pass exceeds evidence scope {evidence_scope!r}"
            )

    if cell_status == "pass" and any(gate_status[gate_id] != "pass" for gate_id in gate_ids):
        raise AlignmentAuditError(f"{source} cell_status=pass requires every A0-A8 gate to pass")
    if cell_status == "unsupported":
        _require_fields(cell, cell_schema["unsupported_requires"], context=source)

    return {
        "source": source,
        "cell_id": str(cell["cell_id"]),
        "comparison_group_id": str(cell["comparison_group_id"]),
        "method_id": method_id,
        "task_id": task_id,
        "alignment_profile": profile_name,
        "direct_ranking_allowed": bool(profile["direct_ranking_allowed"]),
        "evidence_scope": evidence_scope,
        "cell_status": cell_status,
    }


def validate_direct_groups(
    cells: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for cell in cells:
        profile_name = str(cell["alignment_profile"])
        profile = contract["alignment_profiles"][profile_name]
        # ``unsupported`` is a terminal, preregistered disposition rather than
        # a delivered comparison surface. Such cells deliberately may retain
        # non-dereferenced sentinels for inventory/support fields, so comparing
        # them with a peer's fully materialized public cell would manufacture
        # an alignment failure. Exact equality is meaningful only among cells
        # that actually passed the direct-profile gates and can be ranked.
        if not profile["direct_ranking_allowed"] or cell["cell_status"] != "pass":
            continue
        key = (str(cell["comparison_group_id"]), profile_name)
        groups.setdefault(key, []).append(cell)

    reports: list[dict[str, Any]] = []
    for (group_id, profile_name), members in sorted(groups.items()):
        exact_fields = contract["alignment_profiles"][profile_name]["exact_equal_fields"]
        reference = members[0]["comparison_fields"]
        for field in exact_fields:
            reference_value = json.dumps(reference[field], sort_keys=True, separators=(",", ":"))
            for member in members[1:]:
                value = json.dumps(
                    member["comparison_fields"][field], sort_keys=True, separators=(",", ":")
                )
                if value != reference_value:
                    raise AlignmentAuditError(
                        f"direct group {group_id!r} differs on exact field {field!r}: "
                        f"{members[0]['cell_id']!r} != {member['cell_id']!r}"
                    )
        reports.append(
            {
                "comparison_group_id": group_id,
                "alignment_profile": profile_name,
                "cell_count": len(members),
                "exact_fields_checked": list(exact_fields),
            }
        )
    return reports


def audit(contract_path: Path, cell_paths: Sequence[Path]) -> dict[str, Any]:
    contract = _load_mapping(contract_path)
    contract_report = validate_contract(contract)
    cells = [_load_mapping(path) for path in cell_paths]
    cell_reports = [
        validate_cell(cell, contract, source=str(path))
        for path, cell in zip(cell_paths, cells, strict=True)
    ]
    group_reports = validate_direct_groups(cells, contract)
    return {
        "schema": "adapter_alignment_audit_report_v2",
        "status": "pass",
        "contract": contract_report,
        "cell_reports": cell_reports,
        "direct_group_reports": group_reports,
        "protected_test_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cell", nargs="*", type=Path, help="cell evidence YAML/JSON artifacts")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    try:
        report = audit(args.contract, args.cell)
    except (AlignmentAuditError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
