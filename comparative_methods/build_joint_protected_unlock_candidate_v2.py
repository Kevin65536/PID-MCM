#!/usr/bin/env python3
"""Build the non-authorizing adapter-eligibility input for joint unlock review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comparative_methods.audit_adapter_alignment import audit  # noqa: E402


CONTRACT = REPO_ROOT / "comparative_methods/adapter_alignment_gate_contract_v2.yaml"
METHODS = {
    "biot": "comparative_methods/BIOT/evidence/alignment_v2",
    "cbramod": "comparative_methods/CBraMod/evidence/alignment_v2",
    "reve": "comparative_methods/REVE/evidence/alignment_v2",
    "brainfusion_nvc_csp_stacking_reimplementation": (
        "comparative_methods/BrainFusion-NVC-CSP-Stacking/evidence/alignment_v2"
    ),
    "normwear_eeg_fnirs_adapted": "comparative_methods/NormWear/evidence/alignment_v2",
    "efrm_sync_200_10_variable_channel_v1": (
        "comparative_methods/EFRM-PyTorch/evidence/alignment_v2"
    ),
}
TASKS = (
    "motor_imagery",
    "mental_arithmetic",
    "wg",
    "nback",
    "dsr",
    "visual",
    "refed_regression",
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "comparative_methods/evidence/joint_protected_unlock_candidate_v2.json"
)


class CandidateError(RuntimeError):
    """Raised when retained public evidence is not safe to submit for review."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CandidateError(f"expected JSON object: {path}")
    return value


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def build_candidate() -> dict[str, Any]:
    cell_paths = [
        REPO_ROOT / directory / f"{task}.json"
        for directory in METHODS.values()
        for task in TASKS
    ]
    report = audit(CONTRACT, cell_paths)
    if report.get("status") != "pass" or report.get("protected_test_opened") is not False:
        raise CandidateError("global adapter audit is not a protected-closed pass")

    method_summaries: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for expected_method, directory in METHODS.items():
        evidence_dir = REPO_ROOT / directory
        summary_path = evidence_dir / "summary_final.json"
        summary = _read_json(summary_path)
        if summary.get("protected_test_opened") is not False:
            raise CandidateError(f"method summary reports protected access: {summary_path}")
        if summary.get("protected_evaluation_authorized") is not False:
            raise CandidateError(f"method summary is already authorizing: {summary_path}")
        method_summaries.append(
            {
                "method_id": expected_method,
                "evidence_path": _relative(summary_path),
                "evidence_sha256": _sha256(summary_path),
                "retained_status": summary.get("status"),
            }
        )

        for task in TASKS:
            cell_path = evidence_dir / f"{task}.json"
            cell = _read_json(cell_path)
            if cell.get("method_id") != expected_method:
                raise CandidateError(
                    f"method identity mismatch in {cell_path}: {cell.get('method_id')!r}"
                )
            if cell.get("task_id") != task:
                raise CandidateError(f"task identity mismatch in {cell_path}")
            if cell.get("protected_test_opened") is True:
                raise CandidateError(f"cell reports protected access: {cell_path}")
            if cell.get("table_admissible") is True:
                raise CandidateError(f"public cell claims table admission: {cell_path}")
            status = cell.get("cell_status")
            if status not in {"pass", "unsupported"}:
                raise CandidateError(f"non-terminal cell status in {cell_path}: {status!r}")
            cells.append(
                {
                    "cell_id": cell["cell_id"],
                    "method_id": cell["method_id"],
                    "task_id": task,
                    "track": cell["track"],
                    "alignment_profile": cell["alignment_profile"],
                    "comparison_group_id": cell["comparison_group_id"],
                    "cell_status": status,
                    "adapter_eligible_for_unlock_review": status == "pass",
                    "disposition": (
                        "candidate_for_one_time_protected_evaluation"
                        if status == "pass"
                        else "preregistered_unsupported_excluded_from_evaluation"
                    ),
                    "evidence_path": _relative(cell_path),
                    "evidence_sha256": _sha256(cell_path),
                }
            )

    pass_count = sum(cell["cell_status"] == "pass" for cell in cells)
    unsupported_count = sum(cell["cell_status"] == "unsupported" for cell in cells)
    if (len(cells), pass_count, unsupported_count) != (42, 36, 6):
        raise CandidateError(
            "unexpected cell disposition counts: "
            f"total={len(cells)}, pass={pass_count}, unsupported={unsupported_count}"
        )

    group_reports = report["direct_group_reports"]
    return {
        "schema": "joint_protected_unlock_candidate_v2",
        "candidate_version": 2,
        "evidence_snapshot_date": "2026-08-11",
        "status": "ready_for_human_unlock_review",
        "claim_boundary": "adapter_eligibility_only_no_protected_execution_authorized",
        "authorization": {
            "human_review_status": "pending",
            "protected_evaluation_authorized": False,
            "protected_test_opened": False,
            "target_dataset_exposure": False,
            "table_admissible": False,
        },
        "contract": {
            "path": _relative(CONTRACT),
            "sha256": _sha256(CONTRACT),
        },
        "global_alignment_audit": {
            "status": "pass",
            "cell_count": len(cells),
            "pass_cell_count": pass_count,
            "unsupported_cell_count": unsupported_count,
            "direct_group_count": len(group_reports),
            "protected_test_opened": False,
        },
        "reporting_separation": {
            "reve_motor_imagery_and_mental_arithmetic": (
                "target_corpus_overlap_track_not_clean_target_excluded"
            ),
            "sta_net": "method_native_context_reference_not_in_this_candidate",
            "classification_endpoint": "macro_f1",
            "refed_endpoint": "native_coordinate_masked_ccc",
        },
        "required_before_execution": [
            "human_review_and_separately_hashed_authorization_manifest",
            "per_method_runner_verifies_exact_cell_and_public_checkpoint_identities",
            "protected_outer_test_identity_is_opened_exactly_once_per_eligible_cell",
            "no_selection_retraining_or_retry_on_protected_performance",
            "frozen_fold_then_seed_aggregation_and_metric_acceptance_audit",
        ],
        "method_summaries": method_summaries,
        "cells": cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that the retained output exactly matches current public evidence",
    )
    args = parser.parse_args()
    candidate = build_candidate()
    payload = json.dumps(candidate, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output = args.output.resolve()
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != payload:
            raise CandidateError(f"stale or missing candidate: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "pass",
                "output": _relative(output),
                "cell_count": len(candidate["cells"]),
                "pass_cell_count": candidate["global_alignment_audit"]["pass_cell_count"],
                "unsupported_cell_count": candidate["global_alignment_audit"][
                    "unsupported_cell_count"
                ],
                "protected_evaluation_authorized": False,
                "protected_test_opened": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
