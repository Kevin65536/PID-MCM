#!/usr/bin/env python3
"""Create a non-authorizing dual-signature template for a frozen candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comparative_methods.protected_campaign_common import (  # noqa: E402
    AUTHORIZATION_SCHEMA,
    portable_path,
    sha256_file,
    verify_candidate_file,
    write_json_atomic,
)


DEFAULT_OUTPUT = (
    REPO_ROOT / "comparative_methods/evidence/protected_campaign/authorization_template_v1.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    candidate, candidate_sha256 = verify_candidate_file(
        args.candidate.resolve(), verify_artifacts=False
    )
    template = {
        "schema": AUTHORIZATION_SCHEMA,
        "campaign_id": candidate["campaign_id"],
        "candidate_path": portable_path(args.candidate.resolve()),
        "candidate_sha256": candidate_sha256,
        "scope": {"supported_cells": 36, "jobs": 540},
        "authorized_window": {"starts_at": "", "ends_at": ""},
        "technical_recovery_policy": {
            "maximum_attempts_per_job": 2,
            "one_technical_recovery_only": True,
            "performance_based_retry_forbidden": True,
            "attempt_2_device_policy": "same_frozen_gpu_uuid_only",
            "unavailable_assigned_gpu_terminal": (
                "INCOMPLETE_TECHNICAL_requires_new_candidate_and_dual_authorization"
            ),
        },
        "signatures": [
            {
                "role": "protocol_owner",
                "signer_id": "",
                "signed_at": "",
                "attestation": "PENDING",
            },
            {
                "role": "run_owner",
                "signer_id": "",
                "signed_at": "",
                "attestation": "PENDING",
            },
        ],
        "protected_evaluation_authorized": False,
        "note": "Template only. Fill the window, two distinct identities, GO attestations, and set authorization true without modifying the candidate.",
    }
    write_json_atomic(args.output.resolve(), template)
    print(
        json.dumps(
            {
                "status": "template_created_not_authorized",
                "candidate_sha256": candidate_sha256,
                "output": portable_path(args.output.resolve()),
                "authorization_sha256": sha256_file(args.output.resolve()),
                "protected_evaluation_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
