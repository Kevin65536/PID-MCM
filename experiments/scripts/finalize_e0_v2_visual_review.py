#!/usr/bin/env python3
"""Validate and register a human visual review for an E0-v2 archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


def _read(path: Path):
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--review", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    review_path = Path(args.review).resolve()
    manifest_path = run_dir / "visual_audit_manifest.json"
    summary_path = run_dir / "summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    review = _read(review_path)
    decisions = review.get("figures", {})
    allowed = {"pass", "fail"}

    for figure in manifest["figures"]:
        name = figure["name"]
        if name not in decisions or decisions[name].get("status") not in allowed:
            raise ValueError(f"review must assign pass/fail to {name}")
        for artifact, expected in figure["sha256"].items():
            observed = hashlib.sha256((run_dir / artifact).read_bytes()).hexdigest()
            if observed != expected:
                raise ValueError(f"artifact hash changed after rendering: {artifact}")
        figure["review_status"] = decisions[name]["status"]
        figure["review_note"] = str(decisions[name].get("note", ""))

    overall_pass = all(figure["review_status"] == "pass" for figure in manifest["figures"])
    completed = {
        "reviewer": str(review.get("reviewer", "unspecified")),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "overall_pass": overall_pass,
        "checklist_notes": list(review.get("checklist_notes", [])),
        "review_file": str(review_path),
    }
    manifest["review"] = completed
    machine_pass = bool(summary["validation"]["machine_validation_pass"])
    manifest["protected_test_may_open"] = bool(machine_pass and overall_pass)
    summary["validation"]["visual_review"] = "pass" if overall_pass else "fail"
    summary["validation"]["visual_review_completed_at"] = completed["completed_at"]
    summary["protected_test"] = {
        "opened": False,
        "eligible_to_open": manifest["protected_test_may_open"],
        "reason": (
            "machine validation and visual review passed; a separate protected-test command is required"
            if manifest["protected_test_may_open"]
            else "machine validation or visual review failed"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "visual_review.json").write_text(json.dumps(completed, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"visual_review_pass": overall_pass, "protected_test_may_open": manifest["protected_test_may_open"]}))


if __name__ == "__main__":
    main()
