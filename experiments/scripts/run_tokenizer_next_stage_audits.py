#!/usr/bin/env python
"""Run exports, audits, and downstream probes for the next-stage tokenizer suite."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREVIOUS_CAPACITY = PROJECT_ROOT / "experiments/runs/tokenizer_coupling_capacity/20260617_185619_codebook_mission_bias_v1"
MAX_LAG_TOKENS = 5


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def run(command: str) -> None:
    print(f"+ {command}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, shell=True, check=True)


def maybe(command: str, sentinel: Path) -> None:
    if sentinel.exists():
        print(f"skip existing: {sentinel}", flush=True)
        return
    run(command)


def run_specs(suite_dir: Path) -> list[tuple[str, Path, str]]:
    specs: list[tuple[str, Path, str]] = []
    baseline_root = PREVIOUS_CAPACITY / "capacity_sweep"
    for run_dir in sorted(baseline_root.glob("s2_croce_local_highwl_v2_capacity_k128_coupling0_seed*")):
        checkpoint = run_dir / "checkpoints/best_model.pt"
        if checkpoint.exists():
            specs.append((f"baseline_dim48_{run_dir.name}", checkpoint, "baseline"))
    for leaf, source in (("vector_dim_sweep", "vector_dim"), ("cognitive_tokenizer", "cognitive")):
        for run_dir in sorted((suite_dir / leaf).glob("k128_dim*_seed*")):
            checkpoint = run_dir / "checkpoints/best_model.pt"
            if checkpoint.exists():
                specs.append((run_dir.name, checkpoint, source))
        for run_dir in sorted((suite_dir / leaf).glob("k128_dim*_cognitive_nback_wg_seed*")):
            checkpoint = run_dir / "checkpoints/best_model.pt"
            if checkpoint.exists():
                specs.append((run_dir.name, checkpoint, source))
    seen = set()
    deduped = []
    for item in specs:
        if item[0] in seen:
            continue
        seen.add(item[0])
        deduped.append(item)
    return deduped


def audit_one(suite_dir: Path, label: str, checkpoint: Path, device: str, *, downstream: bool) -> None:
    audit_export = suite_dir / "coupling_audit_exports" / label
    position_dir = suite_dir / "position_audit" / label
    geometry_dir = suite_dir / "codebook_geometry" / label
    info_dir = suite_dir / "information_drop_audit" / label
    local_dir = suite_dir / "local_coupling_audit" / label
    token_export = suite_dir / "token_exports" / label
    downstream_dir = suite_dir / "downstream_probe" / label

    maybe(
        ".venv/bin/python experiments/scripts/export_coupling_audit_data.py "
        f"--checkpoint {q(checkpoint)} --output-dir {q(audit_export)} --label {q(label)} "
        f"--device {q(device)} --batch-size 512 --num-workers 8 --clear-entry-filters",
        audit_export / "manifest.json",
    )
    maybe(
        ".venv/bin/python experiments/scripts/analyze_coupling_identifiability.py "
        f"--export-dir {q(audit_export)} --output-dir {q(position_dir)} "
        f"--permutations 500 --bootstraps 1000 --max-lag-tokens {MAX_LAG_TOKENS}",
        position_dir / "summary.json",
    )
    maybe(
        ".venv/bin/python experiments/scripts/analyze_codebook_geometry.py "
        f"--checkpoint {q(checkpoint)} --output-dir {q(geometry_dir)} --run-name {q(label)}",
        geometry_dir / "summary.json",
    )
    maybe(
        ".venv/bin/python experiments/scripts/analyze_information_drop.py "
        f"--checkpoint {q(checkpoint)} --export-dir {q(audit_export)} --output-dir {q(info_dir)} "
        f"--run-name {q(label)} --splits val test --max-tokens 50000",
        info_dir / "summary.json",
    )
    maybe(
        ".venv/bin/python experiments/scripts/analyze_local_coupling.py "
        f"--export-dir {q(audit_export)} --output-dir {q(local_dir)} --run-name {q(label)} "
        f"--max-lag-tokens {MAX_LAG_TOKENS}",
        local_dir / "summary.json",
    )
    if downstream:
        maybe(
            "bash experiments/scripts/launch_training_nohup.sh --task source-observation-token-export --foreground "
            f"--tokenizer-run-dir {q(checkpoint.parent.parent)} --checkpoint {q(checkpoint)} "
            f"--output-root {q(suite_dir / 'token_exports')} --run-name {q(label)} "
            f"--splits train,val,test --batch-size 512 --num-workers 8 --clear-entry-filters",
            token_export / "manifest.json",
        )
        maybe(
            ".venv/bin/python experiments/scripts/run_source_observation_token_downstream_probe.py "
            f"--token-run-dir {q(token_export)} --output-dir {q(downstream_dir)}",
            downstream_dir / "summary.json",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--downstream-sources",
        default="baseline,vector_dim,cognitive",
        help="Comma-separated source classes to run standard token export/downstream probes for.",
    )
    args = parser.parse_args()
    suite_dir = Path(args.suite_dir).resolve()
    downstream_sources = {item.strip() for item in args.downstream_sources.split(",") if item.strip()}
    specs = run_specs(suite_dir)
    for label, checkpoint, source in specs:
        audit_one(suite_dir, label, checkpoint, args.device, downstream=source in downstream_sources)
    run(f".venv/bin/python experiments/scripts/finalize_tokenizer_next_stage_suite.py --suite-dir {q(suite_dir)}")
    print(json.dumps({"processed": len(specs), "suite_dir": str(suite_dir)}, indent=2))


if __name__ == "__main__":
    main()
