#!/usr/bin/env python
"""Generate visual diagnostics for exported source/observation token sequences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.visualization import analyze_source_observation_token_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Croce source/observation token sequence exports")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Token export run directory containing manifest.json and tokens/*_tokens.npz",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Default: <run-dir>/analysis/token_sequence",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Optional split list. Default: infer from tokens/*_tokens.npz",
    )
    parser.add_argument(
        "--max-heatmap-samples",
        type=int,
        default=160,
        help="Maximum samples per split shown in heatmap panels",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Figure DPI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = analyze_source_observation_token_sequences(
        run_dir=Path(args.run_dir),
        output_dir=None if args.output_dir is None else Path(args.output_dir),
        splits=args.splits,
        max_heatmap_samples=args.max_heatmap_samples,
        dpi=args.dpi,
    )
    print(json.dumps({
        "analysis_dir": manifest["analysis_dir"],
        "figures": manifest["figures"],
        "tables": manifest["tables"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
