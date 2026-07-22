#!/usr/bin/env python3
"""Render EFRM CLIP pair evidence without opening protected data implicitly."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from efrm_pytorch.visualization import render_alignment_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--physiology-coupling-evidence")
    args = parser.parse_args()
    render_alignment_report(
        args.evidence,
        args.output_dir,
        physiology_coupling_evidence=args.physiology_coupling_evidence,
    )


if __name__ == "__main__":
    main()
