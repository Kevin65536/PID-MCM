"""Smoke tests for the synthetic-only T3a P0 figure renderer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "experiments/scripts/render_t3a_balloon_robust_p0.py"


def test_t3a_p0_renderer_self_check() -> None:
    result = subprocess.run(
        [sys.executable, str(RENDERER), "--self-check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "self-check passed" in result.stdout
