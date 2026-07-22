#!/usr/bin/env python3
"""Measure dead-code recovery gradients under registered balance temperatures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.losses.physiology_semantic import straight_through_codebook_balance_loss


def measure(temperature: float) -> dict[str, float]:
    logits = torch.full((32, 10, 128), -8.0, requires_grad=True)
    with torch.no_grad():
        logits[..., 0] = 8.0
    loss = straight_through_codebook_balance_loss(
        logits, temperature=temperature
    )
    loss.backward()
    gradient = logits.grad.detach()
    return {
        "temperature": float(temperature),
        "forward_loss": float(loss.detach()),
        "dead_max_abs_gradient": float(gradient[..., 1:].abs().max()),
        "dead_l2_gradient": float(gradient[..., 1:].norm()),
        "winner_l2_gradient": float(gradient[..., 0].norm()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [measure(temperature) for temperature in (1.0, 2.0, 4.0)]
    payload = {
        "schema": "e1_balance_temperature_gradient_probe_v1",
        "input": {
            "shape": [32, 10, 128],
            "winner_logit": 8.0,
            "dead_logit": -8.0,
            "hard_active_codes": 1,
        },
        "rows": rows,
        "temperature_2_over_1_dead_l2_ratio": (
            rows[1]["dead_l2_gradient"] / rows[0]["dead_l2_gradient"]
        ),
        "claim_boundary": (
            "Deterministic implementation diagnostic only; this measures gradient "
            "reachability, not training stability or semantic quality."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
