#!/usr/bin/env python
"""Materialize a T5 config only when frozen Phase B promotes F6."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from experiments.scripts.launch_coupling_identifiability_suite import intervention_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    suite_dir = Path(args.suite_dir).resolve()
    decision_path = suite_dir / "frozen_calibration/neutral/decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    output = Path(args.output).resolve()
    if not decision.get("f6_promoted", False):
        output.with_suffix(".skipped.json").write_text(
            json.dumps({"reason": "F6 did not pass Phase B promotion criteria"}, indent=2) + "\n",
            encoding="utf-8",
        )
        return

    parameters = np.load(
        suite_dir / "frozen_calibration/neutral/global_F6_parameters.npz",
        allow_pickle=False,
    )
    results = json.loads(
        (suite_dir / "frozen_calibration/neutral/frozen_model_results.json").read_text(encoding="utf-8")
    )
    regularization_weight = float(
        results["global"]["F6"]["hyperparameters"]["regularization_weight"]
    )
    config = intervention_config(
        suite_dir.name, "T2", args.seed, args.device, smoke=False,
    )
    config["experiment"]["name"] = f"coupling_identifiability_t5_seed{args.seed}"
    config["experiment"]["description"] = (
        "Promoted F6: fixed train marginal, occupancy gauge, effective-q smoothness, "
        "and interaction-energy lag sparsity."
    )
    coupling = config["loss"]["coupling"]
    coupling["fixed_eeg_marginal"] = parameters["eeg_occupancy"].tolist()
    coupling["fixed_fnirs_marginal"] = parameters["fnirs_prior"].tolist()
    coupling["effective_smoothness_weight"] = regularization_weight
    coupling["interaction_lag_sparsity_weight"] = regularization_weight
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
