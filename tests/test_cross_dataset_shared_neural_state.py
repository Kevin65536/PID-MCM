from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from experiments.evaluate_cross_dataset_shared_neural_state import (
    Segment,
    _eeg_features,
    evaluate_fold_lag,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    config = yaml.safe_load(
        (REPO_ROOT / "experiments/configs/physiology_semantic_tokenizer/cross_dataset_shared_neural_state.yaml").read_text()
    )
    config["uncertainty"]["segment_bootstrap_iterations"] = 10
    config["uncertainty"]["alignment_null_iterations"] = 10
    return config


def test_log_bandpower_preserves_small_unit_variation() -> None:
    time = np.arange(400, dtype=np.float64) / 200.0
    signal = (1e-6 * np.sin(2.0 * np.pi * 10.0 * time))[:, None]
    features = _eeg_features(signal, 200.0, {"alpha": [8.0, 13.0]})
    assert features.shape == (2, 1)
    assert np.isfinite(features).all()
    assert float(np.abs(features).max()) > 1.0


def test_lagged_shared_state_recovers_known_cross_modal_innovation() -> None:
    rng = np.random.default_rng(7)
    segments = []
    for subject in ("train", "validation"):
        for segment_index in range(24):
            shared = rng.normal(size=(30, 3))
            eeg = np.concatenate(
                (shared + 0.2 * rng.normal(size=shared.shape), rng.normal(size=(30, 3))), axis=1
            )
            fnirs = np.zeros((30, 6), dtype=np.float64)
            fnirs[5:, :3] = shared[:-5] + 0.2 * rng.normal(size=(25, 3))
            fnirs[:5, :3] = rng.normal(size=(5, 3))
            fnirs[:, 3:] = rng.normal(size=(30, 3))
            segments.append(
                Segment("synthetic", subject, f"segment_{segment_index}", "condition", eeg, fnirs)
            )

    result, _ = evaluate_fold_lag(
        "synthetic",
        "train",
        "validation",
        [segment for segment in segments if segment.subject == "train"],
        [segment for segment in segments if segment.subject == "validation"],
        5,
        _config(),
        np.random.default_rng(11),
    )
    metrics = {row["direction"]: row for row in result["rows"]}
    assert metrics["eeg_to_fnirs"]["shared_innovation_fraction"] > 0.3
    assert metrics["fnirs_to_eeg"]["shared_innovation_fraction"] > 0.3
    assert metrics["eeg_to_fnirs"]["mean_validation_canonical_correlation"] > 0.8
