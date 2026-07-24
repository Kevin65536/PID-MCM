import json

import numpy as np
import pytest
import yaml

from experiments.analyze_e2_semantic_weight_calibration import summarize_run
from experiments.analyze_e2_semantic_objective_results import (
    _bootstrap_fixed_seed_mean,
)
from experiments.build_adaptive_teacher_sidecar import patch_targets
from experiments.launch_e2_semantic_objective_suite import resolve_row_config
from src.analysis.semantic_token_evaluation import (
    fit_grouped_ridge_probe,
    match_prototype_signatures,
    prototype_signatures,
    r2_per_coordinate,
)


def test_adaptive_sidecar_patch_targets_use_registered_coordinate_order():
    time = np.linspace(0.0, 1.0, 200)
    states = np.column_stack((2 * time, 3 * time, 4 * time, 5 * time, 6 * time))
    std = np.full_like(states, 0.1)

    eeg, eeg_uncertainty, fnirs, fnirs_uncertainty = patch_targets(states, std)

    assert eeg.shape == (10, 6)
    assert fnirs.shape == (10, 9)
    assert np.allclose(eeg[:, 0], states[:, 4].reshape(10, 20).mean(axis=1))
    assert np.allclose(eeg[:, 3], states[:, 0].reshape(10, 20).mean(axis=1))
    assert np.allclose(fnirs[:, 0], states[:, 1].reshape(10, 20).mean(axis=1))
    assert np.all(eeg_uncertainty > 0)
    assert np.all(fnirs_uncertainty > 0)


def test_grouped_frozen_probe_recovers_heldout_linear_target():
    rng = np.random.default_rng(5)
    features = rng.normal(size=(120, 4))
    target = np.column_stack((features[:, 0] - features[:, 1], 2 * features[:, 2]))
    groups = np.repeat([f"s{index}" for index in range(6)], 20)

    probe, selection = fit_grouped_ridge_probe(
        features, target, groups, alphas=(0.0, 0.1, 1.0)
    )
    score = r2_per_coordinate(target, probe.predict(features))

    assert np.all(score > 0.99)
    assert selection["group_count"] == 6


def test_prototype_signature_matching_is_invariant_to_code_permutation():
    hard = np.asarray([[0, 0, 1, 1]])
    target = np.asarray([[[1.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 2.0]]])
    valid = np.ones_like(hard, dtype=bool)
    left, counts = prototype_signatures(hard, target, valid, codebook_size=2)
    right = left[[1, 0]]
    match = match_prototype_signatures(left, right, counts, counts[[1, 0]])

    assert match["matched_count"] == 2
    assert match["mean_cosine"] > 0.99


def test_e2_training_gradient_calibration_overrides_both_semantic_entries():
    with open(
        "experiments/configs/physiology_semantic_tokenizer/"
        "e2_semantic_objective_suite.yaml",
        encoding="utf-8",
    ) as handle:
        base = yaml.safe_load(handle)

    resolved = resolve_row_config(
        base,
        "T1",
        20260719,
        semantic_weight=0.1,
    )

    assert resolved["loss"]["state"]["weight"] == 0.1
    assert resolved["loss"]["prototype"]["weight"] == 0.1
    assert resolved["validation"]["semantic_weight_calibration"] == {
        "scope": "training_gradient_scale_only",
        "validation_target_decoding_used_for_selection": False,
        "weight": 0.1,
    }
    with pytest.raises(ValueError, match="semantic E2 row"):
        resolve_row_config(base, "T0", 20260719, semantic_weight=0.1)


def test_e2_weight_calibration_uses_shared_training_gradients(tmp_path):
    run_dir = tmp_path / "run"
    diagnostics = run_dir / "diagnostics"
    diagnostics.mkdir(parents=True)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump({
            "loss": {
                "state": {"weight": 0.005},
                "prototype": {"weight": 0.005},
            }
        }),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"seed": 20260719}),
        encoding="utf-8",
    )

    def objective(modality, norm):
        return {
            "status": "audited",
            "parameter_gradient_norms": {
                f"{modality}_branch.local_encoder.0.weight": norm,
                f"{modality}_branch.state_head.weight": 1000.0,
            },
        }

    objectives = {}
    for modality in ("eeg", "fnirs"):
        objectives[f"{modality}_reconstruction"] = objective(modality, 1.0)
        objectives[f"{modality}_state"] = objective(modality, 5.0)
        objectives[f"{modality}_prototype"] = objective(modality, 0.5)
    (diagnostics / "gradient_entry_audit.json").write_text(
        json.dumps({
            "all_contracts_passed": True,
            "records": [{"objectives": objectives}],
        }),
        encoding="utf-8",
    )

    summary = summarize_run(run_dir, minimum_ratio=0.1, maximum_ratio=10.0)

    assert summary["candidate_passed"] is True
    assert {
        row["objective"]: row["median_ratio"] for row in summary["objectives"]
    } == {
        "eeg_state": 5.0,
        "eeg_prototype": 0.5,
        "fnirs_state": 5.0,
        "fnirs_prototype": 0.5,
    }


def test_e2_paired_subject_bootstrap_preserves_seed_pairing():
    summary = _bootstrap_fixed_seed_mean(
        {
            1: {"s1": 1.0, "s2": 3.0},
            2: {"s1": -2.0, "s2": 0.0},
        },
        iterations=128,
        rng=np.random.default_rng(7),
    )

    assert summary["subject_mean_delta_by_seed"] == {"1": 2.0, "2": -1.0}
    assert summary["fixed_seed_subject_mean_delta"] == 0.5
    assert summary["positive_seed_count"] == 1
    assert summary["seed_count"] == 2
