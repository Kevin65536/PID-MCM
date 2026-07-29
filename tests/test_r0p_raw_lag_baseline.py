from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "scripts" / "run_r0p_raw_lag_baseline.py"
SPEC = importlib.util.spec_from_file_location("run_r0p_raw_lag_baseline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _synthetic_cells(seed: int = 9):
    rng = np.random.default_rng(seed)
    trials = []
    for subject_index in range(6):
        eeg = rng.normal(size=(10, 10, 3))
        fnirs = rng.normal(scale=0.5, size=(10, 10, 2))
        fnirs[:, 3:, 0] = -eeg[:, :7, 1] + rng.normal(
            scale=0.08, size=(10, 7)
        )
        for trial_index in range(10):
            trials.append(
                MODULE.TrialFeatures(
                    sample_key=(
                        f"synthetic|subject_{subject_index + 1:02d}|"
                        f"session_01|event_{trial_index:02d}"
                    ),
                    subject=f"subject_{subject_index + 1:02d}",
                    session="session_01",
                    condition="MA",
                    eeg=eeg[trial_index],
                    fnirs=fnirs[trial_index],
                )
            )
    return MODULE.prepare_cells(trials)


def test_preregistered_feature_extraction_detects_bands_and_fnirs_means():
    time = np.arange(4000) / 200.0
    eeg = np.stack(
        [
            np.sin(2 * np.pi * 10.0 * time),
            np.sin(2 * np.pi * 10.0 * time),
            np.sin(2 * np.pi * 10.0 * time),
            np.sin(2 * np.pi * 10.0 * time),
            np.sin(2 * np.pi * 10.0 * time),
            np.sin(2 * np.pi * 10.0 * time),
        ]
    )
    fnirs = np.stack([np.full(200, 2.0), np.full(200, -0.5)])
    eeg_features, fnirs_features = MODULE.extract_patch_features(eeg, fnirs)
    assert eeg_features.shape == (10, 3)
    assert fnirs_features.shape == (10, 2)
    assert np.all(eeg_features[:, 1] > eeg_features[:, 0])
    assert np.all(eeg_features[:, 1] > eeg_features[:, 2])
    np.testing.assert_allclose(fnirs_features[:, 0], 2.0)
    np.testing.assert_allclose(fnirs_features[:, 1], -0.5)


def test_known_delayed_negative_alpha_hbo_exceeds_pairing_null():
    cells = _synthetic_cells()
    _, subject_values, _ = MODULE.compute_observed(cells)
    observed = np.nanmean(subject_values, axis=0)
    _, primary_null, _ = MODULE.permutation_null(
        cells, permutations=199, seed=123, batch_size=50
    )
    observed_auc = MODULE.registered_auc(observed[None])[0]
    assert observed_auc > np.percentile(primary_null, 95)


def test_within_cell_permutation_preserves_fnirs_trials_but_changes_pairing():
    cell = _synthetic_cells()[0]
    order = np.array([1, 0, 2, 3, 4, 5, 6, 7, 8, 9])
    permuted = MODULE.permute_fnirs_trials(cell, order)
    np.testing.assert_allclose(permuted.fnirs, cell.fnirs[order])
    np.testing.assert_allclose(
        np.sort(permuted.fnirs.reshape(10, -1), axis=0),
        np.sort(cell.fnirs.reshape(10, -1), axis=0),
    )
    assert not np.array_equal(permuted.fnirs, cell.fnirs)
    np.testing.assert_allclose(permuted.eeg, cell.eeg)


def test_maxstat_pvalues_dominate_unadjusted_member_pvalues():
    rng = np.random.default_rng(4)
    observed = rng.normal(size=(5, 3, 2))
    member_null = rng.normal(size=(300, 5, 3, 2))
    max_abs = np.max(np.abs(member_null), axis=(1, 2, 3))
    unadjusted, adjusted = MODULE.permutation_pvalues(
        observed, member_null, max_abs
    )
    assert np.all(adjusted >= unadjusted)
    assert adjusted.shape == (5, 3, 2)


def test_protected_subject_fails_before_loader_construction():
    with pytest.raises(PermissionError, match="before array dereference"):
        MODULE.assert_development_subjects(["subject_24"])


def test_preregistry_contract_is_machine_readable_and_frozen(tmp_path):
    source = (
        ROOT
        / "docs"
        / "physiology_semantic_tokenizer"
        / "architecture"
        / "r0p_raw_lag_baseline_preregistry.json"
    )
    registry = MODULE.load_preregistry(source)
    assert registry["null_contract"]["formal_permutations"] == 10000
    assert registry["diagnostic_family"]["members"].startswith("all 3 EEG bands")
    mutated = dict(registry)
    mutated["scope"] = dict(registry["scope"])
    mutated["scope"]["claim_ceiling"] = "causal"
    path = tmp_path / "mutated.json"
    import json

    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="claim ceiling"):
        MODULE.load_preregistry(path)
