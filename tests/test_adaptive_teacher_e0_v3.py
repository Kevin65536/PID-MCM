from __future__ import annotations

import numpy as np

from experiments.evaluate_adaptive_teacher_e0_v3 import (
    _apply_local_target_contract,
    _features,
    _physical_patch_rows,
    _required_local_pass,
    _select_alpha_by_train_subject_cv,
    _state_target_variance,
    _state_targets,
)


def test_patch_features_preserve_ten_patch_contract() -> None:
    rng = np.random.default_rng(7)

    eeg = _features(rng.normal(size=200))
    fnirs = _features(rng.normal(size=(200, 2)))

    assert eeg.shape == (10, 12)
    assert fnirs.shape == (10, 24)
    assert np.all(np.isfinite(eeg))
    assert np.all(np.isfinite(fnirs))


def test_adaptive_state_targets_match_original_e0_coordinate_contract() -> None:
    time = np.linspace(-1.0, 1.0, 200)
    states = np.column_stack([(index + 1) * time for index in range(5)])

    eeg, fnirs, means, slopes = _state_targets(states)

    assert eeg.shape == (10, 4)
    assert fnirs.shape == (10, 6)
    assert means.shape == (10, 5)
    assert slopes.shape == (10, 5)
    np.testing.assert_allclose(eeg[:, 0], means[:, 4])
    np.testing.assert_allclose(eeg[:, 2], means[:, 0])
    np.testing.assert_allclose(fnirs[:, :3], means[:, 1:4])


def test_state_uncertainty_is_positive_and_coordinate_aligned() -> None:
    state_std = np.tile(np.arange(1.0, 6.0), (200, 1))

    eeg, fnirs = _state_target_variance(state_std)

    assert eeg.shape == (10, 4)
    assert fnirs.shape == (10, 6)
    assert np.all(eeg > 0.0)
    assert np.all(fnirs > 0.0)
    np.testing.assert_allclose(eeg[:, 0], 25.0)
    np.testing.assert_allclose(eeg[:, 2], 1.0)


def test_physical_rows_include_both_modalities_and_chromophores() -> None:
    rng = np.random.default_rng(11)
    rows, components = _physical_patch_rows(
        subject=19,
        heldout_trial=3,
        eeg_observed=rng.normal(size=200),
        eeg_clean=rng.normal(size=200),
        fnirs_observed=rng.normal(size=(200, 2)),
        fnirs_clean=rng.normal(size=(200, 2)),
    )

    assert len(rows) == 20
    assert {row["modality"] for row in rows} == {"eeg", "fnirs"}
    assert len(components) == 20
    assert {row["component"] for row in components} == {"hbo", "hbr"}


def test_ridge_alpha_is_selected_without_validation_subjects() -> None:
    rng = np.random.default_rng(17)
    groups = np.repeat(np.arange(1, 7), 20)
    feature = rng.normal(size=(len(groups), 3))
    target = 2.0 * feature[:, 0] - 0.5 * feature[:, 1] + rng.normal(scale=0.1, size=len(groups))

    alpha, score = _select_alpha_by_train_subject_cv(
        feature, target, groups, (0.01, 1.0, 100.0), folds=3,
    )

    assert alpha in {0.01, 1.0, 100.0}
    assert score > 0.95


def test_local_contract_requires_declared_coordinates_and_demotes_flow() -> None:
    contract = {
        "required_local_coordinates": {
            "eeg": ["r_mean", "r_slope"],
            "fnirs": ["delta_hbo_mean", "delta_hb_mean"],
        },
        "optional_local_coordinates": {
            "eeg": ["s_mean"],
            "fnirs": [],
        },
    }
    rows = [
        {"modality": "eeg", "coordinate": "r_mean", "statistically_observable": True},
        {"modality": "eeg", "coordinate": "r_slope", "statistically_observable": True},
        {"modality": "eeg", "coordinate": "s_mean", "statistically_observable": False},
        {"modality": "fnirs", "coordinate": "delta_f_mean", "statistically_observable": True},
        {"modality": "fnirs", "coordinate": "delta_hbo_mean", "statistically_observable": True},
        {"modality": "fnirs", "coordinate": "delta_hb_mean", "statistically_observable": False},
    ]

    resolved = _apply_local_target_contract(rows, contract)

    flow = next(row for row in resolved if row["coordinate"] == "delta_f_mean")
    assert flow["target_role"] == "context_only"
    assert not flow["admitted_local_target"]
    assert not _required_local_pass(resolved, contract)
