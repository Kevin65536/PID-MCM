from argparse import Namespace
from pathlib import Path

import numpy as np
import yaml

from experiments.probe_existing_lagged_predictability import (
    CONDITIONS,
    LAG_SECONDS,
    LAG_TOKENS,
    REPRESENTATIONS,
    _pair_data,
    build_lagged_pairs,
    evaluate_cell,
    make_derangement,
    make_synthetic_fixture,
    run,
    validate_config,
)


CONFIG_PATH = Path(
    "experiments/configs/physiology_semantic_tokenizer/probe_existing_lagged_predictability.yaml"
)


def test_probe_config_freezes_lag_bank_and_protected_open():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_config(config)
    assert tuple(config["lag_bank_seconds"]) == LAG_SECONDS
    assert tuple(config["evaluation_conditions"]) == CONDITIONS
    assert config["source"]["protected_open"] is False


def test_pair_sign_convention_and_masks_preserve_negative_lag():
    source = np.arange(1 * 5 * 2, dtype=np.float32).reshape(1, 5, 2)
    target = (100 + np.arange(1 * 5 * 3, dtype=np.float32)).reshape(1, 5, 3)
    token_mask = np.ones((1, 5), dtype=bool)
    source_feature_mask = np.ones_like(source, dtype=bool)
    target_feature_mask = np.ones_like(target, dtype=bool)
    subjects = np.asarray(["s1"])
    sample_ids = np.asarray(["trial-1"])

    positive = _pair_data(
        source,
        target,
        token_mask,
        token_mask,
        source_feature_mask,
        target_feature_mask,
        subjects,
        sample_ids,
        lag_tokens=2,
    )
    assert positive.source_token_index.tolist() == [0, 1, 2]
    assert positive.target_token_index.tolist() == [2, 3, 4]
    assert positive.grid_target_index[0].tolist() == [2, 3, 4, -1, -1]

    negative = _pair_data(
        source,
        target,
        token_mask,
        token_mask,
        source_feature_mask,
        target_feature_mask,
        subjects,
        sample_ids,
        lag_tokens=-2,
    )
    assert negative.source_token_index.tolist() == [2, 3, 4]
    assert negative.target_token_index.tolist() == [0, 1, 2]
    assert negative.grid_target_index[0].tolist() == [-1, -1, 0, 1, 2]

    public = build_lagged_pairs(source, target, token_mask, token_mask, lag_tokens=2)
    assert public.source.shape == (3, 2)
    assert public.target.shape == (3, 3)
    assert public.sample_index.tolist() == [0, 0, 0]
    empty = build_lagged_pairs(source, target, token_mask, token_mask, lag_tokens=10)
    assert empty.source.shape[0] == 0
    assert not empty.pair_mask.any()

    source_feature_mask[0, 2, 1] = False
    masked = _pair_data(
        source,
        target,
        token_mask,
        token_mask,
        source_feature_mask,
        target_feature_mask,
        subjects,
        sample_ids,
        lag_tokens=0,
    )
    assert not bool(masked.grid_mask[0, 2])
    assert len(masked.x) == 4


def test_derangement_is_same_subject_condition_and_nonidentity():
    subjects = np.asarray(["s1", "s1", "s1", "s1", "s2", "s2"])
    conditions = np.asarray(["a", "a", "b", "b", "a", "a"])
    sample_ids = np.asarray([f"trial-{i}" for i in range(len(subjects))])
    donor = make_derangement(subjects, conditions, sample_ids, seed=13)
    assert np.all(donor != np.arange(len(donor)))
    assert np.array_equal(subjects, subjects[donor])
    assert np.array_equal(conditions, conditions[donor])


def test_synthetic_cell_reports_subject_rows_two_representations_and_advantage():
    fixture = make_synthetic_fixture(seed=4)
    metrics, advantages, saved = evaluate_cell(
        "fixture",
        7,
        fixture["fit"],
        fixture["validation"],
        alpha=1.0,
        components=4,
        circular_shift_tokens=1,
        donor_seed=7,
    )
    assert len(metrics) == len(REPRESENTATIONS) * len(CONDITIONS) * len(LAG_TOKENS) * 2
    assert len(advantages) == len(REPRESENTATIONS) * 2 * len(LAG_TOKENS) * 2
    assert {row["evaluation_unit"] for row in metrics} == {"subject"}
    assert {row["token_or_window_as_replicate"] for row in metrics} == {False}
    assert {row["condition"] for row in metrics} == set(CONDITIONS)
    assert {row["lag_seconds"] for row in metrics} == set(LAG_SECONDS)
    assert all(row["pair_mask_sha256"] and row["null_policy"] for row in metrics)
    assert {row["claim_status"] for row in metrics} == {
        "post_selection_development_offline_delayed_association"
    }
    assert {row["causal_future_claim"] for row in metrics} == {False}
    assert all(
        row["negative_lag"] is (row["lag_seconds"] < 0)
        for row in metrics
    )
    assert "shared_pair_mask" in saved
    assert "native_pair_mask" in saved
    assert "fit_eeg_native_mask" in saved
    assert "eeg_shared_source" in saved and "fnirs_shared_target" in saved
    assert saved["fit_target_mask"].shape == fixture["fit"].target_mask.shape
    assert saved["shared_pair_mask"].dtype == bool
    assert saved["native_pair_mask"].dtype == bool
    assert {row["comparison"] for row in advantages} == {
        "matched_minus_deranged_same_subject_same_condition_nonidentity",
        "matched_minus_within_trial_circular_shift",
    }


def test_all_invalid_lag_is_explicitly_skipped_without_false_positive():
    fixture = make_synthetic_fixture(seed=12)
    for bundle in fixture.values():
        bundle.eeg_shared_mask[...] = False
        bundle.fnirs_shared_mask[...] = False
        bundle.eeg_native_mask[...] = False
        bundle.fnirs_native_mask[...] = False
        bundle.target_mask[...] = False
    metrics, advantages, _ = evaluate_cell(
        "invalid_fixture",
        9,
        fixture["fit"],
        fixture["validation"],
        alpha=1.0,
        components=4,
        circular_shift_tokens=1,
        donor_seed=9,
    )
    assert {row["status"] for row in metrics} == {"skipped"}
    assert {row["skipped_reason"] for row in metrics} == {"no_supported_lag_pairs"}
    assert all(np.isnan(float(row["delta_r2"])) for row in metrics)
    assert {row["status"] for row in advantages} == {"skipped"}


def test_fixture_cli_publishes_atomically_with_manifest(tmp_path):
    fixture = make_synthetic_fixture(seed=8)
    fixture_path = tmp_path / "fixture.npz"
    np.savez_compressed(
        fixture_path,
        fit_eeg_shared=fixture["fit"].eeg_shared,
        fit_fnirs_shared=fixture["fit"].fnirs_shared,
        fit_eeg_native=fixture["fit"].eeg_native,
        fit_fnirs_native=fixture["fit"].fnirs_native,
        fit_eeg_shared_mask=fixture["fit"].eeg_shared_mask,
        fit_fnirs_shared_mask=fixture["fit"].fnirs_shared_mask,
        fit_eeg_native_mask=fixture["fit"].eeg_native_mask,
        fit_fnirs_native_mask=fixture["fit"].fnirs_native_mask,
        fit_subject=fixture["fit"].subject,
        fit_condition=fixture["fit"].condition,
        fit_sample_id=fixture["fit"].sample_id,
        validation_eeg_shared=fixture["validation"].eeg_shared,
        validation_fnirs_shared=fixture["validation"].fnirs_shared,
        validation_eeg_native=fixture["validation"].eeg_native,
        validation_fnirs_native=fixture["validation"].fnirs_native,
        validation_eeg_shared_mask=fixture["validation"].eeg_shared_mask,
        validation_fnirs_shared_mask=fixture["validation"].fnirs_shared_mask,
        validation_eeg_native_mask=fixture["validation"].eeg_native_mask,
        validation_fnirs_native_mask=fixture["validation"].fnirs_native_mask,
        validation_subject=fixture["validation"].subject,
        validation_condition=fixture["validation"].condition,
        validation_sample_id=fixture["validation"].sample_id,
    )
    config = CONFIG_PATH
    output = tmp_path / "published"
    result = run(
        Namespace(
            config=config,
            source_run=None,
            fixture=fixture_path,
            output_dir=output,
            smoke=False,
        )
    )
    assert result == output
    assert (output / "manifest.json").exists()
    assert (output / "lag_probe_subject_metrics.csv").exists()
    assert (output / "lag_probe_subject_summary.csv").exists()
    manifest = yaml.safe_load((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["protected_open"] is False
    assert manifest["encoder_retrained"] is False
    assert manifest["evaluation_unit"] == "subject"
    assert manifest["causal_future_claim"] is False
    assert manifest["fresh_fit_held_out"] is False
    assert manifest["forbidden_model_fields_not_used"] == ["target", "eeg_driver", "fnirs_driver"]
    assert not list(tmp_path.glob(".published.staging-*"))
