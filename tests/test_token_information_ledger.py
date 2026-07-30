import numpy as np
import pytest
from sklearn.linear_model import Ridge

from src.analysis.token_information_ledger import (
    TOKEN_INFORMATION_LEDGER_SCHEMA_VERSION,
    InformationLedgerConfig,
    build_token_representations,
    evaluate_information_ledger,
)


def _synthetic_split(rng, subjects, samples_per_subject):
    subject_vector = np.repeat(subjects, samples_per_subject)
    latent = rng.normal(size=(len(subject_vector), 4))
    target = np.column_stack(
        (
            1.5 * latent[:, 0] - 0.75 * latent[:, 1],
            latent[:, 2] + 0.1 * rng.normal(size=len(latent)),
        )
    )
    return subject_vector, latent, target


def test_grouped_ledger_predictable_representation_beats_noise_on_validation():
    rng = np.random.default_rng(23)
    train_subjects, train_latent, train_target = _synthetic_split(
        rng, [f"train_{index}" for index in range(8)], 30
    )
    validation_subjects, validation_latent, validation_target = _synthetic_split(
        rng, [f"validation_{index}" for index in range(4)], 30
    )
    train_noise = rng.normal(size=train_latent.shape)
    validation_noise = rng.normal(size=validation_latent.shape)

    result = evaluate_information_ledger(
        train_target,
        validation_target,
        train_subjects,
        validation_subjects,
        {
            "continuous_latent": train_latent,
            "noise": train_noise,
        },
        {
            "continuous_latent": validation_latent,
            "noise": validation_noise,
        },
        coordinate_names=["linear_exact", "linear_noisy"],
        config=InformationLedgerConfig(
            alphas=(0.0, 0.1, 1.0),
            bootstrap_iterations=128,
            seed=8,
        ),
    )

    predictable = result["representations"]["continuous_latent"]
    noise = result["representations"]["noise"]
    assert result["schema_version"] == TOKEN_INFORMATION_LEDGER_SCHEMA_VERSION
    assert result["status"] == "ok"
    assert predictable["mean_r2"] > 0.98
    assert predictable["mean_r2"] > noise["mean_r2"] + 0.8
    assert predictable["probe_selection"]["selection_data"] == "training_only"
    assert predictable["validation_used_for_model_selection"] is False
    assert set(predictable["subject_r2"]) == set(validation_subjects)
    assert predictable["subject_bootstrap"]["status"] == "ok"
    assert predictable["subject_bootstrap"]["resampling_unit"] == (
        "validation_subject"
    )


def test_common_mask_fast_path_uses_one_multioutput_fit_per_fold_and_alpha(
    monkeypatch,
):
    rng = np.random.default_rng(91)
    train_features = rng.normal(size=(100, 6))
    validation_features = rng.normal(size=(40, 6))
    weights = rng.normal(size=(6, 12))
    train_target = train_features @ weights
    validation_target = validation_features @ weights
    train_subjects = np.repeat([f"s{index}" for index in range(5)], 20)
    validation_subjects = np.repeat(["v1", "v2"], 20)
    original_fit = Ridge.fit
    fit_calls = 0

    def counted_fit(self, features, target, *args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        return original_fit(self, features, target, *args, **kwargs)

    monkeypatch.setattr(Ridge, "fit", counted_fit)
    config = InformationLedgerConfig(
        alphas=(0.0, 0.1, 1.0),
        max_group_folds=5,
        bootstrap_iterations=0,
    )
    result = evaluate_information_ledger(
        train_target,
        validation_target,
        train_subjects,
        validation_subjects,
        {"latent": train_features},
        {"latent": validation_features},
        config=config,
    )

    latent = result["representations"]["latent"]
    expected_cv_fits = config.max_group_folds * len(config.alphas)
    assert fit_calls == expected_cv_fits + 1
    assert latent["probe_selection"]["fit_strategy"] == (
        "multi_output_common_mask"
    )
    assert latent["frozen_fit_strategy"] == "multi_output_common_mask"
    assert latent["probe_selection"]["ridge_fit_count"] == expected_cv_fits
    assert latent["mean_r2"] > 0.99


def test_ledger_skips_when_training_has_fewer_than_two_subjects():
    target = np.arange(12, dtype=float).reshape(6, 2)
    result = evaluate_information_ledger(
        target,
        target,
        np.repeat("only_subject", 6),
        np.repeat(["v1", "v2"], 3),
        {"latent": target},
        {"latent": target},
    )

    assert result["status"] == "skipped"
    assert result["skipped_reason"] == "fewer_than_two_train_subjects"
    assert result["representations"] == {}


def test_nan_target_is_usable_but_constant_coordinate_is_not_scored():
    rng = np.random.default_rng(11)
    train_features = rng.normal(size=(80, 3))
    validation_features = rng.normal(size=(40, 3))
    train_target = np.column_stack(
        (2.0 * train_features[:, 0], np.ones(len(train_features)))
    )
    validation_target = np.column_stack(
        (2.0 * validation_features[:, 0], np.ones(len(validation_features)))
    )
    train_target[[2, 17, 39], 0] = np.nan
    validation_target[[1, 9], 0] = np.nan

    result = evaluate_information_ledger(
        train_target,
        validation_target,
        np.repeat([f"s{index}" for index in range(4)], 20),
        np.repeat(["v1", "v2"], 20),
        {"latent": train_features},
        {"latent": validation_features},
        coordinate_names=["variable", "constant"],
        config=InformationLedgerConfig(
            alphas=(0.0, 1.0),
            bootstrap_iterations=32,
        ),
    )

    representation = result["representations"]["latent"]
    assert representation["status"] == "ok"
    assert representation["coordinate_r2"][0] > 0.99
    assert representation["coordinate_r2"][1] is None
    assert representation["coordinate_status"][1]["skipped_reason"] == (
        "constant_train_target"
    )
    assert result["train_target_nonfinite_count"] == 3


def test_coordinatewise_fallback_preserves_different_nan_masks():
    rng = np.random.default_rng(14)
    train_features = rng.normal(size=(80, 3))
    validation_features = rng.normal(size=(40, 3))
    train_target = np.column_stack(
        (train_features[:, 0], 2.0 * train_features[:, 1])
    )
    validation_target = np.column_stack(
        (validation_features[:, 0], 2.0 * validation_features[:, 1])
    )
    train_target[[1, 8], 0] = np.nan
    train_target[[3, 19, 27], 1] = np.nan

    result = evaluate_information_ledger(
        train_target,
        validation_target,
        np.repeat([f"s{index}" for index in range(4)], 20),
        np.repeat(["v1", "v2"], 20),
        {"latent": train_features},
        {"latent": validation_features},
        config=InformationLedgerConfig(
            alphas=(0.0, 1.0),
            bootstrap_iterations=0,
        ),
    )

    latent = result["representations"]["latent"]
    assert latent["probe_selection"]["fit_strategy"] == (
        "coordinatewise_missing_mask"
    )
    assert latent["frozen_fit_strategy"] == "coordinatewise_missing_mask"
    assert np.all(np.asarray(latent["coordinate_r2"]) > 0.99)


def test_all_constant_targets_return_explicit_skip_instead_of_finite_score():
    representation = np.arange(30, dtype=float).reshape(15, 2)
    result = evaluate_information_ledger(
        np.ones((15, 2)),
        np.ones((10, 2)),
        np.repeat(["s1", "s2", "s3"], 5),
        np.repeat(["v1", "v2"], 5),
        {"latent": representation},
        {"latent": representation[:10]},
        config=InformationLedgerConfig(bootstrap_iterations=0),
    )

    latent = result["representations"]["latent"]
    assert result["status"] == "skipped"
    assert latent["status"] == "skipped"
    assert latent["skipped_reason"] == "no_evaluable_train_target_coordinates"
    assert {
        row["skipped_reason"]
        for row in latent["probe_selection"]["coordinate_train_status"]
    } == {"constant_train_target"}


def test_invalid_shapes_and_no_common_representation_return_skip_reasons():
    target = np.arange(20, dtype=float).reshape(10, 2)
    subjects = np.repeat(["s1", "s2"], 5)

    invalid_target = evaluate_information_ledger(
        target[:, 0],
        target,
        subjects,
        subjects,
        {"latent": target},
        {"latent": target},
    )
    assert invalid_target["status"] == "skipped"
    assert invalid_target["skipped_reason"] == "invalid_train_target_shape"

    invalid_representation = evaluate_information_ledger(
        target,
        target,
        subjects,
        subjects,
        {"latent": target[:-1]},
        {"latent": target},
    )
    assert invalid_representation["status"] == "skipped"
    assert invalid_representation["representations"]["latent"][
        "skipped_reason"
    ] == "invalid_train_representation_shape"

    no_common = evaluate_information_ledger(
        target,
        target,
        subjects,
        subjects,
        {"train_only": target},
        {"validation_only": target},
    )
    assert no_common["status"] == "skipped"
    assert no_common["skipped_reason"] == "no_common_representations"


def test_representation_helper_flattens_tokens_and_marks_invalid_rows():
    latent = np.arange(24, dtype=float).reshape(2, 3, 4)
    hard = np.asarray([[0, 1, 2], [1, -1, 0]])
    posterior = np.full((2, 3, 3), 1 / 3)
    codebook = np.asarray([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
    valid = np.asarray([[True, True, False], [True, True, True]])

    result = build_token_representations(
        continuous_latent=latent,
        hard_ids=hard,
        posterior=posterior,
        codebook=codebook,
        valid_mask=valid,
    )

    assert result["continuous_latent"].shape == (6, 4)
    assert result["hard_one_hot"].shape == (6, 3)
    assert result["posterior"].shape == (6, 3)
    assert result["codebook_embedding"].shape == (6, 2)
    assert np.allclose(result["hard_one_hot"][1], [0.0, 1.0, 0.0])
    assert np.all(np.isnan(result["hard_one_hot"][2]))
    assert np.all(np.isnan(result["hard_one_hot"][4]))
    assert np.allclose(result["codebook_embedding"][5], [1.0, 0.0])

    with pytest.raises(ValueError, match="leading sample shape"):
        build_token_representations(
            continuous_latent=latent,
            posterior=np.ones((2, 4, 3)),
        )
