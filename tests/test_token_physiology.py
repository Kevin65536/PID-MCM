import json

import numpy as np
import pytest

from src.analysis.token_physiology import (
    TOKEN_PHYSIOLOGY_SCHEMA_VERSION,
    TokenPhysiologyConfig,
    _normalized_mutual_information,
    analyze_token_physiology,
    match_token_signatures,
)


def _profile(result, profile_type, token_id, feature_name):
    return next(
        row
        for row in result.profile_rows
        if row["profile_type"] == profile_type
        and row["token_id"] == token_id
        and row["feature_name"] == feature_name
    )


def test_support_metrics_keep_inactive_codes_and_handle_masks_and_nan_ids():
    features = np.arange(12, dtype=float).reshape(6, 2)
    hard_ids = np.asarray([0.0, 0.0, 0.0, 1.0, np.nan, 99.0])
    subjects = np.asarray(["s1", "s1", "s2", "s2", "s3", "s3"])
    valid = np.asarray([True, True, True, True, True, False])

    result = analyze_token_physiology(
        features,
        hard_ids,
        subjects,
        valid_mask=valid,
        config=TokenPhysiologyConfig(
            codebook_size=3,
            min_count=2,
            min_subjects=2,
            rare_count=2,
            bootstrap_iterations=0,
        ),
    )

    token_zero = result.support_rows[0]
    assert token_zero["count"] == 3
    assert token_zero["subject_count"] == 2
    assert token_zero["max_subject_fraction"] == pytest.approx(2 / 3)
    expected_entropy = -(2 / 3 * np.log(2 / 3) + 1 / 3 * np.log(1 / 3))
    assert token_zero["subject_entropy"] == pytest.approx(expected_entropy)
    assert token_zero["effective_subjects"] == pytest.approx(np.exp(expected_entropy))
    assert token_zero["support_status"] == "sufficient"

    assert result.support_rows[1]["support_status"] == "insufficient"
    assert result.support_rows[2]["inactive"] is True
    assert result.manifest["invalid_hard_id_count"] == 1
    assert result.manifest["masked_count"] == 1
    assert result.manifest["analysis_valid_count"] == 4


def test_hard_and_soft_profiles_are_subject_equal_and_bootstrap_is_deterministic():
    features = np.asarray([[0.0], [0.0], [0.0], [10.0], [20.0]])
    hard_ids = np.asarray([0, 0, 0, 0, 1])
    subjects = np.asarray(["s1", "s1", "s1", "s2", "s2"])
    posterior = np.asarray(
        [
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
        ]
    )
    config = TokenPhysiologyConfig(
        codebook_size=2,
        min_count=1,
        min_subjects=1,
        bootstrap_iterations=256,
        seed=17,
    )

    first = analyze_token_physiology(
        features,
        hard_ids,
        subjects,
        feature_names=["alpha_power"],
        posterior=posterior,
        config=config,
    )
    second = analyze_token_physiology(
        features,
        hard_ids,
        subjects,
        feature_names=["alpha_power"],
        posterior=posterior,
        config=config,
    )

    hard = _profile(first, "hard", 0, "alpha_power")
    # Row weighting would give 2.5; equal weighting of s1=0 and s2=10 gives 5.
    assert hard["subject_equal_mean"] == pytest.approx(5.0)
    assert hard["subject_count"] == 2
    assert hard["bootstrap_ci_low"] == _profile(
        second, "hard", 0, "alpha_power"
    )["bootstrap_ci_low"]
    assert hard["bootstrap_ci_high"] == _profile(
        second, "hard", 0, "alpha_power"
    )["bootstrap_ci_high"]

    soft = _profile(first, "soft", 0, "alpha_power")
    expected_s2 = (0.8 * 10.0 + 0.1 * 20.0) / 0.9
    assert soft["subject_equal_mean"] == pytest.approx((0.0 + expected_s2) / 2)
    assert soft["effective_count"] <= soft["finite_count"]
    assert np.isfinite(soft["marginal_standardized_effect"])

    token_zero = first.support_rows[0]
    assert token_zero["posterior_valid_count"] == 4
    assert 0.0 <= token_zero["posterior_normalized_entropy_subject_equal_mean"] <= 1.0
    assert 0.0 <= token_zero["posterior_margin_subject_equal_mean"] <= 1.0


def test_state_contingency_reports_both_directions_lift_and_nmi():
    result = analyze_token_physiology(
        np.asarray([[1.0], [2.0], [3.0], [4.0]]),
        np.asarray([0, 0, 1, 1]),
        np.asarray(["s1", "s2", "s1", "s2"]),
        states={"arousal": np.asarray(["high", "high", "high", "low"])},
        metadata={"task": np.asarray(["a", "a", "b", "b"])},
        config=TokenPhysiologyConfig(
            codebook_size=2,
            min_count=1,
            min_subjects=1,
            bootstrap_iterations=0,
        ),
    )

    row = next(
        row
        for row in result.state_rows
        if row["token_id"] == 0 and row["category"] == "high"
    )
    assert row["p_category_given_token"] == pytest.approx(1.0)
    assert row["p_token_given_category"] == pytest.approx(2 / 3)
    assert row["lift"] == pytest.approx(4 / 3)
    assert row["subject_equal_p_category_given_token"] == pytest.approx(1.0)
    assert row["subject_equal_p_token_given_category"] == pytest.approx(0.75)
    assert row["subject_count_p_token_given_category"] == 2
    assert row["association_scope"].startswith(
        "patch-weighted and subject-equal"
    )
    assert 0.0 < row["normalized_mutual_information"] <= 1.0
    assert {row["variable_name"] for row in result.metadata_rows} == {"task"}


def test_normalized_mutual_information_handles_roundoff_for_one_state():
    # Summing 1 / 20 in float64 produces a state probability just above one.
    # Its entropy must be treated as zero instead of reaching sqrt as negative.
    joint = np.ones((20, 1), dtype=np.float64)

    with np.errstate(invalid="raise"):
        value = _normalized_mutual_information(joint)

    assert value == 0.0


def test_high_cardinality_state_is_skipped_instead_of_continuous_category():
    result = analyze_token_physiology(
        np.arange(4, dtype=float)[:, None],
        np.asarray([0, 1, 0, 1]),
        np.asarray(["s1", "s1", "s2", "s2"]),
        states={"continuous_like": np.asarray([0.1, 0.2, 0.3, 0.4])},
        config=TokenPhysiologyConfig(
            codebook_size=2,
            min_count=1,
            min_subjects=1,
            bootstrap_iterations=0,
            max_state_categories=3,
        ),
    )

    assert result.state_rows == []
    assert result.manifest["skipped_state_fields"] == ["continuous_like"]


def test_signature_matching_is_permutation_invariant_and_filters_low_support():
    features = np.asarray(
        [
            [2.0, 0.0],
            [2.2, 0.1],
            [0.0, 3.0],
            [0.1, 3.2],
            [10.0, 10.0],
        ]
    )
    subjects = np.asarray(["s1", "s2", "s1", "s2", "s1"])
    config = TokenPhysiologyConfig(
        codebook_size=3,
        min_count=2,
        min_subjects=2,
        bootstrap_iterations=0,
    )
    left = analyze_token_physiology(
        features,
        np.asarray([0, 0, 1, 1, 2]),
        subjects,
        feature_names=["f0", "f1"],
        config=config,
    )
    right = analyze_token_physiology(
        features,
        np.asarray([1, 1, 0, 0, 2]),
        subjects,
        feature_names=["f0", "f1"],
        config=config,
    )

    match = match_token_signatures(
        left,
        right,
        bootstrap_iterations=128,
        seed=4,
    )

    assert match["matched_count"] == 2
    assert match["mean_cosine"] == pytest.approx(1.0)
    assert {(row["left_code"], row["right_code"]) for row in match["matches"]} == {
        (0, 1),
        (1, 0),
    }
    assert all(row["left_code"] != 2 for row in match["matches"])
    assert match["fixed_alignment_bootstrap_ci_low"] == pytest.approx(1.0)
    assert match["fixed_alignment_bootstrap_ci_high"] == pytest.approx(1.0)


def test_nan_features_and_invalid_posterior_rows_do_not_poison_other_features():
    features = np.asarray([[1.0, np.nan], [3.0, 2.0], [5.0, 4.0]])
    posterior = np.asarray([[0.8, 0.2], [np.nan, np.nan], [0.2, 0.8]])
    result = analyze_token_physiology(
        features,
        np.asarray([0, 0, 1]),
        np.asarray(["s1", "s2", "s2"]),
        posterior=posterior,
        config=TokenPhysiologyConfig(
            codebook_size=2,
            min_count=1,
            min_subjects=1,
            bootstrap_iterations=0,
        ),
    )

    assert result.manifest["posterior_valid_count"] == 2
    assert result.manifest["posterior_invalid_count"] == 1
    assert _profile(result, "hard", 0, "feature_0")["finite_count"] == 2
    assert _profile(result, "hard", 0, "feature_1")["finite_count"] == 1
    assert _profile(result, "soft", 0, "feature_0")["weight_sum"] == pytest.approx(
        1.0
    )
    json.dumps(result.to_dict(), allow_nan=False)


def test_non_integral_ids_and_negative_posterior_are_rejected():
    with pytest.raises(ValueError, match="integer-valued"):
        analyze_token_physiology(
            np.ones((2, 1)),
            np.asarray([0.0, 0.5]),
            np.asarray(["s1", "s2"]),
            config=TokenPhysiologyConfig(codebook_size=2),
        )
    with pytest.raises(ValueError, match="non-negative"):
        analyze_token_physiology(
            np.ones((2, 1)),
            np.asarray([0, 1]),
            np.asarray(["s1", "s2"]),
            posterior=np.asarray([[1.0, 0.0], [-0.1, 1.1]]),
            config=TokenPhysiologyConfig(codebook_size=2),
        )


def test_result_manifest_is_versioned():
    result = analyze_token_physiology(
        np.ones((2, 1)),
        np.asarray([0, 1]),
        np.asarray(["s1", "s2"]),
        config=TokenPhysiologyConfig(
            codebook_size=2,
            min_count=1,
            min_subjects=1,
            bootstrap_iterations=0,
        ),
    )
    assert result.schema_version == TOKEN_PHYSIOLOGY_SCHEMA_VERSION
    assert result.manifest["schema_version"] == TOKEN_PHYSIOLOGY_SCHEMA_VERSION
