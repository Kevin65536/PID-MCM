from __future__ import annotations

import numpy as np
import pytest

import experiments.evaluate_t3_identifiability as ident
from experiments.evaluate_t3_measured_reconstruction_null import load_config as load_measured_config


def test_invalid_boundary_fails_before_any_experiment_work(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config = ident.load_config()
    config["experiment"]["protected_data_enabled"] = True
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("experiment work must not start")

    monkeypatch.setattr(ident, "_synthetic_problem", forbidden)
    with pytest.raises(ValueError, match="boundary"):
        ident.run(config, tmp_path / "run")
    assert not called


def test_fit_only_loader_view_excludes_validation_and_protected_subjects() -> None:
    source = load_measured_config(ident.REPO_ROOT / ident.MEASURED_CONFIG_PATH)
    view = ident._fit_only_measured_config(source)
    condition = view["data"]["conditions"][0]
    assert condition["subjects"] == [f"subject_{index:02d}" for index in range(1, 19)]
    assert condition["validation_subjects"] == []
    assert condition["protected_subjects"] == []
    assert not set(condition["subjects"]) & {f"subject_{index:02d}" for index in range(19, 30)}


def test_representative_selection_is_deterministic_at_median_tie() -> None:
    rows = ident.select_representative_fit_subjects({"subject_03": 0.0, "subject_02": 2.0, "subject_01": 1.0, "subject_04": 3.0})
    assert [(row["role"], row["subject"]) for row in rows] == [
        ("low", "subject_03"),
        ("median", "subject_01"),
        ("high", "subject_04"),
    ]


def test_transformed_multistarts_are_exact_reproducible_and_in_bounds() -> None:
    specs = {
        "beta": {"bounds": [0.25, 4.0], "prior_mean": 1.0},
        "kappa": {"bounds": [0.2, 1.5], "prior_mean": 0.64},
        "tau": {"bounds": [0.5, 5.0], "prior_mean": 2.0},
    }
    starts = ident.transformed_multistarts(("beta", "kappa", "tau"), specs, 16, 7)
    assert starts == ident.transformed_multistarts(("beta", "kappa", "tau"), specs, 16, 7)
    assert len(starts) == 16
    bounds = ident._transformed_bounds(("beta", "kappa", "tau"), specs)
    assert all(lower <= row[index] <= upper for row in starts for index, (lower, upper) in enumerate(bounds))


def test_profile_point_reoptimizes_companion_instead_of_taking_fixed_slice() -> None:
    objective = lambda vector: float((vector[0] + vector[1] - 3.0) ** 2 + 0.2 * vector[1] ** 2)
    result = ident.minimize_profile_point(
        objective,
        ((-5.0, 5.0), (-5.0, 5.0)),
        0,
        1.0,
        ((0.0, 0.0), (2.0, 2.0)),
        max_iterations=100,
        ftol=1.0e-12,
    )
    assert result["x"][0] == 1.0
    assert result["x"][1] == pytest.approx(2.0 / 1.2, abs=1.0e-5)
    assert result["objective"] < objective(np.asarray([1.0, 0.0]))


def test_expanded_bounds_and_whitened_svd_diagnostics() -> None:
    assert ident.expanded_transformed_bounds(((0.0, 2.0),), 0.25) == ((-0.5, 2.5),)
    with pytest.raises(ValueError, match="finite"):
        ident.expanded_transformed_bounds(((-1.0e308, 1.0e308),), 0.25)
    rank_one = np.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    _, summary_one = ident.svd_diagnostics(rank_one, ("a", "b"), 0.05)
    _, summary_two = ident.svd_diagnostics(np.eye(2), ("a", "b"), 0.05)
    assert summary_one["effective_rank"] == 1
    assert summary_two["effective_rank"] == 2


def test_expansion_status_distinguishes_old_and_new_boundaries() -> None:
    assert ident._bound_expansion_status("beta", ("beta",), ("beta",), ("beta",)) == "persistent_at_expanded_boundary"
    assert ident._bound_expansion_status("beta", ("beta",), ("beta",), ()) == "active_constraint_relieved_estimate_at_or_beyond_registered_limit"
    assert ident._bound_expansion_status("beta", ("beta",), (), ()) == "relieved_after_expansion_into_registered_interior"
    assert ident._bound_expansion_status("beta", (), ("beta",), ()) == "moved_to_or_beyond_registered_boundary"


def test_near_duplicate_parameter_sets_do_not_claim_nonidentifiability() -> None:
    config = ident.load_config()
    support = {
        name: {
            "grid_points": 9,
            "converged_finite_grid_points": 9,
            "support_lower": 1.0,
            "support_upper": 1.0,
            "touches_lower_grid": name == "beta",
            "touches_upper_grid": False,
        }
        for name in ident.ACTIVE_PARAMETERS
    }
    states = [
        {"candidate_source": "reference", "observation_whitened_rmse": 0.0, "driver_nrmse": 0.0, "driver_correlation": 1.0, "parameter_distance_fraction_of_transformed_span": 0.0},
        {"candidate_source": "duplicate", "observation_whitened_rmse": 0.01, "driver_nrmse": 0.01, "driver_correlation": 0.99, "parameter_distance_fraction_of_transformed_span": 0.001},
    ]
    interpretation, _ = ident._interpret_case(support, ("beta",), ("beta",), (), True, {"observation_whitened_rmse": 0.5}, True, states, 5, config)
    assert interpretation == "inconclusive"
    states[1]["parameter_distance_fraction_of_transformed_span"] = 0.02
    interpretation, _ = ident._interpret_case(support, ("beta",), ("beta",), (), True, {"observation_whitened_rmse": 0.5}, True, states, 5, config)
    assert interpretation == "parameters_nonidentifiable_but_state_stable"


def test_near_optimal_candidates_are_not_reduced_to_parameter_extremes() -> None:
    candidates = [
        {"likelihood_nll": 10.0, "values": {"beta": beta, "kappa": 0.5, "tau": 2.0}}
        for beta in (1.0, 2.0, 3.0)
    ]
    assert len(ident._near_optimal_candidates(candidates, ident.ACTIVE_PARAMETERS, 10.0, 1.0)) == 3


def test_disconnected_profile_support_is_not_contiguous() -> None:
    rows = [
        {"parameter": "beta", "grid_index": index, "fixed_value": float(index), "success": True, "likelihood_nll": delta, "delta_nll": delta}
        for index, delta in enumerate((0.0, 10.0, 0.0))
    ]
    support = ident._profile_support(rows, ("beta",), 1.0)["beta"]
    assert not support["support_is_contiguous"]


def test_numeric_and_measured_source_drift_fail_closed() -> None:
    config = ident.load_config()
    config["analysis"]["workers"] = 1.5
    with pytest.raises(ValueError, match="integers"):
        ident.validate_config(config)
    measured = load_measured_config(ident.REPO_ROOT / ident.MEASURED_CONFIG_PATH)
    measured["data"]["cache_root"] = "data/cache/another_cache"
    with pytest.raises(ValueError, match="canonical physiology cache"):
        ident._validate_measured_source(measured)
    measured = load_measured_config(ident.REPO_ROOT / ident.MEASURED_CONFIG_PATH)
    measured["data"]["conditions"].append(dict(measured["data"]["conditions"][0]))
    with pytest.raises(ValueError, match="exactly one measured condition"):
        ident._validate_measured_source(measured)


def test_empty_or_failed_profile_cannot_support_primary_endpoint() -> None:
    case = {
        "case_id": "case",
        "diagnostic_flags": {"profile_reference_consistent": True},
        "profile_support": {
            "beta": {
                "grid_points": 9,
                "converged_finite_grid_points": 0,
                "support_lower": float("nan"),
                "support_upper": float("nan"),
                "touches_lower_grid": False,
                "touches_upper_grid": False,
            }
        },
    }
    assert ident.primary_endpoint_values([case]) == {"case": False}
    case["profile_support"] = {}
    assert ident.primary_endpoint_values([case]) == {"case": False}


def test_profile_reference_consistency_is_required_for_each_parameter() -> None:
    rows = [
        {"parameter": "beta", "success": True, "likelihood_nll": 10.02},
        {"parameter": "kappa", "success": True, "likelihood_nll": 10.0},
        {"parameter": "tau", "success": True, "likelihood_nll": 10.0},
    ]
    differences, consistent = ident._profile_reference_check(rows, ident.ACTIVE_PARAMETERS, 10.0, 0.01)
    assert differences == pytest.approx({"beta": 0.02, "kappa": 0.0, "tau": 0.0})
    assert not consistent
