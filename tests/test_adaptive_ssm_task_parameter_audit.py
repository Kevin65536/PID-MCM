from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "adaptive_ssm_task_parameter_audit",
    ROOT / "experiments/analyze_adaptive_ssm_task_parameters.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_friedman_permutation_returns_null_for_identical_conditions() -> None:
    matrix = np.ones((8, 4), dtype=np.float64)
    statistic, p_value, kendall_w = MODULE.friedman_permutation_test(
        matrix,
        iterations=1000,
        seed=17,
    )
    assert statistic == 0.0
    assert p_value == 1.0
    assert kendall_w == 0.0


def test_friedman_permutation_detects_consistent_ordering() -> None:
    matrix = np.tile(np.asarray([0.0, 1.0, 2.0, 3.0]), (12, 1))
    statistic, p_value, kendall_w = MODULE.friedman_permutation_test(
        matrix,
        iterations=5000,
        seed=19,
    )
    assert statistic > 30.0
    assert p_value < 0.01
    assert np.isclose(kendall_w, 1.0)


def test_multiple_comparison_adjustments_are_monotone_and_restored() -> None:
    values = [0.04, 0.001, 0.03, 0.20]
    bh = MODULE.adjust_pvalues(values, "fdr_bh")
    holm = MODULE.adjust_pvalues(values, "holm")
    assert np.allclose(bh, [0.0533333333, 0.004, 0.0533333333, 0.20])
    assert np.allclose(holm, [0.09, 0.004, 0.09, 0.20])


def test_csv_boolean_strings_do_not_become_truthy() -> None:
    assert MODULE._as_bool("True")
    assert not MODULE._as_bool("False")
    assert not MODULE._as_bool("")
