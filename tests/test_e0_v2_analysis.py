import numpy as np

from experiments.e0_v2_measurement_audit import _split_subjects
from experiments.evaluate_physical_teacher_e0_v2 import _correct_legacy_wavelength_clean


def test_legacy_wavelength_clean_correction_recovers_canonical_mixture():
    state = np.zeros((20, 5), dtype=np.float64)
    state[:, 2] = np.linspace(-1.0, 2.0, 20)
    state[:, 3] = np.linspace(3.0, -2.0, 20)
    old_clean = np.column_stack((5.0 + 2.0 * state[:, 2], -3.0 + 4.0 * state[:, 3]))

    corrected = _correct_legacy_wavelength_clean(old_clean, state)

    np.testing.assert_allclose(corrected[:, 0], 5.0 + 2.0 * (state[:, 2] + 0.25 * state[:, 3]))
    np.testing.assert_allclose(corrected[:, 1], -3.0 + 4.0 * (0.35 * state[:, 2] + state[:, 3]))


def test_measurement_subject_split_keeps_protected_partition_disjoint():
    split = _split_subjects([str(index) for index in range(1, 11)], 0.6, 0.2)

    assert split["train"] == ["1", "2", "3", "4", "5", "6"]
    assert split["validation"] == ["7", "8"]
    assert split["protected_test"] == ["9", "10"]
    assert not (set(split["train"]) & set(split["validation"]))
    assert not (set(split["train"]) & set(split["protected_test"]))
    assert not (set(split["validation"]) & set(split["protected_test"]))
