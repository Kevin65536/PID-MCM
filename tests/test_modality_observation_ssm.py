import numpy as np

from src.inference.modality_observation_ssm import (
    apply_joint_observation_ssm,
    apply_observation_ssm,
    apply_observation_ssm_batch,
    fit_joint_observation_ssm,
    fit_observation_ssm,
)


def _sequences(seed: int = 7, count: int = 12, steps: int = 30):
    rng = np.random.default_rng(seed)
    output = []
    for _ in range(count):
        state = np.zeros((steps, 2), dtype=np.float64)
        for index in range(1, steps):
            state[index, 0] = 0.90 * state[index - 1, 0] + rng.normal(scale=0.15)
            state[index, 1] = 0.75 * state[index - 1, 1] + rng.normal(scale=0.12)
        output.append(state + rng.normal(scale=0.35, size=state.shape))
    return output


def test_modality_specific_observation_ssm_is_stable_and_uncertain():
    sequences = _sequences()
    fit = fit_observation_ssm(
        sequences[:8],
        feature_names=("alpha_left", "alpha_right"),
        provenance_id="fit-only-no-labels",
    )
    assert max(abs(np.linalg.eigvals(fit.transition))) < 1.0
    assert fit.fit_scope == "fit_parameter_all_conditions"
    result = apply_observation_ssm(sequences[8], fit)
    assert result.reconstructed.shape == sequences[8].shape
    assert result.posterior_std.shape == sequences[8].shape
    assert result.observation_predictive_std.shape == sequences[8].shape
    assert np.all(result.posterior_std > 0.0)
    assert np.all(result.observation_predictive_std > 0.0)
    # Smoothing should not exactly copy a noisy observation.
    assert not np.allclose(result.reconstructed, sequences[8])


def test_observation_ssm_supports_featurewise_missingness():
    sequences = _sequences()
    masks = [np.ones_like(value, dtype=bool) for value in sequences[:8]]
    masks[0][4:7, 1] = False
    fit = fit_observation_ssm(
        sequences[:8],
        feature_names=("a", "b"),
        valid_masks=masks,
    )
    apply_mask = np.ones_like(sequences[8], dtype=bool)
    apply_mask[10:13, 0] = False
    result = apply_observation_ssm(sequences[8], fit, valid_mask=apply_mask)
    assert np.all(np.isfinite(result.reconstructed))
    assert np.all(result.residual[~apply_mask] == 0.0)


def test_batch_smoother_matches_single_sequence_results():
    sequences = _sequences()
    fit = fit_observation_ssm(
        sequences[:8], feature_names=("a", "b")
    )
    batch = apply_observation_ssm_batch(np.stack(sequences[8:10]), fit)
    for index in range(2):
        single = apply_observation_ssm(sequences[8 + index], fit)
        np.testing.assert_allclose(batch.reconstructed[index], single.reconstructed)
        np.testing.assert_allclose(
            batch.observation_predictive_std[index],
            single.observation_predictive_std,
        )


def test_joint_teacher_has_named_modality_projections():
    eeg = _sequences(seed=3)
    fnirs = [
        np.column_stack((
            np.roll(value[:, 0], 2),
            -0.5 * np.roll(value[:, 0], 2),
        ))
        for value in eeg
    ]
    fit, slices = fit_joint_observation_ssm(
        {"eeg": eeg[:8], "fnirs": fnirs[:8]},
        feature_names={"eeg": ("left", "right"), "fnirs": ("hbo", "hbr")},
        fit_scope="privileged_fit_parameter_all_conditions",
    )
    result = apply_joint_observation_ssm(
        {"eeg": eeg[8], "fnirs": fnirs[8]}, fit, slices
    )
    assert result.reconstructed["eeg"].shape == eeg[8].shape
    assert result.reconstructed["fnirs"].shape == fnirs[8].shape
    assert set(slices) == {"eeg", "fnirs"}
    assert fit.feature_names == (
        "eeg:left",
        "eeg:right",
        "fnirs:hbo",
        "fnirs:hbr",
    )
