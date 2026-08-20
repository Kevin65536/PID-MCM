from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from experiments.analyze_lag_conditioned_spvq import (
    PROBE_L2,
    PROBE_MAX_ITER,
    PROBE_TARGET_LABEL_SMOOTHING,
    PROBE_TOL,
)
from experiments.run_lag_conditioned_spvq import (
    CANONICAL_SUBJECT_SPLITS,
    POSITIVE_LAGS_SECONDS,
    ChannelStandardizer,
    PreparedPartition,
    PreparedTask,
    PreparedTorchDataset,
    _balanced_indices,
    _jsonable,
    _lc_spvq_pretraining_losses,
    _native_targets_chunked,
    _resolve,
    _validate_prepared_governance,
    apply_channel_standardizer,
    fit_channel_standardizer,
    make_aligned_donor_time_negative_mask,
    make_prepared_loader,
    make_same_group_time_negative_mask,
    prepare_partition,
    prepare_task,
    run_preparation_only,
    run_training_suite,
    train_b0_variant,
    train_lc_spvq_variant,
    validate_config,
)
from src.analysis.lag_conditioned_native_features import (
    MaskedStandardizer,
    NativeFeatureTargets,
    extract_eeg_native_targets,
    extract_fnirs_native_targets,
)
from src.data.lag_conditioned_dataset import LagConditionedSampleIndex


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/physiology_semantic_tokenizer/lag_conditioned_spvq.yaml"


def _config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_reviewed_lc_spvq_config_is_valid_and_keeps_protected_closed():
    config = _config()
    validate_config(config)

    assert config["source"]["protected_open"] is False
    assert config["output"]["protected_open"] is False
    assert config["experiment"]["old_continuous_2_of_16_verdict_mutable"] is False
    assert tuple(config["objective"]["lag_seconds"]) == POSITIVE_LAGS_SECONDS
    assert config["quantizer"]["eeg_codebook_size"] == 16
    assert config["quantizer"]["fnirs_codebook_size"] == 16
    assert config["quantizer"]["independent_codebooks"] is True
    assert (
        config["statistics"]["q0_q1_train_target_label_smoothing"]
        == PROBE_TARGET_LABEL_SMOOTHING
    )
    assert config["statistics"]["q0_q1_evaluation_target_smoothing"] == 0.0
    assert config["statistics"]["q0_q1_probe_l2"] == PROBE_L2
    assert config["statistics"]["q0_q1_max_iter"] == PROBE_MAX_ITER
    assert config["statistics"]["q0_q1_tolerance"] == PROBE_TOL
    assert config["statistics"]["q0_q1_convergence_required"] is True


def test_canonical_subject_split_constants_are_deeply_immutable():
    with pytest.raises(TypeError):
        CANONICAL_SUBJECT_SPLITS["fit_parameter_subjects"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        CANONICAL_SUBJECT_SPLITS["protected_or_unused"][
            "eeg_fnirs_single_trial"
        ] = ()  # type: ignore[index]


def test_relative_source_paths_resolve_against_repository_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _resolve("data/cache/physiology_semantic_clean_v1") == (
        ROOT / "data/cache/physiology_semantic_clean_v1"
    ).resolve()


def test_config_validation_fails_on_protected_overlap_or_direct_map_optimization():
    overlap = _config()
    overlap["data_split"]["fit_parameter_subjects"]["eeg_fnirs_single_trial"].append(
        "subject_24"
    )
    with pytest.raises(PermissionError, match="overlap"):
        validate_config(overlap)

    shifted_fit = _config()
    shifted_fit["data_split"]["fit_parameter_subjects"][
        "eeg_fnirs_single_trial"
    ][0] = "subject_30"
    with pytest.raises(PermissionError, match="canonical reviewed split"):
        validate_config(shifted_fit)

    missing_boundary = _config()
    missing_boundary["data_split"]["protected_or_unused"][
        "eeg_fnirs_single_trial"
    ].remove("subject_29")
    with pytest.raises(PermissionError, match="canonical closed set"):
        validate_config(missing_boundary)

    wrong_source = _config()
    wrong_source["source"]["window_offset_s"] = 0.0
    with pytest.raises(ValueError, match="-5-second"):
        validate_config(wrong_source)

    broad_negative = _config()
    broad_negative["objective"]["hard_negative"] = "all_tokens"
    with pytest.raises(ValueError, match="hard-negative"):
        validate_config(broad_negative)

    cooccurrence = _config()
    cooccurrence["objective"]["token_cooccurrence_loss"] = True
    with pytest.raises(ValueError, match="cannot be optimized"):
        validate_config(cooccurrence)


def test_config_validation_preserves_old_verdict_and_k16_primary_capacity():
    mutable = _config()
    mutable["experiment"]["old_continuous_2_of_16_verdict_mutable"] = True
    with pytest.raises(ValueError, match="immutable"):
        validate_config(mutable)

    large = _config()
    large["quantizer"]["eeg_codebook_size"] = 128
    with pytest.raises(ValueError, match="K=16"):
        validate_config(large)


def test_full_training_remains_fail_closed_before_data_access(tmp_path):
    with pytest.raises(RuntimeError, match="fit-selection-only"):
        run_training_suite(
            _config(),
            CONFIG.resolve(),
            tmp_path / "full",
            smoke=False,
            requested_tasks=None,
            requested_variants=None,
            requested_device="cpu",
        )


def test_direct_training_helpers_cannot_bypass_full_mode_guard(tmp_path):
    common = {
        "prepared": None,
        "config": _config(),
        "seed": 1,
        "device": torch.device("cpu"),
        "output_dir": tmp_path,
        "smoke": False,
    }
    with pytest.raises(RuntimeError, match="fit-selection-only"):
        train_b0_variant(**common)
    with pytest.raises(RuntimeError, match="fit-selection-only"):
        train_lc_spvq_variant(**common, variant="M1")
    with pytest.raises(RuntimeError, match="fit-selection-only"):
        run_preparation_only(
            _config(),
            CONFIG.resolve(),
            tmp_path / "full-preparation",
            smoke=False,
        )
    with pytest.raises(RuntimeError, match="fit-selection-only"):
        prepare_task(_config(), "motor_imagery", smoke=False, derangement_seed=1)
    with pytest.raises(RuntimeError, match="full preparation"):
        prepare_partition(
            None,  # type: ignore[arg-type]
            role="fit_parameter",
            max_per_subject_class=None,
            derangement_seed=1,
        )


def test_direct_preparation_binds_config_object_to_yaml(tmp_path):
    drift = deepcopy(_config())
    drift["training"]["learning_rate"] *= 0.5
    with pytest.raises(ValueError, match="differs from the bound config"):
        run_preparation_only(
            drift,
            CONFIG.resolve(),
            tmp_path / "prepare",
            smoke=True,
        )


def test_channel_standardizer_uses_only_supported_points_and_channels():
    signal = np.asarray(
        [
            [[1.0, 3.0, 99.0], [10.0, 12.0, 14.0]],
            [[5.0, 7.0, 9.0], [1000.0, 1000.0, 1000.0]],
        ],
        dtype=np.float32,
    )
    point_mask = np.asarray([[True, True, False], [True, True, True]])
    channel_mask = np.asarray([[True, True], [True, False]])
    stats = fit_channel_standardizer(signal, point_mask, channel_mask)
    transformed = apply_channel_standardizer(
        signal, point_mask, channel_mask, stats
    )

    np.testing.assert_allclose(stats.mean, [5.0, 11.0])
    assert stats.count.tolist() == [5, 2]
    assert np.equal(transformed[0, 0, 2], 0.0)
    assert np.equal(transformed[1, 1], 0.0).all()
    np.testing.assert_allclose(
        transformed[:, 0][point_mask].mean(), 0.0, atol=1e-6
    )


def test_smoke_selector_retains_two_trials_per_subject_condition():
    rows = []
    for subject in ("s1", "s2"):
        for condition in ("a", "b"):
            for repeat in range(4):
                index = len(rows)
                rows.append(
                    LagConditionedSampleIndex(
                        base_index=index,
                        sample_id=f"{subject}-{condition}-{repeat}",
                        dataset_id="d",
                        task_id="t",
                        subject=subject,
                        record_id="r",
                        condition=condition,
                        class_index=0,
                        event_index=repeat,
                        event_time_ms=float(repeat * 30_000),
                        fnirs_event_time_ms=float(repeat * 30_000),
                    )
                )
    selected = _balanced_indices(rows, max_per_subject_class=2)
    admitted = [rows[int(index)] for index in selected]
    counts = {}
    for row in admitted:
        counts[(row.subject, row.condition)] = counts.get(
            (row.subject, row.condition), 0
        ) + 1
    assert len(selected) == 8
    assert set(counts.values()) == {2}


def test_json_boundary_handles_numpy_boolean_and_integer_scalars():
    payload = _jsonable({"passed": np.bool_(True), "count": np.int64(3)})
    assert payload == {"passed": True, "count": 3}


def _prepared_partition() -> PreparedPartition:
    count, tokens = 4, 2
    eeg_native = NativeFeatureTargets(
        np.zeros((count, tokens, 3), dtype=np.float32),
        np.ones((count, tokens, 3), dtype=bool),
        ("a", "b", "c"),
    )
    fnirs_native = NativeFeatureTargets(
        np.zeros((count, tokens, 4), dtype=np.float32),
        np.ones((count, tokens, 4), dtype=bool),
        ("d", "e", "f", "g"),
    )
    return PreparedPartition(
        role="fit_parameter",
        eeg=np.arange(count * 2 * 8, dtype=np.float32).reshape(count, 2, 8),
        fnirs=np.arange(count * 2 * 4, dtype=np.float32).reshape(count, 2, 4),
        eeg_point_mask=np.ones((count, 8), dtype=bool),
        fnirs_point_mask=np.ones((count, 4), dtype=bool),
        eeg_token_mask=np.ones((count, tokens), dtype=bool),
        fnirs_token_mask=np.ones((count, tokens), dtype=bool),
        eeg_channel_mask=np.ones((count, 2), dtype=bool),
        fnirs_channel_mask=np.ones((count, 2), dtype=bool),
        target=np.asarray([0, 0, 1, 1]),
        sample_id=np.asarray([f"sample-{i}" for i in range(count)]),
        subject=np.asarray(["s1", "s1", "s2", "s2"]),
        condition=np.asarray(["a", "a", "b", "b"]),
        record_id=np.asarray(["r"] * count),
        eeg_event_time_ms=np.arange(count, dtype=np.float64) * 30_000.0,
        fnirs_event_time_ms=np.arange(count, dtype=np.float64) * 30_000.0,
        eeg_channel_names=("e0", "e1"),
        fnirs_channel_names=("f0", "f1"),
        fnirs_component_roles=("HbO", "HbR"),
        eeg_native=eeg_native,
        fnirs_native=fnirs_native,
        donor_index=np.asarray([1, 0, 3, 2]),
    )


def test_prepared_governance_rejects_forged_capability():
    partition = _prepared_partition()
    channel = ChannelStandardizer(
        mean=np.zeros(2, dtype=np.float32),
        scale=np.ones(2, dtype=np.float32),
        count=np.ones(2, dtype=np.int64),
    )
    eeg_native = MaskedStandardizer(
        mean=np.zeros(3, dtype=np.float32),
        scale=np.ones(3, dtype=np.float32),
        count=np.ones(3, dtype=np.int64),
    )
    fnirs_native = MaskedStandardizer(
        mean=np.zeros(4, dtype=np.float32),
        scale=np.ones(4, dtype=np.float32),
        count=np.ones(4, dtype=np.int64),
    )
    forged = PreparedTask(
        task_id="motor_imagery",
        dataset_id="eeg_fnirs_single_trial",
        parameter=partition,
        selection=replace(partition, role="fit_selection"),
        development=replace(partition, role="development_apply"),
        eeg_standardizer=channel,
        fnirs_standardizer=channel,
        eeg_native_standardizer=eeg_native,
        fnirs_native_standardizer=fnirs_native,
        protected_metadata_indexed=True,
        measured_access_count=12,
        protected_measured_access_count=0,
        _preparation_capability=object(),
    )
    with pytest.raises(PermissionError, match="opaque preparation capability"):
        _validate_prepared_governance(forged)


def test_prepared_partition_rejects_identity_or_cross_group_donors():
    partition = _prepared_partition()
    with pytest.raises(ValueError, match="identity"):
        replace(partition, donor_index=np.arange(4, dtype=np.int64))
    with pytest.raises(ValueError, match="changes subject"):
        replace(partition, donor_index=np.asarray([2, 3, 0, 1], dtype=np.int64))
    overlapping = partition.fnirs_event_time_ms.copy()
    overlapping[1] = 10_000.0
    with pytest.raises(ValueError, match="overlapping"):
        replace(partition, fnirs_event_time_ms=overlapping)


def test_prepared_torch_dataset_returns_registered_same_group_donor():
    dataset = PreparedTorchDataset(_prepared_partition())
    item = dataset[0]
    assert int(item["donor_index"]) == 1
    assert item["donor_sample_id"] == "sample-1"
    np.testing.assert_array_equal(
        item["donor_fnirs"].numpy(), dataset.partition.fnirs[1]
    )
    assert item["eeg_native"].shape == (2, 3)
    assert item["fnirs_native_valid_mask"].dtype == torch.bool


def test_same_group_time_negative_mask_is_strict():
    mask = make_same_group_time_negative_mask(
        ("s1", "s1", "s1", "s2"),
        ("a", "a", "b", "a"),
        token_count=3,
    )
    assert mask.shape == (4, 3, 4, 3)
    assert torch.equal(mask[0, :, 1, :], torch.eye(3, dtype=torch.bool))
    assert not mask[0, :, 0, :].any()
    assert not mask[0, :, 2, :].any()
    assert not mask[0, :, 3, :].any()


def test_hard_negative_masks_admit_only_registered_other_trial_same_time():
    in_batch = make_same_group_time_negative_mask(
        ("s1", "s1"),
        ("a", "a"),
        token_count=2,
        query_trial_ids=(10, 11),
        target_trial_ids=(11, 10),
    )
    # Physical trial 10 is target row 1, so it must not be a negative for query 0.
    assert not in_batch[0, :, 1, :].any()
    assert torch.equal(in_batch[0, :, 0, :], torch.eye(2, dtype=torch.bool))

    donor = make_aligned_donor_time_negative_mask(batch_size=2, token_count=2)
    expected = torch.eye(2, dtype=torch.bool)
    assert torch.equal(donor[0, :, 0, :], expected)
    assert not donor[0, :, 1, :].any()
    assert torch.equal(donor[1, :, 1, :], expected)


def test_runner_threads_strict_masks_to_both_deranged_banks(monkeypatch):
    class FakeModel:
        fnirs_shared_encoder = staticmethod(lambda values, mask: values)
        eeg_shared_encoder = staticmethod(lambda values, mask: values)
        fnirs_projection_head = staticmethod(lambda values: values)
        eeg_projection_head = staticmethod(lambda values: values)

    class CaptureLag:
        kwargs = None

        def __call__(self, query, target, **kwargs):
            self.kwargs = kwargs
            return {"loss": query.sum() * 0.0}

    monkeypatch.setattr(
        "experiments.run_lag_conditioned_spvq.native_feature_prediction_loss",
        lambda *args, **kwargs: torch.tensor(0.0),
    )
    monkeypatch.setattr(
        "experiments.run_lag_conditioned_spvq.raw_patch_reconstruction_loss",
        lambda *args, **kwargs: torch.tensor(0.0),
    )
    monkeypatch.setattr(
        "experiments.run_lag_conditioned_spvq.weighted_pretraining_loss",
        lambda losses, weights: (sum(losses.values()), {}),
    )
    batch_size, tokens, dim = 4, 3, 2
    values = torch.randn(batch_size, tokens, dim)
    token_mask = torch.ones(batch_size, tokens, dtype=torch.bool)
    batch = {
        "index": torch.arange(batch_size),
        "donor_index": torch.tensor([1, 0, 3, 2]),
        "subject": ("s1", "s1", "s2", "s2"),
        "condition": ("a", "a", "b", "b"),
        "eeg": values,
        "fnirs": values,
        "donor_eeg": values.roll(1, 0),
        "donor_fnirs": values.roll(1, 0),
        "eeg_token_valid_mask": token_mask,
        "fnirs_token_valid_mask": token_mask,
        "donor_eeg_token_valid_mask": token_mask,
        "donor_fnirs_token_valid_mask": token_mask,
        "eeg_native": values,
        "fnirs_native": values,
        "donor_fnirs_native": values,
        "eeg_native_valid_mask": torch.ones_like(values, dtype=torch.bool),
        "fnirs_native_valid_mask": torch.ones_like(values, dtype=torch.bool),
        "donor_fnirs_native_valid_mask": torch.ones_like(values, dtype=torch.bool),
        "eeg_point_valid_mask": token_mask,
        "fnirs_point_valid_mask": token_mask,
        "donor_fnirs_point_valid_mask": token_mask,
        "eeg_channel_valid_mask": torch.ones(batch_size, tokens, dtype=torch.bool),
        "fnirs_channel_valid_mask": torch.ones(batch_size, tokens, dtype=torch.bool),
        "donor_fnirs_channel_valid_mask": torch.ones(
            batch_size, tokens, dtype=torch.bool
        ),
    }
    output = {
        "eeg_projection": values,
        "fnirs_projection": values,
        "eeg_native_target_prediction": values,
        "fnirs_native_target_prediction": values,
        "eeg_raw": values,
        "fnirs_raw": values,
    }
    capture = CaptureLag()
    _lc_spvq_pretraining_losses(
        FakeModel(),
        capture,
        output,
        batch,
        _config(),
        variant="M1",
        include_commitment=False,
    )
    kwargs = capture.kwargs
    assert kwargs is not None
    aligned = make_aligned_donor_time_negative_mask(
        batch_size=batch_size, token_count=tokens
    )
    assert torch.equal(kwargs["deranged_target_negative_mask"], aligned)
    assert torch.equal(kwargs["deranged_query_negative_mask"], aligned)
    negative = kwargs["negative_mask"]
    assert not negative[0, :, 0, :].any()
    assert torch.equal(negative[0, :, 1, :], torch.eye(tokens, dtype=torch.bool))
    assert not negative[0, :, 2:, :].any()


def test_prepared_loader_shuffle_is_seed_deterministic():
    partition = _prepared_partition()
    first = make_prepared_loader(
        partition, batch_size=2, shuffle=True, seed=9
    )
    second = make_prepared_loader(
        partition, batch_size=2, shuffle=True, seed=9
    )
    first_order = [int(value) for batch in first for value in batch["index"]]
    second_order = [int(value) for batch in second for value in batch["index"]]
    assert first_order == second_order


def test_chunked_native_targets_thread_per_sample_channel_masks():
    rng = np.random.default_rng(13)
    eeg = rng.normal(size=(2, 2, 400)).astype(np.float32)
    fnirs = rng.normal(size=(2, 4, 20)).astype(np.float32)
    eeg_token = np.ones((2, 1), dtype=bool)
    fnirs_token = np.ones((2, 1), dtype=bool)
    eeg_channel = np.asarray([[True, False], [False, True]])
    fnirs_channel = np.asarray(
        [[True, False, True, True], [True, True, False, True]]
    )
    eeg_names = ("eeg_0", "eeg_1")
    fnirs_names = ("hbo_0", "hbo_1", "hbr_0", "hbr_1")
    roles = ("HbO", "HbO", "HbR", "HbR")

    eeg_targets, fnirs_targets = _native_targets_chunked(
        eeg,
        fnirs,
        eeg_token,
        fnirs_token,
        eeg_channel_valid_mask=eeg_channel,
        fnirs_channel_valid_mask=fnirs_channel,
        eeg_channel_names=eeg_names,
        fnirs_channel_names=fnirs_names,
        fnirs_component_roles=roles,
        chunk_size=1,
    )

    expected_eeg_values = []
    expected_eeg_masks = []
    expected_fnirs_values = []
    expected_fnirs_masks = []
    for index in range(len(eeg)):
        eeg_indices = np.flatnonzero(eeg_channel[index])
        fnirs_indices = np.flatnonzero(fnirs_channel[index])
        eeg_expected = extract_eeg_native_targets(
            eeg[index : index + 1, eeg_indices],
            eeg_token[index : index + 1],
            channel_names=tuple(eeg_names[i] for i in eeg_indices),
        )
        fnirs_expected = extract_fnirs_native_targets(
            fnirs[index : index + 1, fnirs_indices],
            fnirs_token[index : index + 1],
            component_roles=tuple(roles[i] for i in fnirs_indices),
            channel_names=tuple(fnirs_names[i] for i in fnirs_indices),
        )
        expected_eeg_values.append(eeg_expected.values)
        expected_eeg_masks.append(eeg_expected.valid_mask)
        expected_fnirs_values.append(fnirs_expected.values)
        expected_fnirs_masks.append(fnirs_expected.valid_mask)

    np.testing.assert_allclose(
        eeg_targets.values, np.concatenate(expected_eeg_values, axis=0)
    )
    np.testing.assert_array_equal(
        eeg_targets.valid_mask, np.concatenate(expected_eeg_masks, axis=0)
    )
    np.testing.assert_allclose(
        fnirs_targets.values, np.concatenate(expected_fnirs_values, axis=0)
    )
    np.testing.assert_array_equal(
        fnirs_targets.valid_mask, np.concatenate(expected_fnirs_masks, axis=0)
    )
