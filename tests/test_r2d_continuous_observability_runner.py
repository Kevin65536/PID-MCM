import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "experiments/scripts/run_r2d_continuous_observability.py"
)
SPEC = importlib.util.spec_from_file_location("r2d_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _config():
    path = (
        ROOT
        / "experiments/configs/physiology_semantic_tokenizer/"
        "r2d_continuous_observability.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_r2d_config_is_exact_development_only_contract():
    config = _config()
    runner.validate_config_contract(config)
    split = config["data"]["split"]
    assert tuple(split["train_subject_keys"]) == runner.TRAIN_SUBJECTS
    assert tuple(split["validation_subject_keys"]) == runner.VALIDATION_SUBJECTS
    assert tuple(split["protected_subject_keys"]) == runner.PROTECTED_SUBJECTS
    assert not {"test_subject_keys", "test_subjects", "test_loader"}.intersection(
        split
    )


def test_r2d_config_rejects_protected_or_forbidden_objective():
    config = _config()
    bad_split = copy.deepcopy(config)
    bad_split["data"]["split"]["train_subject_keys"][-1] = (
        "eeg_fnirs_single_trial|subject_24"
    )
    with pytest.raises(ValueError, match="01–18"):
        runner.validate_config_contract(bad_split)

    bad_objective = copy.deepcopy(config)
    bad_objective["objective"]["vector_quantization"] = True
    with pytest.raises(ValueError, match="vector_quantization"):
        runner.validate_config_contract(bad_objective)

    bad_test = copy.deepcopy(config)
    bad_test["data"]["split"]["test_subject_keys"] = list(
        runner.PROTECTED_SUBJECTS
    )
    with pytest.raises(PermissionError, match="test loader"):
        runner.validate_config_contract(bad_test)


def test_masked_point_mse_uses_only_exact_support():
    prediction = torch.tensor([[[1.0, 100.0], [4.0, 8.0]]])
    target = torch.tensor([[[0.0, 0.0], [2.0, 0.0]]])
    mask = torch.tensor([[[True, False], [True, False]]])
    loss = runner.masked_point_mse(prediction, target, mask)
    assert float(loss) == pytest.approx((1.0 + 4.0) / 2.0)
    with pytest.raises(ValueError, match="no supported points"):
        runner.masked_point_mse(prediction, target, torch.zeros_like(mask))


def test_point_masks_intersect_measurement_teacher_and_point_support():
    batch = {
        "token_valid_mask": {
            "eeg": torch.tensor([[True, False]]),
            "fnirs": torch.tensor([[False, True]]),
        },
        "teacher": {
            "teacher_mask": torch.tensor([[True, True]]),
            "target_point_valid_mask": torch.tensor(
                [[[True, False], [True, True]]]
            ),
        },
    }
    masks = runner.make_point_masks(batch)
    assert masks["eeg"].tolist() == [[[True, False], [False, False]]]
    assert masks["fnirs"].tolist() == [[[False, False], [True, True]]]


def test_phase_baseline_is_condition_by_relative_time_and_train_only():
    target = np.asarray(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
            [[10.0, 20.0]],
        ],
        dtype=np.float32,
    )
    masks = {
        "eeg": np.ones_like(target, dtype=bool),
        "fnirs": np.ones_like(target, dtype=bool),
    }
    baseline = runner.fit_phase_baseline(["A", "A", "B"], target, masks)
    np.testing.assert_allclose(baseline["eeg"]["A"], [[2.0, 3.0]])
    np.testing.assert_allclose(
        runner.phase_predictions(baseline, "fnirs", ["B", "A"]),
        [[[10.0, 20.0]], [[2.0, 3.0]]],
    )
    with pytest.raises(KeyError, match="absent"):
        runner.phase_predictions(baseline, "eeg", ["validation_only"])


def test_subject_metrics_and_cluster_bootstrap_use_subject_as_unit():
    subjects = ["s1", "s1", "s2"]
    target = np.ones((3, 1, 2), dtype=np.float32)
    phase = {
        "eeg": np.zeros_like(target),
        "fnirs": np.zeros_like(target),
    }
    predictions = {
        "eeg": np.asarray([[[0.5, 0.5]], [[0.5, 0.5]], [[2.0, 2.0]]]),
        "fnirs": np.asarray([[[1.0, 1.0]], [[1.0, 1.0]], [[0.0, 0.0]]]),
    }
    masks = {
        "eeg": np.ones_like(target, dtype=bool),
        "fnirs": np.ones_like(target, dtype=bool),
    }
    rows = runner.compute_subject_delta_r2(
        subjects, target, predictions, phase, masks
    )
    assert len(rows) == 6
    eeg = {row["subject"]: row for row in rows if row["modality"] == "eeg"}
    assert eeg["s1"]["delta_r2"] == pytest.approx(0.75)
    assert eeg["s2"]["delta_r2"] == pytest.approx(0.0)
    summary = runner.subject_cluster_bootstrap(
        rows,
        modality="eeg",
        iterations=1000,
        confidence_level=0.95,
        seed=7,
    )
    assert summary["subject_count"] == 2
    assert summary["positive_subject_count"] == 1
    assert summary["subject_equal_mean_delta_r2"] == pytest.approx(0.375)


def test_output_directory_refuses_overwrite(tmp_path):
    config = _config()
    existing = (
        ROOT
        / "experiments/runs/physiology_semantic_tokenizer/"
        "r2_continuous_observability"
        / f"pytest_existing_{tmp_path.name}"
    )
    existing.mkdir(parents=True, exist_ok=False)
    try:
        with pytest.raises(FileExistsError, match="refusing overwrite"):
            runner._prepare_run_dir(config, existing)
    finally:
        existing.rmdir()
