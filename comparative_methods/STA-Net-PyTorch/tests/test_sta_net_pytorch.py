import sys
import json
from pathlib import Path

import numpy as np
import optuna
import torch
import yaml

METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
for path in (REPO_ROOT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sta_net_pytorch.data import (
    OFFICIAL_EEG_GRID,
    OFFICIAL_FNIRS_GRID,
    STANetSampleAdapter,
    get_sta_net_task_spec,
)
from sta_net_pytorch.model import STANet, STANetConfig, STANetObjective, SamePadConv3d
from train import (
    PackedRecordBatchSampler,
    RecordGroupedBatchSampler,
    classification_weights,
    initialize_model_from_checkpoint,
)
from launch_tuning import lane_plan, targeted_lane_plan, tuning_anchors
from tune import RUNG_EPOCHS, best_validation_metric_through_epoch, sample_config
from select_best_checkpoints import metric_contract, select_candidate
from sta_net_pytorch.metrics import classification_metrics as core_classification_metrics, improved
from sta_net_pytorch.splits import development_subject_split, validate_public_manifest
from visualize_results import classification_metrics, plot_regression_diagnostics, regression_metrics


def test_same_pad_conv_matches_keras_ceil_shape_for_strides():
    layer = SamePadConv3d(1, 4, kernel_size=(2, 2, 13), stride=(2, 2, 6))
    output = layer(torch.randn(2, 1, 16, 16, 60))
    assert output.shape == (2, 4, 8, 8, 10)


def test_sta_net_multiclass_forward_backward_is_finite():
    model = STANet(
        STANetConfig(
            task_type="classification",
            output_dim=3,
            dropout=0.0,
            embedding_dim=32,
            attention_heads=2,
            attention_key_dim=16,
            max_lags=3,
        )
    )
    outputs = model(torch.randn(2, 1, 16, 16, 60), torch.randn(2, 3, 2, 16, 16, 10))
    losses = STANetObjective("classification")(outputs, torch.tensor([0, 2]))
    losses["total"].backward()
    assert outputs["prediction"].shape == (2, 3)
    assert outputs["lag_attention"].shape == (2, 3)
    assert torch.isfinite(losses["total"])


def test_sta_net_sequence_regression_consumes_target_mask():
    model = STANet(
        STANetConfig(
            task_type="regression",
            output_dim=2,
            sequence_length=5,
            dropout=0.0,
            embedding_dim=32,
            attention_heads=2,
            attention_key_dim=16,
            max_lags=4,
        )
    )
    outputs = model(torch.randn(2, 1, 16, 16, 80), torch.randn(2, 4, 2, 16, 16, 10))
    target = torch.randn(2, 2, 5)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[:, :, -1] = False
    objective = STANetObjective("regression")
    first = objective(outputs, target, mask)
    modified = target.clone()
    modified[:, :, -1] = 1e9
    second = objective(outputs, modified, mask)
    first["total"].backward()
    assert outputs["prediction"].shape == (2, 2, 5)
    assert torch.allclose(first["main"], second["main"])
    assert torch.isfinite(first["total"])


def test_finetune_initialization_loads_weights_without_optimizer_state(tmp_path):
    config = STANetConfig(
        task_type="classification",
        output_dim=2,
        dropout=0.0,
        embedding_dim=32,
        attention_heads=2,
        attention_key_dim=16,
        max_lags=3,
    )
    source = STANet(config)
    checkpoint = tmp_path / "pretrained.pt"
    torch.save({
        "schema": "sta_net_pytorch_training_v2",
        "task": {"key": "motor_imagery"},
        "model_config": {
            "task_type": "classification",
            "output_dim": 2,
            "sequence_length": 1,
            "dropout": 0.0,
            "embedding_dim": 32,
            "attention_heads": 2,
            "attention_key_dim": 16,
            "max_lags": 3,
        },
        "model_state": source.state_dict(),
        "optimizer_state": {"must_not": "load"},
        "epoch": 17,
        "optimizer_step": 123,
    }, checkpoint)
    target = STANet(config)
    metadata = initialize_model_from_checkpoint(
        target,
        checkpoint_path=checkpoint,
        task_key="motor_imagery",
        model_config=config,
        device=torch.device("cpu"),
    )
    assert metadata["source_epoch"] == 17
    assert metadata["source_optimizer_step"] == 123
    assert metadata["optimizer_state_loaded"] is False
    assert all(
        torch.equal(source_value, target.state_dict()[name])
        for name, source_value in source.state_dict().items()
    )


def test_official_wg_adapter_emits_released_sta_net_tensor_shapes():
    eeg_names = list(OFFICIAL_EEG_GRID)
    fnirs_locations = list(OFFICIAL_FNIRS_GRID)
    fnirs_names = [f"{name}_{component}" for name in fnirs_locations for component in ("HbO", "HbR")]
    fnirs_roles = [component for _ in fnirs_locations for component in ("HbO", "HbR")]
    sample = {
        "eeg": np.random.default_rng(0).normal(size=(28, 4000)).astype(np.float32),
        "fnirs": np.random.default_rng(1).normal(size=(72, 200)).astype(np.float32),
        "analysis_valid_mask": {"eeg": np.ones(4000, dtype=bool), "fnirs": np.ones(200, dtype=bool)},
        "bad_channel_mask": {"eeg": np.zeros(28, dtype=bool), "fnirs": np.zeros(72, dtype=bool)},
        "sample_rate_hz": {"eeg": 200.0, "fnirs": 10.0},
        "channel_names": {"eeg": eeg_names, "fnirs": fnirs_names},
        "component_roles": {"eeg": ["electrical_potential"] * 28, "fnirs": fnirs_roles},
        "channel_geometry": {
            "eeg": [{"x": float(x), "y": float(y)} for x, y in OFFICIAL_EEG_GRID.values()],
            "fnirs": [
                {"x": float(OFFICIAL_FNIRS_GRID[name][0]), "y": float(OFFICIAL_FNIRS_GRID[name][1])}
                for name in fnirs_locations
                for _ in ("HbO", "HbR")
            ],
        },
        "label": {"condition": "WG"},
        "subject": "VP001",
        "record_id": "cnt_wg",
        "join_key": "simultaneous_eeg_nirs|VP001|cnt_wg",
        "event": {"event_index": 0},
        "alignment": {"event_relative_window_start_s": 0.0},
    }
    adapted = STANetSampleAdapter(get_sta_net_task_spec("wg")).adapt(sample)
    assert adapted["eeg"].shape == (1, 16, 16, 600)
    assert adapted["fnirs"].shape == (11, 2, 16, 16, 30)
    assert adapted["target"].item() == 0
    assert torch.isfinite(adapted["eeg"]).all()
    assert torch.isfinite(adapted["fnirs"]).all()
    assert adapted["adapter_state"]["eeg_coordinate_mode"] == "official_sta_net_wg_grid"


def test_task_variants_cover_binary_multiclass_and_regression():
    assert get_sta_net_task_spec("wg").output_dim == 2
    assert get_sta_net_task_spec("nback").output_dim == 3
    regression = get_sta_net_task_spec("refed_regression")
    assert regression.output_dim == 2
    assert regression.target_length == 20
    assert regression.fnirs_lag_count == 18


def test_record_grouped_batch_sampler_preserves_record_locality():
    class FakeDataset:
        records = ("a", "a", "b", "a", "b", "c", "c")

        def lightweight_metadata(self, index):
            return {"join_key": self.records[index]}

    dataset = FakeDataset()
    sampler = RecordGroupedBatchSampler(dataset, range(len(dataset.records)), batch_size=2, shuffle=True, seed=7)
    batches = list(sampler)
    assert sorted(index for batch in batches for index in batch) == list(range(len(dataset.records)))
    assert all(len({dataset.records[index] for index in batch}) == 1 for batch in batches)


def test_packed_record_sampler_fills_batches_across_small_records():
    class FakeDataset:
        records = ("a", "a", "b", "b", "c", "c", "d", "d")

        def lightweight_metadata(self, index):
            return {"join_key": self.records[index]}

    sampler = PackedRecordBatchSampler(
        FakeDataset(), range(8), batch_size=5, shuffle=False, seed=7
    )
    batches = list(sampler)
    assert [len(batch) for batch in batches] == [5, 3]
    assert sorted(index for batch in batches for index in batch) == list(range(8))


def test_reproduction_classification_metrics_include_confusion_and_calibration():
    target = np.asarray([0, 1, 2, 1])
    probability = np.asarray([
        [0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7], [0.6, 0.3, 0.1]
    ])
    metrics = classification_metrics(target, probability, ("a", "b", "c"))
    assert metrics["accuracy"] == 0.75
    assert np.asarray(metrics["confusion_matrix"]).shape == (3, 3)
    assert sum(row["count"] for row in metrics["calibration_bins"]) == 4


def test_reproduction_regression_metrics_respect_coordinate_mask():
    target = np.asarray([[[1.0, 2.0]], [[3.0, 99.0]]])
    prediction = np.asarray([[[2.0, 2.0]], [[1.0, -99.0]]])
    mask = np.asarray([[[True, True]], [[True, False]]])
    metrics = regression_metrics(target, prediction, mask, ("valence",))
    assert metrics["valid_coordinate_count"] == 3
    assert np.isclose(metrics["mae_native"], 1.0)


def test_reproduction_regression_plots_emit_vector_and_raster_files(tmp_path):
    target = np.asarray([
        [[1.0, 2.0, 3.0], [0.0, 0.5, 1.0]],
        [[2.0, 2.5, 3.5], [1.0, 0.5, 0.0]],
    ])
    prediction = target + 0.2
    mask = np.ones_like(target, dtype=bool)
    mask[1, :, -1] = False
    paths = plot_regression_diagnostics(
        target,
        prediction,
        mask,
        ("valence", "arousal"),
        tmp_path,
        max_examples=2,
    )
    assert {Path(path).suffix for path in paths} == {".svg", ".png"}
    assert all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in paths)


def test_tuning_budget_keeps_a_100_epoch_rung():
    assert RUNG_EPOCHS == (2, 8, 20, 40, 100)


def test_tuning_objective_uses_best_checkpoint_through_rung(tmp_path):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    rows = [
        {"epoch": 1, "macro_f1": 0.51},
        {"epoch": 2, "macro_f1": 0.63},
        {"epoch": 3, "macro_f1": 0.55},
        {"epoch": 4, "macro_f1": 0.52},
    ]
    (metrics / "validation_epochs.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    score, selected = best_validation_metric_through_epoch(tmp_path, "motor_imagery", 4)
    assert score == 0.63
    assert selected["epoch"] == 2


def test_tuning_objective_minimizes_best_regression_checkpoint(tmp_path):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    rows = [
        {"epoch": 1, "masked_rmse_scaled": 1.2},
        {"epoch": 2, "masked_rmse_scaled": 0.9},
        {"epoch": 3, "masked_rmse_scaled": 1.0},
    ]
    (metrics / "validation_epochs.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    score, selected = best_validation_metric_through_epoch(tmp_path, "refed_regression", 3)
    assert score == -0.9
    assert selected["epoch"] == 2


def test_tuning_lane_plan_preserves_total_trials_and_shards_long_tasks():
    plan = lane_plan(11)
    totals = {}
    workers = {}
    for lane in plan:
        for task in lane["tasks"]:
            totals[task] = totals.get(task, 0) + lane["quota"]
            workers[task] = workers.get(task, 0) + 1
    assert set(totals.values()) == {11}
    assert workers["visual"] == 2
    assert workers["refed_regression"] == 2


def test_targeted_tuning_plan_gives_each_task_one_gpu_and_preserves_quotas():
    plan = targeted_lane_plan(("motor_imagery", "wg"), 16)
    totals = {}
    task_gpus = {}
    for lane in plan:
        for task in lane["tasks"]:
            totals[task] = totals.get(task, 0) + lane["quota"]
            task_gpus.setdefault(task, set()).add(lane["gpu"])
    assert totals == {"motor_imagery": 16, "wg": 16}
    assert task_gpus == {"motor_imagery": {0}, "wg": {1}}
    assert len({lane["worker_id"] for lane in plan}) == len(plan) == 6


def test_final_mi_wg_anchor_resolves_to_preregistered_training_config():
    path = METHOD_ROOT / "configs" / "final_mi_wg_targeted.yaml"
    base = yaml.safe_load(path.read_text(encoding="utf-8"))
    anchor = tuning_anchors(base, "motor_imagery")[2]
    config = sample_config(optuna.trial.FixedTrial(anchor), base, "motor_imagery")
    assert config["tuning"]["search_profile"] == "final_mi_wg"
    assert config["tuning"]["intervention_regime"] == "regularized"
    assert config["model"]["dropout"] == 0.6
    assert config["loss"]["label_smoothing"] == 0.15
    assert config["training"]["weight_decay"] == 0.01
    assert config["task_overrides"]["motor_imagery"]["batch_size"] == 16
    assert "tuning_search" not in config


def test_checkpoint_metric_prefers_macro_f1_instead_of_lower_loss_proxy():
    metrics = core_classification_metrics([0, 0, 1, 1], [0, 0, 0, 1], class_count=2)
    assert np.isclose(metrics["macro_f1"], (0.8 + 2.0 / 3.0) / 2.0)
    assert improved(metrics["macro_f1"], 0.5, "max")
    assert not improved(metrics["macro_f1"], 0.9, "max")


def test_public_split_manifest_never_exposes_protected_indices():
    class Spec:
        key = "fake"

    class FakeDataset:
        spec = Spec()
        rows = [
            {
                "subject": f"s{index // 2}", "record_id": f"r{index // 2}",
                "join_key": f"j{index // 2}", "condition": str(index % 2),
                "class_index": index % 2, "window_offset_s": 0.0,
                "event_index": index, "trial_group": f"g{index}",
            }
            for index in range(20)
        ]

        def __len__(self):
            return len(self.rows)

        def lightweight_metadata(self, index):
            return dict(self.rows[index])

    dataset = FakeDataset()
    train_indices, validation_indices, manifest = development_subject_split(dataset, seed=7)
    assert "test_indices" not in manifest
    assert "reserved_test_indices" not in manifest
    assert manifest["protected_test_opened"] is False
    assert validate_public_manifest(dataset, manifest) == (train_indices, validation_indices)


def test_none_class_weighting_returns_no_tensor():
    class FakeDataset:
        class Spec:
            output_dim = 2
            class_names = ("a", "b")
        spec = Spec()

    assert classification_weights(FakeDataset(), [0, 1], "none") is None


def test_predictive_checkpoint_selection_uses_historical_metric_not_endpoint():
    candidates = [
        {"trial_number": 2, "checkpoint_metric": 0.61, "trial_endpoint_metric": 0.60},
        {"trial_number": 7, "checkpoint_metric": 0.66, "trial_endpoint_metric": 0.62},
    ]
    assert metric_contract("wg") == ("macro_f1", "max")
    assert select_candidate(candidates, "max")["trial_number"] == 7


def test_predictive_checkpoint_selection_minimizes_regression_rmse():
    candidates = [
        {"trial_number": 0, "checkpoint_metric": 0.94},
        {"trial_number": 5, "checkpoint_metric": 0.95},
    ]
    assert metric_contract("refed_regression") == ("masked_rmse_scaled", "min")
    assert select_candidate(candidates, "min")["trial_number"] == 0
