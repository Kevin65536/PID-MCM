from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from comparative_methods.single_modal_eeg.contract import (
    EEGTaskView,
    load_config,
    load_public_contract,
)
from comparative_methods.single_modal_eeg.build_public_job_matrix import build_matrix
from comparative_methods.single_modal_eeg.run_public_performance import fit_probe


CONFIG = "comparative_methods/single_modal_eeg/configs/public_performance_v1.yaml"


class _Base:
    def __init__(self, sample: dict) -> None:
        self.sample = sample
        self._record_cache = {"record": {"large": "placeholder"}}

    def __getitem__(self, _index: int) -> dict:
        return self.sample


class _TaskDataset:
    class_to_index = {"left": 0, "right": 1}
    indices = [0]

    def __init__(self, sample: dict) -> None:
        self.base = _Base(sample)

    def __len__(self) -> int:
        return 1

    def lightweight_metadata(self, _index: int) -> dict:
        return {
            "subject": "subject_01",
            "event_index": 3,
            "window_offset_s": 0.0,
            "condition": "left",
            "join_key": "record",
        }


def _sample(*, full_mask: bool = True) -> dict:
    names = ["C3", "C4"]
    return {
        "join_key": "record",
        "eeg": np.ones((2, 400), dtype=np.float32),
        "analysis_valid_mask": {
            "eeg": np.full(400, full_mask, dtype=bool),
        },
        "bad_channel_mask": {"eeg": np.zeros(2, dtype=bool)},
        "sample_rate_hz": {"eeg": 200.0},
        "channel_names": {"eeg": names},
        "channel_geometry": {
            "eeg": [
                {"channel_name": name, "position_available": True} for name in names
            ]
        },
        "label": {"condition": "left"},
    }


def _view(*, full_mask: bool = True) -> EEGTaskView:
    dataset = _TaskDataset(_sample(full_mask=full_mask))
    contract = SimpleNamespace(
        dataset=dataset,
        panel=("C3", "C4"),
        duration_s=2.0,
        config={"data": {"eeg_sample_rate_hz": 200.0}},
    )
    return EEGTaskView(contract)


def test_frozen_task_view_selects_real_channels_without_padding() -> None:
    item = _view()[0]
    assert tuple(item["eeg"].shape) == (2, 400)
    assert item["target"].item() == 0
    assert item["sample_id"] == "record|event=3|offset_ms=0"


def test_frozen_task_view_rejects_partial_time_support() -> None:
    with pytest.raises(ValueError, match="invalid or padded time support"):
        _view(full_mask=False)[0]


def test_config_freezes_prest16_and_refed_unsupported() -> None:
    config, _path = load_config(CONFIG)
    assert config["methods"]["biot"]["artifact_id"] == "eeg_prest_16"
    assert config["data"]["panel_size"] == 16
    assert config["tasks"]["refed_regression"]["supported"] is False
    with pytest.raises(RuntimeError, match="frozen unsupported"):
        load_public_contract(CONFIG, task="refed_regression", outer_fold=0)


def test_linear_probe_writes_weights_only_reloadable_checkpoint(tmp_path) -> None:
    rng = np.random.default_rng(17)
    train = np.concatenate(
        (rng.normal(-1.0, 0.1, (8, 4)), rng.normal(1.0, 0.1, (8, 4)))
    )
    validation = np.concatenate(
        (rng.normal(-1.0, 0.1, (4, 4)), rng.normal(1.0, 0.1, (4, 4)))
    )
    arrays = {
        "features": np.concatenate((train, validation)).astype(np.float32),
        "targets": np.asarray([0] * 8 + [1] * 8 + [0] * 4 + [1] * 4),
        "partitions": np.asarray(["train"] * 16 + ["validation"] * 8),
    }
    contract = SimpleNamespace(
        task="motor_imagery",
        outer_fold=0,
        class_names=("left", "right"),
        config={
            "probe": {
                "feature_standardization_epsilon": 1.0e-6,
                "batch_size": 8,
            },
            "smoke": {
                "epochs": 2,
                "learning_rates": [0.01],
                "weight_decays": [0.0],
            },
        },
    )
    report, logits = fit_probe(
        arrays=arrays,
        contract=contract,
        method="biot",
        seed=17,
        device="cpu",
        smoke=True,
        output_dir=tmp_path,
    )
    assert logits.shape == (8, 2)
    assert report["weights_only_reload_match"] is True
    assert (tmp_path / "checkpoint_best.pt").is_file()


def test_public_job_matrix_has_all_supported_cells_and_no_protected_command(tmp_path) -> None:
    matrix = build_matrix(
        config_path=CONFIG,
        output_root=tmp_path / "public_runs",
        device="cuda:1",
    )
    assert matrix["job_count"] == 3 * 6 * 5 * 3
    assert matrix["unsupported_tasks"] == {
        "refed_regression": "partial_terminal_windows_require_time_mask_support_not_available_in_v1_frozen_encoders"
    }
    assert matrix["protected_test_opened"] is False
    assert all("protected" not in " ".join(job["command"]).lower() for job in matrix["jobs"])
    with pytest.raises(ValueError, match="freezes device=cuda:1"):
        build_matrix(
            config_path=CONFIG,
            output_root=tmp_path / "wrong_gpu",
            device="cuda:0",
        )
