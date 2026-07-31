from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from comparative_methods.BIOT.alignment_data import (
    BIOTPublicView,
    SUPPORTED_TASKS,
    load_config,
)
from comparative_methods.BIOT.audit_alignment_v2 import (
    DEFAULT_CONFIG,
    load_alignment_contract,
    unsupported_refed_cell,
)
from comparative_methods.audit_adapter_alignment import validate_cell
from comparative_methods.BIOT.run_public_development_v2 import (
    DEFAULT_CONFIG as DEFAULT_PUBLIC_CONFIG,
    PublicFold,
    class_weights,
    load_runner_config,
    run,
    select_and_refit,
)
from comparative_methods.BIOT.build_public_job_matrix_v2 import build_matrix
from comparative_methods.BIOT.run_public_matrix_v2 import validate_jobs


METHOD_ROOT = Path(__file__).resolve().parents[1]


def test_alignment_config_freezes_biot_only_and_truthful_refed_disposition() -> None:
    config, path = load_config(DEFAULT_CONFIG)
    assert path == (METHOD_ROOT / "configs/alignment_v2.yaml").resolve()
    assert config["method_id"] == "biot"
    assert config["mode"] == "public_audit_only"
    assert config["protected_test_default"] == "locked"
    assert tuple(task for task in config["tasks"] if config["tasks"][task]["supported"]) == (
        *SUPPORTED_TASKS,
    )
    assert config["tasks"]["refed_regression"]["supported"] is False
    assert config["tasks"]["refed_regression"]["unsupported_reason_code"]


def test_refed_unsupported_cell_satisfies_alignment_schema() -> None:
    config, _ = load_config(DEFAULT_CONFIG)
    contract = load_alignment_contract()
    cell = unsupported_refed_cell(config=config, alignment_contract=contract)
    report = validate_cell(cell, contract, source="synthetic_refed_cell")
    assert report["method_id"] == "biot"
    assert report["task_id"] == "refed_regression"
    assert report["cell_status"] == "unsupported"


class _FakeBase:
    def __init__(self, sample: dict[str, Any]) -> None:
        self.sample = sample
        self._record_cache: dict[str, Any] = {}

    def __getitem__(self, index: int) -> dict[str, Any]:
        assert index == 0
        return self.sample


class _FakeDataset:
    def __init__(self, sample: dict[str, Any]) -> None:
        self.base = _FakeBase(sample)
        self.indices = [0]

    def __len__(self) -> int:
        return 1

    def lightweight_metadata(self, index: int) -> dict[str, Any]:
        assert index == 0
        return {
            "join_key": "record-1",
            "event_index": 7,
            "window_offset_s": 0.0,
        }


@dataclass(frozen=True)
class _FakeInventory:
    dataset: _FakeDataset
    panel: tuple[str, ...]
    duration_s: float = 2.0


def _sample(panel: tuple[str, ...]) -> dict[str, Any]:
    return {
        "join_key": "record-1",
        "eeg": np.ones((16, 400), dtype=np.float32),
        "channel_names": {"eeg": list(panel)},
        "sample_rate_hz": {"eeg": 200.0},
        "valid_mask": {"eeg": np.ones(400, dtype=bool)},
        "analysis_valid_mask": {"eeg": np.ones(400, dtype=bool)},
        "bad_channel_mask": {"eeg": np.zeros(16, dtype=bool)},
        "channel_geometry": {
            "eeg": [
                {"channel_name": name, "position_available": True} for name in panel
            ]
        },
    }


def test_public_view_fails_closed_on_recorded_support_not_analysis_alias() -> None:
    panel = tuple(f"E{index}" for index in range(16))
    sample = _sample(panel)
    sample["valid_mask"]["eeg"][-1] = False
    inventory = _FakeInventory(dataset=_FakeDataset(sample), panel=panel)
    view = BIOTPublicView(inventory, sample_rate_hz=200.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unrecorded/padded support"):
        view[0]


def test_public_view_rejects_bad_measured_channel_without_copy_or_padding() -> None:
    panel = tuple(f"E{index}" for index in range(16))
    sample = _sample(panel)
    sample["bad_channel_mask"]["eeg"][3] = True
    inventory = _FakeInventory(dataset=_FakeDataset(sample), panel=panel)
    view = BIOTPublicView(inventory, sample_rate_hz=200.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bad measured channels"):
        view[0]


def test_public_runner_config_freezes_one_biot_matrix_and_reviewed_alignment() -> None:
    config, _config_path, alignment, _alignment_path = load_runner_config(
        DEFAULT_PUBLIC_CONFIG
    )
    assert config["method_id"] == "biot"
    assert config["job_matrix"]["expected_public_jobs"] == 90
    assert tuple(config["job_matrix"]["tasks"]) == SUPPORTED_TASKS
    assert alignment["method_id"] == "biot"
    assert config["protected_test_default"] == "locked"


def test_inverse_frequency_weights_reject_empty_training_class() -> None:
    np.testing.assert_allclose(class_weights(np.asarray([0, 0, 1]), 2), [0.75, 1.5])
    with pytest.raises(RuntimeError, match="empty class"):
        class_weights(np.asarray([0, 0]), 2)


def test_public_probe_selects_then_refits_and_weights_only_reloads(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    features = np.concatenate(
        (
            rng.normal(-1.0, 0.1, (12, 256)),
            rng.normal(1.0, 0.1, (12, 256)),
        )
    ).astype(np.float32)
    arrays = {
        "features": features,
        "targets": np.asarray([0] * 12 + [1] * 12, dtype=np.int64),
        "dataset_indices": np.arange(24, dtype=np.int64),
        "subjects": np.asarray([f"s{index // 4}" for index in range(24)]),
        "sample_ids": np.asarray([f"sample-{index}" for index in range(24)]),
    }
    inventory = SimpleNamespace(
        task="motor_imagery",
        dataset=SimpleNamespace(spec=SimpleNamespace(class_names=("left", "right"))),
    )
    fold = PublicFold(
        inventory=inventory,  # type: ignore[arg-type]
        outer_fold=0,
        public_manifest_path=tmp_path / "public.json",
        public_manifest_sha256="a" * 64,
        train_indices=tuple([*range(8), *range(12, 20)]),
        validation_indices=tuple([*range(8, 12), *range(20, 24)]),
    )
    config, config_path, _alignment, alignment_path = load_runner_config(
        DEFAULT_PUBLIC_CONFIG
    )
    report, logits = select_and_refit(
        arrays=arrays,
        fold=fold,
        train_indices=fold.train_indices,
        validation_indices=fold.validation_indices,
        config=config,
        config_path=config_path,
        alignment_path=alignment_path,
        method_identity={"artifact_id": "synthetic"},
        cache_identity={"feature_cache_key": "synthetic"},
        seed=17,
        device=torch.device("cpu"),
        smoke=True,
        output_dir=tmp_path,
    )
    assert logits.shape == (8, 2)
    assert report["public_refit"]["membership"] == "smoke_train_plus_validation_subset"
    assert report["public_refit"]["weights_only_reload_match"] is True
    assert (tmp_path / "checkpoint_public_refit.pt").is_file()


def test_public_runner_refuses_output_outside_biot_run_root(tmp_path: Path) -> None:
    args = SimpleNamespace(
        config=DEFAULT_PUBLIC_CONFIG,
        task="motor_imagery",
        outer_fold=0,
        seed=17,
        output_dir=tmp_path / "outside",
        smoke=True,
    )
    with pytest.raises(PermissionError, match="must remain under"):
        run(args)


def test_biot_job_matrix_is_serial_public_only_and_not_self_authorizing() -> None:
    matrix = build_matrix()
    assert matrix["job_count"] == 90
    assert matrix["max_concurrent_jobs"] == 1
    assert matrix["automatic_retry_count"] == 0
    assert matrix["public_matrix_launch_authorized"] is False
    assert matrix["protected_evaluation_authorized"] is False
    assert matrix["protected_test_opened"] is False
    assert all(job["initial_status"] == "queued_not_authorized" for job in matrix["jobs"])
    assert all("protected" not in " ".join(job["command"]).lower() for job in matrix["jobs"])
    jobs = validate_jobs(
        matrix,
        run_root=(
            METHOD_ROOT / "runs/public_development_v2/matrix_v2"
        ).resolve(),
    )
    assert [job["order"] for job in jobs] == list(range(90))
