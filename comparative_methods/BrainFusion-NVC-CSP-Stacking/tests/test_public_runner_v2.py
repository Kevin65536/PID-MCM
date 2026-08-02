from __future__ import annotations

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest
import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from run_public_development_v2 import (
    DEFAULT_CONFIG,
    diverse_balanced_subset,
    fold_data_from_cache,
    load_runner_config,
    run,
    validate_tensor_cache,
)


def test_runner_config_freezes_75_serial_public_jobs() -> None:
    config, _config_path, alignment, _alignment_path = load_runner_config(DEFAULT_CONFIG)
    assert config["job_matrix"]["expected_public_jobs"] == 75
    assert config["job_matrix"]["max_concurrent_jobs"] == 1
    assert config["job_matrix"]["automatic_retry_count"] == 0
    assert config["resources"]["tensor_cache_root"].endswith("runs/tensor_cache_v2")
    assert config["reporting"]["protected_evaluation_authorized"] is False
    assert alignment["tasks"]["dsr"]["supported"] is False


class _Dataset:
    class_to_index = {"a": 0, "b": 1}

    def lightweight_metadata(self, index: int) -> dict[str, str]:
        return {
            "condition": "a" if index < 12 else "b",
            "subject": f"s{index % 6}",
            "join_key": f"j{index}",
            "event_index": index,
            "window_offset_s": 0.0,
        }


def test_smoke_subset_is_balanced_and_subject_diverse() -> None:
    inventory = SimpleNamespace(dataset=_Dataset())
    selected = diverse_balanced_subset(inventory, range(24), per_class=6, seed=17)
    labels = [inventory.dataset.class_to_index[inventory.dataset.lightweight_metadata(i)["condition"]] for i in selected]
    groups = [inventory.dataset.lightweight_metadata(i)["subject"] for i in selected]
    assert labels.count(0) == labels.count(1) == 6
    assert len(set(groups)) == 6


def test_public_tensor_cache_is_exact_and_fold_addressable() -> None:
    inventory = SimpleNamespace(
        indices=(2, 14),
        duration_s=0.2,
        eeg_channels=("e1",),
        fnirs_locations=("f1",),
        dataset=_Dataset(),
    )
    payload = {
        "eeg": torch.arange(80, dtype=torch.float32).reshape(2, 1, 40),
        "hbo": torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]),
        "hbr": torch.tensor([[[4.0, 3.0]], [[2.0, 1.0]]]),
        "targets": torch.tensor([0, 1], dtype=torch.long),
        "dataset_indices": torch.tensor([2, 14], dtype=torch.long),
    }
    validate_tensor_cache(payload, inventory)
    fold = fold_data_from_cache(payload, inventory, [14, 2])
    assert fold[3].tolist() == [1, 0]
    assert fold[4] == ["j14|event=14|offset_ms=0", "j2|event=2|offset_ms=0"]
    assert fold[5] == ["s2", "s2"]


def test_runner_refuses_output_outside_method_run_root(tmp_path: Path) -> None:
    args = SimpleNamespace(
        config=DEFAULT_CONFIG,
        task="motor_imagery",
        outer_fold=0,
        seed=17,
        device="cpu",
        output_dir=tmp_path / "outside",
        smoke=True,
    )
    with pytest.raises(PermissionError, match="must remain under"):
        run(args)


def test_retained_public_pilot_passes_without_performance_claim() -> None:
    pilot = json.loads(
        (METHOD_ROOT / "evidence/public_development_v2/pilot_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert pilot["status"] == "pass"
    assert pilot["mode"] == "smoke_only"
    assert pilot["membership_recomputed"] is True
    assert pilot["targets_recomputed"] is True
    assert pilot["metric_recomputed"] is True
    assert pilot["checkpoint_predictions_recomputed"] is True
    assert not Path(pilot["run_report_path"]).is_absolute()
    assert pilot["table_admissible"] is False
    assert pilot["protected_test_opened"] is False


def test_retained_full_fold_pilot_replays_cache_and_checkpoint() -> None:
    pilot = json.loads(
        (METHOD_ROOT / "evidence/public_development_v2/full_fold_pilot_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert pilot["status"] == "pass"
    assert pilot["mode"] == "public_development"
    assert pilot["train_sample_count"] == 900
    assert pilot["validation_sample_count"] == 480
    assert pilot["tensor_cache_audited"] is True
    assert pilot["cached_validation_matches_raw_adapter"] is True
    assert pilot["checkpoint_predictions_recomputed"] is True
    assert pilot["checkpoint_reload_exact"] is True
    assert pilot["table_admissible"] is False
    assert pilot["protected_test_opened"] is False
