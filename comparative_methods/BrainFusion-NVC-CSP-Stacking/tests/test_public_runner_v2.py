from __future__ import annotations

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from run_public_development_v2 import (
    DEFAULT_CONFIG,
    diverse_balanced_subset,
    load_runner_config,
    run,
)


def test_runner_config_freezes_75_serial_public_jobs() -> None:
    config, _config_path, alignment, _alignment_path = load_runner_config(DEFAULT_CONFIG)
    assert config["job_matrix"]["expected_public_jobs"] == 75
    assert config["job_matrix"]["max_concurrent_jobs"] == 1
    assert config["job_matrix"]["automatic_retry_count"] == 0
    assert config["reporting"]["protected_evaluation_authorized"] is False
    assert alignment["tasks"]["dsr"]["supported"] is False


class _Dataset:
    class_to_index = {"a": 0, "b": 1}

    def lightweight_metadata(self, index: int) -> dict[str, str]:
        return {
            "condition": "a" if index < 12 else "b",
            "subject": f"s{index % 6}",
        }


def test_smoke_subset_is_balanced_and_subject_diverse() -> None:
    inventory = SimpleNamespace(dataset=_Dataset())
    selected = diverse_balanced_subset(inventory, range(24), per_class=6, seed=17)
    labels = [inventory.dataset.class_to_index[inventory.dataset.lightweight_metadata(i)["condition"]] for i in selected]
    groups = [inventory.dataset.lightweight_metadata(i)["subject"] for i in selected]
    assert labels.count(0) == labels.count(1) == 6
    assert len(set(groups)) == 6


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
