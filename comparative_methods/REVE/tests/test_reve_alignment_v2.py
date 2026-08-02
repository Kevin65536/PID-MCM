from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from comparative_methods.BIOT.alignment_data import load_config as load_biot_config
from comparative_methods.BIOT.audit_alignment_v2 import comparison_fields as biot_fields
from comparative_methods.REVE.alignment_data import (
    SUPPORTED_TASKS,
    REVEPublicView,
    load_config,
)
from comparative_methods.REVE.audit_alignment_v2 import (
    DEFAULT_CONFIG,
    comparison_fields,
    load_alignment_contract,
    parse_tasks,
    unsupported_refed_cell,
    write_json,
)
from comparative_methods.REVE.build_public_job_matrix_v2 import build_matrix
from comparative_methods.REVE.run_public_development_v2 import (
    DEFAULT_CONFIG as DEFAULT_PUBLIC_CONFIG,
    PublicFold,
    class_weights,
    load_runner_config,
    run,
    select_and_refit,
)
from comparative_methods.REVE.run_public_matrix_v2 import execute, validate_jobs
from comparative_methods.audit_adapter_alignment import validate_cell


METHOD_ROOT = Path(__file__).resolve().parents[1]
BIOT_CONFIG = METHOD_ROOT.parent / "BIOT/configs/alignment_v2.yaml"


def test_alignment_config_is_public_only_support_matched_and_position_covered() -> None:
    config, path = load_config(DEFAULT_CONFIG)
    biot, _ = load_biot_config(BIOT_CONFIG)
    position_config = json.loads(
        (METHOD_ROOT / "checkpoints/reve-positions/config.json").read_text(encoding="utf-8")
    )
    official_names = set(position_config["position_names"])
    assert path == (METHOD_ROOT / "configs/alignment_v2.yaml").resolve()
    assert config["method_id"] == "reve"
    assert config["mode"] == "public_audit_only"
    assert config["protected_test_default"] == "locked"
    assert config["adapter"]["pooling"] == "frozen_pretrained_cls_query_attention_pooling"
    assert config["adapter"]["deterministic_source_declared_sample_transform"] == (
        "none_after_canonical_200hz_coordinate"
    )
    for task in SUPPORTED_TASKS:
        assert config["tasks"][task]["panel"] == biot["tasks"][task]["panel"]
        assert config["tasks"][task]["duration_s"] == biot["tasks"][task]["duration_s"]
        assert set(config["tasks"][task]["panel"]) <= official_names
    assert config["tasks"]["refed_regression"]["supported"] is False


def test_single_trial_tasks_are_confined_to_known_pretraining_overlap_track() -> None:
    config, _ = load_config(DEFAULT_CONFIG)
    for task in ("motor_imagery", "mental_arithmetic"):
        assert config["tasks"][task]["track"] == (
            "open_world_pretrained_with_target_corpus_overlap"
        )
    for task in ("wg", "nback", "dsr", "visual"):
        assert config["tasks"][task]["track"] == (
            "single_modal_eeg_official_pretrained_linear_probe"
        )


def test_method_neutral_comparison_fields_match_biot_exactly() -> None:
    inventory = SimpleNamespace(
        sample_inventory_sha256="a" * 64,
        split_fingerprint="b" * 64,
        panel=("F3", "F4"),
        indices=(1, 2, 3),
        duration_s=8.0,
    )
    contract = load_alignment_contract()
    fingerprints = {"branch": "c" * 64}
    actual = comparison_fields(
        task="motor_imagery",
        inventory=inventory,
        alignment_contract=contract,
        branch_fingerprints=fingerprints,
    )
    expected = biot_fields(
        task="motor_imagery",
        inventory=inventory,
        alignment_contract=contract,
        branch_fingerprints=fingerprints,
    )
    assert actual == expected


def test_refed_unsupported_cell_satisfies_alignment_schema() -> None:
    config, _ = load_config(DEFAULT_CONFIG)
    contract = load_alignment_contract()
    cell = unsupported_refed_cell(config=config, alignment_contract=contract)
    report = validate_cell(cell, contract, source="synthetic_refed_cell")
    assert report["method_id"] == "reve"
    assert report["task_id"] == "refed_regression"
    assert report["cell_status"] == "unsupported"
    assert cell["unsupported_reason_code"] == "REVE_NO_PARTIAL_TIME_MASK_CONTRACT"


def test_evidence_writer_refuses_protected_path(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="protected evidence path"):
        write_json(tmp_path / "protected" / "cell.json", {"status": "forbidden"})


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
        return {"join_key": "record-1", "event_index": 7, "window_offset_s": 0.0}


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


def test_public_view_preserves_canonical_amplitude_without_task_scale_factor() -> None:
    panel = tuple(f"E{index}" for index in range(16))
    inventory = _FakeInventory(dataset=_FakeDataset(_sample(panel)), panel=panel)
    view = REVEPublicView(inventory, sample_rate_hz=200.0)  # type: ignore[arg-type]
    item = view[0]
    np.testing.assert_array_equal(item["eeg"].numpy(), np.ones((16, 400), dtype=np.float32))
    assert int(item["recorded_support_count"]) == 400


def test_public_view_rejects_padding_and_bad_measured_channels() -> None:
    panel = tuple(f"E{index}" for index in range(16))
    padded = _sample(panel)
    padded["valid_mask"]["eeg"][-1] = False
    inventory = _FakeInventory(dataset=_FakeDataset(padded), panel=panel)
    with pytest.raises(ValueError, match="unrecorded/padded support"):
        REVEPublicView(inventory, sample_rate_hz=200.0)[0]  # type: ignore[arg-type]

    bad = _sample(panel)
    bad["bad_channel_mask"]["eeg"][3] = True
    inventory = _FakeInventory(dataset=_FakeDataset(bad), panel=panel)
    with pytest.raises(ValueError, match="bad measured channels"):
        REVEPublicView(inventory, sample_rate_hz=200.0)[0]  # type: ignore[arg-type]


def test_task_parser_is_serial_scope_safe() -> None:
    assert parse_tasks([]) == SUPPORTED_TASKS
    assert parse_tasks(["motor_imagery"]) == ("motor_imagery",)
    with pytest.raises(ValueError, match="unknown or unsupported"):
        parse_tasks(["refed_regression"])
    with pytest.raises(ValueError, match="must be unique"):
        parse_tasks(["wg", "wg"])


def test_retained_full_public_evidence_is_terminal_through_a7() -> None:
    root = METHOD_ROOT / "evidence/alignment_v2"
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "implementation_review_complete_A0_A7_pass_A8_pending"
    assert summary["protected_test_opened"] is False
    assert len(summary["schema_audit"]["direct_group_reports"]) == 7
    assert summary["schema_audit"]["status"] == "pass"

    total = 0
    for task in SUPPORTED_TASKS:
        cell = json.loads((root / f"{task}.json").read_text(encoding="utf-8"))
        assert cell["evidence_scope"] == "public_complete"
        assert cell["cell_status"] == "pending"
        assert [cell["gate_status"][f"A{index}"] for index in range(8)] == ["pass"] * 8
        assert cell["gate_status"]["A8"] == "pending"
        assert cell["public_audit"]["all_unique_public_samples_audited"] is True
        assert cell["public_audit"]["deterministic_replay_exact"] is True
        assert cell["public_audit"]["feature_shape"][1] == 512
        assert cell["public_audit"]["nonconstant_coordinate_count"] == 512
        assert cell["public_audit"]["protected_test_opened"] is False
        total += int(cell["public_audit"]["unique_sample_count"])
    assert total == 22_442

    refed = json.loads((root / "refed_regression.json").read_text(encoding="utf-8"))
    assert refed["cell_status"] == "unsupported"
    assert refed["unsupported_reason_code"] == "REVE_NO_PARTIAL_TIME_MASK_CONTRACT"
    assert refed["protected_test_opened"] is False


def test_public_runner_config_freezes_one_reve_matrix_and_track_map() -> None:
    config, _config_path, alignment, _alignment_path = load_runner_config(
        DEFAULT_PUBLIC_CONFIG
    )
    assert config["method_id"] == "reve"
    assert config["job_matrix"]["expected_public_jobs"] == 90
    assert tuple(config["job_matrix"]["tasks"]) == SUPPORTED_TASKS
    assert alignment["method_id"] == "reve"
    assert config["protected_test_default"] == "locked"
    assert alignment["tasks"]["motor_imagery"]["track"] == (
        "open_world_pretrained_with_target_corpus_overlap"
    )


def test_inverse_frequency_weights_reject_empty_training_class() -> None:
    np.testing.assert_allclose(class_weights(np.asarray([0, 0, 1]), 2), [0.75, 1.5])
    with pytest.raises(RuntimeError, match="empty class"):
        class_weights(np.asarray([0, 0]), 2)


def test_public_probe_selects_refits_512d_and_weights_only_reloads(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    features = np.concatenate(
        (rng.normal(-1.0, 0.1, (12, 512)), rng.normal(1.0, 0.1, (12, 512)))
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
    checkpoint = torch.load(
        tmp_path / "checkpoint_public_refit.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint["head_state"]["weight"].shape == (2, 512)
    assert checkpoint["method_id"] == "reve"


def test_public_runner_refuses_output_outside_reve_run_root(tmp_path: Path) -> None:
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


def test_pilot_audit_passes_without_performance_or_protected_claim() -> None:
    pilot = json.loads(
        (
            METHOD_ROOT / "evidence/public_development_v2/pilot_audit.json"
        ).read_text(encoding="utf-8")
    )
    assert pilot["status"] == "pass"
    assert pilot["protected_test_opened"] is False
    assert len(pilot["run_reports"]) == 1
    report = pilot["run_reports"][0]
    assert report["mode"] == "smoke_only"
    assert report["status"] == "pass"
    assert report["table_admissible"] is False
    assert report["track"] == "open_world_pretrained_with_target_corpus_overlap"


def test_candidate_job_matrix_is_serial_public_only_and_not_self_authorizing() -> None:
    matrix = build_matrix()
    assert matrix["job_count"] == 90
    assert matrix["max_concurrent_jobs"] == 1
    assert matrix["automatic_retry_count"] == 0
    assert matrix["public_matrix_launch_authorized"] is False
    assert matrix["protected_evaluation_authorized"] is False
    assert matrix["protected_test_opened"] is False
    assert all(job["initial_status"] == "queued_not_authorized" for job in matrix["jobs"])
    assert all("protected" not in " ".join(job["command"]).lower() for job in matrix["jobs"])
    assert [job["order"] for job in matrix["jobs"]] == list(range(90))
    jobs = validate_jobs(
        matrix,
        run_root=(METHOD_ROOT / "runs/public_development_v2/matrix_v2").resolve(),
    )
    assert [job["order"] for job in jobs] == list(range(90))


def test_reviewed_launch_authorizes_only_serial_public_matrix() -> None:
    report = execute(METHOD_ROOT / "configs/public_matrix_launch_v2.yaml", dry_run=True)
    assert report["status"] == "pass"
    assert report["job_count"] == 90
    assert report["max_concurrent_jobs"] == 1
    assert report["automatic_retry_count"] == 0
    assert report["public_matrix_launch_authorized"] is True
    assert report["protected_evaluation_authorized"] is False
    assert report["protected_test_opened"] is False
