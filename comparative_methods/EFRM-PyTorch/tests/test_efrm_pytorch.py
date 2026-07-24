from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pytest
import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from efrm_pytorch.data import (
    EFRMPairedWindowAdapter,
    EFRMSyncPretrainDataset,
    InventoryDiverseBatchSampler,
    collate_efrm_pairs,
)
from efrm_pytorch.model import EFRMDownstreamModel, EFRMSyncModel
from efrm_pytorch.pretraining_analysis import analyze_pretraining_run
from efrm_pytorch.protocol import (
    PublicSplitSubjects,
    TrialMixedBoundary,
    load_public_split_subjects,
)
from efrm_pytorch.tasks import TASK_SPECS
from efrm_pytorch.training import cached_pretrain_backward
from efrm_pytorch.visualization import (
    PHYSIOLOGY_EVIDENCE_SCHEMA,
    export_alignment_evidence,
    render_alignment_report,
    retrieval_metrics,
)
from launch_pretrain_detached import launch_detached
from train_pretrain import _archive_incomplete_resume_steps


def _small_model() -> EFRMSyncModel:
    return EFRMSyncModel(
        eeg_patch_samples=10,
        fnirs_patch_samples=5,
        embed_dim=32,
        depth=1,
        num_heads=4,
        decoder_embed_dim=24,
        decoder_depth=1,
        decoder_num_heads=4,
        mlp_ratio=2.0,
    )


def _batch(batch: int = 3) -> dict[str, torch.Tensor]:
    return {
        "eeg": torch.randn(batch, 1, 5, 40),
        "fnirs": torch.randn(batch, 2, 7, 20),
        "eeg_patch_valid": torch.ones(batch, 5, 4, dtype=torch.bool),
        "fnirs_patch_valid": torch.ones(batch, 7, 4, dtype=torch.bool),
    }


def test_variable_channel_model_forward_backward_and_positive_matrix() -> None:
    model = _small_model()
    batch = _batch()
    output = model(**batch)
    assert output["cosine_similarity"].shape == (3, 3)
    assert output["eeg_reconstruction_mask"].shape == (3, 20)
    assert output["fnirs_reconstruction_mask"].shape == (3, 28)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_cached_clip_gradient_matches_full_batch_gradient() -> None:
    torch.manual_seed(9)
    reference = _small_model()
    cached = _small_model()
    cached.load_state_dict(reference.state_dict())
    batch = _batch(batch=4)

    eeg_embedding, fnirs_embedding = reference.encode(**batch)
    reference.alignment(eeg_embedding, fnirs_embedding)["loss"].backward()
    reference_gradient = reference.eeg_model.patch_embed.proj.weight.grad.clone()

    cached_pretrain_backward(
        cached,
        batch,
        chunk_size=2,
        amp_dtype=None,
        eeg_reconstruction_weight=0.0,
        fnirs_reconstruction_weight=0.0,
        clip_alignment_weight=1.0,
    )
    cached_gradient = cached.eeg_model.patch_embed.proj.weight.grad
    torch.testing.assert_close(cached_gradient, reference_gradient, rtol=2e-4, atol=2e-6)


def test_patch_masks_exclude_invalid_tail() -> None:
    model = _small_model()
    batch = _batch()
    batch["eeg_patch_valid"][:, :, -1] = False
    batch["fnirs_patch_valid"][:, :, -1] = False
    output = model(**batch)
    assert torch.isfinite(output["loss"])
    assert not torch.any(output["eeg_reconstruction_mask"][:, 3::4].bool())
    assert not torch.any(output["fnirs_reconstruction_mask"][:, 3::4].bool())


def test_embedding_removes_invalid_tokens_before_attention() -> None:
    model = _small_model().eval()
    values = torch.randn(1, 1, 5, 40)
    valid = torch.ones(1, 5, 4, dtype=torch.bool)
    valid[:, :, -1] = False
    baseline = model.eeg_model.forward_embed(values, valid)
    changed_only_in_invalid_patch = values.clone()
    changed_only_in_invalid_patch[..., -10:] = 1_000_000.0
    comparison = model.eeg_model.forward_embed(changed_only_in_invalid_patch, valid)
    torch.testing.assert_close(baseline, comparison, rtol=1e-5, atol=1e-6)


def test_downstream_classification_and_regression_shapes() -> None:
    batch = _batch()
    classifier = EFRMDownstreamModel(_small_model(), output_dim=4, modality="paired")
    assert classifier(**batch).shape == (3, 4)
    regressor = EFRMDownstreamModel(
        _small_model(), output_dim=2, target_length=20, modality="paired"
    )
    assert regressor(**batch).shape == (3, 2, 20)


def _fake_unified_sample() -> dict:
    return {
        "eeg": np.arange(4 * 400, dtype=np.float32).reshape(4, 400),
        "fnirs": np.arange(6 * 20, dtype=np.float32).reshape(6, 20),
        "analysis_valid_mask": {
            "eeg": np.ones(400, dtype=bool),
            "fnirs": np.ones(20, dtype=bool),
        },
        "bad_channel_mask": {
            "eeg": np.asarray([False, True, False, False]),
            "fnirs": np.asarray([False, False, False, False, True, False]),
        },
        "sample_rate_hz": {"eeg": 200.0, "fnirs": 10.0},
        "channel_names": {
            "eeg": ["E1", "E2", "E3", "E4"],
            "fnirs": ["A_HbO", "A_HbR", "B_HbO", "B_HbR", "C_HbO", "C_HbR"],
        },
        "component_roles": {
            "eeg": ["electrical_potential"] * 4,
            "fnirs": ["HbO", "HbR", "HbO", "HbR", "HbO", "HbR"],
        },
        "dataset_id": "fake",
        "subject": "S1",
        "record_id": "R1",
        "join_key": "fake|S1|R1",
        "event": {"event_index": 2},
        "label": {"namespace": "fake:task", "condition": "A"},
    }


def test_adapter_keeps_measured_channels_and_pairs_components_without_duplication() -> None:
    adapter = EFRMPairedWindowAdapter(
        duration_s=2.0,
        eeg_patch_samples=50,
        fnirs_patch_samples=20,
    )
    result = adapter.adapt(_fake_unified_sample())
    assert result["eeg"].shape == (1, 4, 400)
    assert result["fnirs"].shape == (2, 3, 20)
    assert result["eeg_channel_names"] == ["E1", "E2", "E3", "E4"]
    assert result["fnirs_location_names"] == ["A", "B", "C"]
    assert not result["eeg_patch_valid"][1].any()
    assert not result["fnirs_patch_valid"][2].any()
    assert result["admitted"] is True


def test_collate_requires_one_record_inventory() -> None:
    adapter = EFRMPairedWindowAdapter(duration_s=2.0)
    first = adapter.adapt(_fake_unified_sample())
    second = adapter.adapt(_fake_unified_sample())
    batch = collate_efrm_pairs([first, second])
    assert batch["positive_pair_mask"].equal(torch.eye(2, dtype=torch.bool))
    second["eeg"] = second["eeg"][:, :-1]
    with pytest.raises(ValueError, match="one channel inventory"):
        collate_efrm_pairs([first, second])


def test_contrastive_sampler_draws_distinct_records_per_pass() -> None:
    class FakeDataset:
        def __init__(self) -> None:
            self.rows = [
                {"dataset_id": "d", "join_key": f"r{record}", "subject": f"s{record}"}
                for _window in range(3) for record in range(3)
            ]
            self.epoch = 0

        def __len__(self) -> int:
            return len(self.rows)

        def lightweight_metadata(self, index: int) -> dict:
            return self.rows[index]

        def __getitem__(self, index: int) -> dict:
            return {
                "eeg_channel_names": ["C3", "C4"],
                "fnirs_location_names": ["S1_D1", "S2_D2"],
            }

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

    dataset = FakeDataset()
    sampler = InventoryDiverseBatchSampler(dataset, batch_size=3, seed=4)
    batches = list(sampler)
    assert len(batches) == 3
    for batch in batches:
        records = [dataset.lightweight_metadata(index)["join_key"] for index in batch]
        assert len(records) == len(set(records)) == 3
    assert sampler.manifest()["negative_sampling"].startswith("record_diverse")


def test_pretraining_crop_uses_common_fully_valid_support() -> None:
    dataset = object.__new__(EFRMSyncPretrainDataset)
    dataset.seed = 42
    dataset.epoch = 0
    dataset.adapter = EFRMPairedWindowAdapter(duration_s=2.0)
    sample = _fake_unified_sample()
    sample["analysis_valid_mask"]["eeg"][:200] = False
    sample["analysis_valid_mask"]["fnirs"][:10] = False
    sample["eeg"] = np.zeros((4, 800), dtype=np.float32)
    sample["fnirs"] = np.zeros((6, 40), dtype=np.float32)
    sample["analysis_valid_mask"]["eeg"] = np.concatenate((
        np.zeros(200, dtype=bool), np.ones(600, dtype=bool)
    ))
    sample["analysis_valid_mask"]["fnirs"] = np.concatenate((
        np.zeros(10, dtype=bool), np.ones(30, dtype=bool)
    ))
    start = dataset._crop_start(sample, 0)
    assert start >= 1.0
    assert dataset.adapter.adapt(sample, crop_start_s=start)["admitted"] is True


def test_all_seven_task_contracts_are_present() -> None:
    assert set(TASK_SPECS) == {
        "motor_imagery", "mental_arithmetic", "wg", "nback", "dsr", "visual",
        "refed_regression",
    }
    assert TASK_SPECS["refed_regression"].scientific_scope == "efrm_sync_regression_adapter"
    assert TASK_SPECS["dsr"].input_duration_s == 2.0


def test_alignment_metrics_and_visual_report(tmp_path: Path) -> None:
    eeg = np.eye(4, dtype=np.float32)
    fnirs = np.eye(4, dtype=np.float32)
    metadata = [
        {
            "sample_id": f"pair-{index}", "dataset_id": "d", "subject": f"s{index}",
            "record_id": "r", "join_key": "d|r", "condition": "task",
        }
        for index in range(4)
    ]
    evidence = export_alignment_evidence(
        tmp_path / "evidence", eeg_embeddings=eeg, fnirs_embeddings=fnirs, metadata=metadata,
        filename="full_validation_clip_alignment_evidence.npz",
    )
    assert evidence.name == "full_validation_clip_alignment_evidence.npz"
    metrics = retrieval_metrics(np.eye(4))
    assert metrics["eeg_to_fnirs"]["top1"] == 1.0
    physiology = tmp_path / "physiology.npz"
    np.savez_compressed(
        physiology,
        schema=np.asarray(PHYSIOLOGY_EVIDENCE_SCHEMA),
        lag_seconds=np.asarray([0.0, 2.0, 4.0]),
        coupling_scores=np.asarray([0.1, 0.4, 0.2]),
    )
    rendered = render_alignment_report(
        evidence, tmp_path / "report", physiology_coupling_evidence=physiology
    )
    assert rendered["fnirs_to_eeg"]["top1"] == 1.0
    assert (tmp_path / "report/figures/clip_similarity_positive_pairs.svg").is_file()
    assert (tmp_path / "report/figures/efrm_vs_directional_physiological_coupling.png").is_file()
    assert json.loads((tmp_path / "report/alignment_metrics.json").read_text())["pair_count"] == 4


def test_shared_public_split_matches_efrm_task_ordering() -> None:
    manifest = (
        METHOD_ROOT.parent / "STA-Net-PyTorch/split_registry/motor_imagery/"
        "cross_subject/public/outer0_inner0.json"
    )
    split = load_public_split_subjects(manifest)
    assert split.task == "motor_imagery"
    assert split.dataset_id == "eeg_fnirs_single_trial"
    assert split.train_subjects
    assert split.validation_subjects
    assert set(split.train_subjects).isdisjoint(split.validation_subjects)


def test_trial_mixed_boundary_is_stratified_deterministic_and_disjoint() -> None:
    class FakeDataset:
        def __init__(self) -> None:
            self.rows = []
            for subject in ("s1", "s2"):
                for record in ("r1", "r2"):
                    for condition in ("LMI", "RMI"):
                        for trial in range(5):
                            self.rows.append({
                                "dataset_id": "d",
                                "subject": subject,
                                "record_id": record,
                                "join_key": f"d|{subject}|{record}",
                                "event_index": trial,
                                "window_offset_s": float(trial),
                                "task_namespace": "d:motor_imagery",
                                "condition": condition,
                            })

        def __len__(self) -> int:
            return len(self.rows)

        def lightweight_metadata(self, index: int) -> dict:
            return self.rows[index]

    dataset = FakeDataset()
    source = PublicSplitSubjects(
        task="motor_imagery",
        dataset_id="d",
        manifest_path="/public/split.json",
        manifest_sha256="abc",
        manifest_schema="sta_net_subject_split_v1",
        train_subjects=("s1",),
        validation_subjects=("s2",),
        allowed_subjects=("s1", "s2"),
        metadata_sha256=None,
    )
    first = TrialMixedBoundary(dataset, source, validation_fraction=0.2, seed=7)
    second = TrialMixedBoundary(dataset, source, validation_fraction=0.2, seed=7)
    assert first.train_indices == second.train_indices
    assert first.validation_indices == second.validation_indices
    assert len(first.train_indices) == 32
    assert len(first.validation_indices) == 8
    assert set(first.train_indices).isdisjoint(first.validation_indices)
    manifest = first.manifest()
    assert manifest["subject_overlap_count"] == 2
    assert manifest["trial_overlap_count"] == 0
    assert all(
        counts == {"total": 5, "train": 4, "validation": 1}
        for counts in manifest["stratum_counts"].values()
    )


def test_protected_manifest_path_is_refused_before_split_loading(tmp_path: Path) -> None:
    protected = tmp_path / "protected"
    protected.mkdir()
    path = protected / "outer0.json"
    path.write_text(json.dumps({
        "schema": "sta_net_split_registry_v2",
        "task": "motor_imagery",
        "protected_test_opened": False,
    }))
    with pytest.raises(PermissionError, match="protected split manifest"):
        load_public_split_subjects(path)


def test_pretraining_analysis_audits_logs_and_renders_public_report(tmp_path: Path) -> None:
    run = tmp_path / "run"
    for child in ("metrics", "checkpoints", "figure_data"):
        (run / child).mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "schema": "efrm_sync_pretraining_run_v1",
        "status": "completed",
        "run_id": "synthetic",
        "protected_test_opened": False,
    }))
    (run / "status.json").write_text(json.dumps({
        "status": "completed",
        "epoch": 1,
        "protected_test_opened": False,
    }))
    (run / "resolved_config.yaml").write_text(
        "training:\n  epochs: 2\n  min_epochs: 1\n", encoding="utf-8"
    )
    epoch_rows = [
        {
            "epoch": epoch,
            "seconds": 10.0,
            "learning_rate": 1e-4,
            "train": {
                "batch_count": 1.0, "pair_count": 4.0, "loss": 5.0 - epoch,
                "eeg_reconstruction_loss": 1.0 - 0.2 * epoch,
                "fnirs_reconstruction_loss": 0.8 - 0.2 * epoch,
                "clip_alignment_loss": math.log(4),
            },
            "validation": {
                "batch_count": 1.0, "pair_count": 4.0, "loss": 5.1 - epoch,
                "eeg_reconstruction_loss": 1.1 - 0.2 * epoch,
                "fnirs_reconstruction_loss": 0.9 - 0.2 * epoch,
                "clip_alignment_loss": math.log(4),
            },
            "cuda_peak_allocated_gib": 1.0,
            "cuda_peak_reserved_gib": 1.2,
        }
        for epoch in range(2)
    ]
    (run / "metrics/epochs.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in epoch_rows), encoding="utf-8"
    )
    step_rows = [
        {
            "epoch": float(epoch), "batch": 0.0, "pair_count": 4.0,
            "loss": 5.0 - epoch, "eeg_reconstruction_loss": 1.0,
            "fnirs_reconstruction_loss": 0.8,
            "clip_alignment_loss": math.log(4), "gradient_norm": 1.0,
        }
        for epoch in range(2)
    ]
    (run / "metrics/train_steps.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in step_rows), encoding="utf-8"
    )
    (run / "checkpoints/latest.pt").write_bytes(b"same-checkpoint")
    (run / "checkpoints/best.pt").write_bytes(b"same-checkpoint")
    metadata = [
        {
            "sample_id": f"p{index}", "dataset_id": "public-dataset",
            "subject": f"s{index}", "record_id": f"r{index}",
            "join_key": f"d|r{index}", "condition": "task",
        }
        for index in range(4)
    ]
    export_alignment_evidence(
        run / "figure_data",
        eeg_embeddings=np.eye(4, dtype=np.float32),
        fnirs_embeddings=np.eye(4, dtype=np.float32),
        metadata=metadata,
    )

    result = analyze_pretraining_run(run)
    assert result["audit"]["run_state"] == "completed"
    assert result["audit"]["protected_test_opened"] is False
    assert result["alignment"]["eeg_to_fnirs"]["top1"] == 1.0
    assert result["interpretation"]["alignment_failure_warning"] is False
    assert (
        result["interpretation"]["dataset_level_alignment_impossibility_claim_supported"]
        is False
    )
    assert (run / "analysis/REPORT.md").is_file()
    assert (run / "analysis/figures/training_overview.svg").is_file()
    assert (run / "analysis/tables/epoch_metrics.csv").is_file()


def _wait_for_launcher_terminal_state(path: Path) -> dict[str, object]:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if path.is_file():
            state = json.loads(path.read_text(encoding="utf-8"))
            if state["status"] in {"completed", "failed", "launcher_failed"}:
                return state
        time.sleep(0.05)
    raise AssertionError(f"detached launcher did not reach a terminal state: {path}")


def test_detached_launcher_uses_new_session_file_log_and_exit_state(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control"
    run = tmp_path / "run"
    observation = tmp_path / "child.json"
    child_code = (
        "import json, os, pathlib; "
        f"pathlib.Path({str(observation)!r}).write_text(json.dumps({{"
        "'pid': os.getpid(), 'sid': os.getsid(0), 'pgrp': os.getpgrp(), "
        "'stdin_tty': os.isatty(0), 'stdout_tty': os.isatty(1)"
        "})); "
        "print('detached-child-output', flush=True)"
    )
    launched = launch_detached(
        command=[sys.executable, "-c", child_code],
        run_id="launcher_success",
        run_dir=run,
        control_dir=control,
    )
    state = _wait_for_launcher_terminal_state(control / "state.json")
    child = json.loads(observation.read_text(encoding="utf-8"))

    assert launched["supervisor_pid"] != os.getpid()
    assert state["status"] == "completed"
    assert state["exit_code"] == 0
    assert state["session_id"] == state["supervisor_pid"]
    assert child["sid"] == state["supervisor_pid"]
    assert child["stdin_tty"] is False
    assert child["stdout_tty"] is False
    assert "detached-child-output" in Path(state["log_path"]).read_text(encoding="utf-8")


def test_detached_launcher_records_failure_in_run_artifacts(tmp_path: Path) -> None:
    control = tmp_path / "control"
    run = tmp_path / "run"
    run.mkdir()
    (run / "status.json").write_text(
        json.dumps({"status": "running", "epoch": 2}), encoding="utf-8"
    )
    (run / "manifest.json").write_text(
        json.dumps({"status": "running", "run_id": "launcher_failure"}),
        encoding="utf-8",
    )
    launch_detached(
        command=[sys.executable, "-c", "raise SystemExit(7)"],
        run_id="launcher_failure",
        run_dir=run,
        control_dir=control,
    )
    state = _wait_for_launcher_terminal_state(control / "state.json")

    assert state["status"] == "failed"
    assert state["exit_code"] == 7
    assert json.loads((run / "status.json").read_text())["status"] == "failed"
    assert json.loads((run / "manifest.json").read_text())["exit_code"] == 7


def test_resume_archives_partial_epoch_steps_before_replay(tmp_path: Path) -> None:
    run = tmp_path / "run"
    metrics = run / "metrics"
    metrics.mkdir(parents=True)
    rows = [
        {"epoch": 0.0, "batch": 0.0},
        {"epoch": 0.0, "batch": 1.0},
        {"epoch": 1.0, "batch": 0.0},
        {"epoch": 1.0, "batch": 1.0},
    ]
    step_path = metrics / "train_steps.jsonl"
    step_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    result = _archive_incomplete_resume_steps(run, start_epoch=1)

    assert result is not None
    assert result["discarded_step_count"] == 2
    assert result["discarded_epoch_ids"] == [1]
    retained = [
        json.loads(line) for line in step_path.read_text(encoding="utf-8").splitlines()
    ]
    assert retained == rows[:2]
    archived = [
        json.loads(line)
        for line in Path(result["archive"]).read_text(encoding="utf-8").splitlines()
    ]
    assert archived == rows[2:]
