from __future__ import annotations

import json
from pathlib import Path
import sys

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
from efrm_pytorch.protocol import load_public_split_subjects
from efrm_pytorch.tasks import TASK_SPECS
from efrm_pytorch.training import cached_pretrain_backward
from efrm_pytorch.visualization import (
    PHYSIOLOGY_EVIDENCE_SCHEMA,
    export_alignment_evidence,
    render_alignment_report,
    retrieval_metrics,
)


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
        tmp_path / "evidence", eeg_embeddings=eeg, fnirs_embeddings=fnirs, metadata=metadata
    )
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
