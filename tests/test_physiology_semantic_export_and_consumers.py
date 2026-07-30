import argparse
import copy
import json

import numpy as np
import pytest
import torch

import experiments.scripts.export_physiology_semantic_tokens as exporter
from experiments.scripts.export_physiology_semantic_tokens import (
    EXPORT_SCHEMA,
    build_export_batch,
)
from src.foundation import FoundationModelConfig, TokenBatchAdapter, UnifiedMultimodalFoundationModel
from src.teachers.physical_state_teacher import PhysicalStateTeacher
from src.tokenizers.physiology_semantic_tokenizer import PhysiologySemanticTokenizer


def _model():
    return PhysiologySemanticTokenizer(
        eeg_encoder_dim=32,
        fnirs_encoder_dim=24,
        semantic_dim=8,
        eeg_residual_dim=8,
        fnirs_residual_dim=4,
        codebook_size=16,
    ).eval()


def _batch():
    return {
        "eeg": torch.randn(2, 6, 4000),
        "fnirs": torch.randn(2, 2, 200),
        "teacher": {
            "state_mean": torch.randn(2, 200, 5),
            "state_var": torch.rand(2, 200, 5) + 0.1,
            "neural_driver_eeg_rate": torch.randn(2, 4000, 1),
            "neural_driver_var_eeg_rate": torch.rand(2, 4000, 1) + 0.1,
            "teacher_valid_mask": torch.ones(2, 200, dtype=torch.bool),
        },
        "subject_id": torch.tensor([1, 2]),
        "label": torch.tensor([0, 1]),
        "crop_start_s": torch.tensor([0.0, 2.0]),
        "cache_entry_id": ["a", "b"],
        "source_name": ["source", "source"],
        "source_task": ["task", "task"],
        "anchor": ["anchor", "anchor"],
        "label_name": ["left", "right"],
    }


def test_consumer_modes_preserve_checkpoint_codebook_geometry():
    torch.manual_seed(12)
    model = _model()
    batch = _batch()
    with torch.no_grad():
        outputs = model(batch["eeg"], batch["fnirs"])
    adapter = TokenBatchAdapter()
    codebook = adapter.from_physiology_semantic_outputs(outputs, "codebook")
    expected = model.eeg_branch.quantizer.get_embedding(outputs["eeg"].quantizer.hard_ids)
    assert torch.equal(codebook.eeg.inputs_embeds, expected)

    soft = adapter.from_physiology_semantic_outputs(outputs, "soft")
    combined = adapter.from_physiology_semantic_outputs(outputs, "semantic_residual")
    hard = adapter.from_physiology_semantic_outputs(outputs, "hard")
    assert soft.eeg.inputs_embeds.shape == (2, 10, 8)
    assert combined.eeg.inputs_embeds.shape == (2, 10, 16)
    assert combined.fnirs.inputs_embeds.shape == (2, 10, 12)
    assert hard.eeg.inputs_embeds is None
    assert hard.eeg.input_ids.min().item() >= 2


def test_foundation_model_accepts_transferred_embeddings():
    model = _model()
    batch = _batch()
    with torch.no_grad():
        outputs = model(batch["eeg"], batch["fnirs"])
    foundation_batch = TokenBatchAdapter().from_physiology_semantic_outputs(outputs, "codebook")
    foundation = UnifiedMultimodalFoundationModel(
        FoundationModelConfig(
            eeg_vocab_size=18,
            fnirs_vocab_size=18,
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            max_seq_len=16,
        )
    ).eval()
    with torch.no_grad():
        encoded = foundation.encode(foundation_batch)
    assert encoded["eeg_states"].shape == (2, 10, 16)
    assert encoded["fnirs_states"].shape == (2, 10, 16)


def test_export_schema_and_checkpoint_round_trip():
    torch.manual_seed(13)
    model = _model()
    batch = _batch()
    teacher = PhysicalStateTeacher()(batch["teacher"])
    with torch.no_grad():
        expected = model(batch["eeg"], batch["fnirs"])
    restored = _model()
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    with torch.no_grad():
        observed = restored(batch["eeg"], batch["fnirs"])
    assert torch.equal(expected["eeg"].quantizer.hard_ids, observed["eeg"].quantizer.hard_ids)
    assert torch.equal(expected["fnirs"].quantizer.posterior, observed["fnirs"].quantizer.posterior)

    payload = build_export_batch(observed, teacher, batch, top_k=4)
    assert payload["eeg_hard_ids"].shape == (2, 10)
    assert payload["eeg_posterior_topk_indices"].shape == (2, 10, 4)
    assert payload["fnirs_expected_embedding"].shape == (2, 10, 8)
    assert payload["teacher_valid_mask"].dtype == np.bool_
    assert "eeg_patches" not in payload
    assert "eeg_posterior_entropy" not in payload
    assert "eeg_expected_reconstruction_mse" not in payload


def test_export_v3_optional_patch_diagnostics_and_collated_channel_names():
    torch.manual_seed(14)
    model = _model()
    batch = _batch()
    batch["selected_eeg_channels"] = [
        ("A-Fp1", "B-Fp1"),
        ("A-Fp2", "B-Fp2"),
        ("A-C3", "B-C3"),
        ("A-C4", "B-C4"),
        ("A-O1", "B-O1"),
        ("A-O2", "B-O2"),
    ]
    batch["arousal_state"] = ["calm", "active"]
    with torch.no_grad():
        outputs = model(batch["eeg"], batch["fnirs"])

    payload = build_export_batch(
        outputs,
        teacher=None,
        batch=batch,
        include_patches=True,
        include_assignment_diagnostics=True,
        include_reconstruction_diagnostics=True,
        extra_fields=("arousal_state",),
    )

    assert EXPORT_SCHEMA == "physiology_semantic_export_v3"
    assert payload["eeg_patches"].shape == (2, 10, 6, 400)
    assert payload["fnirs_patches"].shape == (2, 10, 2, 20)
    assert payload["selected_eeg_channels"].shape == (2, 6)
    assert payload["selected_eeg_channels"].tolist() == [
        ["A-Fp1", "A-Fp2", "A-C3", "A-C4", "A-O1", "A-O2"],
        ["B-Fp1", "B-Fp2", "B-C3", "B-C4", "B-O1", "B-O2"],
    ]
    assert payload["arousal_state"].tolist() == ["calm", "active"]

    reconstruction_names = (
        "expected",
        "semantic_only",
        "hard",
        "hard_semantic_only",
        "residual_only",
    )
    for modality in ("eeg", "fnirs"):
        entropy = payload[f"{modality}_posterior_entropy"]
        margin = payload[f"{modality}_posterior_top1_top2_margin"]
        latent_code_l2 = payload[f"{modality}_latent_code_l2"]
        assert entropy.shape == margin.shape == latent_code_l2.shape == (2, 10)
        assert np.isfinite(entropy).all()
        assert np.isfinite(margin).all()
        assert np.isfinite(latent_code_l2).all()
        assert (entropy >= 0.0).all()
        assert ((margin >= 0.0) & (margin <= 1.0)).all()
        assert (latent_code_l2 >= 0.0).all()

        expected_l2 = (
            outputs[modality].semantic_latent - outputs[modality].quantizer.quantized
        ).square().sum(dim=-1).sqrt().detach().numpy()
        np.testing.assert_allclose(latent_code_l2, expected_l2, rtol=1e-6, atol=1e-7)

        for reconstruction_name in reconstruction_names:
            key = f"{modality}_{reconstruction_name}_reconstruction_mse"
            assert payload[key].shape == (2, 10)
            assert np.isfinite(payload[key]).all()
            assert (payload[key] >= 0.0).all()

        expected_mse = (
            outputs[modality].reconstruction - outputs[modality].patches
        ).square().mean(dim=(-1, -2)).detach().numpy()
        np.testing.assert_allclose(
            payload[f"{modality}_expected_reconstruction_mse"],
            expected_mse,
            rtol=1e-6,
            atol=1e-7,
        )


def test_run_refuses_protected_test_before_loading_checkpoint(tmp_path):
    args = argparse.Namespace(
        checkpoint=str(tmp_path / "missing.pt"),
        split="test",
        output=str(tmp_path / "test-export.npz"),
    )
    with pytest.raises(ValueError, match="protected test split"):
        exporter.run(args)


def test_run_is_backward_compatible_atomic_and_records_test_access(tmp_path, monkeypatch):
    torch.manual_seed(15)
    model = _model()
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {"config": {"data": {}}, "model_state": copy.deepcopy(model.state_dict())},
        checkpoint_path,
    )
    batch = _batch()
    monkeypatch.setattr(exporter, "create_tokenizer", lambda _config: model)
    monkeypatch.setattr(
        exporter,
        "create_configured_multimodal_dataloaders",
        lambda _config: {"val": [batch], "test": [batch]},
    )

    output = tmp_path / "tokens.npz"
    # This deliberately resembles a pre-v3 hand-written Namespace: all newly
    # introduced include/force/allow-test attributes are absent.
    args = argparse.Namespace(
        checkpoint=str(checkpoint_path),
        config=None,
        split="val",
        output=str(output),
        top_k=4,
        max_batches=1,
    )
    assert exporter.run(args) == output.resolve()
    manifest_path = output.with_suffix(".npz.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "physiology_semantic_export_v3"
    assert manifest["protected_test_opened"] is False
    assert manifest["include_patches"] is False
    assert manifest["include_assignment_diagnostics"] is False
    assert manifest["include_reconstruction_diagnostics"] is False
    assert manifest["cache_role"] == "checkpoint_assignment"
    assert manifest["checkpoint_independent"] is False
    assert manifest["deterministic_replay"] is True
    assert manifest["drop_last"] is False
    assert manifest["max_batches"] == 1
    assert manifest["replay_scope"] == "first_1_batches"
    assert manifest["npz_sha256"] == exporter._sha256_path(output)
    with np.load(output, allow_pickle=False) as exported:
        assert exported["eeg_codebook"].shape == (16, 8)
        assert exported["fnirs_codebook"].shape == (16, 8)

    output_before = output.read_bytes()
    manifest_before = manifest_path.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        exporter.run(args)
    assert output.read_bytes() == output_before
    assert manifest_path.read_bytes() == manifest_before

    # An explicitly authorized test export can replace both artifacts only
    # when force is also explicit.
    args.split = "test"
    args.allow_test = True
    args.force = True
    assert exporter.run(args) == output.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["split"] == "test"
    assert manifest["protected_test_opened"] is True
    assert not list(tmp_path.glob(".*.tmp"))
