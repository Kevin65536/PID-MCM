import copy

import numpy as np
import torch

from experiments.scripts.export_physiology_semantic_tokens import build_export_batch
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
