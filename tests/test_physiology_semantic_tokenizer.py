import torch

from src.losses.physiology_semantic import PhysiologySemanticLoss
from src.teachers.physical_state_teacher import PhysicalStateTeacher
from src.tokenizers.physiology_semantic_tokenizer import (
    FixedHistoryContext,
    PhysiologySemanticTokenizer,
)


def _small_model():
    return PhysiologySemanticTokenizer(
        eeg_encoder_dim=32,
        fnirs_encoder_dim=24,
        semantic_dim=8,
        eeg_residual_dim=8,
        fnirs_residual_dim=4,
        codebook_size=16,
        quantizer_kwargs={"revive_dead_codes": False},
    )


def _teacher(batch_size=2, valid=True):
    mask = torch.full((batch_size, 200), valid, dtype=torch.bool)
    return {
        "state_mean": torch.randn(batch_size, 200, 5),
        "state_var": torch.rand(batch_size, 200, 5) + 0.1,
        "neural_driver_eeg_rate": torch.randn(batch_size, 4000, 1),
        "neural_driver_var_eeg_rate": torch.rand(batch_size, 4000, 1) + 0.1,
        "teacher_valid_mask": mask,
    }


def test_patch_identity_is_position_invariant():
    torch.manual_seed(3)
    model = _small_model().eval()
    patch = torch.randn(1, 6, 400)
    eeg = torch.randn(1, 6, 4000)
    eeg[:, :, :400] = patch
    eeg[:, :, 2800:3200] = patch
    with torch.no_grad():
        output = model.encode_eeg(eeg)
    assert torch.allclose(output.quantizer.logits[:, 0], output.quantizer.logits[:, 7], atol=1e-6)
    assert torch.equal(output.quantizer.hard_ids[:, 0], output.quantizer.hard_ids[:, 7])
    assert torch.allclose(
        output.quantizer.expected_embedding[:, 0], output.quantizer.expected_embedding[:, 7], atol=1e-6
    )


def test_fixed_history_context_excludes_target_and_future():
    torch.manual_seed(4)
    context = FixedHistoryContext(embedding_dim=8, state_dim=3, history_tokens=5, depth=1, num_heads=2).eval()
    tokens = torch.randn(1, 10, 8)
    changed = tokens.clone()
    changed[:, 7:] = torch.randn_like(changed[:, 7:]) * 100.0
    with torch.no_grad():
        original, original_mask = context(tokens)
        perturbed, changed_mask = context(changed)
    assert torch.equal(original[:, 7], perturbed[:, 7])
    assert torch.equal(original_mask, changed_mask)
    assert not original_mask[:, :5].any()
    assert original_mask[:, 5:].all()


def test_modality_outputs_and_gradients_are_independent():
    torch.manual_seed(5)
    model = _small_model().eval()
    eeg_a = torch.randn(1, 6, 4000)
    eeg_b = torch.randn(1, 6, 4000)
    fnirs = torch.randn(1, 2, 200)
    output_a = model(eeg_a, fnirs)
    output_b = model(eeg_b, fnirs)
    assert torch.equal(output_a["fnirs"].quantizer.logits, output_b["fnirs"].quantizer.logits)

    model.zero_grad(set_to_none=True)
    output_a["eeg"].state_prediction.sum().backward()
    assert any(parameter.grad is not None for parameter in model.eeg_branch.parameters())
    assert all(parameter.grad is None for parameter in model.fnirs_branch.parameters())


def test_model_shapes_and_invalid_teacher_supervision():
    torch.manual_seed(6)
    model = _small_model().eval()
    outputs = model(torch.randn(2, 6, 4000), torch.randn(2, 2, 200))
    teacher = PhysicalStateTeacher()(_teacher(valid=False))
    losses = PhysiologySemanticLoss()(outputs, teacher)
    assert outputs["eeg"].reconstruction.shape == (2, 10, 6, 400)
    assert outputs["fnirs"].reconstruction.shape == (2, 10, 2, 20)
    assert outputs["eeg"].quantizer.hard_ids.shape == (2, 10)
    assert outputs["fnirs"].quantizer.posterior.shape == (2, 10, 16)
    assert losses["state"].item() == 0.0
    assert losses["prototype"].item() == 0.0
    assert losses["masked_state"].item() == 0.0


def test_loss_coordinate_gate_excludes_unadmitted_targets():
    torch.manual_seed(7)
    model = _small_model().eval()
    outputs = model(torch.randn(1, 6, 4000), torch.randn(1, 2, 200))
    teacher = PhysicalStateTeacher()(_teacher(batch_size=1, valid=True))
    criterion = PhysiologySemanticLoss(
        eeg_coordinate_mask=torch.tensor([True, False, False, False, False, False]),
        fnirs_coordinate_mask=torch.tensor([True, False, False, False, False, False, False, False, False]),
    )
    original = criterion(outputs, teacher)["state"]
    teacher.eeg_target[..., 1:] += 1e6
    teacher.fnirs_target[..., 1:] += 1e6
    changed = criterion(outputs, teacher)["state"]
    assert torch.equal(original, changed)
