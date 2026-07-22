import pytest
import torch

from src.losses.physiology_semantic import (
    PhysiologySemanticLoss,
    straight_through_codebook_balance_loss,
)
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


def test_token_mask_blocks_vq_updates_and_invalid_history():
    torch.manual_seed(8)
    model = _small_model().train()
    eeg_mask = torch.ones(1, 10, dtype=torch.bool)
    eeg_mask[:, 2] = False
    fnirs_mask = torch.ones(1, 10, dtype=torch.bool)
    outputs = model(
        torch.randn(1, 6, 4000),
        torch.randn(1, 2, 200),
        token_valid_masks={"eeg": eeg_mask, "fnirs": fnirs_mask},
    )
    # Target token 5 uses history tokens 0..4, which includes invalid token 2.
    assert not outputs["eeg"].context_valid_mask[0, 5]
    # Target token 8 uses history tokens 3..7 and is therefore valid.
    assert outputs["eeg"].context_valid_mask[0, 8]


def test_teacher_free_loss_accepts_no_sidecar_and_masks_reconstruction():
    torch.manual_seed(9)
    model = _small_model().eval()
    masks = {
        "eeg": torch.tensor([[True] + [False] * 9]),
        "fnirs": torch.tensor([[True] + [False] * 9]),
    }
    outputs = model(
        torch.randn(1, 6, 4000),
        torch.randn(1, 2, 200),
        token_valid_masks=masks,
    )
    criterion = PhysiologySemanticLoss(
        state_weight=0.0,
        prototype_weight=0.0,
        masked_state_weight=0.0,
        uncertainty_weighting=False,
    )
    losses = criterion(outputs, None, token_valid_masks=masks)
    assert torch.isfinite(losses["total"])


def test_semantic_only_reconstruction_does_not_train_residual_branch():
    torch.manual_seed(10)
    model = _small_model().train()
    outputs = model(torch.randn(1, 6, 4000), torch.randn(1, 2, 200))
    criterion = PhysiologySemanticLoss(
        state_weight=0.0,
        prototype_weight=0.0,
        masked_state_weight=0.0,
        reconstruction_mode="semantic_only",
    )

    criterion(outputs, None)["total"].backward()

    assert model.eeg_branch.semantic_head.weight.grad is not None
    assert model.eeg_branch.residual_head.weight.grad is None
    assert model.fnirs_branch.semantic_head.weight.grad is not None
    assert model.fnirs_branch.residual_head.weight.grad is None


def test_hard_semantic_reconstruction_uses_straight_through_codebook_lookup():
    torch.manual_seed(11)
    model = _small_model().eval()
    outputs = model(torch.randn(1, 6, 4000), torch.randn(1, 2, 200))
    criterion = PhysiologySemanticLoss(
        state_weight=0.0,
        prototype_weight=0.0,
        masked_state_weight=0.0,
        reconstruction_mode="semantic_only",
        reconstruction_semantic_input="hard",
    )

    assert torch.equal(
        criterion._selected_reconstruction(outputs["eeg"]),
        outputs["eeg"].hard_semantic_reconstruction,
    )
    assert not torch.equal(
        outputs["eeg"].hard_semantic_reconstruction,
        outputs["eeg"].semantic_reconstruction,
    )


def test_annealed_hard_reconstruction_starts_continuous_and_ends_hard():
    torch.manual_seed(12)
    model = _small_model().eval()
    eeg = torch.randn(1, 6, 4000)
    fnirs = torch.randn(1, 2, 200)
    model.set_quantization_strength(0.0)
    continuous = model(eeg, fnirs)
    model.set_quantization_strength(1.0)
    hard = model(eeg, fnirs)
    criterion = PhysiologySemanticLoss(
        state_weight=0.0,
        prototype_weight=0.0,
        masked_state_weight=0.0,
        reconstruction_mode="semantic_only",
        reconstruction_semantic_input="annealed_hard",
    )

    assert model.get_quantization_strength() == 1.0
    assert torch.equal(
        criterion._selected_reconstruction(hard["eeg"]),
        hard["eeg"].hard_semantic_reconstruction,
    )
    assert not torch.equal(
        continuous["eeg"].annealed_hard_semantic_reconstruction,
        hard["eeg"].annealed_hard_semantic_reconstruction,
    )


def test_straight_through_balance_tracks_hard_ids_and_has_soft_gradient():
    collapsed_logits = torch.tensor(
        [[[8.0, 0.0, 0.0, 0.0]] * 4], requires_grad=True
    )
    balanced_logits = torch.tensor(
        [[
            [8.0, 0.0, 0.0, 0.0],
            [0.0, 8.0, 0.0, 0.0],
            [0.0, 0.0, 8.0, 0.0],
            [0.0, 0.0, 0.0, 8.0],
        ]]
    )

    collapsed = straight_through_codebook_balance_loss(collapsed_logits)
    balanced = straight_through_codebook_balance_loss(balanced_logits)
    collapsed.backward()

    assert collapsed.item() > 0.99
    assert balanced.item() == pytest.approx(0.0, abs=1e-5)
    assert collapsed_logits.grad is not None
    assert collapsed_logits.grad.abs().sum().item() > 0.0


def test_balance_smoothing_sends_recovery_gradient_to_unused_codes():
    logits = torch.zeros(1, 32, 128, requires_grad=True)
    with torch.no_grad():
        logits[..., 0] = 0.01

    loss = straight_through_codebook_balance_loss(logits)
    loss.backward()

    assert loss.item() > 0.99
    assert logits.grad is not None
    assert logits.grad[..., 1:].abs().max().item() > 1.0e-6


def test_balance_loss_excludes_invalid_tokens():
    logits = torch.tensor(
        [[
            [8.0, 0.0],
            [0.0, 8.0],
        ]]
    )
    masked = straight_through_codebook_balance_loss(
        logits, torch.tensor([[True, False]])
    )
    unmasked = straight_through_codebook_balance_loss(logits)

    assert masked.item() > 0.99
    assert unmasked.item() == pytest.approx(0.0, abs=1e-5)
