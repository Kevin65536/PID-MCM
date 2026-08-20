import pytest
import torch

from src.losses.ssm_observation import (
    ssm_observation_objective,
    uncertainty_weighted_huber_loss,
)
from src.tokenizers.ssm_observation_shared_private import (
    CausalFIRTransferHead,
    SSMObservationSharedPrivateModel,
)


def _model(cross: bool = True):
    return SSMObservationSharedPrivateModel(
        eeg_channels=2,
        fnirs_channels=2,
        eeg_patch_samples=4,
        fnirs_patch_samples=2,
        num_tokens=5,
        eeg_target_dim=6,
        fnirs_target_dim=4,
        shared_dim=16,
        eeg_private_dim=12,
        fnirs_private_dim=8,
        eeg_shared_history_tokens=2,
        fnirs_shared_history_tokens=3,
        encoder_depth=1,
        encoder_num_heads=4,
        encoder_feedforward_dim=32,
        decoder_hidden_dim=24,
        dropout=0.0,
        class_count=2,
        allowed_lags=(0, 1, 2),
        cross_prediction_lags=(1, 2) if cross else None,
    )


def test_model_separates_clean_shared_and_private_residual_decoders():
    torch.manual_seed(4)
    model = _model().eval()
    eeg = torch.randn(3, 2, 20)
    fnirs = torch.randn(3, 2, 10)
    mask = torch.ones(3, 5, dtype=torch.bool)
    output = model(eeg, fnirs, mask, mask)
    assert model.vector_quantization is False
    assert not any(
        "quantizer" in name.lower() or "codebook" in name.lower()
        for name, _ in model.named_modules()
    )
    assert not any(
        "posterior" in name.lower()
        or "hard_id" in name.lower()
        or "commitment" in name.lower()
        for name in output
    )
    assert output["eeg_clean_prediction"].shape == (3, 5, 6)
    assert output["fnirs_clean_prediction"].shape == (3, 5, 4)
    assert output["eeg_residual_prediction"].shape == (3, 5, 6)
    assert output["fnirs_residual_prediction"].shape == (3, 5, 4)
    assert output["private_only_logits"].shape == (3, 2)
    assert output["private_plus_shared_marginal_logits"].shape == (3, 2)
    assert output["private_shared_interaction_logits"].shape == (3, 2)
    torch.testing.assert_close(
        output["interaction_only_logits"].sum(dim=-1),
        torch.zeros(3),
        atol=1e-6,
        rtol=0.0,
    )
    assert output["fnirs_cross_prediction"].shape == (3, 5, 4)
    # Parameter ownership is structurally disjoint: private residual decoders
    # have no route through either shared decoder.
    shared_ids = {id(value) for value in model.eeg_clean_decoder.parameters()}
    private_ids = {id(value) for value in model.eeg_residual_decoder.parameters()}
    assert shared_ids.isdisjoint(private_ids)


def test_private_residual_loss_does_not_update_shared_encoder():
    torch.manual_seed(5)
    model = _model(cross=False)
    eeg = torch.randn(2, 2, 20)
    fnirs = torch.randn(2, 2, 10)
    output = model(eeg, fnirs)
    loss = output["eeg_residual_prediction"].square().mean()
    loss.backward()
    assert all(value.grad is None for value in model.eeg_shared_encoder.parameters())
    assert any(
        value.grad is not None and bool(value.grad.abs().sum() > 0)
        for value in model.eeg_private_encoder.parameters()
    )


def test_fir_is_asymmetric_and_uses_only_past_eeg_tokens():
    head = CausalFIRTransferHead(3, 2, lags=(1, 2))
    eeg = torch.randn(1, 5, 3)
    changed = eeg.clone()
    changed[:, 3:] += 100.0
    mask = torch.ones(1, 5, dtype=torch.bool)
    baseline, support = head(eeg, mask)
    result, _ = head(changed, mask)
    # Future EEG tokens cannot alter fNIRS predictions at earlier endpoints.
    torch.testing.assert_close(baseline[:, :4], result[:, :4])
    assert not torch.equal(baseline[:, 4], result[:, 4])
    assert not bool(support[:, 0].any())


def test_uncertainty_weighted_huber_downweights_uncertain_teacher_points():
    prediction = torch.tensor([[[2.0, 2.0]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    std = torch.tensor([[[0.1, 10.0]]])
    mask = torch.ones(1, 1, 2, dtype=torch.bool)
    loss = uncertainty_weighted_huber_loss(
        prediction,
        target,
        std,
        mask,
        weight_min=0.01,
        weight_max=100.0,
    )
    loss.backward()
    assert prediction.grad is not None
    assert prediction.grad[0, 0, 0] > 100 * prediction.grad[0, 0, 1]
    with pytest.raises(ValueError, match="no valid"):
        uncertainty_weighted_huber_loss(
            prediction.detach(), target, std, torch.zeros_like(mask)
        )


def test_complete_ssm_observation_objective_has_optional_xpred():
    model = _model(cross=True)
    eeg = torch.randn(2, 2, 20)
    fnirs = torch.randn(2, 2, 10)
    output = model(eeg, fnirs)
    batch = {
        "eeg_clean_target": torch.randn(2, 5, 6),
        "fnirs_clean_target": torch.randn(2, 5, 4),
        "eeg_residual_target": torch.randn(2, 5, 6),
        "fnirs_residual_target": torch.randn(2, 5, 4),
        "eeg_predictive_std": torch.ones(2, 5, 6),
        "fnirs_predictive_std": torch.ones(2, 5, 4),
        "eeg_target_valid_mask": torch.ones(2, 5, dtype=torch.bool),
        "fnirs_target_valid_mask": torch.ones(2, 5, dtype=torch.bool),
    }
    total, components = ssm_observation_objective(
        output, batch, cross_prediction_weight=0.05
    )
    assert torch.isfinite(total)
    assert "eeg_to_fnirs" in components
