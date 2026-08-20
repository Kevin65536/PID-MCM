import pytest
import torch

from src.tokenizers.lag_conditioned_baseline import (
    B0ContinuousSharedPrivate,
    masked_token_mean,
)


def _model() -> B0ContinuousSharedPrivate:
    return B0ContinuousSharedPrivate(
        eeg_channels=3,
        fnirs_channels=4,
        eeg_native_dim=7,
        fnirs_native_dim=8,
        class_count=2,
        eeg_patch_samples=20,
        fnirs_patch_samples=4,
        num_tokens=5,
        shared_dim=16,
        eeg_private_dim=12,
        fnirs_private_dim=8,
        encoder_depth=1,
        encoder_num_heads=4,
        encoder_feedforward_dim=32,
        native_decoder_hidden_dim=16,
        raw_decoder_hidden_dim=16,
        classifier_hidden_dim=12,
        dropout=0.0,
    )


def _nonzero_grad(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for parameter in module.parameters()
    )


def test_b0_shapes_and_logit_ablations():
    model = _model().eval()
    eeg = torch.randn(2, 3, 100)
    fnirs = torch.randn(2, 4, 20)
    with torch.no_grad():
        output = model(eeg, fnirs)

    assert output["eeg_shared"].shape == (2, 5, 16)
    assert output["fnirs_shared"].shape == (2, 5, 16)
    assert output["eeg_private"].shape == (2, 5, 12)
    assert output["fnirs_private"].shape == (2, 5, 8)
    assert output["eeg_native"].shape == (2, 5, 7)
    assert output["fnirs_native"].shape == (2, 5, 8)
    assert output["eeg_raw"].shape == eeg.shape
    assert output["fnirs_raw"].shape == fnirs.shape
    assert set(output["logits"]) == {
        "shared_marginal_only",
        "private_only",
        "combined",
    }
    assert output["logits"]["combined"].shape == (2, 2)


def test_b0_raw_gradient_is_stopped_at_shared_encoders():
    model = _model().train()
    output = model(torch.randn(1, 3, 100), torch.randn(1, 4, 20))
    (output["eeg_raw"].square().mean() + output["fnirs_raw"].square().mean()).backward()

    assert not _nonzero_grad(model.eeg_shared_encoder)
    assert not _nonzero_grad(model.fnirs_shared_encoder)
    assert _nonzero_grad(model.eeg_private_encoder)
    assert _nonzero_grad(model.fnirs_private_encoder)
    assert _nonzero_grad(model.eeg_raw_decoder)
    assert _nonzero_grad(model.fnirs_raw_decoder)


def test_b0_native_loss_updates_shared_but_not_private():
    model = _model().train()
    output = model(torch.randn(1, 3, 100), torch.randn(1, 4, 20))
    (output["eeg_native"].square().mean() + output["fnirs_native"].square().mean()).backward()

    assert _nonzero_grad(model.eeg_shared_encoder)
    assert _nonzero_grad(model.fnirs_shared_encoder)
    assert not _nonzero_grad(model.eeg_private_encoder)
    assert not _nonzero_grad(model.fnirs_private_encoder)


def test_b0_invalid_tokens_are_zeroed_across_outputs():
    model = _model().eval()
    mask = torch.tensor([[True, True, False, True, False]])
    with torch.no_grad():
        output = model(
            torch.randn(1, 3, 100),
            torch.randn(1, 4, 20),
            mask,
            mask,
        )

    assert torch.equal(output["eeg_shared"][0, ~mask[0]], torch.zeros(2, 16))
    assert torch.equal(output["fnirs_native"][0, ~mask[0]], torch.zeros(2, 8))
    eeg_point_mask = mask.repeat_interleave(20, dim=1)
    fnirs_point_mask = mask.repeat_interleave(4, dim=1)
    assert torch.equal(
        output["eeg_raw"].masked_select(~eeg_point_mask.unsqueeze(1)),
        torch.zeros(3 * 40),
    )
    assert torch.equal(
        output["fnirs_raw"].masked_select(~fnirs_point_mask.unsqueeze(1)),
        torch.zeros(4 * 8),
    )


def test_masked_token_mean_is_nan_safe_and_rejects_admitted_nonfinite_values():
    values = torch.tensor(
        [
            [[2.0, 4.0], [float("nan"), float("inf")], [6.0, 8.0]],
            [[float("-inf"), float("nan")], [1.0, 3.0], [float("nan"), 5.0]],
        ]
    )
    mask = torch.tensor([[True, False, True], [False, False, False]])

    pooled = masked_token_mean(values, mask)
    torch.testing.assert_close(pooled, torch.tensor([[4.0, 6.0], [0.0, 0.0]]))

    with pytest.raises(FloatingPointError, match="non-finite admitted"):
        masked_token_mean(values, torch.ones_like(mask))
