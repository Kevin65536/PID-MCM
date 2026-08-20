import torch
import pytest

from src.losses.lag_conditioned import (
    masked_mse,
    native_feature_prediction_loss,
    raw_patch_reconstruction_loss,
    weighted_pretraining_loss,
)


def test_masked_mse_drops_invalid_and_nonfinite_targets():
    prediction = torch.tensor([[[1.0, 3.0], [5.0, 7.0]]], requires_grad=True)
    target = torch.tensor([[[0.0, float("nan")], [3.0, 100.0]]])
    mask = torch.tensor([[[True, True], [True, False]]])
    loss = masked_mse(prediction, target, mask)

    assert loss.item() == pytest.approx((1.0 + 4.0) / 2.0)
    loss.backward()
    assert prediction.grad is not None
    assert prediction.grad[0, 0, 1] == 0.0
    assert prediction.grad[0, 1, 1] == 0.0


def test_masked_mse_fails_on_nonfinite_admitted_prediction():
    with pytest.raises(FloatingPointError, match="prediction"):
        masked_mse(
            torch.tensor([float("nan")]),
            torch.tensor([0.0]),
            torch.tensor([True]),
        )


def test_raw_loss_combines_point_and_channel_support():
    prediction = torch.zeros(1, 2, 4, requires_grad=True)
    target = torch.ones_like(prediction)
    point = torch.tensor([[True, False, True, False]])
    channel = torch.tensor([[True, False]])
    loss = raw_patch_reconstruction_loss(
        prediction,
        target,
        point_valid_mask=point,
        channel_valid_mask=channel,
    )

    assert loss == 1.0
    loss.backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad) == 2


def test_native_loss_requires_coordinate_mask():
    prediction = torch.zeros(2, 3, 4)
    target = torch.ones_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    assert native_feature_prediction_loss(prediction, target, mask) == 1.0
    with pytest.raises(ValueError, match="shapes differ"):
        native_feature_prediction_loss(prediction, target, mask[..., :2])


def test_weighted_objective_keeps_all_named_terms_and_gradients():
    parameter = torch.tensor(2.0, requires_grad=True)
    losses = {"native": parameter.square(), "raw": (parameter - 1).square()}
    total, weighted = weighted_pretraining_loss(
        losses, {"native": 0.5, "raw": 2.0}
    )

    assert total == 4.0
    assert set(weighted) == set(losses)
    total.backward()
    assert parameter.grad == 6.0


def test_weighted_objective_fails_on_name_or_negative_weight_drift():
    with pytest.raises(ValueError, match="names differ"):
        weighted_pretraining_loss({"a": torch.tensor(1.0)}, {"b": 1.0})
    with pytest.raises(ValueError, match="non-negative"):
        weighted_pretraining_loss({"a": torch.tensor(1.0)}, {"a": -1.0})
