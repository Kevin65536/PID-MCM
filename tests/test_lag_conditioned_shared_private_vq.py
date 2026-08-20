"""Focused contracts for the local-causal LC-SPVQ core."""

import pytest
import torch

from src.tokenizers.lag_conditioned_shared_private_vq import (
    FullWindowPatchEncoder,
    LCSPVQModel,
    LagAwareContinuousMatchingLoss,
    LocalCausalPatchEncoder,
    LowRankLagCouplingHead,
    PrivatePooledClassifier,
    _masked_mean,
    _masked_pair_mean,
    lag_aware_continuous_matching_loss,
)


def _small_model() -> LCSPVQModel:
    return LCSPVQModel(
        eeg_channels=2,
        fnirs_channels=1,
        eeg_patch_samples=4,
        fnirs_patch_samples=2,
        num_tokens=4,
        shared_dim=64,
        eeg_private_dim=16,
        fnirs_private_dim=8,
        eeg_shared_history_patches=2,
        fnirs_shared_history_patches=3,
        encoder_depth=1,
        encoder_num_heads=4,
        encoder_feedforward_dim=128,
        private_encoder_num_heads=4,
        private_encoder_feedforward_dim=64,
        dropout=0.0,
        native_decoder_hidden_dim=32,
        eeg_native_feature_dim=7,
        fnirs_native_feature_dim=5,
        num_classes=3,
    )


def _inputs(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(batch, 2, 16), torch.randn(batch, 1, 8)


def _has_nonzero_grad(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for parameter in module.parameters()
    )


def test_local_causal_encoder_never_reads_future_patch():
    torch.manual_seed(3)
    encoder = LocalCausalPatchEncoder(
        input_channels=1,
        patch_samples=4,
        num_tokens=6,
        latent_dim=8,
        history_patches=2,
        depth=1,
        num_heads=2,
        feedforward_dim=24,
        dropout=0.0,
    ).eval()
    signal = torch.randn(1, 1, 24)
    changed = signal.clone()
    # Change patches 3 onward.  Tokens 0, 1, and 2 must be unchanged because
    # each token sees itself and at most one past patch.
    changed[..., 12:] += 100.0
    with torch.no_grad():
        baseline = encoder(signal)
        result = encoder(changed)
    torch.testing.assert_close(baseline[:, :3], result[:, :3], atol=0.0, rtol=0.0)
    assert not torch.equal(baseline[:, 4], result[:, 4])
    assert encoder.token_temporal_scope == "causal_local_history"
    # A true entry is a forbidden attention edge: future keys are forbidden.
    assert bool(encoder.attention_mask[0, 1])
    assert not bool(encoder.attention_mask[2, 1])


def test_private_encoder_is_explicitly_full_window_bidirectional():
    encoder = FullWindowPatchEncoder(
        input_channels=1,
        patch_samples=4,
        num_tokens=5,
        latent_dim=8,
        depth=1,
        num_heads=2,
        feedforward_dim=24,
        dropout=0.0,
    )
    assert encoder.token_temporal_scope == "bidirectional_full_window"
    assert not hasattr(encoder, "attention_mask")


def test_model_has_four_independent_encoders_and_complete_vq_surfaces():
    torch.manual_seed(5)
    model = _small_model().eval()
    eeg, fnirs = _inputs()
    with torch.no_grad():
        output = model(eeg, fnirs)

    assert model.shared_dim == 64
    assert model.eeg_quantizer.codebook_size == 16
    assert model.fnirs_quantizer.codebook_size == 16
    assert model.eeg_shared_encoder is not model.fnirs_shared_encoder
    assert model.eeg_shared_encoder is not model.eeg_private_encoder
    assert model.fnirs_shared_encoder is not model.fnirs_private_encoder
    assert model.eeg_quantizer.codebook.data_ptr() != model.fnirs_quantizer.codebook.data_ptr()

    for modality, private_dim in (("eeg", 16), ("fnirs", 8)):
        surface = output[modality]
        assert surface["pre_vq_latent"].shape == (2, 4, 64)
        assert surface["posterior"].shape == (2, 4, 16)
        assert surface["hard_ids"].shape == (2, 4)
        assert surface["expected_embedding"].shape == (2, 4, 64)
        assert surface["annealed_embedding"].shape == (2, 4, 64)
        assert surface["quantized_embedding"].shape == (2, 4, 64)
        assert surface["private"].shape == (2, 4, private_dim)
        assert surface["quantizer"].posterior.shape == (2, 4, 16)

    assert output["coupling_logits"].shape == (2, 4, 4, 3)
    assert output["coupling_only_logits"].shape == (2, 3)
    assert output["shared_marginal_only_logits"].shape == (2, 3)
    assert output["private_only_logits"].shape == (2, 3)
    assert output["private_logits"].shape == (2, 3)
    assert output["combined_logits"].shape == (2, 3)
    assert output["eeg_raw"].shape == eeg.shape
    assert output["fnirs_raw"].shape == fnirs.shape
    assert output["eeg_native_features"].shape == eeg.shape  # raw compatibility alias
    assert output["fnirs_native_features"].shape == fnirs.shape
    assert output["eeg_native_target_prediction"].shape == (2, 4, 7)
    assert output["fnirs_native_target_prediction"].shape == (2, 4, 5)
    assert model.coupling_head.input_dim == 16
    assert model.coupling_head.allowed_lags == (0, 1, 2, 3, 4, 5)
    torch.testing.assert_close(
        output["eeg_projection"],
        model.eeg_projection_head(output["eeg_pre_vq"]),
    )
    torch.testing.assert_close(
        output["combined_logits"],
        output["coupling_only_logits"] + output["private_only_logits"],
    )


def test_vq_codebooks_are_independent_and_ids_are_not_swapped():
    torch.manual_seed(7)
    model = _small_model().eval()
    before = model.fnirs_quantizer.codebook.detach().clone()
    with torch.no_grad():
        model.eeg_quantizer.codebook.add_(1.0)
    torch.testing.assert_close(model.fnirs_quantizer.codebook, before)
    assert model.eeg_quantizer is not model.fnirs_quantizer
    assert model.eeg_quantizer.get_codebook_weight().data_ptr() != model.fnirs_quantizer.get_codebook_weight().data_ptr()


def test_raw_decoder_gradient_isolated_from_shared_encoder():
    torch.manual_seed(11)
    model = _small_model().train()
    eeg, fnirs = _inputs(batch=1)
    output = model(eeg, fnirs)
    loss = output["eeg_native_features"].square().mean() + output["fnirs_native_features"].square().mean()
    loss.backward()

    assert not _has_nonzero_grad(model.eeg_shared_encoder)
    assert not _has_nonzero_grad(model.fnirs_shared_encoder)
    assert _has_nonzero_grad(model.eeg_private_encoder)
    assert _has_nonzero_grad(model.fnirs_private_encoder)
    assert model.native_feature_decoders is model.raw_decoders
    assert _has_nonzero_grad(model.raw_decoders["eeg"])
    assert _has_nonzero_grad(model.raw_decoders["fnirs"])


def test_native_target_feature_loss_reaches_shared_pre_vq_encoders():
    torch.manual_seed(12)
    model = _small_model().train()
    eeg, fnirs = _inputs(batch=1)
    output = model(eeg, fnirs)
    loss = (
        output["eeg_native_target_prediction"].square().mean()
        + output["fnirs_native_target_prediction"].square().mean()
    )
    loss.backward()
    assert _has_nonzero_grad(model.eeg_shared_encoder)
    assert _has_nonzero_grad(model.fnirs_shared_encoder)
    assert not _has_nonzero_grad(model.eeg_private_encoder)
    assert not _has_nonzero_grad(model.fnirs_private_encoder)
    assert _has_nonzero_grad(model.native_target_decoders["eeg"])
    assert _has_nonzero_grad(model.native_target_decoders["fnirs"])


def test_lag_matching_prefers_matched_target_over_explicit_derangement():
    torch.manual_seed(13)
    query = torch.randn(4, 6, 12, requires_grad=True)
    matched = query.detach() + 0.02 * torch.randn(4, 6, 12)
    deranged = matched.roll(shifts=1, dims=0)
    deranged[0, -1] = float("nan")
    deranged_mask = torch.ones(4, 6, dtype=torch.bool)
    deranged_mask[0, -1] = False
    kwargs = dict(
        positive_lag_weights={0: 1.0},
        temperature=0.1,
        bidirectional=False,
        subject_ids=torch.tensor([0, 0, 1, 1]),
        relative_time=torch.arange(6).expand(4, 6),
    )
    matched_loss = lag_aware_continuous_matching_loss(
        query,
        matched,
        deranged_target=deranged,
        deranged_target_valid_mask=deranged_mask,
        **kwargs,
    )
    deranged_loss = lag_aware_continuous_matching_loss(
        query,
        deranged,
        target_valid_mask=deranged_mask,
        deranged_target=matched,
        deranged_target_valid_mask=torch.ones_like(deranged_mask),
        **kwargs,
    )
    assert torch.isfinite(matched_loss)
    assert matched_loss < deranged_loss
    matched_loss.backward()
    assert query.grad is not None and bool(query.grad.abs().sum() > 0)


def test_deranged_banks_use_only_explicit_registered_negative_pairs():
    torch.manual_seed(14)
    query = torch.randn(2, 2, 6)
    target = query + 0.01 * torch.randn(2, 2, 6)
    deranged_target = target.roll(1, 0)
    deranged_query = query.roll(1, 0)
    no_base = torch.zeros(2, 2, 2, 2, dtype=torch.bool)
    aligned = torch.eye(2, dtype=torch.bool)[:, None, :, None] & torch.eye(
        2, dtype=torch.bool
    )[None, :, None, :]
    details = lag_aware_continuous_matching_loss(
        query,
        target,
        positive_lag_weights={0: 1.0},
        negative_mask=no_base,
        deranged_target=deranged_target,
        deranged_target_negative_mask=aligned,
        deranged_query=deranged_query,
        deranged_query_negative_mask=aligned,
        return_details=True,
    )
    # Four positives plus four separately appended registered negatives.
    assert int(details["forward"]["candidate_count"]) == 8
    assert int(details["reverse"]["candidate_count"]) == 8


def test_posterior_temperature_setter_updates_both_independent_vqs():
    model = _small_model()
    model.set_posterior_temperature(0.37)
    assert model.get_posterior_temperature() == 0.37
    assert model.eeg_quantizer.temperature == 0.37
    assert model.fnirs_quantizer.temperature == 0.37
    try:
        model.set_posterior_temperature(0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive posterior temperature must fail")


def test_lag_matching_is_bidirectional_and_strictly_masked():
    torch.manual_seed(17)
    query = torch.randn(3, 5, 10, requires_grad=True)
    target = query.detach() + 0.03 * torch.randn(3, 5, 10)
    target[0, -1] = float("nan")
    query_mask = torch.ones(3, 5, dtype=torch.bool)
    target_mask = torch.ones(3, 5, dtype=torch.bool)
    target_mask[0, -1] = False
    result = lag_aware_continuous_matching_loss(
        query,
        target,
        positive_lag_weights={0: 1.0, 1: 0.5},
        query_valid_mask=query_mask,
        target_valid_mask=target_mask,
        bidirectional=True,
        return_details=True,
    )
    assert torch.isfinite(result["loss"])
    assert result["forward"]["valid_query_mask"].any()
    result["loss"].backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()


def test_learnable_lag_mixture_computes_per_lag_symmetric_terms():
    torch.manual_seed(18)
    query = torch.randn(3, 5, 10, requires_grad=True)
    target = query.detach() + 0.05 * torch.randn(3, 5, 10)
    loss_module = LagAwareContinuousMatchingLoss(
        positive_lag_weights={0: 1.0, 1: 0.5},
        bidirectional=True,
        learnable_lag_mixture=True,
    )
    result = loss_module(query, target, return_details=True)
    assert torch.isfinite(result["loss"])
    assert len(result["per_lag"]) == 2
    assert all("reverse" in entry for entry in result["per_lag"])
    torch.testing.assert_close(result["lag_weights"].sum(), torch.ones(()))
    result["loss"].backward()
    assert loss_module.lag_mixture_logits.grad is not None
    assert bool(loss_module.lag_mixture_logits.grad.abs().sum() > 0)


def test_learnable_lag_mixture_recovers_known_two_patch_delay():
    torch.manual_seed(8)
    batch, tokens, dimension = 48, 8, 12
    query = torch.randn(batch, tokens, dimension)
    target = torch.randn(batch, tokens, dimension)
    target[:, 2:] = query[:, :-2] + 0.02 * torch.randn(
        batch, tokens - 2, dimension
    )
    objective = LagAwareContinuousMatchingLoss(
        positive_lag_weights={lag: 1.0 for lag in range(5)},
        temperature=0.1,
        bidirectional=True,
        target_stop_gradient=True,
        learnable_lag_mixture=True,
    )
    optimizer = torch.optim.Adam([objective.lag_mixture_logits], lr=0.2)
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(query, target)
        loss.backward()
        optimizer.step()
    assert int(objective.lag_mixture_weights.argmax()) == 2
    assert float(objective.lag_mixture_weights[2].detach()) > 0.95


def test_rank_eight_coupling_and_private_classifier_have_shapes_and_gradients():
    torch.manual_seed(19)
    coupling = LowRankLagCouplingHead(
        input_dim=16, rank=8, allowed_lags=(0, 2, 5), num_classes=4
    )
    eeg = torch.randn(2, 5, 16, requires_grad=True)
    fnirs = torch.randn(2, 4, 16, requires_grad=True)
    eeg_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    fnirs_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    logits, pair_mask = coupling(
        eeg,
        fnirs,
        eeg_valid_mask=eeg_mask,
        fnirs_valid_mask=fnirs_mask,
        return_mask=True,
    )
    assert coupling.rank == 8
    assert coupling.allowed_lags == (0, 2, 5)
    assert logits.shape == (2, 5, 4, 4)
    assert pair_mask.shape == (2, 5, 4)
    expected_lag_mask = torch.tensor(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=torch.bool,
    )
    assert torch.equal(pair_mask[0], expected_lag_mask)
    assert not bool(pair_mask[:, 1, 0].any())  # negative lag is forbidden
    assert not bool(pair_mask[:, 0, 1].any())  # lag 1 is not configured
    (logits * pair_mask.unsqueeze(-1)).square().mean().backward()
    assert eeg.grad is not None and bool(eeg.grad.abs().sum() > 0)
    assert fnirs.grad is not None and bool(fnirs.grad.abs().sum() > 0)
    assert _has_nonzero_grad(coupling)

    classifier = PrivatePooledClassifier(
        eeg_private_dim=16, fnirs_private_dim=8, num_classes=4
    )
    eeg_private = torch.randn(2, 5, 16, requires_grad=True)
    fnirs_private = torch.randn(2, 4, 8, requires_grad=True)
    private_logits, pooled = classifier(
        eeg_private,
        fnirs_private,
        eeg_valid_mask=eeg_mask,
        fnirs_valid_mask=fnirs_mask,
        return_pooled=True,
    )
    assert private_logits.shape == (2, 4)
    assert pooled.shape == (2, 24)
    private_logits.square().mean().backward()
    assert eeg_private.grad is not None and bool(eeg_private.grad.abs().sum() > 0)
    assert fnirs_private.grad is not None and bool(fnirs_private.grad.abs().sum() > 0)


def test_masked_mean_is_nan_safe_and_rejects_admitted_nonfinite_values():
    values = torch.tensor(
        [
            [[2.0, 4.0], [float("nan"), float("inf")], [6.0, 8.0]],
            [[float("-inf"), float("nan")], [1.0, 3.0], [float("nan"), 5.0]],
        ]
    )
    mask = torch.tensor([[True, False, True], [False, False, False]])

    pooled = _masked_mean(values, mask)
    torch.testing.assert_close(pooled, torch.tensor([[4.0, 6.0], [0.0, 0.0]]))

    with pytest.raises(FloatingPointError, match="non-finite admitted"):
        _masked_mean(values, torch.ones_like(mask))


def test_masked_pair_mean_is_nan_safe_and_rejects_admitted_nonfinite_values():
    values = torch.tensor(
        [
            [
                [[2.0, 4.0], [float("nan"), float("inf")]],
                [[6.0, 8.0], [10.0, 12.0]],
            ],
            [
                [[float("-inf"), float("nan")], [float("nan"), 5.0]],
                [[float("inf"), 7.0], [8.0, float("nan")]],
            ],
        ]
    )
    mask = torch.tensor(
        [
            [[True, False], [False, False]],
            [[False, False], [False, False]],
        ]
    )

    pooled = _masked_pair_mean(values, mask)
    torch.testing.assert_close(pooled, torch.tensor([[2.0, 4.0], [0.0, 0.0]]))

    with pytest.raises(FloatingPointError, match="non-finite admitted"):
        _masked_pair_mean(values, torch.ones_like(mask))
