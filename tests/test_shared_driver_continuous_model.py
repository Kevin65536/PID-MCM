import inspect

import pytest
import torch

from src.tokenizers.shared_driver_semantic_vq import (
    SharedDriverContinuousModel,
)


def _small_model() -> SharedDriverContinuousModel:
    return SharedDriverContinuousModel(
        latent_dim=16,
        encoder_depth=1,
        encoder_num_heads=4,
        encoder_feedforward_dim=32,
        decoder_hidden_dim=24,
        dropout=0.0,
    )


def _inputs(batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.randn(batch_size, 6, 4000),
        torch.randn(batch_size, 2, 200),
    )


def _has_nonzero_gradient(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and bool(parameter.grad.abs().sum() > 0)
        for parameter in module.parameters()
    )


def test_continuous_model_public_contract_and_shapes():
    torch.manual_seed(21)
    model = _small_model().eval()
    eeg, fnirs = _inputs()

    with torch.no_grad():
        output = model(eeg, fnirs)

    assert tuple(inspect.signature(model.forward).parameters) == (
        "eeg",
        "fnirs",
        "eeg_token_valid_mask",
        "fnirs_token_valid_mask",
    )
    assert model.token_temporal_scope == "bidirectional_full_window"
    assert output["eeg_latent"].shape == (2, 10, 16)
    assert output["fnirs_latent"].shape == (2, 10, 16)
    assert output["eeg_decoded"].shape == (2, 10, 20)
    assert output["fnirs_decoded"].shape == (2, 10, 20)
    assert output["eeg_token_valid_mask"].shape == (2, 10)
    assert output["fnirs_token_valid_mask"].shape == (2, 10)
    assert output["eeg_token_valid_mask"].dtype == torch.bool
    assert output["fnirs_token_valid_mask"].all()


def test_changing_one_modality_cannot_change_the_other_latent_or_decode():
    torch.manual_seed(22)
    model = _small_model().eval()
    eeg, fnirs = _inputs(batch_size=1)

    with torch.no_grad():
        baseline = model(eeg, fnirs)
        changed_eeg = model(eeg + torch.randn_like(eeg) * 100.0, fnirs)
        changed_fnirs = model(eeg, fnirs + torch.randn_like(fnirs) * 100.0)

    assert torch.equal(
        baseline["fnirs_latent"], changed_eeg["fnirs_latent"]
    )
    assert torch.equal(
        baseline["fnirs_decoded"], changed_eeg["fnirs_decoded"]
    )
    assert torch.equal(baseline["eeg_latent"], changed_fnirs["eeg_latent"])
    assert torch.equal(baseline["eeg_decoded"], changed_fnirs["eeg_decoded"])


@pytest.mark.parametrize(
    ("modality", "patch_samples"),
    (("eeg", 400), ("fnirs", 20)),
)
def test_invalid_patch_cannot_act_as_attention_key_or_value(
    modality: str,
    patch_samples: int,
):
    torch.manual_seed(23)
    model = _small_model().eval()
    eeg, fnirs = _inputs(batch_size=1)
    mask = torch.ones(1, 10, dtype=torch.bool)
    invalid_index = 4
    mask[:, invalid_index] = False

    changed_eeg = eeg.clone()
    changed_fnirs = fnirs.clone()
    signal = changed_eeg if modality == "eeg" else changed_fnirs
    start = invalid_index * patch_samples
    signal[..., start : start + patch_samples] = float("nan")

    kwargs = {
        "eeg_token_valid_mask": mask if modality == "eeg" else None,
        "fnirs_token_valid_mask": mask if modality == "fnirs" else None,
    }
    with torch.no_grad():
        baseline = model(eeg, fnirs, **kwargs)
        changed = model(changed_eeg, changed_fnirs, **kwargs)

    valid = mask[0]
    latent_key = f"{modality}_latent"
    decoded_key = f"{modality}_decoded"
    torch.testing.assert_close(
        baseline[latent_key][0, valid],
        changed[latent_key][0, valid],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        baseline[decoded_key][0, valid],
        changed[decoded_key][0, valid],
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(
        changed[latent_key][0, invalid_index],
        torch.zeros_like(changed[latent_key][0, invalid_index]),
    )
    assert torch.equal(
        changed[decoded_key][0, invalid_index],
        torch.zeros_like(changed[decoded_key][0, invalid_index]),
    )
    assert torch.isfinite(changed[latent_key]).all()
    assert torch.isfinite(changed[decoded_key]).all()


def test_all_invalid_rows_bypass_attention_and_return_finite_zeros():
    torch.manual_seed(24)
    model = _small_model().eval()
    eeg, fnirs = _inputs(batch_size=2)
    eeg_mask = torch.ones(2, 10, dtype=torch.bool)
    eeg_mask[0] = False
    fnirs_mask = torch.zeros(2, 10, dtype=torch.bool)

    with torch.no_grad():
        output = model(
            eeg,
            fnirs,
            eeg_token_valid_mask=eeg_mask,
            fnirs_token_valid_mask=fnirs_mask,
        )

    assert torch.isfinite(output["eeg_latent"]).all()
    assert torch.isfinite(output["fnirs_latent"]).all()
    assert torch.equal(
        output["eeg_latent"][0], torch.zeros_like(output["eeg_latent"][0])
    )
    assert torch.equal(
        output["eeg_decoded"][0], torch.zeros_like(output["eeg_decoded"][0])
    )
    assert torch.equal(
        output["fnirs_latent"], torch.zeros_like(output["fnirs_latent"])
    )
    assert torch.equal(
        output["fnirs_decoded"], torch.zeros_like(output["fnirs_decoded"])
    )


def test_shared_decoder_is_modality_agnostic_and_gradient_isolation_holds():
    torch.manual_seed(25)
    model = _small_model().train()
    eeg, fnirs = _inputs(batch_size=1)
    output = model(eeg, fnirs)

    assert tuple(inspect.signature(model.driver_decoder.forward).parameters) == (
        "latent",
    )
    torch.testing.assert_close(
        output["eeg_decoded"],
        model.driver_decoder(output["eeg_latent"]),
    )
    torch.testing.assert_close(
        output["fnirs_decoded"],
        model.driver_decoder(output["fnirs_latent"]),
    )

    output["eeg_decoded"].square().mean().backward()
    assert _has_nonzero_gradient(model.eeg_encoder)
    assert _has_nonzero_gradient(model.driver_decoder)
    assert all(
        parameter.grad is None for parameter in model.fnirs_encoder.parameters()
    )

    model.zero_grad(set_to_none=True)
    output = model(eeg, fnirs)
    output["fnirs_decoded"].square().mean().backward()
    assert _has_nonzero_gradient(model.fnirs_encoder)
    assert _has_nonzero_gradient(model.driver_decoder)
    assert all(
        parameter.grad is None for parameter in model.eeg_encoder.parameters()
    )


def test_invalid_raw_patch_has_zero_gradient_under_valid_only_output():
    torch.manual_seed(26)
    model = _small_model().eval()
    eeg, fnirs = _inputs(batch_size=1)
    eeg.requires_grad_(True)
    mask = torch.ones(1, 10, dtype=torch.bool)
    mask[:, 7] = False

    output = model(eeg, fnirs, eeg_token_valid_mask=mask)
    output["eeg_decoded"].square().sum().backward()

    invalid_gradient = eeg.grad[..., 7 * 400 : 8 * 400]
    valid_gradient = torch.cat(
        (eeg.grad[..., : 7 * 400], eeg.grad[..., 8 * 400 :]),
        dim=-1,
    )
    assert torch.equal(invalid_gradient, torch.zeros_like(invalid_gradient))
    assert bool(valid_gradient.abs().sum() > 0)


def test_signal_and_mask_shape_contracts_fail_closed():
    model = _small_model()
    eeg, fnirs = _inputs(batch_size=1)

    with pytest.raises(ValueError, match="exactly 4000"):
        model(eeg[..., :-1], fnirs)
    with pytest.raises(ValueError, match="token_valid_mask"):
        model(eeg, fnirs, eeg_token_valid_mask=torch.ones(1, 9, dtype=torch.bool))
