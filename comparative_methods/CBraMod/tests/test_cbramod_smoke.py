from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from comparative_methods.CBraMod.adapters.cbramod import (
    CBraModFrozenEncoder,
    CBraModLinearProbe,
    load_verified_cbramod_encoder,
)


METHOD_ROOT = Path(__file__).resolve().parents[1]


def _require_local_assets() -> None:
    if not (METHOD_ROOT / "upstream/models/cbramod.py").is_file():
        pytest.skip("pinned CBraMod upstream checkout is not available locally")
    if not (METHOD_ROOT / "checkpoints/pretrained_weights.pth").is_file():
        pytest.skip("official CBraMod checkpoint is not available locally")


def test_official_checkpoint_is_hash_verified_and_strictly_loadable() -> None:
    _require_local_assets()
    encoder, metadata = load_verified_cbramod_encoder()
    assert metadata.patch_samples == 200
    assert metadata.embedding_dim == 200
    assert metadata.representation_layer == "encoder_latent_before_pretraining_proj_out"
    assert len(metadata.sha256) == 64
    assert isinstance(encoder.proj_out, torch.nn.Identity)
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    assert encoder.training is False


def test_adapter_rejects_wrong_rate_nonpatch_window_and_missing_support() -> None:
    _require_local_assets()
    encoder, _ = load_verified_cbramod_encoder()
    adapter = CBraModFrozenEncoder(encoder)
    eeg = torch.randn(2, 18, 400)
    names = tuple(f"EEG{index:02d}" for index in range(18))

    with pytest.raises(ValueError, match="requires 200 Hz"):
        adapter(eeg, sampling_rate_hz=100.0, channel_names=names)
    with pytest.raises(ValueError, match="positive multiple of 200"):
        adapter(eeg[..., :-1], sampling_rate_hz=200.0, channel_names=names)
    with pytest.raises(ValueError, match="missing or padded channels"):
        adapter(
            eeg,
            sampling_rate_hz=200.0,
            channel_names=names,
            channel_valid=torch.zeros(2, 18, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="missing or padded samples"):
        adapter(
            eeg,
            sampling_rate_hz=200.0,
            channel_names=names,
            sample_valid=torch.zeros(2, 400, dtype=torch.bool),
        )


def test_adapter_rejects_pretraining_reconstruction_projection() -> None:
    _require_local_assets()
    encoder, _ = load_verified_cbramod_encoder()
    encoder.proj_out = torch.nn.Linear(200, 200)
    with pytest.raises(ValueError, match="proj_out=Identity"):
        CBraModFrozenEncoder(encoder)


def test_gpu_frozen_probe_forward_backward_optimizer_and_reload() -> None:
    _require_local_assets()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the CBraMod GPU smoke")
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    encoder, _ = load_verified_cbramod_encoder(device=device)
    probe = CBraModLinearProbe(CBraModFrozenEncoder(encoder), output_dim=4).to(device)
    probe.train()
    assert probe.frozen_encoder.encoder.training is False

    eeg = torch.randn(2, 18, 400, device=device)
    target = torch.tensor([0, 1], device=device)
    names = tuple(f"EEG{index:02d}" for index in range(18))
    optimizer = torch.optim.AdamW(probe.head.parameters(), lr=1e-3)
    initial_head = probe.head.weight.detach().clone()
    with torch.no_grad():
        latent_tokens = encoder(eeg.reshape(2, 18, 2, 200))
        expected_embedding = latent_tokens.mean(dim=(1, 2))
        actual_embedding = probe.frozen_encoder(
            eeg,
            sampling_rate_hz=200.0,
            channel_names=names,
        )
    torch.testing.assert_close(actual_embedding, expected_embedding)
    logits = probe(eeg, sampling_rate_hz=200.0, channel_names=names)
    loss = F.cross_entropy(logits, target)
    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert probe.head.weight.grad is not None
    assert torch.isfinite(probe.head.weight.grad).all()
    assert not any(parameter.grad is not None for parameter in encoder.parameters())
    optimizer.step()
    assert not torch.equal(initial_head, probe.head.weight.detach())

    buffer = BytesIO()
    torch.save(probe.state_dict(), buffer)
    buffer.seek(0)
    reloaded_state = torch.load(buffer, map_location=device, weights_only=True)
    reloaded_encoder, _ = load_verified_cbramod_encoder(device=device)
    reloaded = CBraModLinearProbe(CBraModFrozenEncoder(reloaded_encoder), output_dim=4).to(device)
    reloaded.load_state_dict(reloaded_state, strict=True)
    probe.eval()
    reloaded.eval()
    with torch.no_grad():
        expected = probe(eeg, sampling_rate_hz=200.0, channel_names=names)
        actual = reloaded(eeg, sampling_rate_hz=200.0, channel_names=names)
    torch.testing.assert_close(actual, expected)
