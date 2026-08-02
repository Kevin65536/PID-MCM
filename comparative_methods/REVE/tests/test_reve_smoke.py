from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from comparative_methods.REVE.adapters.reve import (
    REVEFrozenEncoder,
    REVELinearProbe,
    load_verified_reve_base,
)


METHOD_ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ("F3", "F4", "C3", "C4")


def _require_local_assets() -> None:
    required = [
        METHOD_ROOT / "checkpoints/reve-base/model.safetensors",
        METHOD_ROOT / "checkpoints/reve-base/modeling_reve.py",
        METHOD_ROOT / "checkpoints/reve-positions/model.safetensors",
        METHOD_ROOT / "checkpoints/reve-positions/position_bank.py",
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("gated REVE-base and position snapshots are not available locally")


def test_base_and_position_bank_are_hash_verified_and_locally_loadable() -> None:
    _require_local_assets()
    encoder, position_bank, metadata = load_verified_reve_base()
    assert metadata.artifact_id == "reve_base"
    assert metadata.patch_samples == 200
    assert metadata.patch_overlap == 20
    assert metadata.embedding_dim == 512
    assert metadata.position_artifact_id == "reve_positions"
    assert metadata.position_bank_size == 543
    assert len(metadata.position_sha256) == 64
    assert len(metadata.position_source_revision) == 40
    assert len(metadata.upstream_code_revision) == 40
    assert metadata.representation_layer == (
        "final_transformer_latent_tokens_after_identity_final_layer"
    )
    assert metadata.pooling == "frozen_pretrained_cls_query_attention_pooling"
    assert len(metadata.sha256) == 64
    assert metadata.path.is_file()
    assert metadata.position_path.is_file()
    assert all(name in position_bank.mapping for name in CHANNELS)
    assert position_bank.embedding.shape == (543, 3)
    assert encoder.cls_query_token.shape == (1, 1, 512)
    assert isinstance(encoder.final_layer, torch.nn.Identity)
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    assert not any(parameter.requires_grad for parameter in position_bank.parameters())
    assert encoder.training is False
    assert position_bank.training is False


def test_adapter_rejects_unknown_coordinates_wrong_rate_and_missing_support() -> None:
    _require_local_assets()
    encoder, position_bank, _ = load_verified_reve_base()
    adapter = REVEFrozenEncoder(encoder, position_bank)
    eeg = torch.randn(2, 4, 200)

    with pytest.raises(ValueError, match="requires 200 Hz"):
        adapter(eeg, sampling_rate_hz=100.0, channel_names=CHANNELS)
    with pytest.raises(ValueError, match="no coordinates"):
        adapter(
            eeg,
            sampling_rate_hz=200.0,
            channel_names=("F3", "F4", "C3", "UNKNOWN"),
        )
    with pytest.raises(ValueError, match="missing or padded channels"):
        adapter(
            eeg,
            sampling_rate_hz=200.0,
            channel_names=CHANNELS,
            channel_valid=torch.zeros(2, 4, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="missing or padded samples"):
        adapter(
            eeg,
            sampling_rate_hz=200.0,
            channel_names=CHANNELS,
            sample_valid=torch.zeros(2, 200, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="only frozen_pretrained"):
        REVEFrozenEncoder(encoder, position_bank, pooling="mean")


def test_adapter_matches_frozen_pretrained_query_pooling_exactly() -> None:
    _require_local_assets()
    encoder, position_bank, _ = load_verified_reve_base()
    adapter = REVEFrozenEncoder(encoder, position_bank)
    eeg = torch.randn(2, 4, 380)
    with torch.no_grad():
        indices = torch.tensor([position_bank.mapping[name] for name in CHANNELS])
        positions = position_bank.embedding[indices].unsqueeze(0).expand(2, -1, -1)
        tokens = encoder(eeg=eeg, pos=positions)
        flattened = tokens.flatten(1, 2)
        query = encoder.cls_query_token.expand(2, -1, -1)
        weights = torch.softmax(
            torch.matmul(query, flattened.transpose(-1, -2)) / (512**0.5),
            dim=-1,
        )
        expected = torch.matmul(weights, flattened).squeeze(1)
        actual = adapter(eeg, sampling_rate_hz=200.0, channel_names=CHANNELS)
    assert tokens.shape == (2, 4, 2, 512)
    torch.testing.assert_close(actual, expected)


def test_gpu_frozen_probe_forward_backward_optimizer_and_reload() -> None:
    _require_local_assets()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the REVE GPU smoke")
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    encoder, position_bank, _ = load_verified_reve_base(device=device)
    probe = REVELinearProbe(REVEFrozenEncoder(encoder, position_bank), output_dim=4).to(device)
    probe.train()
    assert probe.frozen_encoder.encoder.training is False
    assert probe.frozen_encoder.position_bank.training is False
    assert probe.frozen_encoder.encoder.cls_query_token.requires_grad is False
    assert {
        name for name, parameter in probe.named_parameters() if parameter.requires_grad
    } == {"head.weight", "head.bias"}

    eeg = torch.randn(2, 4, 200, device=device)
    target = torch.tensor([0, 1], device=device)
    optimizer = torch.optim.AdamW(probe.head.parameters(), lr=1e-3)
    initial_head = probe.head.weight.detach().clone()
    logits = probe(eeg, sampling_rate_hz=200.0, channel_names=CHANNELS)
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
    reloaded_encoder, reloaded_bank, _ = load_verified_reve_base(device=device)
    reloaded = REVELinearProbe(
        REVEFrozenEncoder(reloaded_encoder, reloaded_bank),
        output_dim=4,
    ).to(device)
    reloaded.load_state_dict(reloaded_state, strict=True)
    probe.eval()
    reloaded.eval()
    with torch.no_grad():
        expected = probe(eeg, sampling_rate_hz=200.0, channel_names=CHANNELS)
        actual = reloaded(eeg, sampling_rate_hz=200.0, channel_names=CHANNELS)
    torch.testing.assert_close(actual, expected)
