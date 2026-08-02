from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import signal
import torch
import torch.nn.functional as F

from comparative_methods.NormWear.adapters.normwear import (
    NormWearFrozenEncoder,
    NormWearLinearProbe,
    load_verified_normwear_encoder,
    resample_polyphase,
)
from comparative_methods.NormWear.audit_adapter_smoke_v2 import write_json


METHOD_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def loaded() -> tuple[torch.nn.Module, object, object, torch.device]:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the NormWear executable adapter tests")
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    backbone, upstream_module, metadata = load_verified_normwear_encoder(device=device)
    return backbone, upstream_module, metadata, device


def test_polyphase_resampling_is_exactly_declared_and_deterministic() -> None:
    eeg = torch.arange(2 * 3 * 1600, dtype=torch.float32).reshape(2, 3, 1600)
    fnirs = torch.arange(2 * 4 * 80, dtype=torch.float32).reshape(2, 4, 80)
    eeg_actual = resample_polyphase(eeg, source_rate_hz=200.0)
    fnirs_actual = resample_polyphase(fnirs, source_rate_hz=10.0)
    eeg_expected = signal.resample_poly(
        eeg.numpy(), 13, 40, axis=-1, window=("kaiser", 5.0), padtype="constant"
    )
    fnirs_expected = signal.resample_poly(
        fnirs.numpy(), 13, 2, axis=-1, window=("kaiser", 5.0), padtype="constant"
    )
    assert eeg_actual.shape == (2, 3, 520)
    assert fnirs_actual.shape == (2, 4, 520)
    np.testing.assert_array_equal(eeg_actual.numpy(), eeg_expected)
    np.testing.assert_array_equal(fnirs_actual.numpy(), fnirs_expected)
    assert torch.equal(eeg_actual, resample_polyphase(eeg, source_rate_hz=200.0))


def test_verified_encoder_excludes_decoder_and_is_frozen(loaded: tuple) -> None:
    backbone, _, metadata, _ = loaded
    assert metadata.sha256 == (
        "36d0bca18356ccfc8e8916058bf838f26f1212a646f5780b487ad78581a92561"
    )
    assert metadata.encoder_entry_count == 222
    assert metadata.excluded_decoder_entry_count == 39
    assert metadata.encoder_parameter_count == 128_118_528
    assert metadata.embedding_dim_per_channel == 768
    assert not hasattr(backbone, "decoder_blocks")
    assert backbone.training is False
    assert not any(parameter.requires_grad for parameter in backbone.parameters())


def test_chunked_encoder_matches_pinned_upstream_execution(loaded: tuple) -> None:
    backbone, upstream_module, _, device = loaded
    chunked_adapter = NormWearFrozenEncoder(backbone, upstream_module, channel_chunk_size=1)
    unchunked_adapter = NormWearFrozenEncoder(backbone, upstream_module, channel_chunk_size=3)
    generator = torch.Generator(device=device).manual_seed(19)
    eeg = torch.randn(1, 1, 400, generator=generator, device=device)
    hbo = torch.randn(1, 1, 20, generator=generator, device=device)
    hbr = torch.randn(1, 1, 20, generator=generator, device=device)
    with torch.no_grad():
        model_input, _ = chunked_adapter.prepare_model_input(
            eeg,
            hbo,
            hbr,
            eeg_sampling_rate_hz=200.0,
            fnirs_sampling_rate_hz=10.0,
            eeg_channel_names=("E1",),
            fnirs_location_names=("L1",),
        )
        cwt = chunked_adapter.calculate_cwt(model_input)
        expected = backbone.get_signal_embedding(cwt, hidden_out=False, device=device)
        unchunked = unchunked_adapter.encode_cwt(cwt)
        chunked = chunked_adapter.encode_cwt(cwt)
    assert chunked.shape == expected.shape
    torch.testing.assert_close(unchunked, expected, rtol=0.0, atol=0.0)
    token_difference = (chunked - expected).abs()
    assert float(token_difference.max()) < 0.01
    assert float(token_difference.mean()) < 2e-4
    expected_features = expected.mean(dim=2).flatten(start_dim=1)
    chunked_features = chunked.mean(dim=2).flatten(start_dim=1)
    assert float((chunked_features - expected_features).abs().max()) < 2e-4
    similarity = F.cosine_similarity(chunked_features, expected_features)
    assert bool((similarity > 0.99999).all())


def test_adapter_rejects_wrong_rates_identity_and_missing_support(loaded: tuple) -> None:
    backbone, upstream_module, _, device = loaded
    adapter = NormWearFrozenEncoder(backbone, upstream_module)
    eeg = torch.randn(1, 1, 400, device=device)
    hbo = torch.randn(1, 1, 20, device=device)
    hbr = torch.randn(1, 1, 20, device=device)
    kwargs = {
        "eeg_sampling_rate_hz": 200.0,
        "fnirs_sampling_rate_hz": 10.0,
        "eeg_channel_names": ("E1",),
        "fnirs_location_names": ("L1",),
    }
    with pytest.raises(ValueError, match="unsupported frozen NormWear rate"):
        adapter.prepare_model_input(eeg, hbo, hbr, **{**kwargs, "eeg_sampling_rate_hz": 100.0})
    with pytest.raises(ValueError, match="identities"):
        adapter.prepare_model_input(eeg, hbo, hbr, **{**kwargs, "eeg_channel_names": ()})
    with pytest.raises(ValueError, match="missing or padded channel"):
        adapter.prepare_model_input(
            eeg,
            hbo,
            hbr,
            **kwargs,
            channel_valid=torch.zeros(1, 3, dtype=torch.bool, device=device),
        )
    with pytest.raises(ValueError, match="missing or padded EEG"):
        adapter.prepare_model_input(
            eeg,
            hbo,
            hbr,
            **kwargs,
            eeg_sample_valid=torch.zeros(1, 400, dtype=torch.bool, device=device),
        )


def test_frozen_probe_updates_only_head_and_head_reloads(loaded: tuple) -> None:
    backbone, upstream_module, _, device = loaded
    encoder = NormWearFrozenEncoder(backbone, upstream_module, channel_chunk_size=2)
    probe = NormWearLinearProbe(encoder, channel_count=3, output_dim=2).to(device)
    probe.train()
    assert probe.frozen_encoder.backbone.training is False
    assert {name for name, value in probe.named_parameters() if value.requires_grad} == {
        "head.weight",
        "head.bias",
    }
    generator = torch.Generator(device=device).manual_seed(23)
    eeg = torch.randn(1, 1, 400, generator=generator, device=device)
    hbo = torch.randn(1, 1, 20, generator=generator, device=device)
    hbr = torch.randn(1, 1, 20, generator=generator, device=device)
    kwargs = {
        "eeg_sampling_rate_hz": 200.0,
        "fnirs_sampling_rate_hz": 10.0,
        "eeg_channel_names": ("E1",),
        "fnirs_location_names": ("L1",),
    }
    optimizer = torch.optim.AdamW(probe.head.parameters(), lr=1e-3)
    initial = probe.head.weight.detach().clone()
    logits = probe(eeg, hbo, hbr, **kwargs)
    loss = F.cross_entropy(logits, torch.tensor([1], device=device))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()
    assert probe.head.weight.grad is not None
    assert not torch.equal(initial, probe.head.weight.detach())
    assert not any(parameter.grad is not None for parameter in backbone.parameters())

    buffer = BytesIO()
    torch.save(probe.head.state_dict(), buffer)
    buffer.seek(0)
    reloaded = torch.nn.Linear(3 * 768, 2).to(device)
    reloaded.load_state_dict(torch.load(buffer, map_location=device, weights_only=True))
    torch.testing.assert_close(reloaded.weight, probe.head.weight)
    torch.testing.assert_close(reloaded.bias, probe.head.bias)


def test_smoke_evidence_writer_refuses_protected_path(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="protected NormWear smoke"):
        write_json(tmp_path / "protected" / "summary.json", {"status": "forbidden"})


def test_retained_adapter_smoke_passes_frozen_bounds() -> None:
    report = json.loads(
        (METHOD_ROOT / "evidence/adapter_smoke_v2/summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "pass"
    assert report["execution"]["upstream_unchunked_bitwise_exact"] is True
    metrics = report["execution"]["chunked_numerical_metrics"]
    thresholds = report["execution"]["chunked_numerical_thresholds"]
    assert metrics["maximum_token_absolute_difference"] <= thresholds[
        "maximum_token_absolute_difference"
    ]
    assert metrics["mean_token_absolute_difference"] <= thresholds[
        "maximum_mean_token_absolute_difference"
    ]
    assert metrics["maximum_pooled_feature_absolute_difference"] <= thresholds[
        "maximum_pooled_feature_absolute_difference"
    ]
    assert metrics["pooled_feature_cosine_similarity"] >= thresholds[
        "minimum_pooled_feature_cosine_similarity"
    ]
    assert report["public_smoke"]["dsr"]["feature_shape"] == [1, 76_800]
    assert report["public_smoke"]["motor_imagery"]["feature_shape"] == [1, 78_336]
    assert report["public_smoke"]["dsr"]["bitwise_replay_exact"] is True
    assert report["public_smoke"]["motor_imagery"]["bitwise_replay_exact"] is True
    assert report["linear_probe"]["encoder_gradient_count"] == 0
    assert report["protected_test_opened"] is False
