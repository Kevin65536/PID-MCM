import copy

import torch

from src.tokenizers.ema_vector_quantizer import EMAVectorQuantizer


def test_quantizer_zero_assignment_code_does_not_move():
    quantizer = EMAVectorQuantizer(codebook_size=4, embedding_dim=2, decay=0.5)
    with torch.no_grad():
        quantizer.codebook.copy_(torch.tensor([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]))
        quantizer.ema_sum.copy_(quantizer.codebook)
    before = quantizer.codebook.clone()
    quantizer.train()
    quantizer(torch.zeros(2, 3, 2))
    assert torch.equal(quantizer.codebook[1:], before[1:])


def test_quantizer_count_and_sum_ema_converges_to_centroid():
    quantizer = EMAVectorQuantizer(codebook_size=1, embedding_dim=2, decay=0.5)
    quantizer.train()
    target = torch.tensor([[[2.0, -1.0], [2.0, -1.0]]])
    for _ in range(12):
        quantizer(target)
    assert torch.allclose(quantizer.codebook[0], torch.tensor([2.0, -1.0]), atol=1e-3)


def test_quantizer_outputs_and_reload_are_identical_in_eval():
    torch.manual_seed(7)
    quantizer = EMAVectorQuantizer(codebook_size=8, embedding_dim=3)
    latent = torch.randn(2, 5, 3)
    quantizer.train()
    quantizer(latent)
    quantizer.eval()
    expected = quantizer(latent)

    restored = EMAVectorQuantizer(codebook_size=8, embedding_dim=3)
    restored.load_state_dict(copy.deepcopy(quantizer.state_dict()))
    restored.eval()
    observed = restored(latent)
    assert torch.equal(observed.hard_ids, expected.hard_ids)
    assert torch.equal(observed.logits, expected.logits)
    assert torch.equal(observed.posterior, expected.posterior)
    assert torch.equal(observed.expected_embedding, expected.expected_embedding)
    assert torch.equal(observed.hard_ids, observed.posterior.argmax(dim=-1))


def test_quantizer_dead_code_revival_is_explicit_and_counted():
    quantizer = EMAVectorQuantizer(
        codebook_size=4,
        embedding_dim=2,
        revive_dead_codes=True,
        revival_warmup_steps=1,
        revival_interval=1,
        dead_code_threshold=0.1,
    )
    quantizer.train()
    output = quantizer(torch.tensor([[[1.0, 1.0], [1.1, 1.1]]]))
    assert output.health["revived_codes"].item() > 0
    assert quantizer.revival_count.item() == output.health["revived_codes"].item()


def test_quantizer_device_and_state_buffers_follow_module():
    quantizer = EMAVectorQuantizer(codebook_size=4, embedding_dim=2).to(dtype=torch.float64)
    output = quantizer.eval()(torch.randn(1, 2, 2, dtype=torch.float64))
    assert output.quantized.dtype == torch.float64
    assert quantizer.ema_sum.dtype == torch.float64


def test_quantizer_reduces_assignment_statistics_when_distributed(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fake_all_reduce(value, op):
        calls.append((value.shape, op))
        value.mul_(2.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    quantizer = EMAVectorQuantizer(codebook_size=2, embedding_dim=2, decay=0.0)
    quantizer.train()
    quantizer(torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]))
    assert len(calls) == 2
    assert sorted(shape for shape, _ in calls) == [(2,), (2, 2)]
