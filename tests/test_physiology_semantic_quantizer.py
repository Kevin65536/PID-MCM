import copy

import pytest
import torch

from src.tokenizers.ema_vector_quantizer import EMAVectorQuantizer


def test_quantizer_zero_assignment_code_does_not_move():
    quantizer = EMAVectorQuantizer(codebook_size=4, embedding_dim=2, decay=0.5)
    with torch.no_grad():
        quantizer.codebook.copy_(torch.tensor([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]))
        quantizer.ema_count.fill_(2.0)
        quantizer.ema_sum.copy_(quantizer.codebook * quantizer.ema_count.unsqueeze(-1))
    before = quantizer.codebook.clone()
    quantizer.train()
    quantizer(torch.zeros(2, 3, 2))
    assert torch.equal(quantizer.codebook[1:], before[1:])
    assert torch.equal(quantizer.ema_count[1:], torch.ones(3))
    assert torch.equal(
        quantizer.ema_sum[1:],
        before[1:] * quantizer.ema_count[1:].unsqueeze(-1),
    )


def test_quantizer_count_and_sum_ema_converges_to_centroid():
    quantizer = EMAVectorQuantizer(codebook_size=1, embedding_dim=2, decay=0.5)
    quantizer.train()
    target = torch.tensor([[[2.0, -1.0], [2.0, -1.0]]])
    for _ in range(12):
        quantizer(target)
    assert torch.allclose(quantizer.codebook[0], torch.tensor([2.0, -1.0]), atol=1e-3)


def test_quantizer_first_assignment_has_no_unmatched_random_prior_mass():
    quantizer = EMAVectorQuantizer(codebook_size=2, embedding_dim=2, decay=0.99)
    with torch.no_grad():
        quantizer.codebook.copy_(torch.tensor([[0.0, 0.0], [20.0, 20.0]]))
    quantizer.train()

    quantizer(torch.tensor([[[2.0, -1.0], [2.0, -1.0]]]))

    assert torch.allclose(quantizer.codebook[0], torch.tensor([2.0, -1.0]))
    assert torch.equal(quantizer.codebook[1], torch.tensor([20.0, 20.0]))


def test_kmeans_initializes_codebook_and_matching_ema_state_from_valid_latents():
    torch.manual_seed(17)
    quantizer = EMAVectorQuantizer(
        codebook_size=2,
        embedding_dim=2,
        decay=0.5,
        kmeans_init=True,
        kmeans_iters=5,
    ).train()
    latent = torch.tensor(
        [[[-10.0, -10.0], [-9.0, -11.0], [10.0, 10.0], [9.0, 11.0]]]
    )

    quantizer(latent)

    expected = torch.tensor([[-9.5, -10.5], [9.5, 10.5]])
    observed = quantizer.codebook[quantizer.codebook[:, 0].argsort()]
    assert quantizer.initialized.item()
    assert torch.allclose(observed, expected)
    assert torch.all(quantizer.ema_count > 0)
    assert torch.allclose(
        quantizer.ema_sum,
        quantizer.codebook * quantizer.ema_count.unsqueeze(-1),
    )


def test_kmeans_initialization_waits_for_training_and_round_trips():
    torch.manual_seed(18)
    latent = torch.randn(2, 8, 3)
    quantizer = EMAVectorQuantizer(
        codebook_size=4, embedding_dim=3, kmeans_init=True
    )
    before = quantizer.codebook.clone()
    quantizer.eval()(latent)
    assert not quantizer.initialized.item()
    assert torch.equal(quantizer.codebook, before)

    quantizer.train()(latent)
    restored = EMAVectorQuantizer(
        codebook_size=4, embedding_dim=3, kmeans_init=True
    )
    restored.load_state_dict(copy.deepcopy(quantizer.state_dict()))
    assert restored.initialized.item()
    assert torch.equal(restored.codebook, quantizer.codebook)


def test_cosine_assignment_is_invariant_to_positive_latent_and_codebook_scale():
    quantizer = EMAVectorQuantizer(
        codebook_size=3,
        embedding_dim=2,
        assignment="cosine",
    ).eval()
    with torch.no_grad():
        quantizer.codebook.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0]])
        )
        latent = torch.tensor([[[2.0, 1.0], [-1.0, 4.0]]])
        baseline = quantizer(latent)
        quantizer.codebook.mul_(7.0)
        scaled = quantizer(latent * 11.0)

    assert torch.allclose(scaled.logits, baseline.logits, atol=1.0e-6)
    assert torch.equal(scaled.hard_ids, baseline.hard_ids)


def test_quantization_strength_anneals_hard_path_without_changing_hard_ids():
    quantizer = EMAVectorQuantizer(codebook_size=2, embedding_dim=2).eval()
    latent = torch.tensor([[[2.0, -1.0], [0.5, 3.0]]])
    with torch.no_grad():
        quantizer.codebook.copy_(torch.tensor([[2.0, -1.0], [0.0, 4.0]]))
    quantizer.set_quantization_strength(0.0)
    continuous = quantizer(latent)
    quantizer.set_quantization_strength(1.0)
    hard = quantizer(latent)

    assert torch.equal(continuous.hard_ids, hard.hard_ids)
    assert torch.equal(continuous.quantized, hard.quantized)
    assert torch.equal(continuous.annealed_quantized, latent)
    assert torch.equal(hard.annealed_quantized, hard.quantized)
    assert continuous.commitment_loss.item() == 0.0
    assert hard.commitment_loss.item() > 0.0


def test_quantization_strength_rejects_out_of_range_values():
    quantizer = EMAVectorQuantizer(codebook_size=2, embedding_dim=2)
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        quantizer.set_quantization_strength(1.1)


def test_normalized_latent_warmup_removes_magnitude_channel_and_keeps_unit_codebook():
    quantizer = EMAVectorQuantizer(
        codebook_size=2,
        embedding_dim=2,
        assignment="cosine",
        normalize_latents=True,
        decay=0.0,
    ).train()
    with torch.no_grad():
        quantizer.codebook.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    latent = torch.tensor([[[10.0, 0.0], [0.0, 3.0]]])
    quantizer.set_quantization_strength(0.0)
    output = quantizer(latent)

    assert torch.allclose(
        output.annealed_quantized,
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
    )
    assert torch.allclose(quantizer.codebook.norm(dim=-1), torch.ones(2))
    assert output.commitment_loss.item() == 0.0


def test_normalized_cosine_kmeans_initializes_unit_centroids():
    torch.manual_seed(19)
    quantizer = EMAVectorQuantizer(
        codebook_size=3,
        embedding_dim=2,
        assignment="cosine",
        normalize_latents=True,
        kmeans_init=True,
    ).train()
    quantizer(torch.tensor([[[10.0, 0.0], [8.0, 1.0], [0.0, 4.0], [-3.0, 0.0]]]))

    assert torch.allclose(quantizer.codebook.norm(dim=-1), torch.ones(3), atol=1e-6)


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


def test_quantizer_mask_excludes_invalid_tokens_from_ema_and_health():
    quantizer = EMAVectorQuantizer(codebook_size=1, embedding_dim=2, decay=0.0)
    quantizer.train()
    output = quantizer(
        torch.tensor([[[2.0, -1.0], [1000.0, 1000.0]]]),
        valid_mask=torch.tensor([[True, False]]),
    )
    assert torch.allclose(quantizer.codebook[0], torch.tensor([2.0, -1.0]))
    assert output.health["batch_active_codes"].item() == 1


def test_quantizer_rejects_mismatched_validity_mask():
    quantizer = EMAVectorQuantizer(codebook_size=2, embedding_dim=2)
    with torch.no_grad(), pytest.raises(ValueError, match="valid_mask shape"):
        quantizer(torch.zeros(1, 3, 2), valid_mask=torch.ones(1, 2, dtype=torch.bool))


def test_training_forward_uses_pre_update_codebook_for_backward():
    quantizer = EMAVectorQuantizer(codebook_size=4, embedding_dim=2, decay=0.99).train()
    latent = torch.randn(2, 3, 2, requires_grad=True)

    output = quantizer(latent)
    (output.expected_embedding.square().mean() + output.commitment_loss).backward()

    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()


def test_dead_code_revival_is_bounded_and_uses_highest_error_latent():
    quantizer = EMAVectorQuantizer(
        codebook_size=3,
        embedding_dim=2,
        decay=0.0,
        revive_dead_codes=True,
        revival_warmup_steps=1,
        revival_interval=1,
        max_revivals_per_event=1,
    )
    with torch.no_grad():
        quantizer.codebook.copy_(
            torch.tensor([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        )
    quantizer.train()

    output = quantizer(
        torch.tensor([[[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]]])
    )

    assert output.health["revived_codes"].item() == 1
    assert torch.equal(quantizer.codebook[1], torch.tensor([5.0, 0.0]))


def test_revival_casts_amp_latents_to_codebook_dtype():
    quantizer = EMAVectorQuantizer(
        codebook_size=3,
        embedding_dim=2,
        decay=0.0,
        revive_dead_codes=True,
        revival_warmup_steps=1,
        revival_interval=1,
        max_revivals_per_event=1,
    )
    with torch.no_grad():
        quantizer.codebook.copy_(
            torch.tensor([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        )
        revived = quantizer._ema_update(
            torch.tensor([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]], dtype=torch.float16),
            torch.tensor([0, 0, 0]),
        )

    assert revived.item() == 1
    assert quantizer.codebook.dtype == torch.float32
    assert torch.equal(quantizer.codebook[1], torch.tensor([5.0, 0.0]))


def test_normalized_revival_noise_preserves_unit_sphere_and_is_counted():
    torch.manual_seed(23)
    quantizer = EMAVectorQuantizer(
        codebook_size=3,
        embedding_dim=2,
        decay=0.0,
        assignment="cosine",
        normalize_latents=True,
        revive_dead_codes=True,
        revival_warmup_steps=1,
        revival_interval=1,
        max_revivals_per_event=2,
        revival_noise_std=0.01,
    ).train()
    output = quantizer(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))

    assert output.health["revived_codes"].item() == 2
    assert torch.allclose(quantizer.codebook.norm(dim=-1), torch.ones(3), atol=1e-6)


def test_diverse_revival_spreads_replacements_across_high_error_regions():
    quantizer = EMAVectorQuantizer(
        codebook_size=4,
        embedding_dim=2,
        decay=0.0,
        revive_dead_codes=True,
        revival_warmup_steps=1,
        revival_interval=1,
        max_revivals_per_event=2,
        revival_strategy="diverse_farthest",
    )
    with torch.no_grad():
        quantizer.codebook.copy_(
            torch.tensor([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
        )
    quantizer.train()

    output = quantizer(
        torch.tensor([[[5.0, 0.0], [4.9, 0.0], [0.0, 4.0], [0.0, 0.1]]])
    )

    assert output.health["revived_codes"].item() == 2
    replacements = {tuple(row.tolist()) for row in quantizer.codebook[1:3]}
    assert replacements == {(5.0, 0.0), (0.0, 4.0)}


def test_quantizer_rejects_unknown_revival_strategy():
    with pytest.raises(ValueError, match="revival_strategy"):
        EMAVectorQuantizer(revival_strategy="unknown")


def test_uniform_batch_revival_prior_matches_expected_per_code_occupancy():
    quantizer = EMAVectorQuantizer(
        codebook_size=4,
        embedding_dim=2,
        decay=0.0,
        revive_dead_codes=True,
        revival_warmup_steps=1,
        revival_interval=1,
        max_revivals_per_event=1,
        dead_code_threshold=0.1,
        revival_count_prior="uniform_batch",
    ).train()
    with torch.no_grad():
        quantizer.codebook.copy_(
            torch.tensor([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
        )

    output = quantizer(torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]]))

    assert output.health["revived_codes"].item() == 1
    assert quantizer.ema_count[1].item() == pytest.approx(1.0)


def test_quantizer_rejects_unknown_revival_count_prior():
    with pytest.raises(ValueError, match="revival_count_prior"):
        EMAVectorQuantizer(revival_count_prior="unknown")


def test_revival_stop_step_prevents_later_periodic_revival():
    quantizer = EMAVectorQuantizer(
        codebook_size=4,
        embedding_dim=2,
        decay=0.0,
        revive_dead_codes=True,
        revival_warmup_steps=1,
        revival_interval=1,
        revival_stop_after_steps=1,
        max_revivals_per_event=1,
    ).train()
    first = quantizer(torch.zeros(1, 2, 2))
    count_after_first = quantizer.revival_count.item()
    second = quantizer(torch.zeros(1, 2, 2))

    assert first.health["revived_codes"].item() == 1
    assert second.health["revived_codes"].item() == 0
    assert quantizer.revival_count.item() == count_after_first


def test_revival_stop_step_cannot_precede_warmup():
    with pytest.raises(ValueError, match="revival_stop_after_steps"):
        EMAVectorQuantizer(
            revival_warmup_steps=10,
            revival_stop_after_steps=9,
        )
