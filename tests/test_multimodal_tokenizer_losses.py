import unittest

import torch

from src.losses.multimodal_tokenizer import (
    batch_usage_entropy_loss,
    coupling_eeg_neighbor_smoothness_loss,
    coupling_lag_evidence_loss,
    coupling_lag_focus_loss,
    coupling_pair_likelihood_loss,
    straight_through_assignment_probs,
)


class MultimodalTokenizerLossTests(unittest.TestCase):
    def test_straight_through_assignments_are_hard_in_forward(self):
        logits = torch.tensor(
            [[[2.0, 0.5, -1.0], [0.1, 0.2, 0.3], [1.0, 3.0, 2.0]]],
            requires_grad=True,
        )

        probs = straight_through_assignment_probs(logits, temperature=1.0)
        expected = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]])

        self.assertTrue(torch.equal(probs.detach().argmax(dim=-1), expected.argmax(dim=-1)))
        self.assertTrue(torch.allclose(probs.detach().sum(dim=-1), torch.ones_like(probs.detach().sum(dim=-1))))

        loss = batch_usage_entropy_loss(probs)
        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum().item()), 0.0)

    def test_hard_assignment_balance_loss_penalizes_collapsed_usage(self):
        collapsed_logits = torch.tensor(
            [[[5.0, 4.0, 0.0], [5.2, 4.1, 0.0], [5.4, 4.2, 0.0], [5.6, 4.3, 0.0]]]
        )
        balanced_logits = torch.tensor(
            [[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0], [5.0, 0.0, 0.0]]]
        )

        collapsed_loss = batch_usage_entropy_loss(
            straight_through_assignment_probs(collapsed_logits, temperature=1.0)
        )
        balanced_loss = batch_usage_entropy_loss(
            straight_through_assignment_probs(balanced_logits, temperature=1.0)
        )

        self.assertGreater(float(collapsed_loss.item()), float(balanced_loss.item()))
        self.assertGreater(float(collapsed_loss.item()), 0.9)

    def test_coupling_lag_focus_loss_penalizes_diffuse_delay_usage(self):
        diffuse_logits = torch.zeros(3, 2, 2)
        focused_logits = torch.zeros(3, 2, 2)
        focused_logits[1] = 4.0

        diffuse_loss = coupling_lag_focus_loss(diffuse_logits)
        focused_loss = coupling_lag_focus_loss(focused_logits)

        self.assertGreater(float(diffuse_loss.item()), float(focused_loss.item()))
        self.assertGreater(float(diffuse_loss.item()), 0.9)

    def test_coupling_neighbor_smoothness_uses_eeg_codebook_geometry(self):
        eeg_codebook = torch.tensor(
            [
                [1.0, 0.0],
                [0.98, 0.02],
                [-1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        consistent_logits = torch.tensor(
            [
                [[4.0, 0.0], [4.0, 0.0], [0.0, 4.0]],
                [[0.0, 4.0], [0.0, 4.0], [4.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        inconsistent_logits = consistent_logits.clone()
        inconsistent_logits[:, 1] = torch.tensor([[0.0, 4.0], [4.0, 0.0]], dtype=torch.float32)

        consistent_loss = coupling_eeg_neighbor_smoothness_loss(
            consistent_logits,
            eeg_codebook,
            n_neighbors=1,
        )
        inconsistent_loss = coupling_eeg_neighbor_smoothness_loss(
            inconsistent_logits,
            eeg_codebook,
            n_neighbors=1,
        )

        self.assertGreater(float(inconsistent_loss.item()), float(consistent_loss.item()))

    def test_coupling_pair_likelihood_rewards_observed_fnirs_tokens(self):
        eeg_logits = torch.tensor(
            [
                [
                    [5.0, 0.0],
                    [0.0, 5.0],
                ]
            ],
            dtype=torch.float32,
        )
        fnirs_logits = torch.tensor(
            [
                [
                    [5.0, 0.0],
                    [0.0, 5.0],
                ]
            ],
            dtype=torch.float32,
        )
        uniform_logits = torch.zeros(2, 2, 2, dtype=torch.float32)
        matched_logits = torch.zeros(2, 2, 2, dtype=torch.float32)
        matched_logits[0, 0, 0] = 5.0
        matched_logits[0, 1, 1] = 5.0

        uniform_loss = coupling_pair_likelihood_loss(
            uniform_logits,
            eeg_logits,
            fnirs_logits,
            detach_tokens=True,
        )
        matched_loss = coupling_pair_likelihood_loss(
            matched_logits,
            eeg_logits,
            fnirs_logits,
            detach_tokens=True,
        )

        self.assertGreater(float(uniform_loss.item()), float(matched_loss.item()))

    def test_coupling_pair_likelihood_can_backpropagate_to_assignments(self):
        eeg_logits = torch.tensor(
            [[[3.0, 0.0], [0.0, 3.0]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        fnirs_logits = torch.tensor(
            [[[0.0, 3.0], [3.0, 0.0]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        coupling_logits = torch.zeros(2, 2, 2, dtype=torch.float32, requires_grad=True)
        coupling_logits.data[1, 0, 0] = 4.0
        coupling_logits.data[1, 1, 1] = 4.0

        loss = coupling_pair_likelihood_loss(
            coupling_logits,
            eeg_logits,
            fnirs_logits,
            detach_tokens=False,
        )
        loss.backward()

        self.assertIsNotNone(coupling_logits.grad)
        self.assertIsNotNone(eeg_logits.grad)
        self.assertIsNotNone(fnirs_logits.grad)
        self.assertGreater(float(coupling_logits.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(eeg_logits.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(fnirs_logits.grad.abs().sum().item()), 0.0)

    def test_coupling_pair_likelihood_weights_each_lag_equally(self):
        eeg_logits = torch.tensor(
            [[[5.0, 0.0], [0.0, 5.0], [5.0, 0.0]]],
            dtype=torch.float32,
        )
        fnirs_logits = eeg_logits.clone()
        coupling_logits = torch.zeros(3, 2, 2, dtype=torch.float32)

        loss = coupling_pair_likelihood_loss(
            coupling_logits,
            eeg_logits,
            fnirs_logits,
            detach_tokens=True,
        )

        self.assertAlmostEqual(float(loss.item()), float(torch.log(torch.tensor(2.0)).item()), places=5)

    def test_residualized_pair_likelihood_ignores_eeg_independent_column_bias(self):
        eeg_logits = torch.tensor(
            [[[5.0, 0.0], [0.0, 5.0], [5.0, 0.0], [0.0, 5.0]]],
            dtype=torch.float32,
        )
        fnirs_logits = torch.tensor(
            [[[5.0, 0.0], [5.0, 0.0], [5.0, 0.0], [0.0, 5.0]]],
            dtype=torch.float32,
        )
        uniform_logits = torch.zeros(1, 2, 2, dtype=torch.float32)
        marginal_bias_logits = uniform_logits.clone()
        marginal_bias_logits[:, :, 0] = 1.0

        plain_uniform = coupling_pair_likelihood_loss(
            uniform_logits,
            eeg_logits,
            fnirs_logits,
            detach_tokens=True,
        )
        plain_biased = coupling_pair_likelihood_loss(
            marginal_bias_logits,
            eeg_logits,
            fnirs_logits,
            detach_tokens=True,
        )
        residual_uniform = coupling_pair_likelihood_loss(
            uniform_logits,
            eeg_logits,
            fnirs_logits,
            detach_tokens=True,
            residualize_fnirs_marginal=True,
        )
        residual_biased = coupling_pair_likelihood_loss(
            marginal_bias_logits,
            eeg_logits,
            fnirs_logits,
            detach_tokens=True,
            residualize_fnirs_marginal=True,
        )

        self.assertLess(float(plain_biased.item()), float(plain_uniform.item()))
        self.assertTrue(torch.allclose(residual_biased, residual_uniform, atol=1e-6))

    def test_coupling_lag_evidence_prefers_conditionally_informative_lag(self):
        eeg_logits = torch.tensor(
            [[[5.0, 0.0], [0.0, 5.0], [5.0, 0.0]]],
            dtype=torch.float32,
        )
        fnirs_logits = torch.tensor(
            [[[0.0, 5.0], [5.0, 0.0], [0.0, 5.0]]],
            dtype=torch.float32,
        )
        base_logits = torch.zeros(2, 2, 2, dtype=torch.float32)
        base_logits[1, 0, 0] = 5.0
        base_logits[1, 1, 1] = 5.0
        wrong_lag_logits = base_logits.clone()
        wrong_lag_logits[0] += 4.0
        correct_lag_logits = base_logits.clone()
        correct_lag_logits[1] += 4.0

        wrong_loss = coupling_lag_evidence_loss(
            wrong_lag_logits,
            eeg_logits,
            fnirs_logits,
            detach_tokens=True,
            evidence_temperature=0.1,
        )
        correct_loss = coupling_lag_evidence_loss(
            correct_lag_logits,
            eeg_logits,
            fnirs_logits,
            detach_tokens=True,
            evidence_temperature=0.1,
        )

        self.assertGreater(float(wrong_loss.item()), float(correct_loss.item()))


if __name__ == '__main__':
    unittest.main()
