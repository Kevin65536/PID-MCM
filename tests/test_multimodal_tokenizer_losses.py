import unittest

import torch

from src.losses.multimodal_tokenizer import batch_usage_entropy_loss, straight_through_assignment_probs


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


if __name__ == '__main__':
    unittest.main()