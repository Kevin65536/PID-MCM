import unittest

import torch
from torch import nn

from experiments.scripts.calibrate_source_observation_coupling import (
    compute_calibration_objective,
    configure_frozen_token_calibration,
)


class _CalibrationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 3)
        self.coupling_logits = nn.Parameter(torch.full((2, 3, 3), 4.0))


class CouplingCalibrationTests(unittest.TestCase):
    def test_configuration_freezes_everything_except_coupling(self):
        model = _CalibrationModel()

        coupling = configure_frozen_token_calibration(model, reset_coupling=False)

        self.assertIs(coupling, model.coupling_logits)
        self.assertTrue(model.coupling_logits.requires_grad)
        self.assertFalse(model.encoder.weight.requires_grad)
        self.assertFalse(model.encoder.bias.requires_grad)
        self.assertFalse(model.training)

    def test_random_reset_reinitializes_coupling_tensor(self):
        torch.manual_seed(7)
        model = _CalibrationModel()

        configure_frozen_token_calibration(model, reset_coupling=True, reset_std=0.02)

        self.assertLess(float(model.coupling_logits.detach().abs().mean()), 0.1)
        self.assertFalse(torch.allclose(model.coupling_logits, torch.full_like(model.coupling_logits, 4.0)))

    def test_objective_uses_only_enabled_components(self):
        pair = torch.tensor(3.0, requires_grad=True)
        focus = torch.tensor(2.0, requires_grad=True)
        outputs = {
            'source_coupling_pair_likelihood_loss': pair,
            'source_coupling_lag_focus_loss': focus,
        }

        objective, weighted = compute_calibration_objective(
            outputs,
            {'pair_likelihood': 1.0, 'lag_focus': 0.0},
        )
        objective.backward()

        self.assertEqual(float(objective.detach()), 3.0)
        self.assertEqual(set(weighted), {'weighted_pair_likelihood_loss'})
        self.assertEqual(float(pair.grad), 1.0)
        self.assertIsNone(focus.grad)


if __name__ == '__main__':
    unittest.main()
