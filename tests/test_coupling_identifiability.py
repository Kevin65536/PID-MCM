import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.analysis.coupling_identifiability import (
    effective_conditional_probabilities,
    lag_mutual_information,
    load_export_split,
    occupancy_weighted_gauge,
    patch_features,
    patch_features_torch,
)
from src.losses.multimodal_tokenizer import coupling_pair_likelihood_loss


class CouplingIdentifiabilityTests(unittest.TestCase):
    def test_torch_patch_features_match_numpy(self):
        rng = np.random.default_rng(11)
        for channels, sample_rate, patch_size, eeg in (
            (2, 200.0, 400, True),
            (1, 10.0, 20, False),
        ):
            signal = rng.normal(size=(3, channels, patch_size * 2)).astype(np.float32)
            expected = patch_features(
                signal, sample_rate_hz=sample_rate, patch_size=patch_size, eeg=eeg,
            )
            actual = patch_features_torch(
                torch.from_numpy(signal),
                sample_rate_hz=sample_rate,
                patch_size=patch_size,
                eeg=eeg,
            ).numpy()
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)

    def test_load_export_split_supports_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard_dir = root / "shards" / "test"
            shard_dir.mkdir(parents=True)
            np.savez(shard_dir / "part_00000.npz", value=np.asarray([[1], [2]]))
            np.savez(shard_dir / "part_00001.npz", value=np.asarray([[3]]))
            (root / "test.manifest.json").write_text(json.dumps({
                "shards": [
                    "shards/test/part_00000.npz",
                    "shards/test/part_00001.npz",
                ],
            }))
            loaded = load_export_split(root, "test")
            np.testing.assert_array_equal(loaded["value"], np.asarray([[1], [2], [3]]))

    def test_gradient_routing_targets_only_requested_assignment_logits(self):
        for target, expect_eeg, expect_fnirs in (
            ("none", False, False),
            ("eeg", True, False),
            ("fnirs", False, True),
            ("both", True, True),
        ):
            coupling = torch.randn(3, 4, 4, requires_grad=True)
            eeg = torch.randn(2, 3, 4, requires_grad=True)
            fnirs = torch.randn(2, 3, 4, requires_grad=True)
            loss = coupling_pair_likelihood_loss(
                coupling, eeg, fnirs, gradient_target=target,
            )
            loss.backward()
            self.assertIsNotNone(coupling.grad)
            self.assertEqual(eeg.grad is not None and bool(torch.any(eeg.grad != 0)), expect_eeg)
            self.assertEqual(fnirs.grad is not None and bool(torch.any(fnirs.grad != 0)), expect_fnirs)

    def test_legacy_detach_mapping_is_preserved(self):
        coupling = torch.randn(2, 3, 3, requires_grad=True)
        eeg = torch.randn(2, 2, 3, requires_grad=True)
        fnirs = torch.randn(2, 2, 3, requires_grad=True)
        coupling_pair_likelihood_loss(coupling, eeg, fnirs, detach_tokens=True).backward()
        self.assertIsNone(eeg.grad)
        self.assertIsNone(fnirs.grad)

    def test_occupancy_weighted_gauge_removes_column_bias(self):
        logits = torch.randn(4, 5, 6)
        occupancy = torch.rand(4, 5)
        gauged = occupancy_weighted_gauge(logits, occupancy)
        weights = occupancy / occupancy.sum(dim=-1, keepdim=True)
        column_mean = torch.einsum("le,lef->lf", weights, gauged)
        self.assertTrue(torch.allclose(column_mean, torch.zeros_like(column_mean), atol=1e-6))

    def test_eeg_independent_column_bias_does_not_change_effective_q(self):
        logits = torch.randn(3, 4, 5)
        column_bias = torch.randn(3, 1, 5)
        prior = torch.softmax(torch.randn(3, 5), dim=-1)
        occupancy = torch.softmax(torch.randn(3, 4), dim=-1)
        first = effective_conditional_probabilities(logits, prior, occupancy)
        second = effective_conditional_probabilities(logits + column_bias, prior, occupancy)
        self.assertTrue(torch.allclose(first, second, atol=1e-6))

    def test_triangular_support_can_create_spurious_lag_pattern(self):
        rng = np.random.default_rng(3)
        samples, tokens = 4000, 10
        position_state = np.tile(np.arange(tokens) // 2, (samples, 1))
        eeg = np.bitwise_xor(position_state, rng.integers(0, 2, size=(samples, tokens)))
        fnirs = position_state.copy()
        observed = lag_mutual_information(eeg, fnirs, k_eeg=8, k_fnirs=5)
        permuted = lag_mutual_information(
            eeg[:, rng.permutation(tokens)], fnirs, k_eeg=8, k_fnirs=5,
        )
        self.assertGreater(float(observed.max()), float(permuted.max()))


if __name__ == "__main__":
    unittest.main()
