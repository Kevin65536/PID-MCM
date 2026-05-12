import unittest

import numpy as np
import torch

from src.data.channel_adjacency import (
    build_channel_adjacency,
    compute_per_channel_rms_envelope,
    compute_spatial_fnirs_driver,
)
from src.data.eeg_fnirs_dataset import MultiModalEEGfNIRSDataset


class ChannelAdjacencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = MultiModalEEGfNIRSDataset(
            data_root='data/EEG+NIRS Single-Trial',
            subject_ids=[1],
            task='motor_imagery',
            window_duration_s=20.0,
            normalize=True,
            normalization_mode='session',
            exclude_eog=True,
            hbo_only=True,
            hbr_only=False,
        )

    def test_build_channel_adjacency_is_row_normalized_for_single_trial(self):
        info = build_channel_adjacency(
            'eeg_fnirs_single_trial',
            'data/EEG+NIRS Single-Trial',
            self.dataset.get_eeg_channel_names(),
            self.dataset.get_fnirs_channel_names(),
            reference_subject_id=1,
            use_artifact_data=True,
        )

        self.assertEqual(info.adjacency_matrix.shape, (36, 30))
        self.assertTrue(np.allclose(info.adjacency_matrix.sum(axis=1), 1.0, atol=1e-5))
        self.assertEqual(info.anchor_matches[0]['base_channel'], 'AF7Fp1')
        self.assertIn('F7', info.anchor_matches[0]['source_anchor']['direct_labels'])
        self.assertEqual(info.warnings, [])

    def test_compute_per_channel_rms_envelope_reduces_to_abs_without_smoothing(self):
        eeg = torch.tensor(
            [[[1.0, -1.0, 1.0, -1.0], [0.0, 0.0, 2.0, -2.0]]],
            dtype=torch.float32,
        )

        target = compute_per_channel_rms_envelope(eeg, smoothing_samples=1, eps=0.0)
        expected = torch.tensor(
            [[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]]],
            dtype=torch.float32,
        )

        self.assertTrue(torch.allclose(target, expected, atol=1e-6))

    def test_compute_spatial_fnirs_driver_uses_weighted_eeg_power(self):
        eeg = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]]],
            dtype=torch.float32,
        )
        adjacency = torch.tensor(
            [[1.0, 0.0], [0.25, 0.75]],
            dtype=torch.float32,
        )

        driver = compute_spatial_fnirs_driver(eeg, adjacency, target_length=2)
        expected = torch.tensor(
            [[[1.0, 4.0], [7.0, 13.0]]],
            dtype=torch.float32,
        )

        self.assertTrue(torch.allclose(driver, expected, atol=1e-6))


if __name__ == '__main__':
    unittest.main()