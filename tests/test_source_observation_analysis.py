import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.visualization import source_observation_analysis as soa


class SourceObservationAnalysisTests(unittest.TestCase):
    def test_resolve_token_pattern_visualization_config_defaults_to_enabled(self):
        resolved = soa._resolve_token_pattern_visualization_config({'analysis': {}})

        self.assertTrue(resolved['enabled'])
        self.assertEqual(resolved['subdir'], 'token_pattern_visualizations')
        self.assertEqual(resolved['top_k'], 8)

    def test_resolve_token_pattern_visualization_config(self):
        config = {
            'analysis': {
                'token_pattern_visualization': {
                    'enabled': True,
                    'subdir': 'custom_token_patterns',
                    'max_samples': 24,
                    'top_k': 5,
                    'max_patches_per_code': 18,
                    'max_overlay_patches': 7,
                    'channel_indices': {'eeg': 2, 'fnirs': 1},
                    'frequency_range_hz': {'eeg': [0.0, 30.0], 'fnirs': [0.0, 0.4]},
                }
            }
        }

        resolved = soa._resolve_token_pattern_visualization_config(config)

        self.assertTrue(resolved['enabled'])
        self.assertEqual(resolved['subdir'], 'custom_token_patterns')
        self.assertEqual(resolved['max_samples'], 24)
        self.assertEqual(resolved['top_k'], 5)
        self.assertEqual(resolved['max_patches_per_code'], 18)
        self.assertEqual(resolved['max_overlay_patches'], 7)
        self.assertEqual(resolved['channel_indices']['eeg'], 2)
        self.assertEqual(resolved['channel_indices']['fnirs'], 1)
        self.assertEqual(resolved['frequency_range_hz']['eeg'], (0.0, 30.0))
        self.assertEqual(resolved['frequency_range_hz']['fnirs'], (0.0, 0.4))

    def test_generate_token_pattern_visualizations_writes_expected_artifact(self):
        cfg = {
            'enabled': True,
            'subdir': 'token_pattern_visualizations',
            'top_k': 2,
            'max_patches_per_code': 12,
            'max_overlay_patches': 4,
            'frequency_range_hz': {'eeg': (0.0, 20.0), 'fnirs': (0.0, 0.5)},
        }
        time = np.linspace(0.0, 1.0, num=16, endpoint=False, dtype=np.float32)
        signals = np.stack(
            [
                np.sin(2.0 * np.pi * (index + 1) * time) + 0.05 * index
                for index in range(6)
            ],
            axis=0,
        ).astype(np.float32)
        tokens = np.asarray(
            [
                [0, 1, 0, 2],
                [1, 1, 2, 2],
                [0, 0, 1, 2],
                [2, 1, 1, 0],
                [0, 2, 2, 1],
                [1, 0, 2, 0],
            ],
            dtype=np.int64,
        )
        split_stats = {
            'codebooks': {
                'eeg_source': {'codebook_size': 3},
            },
            'token_pattern_samples': {
                'signals': {'eeg': signals},
                'tokens': {'eeg_source': tokens},
                'sample_rates': {'eeg': 16.0},
                'patch_sizes': {'eeg': 4},
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = soa._generate_token_pattern_visualizations(
                split_name='test',
                split_stats=split_stats,
                analysis_root=Path(tmpdir),
                token_pattern_viz_cfg=cfg,
            )

            self.assertIn('eeg_source', artifacts)
            if soa.HAS_MATPLOTLIB:
                artifact_path = Path(artifacts['eeg_source'])
                self.assertTrue(artifact_path.exists())
                self.assertEqual(artifact_path.name, 'test_eeg_source_token_patterns.png')
            else:
                self.assertIsNone(artifacts['eeg_source'])


if __name__ == '__main__':
    unittest.main()