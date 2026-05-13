import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.visualization import source_observation_analysis as soa


class SourceObservationAnalysisTests(unittest.TestCase):
    def test_build_gate_1_ignores_disabled_fnirs_observation_codebook(self):
        split_stats = {
            'codebooks': {
                'eeg_source': {'passes_thresholds': True},
                'fnirs_source': {'passes_thresholds': True},
                'eeg_observation': {'passes_thresholds': True},
                'fnirs_observation': {'passes_thresholds': False},
            },
            'branch_reconstruction': {},
        }
        metrics_payload = {
            'epochs': [
                {'metrics': {'val_eeg_rec_loss': 2.0, 'val_fnirs_rec_loss': 1.5}},
                {'metrics': {'val_eeg_rec_loss': 1.0, 'val_fnirs_rec_loss': 1.0}},
            ]
        }
        branch_policy = {
            'active_codebooks': ['eeg_source', 'fnirs_source', 'eeg_observation'],
            'ignored_branches': ['fnirs_observation'],
            'active_observation_modalities': ['eeg'],
            'ignored_observation_modalities': ['fnirs'],
        }

        gate_1 = soa._build_gate_1(split_stats, metrics_payload, branch_policy=branch_policy)

        self.assertEqual(gate_1['status'], 'pass')
        self.assertNotIn('fnirs_observation', gate_1['metrics']['codebooks'])
        self.assertTrue(any('fnirs_observation' in note for note in gate_1['notes']))

    def test_build_gate_2_passes_when_source_target_beats_baseline(self):
        split_stats = {
            'eeg_source_tokens': np.asarray([[0, 1, 0], [1, 0, 1]], dtype=np.int64),
            'fnirs_source_tokens': np.asarray([[0, 1, 0], [1, 0, 1]], dtype=np.int64),
            'mean_scalars': {
                'source_target_loss': 0.20,
                'source_target_random_baseline': 0.45,
                'eeg_source_aux_loss': 0.18,
                'observation_loss': 0.14,
            },
            'branch_reconstruction': {
                'eeg_source_target_random_baseline': 0.40,
                'eeg_observation_target_mse': 0.09,
                'fnirs_observation_target_mse': 0.11,
                'eeg_observation_contribution_gap': 0.10,
                'fnirs_observation_contribution_gap': 0.12,
            },
            'codebooks': {
                'eeg_source': {'active_code_ratio': 0.62},
                'fnirs_source': {'active_code_ratio': 0.58},
            },
        }
        coupling = {
            'available': True,
            'lag': 0,
            'transition': np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64),
        }

        gate_2 = soa._build_gate_2(split_stats, best_lag=0, coupling=coupling)

        self.assertEqual(gate_2['status'], 'pass')
        self.assertAlmostEqual(float(gate_2['metrics']['source_target_mse']), 0.20, places=6)

    def test_build_gate_2_ignores_disabled_fnirs_observation_branch(self):
        split_stats = {
            'eeg_source_tokens': np.asarray([[0, 1, 0], [1, 0, 1]], dtype=np.int64),
            'fnirs_source_tokens': np.asarray([[0, 1, 0], [1, 0, 1]], dtype=np.int64),
            'mean_scalars': {
                'source_target_loss': 0.20,
                'source_target_random_baseline': 0.45,
                'eeg_source_aux_loss': 0.18,
                'observation_loss': 0.14,
            },
            'branch_reconstruction': {
                'eeg_source_target_random_baseline': 0.40,
                'eeg_observation_target_mse': 0.09,
                'eeg_observation_contribution_gap': 0.10,
                'fnirs_observation_contribution_gap': -0.50,
            },
            'codebooks': {
                'eeg_source': {'active_code_ratio': 0.62},
                'fnirs_source': {'active_code_ratio': 0.58},
            },
        }
        coupling = {
            'available': True,
            'lag': 0,
            'transition': np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64),
        }
        branch_policy = {
            'active_codebooks': ['eeg_source', 'fnirs_source', 'eeg_observation'],
            'ignored_branches': ['fnirs_observation'],
            'active_observation_modalities': ['eeg'],
            'ignored_observation_modalities': ['fnirs'],
        }

        gate_2 = soa._build_gate_2(split_stats, best_lag=0, coupling=coupling, branch_policy=branch_policy)

        self.assertEqual(gate_2['status'], 'pass')
        self.assertEqual(gate_2['metrics']['active_observation_modalities'], ['eeg'])
        self.assertEqual(gate_2['metrics']['ignored_observation_modalities'], ['fnirs'])

    def test_build_gate_2_is_pending_when_source_target_metrics_missing(self):
        split_stats = {
            'eeg_source_tokens': np.asarray([[0, 1, 0], [1, 0, 1]], dtype=np.int64),
            'fnirs_source_tokens': np.asarray([[0, 1, 0], [1, 0, 1]], dtype=np.int64),
            'mean_scalars': {
                'observation_loss': 0.14,
            },
            'branch_reconstruction': {
                'eeg_observation_contribution_gap': 0.10,
                'fnirs_observation_contribution_gap': 0.12,
            },
            'codebooks': {
                'eeg_source': {'active_code_ratio': 0.62},
                'fnirs_source': {'active_code_ratio': 0.58},
            },
        }
        coupling = {
            'available': True,
            'lag': 0,
            'transition': np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64),
        }

        gate_2 = soa._build_gate_2(split_stats, best_lag=0, coupling=coupling)

        self.assertEqual(gate_2['status'], 'pending')
        self.assertTrue(gate_2['notes'])

    def test_build_gate_2_is_pending_when_observation_metrics_missing(self):
        split_stats = {
            'eeg_source_tokens': np.asarray([[0, 1, 0], [1, 0, 1]], dtype=np.int64),
            'fnirs_source_tokens': np.asarray([[0, 1, 0], [1, 0, 1]], dtype=np.int64),
            'mean_scalars': {
                'source_target_loss': 0.20,
                'source_target_random_baseline': 0.45,
                'eeg_source_aux_loss': 0.18,
            },
            'branch_reconstruction': {
                'eeg_source_target_random_baseline': 0.40,
                'eeg_observation_contribution_gap': 0.10,
                'fnirs_observation_contribution_gap': 0.12,
            },
            'codebooks': {
                'eeg_source': {'active_code_ratio': 0.62},
                'fnirs_source': {'active_code_ratio': 0.58},
            },
        }
        coupling = {
            'available': True,
            'lag': 0,
            'transition': np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64),
        }

        gate_2 = soa._build_gate_2(split_stats, best_lag=0, coupling=coupling)

        self.assertEqual(gate_2['status'], 'pending')
        self.assertTrue(any('Observation target metrics' in note for note in gate_2['notes']))

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

    def test_generate_reconstruction_visualizations_uses_target_aware_payloads(self):
        cfg = {
            'enabled': True,
            'subdir': 'reconstruction_visualizations',
            'domains': ['time'],
            'channel_indices': {'eeg': 0, 'fnirs': 0},
            'frequency_range_hz': {'eeg': (0.0, 20.0), 'fnirs': (0.0, 0.5)},
            'time_window_s': {'eeg': 1.0, 'fnirs': 1.0},
            'max_time_points': {'eeg': 32, 'fnirs': 32},
        }
        time = np.linspace(0.0, 1.0, num=16, endpoint=False, dtype=np.float32)
        original = np.stack(
            [
                np.sin(2.0 * np.pi * time),
                np.cos(2.0 * np.pi * time),
            ],
            axis=0,
        ).astype(np.float32)[:, None, :]
        source_target = (0.5 * original).astype(np.float32)
        residual_target = (original - source_target).astype(np.float32)
        split_stats = {
            'reconstruction_samples': {
                'original': {'fnirs': original},
                'targets': {
                    'source': {'fnirs': source_target},
                    'residual': {'fnirs': residual_target},
                },
                'branches': {
                    'full': {'fnirs': original * 0.9},
                    'source_only': {'fnirs': source_target * 1.05},
                    'observation_only': {'fnirs': residual_target * 0.95},
                },
                'sample_rates': {'fnirs': 16.0},
                'patch_sizes': {'fnirs': 4},
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = soa._generate_reconstruction_visualizations(
                split_name='test',
                split_stats=split_stats,
                analysis_root=Path(tmpdir),
                reconstruction_viz_cfg=cfg,
            )

            self.assertIn('fnirs', artifacts)
            if soa.HAS_MATPLOTLIB:
                artifact_path = Path(artifacts['fnirs']['time_path'])
                self.assertTrue(artifact_path.exists())
                self.assertEqual(artifact_path.name, 'test_fnirs_time_reconstruction.png')
            else:
                self.assertIsNone(artifacts['fnirs']['time_path'])


if __name__ == '__main__':
    unittest.main()