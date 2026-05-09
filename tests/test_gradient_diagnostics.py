import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.scripts.train_shared_tokenizer import compute_gradient_attribution
from src.tokenizers.factorized_labram_vqnsp import SourceObservationLaBraMVQNSP
from src.visualization.gradient_diagnostics import (
    plot_gradient_influence_dashboard,
    summarize_total_gradient_groups,
)


class GradientDiagnosticsTests(unittest.TestCase):
    def _build_tiny_model(self) -> SourceObservationLaBraMVQNSP:
        return SourceObservationLaBraMVQNSP(
            eeg_seq_length=40,
            eeg_patch_size=20,
            eeg_channels=3,
            eeg_encoder_embed_dim=16,
            eeg_encoder_depth=1,
            eeg_encoder_num_heads=4,
            eeg_decoder_embed_dim=16,
            eeg_decoder_depth=1,
            eeg_decoder_num_heads=4,
            fnirs_seq_length=20,
            fnirs_patch_size=10,
            fnirs_channels=4,
            fnirs_encoder_embed_dim=12,
            fnirs_encoder_depth=1,
            fnirs_encoder_num_heads=4,
            fnirs_decoder_embed_dim=12,
            fnirs_decoder_depth=1,
            fnirs_decoder_num_heads=4,
            source_codebook_size=4,
            eeg_source_codebook_dim=8,
            fnirs_source_codebook_dim=8,
            eeg_observation_codebook_size=4,
            eeg_observation_codebook_dim=8,
            fnirs_observation_codebook_size=4,
            fnirs_observation_codebook_dim=8,
            alignment_lag_candidates=[0],
            kmeans_init=False,
            revive_dead_codes=False,
            drop_path=0.0,
            dropout=0.0,
        )

    def test_component_group_attribution_matches_architecture_groups(self):
        torch.manual_seed(7)
        model = self._build_tiny_model()
        eeg = torch.randn(2, 3, 40)
        fnirs = torch.randn(2, 4, 20)

        outputs = model(eeg, fnirs)
        metrics, artifacts = compute_gradient_attribution(model, outputs)

        self.assertIn('grad_component_count', metrics)
        self.assertIsNotNone(artifacts)
        self.assertIn('group_names', artifacts)
        self.assertIn('component_group_shares', artifacts)
        self.assertIn('group_component_shares', artifacts)
        self.assertIn('coupling_logits', artifacts['group_names'])
        self.assertIn('eeg_encoder', artifacts['group_names'])
        self.assertIn('fnirs_decoder', artifacts['group_names'])

        component_group_shares = np.asarray(artifacts['component_group_shares'], dtype=np.float32)
        group_component_shares = np.asarray(artifacts['group_component_shares'], dtype=np.float32)
        self.assertEqual(component_group_shares.shape[0], len(artifacts['component_names']))
        self.assertEqual(component_group_shares.shape[1], len(artifacts['group_names']))
        self.assertEqual(group_component_shares.shape, component_group_shares.shape)

        row_sums = component_group_shares.sum(axis=1)
        self.assertTrue(np.allclose(row_sums[row_sums > 0.0], 1.0, atol=1e-4))

    def test_total_gradient_group_summary_produces_ordered_quantiles(self):
        torch.manual_seed(11)
        model = self._build_tiny_model()
        eeg = torch.randn(2, 3, 40)
        fnirs = torch.randn(2, 4, 20)

        outputs = model(eeg, fnirs)
        outputs['loss'].backward()

        summary = summarize_total_gradient_groups(
            model,
            group_specs=model.get_gradient_parameter_group_specs(),
        )
        self.assertIsNotNone(summary)
        self.assertIn('group_total_shares', summary)
        self.assertIn('group_abs_grad_quantiles', summary)

        total_shares = np.asarray(summary['group_total_shares'], dtype=np.float32)
        self.assertAlmostEqual(float(total_shares.sum()), 1.0, places=4)

        quantiles = summary['group_abs_grad_quantiles']
        p05 = np.asarray(quantiles['p05'], dtype=np.float32)
        p25 = np.asarray(quantiles['p25'], dtype=np.float32)
        p50 = np.asarray(quantiles['p50'], dtype=np.float32)
        p75 = np.asarray(quantiles['p75'], dtype=np.float32)
        p95 = np.asarray(quantiles['p95'], dtype=np.float32)
        self.assertTrue(np.all(p05 <= p25 + 1e-12))
        self.assertTrue(np.all(p25 <= p50 + 1e-12))
        self.assertTrue(np.all(p50 <= p75 + 1e-12))
        self.assertTrue(np.all(p75 <= p95 + 1e-12))

    def test_gradient_influence_dashboard_plot_smoke(self):
        component_names = ['eeg_rec_loss', 'fnirs_rec_loss', 'vq_loss']
        group_labels = ['EEG Encoder', 'Coupling', 'EEG Decoder']
        component_group_shares = [
            [0.7, 0.0, 0.3],
            [0.2, 0.1, 0.7],
            [0.4, 0.6, 0.0],
        ]
        group_component_shares = [
            [0.5, 0.0, 1.0],
            [0.2, 0.2, 0.0],
            [0.3, 0.8, 0.0],
        ]
        group_total_shares = [0.35, 0.25, 0.40]
        group_abs_grad_quantiles = {
            'p05': [1e-4, 2e-4, 1e-5],
            'p25': [4e-4, 7e-4, 4e-5],
            'p50': [1e-3, 1.2e-3, 7e-5],
            'p75': [2e-3, 3e-3, 1.5e-4],
            'p95': [5e-3, 7e-3, 5e-4],
        }

        fig = plot_gradient_influence_dashboard(
            component_names=component_names,
            group_labels=group_labels,
            component_group_shares=component_group_shares,
            group_component_shares=group_component_shares,
            group_total_shares=group_total_shares,
            group_abs_grad_quantiles=group_abs_grad_quantiles,
            step=3,
        )
        self.assertGreaterEqual(len(fig.axes), 4)
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()