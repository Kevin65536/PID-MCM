import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from experiments.scripts.train_source_observation_tokenizer import apply_epoch_schedules, compute_gradient_attribution
from src.tokenizers.factorized_labram_vqnsp import SourceObservationLaBraMVQNSP
from src.tokenizers.labram_vqnsp import NormEMAVectorQuantizer
from src.visualization.gradient_diagnostics import (
    plot_gradient_influence_dashboard,
    summarize_total_gradient_groups,
)


class GradientDiagnosticsTests(unittest.TestCase):
    def _build_tiny_model(self, **overrides) -> SourceObservationLaBraMVQNSP:
        params = dict(
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
            kmeans_init=False,
            revive_dead_codes=False,
            drop_path=0.0,
            dropout=0.0,
        )
        params.update(overrides)
        return SourceObservationLaBraMVQNSP(**params)

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
        self.assertIn('fnirs_source_decoder', artifacts['group_names'])
        self.assertIn('fnirs_observation_decoder', artifacts['group_names'])

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

    def test_learnable_codebook_transform_receives_gradient(self):
        torch.manual_seed(5)
        quantizer = NormEMAVectorQuantizer(
            n_embed=4,
            embedding_dim=3,
            beta=0.5,
            decay=0.95,
            kmeans_init=False,
            revive_dead_codes=False,
            learnable_codebook_transform=True,
            codebook_transform_loss_weight=0.5,
        )
        z = torch.randn(6, 3, requires_grad=True)

        _, _, info = quantizer(z)
        info['vq_loss'].backward()

        self.assertGreater(float(info['codebook_loss'].item()), 0.0)
        self.assertIsNotNone(quantizer.codebook_transform.weight.grad)
        self.assertGreater(float(quantizer.codebook_transform.weight.grad.abs().sum().item()), 0.0)

    def test_quantization_strength_zero_returns_continuous_normalized_latents(self):
        torch.manual_seed(17)
        quantizer = NormEMAVectorQuantizer(
            n_embed=4,
            embedding_dim=3,
            beta=0.5,
            decay=0.95,
            kmeans_init=False,
            revive_dead_codes=False,
        )
        quantizer.set_quantization_strength(0.0)
        z = torch.randn(5, 3)

        z_q, _, info = quantizer(z)

        self.assertTrue(torch.allclose(z_q, F.normalize(z, p=2, dim=-1), atol=1e-6))
        self.assertAlmostEqual(float(info['vq_loss'].item()), 0.0, places=6)
        self.assertAlmostEqual(float(info['quantization_strength'].item()), 0.0, places=6)

    def test_source_only_balance_scale_disables_observation_pressure(self):
        torch.manual_seed(13)
        model = self._build_tiny_model(
            source_balance_scale=1.0,
            observation_balance_scale=0.0,
            source_balance_temperature=2.0,
            observation_balance_temperature=1.0,
        )
        eeg = torch.randn(2, 3, 40)
        fnirs = torch.randn(2, 4, 20)

        outputs = model(eeg, fnirs)

        self.assertGreater(float(outputs['observation_balance_loss'].item()), 0.0)
        self.assertAlmostEqual(
            float(outputs['codebook_balance_loss'].item()),
            0.5 * float(outputs['source_balance_loss'].item()),
            places=5,
        )

    def test_forward_outputs_drop_explicit_phase_losses(self):
        model = self._build_tiny_model()
        eeg = torch.randn(2, 3, 40)
        fnirs = torch.randn(2, 4, 20)

        outputs = model(eeg, fnirs)

        self.assertNotIn('eeg_phase_loss', outputs)
        self.assertNotIn('fnirs_phase_loss', outputs)
        self.assertIn('eeg_amp_loss', outputs)
        self.assertIn('fnirs_time_loss', outputs)

    def test_forward_outputs_include_phase2a_branch_target_metrics(self):
        model = self._build_tiny_model(source_target_weight=0.15, observation_target_weight=0.15)
        eeg = torch.randn(2, 3, 40)
        fnirs = torch.randn(2, 4, 20)

        outputs = model(eeg, fnirs)

        self.assertIn('source_target_loss', outputs)
        self.assertIn('source_target_random_baseline', outputs)
        self.assertIn('eeg_source_aux_loss', outputs)
        self.assertIn('fnirs_source_target', outputs)
        self.assertIn('eeg_source_aux_target', outputs)
        self.assertIn('observation_loss', outputs)
        self.assertIn('eeg_observation_loss', outputs)
        self.assertIn('fnirs_observation_loss', outputs)
        self.assertIn('eeg_observation_target', outputs)
        self.assertIn('fnirs_observation_target', outputs)
        self.assertIn('eeg_source_reconstructed', outputs)
        self.assertIn('fnirs_observation_reconstructed', outputs)
        self.assertEqual(outputs['fnirs_source_target'].shape, fnirs.shape)
        self.assertEqual(outputs['eeg_source_aux_target'].shape, eeg.shape)
        self.assertEqual(outputs['eeg_observation_target'].shape, eeg.shape)
        self.assertEqual(outputs['fnirs_observation_target'].shape, fnirs.shape)
        self.assertGreaterEqual(float(outputs['source_target_random_baseline'].item()), 0.0)

        self.assertTrue(
            torch.allclose(
                outputs['eeg_reconstructed'],
                outputs['eeg_source_reconstructed'] + outputs['eeg_observation_reconstructed'],
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs['fnirs_reconstructed'],
                outputs['fnirs_source_reconstructed'] + outputs['fnirs_observation_reconstructed'],
                atol=1e-5,
            )
        )

    def test_coupling_weight_changes_total_loss_when_enabled(self):
        torch.manual_seed(29)
        eeg = torch.randn(2, 3, 40)
        fnirs = torch.randn(2, 4, 20)
        base_model = self._build_tiny_model(coupling_weight=0.0)
        comparison_model = self._build_tiny_model(coupling_weight=0.7)
        comparison_model.load_state_dict(base_model.state_dict())
        base_model.eval()
        comparison_model.eval()

        with torch.no_grad():
            base_outputs = base_model(eeg, fnirs)
            comparison_outputs = comparison_model(eeg, fnirs)

        self.assertGreater(float(base_outputs['source_coupling_loss'].item()), 0.0)
        self.assertTrue(
            torch.allclose(
                base_outputs['source_coupling_loss'],
                comparison_outputs['source_coupling_loss'],
                atol=1e-6,
            )
        )
        self.assertFalse(torch.allclose(base_outputs['loss'], comparison_outputs['loss'], atol=1e-6))
        self.assertTrue(
            torch.allclose(
                comparison_outputs['loss'] - base_outputs['loss'],
                0.7 * comparison_outputs['source_coupling_loss'],
                atol=1e-6,
            )
        )
        self.assertAlmostEqual(
            float(comparison_model.get_gradient_component_weights()['source_coupling_loss']),
            0.7,
            places=6,
        )

        comparison_model.set_alignment_scale(0.5)
        with torch.no_grad():
            half_scale_outputs = comparison_model(eeg, fnirs)
        self.assertTrue(
            torch.allclose(
                half_scale_outputs['loss'] - base_outputs['loss'],
                0.35 * half_scale_outputs['source_coupling_loss'],
                atol=1e-6,
            )
        )
        self.assertAlmostEqual(
            float(comparison_model.get_gradient_component_weights()['source_coupling_loss']),
            0.35,
            places=6,
        )

        comparison_model.set_alignment_scale(0.0)
        with torch.no_grad():
            zero_scale_outputs = comparison_model(eeg, fnirs)
        self.assertTrue(torch.allclose(base_outputs['loss'], zero_scale_outputs['loss'], atol=1e-6))
        self.assertAlmostEqual(
            float(comparison_model.get_gradient_component_weights()['source_coupling_loss']),
            0.0,
            places=6,
        )

    def test_structural_coupling_loss_updates_coupling_logits(self):
        torch.manual_seed(31)
        eeg = torch.randn(2, 3, 40)
        fnirs = torch.randn(2, 4, 20)
        model = self._build_tiny_model(
            coupling_weight=0.4,
            coupling_lag_focus_weight=1.0,
            coupling_smoothness_weight=0.0,
        )
        model.train()

        outputs = model(eeg, fnirs)
        self.assertNotIn('source_coupling_association_loss', outputs)
        self.assertGreater(float(outputs['source_coupling_lag_focus_loss'].detach().item()), 0.0)
        self.assertTrue(
            torch.allclose(
                outputs['source_coupling_loss'],
                outputs['source_coupling_lag_focus_loss'],
                atol=1e-6,
            )
        )

        outputs['loss'].backward()
        self.assertIsNotNone(model.coupling_logits.grad)
        self.assertGreater(float(model.coupling_logits.grad.abs().sum().item()), 0.0)

    def test_fnirs_observation_dropout_one_disables_branch_in_eval(self):
        torch.manual_seed(23)
        model = self._build_tiny_model(
            source_target_weight=0.15,
            observation_target_weight=0.15,
            fnirs_observation_branch_dropout=1.0,
        )
        model.eval()
        eeg = torch.randn(2, 3, 40)
        fnirs = torch.randn(2, 4, 20)

        with torch.no_grad():
            outputs = model(eeg, fnirs)

        self.assertTrue(torch.allclose(
            outputs['fnirs_observation_reconstructed'],
            torch.zeros_like(outputs['fnirs_observation_reconstructed']),
            atol=1e-6,
        ))
        self.assertAlmostEqual(float(outputs['fnirs_observation_loss'].item()), 0.0, places=6)
        self.assertTrue(torch.allclose(
            outputs['fnirs_reconstructed'],
            outputs['fnirs_source_reconstructed'],
            atol=1e-6,
        ))
        self.assertAlmostEqual(
            float(outputs['observation_loss'].item()),
            float(outputs['eeg_observation_loss'].item()),
            places=6,
        )

    def test_signed_rms_carrier_eeg_source_target_preserves_polarity(self):
        model = self._build_tiny_model(
            eeg_source_target_mode='signed_rms_carrier',
            eeg_source_target_smoothing_ms=50.0,
        )
        time = torch.linspace(0.0, 2.0 * torch.pi, steps=40)
        waveform = torch.sin(time)
        eeg = torch.stack([waveform, -waveform, 0.5 * waveform], dim=0).unsqueeze(0)

        target = model._compute_eeg_source_target(eeg)

        self.assertLess(float(target.min().item()), 0.0)
        self.assertGreater(float(target.max().item()), 0.0)
        self.assertEqual(target.shape, eeg.shape)

    def test_apply_epoch_schedules_updates_branch_target_scales(self):
        model = self._build_tiny_model(source_target_weight=0.15, observation_target_weight=0.15)
        config = {
            'loss': {
                'source_target': {'weight': 0.15, 'warmup_epochs': 5},
                'observation_target': {'weight': 0.15, 'warmup_epochs': 3},
            },
            'training': {},
        }

        early_metrics = apply_epoch_schedules(model, 1, config)
        middle_metrics = apply_epoch_schedules(model, 3, config)
        final_metrics = apply_epoch_schedules(model, 5, config)

        self.assertAlmostEqual(float(early_metrics['source_target_scale']), 0.0, places=6)
        self.assertAlmostEqual(float(early_metrics['observation_target_scale']), 0.0, places=6)
        self.assertAlmostEqual(float(middle_metrics['observation_target_scale']), 1.0, places=6)
        self.assertAlmostEqual(float(final_metrics['source_target_scale']), 1.0, places=6)
        self.assertAlmostEqual(float(final_metrics['observation_target_scale']), 1.0, places=6)
        self.assertAlmostEqual(float(model.get_source_target_scale()), 1.0, places=6)
        self.assertAlmostEqual(float(model.get_observation_target_scale()), 1.0, places=6)

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
