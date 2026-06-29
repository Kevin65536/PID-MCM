import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from experiments.scripts.train_source_observation_tokenizer import (
    apply_epoch_schedules,
    compute_gradient_attribution,
    update_cross_modal_gradient_scale,
)
from src.tokenizers.factorized_labram_vqnsp import (
    CausalLowRankCrossAdapter,
    LagAwareCrossModalFusion,
    SourceObservationLaBraMVQNSP,
)
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

    def test_coupling_max_lag_tokens_limits_tensor_support(self):
        model = self._build_tiny_model(coupling_max_lag_tokens=0)

        self.assertEqual(model.n_coupling_lags, 1)
        self.assertEqual(model.coupling_lags, [0])
        self.assertEqual(tuple(model.coupling_logits.shape), (1, 4, 4))

        clamped = self._build_tiny_model(coupling_max_lag_tokens=99)
        self.assertEqual(clamped.n_coupling_lags, clamped.n_patches)
        self.assertEqual(tuple(clamped.coupling_logits.shape), (clamped.n_patches, 4, 4))

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
        self.assertIn('context_coupling', artifacts['group_names'])
        self.assertIn('eeg_encoder', artifacts['group_names'])
        self.assertIn('fnirs_source_decoder', artifacts['group_names'])
        self.assertIn('fnirs_observation_decoder', artifacts['group_names'])
        self.assertTrue(any(key.startswith('grad_group_norm_') for key in metrics))
        self.assertTrue(any(key.startswith('grad_group_share_') for key in metrics))

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
        self.assertIn('source_coupling_joint_entropy_loss', base_outputs)
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
                comparison_outputs['source_coupling_weighted_loss'],
                0.7 * comparison_outputs['source_coupling_loss'],
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                comparison_outputs['loss'] - base_outputs['loss'],
                0.7 * comparison_outputs['source_coupling_loss'],
                atol=1e-6,
            )
        )
        component_weights = comparison_model.get_gradient_component_weights()
        self.assertAlmostEqual(float(component_weights['source_coupling_lag_focus_loss']), 0.7, places=6)
        self.assertIn('source_coupling_joint_entropy_loss', component_weights)
        self.assertAlmostEqual(float(component_weights['source_coupling_smoothness_loss']), 0.14, places=6)

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
        component_weights = comparison_model.get_gradient_component_weights()
        self.assertAlmostEqual(float(component_weights['source_coupling_lag_focus_loss']), 0.35, places=6)
        self.assertAlmostEqual(float(component_weights['source_coupling_smoothness_loss']), 0.07, places=6)

        comparison_model.set_alignment_scale(0.0)
        with torch.no_grad():
            zero_scale_outputs = comparison_model(eeg, fnirs)
        self.assertTrue(torch.allclose(base_outputs['loss'], zero_scale_outputs['loss'], atol=1e-6))
        for name, weight in comparison_model.get_gradient_component_weights().items():
            if name.startswith('source_coupling_'):
                self.assertAlmostEqual(float(weight), 0.0, places=6)

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

    def test_local_residual_coupling_updates_only_selected_assignment_side(self):
        torch.manual_seed(37)
        eeg = torch.randn(2, 3, 40)
        fnirs = torch.randn(2, 4, 20)
        model = self._build_tiny_model(
            coupling_weight=0.2,
            coupling_lag_focus_weight=0.0,
            coupling_smoothness_weight=0.0,
            coupling_pair_likelihood_weight=0.0,
            coupling_local_residual_enabled=True,
            coupling_pair_gradient_target="eeg",
            coupling_effective_smoothness_weight=0.0,
            coupling_interaction_lag_sparsity_weight=0.0,
        )
        outputs = model(eeg, fnirs)

        self.assertGreater(float(outputs['source_coupling_local_residual_loss'].item()), 0.0)
        outputs['loss'].backward()
        eeg_grad = sum(
            float(param.grad.abs().sum().item())
            for name, param in model.named_parameters()
            if name.startswith('eeg_source_proj') and param.grad is not None
        )
        fnirs_grad = sum(
            float(param.grad.abs().sum().item())
            for name, param in model.named_parameters()
            if name.startswith('fnirs_source_proj') and param.grad is not None
        )
        self.assertGreater(eeg_grad, 0.0)
        self.assertGreaterEqual(fnirs_grad, 0.0)

    def test_context_residual_coupling_outputs_and_gradients_are_exposed(self):
        torch.manual_seed(39)
        eeg = torch.randn(2, 3, 40)
        fnirs = torch.randn(2, 4, 20)
        model = self._build_tiny_model(
            coupling_weight=0.2,
            coupling_lag_focus_weight=0.0,
            coupling_smoothness_weight=0.0,
            coupling_pair_likelihood_weight=0.0,
            coupling_context_residual_enabled=True,
            coupling_context_states=4,
            coupling_context_rank=2,
            coupling_context_gradient_target="fnirs",
        )

        self.assertEqual(tuple(model.context_coupling_eeg_factors.shape), (4, model.n_coupling_lags, 4, 2))
        self.assertEqual(tuple(model.context_coupling_fnirs_factors.shape), (4, model.n_coupling_lags, 4, 2))
        self.assertIsNotNone(model.context_coupling_router)

        outputs = model(eeg, fnirs)

        self.assertGreater(float(outputs['source_coupling_context_residual_loss'].item()), 0.0)
        self.assertEqual(tuple(outputs['source_coupling_context_probs'].shape), (2, 4))
        self.assertGreaterEqual(float(outputs['source_coupling_context_pair_likelihood_loss'].item()), 0.0)
        self.assertIn('source_coupling_context_entropy_loss', outputs)
        self.assertIn('source_coupling_context_balance_loss', outputs)
        self.assertIn('source_coupling_context_residual_l1_loss', outputs)
        self.assertIn('source_coupling_context_residual_loss', model.get_gradient_component_weights())

        outputs['loss'].backward()
        self.assertIsNotNone(model.context_coupling_eeg_factors.grad)
        self.assertIsNotNone(model.context_coupling_fnirs_factors.grad)
        self.assertGreater(float(model.context_coupling_eeg_factors.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(model.context_coupling_fnirs_factors.grad.abs().sum().item()), 0.0)
        router_grad = sum(
            float(param.grad.abs().sum().item())
            for name, param in model.named_parameters()
            if name.startswith('context_coupling_router') and param.grad is not None
        )
        self.assertGreater(router_grad, 0.0)

    def test_interaction_auxiliary_loss_is_configurable(self):
        torch.manual_seed(41)
        model = self._build_tiny_model(interaction_aux_weight=0.3)
        outputs = model(torch.randn(2, 3, 40), torch.randn(2, 4, 20))

        self.assertGreater(float(outputs['interaction_aux_loss'].item()), 0.0)
        self.assertAlmostEqual(model.get_gradient_component_weights()['interaction_aux_loss'], 0.3)

    def test_interaction_auxiliary_direct_gradient_reaches_both_source_paths(self):
        torch.manual_seed(43)
        model = self._build_tiny_model(
            interaction_aux_weight=0.3,
            interaction_aux_direction='eeg_to_fnirs',
            interaction_aux_stop_gradient=False,
        )
        outputs = model(torch.randn(2, 3, 40), torch.randn(2, 4, 20))
        outputs['interaction_aux_loss'].backward()

        eeg_grad = sum(
            float(param.grad.abs().sum().item())
            for name, param in model.named_parameters()
            if name.startswith('eeg_source_proj') and param.grad is not None
        )
        fnirs_grad = sum(
            float(param.grad.abs().sum().item())
            for name, param in model.named_parameters()
            if name.startswith('fnirs_source_proj') and param.grad is not None
        )
        self.assertGreater(eeg_grad, 0.0)
        self.assertGreater(fnirs_grad, 0.0)

    def test_shared_state_bottleneck_stop_gradient_default_preserves_old_behavior(self):
        torch.manual_seed(47)
        detached = self._build_tiny_model(shared_state_bottleneck_weight=0.3)
        outputs = detached(torch.randn(2, 3, 40), torch.randn(2, 4, 20))
        outputs['shared_state_bottleneck_loss'].backward()
        detached_fnirs_grad = sum(
            float(param.grad.abs().sum().item())
            for name, param in detached.named_parameters()
            if name.startswith('fnirs_shared_state_proj') and param.grad is not None
        )

        direct = self._build_tiny_model(
            shared_state_bottleneck_weight=0.3,
            shared_state_bottleneck_stop_gradient=False,
        )
        outputs = direct(torch.randn(2, 3, 40), torch.randn(2, 4, 20))
        outputs['shared_state_bottleneck_loss'].backward()
        direct_fnirs_grad = sum(
            float(param.grad.abs().sum().item())
            for name, param in direct.named_parameters()
            if name.startswith('fnirs_shared_state_proj') and param.grad is not None
        )

        self.assertAlmostEqual(detached_fnirs_grad, 0.0, places=6)
        self.assertGreater(direct_fnirs_grad, 0.0)

    def test_cross_modal_exchange_causal_mask_blocks_future_eeg_tokens(self):
        torch.manual_seed(51)
        adapter = CausalLowRankCrossAdapter(
            eeg_dim=6,
            fnirs_dim=5,
            rank=3,
            adapter_dim=4,
            max_lag_tokens=2,
            residual_init=0.1,
            dropout=0.0,
        )
        _, lag_weights = adapter(torch.randn(2, 4, 6), torch.randn(2, 4, 5), detach_context=False)

        self.assertEqual(tuple(lag_weights.shape), (2, 4, 3))
        self.assertTrue(torch.allclose(lag_weights[:, 0, 1:], torch.zeros_like(lag_weights[:, 0, 1:]), atol=1e-6))
        self.assertTrue(torch.allclose(lag_weights[:, 1, 2], torch.zeros_like(lag_weights[:, 1, 2]), atol=1e-6))

    def test_cross_modal_exchange_detach_context_blocks_eeg_gradient(self):
        torch.manual_seed(53)
        eeg = torch.randn(2, 3, 40)
        fnirs = torch.randn(2, 4, 20)
        detached = self._build_tiny_model(
            cross_modal_exchange_enabled=True,
            cross_modal_exchange_detach_context=True,
            cross_modal_exchange_rank=4,
            cross_modal_exchange_adapter_dim=8,
            cross_modal_exchange_max_lag_tokens=1,
            cross_modal_exchange_dropout=0.0,
        )
        latents = detached.encode_modalities(eeg, fnirs)
        latents['fnirs_source'].pow(2).mean().backward()
        detached_eeg_grad = sum(
            float(param.grad.abs().sum().item())
            for name, param in detached.named_parameters()
            if (name.startswith('eeg_patch_embed') or name.startswith('eeg_encoder')) and param.grad is not None
        )

        direct = self._build_tiny_model(
            cross_modal_exchange_enabled=True,
            cross_modal_exchange_detach_context=False,
            cross_modal_exchange_rank=4,
            cross_modal_exchange_adapter_dim=8,
            cross_modal_exchange_max_lag_tokens=1,
            cross_modal_exchange_dropout=0.0,
        )
        latents = direct.encode_modalities(eeg, fnirs)
        latents['fnirs_source'].pow(2).mean().backward()
        direct_eeg_grad = sum(
            float(param.grad.abs().sum().item())
            for name, param in direct.named_parameters()
            if (name.startswith('eeg_patch_embed') or name.startswith('eeg_encoder')) and param.grad is not None
        )
        direct_fnirs_grad = sum(
            float(param.grad.abs().sum().item())
            for name, param in direct.named_parameters()
            if name.startswith('fnirs_source_proj') and param.grad is not None
        )

        self.assertAlmostEqual(detached_eeg_grad, 0.0, places=6)
        self.assertGreater(direct_eeg_grad, 0.0)
        self.assertGreater(direct_fnirs_grad, 0.0)

    def test_cross_modal_exchange_updates_only_fnirs_source_branch(self):
        torch.manual_seed(55)
        eeg = torch.randn(2, 3, 40)
        fnirs = torch.randn(2, 4, 20)
        base = self._build_tiny_model()
        exchange = self._build_tiny_model(
            cross_modal_exchange_enabled=True,
            cross_modal_exchange_detach_context=False,
            cross_modal_exchange_rank=4,
            cross_modal_exchange_adapter_dim=8,
            cross_modal_exchange_max_lag_tokens=1,
            cross_modal_exchange_dropout=0.0,
        )
        exchange.load_state_dict(base.state_dict(), strict=False)
        base.eval()
        exchange.eval()

        with torch.no_grad():
            base_latents = base.encode_modalities(eeg, fnirs)
            exchange_latents = exchange.encode_modalities(eeg, fnirs)

        self.assertTrue(torch.allclose(base_latents['eeg_source'], exchange_latents['eeg_source'], atol=1e-6))
        self.assertTrue(torch.allclose(base_latents['eeg_observation'], exchange_latents['eeg_observation'], atol=1e-6))
        self.assertTrue(torch.allclose(base_latents['fnirs_observation'], exchange_latents['fnirs_observation'], atol=1e-6))
        self.assertGreater(float(exchange_latents['cross_modal_exchange_update'].abs().sum().item()), 0.0)
        self.assertAlmostEqual(float(exchange.cross_modal_exchange.residual_gate.item()), 0.1, places=6)

    def test_cross_modal_exchange_config_and_gradient_group_are_exposed(self):
        config = {
            'model': {
                'eeg': {
                    'seq_length': 40,
                    'patch_size': 20,
                    'channels': 3,
                    'encoder_embed_dim': 16,
                    'encoder_depth': 1,
                    'encoder_num_heads': 4,
                    'decoder_embed_dim': 16,
                    'decoder_depth': 1,
                    'decoder_num_heads': 4,
                },
                'fnirs': {
                    'seq_length': 20,
                    'patch_size': 10,
                    'channels': 4,
                    'encoder_embed_dim': 12,
                    'encoder_depth': 1,
                    'encoder_num_heads': 4,
                    'decoder_embed_dim': 12,
                    'decoder_depth': 1,
                    'decoder_num_heads': 4,
                },
                'source': {'codebook_size': 4, 'codebook_dim': 8},
                'eeg_observation': {'codebook_size': 4, 'codebook_dim': 8},
                'fnirs_observation': {'codebook_size': 4, 'codebook_dim': 8},
                'quantizer': {'kmeans_init': False, 'revive_dead_codes': False},
                'cross_modal_exchange': {
                    'enabled': True,
                    'mode': 'low_rank_causal_adapter',
                    'direction': 'eeg_to_fnirs',
                    'target_branch': 'fnirs_source',
                    'rank': 4,
                    'adapter_dim': 8,
                    'max_lag_tokens': 1,
                    'detach_context': False,
                    'dropout': 0.0,
                },
                'drop_path': 0.0,
                'dropout': 0.0,
            },
            'loss': {},
        }
        model = SourceObservationLaBraMVQNSP.from_config(config)
        outputs = model(torch.randn(2, 3, 40), torch.randn(2, 4, 20))

        self.assertIn('cross_modal_exchange_update_norm', outputs)
        self.assertGreater(float(outputs['cross_modal_exchange_update_norm'].item()), 0.0)
        group_names = [group['name'] for group in model.get_gradient_parameter_group_specs()]
        self.assertIn('cross_modal_exchange', group_names)

    def test_full_fusion_causal_and_bidirectional_masks(self):
        torch.manual_seed(57)
        causal = LagAwareCrossModalFusion(
            eeg_dim=8,
            fnirs_dim=8,
            embed_dim=8,
            depth=1,
            num_heads=2,
            max_lag_tokens=1,
            relative_lag_bias=True,
            dropout=0.0,
            mode='causal_cross_attention',
        )
        outputs = causal(torch.randn(2, 4, 8), torch.randn(2, 4, 8))
        weights = outputs['fnirs_attention'][:, 0]
        self.assertTrue(torch.allclose(weights[:, :, 0, 1:], torch.zeros_like(weights[:, :, 0, 1:]), atol=1e-7))
        self.assertTrue(torch.allclose(weights[:, :, 2, 0], torch.zeros_like(weights[:, :, 2, 0]), atol=1e-7))
        self.assertEqual(outputs['eeg_attention'].numel(), 0)

        bidirectional = LagAwareCrossModalFusion(
            eeg_dim=8,
            fnirs_dim=8,
            embed_dim=8,
            depth=1,
            num_heads=2,
            max_lag_tokens=1,
            relative_lag_bias=True,
            dropout=0.0,
            mode='bidirectional_cross_attention',
        )
        outputs = bidirectional(torch.randn(2, 4, 8), torch.randn(2, 4, 8))
        reverse = outputs['eeg_attention'][:, 0]
        self.assertTrue(torch.allclose(reverse[:, :, 0, 2:], torch.zeros_like(reverse[:, :, 0, 2:]), atol=1e-7))
        self.assertTrue(torch.allclose(reverse[:, :, 2, 0], torch.zeros_like(reverse[:, :, 2, 0]), atol=1e-7))

    def test_full_fusion_leaves_observation_inputs_unchanged(self):
        torch.manual_seed(59)
        baseline = self._build_tiny_model()
        fusion = self._build_tiny_model(
            cross_modal_fusion_enabled=True,
            cross_modal_fusion_mode='bidirectional_cross_attention',
            cross_modal_fusion_embed_dim=8,
            cross_modal_fusion_depth=1,
            cross_modal_fusion_num_heads=2,
            cross_modal_fusion_max_lag_tokens=1,
            cross_modal_fusion_dropout=0.0,
        )
        fusion.load_state_dict(baseline.state_dict(), strict=False)
        baseline.eval()
        fusion.eval()
        eeg = torch.randn(2, 3, 40)
        fnirs = torch.randn(2, 4, 20)
        with torch.no_grad():
            baseline_latents = baseline.encode_modalities(eeg, fnirs)
            fusion_latents = fusion.encode_modalities(eeg, fnirs)
        self.assertTrue(torch.allclose(baseline_latents['eeg_observation'], fusion_latents['eeg_observation']))
        self.assertTrue(torch.allclose(baseline_latents['fnirs_observation'], fusion_latents['fnirs_observation']))
        self.assertFalse(torch.allclose(baseline_latents['fnirs_source'], fusion_latents['fnirs_source']))

    def test_shared_joint_quantizer_updates_once_and_is_order_invariant(self):
        torch.manual_seed(61)
        left = self._build_tiny_model(source_codebook_mode='shared_joint')
        right = self._build_tiny_model(source_codebook_mode='shared_joint')
        right.load_state_dict(left.state_dict())
        eeg_source = torch.randn(3, 2, 8)
        fnirs_source = torch.randn(3, 2, 8)
        left._quantize_source_pair(eeg_source, fnirs_source)
        right._quantize_source_pair(fnirs_source, eeg_source)
        self.assertEqual(int(left.shared_source_quantizer.update_count.item()), 1)
        self.assertEqual(int(right.shared_source_quantizer.update_count.item()), 1)
        self.assertTrue(torch.allclose(left.shared_source_quantizer.weight, right.shared_source_quantizer.weight, atol=1e-6))
        self.assertTrue(torch.allclose(left.shared_source_quantizer.cluster_size, right.shared_source_quantizer.cluster_size))

    def test_full_alignment_losses_are_finite_and_reach_fusion(self):
        torch.manual_seed(63)
        model = self._build_tiny_model(
            cross_modal_fusion_enabled=True,
            cross_modal_fusion_mode='causal_cross_attention',
            cross_modal_fusion_embed_dim=8,
            cross_modal_fusion_depth=1,
            cross_modal_fusion_num_heads=2,
            cross_modal_fusion_max_lag_tokens=1,
            cross_modal_fusion_dropout=0.0,
            cross_modal_temporal_nce_weight=0.2,
            cross_modal_masked_latent_weight=0.2,
            cross_modal_soft_code_weight=0.05,
            cross_modal_positive_lag_weights=(0.0, 1.0),
        )
        outputs = model(torch.randn(4, 3, 40), torch.randn(4, 4, 20))
        for key in (
            'cross_modal_temporal_nce_loss',
            'cross_modal_masked_latent_loss',
            'cross_modal_soft_code_distillation_loss',
        ):
            self.assertTrue(torch.isfinite(outputs[key]))
        outputs['cross_modal_alignment_weighted_loss'].backward()
        fusion_grad = sum(
            float(param.grad.abs().sum().item())
            for param in model.cross_modal_fusion.parameters()
            if param.grad is not None
        )
        self.assertGreater(fusion_grad, 0.0)
        model.eval()
        with torch.no_grad():
            evaluation = model(torch.randn(4, 3, 40), torch.randn(4, 4, 20))
        self.assertTrue(torch.isfinite(evaluation['cross_modal_masked_shuffled_latent_loss']))
        self.assertTrue(torch.isfinite(evaluation['cross_modal_masked_pairing_gain']))

    def test_alignment_gradient_controller_respects_scale_bounds(self):
        torch.manual_seed(65)
        model = self._build_tiny_model(
            cross_modal_fusion_enabled=True,
            cross_modal_fusion_embed_dim=8,
            cross_modal_fusion_depth=1,
            cross_modal_fusion_num_heads=2,
            cross_modal_fusion_dropout=0.0,
            cross_modal_temporal_nce_weight=0.2,
            cross_modal_positive_lag_weights=(0.0, 1.0),
        )
        outputs = model(torch.randn(4, 3, 40), torch.randn(4, 4, 20))
        metrics = update_cross_modal_gradient_scale(model, outputs, {
            'training': {'alignment_gradient_control': {
                'enabled': True,
                'target_ratio': 0.3,
                'min_scale': 0.1,
                'max_scale': 2.0,
                'ema': 0.0,
            }}
        })
        self.assertIn('alignment_gradient_ratio', metrics)
        self.assertGreaterEqual(model.get_cross_modal_gradient_scale(), 0.1)
        self.assertLessEqual(model.get_cross_modal_gradient_scale(), 2.0)

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
