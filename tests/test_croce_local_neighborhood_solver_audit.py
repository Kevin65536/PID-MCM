import importlib.util
import sys
import tempfile
import unittest
from unittest import mock
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / 'croce_validation' / 'scripts' / 'run_local_neighborhood_solver_audit.py'
MODULE_SPEC = importlib.util.spec_from_file_location('croce_local_neighborhood_solver_audit', MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f'Unable to load module spec from {MODULE_PATH}')

AUDIT = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = AUDIT
MODULE_SPEC.loader.exec_module(AUDIT)


class CroceLocalNeighborhoodSolverAuditTests(unittest.TestCase):
    def build_args(self) -> Namespace:
        return Namespace(
            duration_s=6.0,
            observation_fs=10.0,
            eeg_fs=200.0,
            integration_dt=0.005,
            snr_db=5.0,
            synthetic_eeg_channels=8,
            synthetic_fnirs_channels=4,
            eeg_neighbors=4,
            fnirs_neighbors=3,
            eeg_radius_mm=60.0,
            fnirs_radius_mm=45.0,
            eeg_sigma_mm=30.0,
            fnirs_sigma_mm=22.0,
            eeg_sign_mode='covariance',
            eeg_unit='uV',
            fnirs_primary_unit='a.u.',
            fnirs_secondary_unit='a.u.',
        )

    def build_spatial_config(self) -> Any:
        args = self.build_args()
        return AUDIT.SpatialConfig(
            eeg_neighbors=args.eeg_neighbors,
            fnirs_neighbors=args.fnirs_neighbors,
            eeg_radius_mm=args.eeg_radius_mm,
            fnirs_radius_mm=args.fnirs_radius_mm,
            eeg_sigma_mm=args.eeg_sigma_mm,
            fnirs_sigma_mm=args.fnirs_sigma_mm,
            eeg_sign_mode=args.eeg_sign_mode,
        )

    def build_filter_config(self, bundle: Any) -> Any:
        return AUDIT.FilterConfig(
            integration_dt_s=1.0 / bundle.eeg_fs_hz,
            observation_fs_hz=bundle.fnirs_fs_hz,
            num_particles=32,
            resample_fraction=0.5,
            prior_std=np.asarray([0.05, 0.05, 0.05, 0.05, 0.0], dtype=np.float64),
            state_noise_std=np.asarray([0.02, 0.015, 0.015, 0.015, 0.0], dtype=np.float64),
            sigma_prop=0.35,
            sigma_nirs=1.0,
            seed_list=(11, 23),
            time_shift_null_s=2.0,
            run_spatial_null=False,
            solver_backend='python_exact',
            torch_device='cpu',
        )

    def test_synthetic_bundle_keeps_dual_rate_structure(self):
        bundle = AUDIT.simulate_synthetic_bundle(self.build_args(), self.build_spatial_config())

        self.assertEqual(bundle.eeg_obs.shape[0], bundle.eeg_time_s.shape[0])
        self.assertEqual(bundle.fnirs_primary_obs.shape[0], bundle.time_s.shape[0])
        self.assertEqual(bundle.fnirs_secondary_obs.shape[0], bundle.time_s.shape[0])
        self.assertEqual(bundle.eeg_time_s.shape[0], bundle.time_s.shape[0] * bundle.eeg_substeps_per_fnirs)
        self.assertEqual(bundle.r_eeg_projection.shape[0], bundle.eeg_time_s.shape[0])

    def test_particle_filter_reports_new_recovery_metrics(self):
        bundle = AUDIT.simulate_synthetic_bundle(self.build_args(), self.build_spatial_config())
        filter_config = self.build_filter_config(bundle)

        result = AUDIT.run_particle_filter(bundle, filter_config, AUDIT.ModelParams(), seed=11)
        metrics = AUDIT.compute_fit_metrics(bundle, result, filter_config)
        reproducibility = AUDIT.summarise_seed_reproducibility(
            [
                result,
                AUDIT.run_particle_filter(bundle, filter_config, AUDIT.ModelParams(), seed=23),
            ]
        )

        self.assertEqual(result['pred_eeg'].shape, bundle.eeg_obs.shape)
        self.assertEqual(result['pred_primary'].shape, bundle.fnirs_primary_obs.shape)
        self.assertEqual(result['pred_secondary'].shape, bundle.fnirs_secondary_obs.shape)
        self.assertEqual(result['r_estimates_eeg'].shape, bundle.r_eeg_projection.shape)
        self.assertEqual(result['ess_trace'].shape[0], bundle.time_s.shape[0])

        for metric_name in (
            'eeg_corr_mean',
            'fnirs_primary_corr_mean',
            'fnirs_secondary_corr_mean',
            'r_alpha_delta_power_ratio',
            'r_low_modification_ratio',
            'r_total_modification_ratio',
            'r_projection_corr',
            'r_norm_rmse',
            'r_high_corr',
            'r_low_corr',
        ):
            self.assertIn(metric_name, metrics)
            self.assertTrue(np.isfinite(metrics[metric_name]))

        self.assertEqual(reproducibility['seed_pairwise_count'], 1)
        self.assertTrue(np.isfinite(reproducibility['seed_pairwise_r_corr_mean']))

    def test_torch_exact_backend_matches_python_exact(self):
        if AUDIT.torch is None:
            self.skipTest('PyTorch is unavailable in this environment')

        bundle = AUDIT.simulate_synthetic_bundle(self.build_args(), self.build_spatial_config())
        python_filter = self.build_filter_config(bundle)
        torch_filter = AUDIT.FilterConfig(
            integration_dt_s=python_filter.integration_dt_s,
            observation_fs_hz=python_filter.observation_fs_hz,
            num_particles=python_filter.num_particles,
            resample_fraction=python_filter.resample_fraction,
            prior_std=python_filter.prior_std.copy(),
            state_noise_std=python_filter.state_noise_std.copy(),
            sigma_prop=python_filter.sigma_prop,
            sigma_nirs=python_filter.sigma_nirs,
            seed_list=python_filter.seed_list,
            time_shift_null_s=python_filter.time_shift_null_s,
            run_spatial_null=python_filter.run_spatial_null,
            solver_backend='torch_exact',
            torch_device='cpu',
        )

        python_result = AUDIT.run_particle_filter_python_exact(bundle, python_filter, AUDIT.ModelParams(), seed=11)
        torch_result = AUDIT.run_particle_filter(bundle, torch_filter, AUDIT.ModelParams(), seed=11)

        for key in ('state_estimates', 'state_std', 'ess_trace', 'pred_eeg', 'pred_primary', 'pred_secondary', 'r_estimates_eeg', 'r_std_eeg'):
            self.assertTrue(np.allclose(python_result[key], torch_result[key], atol=1e-10, rtol=1e-10), msg=key)
        self.assertAlmostEqual(float(python_result['log_likelihood']), float(torch_result['log_likelihood']), places=10)

    def test_target_time_frequency_diagnostics_plot_is_written(self):
        bundle = AUDIT.simulate_synthetic_bundle(self.build_args(), self.build_spatial_config())
        filter_config = self.build_filter_config(bundle)
        result = AUDIT.run_particle_filter(bundle, filter_config, AUDIT.ModelParams(), seed=11)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / 'target_time_frequency_diagnostics.png'
            AUDIT.plot_target_time_frequency_diagnostics(output_path, bundle, result)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_target_time_frequency_diagnostics_keeps_eeg_and_fnirs_psd_ranges_separate(self):
        bundle = AUDIT.simulate_synthetic_bundle(self.build_args(), self.build_spatial_config())
        filter_config = self.build_filter_config(bundle)
        result = AUDIT.run_particle_filter(bundle, filter_config, AUDIT.ModelParams(), seed=11)
        captured_axes: list[Any] = []

        original_subplots = AUDIT.plt.subplots

        def capture_subplots(*args: Any, **kwargs: Any) -> Any:
            fig, axes = original_subplots(*args, **kwargs)
            captured_axes.append(axes)
            return fig, axes

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / 'target_time_frequency_diagnostics.png'
            with mock.patch.object(AUDIT.plt, 'subplots', side_effect=capture_subplots):
                AUDIT.plot_target_time_frequency_diagnostics(output_path, bundle, result)

        self.assertEqual(len(captured_axes), 1)
        axes = captured_axes[0]
        eeg_psd_xlim = axes[0, 1].get_xlim()
        fnirs_psd_xlim = axes[1, 1].get_xlim()
        self.assertGreater(eeg_psd_xlim[1], 10.0)
        self.assertLessEqual(fnirs_psd_xlim[1], 0.5 + 1e-9)

    def test_target_psd_comparison_plot_is_written(self):
        bundle = AUDIT.simulate_synthetic_bundle(self.build_args(), self.build_spatial_config())
        filter_config = self.build_filter_config(bundle)
        result = AUDIT.run_particle_filter(bundle, filter_config, AUDIT.ModelParams(), seed=11)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / 'target_psd_comparison.png'
            AUDIT.plot_target_psd_comparison(output_path, bundle, result)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()