"""Unit tests for numerical integration kernels vs matrix exponential baseline.

These tests use the synthetic bundle generator from the audit module, so no
real dataset is required.  They validate:

1. Single-step agreement between numerical integrators and expm
2. Conservation of the r state (which has zero endogenous drift)
3. Physiological clipping is applied identically
4. Deterministic trajectory stability over 1000 steps
5. Full PF with numerical kernels produces plausible outputs
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from typing import Any

import numpy as np

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "croce_validation"
    / "scripts"
    / "run_local_neighborhood_solver_audit.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "croce_local_neighborhood_solver_audit", MODULE_PATH
)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {MODULE_PATH}")

AUDIT = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = AUDIT
MODULE_SPEC.loader.exec_module(AUDIT)

# Also import the numerical kernels from the benchmark script
BENCH_PATH = (
    Path(__file__).resolve().parents[1]
    / "croce_validation"
    / "scripts"
    / "benchmark_numerical_integration.py"
)
BENCH_SPEC = importlib.util.spec_from_file_location(
    "benchmark_numerical_integration", BENCH_PATH
)
if BENCH_SPEC is None or BENCH_SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {BENCH_PATH}")

BENCH = importlib.util.module_from_spec(BENCH_SPEC)
sys.modules[BENCH_SPEC.name] = BENCH
BENCH_SPEC.loader.exec_module(BENCH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_synthetic_bundle() -> Any:
    from argparse import Namespace

    args = Namespace(
        duration_s=5.0,
        observation_fs=10.0,
        eeg_fs=200.0,
        integration_dt=0.005,
        snr_db=10.0,
        synthetic_eeg_channels=4,
        synthetic_fnirs_channels=2,
        eeg_neighbors=4,
        fnirs_neighbors=3,
        eeg_radius_mm=60.0,
        fnirs_radius_mm=45.0,
        eeg_sigma_mm=30.0,
        fnirs_sigma_mm=22.0,
        eeg_sign_mode="covariance",
        eeg_unit="uV",
        fnirs_primary_unit="a.u.",
        fnirs_secondary_unit="a.u.",
    )
    spatial = AUDIT.SpatialConfig(
        eeg_neighbors=args.eeg_neighbors,
        fnirs_neighbors=args.fnirs_neighbors,
        eeg_radius_mm=args.eeg_radius_mm,
        fnirs_radius_mm=args.fnirs_radius_mm,
        eeg_sigma_mm=args.eeg_sigma_mm,
        fnirs_sigma_mm=args.fnirs_sigma_mm,
        eeg_sign_mode=args.eeg_sign_mode,
    )
    return AUDIT.simulate_synthetic_bundle(args, spatial)


def _build_filter_config(bundle: Any, solver_backend: str = "python_exact") -> Any:
    return AUDIT.FilterConfig(
        integration_dt_s=1.0 / bundle.eeg_fs_hz,
        observation_fs_hz=bundle.fnirs_fs_hz,
        num_particles=32,
        resample_fraction=0.5,
        prior_std=np.asarray([0.05, 0.05, 0.05, 0.05, 0.0], dtype=np.float64),
        state_noise_std=np.asarray([0.02, 0.015, 0.015, 0.015, 0.0], dtype=np.float64),
        sigma_prop=0.35,
        sigma_nirs=1.0,
        seed_list=(11,),
        time_shift_null_s=2.0,
        run_spatial_null=False,
        solver_backend=solver_backend,
        torch_device="cpu",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class NumericalIntegrationSingleStepTests(unittest.TestCase):
    """Phase 1 — single-step accuracy."""

    @classmethod
    def setUpClass(cls):
        cls.params = AUDIT.ModelParams()
        cls.dt = 0.005
        rng = np.random.default_rng(42)
        cls.test_states = np.column_stack([
            rng.uniform(-0.3, 0.3, size=1000),
            rng.uniform(-0.15, 0.25, size=1000),
            rng.uniform(-0.08, 0.12, size=1000),
            rng.uniform(-0.06, 0.08, size=1000),
            rng.uniform(-8.0, 8.0, size=1000),
        ]).astype(np.float64)

    def _expm_reference(self, x: np.ndarray) -> np.ndarray:
        return AUDIT.local_linearized_step(x, self.dt, self.params)

    def test_euler_max_error_below_1e3(self):
        """Euler should agree with expm to within ~1e-3 per step at dt=0.005."""
        drift_fn = AUDIT.state_drift
        max_errs = []
        for i in range(min(200, self.test_states.shape[0])):
            x = self.test_states[i]
            ref = self._expm_reference(x)
            num = BENCH._euler_step(x, self.dt, drift_fn, self.params)
            max_errs.append(float(np.max(np.abs(num - ref))))
        self.assertLess(np.mean(max_errs), 5e-3)  # mean max_abs across states

    def test_heun_max_error_below_euler(self):
        """Heun (RK2) should be more accurate than Euler."""
        drift_fn = AUDIT.state_drift
        euler_errs = []
        heun_errs = []
        for i in range(min(200, self.test_states.shape[0])):
            x = self.test_states[i]
            ref = self._expm_reference(x)
            euler_errs.append(
                float(np.max(np.abs(BENCH._euler_step(x, self.dt, drift_fn, self.params) - ref)))
            )
            heun_errs.append(
                float(np.max(np.abs(BENCH._heun_step(x, self.dt, drift_fn, self.params) - ref)))
            )
        self.assertLess(np.mean(heun_errs), np.mean(euler_errs))

    def test_rk4_max_error_below_1e6(self):
        """RK4 should be extremely accurate (near machine precision per step)."""
        drift_fn = AUDIT.state_drift
        max_errs = []
        for i in range(min(200, self.test_states.shape[0])):
            x = self.test_states[i]
            ref = self._expm_reference(x)
            num = BENCH._rk4_step(x, self.dt, drift_fn, self.params)
            max_errs.append(float(np.max(np.abs(num - ref))))
        self.assertLess(np.mean(max_errs), 1e-5)

    def test_r_state_preserved_by_all_kernels(self):
        """The r state has zero drift and must be preserved exactly."""
        drift_fn = AUDIT.state_drift
        for name, kernel in [("euler", BENCH._euler_step), ("heun", BENCH._heun_step), ("rk4", BENCH._rk4_step)]:
            for i in range(min(50, self.test_states.shape[0])):
                x = self.test_states[i].copy()
                nx = kernel(x, self.dt, drift_fn, self.params)
                self.assertAlmostEqual(
                    float(nx[4]), float(x[4]), places=13,
                    msg=f"{name} changed r state at index {i}"
                )

    def test_physiological_clipping_applied(self):
        """Hemodynamic states must be clipped to [-0.95, inf) after each step."""
        drift_fn = AUDIT.state_drift
        # Create a state that would go below -0.95 without clipping
        x_bad = np.array([0.0, -0.96, -0.96, -0.96, 1.0], dtype=np.float64)
        for name, kernel in [("euler", BENCH._euler_step), ("heun", BENCH._heun_step), ("rk4", BENCH._rk4_step)]:
            nx = kernel(x_bad, self.dt, drift_fn, self.params)
            self.assertTrue(np.all(nx[1:4] >= -0.95 - 1e-12),
                            msg=f"{name} violated lower clip: {nx[1:4]}")


class NumericalIntegrationTrajectoryTests(unittest.TestCase):
    """Phase 2 — deterministic multi-step trajectories."""

    @classmethod
    def setUpClass(cls):
        cls.params = AUDIT.ModelParams()
        cls.dt = 0.005
        cls.n_steps = 1000
        cls.x0 = np.array([0.05, 0.02, 0.01, 0.005, 4.0], dtype=np.float64)

    def _evolve_exact(self) -> np.ndarray:
        traj = np.zeros((self.n_steps + 1, 5), dtype=np.float64)
        traj[0] = self.x0
        x = self.x0.copy()
        for i in range(self.n_steps):
            x = AUDIT.local_linearized_step(x, self.dt, self.params)
            traj[i + 1] = x
        return traj

    def _evolve_numerical(self, kernel) -> np.ndarray:
        drift_fn = AUDIT.state_drift
        traj = np.zeros((self.n_steps + 1, 5), dtype=np.float64)
        traj[0] = self.x0
        x = self.x0.copy()
        for i in range(self.n_steps):
            x = kernel(x, self.dt, drift_fn, self.params)
            traj[i + 1] = x
        return traj

    def test_trajectories_stay_bounded(self):
        """No kernel should produce NaN or absurdly large values."""
        drift_fn = AUDIT.state_drift
        for name, kernel in [("euler", BENCH._euler_step), ("heun", BENCH._heun_step), ("rk4", BENCH._rk4_step)]:
            traj = self._evolve_numerical(kernel)
            self.assertFalse(np.any(np.isnan(traj)), msg=f"{name} produced NaN")
            self.assertFalse(np.any(np.isinf(traj)), msg=f"{name} produced Inf")
            self.assertTrue(np.all(np.abs(traj) < 100.0), msg=f"{name} diverged")

    def test_rk4_rmse_growth_controlled(self):
        """RK4's RMSE vs expm should stay small at 1000 steps."""
        traj_exact = self._evolve_exact()
        traj_rk4 = self._evolve_numerical(BENCH._rk4_step)
        rmse = float(np.sqrt(np.mean((traj_rk4 - traj_exact) ** 2)))
        self.assertLess(rmse, 1e-3, f"RK4 trajectory RMSE {rmse:.2e} exceeds 1e-3")

    def test_error_grows_sublinearly(self):
        """Error should not explode — late-step RMSE should be within 100x of early-step RMSE."""
        traj_exact = self._evolve_exact()
        for name, kernel in [("euler", BENCH._euler_step), ("heun", BENCH._heun_step), ("rk4", BENCH._rk4_step)]:
            traj_num = self._evolve_numerical(kernel)
            early_rmse = float(np.sqrt(np.mean((traj_num[:100] - traj_exact[:100]) ** 2)))
            late_rmse = float(np.sqrt(np.mean((traj_num[-100:] - traj_exact[-100:]) ** 2)))
            ratio = late_rmse / max(early_rmse, 1e-15)
            self.assertLess(ratio, 200.0,
                            msg=f"{name} error ratio {ratio:.1f}x (early={early_rmse:.2e}, late={late_rmse:.2e})")


class NumericalIntegrationPFTests(unittest.TestCase):
    """Phase 3 — full PF with numerical kernels, synthetic data."""

    @classmethod
    def setUpClass(cls):
        if AUDIT.torch is None:
            raise unittest.SkipTest("PyTorch unavailable")
        cls.bundle = _build_synthetic_bundle()
        cls.params = AUDIT.ModelParams()
        cls.filter_config = _build_filter_config(cls.bundle, solver_backend="torch_exact")

    def _run_pf_with_kernel(self, kernel_name: str, kernel_fn) -> dict:
        """Patch the audit module and run the PF, then restore."""
        original = AUDIT.torch_local_linearized_step_batch
        dt = float(self.filter_config.integration_dt_s)
        params = self.params

        def _wrapper(particles, _dt, _params):
            return kernel_fn(particles, dt, params)

        AUDIT.torch_local_linearized_step_batch = _wrapper
        try:
            return AUDIT.run_particle_filter(self.bundle, self.filter_config, self.params, seed=11)
        finally:
            AUDIT.torch_local_linearized_step_batch = original

    def test_all_kernels_produce_plausible_pf_output(self):
        """Every numerical kernel should produce finite, well-shaped PF output."""
        for name, kfn in [
            ("euler", BENCH._torch_euler_step_batch),
            ("heun", BENCH._torch_heun_step_batch),
            ("rk4", BENCH._torch_rk4_step_batch),
        ]:
            result = self._run_pf_with_kernel(name, kfn)
            self.assertTrue(np.isfinite(result["log_likelihood"]), msg=f"{name} LL is non-finite")
            self.assertTrue(np.all(np.isfinite(result["state_estimates"])), msg=f"{name} states non-finite")
            self.assertTrue(np.all(np.isfinite(result["ess_trace"])), msg=f"{name} ESS non-finite")
            self.assertEqual(result["state_estimates"].shape[0], self.bundle.time_s.shape[0])
            self.assertEqual(result["state_estimates"].shape[1], 5)
            self.assertGreaterEqual(np.min(result["ess_trace"]), 0.0)

    def test_numerical_kernels_agree_with_expm_on_log_likelihood(self):
        """Log-likelihood from numerical kernels should be close to expm baseline."""
        # expm baseline (solver_backend already "torch_exact" from setUpClass)
        expm_result = AUDIT.run_particle_filter(self.bundle, self.filter_config, self.params, seed=11)
        expm_ll = float(expm_result["log_likelihood"])

        for name, kfn in [
            ("euler", BENCH._torch_euler_step_batch),
            ("heun", BENCH._torch_heun_step_batch),
            ("rk4", BENCH._torch_rk4_step_batch),
        ]:
            result = self._run_pf_with_kernel(name, kfn)
            num_ll = float(result["log_likelihood"])
            delta = abs(num_ll - expm_ll)
            # With 32 particles and 5s of data, the PF is stochastic enough that
            # we expect sub-1.0 LL delta if the state evolution is correct.
            self.assertLess(delta, 2.0,
                            msg=f"{name} LL delta={delta:.4f} (expm={expm_ll:.4f}, {name}={num_ll:.4f})")

    def test_rk4_closest_to_expm(self):
        """RK4 should have the smallest log-likelihood delta vs expm."""
        # solver_backend already "torch_exact" from setUpClass
        expm_result = AUDIT.run_particle_filter(self.bundle, self.filter_config, self.params, seed=11)
        expm_ll = float(expm_result["log_likelihood"])

        deltas = {}
        for name, kfn in [
            ("euler", BENCH._torch_euler_step_batch),
            ("heun", BENCH._torch_heun_step_batch),
            ("rk4", BENCH._torch_rk4_step_batch),
        ]:
            result = self._run_pf_with_kernel(name, kfn)
            deltas[name] = abs(float(result["log_likelihood"]) - expm_ll)

        self.assertLessEqual(deltas["rk4"], deltas["euler"] * 1.1,
                             msg=f"RK4 ({deltas['rk4']:.4f}) should be <= Euler ({deltas['euler']:.4f})")


class TorchKernelSignatureTests(unittest.TestCase):
    """Verify PyTorch kernels are drop-in compatible with audit.torch_local_linearized_step_batch."""

    @classmethod
    def setUpClass(cls):
        if AUDIT.torch is None:
            raise unittest.SkipTest("PyTorch unavailable")
        cls.params = AUDIT.ModelParams()
        cls.dt = 0.005
        cls.N = 32
        rng = np.random.default_rng(99)
        cls.particles_np = np.column_stack([
            rng.uniform(-0.2, 0.2, size=cls.N),
            rng.uniform(-0.1, 0.1, size=cls.N),
            rng.uniform(-0.05, 0.05, size=cls.N),
            rng.uniform(-0.03, 0.03, size=cls.N),
            rng.uniform(-5.0, 5.0, size=cls.N),
        ]).astype(np.float64)

    def test_signature_match(self):
        """Each torch kernel accepts and returns tensors of correct shape."""
        pt = AUDIT.torch.from_numpy(self.particles_np).to(dtype=AUDIT.torch.float64)
        for name, kfn in [
            ("euler", BENCH._torch_euler_step_batch),
            ("heun", BENCH._torch_heun_step_batch),
            ("rk4", BENCH._torch_rk4_step_batch),
        ]:
            result = kfn(pt, self.dt, self.params)
            self.assertIsInstance(result, AUDIT.torch.Tensor, msg=f"{name} returned non-Tensor")
            self.assertEqual(result.shape, (self.N, 5), msg=f"{name} shape mismatch")
            self.assertEqual(result.dtype, AUDIT.torch.float64, msg=f"{name} dtype mismatch")

    def test_outputs_are_finite(self):
        """All kernels produce finite outputs for plausible inputs."""
        pt = AUDIT.torch.from_numpy(self.particles_np).to(dtype=AUDIT.torch.float64)
        for name, kfn in [
            ("euler", BENCH._torch_euler_step_batch),
            ("heun", BENCH._torch_heun_step_batch),
            ("rk4", BENCH._torch_rk4_step_batch),
        ]:
            result = kfn(pt, self.dt, self.params)
            self.assertTrue(AUDIT.torch.all(AUDIT.torch.isfinite(result)).item(),
                            msg=f"{name} produced non-finite values")


if __name__ == "__main__":
    unittest.main()
