import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / 'croce_validation' / 'scripts' / 'benchmark_solver_optimizations.py'
MODULE_SPEC = importlib.util.spec_from_file_location('benchmark_solver_optimizations', MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f'Unable to load module spec from {MODULE_PATH}')

BENCH = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = BENCH
MODULE_SPEC.loader.exec_module(BENCH)


class BenchmarkSolverOptimizationsTests(unittest.TestCase):
    def test_parallel_pool_initializer_caps_worker_threads(self):
        fake_torch = mock.Mock()
        fake_torch.get_num_threads.return_value = 2
        fake_torch.get_num_interop_threads.return_value = 1

        with mock.patch.object(BENCH, 'threadpool_limits', autospec=True) as mock_limits, \
                mock.patch.object(BENCH, 'threadpool_info', autospec=True, return_value=[{'num_threads': 2}]), \
                mock.patch.object(BENCH, 'torch', fake_torch), \
                mock.patch.dict(os.environ, {}, clear=False):
            BENCH._parallel_pool_initializer(worker_threads=2, torch_interop_threads=1)

            self.assertEqual(os.environ['OMP_NUM_THREADS'], '2')
            self.assertEqual(os.environ['MKL_NUM_THREADS'], '2')
            self.assertEqual(os.environ['OPENBLAS_NUM_THREADS'], '2')
            self.assertEqual(os.environ['NUMEXPR_NUM_THREADS'], '2')

        mock_limits.assert_called_once_with(limits=2)
        fake_torch.set_num_threads.assert_called_once_with(2)
        fake_torch.set_num_interop_threads.assert_called_once_with(1)
        self.assertEqual(BENCH._WORKER_DIAGNOSTICS['blas_num_threads'], 2)
        self.assertEqual(BENCH._WORKER_DIAGNOSTICS['torch_num_threads'], 2)
        self.assertEqual(BENCH._WORKER_DIAGNOSTICS['torch_num_interop_threads'], 1)