import argparse
import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "croce_validation"
    / "scripts"
    / "run_local_neighborhood_solver_audit.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location("run_local_neighborhood_solver_audit", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {MODULE_PATH}")

AUDIT = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = AUDIT
MODULE_SPEC.loader.exec_module(AUDIT)


class SingleTrialTaskSessionMappingTests(unittest.TestCase):
    def test_mental_arithmetic_session_zero_maps_to_raw_session_one(self):
        args = argparse.Namespace(task="mental_arithmetic", session_idx=0)

        task_session_idx, raw_session_idx, task = AUDIT.resolve_task_session_index(
            args,
            total_sessions=6,
        )

        self.assertEqual(task, "mental_arithmetic")
        self.assertEqual(task_session_idx, 0)
        self.assertEqual(raw_session_idx, 1)

    def test_motor_imagery_session_one_maps_to_second_mi_raw_session(self):
        args = argparse.Namespace(task="motor_imagery", session_idx=1)

        task_session_idx, raw_session_idx, task = AUDIT.resolve_task_session_index(
            args,
            total_sessions=6,
        )

        self.assertEqual(task, "motor_imagery")
        self.assertEqual(task_session_idx, 1)
        self.assertEqual(raw_session_idx, 2)


if __name__ == "__main__":
    unittest.main()
