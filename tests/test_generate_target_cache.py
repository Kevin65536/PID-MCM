import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "croce_validation"
    / "scripts"
    / "generate_target_cache.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location("generate_target_cache", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {MODULE_PATH}")

GEN = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = GEN
MODULE_SPEC.loader.exec_module(GEN)


class _AuditStub:
    @staticmethod
    def predict_observations(particles, lead_field, jac_primary, jac_secondary, pair_mode):
        pred_eeg = 2.0 * particles[:, 4:5]
        pred_primary = particles[:, 2:3] + particles[:, 3:4]
        pred_secondary = particles[:, 3:4] - particles[:, 2:3]
        return pred_eeg, pred_primary, pred_secondary


class GenerateTargetCacheTests(unittest.TestCase):
    def test_build_cache_entry_uses_optical_channel_field_names(self):
        bundle = SimpleNamespace(
            normalization={},
            lead_field=np.asarray([1.0], dtype=np.float64),
            jac_primary=np.asarray([1.0], dtype=np.float64),
            jac_secondary=np.asarray([1.0], dtype=np.float64),
            pair_mode="wavelength",
            eeg_obs_raw=np.asarray([[5.0], [6.0], [7.0]], dtype=np.float64),
            fnirs_primary_obs_raw=np.asarray([[11.0], [12.0], [13.0]], dtype=np.float64),
            fnirs_secondary_obs_raw=np.asarray([[17.0], [18.0], [19.0]], dtype=np.float64),
        )
        pf_result = {
            "r_estimates_eeg": np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
            "state_estimates": np.asarray(
                [
                    [0.0, 0.0, 1.0, 2.0, 0.0],
                    [0.0, 0.0, 2.0, 3.0, 0.0],
                    [0.0, 0.0, 3.0, 4.0, 0.0],
                ],
                dtype=np.float64,
            ),
        }

        entry = GEN._build_cache_entry(bundle, pf_result, _AuditStub())

        self.assertIn("source_fnirs_optical_channel_0", entry)
        self.assertIn("obs_fnirs_optical_channel_1", entry)
        self.assertNotIn("source_fnirs_hbo", entry)

        np.testing.assert_allclose(entry["source_eeg"].ravel(), np.asarray([2.0, 4.0, 6.0], dtype=np.float32))
        np.testing.assert_allclose(
            entry["source_fnirs_optical_channel_0"].ravel(),
            np.asarray([3.0, 5.0, 7.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            entry["obs_fnirs_optical_channel_1"].ravel(),
            np.asarray([16.0, 17.0, 18.0], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()