import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "croce_validation"
    / "scripts"
    / "analyze_target_cache.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location("analyze_target_cache", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {MODULE_PATH}")

ANALYZE = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = ANALYZE
MODULE_SPEC.loader.exec_module(ANALYZE)


def _write_subject_cache(cache_path: Path, field_mode: str = "optical") -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if field_mode == "primary_secondary":
        payload = {
            "AF7Fp1/source_eeg": np.zeros((400, 2), dtype=np.float32),
            "AF7Fp1/obs_eeg": np.ones((400, 2), dtype=np.float32),
            "AF7Fp1/source_fnirs_primary": np.zeros((20, 1), dtype=np.float32),
            "AF7Fp1/obs_fnirs_primary": np.ones((20, 1), dtype=np.float32),
            "AF7Fp1/source_fnirs_secondary": np.zeros((20, 1), dtype=np.float32),
            "AF7Fp1/obs_fnirs_secondary": np.ones((20, 1), dtype=np.float32),
        }
    elif field_mode == "concentration":
        payload = {
            "AF7Fp1/source_eeg": np.zeros((400, 2), dtype=np.float32),
            "AF7Fp1/obs_eeg": np.ones((400, 2), dtype=np.float32),
            "AF7Fp1/source_fnirs_hbo": np.zeros((20, 1), dtype=np.float32),
            "AF7Fp1/obs_fnirs_hbo": np.ones((20, 1), dtype=np.float32),
            "AF7Fp1/source_fnirs_hbr": np.zeros((20, 1), dtype=np.float32),
            "AF7Fp1/obs_fnirs_hbr": np.ones((20, 1), dtype=np.float32),
        }
    else:
        payload = {
            "AF7Fp1/source_eeg": np.zeros((400, 2), dtype=np.float32),
            "AF7Fp1/obs_eeg": np.ones((400, 2), dtype=np.float32),
            "AF7Fp1/source_fnirs_optical_channel_0": np.zeros((20, 1), dtype=np.float32),
            "AF7Fp1/obs_fnirs_optical_channel_0": np.ones((20, 1), dtype=np.float32),
            "AF7Fp1/source_fnirs_optical_channel_1": np.zeros((20, 1), dtype=np.float32),
            "AF7Fp1/obs_fnirs_optical_channel_1": np.ones((20, 1), dtype=np.float32),
        }
    np.savez_compressed(
        cache_path,
        **payload,
    )


class AnalyzeTargetCacheTests(unittest.TestCase):
    def test_find_cache_files_prefers_subject_subdirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            canonical = root / "subject_1" / "subject1_cache.npz"
            duplicate = root / "subject1_cache.npz"
            legacy_only = root / "subject2_cache.npz"

            _write_subject_cache(canonical)
            _write_subject_cache(duplicate)
            _write_subject_cache(legacy_only)

            files = ANALYZE.find_cache_files(root)

            self.assertEqual(files[1], canonical)
            self.assertEqual(files[2], legacy_only)

    def test_resolve_manifest_path_prefers_canonical_subject_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            canonical_cache = root / "subject_3" / "subject3_cache.npz"
            duplicate_cache = root / "subject3_cache.npz"
            manifest_path = root / "subject_3" / "cache_manifest.json"

            _write_subject_cache(canonical_cache)
            _write_subject_cache(duplicate_cache)
            manifest_path.write_text(
                json.dumps({"config": {"segment_duration_s": 2.0, "pair_mode": "wavelength", "pair_labels": ["highWL", "lowWL"]}}),
                encoding="utf-8",
            )

            self.assertEqual(ANALYZE.resolve_manifest_path(duplicate_cache), manifest_path)
            self.assertEqual(ANALYZE.resolve_manifest_path(canonical_cache), manifest_path)

    def test_build_time_axis_uses_segment_duration(self):
        axis = ANALYZE.build_time_axis(400, 2.0)

        np.testing.assert_allclose(axis[:3], np.asarray([0.0, 0.005, 0.010]))
        self.assertAlmostEqual(float(axis[-1]), 1.995)

    def test_load_cache_supports_legacy_primary_secondary_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "subject_1" / "subject1_cache.npz"
            _write_subject_cache(cache_path, field_mode="primary_secondary")

            data = ANALYZE.load_cache(cache_path, max_anchors=1)

            self.assertIn("source_fnirs_optical_channel_0", data)
            self.assertIn("obs_fnirs_optical_channel_1", data)
            self.assertEqual(data["source_fnirs_optical_channel_0"].shape[0], 20)

    def test_load_cache_supports_legacy_concentration_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "subject_1" / "subject1_cache.npz"
            _write_subject_cache(cache_path, field_mode="concentration")

            data = ANALYZE.load_cache(cache_path, max_anchors=1)

            self.assertIn("source_fnirs_optical_channel_0", data)
            self.assertIn("obs_fnirs_optical_channel_1", data)
            self.assertEqual(data["obs_fnirs_optical_channel_1"].shape[0], 20)

    def test_infer_fnirs_target_labels_uses_optical_contract(self):
        primary_title, secondary_title, note = ANALYZE.infer_fnirs_target_labels(
            "wavelength",
            ["highWL", "lowWL"],
        )

        self.assertEqual(primary_title, "fNIRS highWL Targets")
        self.assertEqual(secondary_title, "fNIRS lowWL Targets")
        self.assertIn("optical measurement space", note)



if __name__ == "__main__":
    unittest.main()