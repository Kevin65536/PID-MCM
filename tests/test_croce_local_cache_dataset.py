import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data import CroceLocalCacheDataset, create_configured_multimodal_dataloaders


class CroceLocalCacheDatasetTests(unittest.TestCase):
    def _write_cache(self, root: Path) -> Path:
        subject_dir = root / "subject_1"
        subject_dir.mkdir(parents=True, exist_ok=True)
        prefix = "AF7_Fp1/event_000"
        eeg_steps = 10_000
        fnirs_steps = 500

        arrays = {
            f"{prefix}/source_eeg": np.ones((eeg_steps, 6), dtype=np.float32),
            f"{prefix}/obs_eeg": np.full((eeg_steps, 6), 2.0, dtype=np.float32),
            f"{prefix}/source_fnirs_optical_channel_0": np.full((fnirs_steps, 1), 3.0, dtype=np.float32),
            f"{prefix}/obs_fnirs_optical_channel_0": np.full((fnirs_steps, 1), 4.0, dtype=np.float32),
            f"{prefix}/source_fnirs_optical_channel_1": np.full((fnirs_steps, 1), 999.0, dtype=np.float32),
            f"{prefix}/obs_fnirs_optical_channel_1": np.full((fnirs_steps, 1), -999.0, dtype=np.float32),
        }
        cache_path = subject_dir / "subject1_cache.npz"
        np.savez(cache_path, **arrays)

        manifest = {
            "cache_file": cache_path.name,
            "config": {
                "subject_id": 1,
                "task": "mental_arithmetic",
                "pair_mode": "wavelength",
                "pair_labels": ["highWL", "lowWL"],
                "bundle_segment_label": "math",
            },
            "per_job_results": [
                {
                    "anchor": "AF7 Fp1",
                    "event_idx": 0,
                    "num_eeg_steps": eeg_steps,
                    "num_fnirs_steps": fnirs_steps,
                    "event_label_name_fnirs": "math",
                }
            ],
        }
        (subject_dir / "cache_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return subject_dir

    def test_dataset_uses_highwl_only_and_keeps_local_shapes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_cache(root)

            dataset = CroceLocalCacheDataset(
                cache_sources=[{"name": "toy", "root": str(root), "task": "mental_arithmetic"}],
                subject_ids=[1],
                split="val",
                crop_duration_s=20.0,
                eval_event_offsets_s=[0.0],
            )
            item = dataset[0]

            self.assertEqual(tuple(item["eeg"].shape), (6, 4000))
            self.assertEqual(tuple(item["fnirs"].shape), (1, 200))
            self.assertEqual(tuple(item["targets"]["eeg_source"].shape), (6, 4000))
            self.assertEqual(tuple(item["targets"]["eeg_observation"].shape), (6, 4000))
            self.assertEqual(tuple(item["targets"]["fnirs_source"].shape), (1, 200))
            self.assertEqual(tuple(item["targets"]["fnirs_observation"].shape), (1, 200))
            self.assertTrue(torch.allclose(item["fnirs"], torch.full((1, 200), 7.0)))
            self.assertTrue(torch.allclose(item["targets"]["fnirs_source"], torch.full((1, 200), 3.0)))
            self.assertFalse(torch.any(item["fnirs"] == 999.0))
            self.assertEqual(item["fnirs_component"], "highWL")

            gate0 = dataset.get_gate0_metadata()
            self.assertEqual(gate0["selected_fnirs_component"], "highWL")
            self.assertEqual(gate0["ignored_fnirs_components"], ["lowWL"])
            self.assertEqual(gate0["pair_mode"], "wavelength")
            self.assertEqual(gate0["pair_labels"], ["highWL", "lowWL"])
            self.assertEqual(gate0["fnirs_channels"], 1)

    def test_dataset_reuses_npz_handles_when_cache_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_cache(root)

            dataset = CroceLocalCacheDataset(
                cache_sources=[{"name": "toy", "root": str(root), "task": "mental_arithmetic"}],
                subject_ids=[1],
                split="val",
                crop_duration_s=20.0,
                eval_event_offsets_s=[0.0],
                cache_npz_handles=True,
            )
            _ = dataset[0]
            _ = dataset[0]

            self.assertEqual(len(dataset._npz_cache), 1)
            dataset.close()
            self.assertEqual(len(dataset._npz_cache), 0)

    def test_factory_creates_croce_local_dataloaders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_cache(root)
            config = {
                "data": {
                    "dataset": "croce_local_cache",
                    "data_root": str(root),
                    "cache_sources": [{"name": "toy", "root": str(root), "task": "mental_arithmetic"}],
                    "split": {
                        "train_subjects": [1],
                        "val_subjects": [1],
                        "test_subjects": [1],
                    },
                    "window": {"duration_s": 20.0},
                    "crop": {"train_random": False, "eval_event_offsets_s": [0.0]},
                    "num_workers": 0,
                    "dataloader": {"drop_last": True},
                },
                "training": {"batch_size": 2},
            }

            dataloaders = create_configured_multimodal_dataloaders(config)

            self.assertEqual(dataloaders["train"].dataset.get_num_fnirs_channels(), 1)
            self.assertEqual(dataloaders["val"].dataset.get_num_eeg_channels(), 6)
            batch = next(iter(dataloaders["val"]))
            self.assertEqual(tuple(batch["fnirs"].shape), (1, 1, 200))


if __name__ == "__main__":
    unittest.main()
