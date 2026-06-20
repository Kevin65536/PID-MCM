import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.scripts.export_source_observation_tokens import prepare_export_config
from experiments.scripts.launch_tokenizer_next_stage_suite import training_config
from experiments.scripts.run_source_observation_token_downstream_probe import run_probe


class TokenizerNextStageSuiteTests(unittest.TestCase):
    def test_k128_dim_config_changes_source_dim_only(self):
        config = training_config(
            "toy_suite",
            dim=96,
            seed=20260623,
            device="cuda:0",
            smoke=False,
            cognitive_only=False,
        )

        self.assertEqual(config["model"]["source"]["codebook_size"], 128)
        self.assertEqual(config["model"]["source"]["codebook_dim"], 96)
        self.assertEqual(config["model"]["eeg_observation"]["codebook_size"], 256)
        self.assertEqual(config["model"]["fnirs_observation"]["codebook_size"], 128)
        self.assertEqual(config["loss"]["coupling"]["weight"], 0.0)
        self.assertNotIn("entry_filters", config["data"])

    def test_cognitive_config_filters_nback_wg(self):
        config = training_config(
            "toy_suite",
            dim=96,
            seed=20260625,
            device="cuda:0",
            smoke=False,
            cognitive_only=True,
        )

        filters = config["data"]["entry_filters"]
        self.assertEqual(filters["include_source_names"], ["simultaneous_cognitive"])
        self.assertEqual(filters["include_label_names"], ["nback", "wg"])

    def test_export_config_can_clear_checkpoint_entry_filters(self):
        config = {
            "training": {"batch_size": 2},
            "data": {
                "entry_filters": {"include_label_names": ["nback", "wg"]},
                "dataloader": {"drop_last": True},
            },
        }

        prepared = prepare_export_config(
            config,
            batch_size=4,
            num_workers=0,
            seed=1,
            train_random_crop=False,
            clear_entry_filters=True,
        )

        self.assertNotIn("entry_filters", prepared["data"])
        self.assertFalse(prepared["data"]["dataloader"]["drop_last"])

    def test_downstream_probe_trains_on_token_histograms(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "tokens_run"
            token_dir = run_dir / "tokens"
            token_dir.mkdir(parents=True)
            manifest = {
                "token_semantics": {
                    "eeg_source_vocab_size": 4,
                    "fnirs_source_vocab_size": 4,
                    "eeg_observation_vocab_size": 4,
                    "fnirs_observation_vocab_size": 4,
                }
            }
            (run_dir / "manifest.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")

            def write_split(split: str, labels):
                n = len(labels)
                base = np.asarray([0 if label == "nback" else 2 for label in labels], dtype=np.int64)
                arrays = {
                    "eeg_source_tokens": np.stack([base, base, base + 1], axis=1),
                    "fnirs_source_tokens": np.stack([base, base + 1, base + 1], axis=1),
                    "eeg_observation_tokens": np.stack([base, base, base + 1], axis=1),
                    "fnirs_observation_tokens": np.stack([base, base + 1, base + 1], axis=1),
                    "label_name": np.asarray(labels, dtype=str),
                    "source_task": np.asarray(["cognitive"] * n, dtype=str),
                    "subject_id": np.arange(n, dtype=np.int64),
                }
                np.savez(token_dir / f"{split}_tokens.npz", **arrays)

            write_split("train", ["nback", "wg", "nback", "wg", "nback", "wg"])
            write_split("val", ["nback", "wg"])
            write_split("test", ["nback", "wg"])

            output_dir = Path(tmpdir) / "probe"
            summary = run_probe(run_dir, output_dir, seed=7)
            rows = summary["rows"]
            self.assertTrue(rows)
            self.assertTrue(any(row["task"] == "nback_vs_wg" and row["split"] == "test" for row in rows))
            self.assertTrue((output_dir / "summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
