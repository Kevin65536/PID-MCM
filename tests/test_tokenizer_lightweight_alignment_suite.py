import json
import tempfile
import unittest
from pathlib import Path

from experiments.scripts.finalize_tokenizer_lightweight_alignment_suite import main as finalize_main
from experiments.scripts.launch_tokenizer_lightweight_alignment_suite import CONDITIONS, initialize_suite


class TokenizerLightweightAlignmentSuiteTests(unittest.TestCase):
    def test_launcher_generates_a0_to_a5_configs_and_queues(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_dir = Path(tmpdir) / "20260624_causal_pre_vq_alignment_v1"
            manifest = initialize_suite(suite_dir, formal_epochs=40, seeds=(20260641, 20260642))

            self.assertEqual(manifest["schema_version"], "tokenizer_lightweight_alignment_suite_v1")
            self.assertEqual([condition["key"] for condition in manifest["conditions"]], [c.key for c in CONDITIONS])
            self.assertEqual(len(manifest["formal_configs"]), 12)
            self.assertTrue((suite_dir / "queue_logs/smoke_gpu0.sh").exists())
            self.assertTrue((suite_dir / "queue_logs/formal_gpu1.sh").exists())
            self.assertTrue((suite_dir / "queue_logs/audit_queue.sh").exists())

            a3_path = suite_dir / "tokenizer_interventions/configs/k128_dim128_a3_causal_alignment_seed20260641.yaml"
            a4_path = suite_dir / "tokenizer_interventions/configs/k128_dim128_a4_causal_alignment_seed20260641.yaml"
            a5_path = suite_dir / "tokenizer_interventions/configs/k128_dim128_a5_causal_alignment_seed20260641.yaml"
            self.assertTrue(a3_path.exists())
            self.assertTrue(a4_path.exists())
            self.assertTrue(a5_path.exists())

            import yaml

            a3 = yaml.safe_load(a3_path.read_text(encoding="utf-8"))
            a4 = yaml.safe_load(a4_path.read_text(encoding="utf-8"))
            a5 = yaml.safe_load(a5_path.read_text(encoding="utf-8"))

            self.assertEqual(a3["loss"]["interaction_aux"]["direction"], "eeg_to_fnirs")
            self.assertFalse(a3["loss"]["interaction_aux"]["stop_gradient"])
            self.assertEqual(a3["loss"]["coupling"]["pair_gradient_target"], "fnirs")
            self.assertTrue(a3["loss"]["coupling"]["residualize_fnirs_marginal"])
            self.assertFalse(a3["loss"]["coupling"]["context_residual_enabled"])
            self.assertTrue(a4["loss"]["coupling"]["context_residual_enabled"])
            self.assertEqual(a4["loss"]["coupling"]["context_gradient_target"], "fnirs")
            self.assertEqual(a5["loss"]["shared_state_bottleneck"]["weight"], 0.05)
            self.assertEqual(a5["loss"]["coupling"]["weight"], 0.0)

    def test_finalizer_outputs_required_summary_categories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_dir = Path(tmpdir)
            for condition, primary in (("a0", 1.0), ("a2", 0.99)):
                run_dir = suite_dir / "tokenizer_interventions" / f"k128_dim128_{condition}_causal_alignment_seed20260641"
                run_dir.mkdir(parents=True)
                metrics = {
                    "epochs": [
                        {
                            "epoch": 1,
                            "val_loss": primary,
                            "metrics": {
                                "val_primary_loss": primary,
                                "val_eeg_rec_loss": 0.7,
                                "val_fnirs_rec_loss": 0.2,
                                "val_eeg_source_perplexity": 80.0,
                                "val_fnirs_source_perplexity": 70.0,
                                "grad_cosine_eeg_rec_loss__vs__interaction_aux_loss": 0.1,
                                "grad_share_interaction_aux_loss": 0.05,
                            },
                        }
                    ]
                }
                (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

            import sys

            old_argv = sys.argv
            try:
                sys.argv = ["finalize", "--suite-dir", str(suite_dir)]
                finalize_main()
            finally:
                sys.argv = old_argv

            summary = json.loads((suite_dir / "summary.json").read_text(encoding="utf-8"))
            decision = json.loads((suite_dir / "decision.json").read_text(encoding="utf-8"))

            self.assertEqual(summary["schema_version"], "tokenizer_lightweight_alignment_summary_v1")
            for key in (
                "information_retention",
                "fine_task_probe",
                "gate3_coupling",
                "gradient_conflict",
                "token_health",
                "decision",
            ):
                self.assertIn(key, summary["categories"])
            self.assertIn("A0", summary["conditions"])
            self.assertIn("A2", summary["conditions"])
            self.assertEqual(decision["status"], "complete")


if __name__ == "__main__":
    unittest.main()
