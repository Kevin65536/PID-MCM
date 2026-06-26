import json
import tempfile
import unittest
from pathlib import Path

from experiments.scripts.finalize_tokenizer_cross_modal_exchange_suite import main as finalize_main
from experiments.scripts.launch_tokenizer_cross_modal_exchange_suite import CONDITIONS, initialize_suite


class TokenizerCrossModalExchangeSuiteTests(unittest.TestCase):
    def test_launcher_generates_x0_to_x4_configs_and_queues(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_dir = Path(tmpdir) / "20260626_causal_cross_adapter_v1"
            manifest = initialize_suite(suite_dir, formal_epochs=40, seeds=(20260651, 20260652))

            self.assertEqual(manifest["schema_version"], "tokenizer_cross_modal_exchange_suite_v1")
            self.assertEqual([condition["key"] for condition in manifest["conditions"]], [c.key for c in CONDITIONS])
            self.assertEqual(len(manifest["formal_configs"]), 10)
            self.assertTrue((suite_dir / "queue_logs/smoke_gpu0.sh").exists())
            self.assertTrue((suite_dir / "queue_logs/formal_gpu1.sh").exists())
            self.assertTrue((suite_dir / "queue_logs/audit_queue.sh").exists())

            formal_queue = (suite_dir / "queue_logs/formal_gpu0.sh").read_text(encoding="utf-8")
            self.assertIn("audit_queue.sh", formal_queue)

            import yaml

            x1 = yaml.safe_load(
                (suite_dir / "tokenizer_interventions/configs/k128_dim128_x1_causal_exchange_seed20260651.yaml")
                .read_text(encoding="utf-8")
            )
            x2 = yaml.safe_load(
                (suite_dir / "tokenizer_interventions/configs/k128_dim128_x2_causal_exchange_seed20260651.yaml")
                .read_text(encoding="utf-8")
            )
            x3 = yaml.safe_load(
                (suite_dir / "tokenizer_interventions/configs/k128_dim128_x3_causal_exchange_seed20260651.yaml")
                .read_text(encoding="utf-8")
            )
            x4 = yaml.safe_load(
                (suite_dir / "tokenizer_interventions/configs/k128_dim128_x4_causal_exchange_seed20260651.yaml")
                .read_text(encoding="utf-8")
            )

            self.assertTrue(x1["model"]["cross_modal_exchange"]["enabled"])
            self.assertTrue(x1["model"]["cross_modal_exchange"]["detach_context"])
            self.assertFalse(x2["model"]["cross_modal_exchange"]["detach_context"])
            self.assertEqual(x2["model"]["cross_modal_exchange"]["target_branch"], "fnirs_source")
            self.assertEqual(x3["loss"]["coupling"]["pair_gradient_target"], "fnirs")
            self.assertTrue(x3["loss"]["coupling"]["residualize_fnirs_marginal"])
            self.assertFalse(x3["loss"]["coupling"]["context_residual_enabled"])
            self.assertTrue(x4["loss"]["coupling"]["context_residual_enabled"])
            self.assertEqual(x4["loss"]["coupling"]["context_gradient_target"], "fnirs")

    def test_finalizer_outputs_required_summary_categories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_dir = Path(tmpdir)
            for condition, primary in (("x0", 1.0), ("x2", 0.99)):
                run_dir = suite_dir / "tokenizer_interventions" / f"k128_dim128_{condition}_causal_exchange_seed20260651"
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
                                "val_cross_modal_exchange_update_norm": 0.01,
                                "grad_group_share_fnirs_rec_loss__cross_modal_exchange": 0.05,
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

            self.assertEqual(summary["schema_version"], "tokenizer_cross_modal_exchange_summary_v1")
            for key in (
                "information_retention",
                "fine_task_probe",
                "gate3_coupling",
                "gradient_conflict",
                "token_health",
                "decision",
            ):
                self.assertIn(key, summary["categories"])
            self.assertIn("X0", summary["conditions"])
            self.assertIn("X2", summary["conditions"])
            self.assertEqual(decision["status"], "complete")


if __name__ == "__main__":
    unittest.main()
