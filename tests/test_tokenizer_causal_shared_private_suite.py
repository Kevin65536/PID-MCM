import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from experiments.scripts.finalize_tokenizer_causal_shared_private_suite import main as finalize_main
from experiments.scripts.launch_tokenizer_causal_shared_private_suite import CONDITIONS, initialize_suite


class CausalSharedPrivateSuiteTests(unittest.TestCase):
    def test_launcher_generates_m0_through_m5(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            suite = Path(tmpdir) / "causal_shared_private_v1"
            manifest = initialize_suite(suite, screening_epochs=20, confirm_epochs=40)
            self.assertEqual([item["key"] for item in manifest["conditions"]], [item.key for item in CONDITIONS])
            self.assertEqual(len(manifest["configs"]["smoke"]), 6)
            self.assertEqual(len(manifest["configs"]["screening"]), 6)
            self.assertEqual(len(manifest["configs"]["confirmatory"]), 12)

            def config(key):
                path = next((suite / "screening/configs").glob(f"*_{key.lower()}_*.yaml"))
                return yaml.safe_load(path.read_text(encoding="utf-8"))

            m0, m3, m4, m5 = (config(key) for key in ("M0", "M3", "M4", "M5"))
            self.assertFalse(m0["model"]["cross_modal_fusion"]["apply_to_source"])
            self.assertIn("cross_modal_fusion.", m0["training"]["trainable_parameter_prefixes"])
            self.assertTrue(m3["model"]["cross_modal_token"]["enabled"])
            self.assertEqual(m3["model"]["source_codebook"]["mode"], "independent")
            self.assertTrue(m4["model"]["cross_modal_fusion"]["adaptive_lag_enabled"])
            self.assertEqual(m5["loss"]["coupling"]["weight"], 0.01)

    def test_finalizer_reads_cross_token_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            suite = Path(tmpdir)
            run = suite / "smoke/smoke_k128_dim128_m3_causal_shared_private_seed20260671"
            run.mkdir(parents=True)
            payload = {
                "epochs": [{
                    "epoch": 2,
                    "val_loss": 1.0,
                    "val_primary_loss": 0.8,
                    "val_cross_modal_alignment_unscaled_loss": 0.2,
                    "val_cross_token_pairing_gain": 0.1,
                    "val_cross_token_hard_pairing_gain": 0.05,
                    "val_cross_token_predicted_perplexity": 32.0,
                }]
            }
            (run / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
            old_argv = sys.argv
            try:
                sys.argv = ["finalize", "--suite-dir", str(suite)]
                finalize_main()
            finally:
                sys.argv = old_argv
            summary = json.loads((suite / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["rows"][0]["condition"], "M3")
            self.assertEqual(summary["rows"][0]["cross_token_pairing_gain"], 0.1)


if __name__ == "__main__":
    unittest.main()
