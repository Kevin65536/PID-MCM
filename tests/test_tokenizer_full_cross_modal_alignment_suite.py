import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from experiments.scripts.finalize_tokenizer_full_cross_modal_alignment_suite import main as finalize_main
from experiments.scripts.launch_tokenizer_full_cross_modal_alignment_suite import (
    CONDITIONS,
    initialize_suite,
    prepare_z7,
)
from experiments.scripts.reselect_tokenizer_alignment_checkpoints import select_alignment_checkpoint
from experiments.scripts.train_source_observation_tokenizer import alignment_checkpoint_min_epoch


class FullCrossModalAlignmentSuiteTests(unittest.TestCase):
    def test_alignment_checkpoint_waits_for_all_schedules(self):
        config = {
            'training': {
                'alignment_warmup': {'enabled': True, 'start_epoch': 1, 'ramp_epochs': 5},
                'quantization_warmup': {'enabled': True, 'start_epoch': 1, 'ramp_epochs': 10},
                'staged_unfreeze': {
                    'enabled': True,
                    'freeze_encoder_epochs': 5,
                    'freeze_codebook_epochs': 5,
                },
            },
        }
        self.assertEqual(alignment_checkpoint_min_epoch(config), 10)

    def test_reselection_excludes_warmup_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / 'checkpoints').mkdir()
            config = {
                'loss': {'cross_modal_alignment': {'temporal_nce_weight': 0.2}},
                'training': {
                    'alignment_warmup': {'enabled': True, 'start_epoch': 1, 'ramp_epochs': 5},
                    'quantization_warmup': {'enabled': True, 'start_epoch': 1, 'ramp_epochs': 10},
                    'staged_unfreeze': {'enabled': True, 'freeze_encoder_epochs': 5, 'freeze_codebook_epochs': 5},
                },
            }
            (run_dir / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
            epochs = []
            for epoch, alignment, primary in ((5, 0.1, 1.0), (10, 0.5, 1.01), (20, 0.3, 1.02)):
                torch_payload = {'epoch': epoch, 'model_state_dict': {'value': epoch}}
                torch.save(torch_payload, run_dir / 'checkpoints' / f'checkpoint_epoch_{epoch}.pt')
                epochs.append({
                    'epoch': epoch,
                    'val_cross_modal_alignment_unscaled_loss': alignment,
                    'val_primary_loss': primary,
                })
            (run_dir / 'metrics.json').write_text(json.dumps({'epochs': epochs}), encoding='utf-8')

            result = select_alignment_checkpoint(run_dir)

            self.assertEqual(result['status'], 'selected')
            self.assertEqual(result['selected_epoch'], 20)
            self.assertTrue((run_dir / 'checkpoints/best_alignment_eligible.pt').exists())

    def test_launcher_generates_screening_and_confirmatory_matrix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            suite = Path(tmpdir) / "20260627_lag_aware_cross_transformer_v1"
            manifest = initialize_suite(
                suite,
                screening_epochs=20,
                confirm_epochs=40,
                seeds=(20260661, 20260662),
            )
            self.assertEqual(manifest["schema_version"], "tokenizer_full_cross_modal_alignment_suite_v1")
            self.assertEqual([item["key"] for item in manifest["conditions"]], [item.key for item in CONDITIONS])
            self.assertEqual(len(manifest["configs"]["smoke"]), 7)
            self.assertEqual(len(manifest["configs"]["screening"]), 7)
            self.assertEqual(len(manifest["configs"]["confirmatory"]), 10)
            self.assertTrue((suite / "queue_logs/smoke_gpu0.sh").exists())
            self.assertTrue((suite / "queue_logs/screen_gpu1.sh").exists())
            self.assertTrue((suite / "queue_logs/formal_gpu0.sh").exists())

            z3 = yaml.safe_load((
                suite / "tokenizer_interventions/configs/k128_dim128_z3_full_alignment_seed20260661.yaml"
            ).read_text(encoding="utf-8"))
            z4 = yaml.safe_load((
                suite / "tokenizer_interventions/configs/k128_dim128_z4_full_alignment_seed20260661.yaml"
            ).read_text(encoding="utf-8"))
            z5 = yaml.safe_load((
                suite / "tokenizer_interventions/configs/k128_dim128_z5_full_alignment_seed20260661.yaml"
            ).read_text(encoding="utf-8"))
            self.assertEqual(z3["model"]["cross_modal_fusion"]["mode"], "causal_cross_attention")
            self.assertEqual(z4["model"]["cross_modal_fusion"]["mode"], "bidirectional_cross_attention")
            self.assertEqual(z5["model"]["source_codebook"]["mode"], "shared_joint")
            self.assertEqual(z3["loss"]["cross_modal_alignment"]["positive_lag_weights"], [0.0, 0.1, 0.4, 0.4, 0.1, 0.0])
            self.assertTrue(z3["training"]["validation"]["forced_hard"])
            self.assertTrue(z3["training"]["alignment_gradient_control"]["enabled"])
            self.assertIn("best_hard_primary.pt", (suite / "queue_logs/audit_queue.sh").read_text(encoding="utf-8"))
            self.assertIn("best_alignment.pt", (suite / "queue_logs/audit_queue.sh").read_text(encoding="utf-8"))
            self.assertIn("final_model.pt", (suite / "queue_logs/audit_queue.sh").read_text(encoding="utf-8"))

            generated = prepare_z7(suite, "Z3")
            self.assertEqual(len(generated), 2)
            z7 = yaml.safe_load((
                suite / "tokenizer_interventions/configs/k128_dim128_z7_full_alignment_seed20260661.yaml"
            ).read_text(encoding="utf-8"))
            self.assertEqual(z7["loss"]["coupling"]["weight"], 0.01)
            self.assertEqual(z7["model"]["cross_modal_fusion"]["mode"], "causal_cross_attention")

    def test_finalizer_exposes_required_categories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            suite = Path(tmpdir)
            for key, primary in (("z0", 1.0), ("z3", 0.98)):
                run = suite / "tokenizer_interventions" / f"k128_dim128_{key}_full_alignment_seed20260661"
                run.mkdir(parents=True)
                payload = {
                    "epochs": [{
                        "epoch": 1,
                        "val_loss": primary,
                        "metrics": {
                            "val_primary_loss": primary,
                            "val_eeg_rec_loss": 0.7,
                            "val_fnirs_rec_loss": 0.2,
                            "val_eeg_source_perplexity": 80.0,
                            "val_fnirs_source_perplexity": 70.0,
                            "val_forced_hard_quantization": 1.0,
                            "val_cross_modal_fusion_physiologic_lag_mass": 0.4,
                            "alignment_gradient_ratio": 0.3,
                        },
                    }],
                }
                (run / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
            old = sys.argv
            try:
                sys.argv = ["finalize", "--suite-dir", str(suite)]
                finalize_main()
            finally:
                sys.argv = old
            summary = json.loads((suite / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["schema_version"], "tokenizer_full_cross_modal_alignment_summary_v1")
            for category in ("architecture", "information_retention", "fine_task", "coupling", "gradient", "token_health", "leakage"):
                self.assertIn(category, summary["categories"])
            self.assertIn("Z0", summary["conditions"])
            self.assertIn("Z3", summary["conditions"])


if __name__ == "__main__":
    unittest.main()
