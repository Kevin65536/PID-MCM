import unittest

import torch

from src.tokenizers.factorized_labram_vqnsp import SourceObservationLaBraMVQNSP


class SourceObservationExplicitTargetTests(unittest.TestCase):
    def _build_model(self) -> SourceObservationLaBraMVQNSP:
        return SourceObservationLaBraMVQNSP(
            eeg_seq_length=4000,
            eeg_patch_size=400,
            eeg_channels=6,
            eeg_encoder_embed_dim=16,
            eeg_encoder_depth=1,
            eeg_encoder_num_heads=1,
            eeg_decoder_embed_dim=16,
            eeg_decoder_depth=1,
            eeg_decoder_num_heads=1,
            fnirs_seq_length=200,
            fnirs_patch_size=20,
            fnirs_channels=1,
            fnirs_spatial_anchors=1,
            fnirs_optical_components=1,
            fnirs_component_labels=["highWL"],
            fnirs_encoder_embed_dim=16,
            fnirs_encoder_depth=1,
            fnirs_encoder_num_heads=1,
            fnirs_decoder_embed_dim=16,
            fnirs_decoder_depth=1,
            fnirs_decoder_num_heads=1,
            source_codebook_size=8,
            eeg_source_codebook_dim=8,
            fnirs_source_codebook_dim=8,
            eeg_observation_codebook_size=8,
            eeg_observation_codebook_dim=8,
            fnirs_observation_codebook_size=8,
            fnirs_observation_codebook_dim=8,
            kmeans_init=False,
            revive_dead_codes=False,
            source_target_weight=0.3,
            eeg_source_aux_weight=1.0,
            observation_target_weight=0.15,
            codebook_balance_weight=0.0,
            coupling_weight=0.0,
            orthogonality_weight=0.0,
            window_duration_s=20.0,
            dropout=0.0,
            drop_path=0.0,
        )

    def test_forward_accepts_explicit_source_observation_targets(self):
        torch.manual_seed(7)
        model = self._build_model()
        model.eval()
        eeg = torch.randn(2, 6, 4000)
        fnirs = torch.randn(2, 1, 200)
        targets = {
            "eeg_source": eeg * 0.25,
            "eeg_observation": eeg * 0.75,
            "fnirs_source": fnirs * 0.4,
            "fnirs_observation": fnirs * 0.6,
        }

        with torch.no_grad():
            outputs = model(eeg, fnirs, targets=targets)

        self.assertEqual(tuple(outputs["fnirs_reconstructed"].shape), (2, 1, 200))
        self.assertEqual(tuple(outputs["fnirs_source_reconstructed"].shape), (2, 1, 200))
        self.assertTrue(torch.allclose(outputs["fnirs_source_target"], targets["fnirs_source"]))
        self.assertTrue(torch.allclose(outputs["fnirs_observation_target"], targets["fnirs_observation"]))
        self.assertTrue(torch.allclose(outputs["eeg_source_target"], targets["eeg_source"]))
        self.assertEqual(float(outputs["croce_targets_used"].item()), 1.0)


if __name__ == "__main__":
    unittest.main()
