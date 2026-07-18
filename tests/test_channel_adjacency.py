import unittest
from pathlib import Path

import numpy as np
import torch

from src.data.channel_adjacency import (
    build_channel_adjacency,
    build_eeg_adjacency_from_geometry,
    build_fnirs_adjacency_from_shared_optodes,
    compute_per_channel_rms_envelope,
    compute_spatial_fnirs_driver,
)
from src.data.channel_geometry import (
    records_from_template_montage_csv,
    records_from_visual_fnirs_graphical_projection,
)
from src.data.eeg_fnirs_dataset import MultiModalEEGfNIRSDataset


class ChannelAdjacencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = MultiModalEEGfNIRSDataset(
            data_root='data/EEG+NIRS Single-Trial',
            subject_ids=[1],
            task='motor_imagery',
            window_duration_s=20.0,
            normalize=True,
            normalization_mode='session',
            exclude_eog=True,
            hbo_only=True,
            hbr_only=False,
        )

    def test_build_channel_adjacency_is_row_normalized_for_single_trial(self):
        info = build_channel_adjacency(
            'eeg_fnirs_single_trial',
            'data/EEG+NIRS Single-Trial',
            self.dataset.get_eeg_channel_names(),
            self.dataset.get_fnirs_channel_names(),
            reference_subject_id=1,
            use_artifact_data=True,
        )

        self.assertEqual(info.adjacency_matrix.shape, (36, 30))
        self.assertTrue(np.allclose(info.adjacency_matrix.sum(axis=1), 1.0, atol=1e-5))
        self.assertEqual(info.anchor_matches[0]['base_channel'], 'AF7Fp1')
        self.assertIn('F7', info.anchor_matches[0]['source_anchor']['direct_labels'])
        self.assertEqual(info.warnings, [])

    def test_compute_per_channel_rms_envelope_reduces_to_abs_without_smoothing(self):
        eeg = torch.tensor(
            [[[1.0, -1.0, 1.0, -1.0], [0.0, 0.0, 2.0, -2.0]]],
            dtype=torch.float32,
        )

        target = compute_per_channel_rms_envelope(eeg, smoothing_samples=1, eps=0.0)
        expected = torch.tensor(
            [[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]]],
            dtype=torch.float32,
        )

        self.assertTrue(torch.allclose(target, expected, atol=1e-6))

    def test_compute_spatial_fnirs_driver_uses_weighted_eeg_power(self):
        eeg = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]]],
            dtype=torch.float32,
        )
        adjacency = torch.tensor(
            [[1.0, 0.0], [0.25, 0.75]],
            dtype=torch.float32,
        )

        driver = compute_spatial_fnirs_driver(eeg, adjacency, target_length=2)
        expected = torch.tensor(
            [[[1.0, 4.0], [7.0, 13.0]]],
            dtype=torch.float32,
        )

        self.assertTrue(torch.allclose(driver, expected, atol=1e-6))

    def test_refed_template_geometry_builds_connected_symmetric_eeg_topology(self):
        root = Path(__file__).resolve().parents[1]
        records = records_from_template_montage_csv(
            root / "src/data/assets/refed_standard_1005_montage_v1.csv",
            dataset_id="refed",
            subject="all",
            record_id="global_eeg_standard_1005_v1",
            template_name="fieldtrip_standard_1005_refed_1010_subset",
            coordinate_system="fieldtrip_standard_1005_mni_template",
            coordinate_units="mm",
        )
        rows = []
        for record in records:
            row = record.to_dict()
            row["coordinate_status"] = record.metadata["coordinate_status"]
            rows.append(row)
        names = [row["channel_name"] for row in rows]

        info = build_eeg_adjacency_from_geometry("refed", rows, names)
        adjacency = info.adjacency_matrix

        self.assertEqual(adjacency.shape, (64, 64))
        self.assertTrue(np.array_equal(adjacency, adjacency.T))
        self.assertTrue(np.all(np.diag(adjacency) == 0))
        self.assertEqual(len(info.edges), 168)
        self.assertTrue(np.all(adjacency.sum(axis=1) >= 3))
        visited = {0}
        frontier = [0]
        while frontier:
            current = frontier.pop()
            for neighbor in np.flatnonzero(adjacency[current]):
                neighbor = int(neighbor)
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append(neighbor)
        self.assertEqual(len(visited), 64)
        cb1 = names.index("CB1")
        cb2 = names.index("CB2")
        self.assertGreater(adjacency[cb1, names.index("PO7")], 0)
        self.assertGreater(adjacency[cb1, names.index("O1")], 0)
        self.assertGreater(adjacency[cb2, names.index("PO8")], 0)
        self.assertGreater(adjacency[cb2, names.index("O2")], 0)

    def test_visual_4x4_geometry_builds_connected_shared_optode_topology(self):
        root = Path(__file__).resolve().parents[1]
        data_root = (
            root
            / "data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults"
        )
        records = records_from_visual_fnirs_graphical_projection(
            data_root / "fNIRS_to_EEG_channel_reference.xlsx",
            data_root / "Location.ced",
            root / "src/data/assets/visual_fnirs_4x4_topology_v1.csv",
            graphical_model_path=data_root / "Graphical_recording_head_model.pdf",
        )
        channel_names = [f"CH{index}" for index in range(1, 25)]
        for probe in ("Probe1", "Probe2"):
            rows = [record.to_dict() for record in records if record.record_id == probe]
            info = build_fnirs_adjacency_from_shared_optodes(
                "visual_cognitive_motivation", probe, rows, channel_names
            )
            adjacency = info.adjacency_matrix
            self.assertEqual(adjacency.shape, (24, 24))
            self.assertEqual(len(info.edges), 52)
            self.assertTrue(np.array_equal(adjacency, adjacency.T))
            self.assertTrue(np.all(np.diag(adjacency) == 0))
            self.assertTrue(np.all(adjacency.sum(axis=1) >= 3))
            self.assertEqual(
                np.linalg.matrix_rank(np.diag(adjacency.sum(axis=1)) - adjacency), 23
            )


if __name__ == '__main__':
    unittest.main()
