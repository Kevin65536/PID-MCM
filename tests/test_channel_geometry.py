import csv

import numpy as np
from scipy.io import savemat

from src.data.channel_geometry import records_from_mnt, records_from_refed_coordinates


def test_records_from_mnt_preserves_channel_coordinates_and_sd(tmp_path):
    path = tmp_path / "mnt.mat"
    savemat(
        path,
        {
            "mnt": {
                "clab": np.array(["AF7Fp1", "AF3Fp1"], dtype=object),
                "pos_3d": np.array([[-0.4, -0.2], [0.9, 0.8], [0.1, 0.2]], dtype=float),
                "sd": np.array([[2, 1], [3, 1]], dtype=np.uint8),
            }
        },
    )

    rows = records_from_mnt(
        path,
        dataset_id="eeg_fnirs_single_trial",
        subject="subject 01",
        record_id="global_nirs_mnt",
        modality="fnirs",
        channel_role="fnirs_channel_midpoint",
    )

    assert [row.channel_name for row in rows] == ["AF7Fp1", "AF3Fp1"]
    assert rows[0].x == -0.4
    assert rows[0].detector_index == 1
    assert rows[1].source_index == 3


def test_records_from_refed_coordinates_preserves_source_detector_indices(tmp_path):
    path = tmp_path / "fNIRS_coordinates.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Channel", "Source", "Detector", "X", "Y", "Z"])
        writer.writeheader()
        writer.writerow({"Channel": "1", "Source": "7", "Detector": "8", "X": "-1.5", "Y": "2.5", "Z": "3.5"})

    rows = records_from_refed_coordinates(path)

    assert len(rows) == 1
    assert rows[0].channel_name == "CH1"
    assert rows[0].source_index == 7
    assert rows[0].detector_index == 8
    assert rows[0].coordinate_system == "dataset_head_coordinates"
