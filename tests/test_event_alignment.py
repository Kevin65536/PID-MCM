from pathlib import Path
from zipfile import ZipFile

import numpy as np

from src.data.event_alignment import (
    EVENT_ALIGNMENT_SCHEMA,
    align_paired_marker_streams,
    detect_offset_blocks,
    drift_slope_ms_per_min,
    normalize_marker_targets,
    read_xlsx_rows,
)


def _marker(times, labels):
    class_names = ["A", "B"]
    y = np.zeros((2, len(times)), dtype=np.float32)
    for index, label in enumerate(labels):
        y[label, index] = 1.0
    return {"time": np.asarray(times, dtype=np.float64), "y": y, "className": class_names}


def test_normalize_marker_targets_accepts_event_major_or_class_major():
    event_major = np.asarray([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
    class_major = event_major.T

    np.testing.assert_array_equal(normalize_marker_targets(event_major, 3), class_major)
    np.testing.assert_array_equal(normalize_marker_targets(class_major, 3), class_major)


def test_align_paired_marker_streams_records_fixed_offset():
    eeg = _marker([1000, 2000, 3000], [0, 1, 0])
    fnirs = _marker([1500, 2500, 3500], [0, 1, 0])

    events, report = align_paired_marker_streams(
        dataset_id="synthetic",
        subject="s1",
        record_id="r1",
        eeg_marker=eeg,
        fnirs_marker=fnirs,
    )

    assert len(events) == 3
    assert report.schema == EVENT_ALIGNMENT_SCHEMA
    assert report.alignment_case == "stable_fixed_offset"
    assert report.offset_mean_ms == 500.0
    assert report.label_sequence_match is True
    assert events[0].metadata["offset_ms"] == 500.0


def test_alignment_reports_continuous_drift_slope():
    eeg_times = np.linspace(0, 10 * 60_000, 20)
    residual = np.linspace(0, 500, 20)
    fnirs_times = eeg_times + residual
    eeg = _marker(eeg_times, [0, 1] * 10)
    fnirs = _marker(fnirs_times, [0, 1] * 10)

    _, report = align_paired_marker_streams(
        dataset_id="synthetic",
        subject="s1",
        record_id="drift",
        eeg_marker=eeg,
        fnirs_marker=fnirs,
    )

    assert report.alignment_case == "continuous_drift"
    assert report.drift_slope_ms_per_min is not None
    assert report.drift_slope_ms_per_min > 40.0


def test_detect_offset_blocks_splits_large_jumps():
    blocks = detect_offset_blocks(np.asarray([100, 101, 102, 50_000, 50_001], dtype=np.float64))
    assert len(blocks) == 2
    assert blocks[0]["count"] == 3
    assert blocks[1]["count"] == 2


def test_drift_slope_returns_none_for_single_event():
    assert drift_slope_ms_per_min(np.asarray([1.0]), np.asarray([2.0])) is None


def test_read_xlsx_rows_uses_stdlib(tmp_path: Path):
    path = tmp_path / "labels.xlsx"
    with ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "")
        archive.writestr("_rels/.rels", "")
        archive.writestr(
            "xl/sharedStrings.xml",
            """<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
            <si><t>Epoch_ID</t></si><si><t>Type</t></si><si><t>RR</t></si>
            </sst>""",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            """<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
            <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>
            <row r="2"><c r="A2"><v>1</v></c><c r="B2" t="s"><v>2</v></c></row>
            </sheetData></worksheet>""",
        )

    assert read_xlsx_rows(str(path)) == [{"Epoch_ID": "1", "Type": "RR"}]
