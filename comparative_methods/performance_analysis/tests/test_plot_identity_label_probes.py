from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comparative_methods.performance_analysis.plot_identity_label_probes import aggregate, run


def _rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for method, values in {"biot": {"task": ["0.5", "0.6"], "session": ["0.3", "0.4"], "subject_closed_set": ["0.8", "0.9"]}, "normwear": {"task": ["0.5", "0.5"], "session": ["0.3", "0.3"], "subject_closed_set": ["0.4", "0.5"]}}.items():
        for probe, entries in values.items():
            for index, value in enumerate(entries):
                rows.append({"task": "motor_imagery", "method": method, "probe": probe, "macro_f1": value, "n_test_subjects": "5", "split_index": str(index)})
    return rows


def test_aggregate_keeps_probe_estimands_and_local_chance() -> None:
    summary = aggregate(_rows())
    by_key = {(row["method"], row["probe"]): row for row in summary}
    assert by_key[("biot", "task")]["chance"] == 0.5
    assert by_key[("biot", "session")]["chance"] == 1 / 3
    assert by_key[("normwear", "subject_closed_set")]["chance"] == 1 / 29
    assert by_key[("biot", "subject_closed_set")]["n"] == 2


def test_run_writes_png_pdf_source_manifest_and_alt_text(tmp_path: Path) -> None:
    source = tmp_path / "metrics.csv"
    rows = _rows()
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "figures"
    summary = run(source, output)
    assert len(summary) == 6
    assert (output / "identity_probe_macro_f1.png").stat().st_size > 0
    assert (output / "identity_probe_macro_f1.pdf").stat().st_size > 0
    assert "upper-bound identity-retention" in (output / "alt_text.txt").read_text(encoding="utf-8")
    assert (output / "figure_manifest.json").is_file()

