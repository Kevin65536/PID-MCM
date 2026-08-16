import csv
import json
import tempfile
import unittest
from pathlib import Path

from comparative_methods.performance_analysis.global_diagnostics import (
    build_descriptive_summary,
    build_task_method_rows,
    load_records,
    run_diagnostics,
)


class GlobalDiagnosticsTests(unittest.TestCase):
    def _write_sealed_aggregate(self, root: Path) -> Path:
        aggregate_dir = root / "aggregate"
        aggregate_dir.mkdir(parents=True)
        fields = [
            "method_id",
            "method_slug",
            "task",
            "track",
            "metric",
            "value",
            "fold_sample_sd",
            "B0",
            "minimum_admissible",
            "preferred_target",
            "numeric_acceptance",
            "terminal",
        ]
        rows = [
            {
                "method_id": "biot",
                "method_slug": "biot",
                "task": "dsr",
                "track": "frozen",
                "metric": "macro_f1",
                "value": "0.6",
                "fold_sample_sd": "0.1",
                "B0": "0.5",
                "minimum_admissible": "0.52",
                "preferred_target": "0.52",
                "numeric_acceptance": "TABLE_READY",
                "terminal": "TABLE_READY",
            },
            {
                "method_id": "biot",
                "method_slug": "biot",
                "task": "visual",
                "track": "frozen",
                "metric": "macro_f1",
                "value": "0.2",
                "fold_sample_sd": "0.02",
                "B0": "0.25",
                "minimum_admissible": "0.27",
                "preferred_target": "0.27",
                "numeric_acceptance": "REJECTED_VALUE",
                "terminal": "REJECTED_VALUE",
            },
            {
                "method_id": "cbramod",
                "method_slug": "cbramod",
                "task": "dsr",
                "track": "frozen",
                "metric": "macro_f1",
                "value": "",
                "fold_sample_sd": "",
                "B0": "",
                "minimum_admissible": "",
                "preferred_target": "",
                "numeric_acceptance": "UNSUPPORTED",
                "terminal": "UNSUPPORTED",
            },
            {
                "method_id": "cbramod",
                "method_slug": "cbramod",
                "task": "visual",
                "track": "frozen",
                "metric": "macro_f1",
                "value": "0.4",
                "fold_sample_sd": "0.03",
                "B0": "0.25",
                "minimum_admissible": "0.27",
                "preferred_target": "0.27",
                "numeric_acceptance": "TABLE_READY",
                "terminal": "TABLE_READY",
            },
        ]
        with (aggregate_dir / "cells.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        cells = []
        for row, folds in zip(rows, ([0.55, 0.65], [0.18, 0.22], [], [0.35, 0.45])):
            cells.append(
                {
                    "method_id": row["method_id"],
                    "method_slug": row["method_slug"],
                    "task": row["task"],
                    "track": row["track"],
                    "metric": row["metric"],
                    "fold_values": folds,
                }
            )
        fold_rows = []
        for cell in cells:
            for outer_fold, value in enumerate(cell["fold_values"]):
                fold_rows.append(
                    {
                        **{key: cell[key] for key in ("method_id", "method_slug", "task", "track", "metric")},
                        "outer_fold": outer_fold,
                        "seed_mean": value,
                    }
                )
        (aggregate_dir / "aggregate.json").write_text(
            json.dumps(
                {
                    "schema": "synthetic_sealed_aggregate_v1",
                    "campaign_id": "synthetic",
                    "cells": cells,
                    "fold_rows": fold_rows,
                }
            ),
            encoding="utf-8",
        )
        return aggregate_dir

    def test_loader_attaches_fold_values_and_delta_without_recomputing_test(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            aggregate_dir = self._write_sealed_aggregate(Path(tmpdir))
            records, payload = load_records(
                aggregate_dir / "cells.csv", aggregate_dir / "aggregate.json"
            )
            self.assertEqual(payload["campaign_id"], "synthetic")
            rows = build_task_method_rows(records)
            by_key = {(row["method_id"], row["task"]): row for row in rows}
            self.assertAlmostEqual(by_key[("biot", "dsr")]["value_minus_B0"], 0.1)
            self.assertEqual(by_key[("biot", "dsr")]["n_folds"], 2)
            self.assertTrue(by_key[("cbramod", "dsr")]["missing_cell"])
            self.assertIsNone(by_key[("cbramod", "dsr")]["value_minus_B0"])

    def test_summary_is_descriptive_and_complete_case_partition_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            aggregate_dir = self._write_sealed_aggregate(Path(tmpdir))
            records, _ = load_records(aggregate_dir / "cells.csv")
            fold_rows = []
            for record in records:
                for index, value in enumerate(record.fold_values):
                    fold_rows.append(
                        {
                            "metric": record.metric,
                            "outer_fold": index,
                            "fold_value": value,
                            "fold_value_minus_B0": record.delta_b0,
                            "missing_cell": False,
                        }
                    )
            summary = build_descriptive_summary(records, fold_rows)
            components = summary["variance_components"]["macro_f1"]["raw_value"]
            self.assertEqual(components["n_methods"], 1)
            self.assertEqual(components["n_tasks"], 2)
            self.assertAlmostEqual(
                components["method_pct"] + components["task_pct"] + components["residual_pct"],
                1.0,
            )
            self.assertIn("not an inferential", components["unit_note"])

    def test_run_writes_tables_figures_manifest_and_explicit_missing_cells(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            aggregate_dir = self._write_sealed_aggregate(root)
            output_dir = root / "out"
            manifest = run_diagnostics(aggregate_dir=aggregate_dir, output_dir=output_dir)
            expected = [
                "task_method_table.csv",
                "fold_level.csv",
                "task_method_value_macro_f1.csv",
                "task_method_delta_b0_macro_f1.csv",
                "descriptive_decomposition.csv",
                "descriptive_decomposition.json",
                "value_minus_b0_heatmap.png",
                "value_minus_b0_heatmap.pdf",
                "descriptive_decomposition.png",
                "descriptive_decomposition.pdf",
                "figure_manifest.json",
                "figure_alt_text.md",
                "analysis_manifest.json",
            ]
            for name in expected:
                self.assertTrue((output_dir / name).exists(), name)
            self.assertEqual(manifest["protected_data_policy"].split(";")[0], "frozen descriptive/post-hoc")
            table = (output_dir / "task_method_table.csv").read_text(encoding="utf-8")
            self.assertIn("cbramod,", table)
            self.assertIn("true", table)
            manifest_json = json.loads((output_dir / "figure_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest_json["n_missing_or_unsupported_cells"], 1)
            self.assertEqual(len(manifest_json["figures"]), 2)


if __name__ == "__main__":
    unittest.main()

