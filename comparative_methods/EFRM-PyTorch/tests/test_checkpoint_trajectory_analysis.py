from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efrm_pytorch.checkpoint_trajectory_analysis import (  # noqa: E402
    discover_runs,
    load_run_rows,
    run_analysis,
)


def _write_run(
    root: Path,
    run_id: str,
    *,
    include_analysis: bool = True,
    include_checkpoint: bool = False,
) -> Path:
    run = root / run_id
    (run / "metrics").mkdir(parents=True)
    (run / "analysis/checkpoints").mkdir(parents=True)
    manifest = {
        "schema": "efrm_sync_pretraining_run_v1",
        "status": "completed",
        "run_id": run_id,
        "lodo_stage": "selection" if "stage_a" in run_id else None,
        "excluded_target_dataset": "held_out",
        "protected_test_opened": False,
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run / "status.json").write_text(
        json.dumps({"status": "completed", "protected_test_opened": False}),
        encoding="utf-8",
    )
    epochs = []
    for epoch in range(2):
        epochs.append(
            {
                "epoch": epoch,
                "seconds": 1.0 + epoch,
                "learning_rate": 1e-4,
                "train": {
                    "loss": 5.0 - epoch,
                    "eeg_reconstruction_loss": 2.0 - epoch * 0.2,
                    "fnirs_reconstruction_loss": 1.5 - epoch * 0.1,
                    "clip_alignment_loss": 3.4,
                    "pair_count": 4,
                },
                "validation": {
                    "loss": 5.2 - epoch,
                    "eeg_reconstruction_loss": 2.1 - epoch * 0.2,
                    "fnirs_reconstruction_loss": 1.4 - epoch * 0.1,
                    "clip_alignment_loss": 3.42,
                    "pair_count": 4,
                },
                "cuda_peak_allocated_gib": 1.0,
                "cuda_peak_reserved_gib": 1.1,
            }
        )
    (run / "metrics/epochs.jsonl").write_text(
        "\n".join(json.dumps(row) for row in epochs) + "\n", encoding="utf-8"
    )
    if include_analysis:
        analysis = {
            "schema": "efrm_pretraining_analysis_v1",
            "audit": {
                "best_epoch": 1,
                "checkpoint": {
                    "checkpoint_epoch_inferred_from_log": None,
                },
            },
            "alignment": {
                "pair_count": 4,
                "positive_cosine_mean": 0.3,
                "negative_cosine_mean": 0.1,
                "positive_minus_negative_cosine": 0.2,
                "positive_vs_all_negative_auc": 0.75,
                "positive_minus_hardest_negative_mean": 0.1,
                "eeg_to_fnirs": {"mrr": 0.4, "top1": 0.25, "top5": 0.5},
                "fnirs_to_eeg": {"mrr": 0.35, "top1": 0.25, "top5": 0.5},
                "eeg_embedding_geometry": {
                    "effective_rank": 3.0,
                    "first_axis_energy_fraction": 0.4,
                    "off_diagonal_cosine_mean": 0.1,
                },
                "fnirs_embedding_geometry": {
                    "effective_rank": 2.0,
                    "first_axis_energy_fraction": 0.5,
                    "off_diagonal_cosine_mean": 0.2,
                },
            },
        }
        (run / "analysis/analysis_metrics.json").write_text(
            json.dumps(analysis), encoding="utf-8"
        )
    if include_checkpoint:
        checkpoint = {
            "alignment": {
                "pair_count": 4,
                "positive_vs_all_negative_auc": 0.7,
                "eeg_to_fnirs": {"mrr": 0.3},
                "fnirs_to_eeg": {"mrr": 0.2},
                "eeg_embedding_geometry": {"effective_rank": 1.5},
                "fnirs_embedding_geometry": {"effective_rank": 1.2},
            }
        }
        (run / "analysis/checkpoints/latest_full_validation_metrics.json").write_text(
            json.dumps(checkpoint), encoding="utf-8"
        )
    return run


def test_epoch_rows_keep_alignment_missing_and_analysis_export_unassigned_epoch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pretraining"
    run = _write_run(root, "synthetic_stage_a", include_analysis=True)
    rows, inventory = load_run_rows(run, root)
    epochs = [row for row in rows if row["record_kind"] == "epoch_metric"]
    exports = [row for row in rows if row["record_kind"] == "checkpoint_metric"]
    assert len(epochs) == 2
    assert all(row["validation_eeg_reconstruction_loss"] is not None for row in epochs)
    assert all(row["alignment_observed"] is False for row in epochs)
    assert all(row["positive_vs_all_negative_auc"] is None for row in epochs)
    assert len(exports) == 1
    assert exports[0]["checkpoint_label"] == "analysis_export"
    assert exports[0]["checkpoint_epoch_known"] is False
    assert exports[0]["positive_vs_all_negative_auc"] == pytest.approx(0.75)
    assert exports[0]["alignment_metric_basis"] == "row_weighted_duplicate_unaware_existing_export"
    assert inventory["field_availability"]["alignment"] is True


def test_checkpoint_summary_uses_explicit_epoch_only_and_marks_basis(tmp_path: Path) -> None:
    root = tmp_path / "pretraining"
    run = _write_run(root, "synthetic_stage_a", include_analysis=True, include_checkpoint=True)
    # Make the source analysis explicitly identify the latest checkpoint epoch.
    analysis_path = run / "analysis/analysis_metrics.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["audit"]["checkpoint"]["checkpoint_epoch_inferred_from_log"] = 1
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    rows, _inventory = load_run_rows(run, root)
    latest = [
        row
        for row in rows
        if row["record_kind"] == "checkpoint_metric" and row["checkpoint_label"] == "latest"
    ][0]
    assert latest["checkpoint_epoch"] == 1
    assert latest["checkpoint_epoch_known"] is True
    assert latest["checkpoint_epoch_source"] == "analysis.audit.checkpoint_epoch_inferred_from_log"
    assert latest["alignment_metric_basis"] == "row_weighted_duplicate_unaware_existing_export"


def test_discovery_default_excludes_smoke_and_include_all_is_explicit(tmp_path: Path) -> None:
    root = tmp_path / "pretraining"
    _write_run(root, "synthetic_stage_a", include_analysis=False)
    _write_run(root, "synthetic_smoke", include_analysis=False)
    assert [path.name for path in discover_runs(root)] == ["synthetic_stage_a"]
    assert [path.name for path in discover_runs(root, include_all=True)] == [
        "synthetic_smoke",
        "synthetic_stage_a",
    ]


def test_end_to_end_bundle_declares_no_interpolation_and_writes_figures(tmp_path: Path) -> None:
    root = tmp_path / "pretraining"
    _write_run(root, "synthetic_stage_a", include_analysis=True)
    output = tmp_path / "output"
    manifest = run_analysis(root, output)
    assert manifest["alignment_epoch_interpolation"] is False
    assert manifest["model_checkpoints_loaded"] is False
    assert manifest["similarity_matrices_loaded"] is False
    for filename in (
        "efrm_checkpoint_trajectory_tidy.csv",
        "summary.json",
        "manifest.json",
        "REPORT.md",
        "trajectory.png",
        "trajectory.pdf",
        "trajectory_all_runs.png",
        "trajectory_all_runs.pdf",
        "alignment_geometry.png",
        "alignment_geometry.pdf",
    ):
        assert (output / filename).is_file(), filename
    assert not list(output.glob("*alt*text*"))
