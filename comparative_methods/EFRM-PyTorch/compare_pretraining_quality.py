#!/usr/bin/env python3
"""Compare completed EFRM pretraining runs using public validation artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _label(run_id: str) -> str:
    marker = "__exclude_"
    if marker not in run_id:
        return run_id
    value = run_id.split(marker, 1)[1].split("__stage_", 1)[0]
    aliases = {
        "eeg_fnirs_single_trial": "exclude Single-Trial",
        "simultaneous_eeg_nirs": "exclude Simultaneous",
        "refed": "exclude REFED",
        "visual_cognitive_motivation": "exclude Visual",
    }
    return aliases.get(value, f"exclude {value}")


def _load_run(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "analysis/analysis_metrics.json"
    epochs_path = run_dir / "metrics/epochs.jsonl"
    if not metrics_path.is_file() or not epochs_path.is_file():
        raise FileNotFoundError(
            f"{run_dir} requires analysis/analysis_metrics.json and metrics/epochs.jsonl"
        )
    metrics = _read_json(metrics_path)
    epochs = _read_jsonl(epochs_path)
    if metrics["audit"]["run_state"] != "completed":
        raise ValueError(f"comparison accepts completed runs only: {run_dir}")
    if metrics["audit"]["protected_test_opened"]:
        raise PermissionError(f"protected test was opened for {run_dir}")
    return {
        "run_dir": str(run_dir.resolve()),
        "run_id": metrics["run_id"],
        "label": _label(metrics["run_id"]),
        "metrics": metrics,
        "epochs": epochs,
    }


def _configure_style() -> None:
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.13,
        1.07,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def _render_dashboard(runs: list[dict[str, Any]], output: Path) -> None:
    _configure_style()
    figure, axes = plt.subplots(2, 3, figsize=(14.2, 8.2), constrained_layout=True)

    axis = axes[0, 0]
    for index, run in enumerate(runs):
        epochs = run["epochs"]
        x = np.asarray([int(row["epoch"]) + 1 for row in epochs])
        values = np.asarray([row["validation"]["loss"] for row in epochs], dtype=float)
        axis.plot(x, values / values[0], color=COLORS[index], label=run["label"])
        best = int(run["metrics"]["audit"]["best_epoch"]) + 1
        axis.axvline(best, color=COLORS[index], ls=":", lw=1)
    axis.set(
        xlabel="completed epoch",
        ylabel="validation loss / epoch-1 loss",
        title="Total validation objective",
    )
    axis.axhline(1.0, color="#666666", ls="--", lw=0.8)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    _panel_label(axis, "A")

    axis = axes[0, 1]
    for index, run in enumerate(runs):
        epochs = run["epochs"]
        x = np.asarray([int(row["epoch"]) + 1 for row in epochs])
        eeg = np.asarray(
            [row["validation"]["eeg_reconstruction_loss"] for row in epochs],
            dtype=float,
        )
        fnirs = np.asarray(
            [row["validation"]["fnirs_reconstruction_loss"] for row in epochs],
            dtype=float,
        )
        axis.plot(
            x,
            eeg / eeg[0],
            color=COLORS[index],
            ls="-",
            label=f"{run['label']} · EEG",
        )
        axis.plot(
            x,
            fnirs / fnirs[0],
            color=COLORS[index],
            ls="--",
            label=f"{run['label']} · fNIRS",
        )
    axis.set(
        xlabel="completed epoch",
        ylabel="component loss / epoch-1 loss",
        title="Masked-reconstruction learning",
    )
    axis.axhline(1.0, color="#666666", ls=":", lw=0.8)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    _panel_label(axis, "B")

    axis = axes[0, 2]
    positions = np.arange(len(runs))
    width = 0.34
    eeg_ratios = [
        run["metrics"]["alignment"]["eeg_to_fnirs"]["mrr"]
        / run["metrics"]["alignment"]["chance"]["mrr"]
        for run in runs
    ]
    fnirs_ratios = [
        run["metrics"]["alignment"]["fnirs_to_eeg"]["mrr"]
        / run["metrics"]["alignment"]["chance"]["mrr"]
        for run in runs
    ]
    axis.bar(
        positions - width / 2,
        eeg_ratios,
        width,
        color="#0072B2",
        label="EEG→fNIRS",
    )
    axis.bar(
        positions + width / 2,
        fnirs_ratios,
        width,
        color="#E69F00",
        label="fNIRS→EEG",
    )
    axis.axhline(1.0, color="#666666", ls="--", lw=1, label="chance")
    axis.set(
        xticks=positions,
        xticklabels=[run["label"] for run in runs],
        ylabel="MRR / random-ranking MRR",
        title="Bidirectional exact-pair retrieval",
    )
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    _panel_label(axis, "C")

    axis = axes[1, 0]
    auc = [run["metrics"]["alignment"]["positive_vs_all_negative_auc"] for run in runs]
    cosine_gap = [
        run["metrics"]["alignment"]["positive_minus_negative_cosine"] for run in runs
    ]
    for index, (value, gap) in enumerate(zip(auc, cosine_gap)):
        axis.scatter(index, value, color=COLORS[index], s=70, zorder=3)
        axis.text(index, value + 0.002, f"Δcos={gap:.3f}", ha="center", fontsize=8)
    axis.axhline(0.5, color="#666666", ls="--", lw=1, label="chance")
    axis.set(
        xticks=np.arange(len(runs)),
        xticklabels=[run["label"] for run in runs],
        ylabel="positive-vs-negative AUC",
        title="Mean pair separation",
        ylim=(0.49, max(auc) + 0.012),
    )
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    _panel_label(axis, "D")

    axis = axes[1, 1]
    eeg_rank = [
        run["metrics"]["alignment"]["eeg_embedding_geometry"]["effective_rank"]
        for run in runs
    ]
    fnirs_rank = [
        run["metrics"]["alignment"]["fnirs_embedding_geometry"]["effective_rank"]
        for run in runs
    ]
    axis.bar(
        positions - width / 2,
        eeg_rank,
        width,
        color="#0072B2",
        label="EEG",
    )
    axis.bar(
        positions + width / 2,
        fnirs_rank,
        width,
        color="#009E73",
        label="fNIRS",
    )
    for index, run in enumerate(runs):
        eeg_energy = (
            100
            * run["metrics"]["alignment"]["eeg_embedding_geometry"][
                "first_axis_energy_fraction"
            ]
        )
        fnirs_energy = (
            100
            * run["metrics"]["alignment"]["fnirs_embedding_geometry"][
                "first_axis_energy_fraction"
            ]
        )
        axis.text(
            index - width / 2,
            eeg_rank[index] + 0.035,
            f"{eeg_energy:.1f}%",
            ha="center",
            fontsize=7,
        )
        axis.text(
            index + width / 2,
            fnirs_rank[index] + 0.035,
            f"{fnirs_energy:.1f}%",
            ha="center",
            fontsize=7,
        )
    axis.axhline(1.0, color="#666666", ls="--", lw=1, label="rank-1 limit")
    axis.set(
        xticks=positions,
        xticklabels=[run["label"] for run in runs],
        ylabel="centered effective rank",
        title="Embedding diversity (labels: first-axis energy)",
        ylim=(0, max(eeg_rank + fnirs_rank) + 0.3),
    )
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    _panel_label(axis, "E")

    axis = axes[1, 2]
    for index, run in enumerate(runs):
        training = run["metrics"]["training"]
        gap = training["validation_generalization_gap_last"]
        p95 = training["gradient_norm"]["p95"]
        clipped = 100 * training["gradient_norm"]["fraction_above_clip_threshold"]
        axis.scatter(
            gap,
            p95,
            s=35 + 20 * clipped,
            color=COLORS[index],
            alpha=0.85,
            label=f"{run['label']} ({clipped:.1f}% > 5)",
        )
    axis.axhline(5.0, color="#666666", ls="--", lw=1, label="clip threshold")
    axis.set(
        xlabel="final validation−train loss gap",
        ylabel="p95 pre-clip gradient norm",
        title="Generalization gap and optimization stability",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    _panel_label(axis, "F")

    figure.suptitle(
        "EFRM LODO Stage-A pretraining quality · complete public validation only",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for run in runs:
        metrics = run["metrics"]
        alignment = metrics["alignment"]
        training = metrics["training"]
        interpretation = metrics["interpretation"]
        if interpretation["full_validation_alignment_claim_supported"]:
            verdict = "detectable_bidirectional_alignment_but_low_rank"
        elif interpretation["saved_batch_alignment_above_chance"]:
            verdict = "partial_directionally_inconsistent_alignment_and_low_rank"
        else:
            verdict = "alignment_not_detected"
        rows.append({
            "run_id": run["run_id"],
            "label": run["label"],
            "completed_epochs": metrics["audit"]["completed_epoch_count"],
            "best_epoch_1_based": metrics["audit"]["best_epoch"] + 1,
            "best_validation_loss": metrics["audit"]["best_validation_loss"],
            "validation_relative_change": training["first_to_last_relative_change"][
                "validation"
            ],
            "validation_generalization_gap_last": training[
                "validation_generalization_gap_last"
            ],
            "gradient_norm": training["gradient_norm"],
            "alignment": {
                "pair_count": alignment["pair_count"],
                "dataset_ids": alignment["evidence_scope"]["dataset_ids"],
                "eeg_to_fnirs_mrr_over_chance": (
                    alignment["eeg_to_fnirs"]["mrr"] / alignment["chance"]["mrr"]
                ),
                "fnirs_to_eeg_mrr_over_chance": (
                    alignment["fnirs_to_eeg"]["mrr"] / alignment["chance"]["mrr"]
                ),
                "positive_vs_all_negative_auc": alignment[
                    "positive_vs_all_negative_auc"
                ],
                "positive_minus_negative_cosine": alignment[
                    "positive_minus_negative_cosine"
                ],
                "positive_minus_hardest_negative_mean": alignment[
                    "positive_minus_hardest_negative_mean"
                ],
                "identity_pair_permutation_p_one_sided": alignment[
                    "identity_pair_permutation_p_one_sided"
                ],
                "eeg_effective_rank": alignment["eeg_embedding_geometry"][
                    "effective_rank"
                ],
                "fnirs_effective_rank": alignment["fnirs_embedding_geometry"][
                    "effective_rank"
                ],
            },
            "verdict": verdict,
            "protected_test_opened": metrics["audit"]["protected_test_opened"],
        })
    return {
        "schema": "efrm_completed_pretraining_quality_comparison_v1",
        "generated_at": datetime.now().isoformat(),
        "scope": "completed Stage-A runs; public train/validation artifacts only",
        "runs": rows,
    }


def _write_report(summary: dict[str, Any], output: Path) -> None:
    lines = [
        "# EFRM completed Stage-A pretraining quality comparison",
        "",
        "Scope: completed formal Stage-A runs and complete public-validation exports only. "
        "Protected test data remained locked.",
        "",
        "| Excluded target | Epochs (best) | Val total Δ | EEG recon Δ | fNIRS recon Δ | "
        "AUC | MRR/chance E→F / F→E | Effective rank E / F | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["runs"]:
        change = row["validation_relative_change"]
        alignment = row["alignment"]
        lines.append(
            f"| {row['label']} | {row['completed_epochs']} ({row['best_epoch_1_based']}) | "
            f"{100 * change['loss']:.1f}% | "
            f"{100 * change['eeg_reconstruction_loss']:.1f}% | "
            f"{100 * change['fnirs_reconstruction_loss']:.1f}% | "
            f"{alignment['positive_vs_all_negative_auc']:.3f} | "
            f"{alignment['eeg_to_fnirs_mrr_over_chance']:.2f} / "
            f"{alignment['fnirs_to_eeg_mrr_over_chance']:.2f} | "
            f"{alignment['eeg_effective_rank']:.2f} / "
            f"{alignment['fnirs_effective_rank']:.2f} | `{row['verdict']}` |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Both runs learned their masked-reconstruction objectives strongly.",
        "- Mean positive-pair separation is statistically detectable, but AUC remains close "
        "to 0.5 and the hardest-negative margins remain strongly negative.",
        "- Effective ranks near 1 show severe low-rank/bipolar embedding compression. "
        "This is the main quality limitation even where bidirectional MRR exceeds chance.",
        "- These are Stage-A selection checkpoints. Final LODO pretraining is not complete "
        "until all four Stage-B refits have terminal completed markers.",
        "",
        "## Artifacts",
        "",
        "- `pretraining_quality_comparison.{svg,png}`",
        "- `comparison_summary.json`",
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    runs = [_load_run(Path(value).resolve()) for value in args.run_dir]
    if len(runs) < 2:
        raise ValueError("at least two completed runs are required")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _render_dashboard(runs, output_dir / "pretraining_quality_comparison")
    summary = _summary(runs)
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(summary, output_dir / "REPORT.md")
    print(json.dumps({
        "output_dir": str(output_dir),
        "run_count": len(runs),
        "protected_test_opened": any(
            row["protected_test_opened"] for row in summary["runs"]
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
