#!/usr/bin/env python3
"""Audit repeated public folds and report subject-level baseline performance.

The public v1 manifests intentionally provide several training repetitions, and
their validation inventories overlap.  Therefore the report never concatenates
all validation windows into one pseudo-OOF metric.  It first computes one
macro-F1 per subject in each outer fold, averages those fold-level values for
each subject, and only then reports the mean/SD/subject bootstrap interval.
Repeated sample identities are audited explicitly with SHA-256 identifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = (
    REPO_ROOT
    / "comparative_methods"
    / "runs"
    / "performance_analysis"
    / "20260816_p0"
    / "classical_baselines"
)
TASKS = ("motor_imagery", "nback", "visual")
TASK_LABELS = {"motor_imagery": "Motor imagery", "nback": "N-back", "visual": "Visual"}
BOOTSTRAP_SEED = 20260816
BOOTSTRAP_REPLICATES = 10_000


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _bootstrap_mean(values: np.ndarray, *, seed: int, replicates: int) -> tuple[float, float]:
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty subject metric vector")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(int(replicates), values.size))
    means = values[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _identity_hash(task: str, dataset_index: int, subject: str, join_key: str, sample_id: str | None) -> tuple[str, str]:
    """Return a canonical sample ID and its hash.

    New prediction artifacts carry the adapter-compatible event/offset sample
    ID.  Older outer0 artifacts are upgraded deterministically from the frozen
    dataset index + subject + record key; the source is recorded so this
    fallback cannot be mistaken for an unverified opaque index.
    """
    if sample_id:
        canonical = str(sample_id)
        source = "prediction_artifact_sample_id"
    else:
        canonical = f"{task}|dataset_index={int(dataset_index)}|subject={subject}|join_key={join_key}"
        source = "dataset_index_subject_join_key_fallback"
    return canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_fold_predictions(task_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load all fold predictions and audit repeated sample identities."""
    summary_path = task_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("protected_test_opened") is not False:
        raise RuntimeError(f"baseline summary crossed protected boundary: {summary_path}")
    rows: list[dict[str, Any]] = []
    identity_sources: set[str] = set()
    for fold in sorted(summary.get("folds", ()), key=lambda row: int(row["outer_fold"])):
        fold_id = int(fold["outer_fold"])
        npz_path = task_root / f"outer{fold_id}_predictions.npz"
        with np.load(npz_path, allow_pickle=False) as payload:
            required = {"dataset_index", "y_true", "y_pred", "subject", "join_key"}
            missing = required.difference(payload.files)
            if missing:
                raise RuntimeError(f"prediction artifact missing {sorted(missing)}: {npz_path}")
            indices = np.asarray(payload["dataset_index"], dtype=np.int64)
            y_true = np.asarray(payload["y_true"], dtype=np.int64)
            y_pred = np.asarray(payload["y_pred"], dtype=np.int64)
            subjects = np.asarray(payload["subject"], dtype=str)
            join_keys = np.asarray(payload["join_key"], dtype=str)
            sample_ids = np.asarray(payload["sample_id"], dtype=str) if "sample_id" in payload.files else None
        if not (indices.size == y_true.size == y_pred.size == subjects.size == join_keys.size):
            raise RuntimeError(f"prediction artifact has inconsistent lengths: {npz_path}")
        if sample_ids is not None and sample_ids.size != indices.size:
            raise RuntimeError(f"sample_id length differs from predictions: {npz_path}")
        for position, (index, truth, prediction, subject, join_key) in enumerate(
            zip(indices, y_true, y_pred, subjects, join_keys, strict=True)
        ):
            canonical, identity_hash = _identity_hash(
                str(summary["task"]), int(index), str(subject), str(join_key),
                None if sample_ids is None else str(sample_ids[position]),
            )
            identity_sources.add(
                "prediction_artifact_sample_id" if sample_ids is not None else "dataset_index_subject_join_key_fallback"
            )
            rows.append(
                {
                    "outer_fold": fold_id,
                    "dataset_index": int(index),
                    "sample_id": canonical,
                    "sample_id_sha256": identity_hash,
                    "y_true": int(truth),
                    "y_pred": int(prediction),
                    "subject": str(subject),
                    "join_key": str(join_key),
                }
            )
    if not rows:
        raise RuntimeError(f"no public validation predictions found: {task_root}")
    by_hash: dict[str, list[int]] = {}
    for row_index, row in enumerate(rows):
        by_hash.setdefault(str(row["sample_id_sha256"]), []).append(row_index)
    repeated = {key: values for key, values in by_hash.items() if len(values) > 1}
    identity_conflicts: list[dict[str, Any]] = []
    for identity_hash, row_indices in repeated.items():
        identities = {
            (str(rows[index]["sample_id"]), str(rows[index]["subject"]), int(rows[index]["y_true"]))
            for index in row_indices
        }
        if len(identities) > 1:
            identity_conflicts.append(
                {"sample_id_sha256": identity_hash, "rows": row_indices}
            )
    if identity_conflicts:
        raise RuntimeError(f"sample identity metadata conflicts across folds: {identity_conflicts[:3]}")
    provenance = {
        "raw_validation_prediction_count": len(rows),
        "unique_sample_id_count": len(by_hash),
        "repeated_sample_id_count": len(repeated),
        "repeated_prediction_row_count": int(sum(len(values) - 1 for values in repeated.values())),
        "sample_identity_sources": sorted(identity_sources),
        "sample_id_hash": "sha256(canonical_sample_id_utf8)",
        "sample_identity_conflicts": 0,
        "repetition_policy": "primary endpoint averages fold-level subject macro-F1 before subject aggregation; no repeated row is treated as an independent subject",
    }
    return provenance, rows


def summarize_task(
    task_root: Path,
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    provenance, rows = load_fold_predictions(task_root)
    fold_rows: list[dict[str, Any]] = []
    subject_fold_values: dict[str, list[float]] = {}
    subject_sample_counts: dict[str, int] = {}
    task = str(json.loads((task_root / "summary.json").read_text(encoding="utf-8"))["task"])
    for fold_id in sorted({int(row["outer_fold"]) for row in rows}):
        fold = [row for row in rows if int(row["outer_fold"]) == fold_id]
        y_true = np.asarray([row["y_true"] for row in fold], dtype=np.int64)
        y_pred = np.asarray([row["y_pred"] for row in fold], dtype=np.int64)
        fold_metrics = _metrics(y_true, y_pred)
        per_subject: dict[str, dict[str, float | int]] = {}
        subjects = sorted({str(row["subject"]) for row in fold})
        for subject in subjects:
            mask = np.asarray([str(row["subject"]) == subject for row in fold], dtype=bool)
            value = _metrics(y_true[mask], y_pred[mask])
            per_subject[subject] = {"sample_count": int(mask.sum()), **value}
            subject_fold_values.setdefault(subject, []).append(float(value["macro_f1"]))
            subject_sample_counts[subject] = subject_sample_counts.get(subject, 0) + int(mask.sum())
        fold_rows.append({"outer_fold": fold_id, "sample_count": len(fold), "subject_count": len(subjects), "metrics": fold_metrics, "subject_metrics": per_subject})
    subject_rows: list[dict[str, Any]] = []
    for subject in sorted(subject_fold_values):
        values = np.asarray(subject_fold_values[subject], dtype=np.float64)
        subject_rows.append(
            {
                "subject": subject,
                "fold_count": int(values.size),
                "sample_count_across_fold_predictions": int(subject_sample_counts[subject]),
                "fold_macro_f1_values": values.tolist(),
                "macro_f1": float(values.mean()),
            }
        )
    subject_values = np.asarray([row["macro_f1"] for row in subject_rows], dtype=np.float64)
    ci_low, ci_high = _bootstrap_mean(subject_values, seed=int(bootstrap_seed), replicates=int(bootstrap_replicates))
    fold_metric_means = {
        name: {
            "mean_across_outer_folds": float(np.mean([row["metrics"][name] for row in fold_rows])),
            "sd_across_outer_folds": float(np.std([row["metrics"][name] for row in fold_rows], ddof=1)) if len(fold_rows) > 1 else 0.0,
            "fold_values": [float(row["metrics"][name]) for row in fold_rows],
        }
        for name in ("accuracy", "balanced_accuracy", "macro_f1")
    }
    return {
        "task": task,
        **provenance,
        "outer_fold_count": len(fold_rows),
        "subject_count": int(subject_values.size),
        "primary_endpoint": "subject_level_macro_f1_after_fold_average",
        "subject_level_macro_f1": {
            "mean": float(subject_values.mean()),
            "sd": float(subject_values.std(ddof=1)) if subject_values.size > 1 else 0.0,
            "bootstrap_95ci": [ci_low, ci_high],
            "bootstrap_seed": int(bootstrap_seed),
            "bootstrap_replicates": int(bootstrap_replicates),
            "values": subject_values.tolist(),
        },
        "fold_level_metrics_secondary": fold_metric_means,
        "folds": fold_rows,
        "subject_metrics": subject_rows,
    }


def _plot(task_summaries: Sequence[dict[str, Any]], destination_png: Path, destination_pdf: Path) -> str:
    fig, axes = plt.subplots(
        1, len(task_summaries), figsize=(14.5, 5.0), squeeze=False, layout="constrained"
    )
    for ax, item in zip(axes[0], task_summaries, strict=True):
        rows = item["subject_metrics"]
        labels = [str(row["subject"]) for row in rows]
        values = np.asarray([float(row["macro_f1"]) for row in rows], dtype=float)
        x = np.arange(values.size)
        ax.scatter(x, values, s=34, color="#245a88", alpha=0.9, zorder=3)
        mean = float(item["subject_level_macro_f1"]["mean"])
        low, high = item["subject_level_macro_f1"]["bootstrap_95ci"]
        ax.axhline(mean, color="#c33b3b", linewidth=1.4, label=f"mean={mean:.3f}")
        ax.axhspan(float(low), float(high), color="#d86c6c", alpha=0.16, label="95% bootstrap CI")
        ax.set_title(TASK_LABELS.get(item["task"], item["task"]))
        ax.set_ylabel("Subject-level macro-F1")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x, labels, rotation=65, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right", fontsize=7, frameon=False)
    fig.suptitle("Classical EEG band-power baseline: public validation subject averages")
    fig.text(0.5, 0.005, "Each point is one subject's mean across public outer folds in which that subject was validation.", ha="center", fontsize=9)
    destination_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination_png, dpi=220, facecolor="white", transparent=False)
    fig.savefig(destination_pdf, facecolor="white", transparent=False)
    plt.close(fig)
    return (
        "Three panels (motor imagery, n-back, visual). Each dot is one subject's "
        "macro-F1 averaged across the public outer folds in which that subject "
        "was validation. The red line is the unweighted subject mean and the "
        "translucent band is its percentile bootstrap 95% CI (seed 20260816, "
        "10,000 draws). Repeated sample identities across folds were audited by "
        "SHA-256 and were not concatenated as independent OOF observations."
    )


def build_report(root: Path = DEFAULT_ROOT, *, tasks: Sequence[str] = TASKS) -> dict[str, Any]:
    root = root.expanduser().resolve()
    summaries = [summarize_task(root / task / "lda") for task in tasks]
    figure_png = root / "subject_macro_f1_public_fold_average.png"
    figure_pdf = root / "subject_macro_f1_public_fold_average.pdf"
    alt_text = _plot(summaries, figure_png, figure_pdf)
    result = {
        "schema": "classical_eeg_bandpower_public_report_v2",
        "status": "completed",
        "analysis_label": "exploratory_public_development_only",
        "protected_test_opened": False,
        "protocol": "public_v1_strict_cross_subject_outer0_to_outer4",
        "primary_endpoint": "subject_level_macro_f1_after_fold_average",
        "subject_unit": "one subject contributes one value after averaging its fold-level validation macro-F1 values",
        "no_pooled_concatenation": True,
        "sample_identity": "sha256(canonical_sample_id_utf8)",
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "interval": "percentile_95"},
        "tasks": summaries,
        "figures": {"png": str(figure_png), "pdf": str(figure_pdf), "alt_text": alt_text},
    }
    report_lines = [
        "# Classical EEG band-power baseline: public validation report",
        "",
        "> Exploratory public-development evidence only. This report does not open or replace the protected comparison table.",
        "",
        "## Protocol and unit of analysis",
        "",
        "The baseline uses fixed 16-channel task panels, Welch log-bandpower and shrinkage LDA. The five public v1 strict-cross-subject manifests overlap in validation inventories. We first audit each prediction's canonical sample ID and SHA-256 hash. We do not concatenate repeated windows into an OOF score. Instead, each outer fold is treated as a training repetition: compute one macro-F1 per validation subject within each fold, average those fold-level values for each subject, and use subjects as the primary statistical unit. SD and the percentile bootstrap 95% CI are over subject values (seed `20260816`, 10,000 draws).",
        "",
        "## Coverage limitation",
        "",
        "These are public-v1 development folds only. The subject counts in the table are the union of subjects appearing in public validation (17 for motor imagery, 16 for n-back, and 8 for visual); reserved/protected subjects and labels were not opened. The results therefore diagnose public cross-subject difficulty and subject heterogeneity, but are not a replacement for the protected campaign estimate.",
        "",
        "## Results",
        "",
        "| task | outer folds | subjects | validation rows | unique sample IDs | repeated rows | subject macro-F1 mean ± SD | bootstrap 95% CI | mean fold macro-F1 (secondary) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        primary = item["subject_level_macro_f1"]
        secondary = item["fold_level_metrics_secondary"]["macro_f1"]["mean_across_outer_folds"]
        report_lines.append(
            f"| {item['task']} | {item['outer_fold_count']} | {item['subject_count']} | {item['raw_validation_prediction_count']} | {item['unique_sample_id_count']} | {item['repeated_prediction_row_count']} | {primary['mean']:.4f} ± {primary['sd']:.4f} | [{primary['bootstrap_95ci'][0]:.4f}, {primary['bootstrap_95ci'][1]:.4f}] | {secondary:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "## Subject-level values",
            "",
            "Every point in the figure is one subject. Values are averaged over that subject's public validation folds; repeated samples are retained only as training-repetition evidence and are never treated as additional subjects.",
            "",
            "![Subject-level macro-F1 points](subject_macro_f1_public_fold_average.png)",
            "",
            f"Alt text: {alt_text}",
            "",
            "Fold metrics and subject-level rows remain under each task's `lda/` directory.",
        ]
    )
    (root / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--task", action="append", choices=TASKS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks = tuple(args.task) if args.task else TASKS
    manifest = build_report(args.root, tasks=tasks)
    for item in manifest["tasks"]:
        primary = item["subject_level_macro_f1"]
        print(
            f"[{item['task']}] folds={item['outer_fold_count']} subjects={item['subject_count']} "
            f"unique_sample_ids={item['unique_sample_id_count']} repeated_rows={item['repeated_prediction_row_count']} "
            f"subject_macro_f1={primary['mean']:.4f}±{primary['sd']:.4f} "
            f"CI=[{primary['bootstrap_95ci'][0]:.4f},{primary['bootstrap_95ci'][1]:.4f}]",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
