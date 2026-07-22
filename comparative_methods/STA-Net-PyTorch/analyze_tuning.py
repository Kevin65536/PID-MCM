#!/usr/bin/env python3
"""Audit and visualize a completed STA-Net Optuna tuning run.

The analyzer is read-only with respect to the tuning artifacts.  It reconstructs
trial states from Optuna, rung trajectories from intermediate values, and
checkpoint-selection trajectories from the validation JSONL files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import optuna
from scipy.stats import spearmanr


TASK_ORDER = (
    "motor_imagery",
    "mental_arithmetic",
    "wg",
    "nback",
    "dsr",
    "visual",
    "refed_regression",
)
TPE_STARTUP_TRIALS = 6
TASK_LABELS = {
    "motor_imagery": "Motor imagery",
    "mental_arithmetic": "Mental arithmetic",
    "wg": "Word generation",
    "nback": "N-back",
    "dsr": "DSR",
    "visual": "Visual",
    "refed_regression": "REFED regression",
}
STATE_STYLE = {
    "COMPLETE": ("#0072B2", "o", "-"),
    "PRUNED": ("#E69F00", "s", "--"),
    "FAIL": ("#D55E00", "X", ":"),
    "RUNNING": ("#999999", "^", "-."),
    "WAITING": ("#999999", "v", "-."),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def metric_contract(task: str) -> tuple[str, str]:
    return (
        ("masked_rmse_scaled", "min")
        if task == "refed_regression"
        else ("macro_f1", "max")
    )


def natural_objective(task: str, value: float | None) -> float | None:
    if value is None:
        return None
    return -float(value) if task == "refed_regression" else float(value)


def better(mode: str, left: float, right: float) -> bool:
    return left < right if mode == "min" else left > right


def failure_reason(path: Path) -> str | None:
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    exception = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception):\s+.+")
    for line in reversed(lines):
        stripped = line.strip()
        if exception.match(stripped):
            return stripped
    return "process exited non-zero; no exception line was found"


def finite_spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    value = float(spearmanr(xs, ys).statistic)
    return value if math.isfinite(value) else None


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
    })


def collect_run(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    launch = read_json(root / "launch_manifest.json")
    rung_epochs = [int(value) for value in launch["rung_epochs"]]
    study_id = str(launch["study_id"])
    storage = f"sqlite:///{root / 'optuna.sqlite3'}"
    summaries = {summary.study_name: summary for summary in optuna.study.get_all_study_summaries(storage=storage)}
    tasks = [task for task in TASK_ORDER if f"{study_id}__{task}__development_cross_subject" in summaries]

    trial_rows: list[dict[str, Any]] = []
    rung_rows: list[dict[str, Any]] = []
    curves: dict[str, list[dict[str, Any]]] = {task: [] for task in tasks}
    task_summary: dict[str, Any] = {}
    all_trainer_hashes: Counter[str] = Counter()

    for task in tasks:
        name = f"{study_id}__{task}__development_cross_subject"
        study = optuna.load_study(study_name=name, storage=storage)
        metric_name, mode = metric_contract(task)
        split = read_json(root / "splits" / task / "development_cross_subject.json")
        task_trials: list[dict[str, Any]] = []

        for trial in study.trials:
            state = trial.state.name
            trial_dir = root / "trials" / task / f"trial_{trial.number:05d}"
            validation = read_jsonl(trial_dir / "run" / "metrics" / "validation_epochs.jsonl")
            metric_rows = [row for row in validation if metric_name in row]
            checkpoint_best_metric: float | None = None
            checkpoint_best_epoch: int | None = None
            checkpoint_metrics: dict[str, float] = {}
            endpoint_metric: float | None = None
            if metric_rows:
                selected = min(metric_rows, key=lambda row: float(row[metric_name])) if mode == "min" else max(
                    metric_rows, key=lambda row: float(row[metric_name])
                )
                checkpoint_best_metric = float(selected[metric_name])
                checkpoint_best_epoch = int(selected["epoch"])
                checkpoint_metrics = {
                    key: float(selected[key])
                    for key in (
                        "accuracy", "balanced_accuracy", "macro_f1", "cohen_kappa",
                        "masked_mae_scaled", "masked_rmse_scaled", "sample_count",
                    )
                    if key in selected
                }
                endpoint = [row for row in metric_rows if int(row["epoch"]) == rung_epochs[-1]]
                if endpoint:
                    endpoint_metric = float(endpoint[-1][metric_name])
                curves[task].append({
                    "trial_number": int(trial.number),
                    "state": state,
                    "epochs": [int(row["epoch"]) for row in metric_rows],
                    "metrics": [float(row[metric_name]) for row in metric_rows],
                })

            trainer_hash: str | None = None
            manifest_path = trial_dir / "run" / "manifest.json"
            if manifest_path.exists():
                trainer_hash = read_json(manifest_path).get("implementation_sha256", {}).get("trainer")
                if trainer_hash:
                    all_trainer_hashes[trainer_hash] += 1

            duration = None
            if trial.datetime_start and trial.datetime_complete:
                duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
            max_step = max(trial.intermediate_values, default=0)
            max_budget = rung_epochs[max_step - 1] if max_step else 0
            row: dict[str, Any] = {
                "task": task,
                "trial_number": int(trial.number),
                "state": state,
                "objective_raw": None if trial.value is None else float(trial.value),
                "objective_metric": natural_objective(task, trial.value),
                "duration_seconds": duration,
                "max_rung_step": max_step,
                "max_epoch_budget": max_budget,
                "checkpoint_best_metric": checkpoint_best_metric,
                "checkpoint_best_epoch": checkpoint_best_epoch,
                "endpoint_metric": endpoint_metric,
                "failure_reason": failure_reason(trial_dir / "process.log") if state == "FAIL" else None,
                "trainer_sha256": trainer_hash,
            }
            row.update({f"checkpoint_best_{key}": value for key, value in checkpoint_metrics.items()})
            row.update({f"param_{key}": value for key, value in trial.params.items()})
            trial_rows.append(row)
            task_trials.append(row)
            for step, raw_value in sorted(trial.intermediate_values.items()):
                epoch = rung_epochs[int(step) - 1]
                metric_value = natural_objective(task, float(raw_value))
                rung_rows.append({
                    "task": task,
                    "trial_number": int(trial.number),
                    "state": state,
                    "rung_step": int(step),
                    "epoch_budget": epoch,
                    "objective_raw": float(raw_value),
                    "objective_metric": metric_value,
                })

        completed = [row for row in task_trials if row["state"] == "COMPLETE"]
        valid = [row for row in task_trials if row["state"] in {"COMPLETE", "PRUNED"}]
        endpoint_trial = int(study.best_trial.number)
        endpoint_value = natural_objective(task, study.best_value)
        eligible = [row for row in completed if row["checkpoint_best_metric"] is not None]
        checkpoint_winner = None
        if eligible:
            checkpoint_winner = eligible[0]
            for row in eligible[1:]:
                if better(mode, float(row["checkpoint_best_metric"]), float(checkpoint_winner["checkpoint_best_metric"])):
                    checkpoint_winner = row

        correlations: dict[str, float | None] = {}
        complete_trials = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
        for step, epoch in enumerate(rung_epochs[:-1], start=1):
            xs = [float(trial.intermediate_values[step]) for trial in complete_trials if step in trial.intermediate_values]
            ys = [float(trial.value) for trial in complete_trials if step in trial.intermediate_values and trial.value is not None]
            correlations[str(epoch)] = finite_spearman(xs, ys)

        degradation = []
        for row in completed:
            if row["checkpoint_best_metric"] is None or row["endpoint_metric"] is None:
                continue
            if mode == "max":
                degradation.append(float(row["checkpoint_best_metric"]) - float(row["endpoint_metric"]))
            else:
                degradation.append(float(row["endpoint_metric"]) - float(row["checkpoint_best_metric"]))
        states = Counter(row["state"] for row in task_trials)
        consumed = sum(int(row["max_epoch_budget"]) for row in task_trials)
        planned = int(launch["n_trials_per_task"]) * rung_epochs[-1]
        task_summary[task] = {
            "metric": metric_name,
            "mode": mode,
            "trial_count": len(task_trials),
            "states": dict(states),
            "valid_trial_count": len(valid),
            "tpe_startup_trial_count": min(TPE_STARTUP_TRIALS, len(valid)),
            "estimated_tpe_guided_trial_count": max(0, len(valid) - TPE_STARTUP_TRIALS),
            "completed_100_epoch_count": len(completed),
            "failure_rate": states.get("FAIL", 0) / len(task_trials) if task_trials else None,
            "consumed_epoch_equivalents": consumed,
            "planned_full_epoch_equivalents": planned,
            "budget_fraction": consumed / planned if planned else None,
            "endpoint_winner_trial": endpoint_trial,
            "endpoint_winner_metric": endpoint_value,
            "checkpoint_winner_trial": None if checkpoint_winner is None else int(checkpoint_winner["trial_number"]),
            "checkpoint_winner_metric": None if checkpoint_winner is None else float(checkpoint_winner["checkpoint_best_metric"]),
            "checkpoint_winner_epoch": None if checkpoint_winner is None else int(checkpoint_winner["checkpoint_best_epoch"]),
            "checkpoint_winner_validation_metrics": {} if checkpoint_winner is None else {
                key.removeprefix("checkpoint_best_"): value
                for key, value in checkpoint_winner.items()
                if key.startswith("checkpoint_best_")
                and key not in {"checkpoint_best_metric", "checkpoint_best_epoch"}
            },
            "selection_changed_trial": checkpoint_winner is not None and int(checkpoint_winner["trial_number"]) != endpoint_trial,
            "median_best_to_endpoint_degradation": float(np.median(degradation)) if degradation else None,
            "rung_to_final_spearman": correlations,
            "failure_reasons": dict(Counter(row["failure_reason"] for row in task_trials if row["failure_reason"])),
            "split_sha256": split.get("split_sha256"),
            "train_subject_count": len(split.get("train_subjects", [])),
            "validation_subject_count": len(split.get("validation_subjects", [])),
            "reserved_test_subject_count": len(split.get("reserved_test_subjects", [])),
            "protected_test_opened": bool(split.get("protected_test_opened", False)),
        }

    overall_states = Counter(row["state"] for row in trial_rows)
    intended = len(tasks) * int(launch["n_trials_per_task"]) * int(rung_epochs[-1])
    consumed = sum(int(row["max_epoch_budget"]) for row in trial_rows)
    changed = sum(bool(row["selection_changed_trial"]) for row in task_summary.values())
    report = {
        "schema": "sta_net_tuning_audit_v1",
        "generated_at": utc_now(),
        "source_run_root": str(root),
        "study_id": study_id,
        "rung_epochs": rung_epochs,
        "task_count": len(tasks),
        "trial_count": len(trial_rows),
        "overall_states": dict(overall_states),
        "consumed_epoch_equivalents": consumed,
        "planned_full_epoch_equivalents": intended,
        "budget_fraction": consumed / intended if intended else None,
        "endpoint_checkpoint_selection_changed_tasks": changed,
        "trainer_hash_counts": dict(all_trainer_hashes),
        "tasks": task_summary,
        "claim_boundary": "Development split tuning audit only; protected test data were not opened.",
    }
    return report, trial_rows, rung_rows, curves


def plot_rungs(report: Mapping[str, Any], rung_rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    apply_style()
    tasks = list(report["tasks"])
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 9.0), constrained_layout=True)
    axes_flat = list(axes.flat)
    by_trial: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rung_rows:
        by_trial.setdefault((str(row["task"]), int(row["trial_number"])), []).append(row)
    for ax, task in zip(axes_flat, tasks):
        summary = report["tasks"][task]
        endpoint_trial = summary["endpoint_winner_trial"]
        checkpoint_trial = summary["checkpoint_winner_trial"]
        for (row_task, trial_number), rows in by_trial.items():
            if row_task != task:
                continue
            rows = sorted(rows, key=lambda row: int(row["epoch_budget"]))
            state = str(rows[-1]["state"])
            color, marker, linestyle = STATE_STYLE.get(state, STATE_STYLE["RUNNING"])
            width = 2.2 if trial_number == checkpoint_trial else (1.6 if trial_number == endpoint_trial else 0.8)
            alpha = 1.0 if trial_number in {endpoint_trial, checkpoint_trial} else 0.55
            ax.plot(
                [row["epoch_budget"] for row in rows],
                [row["objective_metric"] for row in rows],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=width,
                markersize=3.5,
                alpha=alpha,
            )
            if trial_number in {endpoint_trial, checkpoint_trial}:
                tag = "E" if trial_number == endpoint_trial else "C"
                if trial_number == endpoint_trial == checkpoint_trial:
                    tag = "E/C"
                ax.annotate(
                    f"{tag}: T{trial_number}",
                    (rows[-1]["epoch_budget"], rows[-1]["objective_metric"]),
                    xytext=(3, 2), textcoords="offset points", fontsize=6,
                )
        ax.set_xscale("log")
        ax.set_xticks(report["rung_epochs"], labels=[str(value) for value in report["rung_epochs"]])
        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("Training budget (epochs)")
        ax.set_ylabel(summary["metric"])
        ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    for ax in axes_flat[len(tasks):]:
        ax.axis("off")
    handles = [
        plt.Line2D([], [], color=color, marker=marker, linestyle=linestyle, label=state.title())
        for state, (color, marker, linestyle) in STATE_STYLE.items()
        if state in {"COMPLETE", "PRUNED", "FAIL"}
    ]
    handles.extend([
        plt.Line2D([], [], color="black", linewidth=1.6, label="E: endpoint winner"),
        plt.Line2D([], [], color="black", linewidth=2.2, label="C: checkpoint winner"),
    ])
    fig.legend(handles=handles, loc="lower right", frameon=False, ncol=1)
    fig.suptitle("STA-Net multi-fidelity tuning trajectories", fontsize=11)
    save_figure(fig, output, "tuning_rung_trajectories")


def plot_curves(report: Mapping[str, Any], curves: Mapping[str, Sequence[Mapping[str, Any]]], output: Path) -> None:
    apply_style()
    tasks = list(report["tasks"])
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 9.0), constrained_layout=True)
    axes_flat = list(axes.flat)
    for ax, task in zip(axes_flat, tasks):
        summary = report["tasks"][task]
        selected = summary["checkpoint_winner_trial"]
        for row in curves[task]:
            if row["state"] != "COMPLETE":
                continue
            is_selected = int(row["trial_number"]) == selected
            ax.plot(
                row["epochs"], row["metrics"],
                color="#0072B2" if is_selected else "#999999",
                linewidth=2.0 if is_selected else 0.7,
                alpha=1.0 if is_selected else 0.5,
            )
        if selected is not None:
            ax.axvline(summary["checkpoint_winner_epoch"], color="#D55E00", linestyle="--", linewidth=0.9)
            ax.text(
                0.98, 0.03,
                f"checkpoint T{selected} @ epoch {summary['checkpoint_winner_epoch']}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6,
            )
        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(summary["metric"])
        ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    for ax in axes_flat[len(tasks):]:
        ax.axis("off")
    fig.suptitle("Validation trajectories of 100-epoch trials", fontsize=11)
    save_figure(fig, output, "completed_trial_validation_curves")


def plot_overview(report: Mapping[str, Any], output: Path) -> None:
    apply_style()
    tasks = list(report["tasks"])
    labels = [TASK_LABELS[task] for task in tasks]
    y = np.arange(len(tasks))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), constrained_layout=True)
    left = np.zeros(len(tasks))
    for state in ("COMPLETE", "PRUNED", "FAIL"):
        values = np.asarray([report["tasks"][task]["states"].get(state, 0) for task in tasks])
        color, _, _ = STATE_STYLE[state]
        axes[0].barh(y, values, left=left, color=color, label=state.title())
        left += values
    axes[0].set_yticks(y, labels=labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Trial count")
    axes[0].set_title("Trial outcomes")
    axes[0].legend(frameon=False, loc="lower right")

    fractions = [100.0 * report["tasks"][task]["budget_fraction"] for task in tasks]
    axes[1].barh(y, fractions, color="#56B4E9")
    axes[1].axvline(100, color="#333333", linestyle="--", linewidth=0.8)
    axes[1].set_yticks(y, labels=labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 105)
    axes[1].set_xlabel("Used / full-grid epoch budget (%)")
    axes[1].set_title("Multi-fidelity budget use")
    fig.suptitle("STA-Net tuning audit overview", fontsize=11)
    save_figure(fig, output, "tuning_audit_overview")


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# STA-Net tuning audit: `{report['study_id']}`",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "> Development-split audit only. Protected test data were not opened.",
        "",
        "## Executive summary",
        "",
        (
            f"The database contains {report['trial_count']} trials across {report['task_count']} tasks: "
            + ", ".join(f"{state}={count}" for state, count in sorted(report["overall_states"].items()))
            + "."
        ),
        "",
        (
            f"Multi-fidelity execution consumed {report['consumed_epoch_equivalents']} of "
            f"{report['planned_full_epoch_equivalents']} full-grid epoch equivalents "
            f"({100.0 * report['budget_fraction']:.1f}%)."
        ),
        "",
        (
            f"Endpoint-based Optuna selection and historical best-checkpoint selection choose different "
            f"trials for {report['endpoint_checkpoint_selection_changed_tasks']} of {report['task_count']} tasks."
        ),
        "",
        f"Observed trainer hashes: {len(report['trainer_hash_counts'])}. Counts: `{json.dumps(report['trainer_hash_counts'], sort_keys=True)}`.",
        "",
        "## Per-task audit",
        "",
        "| Task | Metric | Val subjects | Complete / Pruned / Failed | TPE-guided est. | Budget | Endpoint winner | Checkpoint winner | Changed | Median endpoint degradation |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |",
    ]
    for task, row in report["tasks"].items():
        states = row["states"]
        lines.append(
            "| " + " | ".join([
                TASK_LABELS[task],
                row["metric"],
                str(row["validation_subject_count"]),
                f"{states.get('COMPLETE', 0)} / {states.get('PRUNED', 0)} / {states.get('FAIL', 0)}",
                str(row["estimated_tpe_guided_trial_count"]),
                f"{100.0 * row['budget_fraction']:.1f}%",
                f"T{row['endpoint_winner_trial']} ({fmt(row['endpoint_winner_metric'])})",
                (
                    "NA" if row["checkpoint_winner_trial"] is None else
                    f"T{row['checkpoint_winner_trial']} @ e{row['checkpoint_winner_epoch']} ({fmt(row['checkpoint_winner_metric'])})"
                ),
                "yes" if row["selection_changed_trial"] else "no",
                fmt(row["median_best_to_endpoint_degradation"]),
            ]) + " |"
        )
    lines.extend([
        "",
        "`Median endpoint degradation` is the non-negative loss from each completed trial's best validation checkpoint to epoch 100; it is not a cross-task aggregate endpoint.",
        "`TPE-guided est.` assumes the tuning-v1 setting of six COMPLETE/PRUNED startup trials; failed trials do not count toward TPE startup.",
        "",
        "## Exploratory development checkpoint metrics",
        "",
        "These are pooled validation-window metrics selected on the same development partition; they are not subject-level outer-fold estimates.",
        "",
        "| Task | Trial @ epoch | Accuracy | Balanced accuracy | Macro F1 | Kappa |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for task, row in report["tasks"].items():
        if row["mode"] != "max":
            continue
        metrics = row["checkpoint_winner_validation_metrics"]
        lines.append(
            f"| {TASK_LABELS[task]} | T{row['checkpoint_winner_trial']} @ e{row['checkpoint_winner_epoch']} | "
            f"{fmt(metrics.get('accuracy'))} | {fmt(metrics.get('balanced_accuracy'))} | "
            f"{fmt(metrics.get('macro_f1'))} | {fmt(metrics.get('cohen_kappa'))} |"
        )
    regression = report["tasks"].get("refed_regression")
    if regression:
        metrics = regression["checkpoint_winner_validation_metrics"]
        lines.extend([
            "",
            "| Regression task | Trial @ epoch | Masked MAE (scaled) | Masked RMSE (scaled) |",
            "| --- | ---: | ---: | ---: |",
            f"| REFED regression | T{regression['checkpoint_winner_trial']} @ e{regression['checkpoint_winner_epoch']} | "
            f"{fmt(metrics.get('masked_mae_scaled'))} | {fmt(metrics.get('masked_rmse_scaled'))} |",
        ])
    lines.extend([
        "",
        "## Rung predictiveness",
        "",
        "Spearman correlations are computed only among trials that reached 100 epochs. Small completed-trial counts make these descriptive, not inferential.",
        "",
        "| Task | 2→100 | 8→100 | 20→100 | 40→100 | n complete |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for task, row in report["tasks"].items():
        corr = row["rung_to_final_spearman"]
        lines.append(
            f"| {TASK_LABELS[task]} | {fmt(corr.get('2'))} | {fmt(corr.get('8'))} | "
            f"{fmt(corr.get('20'))} | {fmt(corr.get('40'))} | {row['completed_100_epoch_count']} |"
        )
    failures = Counter()
    for row in report["tasks"].values():
        failures.update(row["failure_reasons"])
    lines.extend(["", "## Failures", ""])
    if failures:
        for reason, count in failures.most_common():
            lines.append(f"- {count} × `{reason}`")
    else:
        lines.append("No failed trials were recorded.")
    lines.extend([
        "",
        "## Generated views",
        "",
        "- `tuning_rung_trajectories.svg/png`: rung-level trial trajectories; E is the Optuna endpoint winner and C is the historical checkpoint winner.",
        "- `completed_trial_validation_curves.svg/png`: per-epoch curves for full-budget trials, highlighting the checkpoint winner.",
        "- `tuning_audit_overview.svg/png`: state counts and effective budget use.",
        "- `trials.csv`, `rungs.csv`, and `summary.json`: machine-readable source tables.",
        "",
    ])
    return "\n".join(lines)


def run(root: Path, output: Path, *, plots: bool = True) -> dict[str, Any]:
    if not (root / "optuna.sqlite3").exists() or not (root / "launch_manifest.json").exists():
        raise FileNotFoundError(f"not a STA-Net tuning root: {root}")
    output.mkdir(parents=True, exist_ok=True)
    report, trials, rungs, curves = collect_run(root)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(output / "trials.csv", trials)
    write_csv(output / "rungs.csv", rungs)
    (output / "tuning_report.md").write_text(render_markdown(report), encoding="utf-8")
    if plots:
        plot_rungs(report, rungs, output)
        plot_curves(report, curves, output)
        plot_overview(report, output)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, help="Tuning directory containing optuna.sqlite3")
    parser.add_argument("--output-dir", default=None, help="Default: <run-root>/analysis")
    parser.add_argument("--no-plots", action="store_true", help="Write only JSON, CSV, and Markdown")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_root / "analysis"
    result = run(run_root, output_dir, plots=not args.no_plots)
    print(json.dumps({
        "status": "completed",
        "output_dir": str(output_dir),
        "trial_count": result["trial_count"],
        "protected_test_opened": False,
    }, indent=2))
