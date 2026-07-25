#!/usr/bin/env python3
"""Re-audit E2 semantic-objective results and generate diagnostic figures.

The registered E2 decision remains authoritative.  This script adds:

* artifact-integrity checks;
* representation-, coordinate-, subject-, and training-dynamics summaries;
* a subject-clustered bootstrap that preserves subject identity across seeds;
* a modality-balanced sensitivity estimand;
* publication-ready PNG/PDF figures.

No protected-test data are read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml


ROWS = ("T0", "T1", "T2")
SEMANTIC_ROWS = ("T1", "T2")
MODALITIES = ("eeg", "fnirs")
SEEDS = (20260719, 20260720, 20260721)
REPRESENTATIONS = (
    "hard_id",
    "continuous_latent",
    "posterior",
    "codebook_embedding",
)
COLORS = {
    "T0": "#0072B2",
    "T1": "#E69F00",
    "T2": "#D55E00",
    "eeg": "#CC79A7",
    "fnirs": "#009E73",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: mpl.figure.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )


def exact_cluster_bootstrap(values: Sequence[float]) -> dict[str, Any]:
    """Exact n-out-of-n subject bootstrap for small validation cohorts."""

    array = np.asarray(values, dtype=np.float64)
    if array.size > 8:
        raise ValueError("Exact bootstrap is intentionally limited to <=8 clusters")
    choices = itertools.product(range(array.size), repeat=array.size)
    draws = np.fromiter(
        (float(array[list(indices)].mean()) for indices in choices),
        dtype=np.float64,
        count=array.size ** array.size,
    )
    return {
        "cluster_count": int(array.size),
        "exact_draw_count": int(draws.size),
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "positive_cluster_count": int((array > 0).sum()),
    }


def subject_delta(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    subject: str,
    *,
    modality_balanced: bool,
) -> float:
    modality_values: list[float] = []
    coordinate_values: list[float] = []
    for modality in MODALITIES:
        left = np.asarray(
            baseline["representations"][modality]["hard_id"]["subject_r2"][subject],
            dtype=np.float64,
        )
        right = np.asarray(
            candidate["representations"][modality]["hard_id"]["subject_r2"][subject],
            dtype=np.float64,
        )
        delta = right - left
        modality_values.append(float(delta.mean()))
        coordinate_values.extend(delta.tolist())
    return float(
        np.mean(modality_values if modality_balanced else coordinate_values)
    )


def audit_integrity(
    suite_dir: Path,
    runs: Mapping[tuple[str, int], Mapping[str, Any]],
    suite_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    expected = {(row, seed) for row in ROWS for seed in SEEDS}
    record(
        "complete_row_seed_grid",
        set(runs) == expected,
        f"observed={len(runs)}, expected={len(expected)}",
    )
    record(
        "protected_test_closed_evaluation",
        not bool(
            load_json(suite_dir / "evaluation" / "manifest.json").get(
                "protected_test_opened", True
            )
        ),
        "evaluation manifest",
    )
    record(
        "protected_test_closed_suite",
        not bool(suite_manifest.get("protected_test_opened", True)),
        "suite manifest",
    )

    split_hashes: set[str] = set()
    run_status: list[str] = []
    step_counts: list[int] = []
    metric_lengths: list[int] = []
    checkpoint_matches = 0
    checkpoint_total = 0
    for row, seed in sorted(expected):
        run_dir = Path(str(runs[(row, seed)]["run_dir"]))
        manifest = load_json(run_dir / "manifest.json")
        run_status.append(str(manifest.get("status")))
        step_counts.append(int(manifest.get("global_step", -1)))
        split_hashes.add(str(manifest.get("split_sha256")))
        metric_lengths.append(len(load_jsonl(run_dir / "metrics" / "validation.jsonl")))
        for filename, expected_hash in manifest["checkpoint_sha256"].items():
            checkpoint_total += 1
            if sha256(run_dir / "checkpoints" / filename) == expected_hash:
                checkpoint_matches += 1
    record(
        "all_runs_training_complete",
        all(value == "training_complete" for value in run_status),
        f"{sum(value == 'training_complete' for value in run_status)}/9",
    )
    record(
        "all_runs_462_steps",
        all(value == 462 for value in step_counts),
        f"range={min(step_counts)}-{max(step_counts)}",
    )
    record(
        "all_runs_14_validation_epochs",
        all(value == 14 for value in metric_lengths),
        f"range={min(metric_lengths)}-{max(metric_lengths)}",
    )
    record(
        "checkpoint_hashes_match",
        checkpoint_matches == checkpoint_total,
        f"{checkpoint_matches}/{checkpoint_total}",
    )
    record(
        "single_split_hash",
        len(split_hashes) == 1,
        ",".join(sorted(split_hashes)),
    )

    sidecar_manifest = load_json(
        Path(
            "data/cache/physiology_semantic_targets_v1/"
            "adaptive_ssm_e2_development/manifest.json"
        )
    )
    sidecar_path = Path(
        "data/cache/physiology_semantic_targets_v1/"
        "adaptive_ssm_e2_development/targets.npz"
    )
    record(
        "sidecar_array_hash_matches",
        sha256(sidecar_path) == sidecar_manifest["arrays_sha256"],
        sidecar_manifest["arrays_sha256"],
    )
    calibration_path = Path(
        str(suite_manifest["semantic_weight_calibration_path"])
    )
    record(
        "weight_calibration_hash_matches",
        sha256(calibration_path)
        == suite_manifest["semantic_weight_calibration_sha256"],
        suite_manifest["semantic_weight_calibration_sha256"],
    )
    return checks


def representation_rows(
    runs: Mapping[tuple[str, int], Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in ROWS:
        for seed in SEEDS:
            run = runs[(row, seed)]
            for modality in MODALITIES:
                for representation in REPRESENTATIONS:
                    payload = run["representations"][modality][representation]
                    null_q95 = ""
                    null_margin = ""
                    above_null = ""
                    if representation == "hard_id":
                        null_q95 = float(
                            payload["shuffled_target_null"]["mean_r2_q95"]
                        )
                        null_margin = float(payload["mean_r2"]) - null_q95
                        above_null = bool(
                            payload["shuffled_target_null"]["observed_above_q95"]
                        )
                    rows.append(
                        {
                            "row": row,
                            "seed": seed,
                            "modality": modality,
                            "representation": representation,
                            "mean_r2": float(payload["mean_r2"]),
                            "null_q95": null_q95,
                            "null_margin": null_margin,
                            "above_null": above_null,
                            "train_token_count": int(payload["train_token_count"]),
                            "validation_token_count": int(
                                payload["validation_token_count"]
                            ),
                        }
                    )
    return rows


def coordinate_rows(
    runs: Mapping[tuple[str, int], Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in SEMANTIC_ROWS:
        for seed in SEEDS:
            for modality in MODALITIES:
                baseline = runs[("T0", seed)]["representations"][modality]["hard_id"]
                payload = runs[(candidate, seed)]["representations"][modality][
                    "hard_id"
                ]
                for index, coordinate in enumerate(payload["coordinate_names"]):
                    left = float(baseline["coordinate_r2"][index])
                    right = float(payload["coordinate_r2"][index])
                    rows.append(
                        {
                            "comparison": f"{candidate}-T0",
                            "seed": seed,
                            "modality": modality,
                            "coordinate": coordinate,
                            "baseline_r2": left,
                            "candidate_r2": right,
                            "delta_r2": right - left,
                        }
                    )
    return rows


def subject_sensitivity(
    runs: Mapping[tuple[str, int], Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    subjects = sorted(
        runs[("T0", SEEDS[0])]["representations"]["eeg"]["hard_id"][
            "subject_r2"
        ]
    )
    for candidate in SEMANTIC_ROWS:
        coordinate_weighted: list[float] = []
        modality_balanced: list[float] = []
        for subject in subjects:
            coordinate_by_seed = [
                subject_delta(
                    runs[("T0", seed)],
                    runs[(candidate, seed)],
                    subject,
                    modality_balanced=False,
                )
                for seed in SEEDS
            ]
            modality_by_seed = [
                subject_delta(
                    runs[("T0", seed)],
                    runs[(candidate, seed)],
                    subject,
                    modality_balanced=True,
                )
                for seed in SEEDS
            ]
            coordinate_mean = float(np.mean(coordinate_by_seed))
            modality_mean = float(np.mean(modality_by_seed))
            coordinate_weighted.append(coordinate_mean)
            modality_balanced.append(modality_mean)
            rows.append(
                {
                    "comparison": f"{candidate}-T0",
                    "subject": subject,
                    "coordinate_weighted_delta": coordinate_mean,
                    "modality_balanced_delta": modality_mean,
                    "coordinate_weighted_seed_values": ";".join(
                        f"{value:.10g}" for value in coordinate_by_seed
                    ),
                    "modality_balanced_seed_values": ";".join(
                        f"{value:.10g}" for value in modality_by_seed
                    ),
                }
            )
        summary[candidate] = {
            "coordinate_weighted": exact_cluster_bootstrap(coordinate_weighted),
            "modality_balanced": exact_cluster_bootstrap(modality_balanced),
        }
    return rows, summary


def training_rows(
    suite_dir: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, str], list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    history: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    objectives = (
        "eeg_state",
        "eeg_prototype",
        "fnirs_state",
        "fnirs_prototype",
    )
    for row in SEMANTIC_ROWS:
        for seed in SEEDS:
            run_dir = suite_dir / "runs" / f"{row.lower()}_seed{seed}"
            train = load_jsonl(run_dir / "metrics" / "train.jsonl")
            validation = load_jsonl(run_dir / "metrics" / "validation.jsonl")
            history[(row, seed, "train")] = train
            history[(row, seed, "validation")] = validation
            for objective in objectives:
                rows.append(
                    {
                        "row": row,
                        "seed": seed,
                        "objective": objective,
                        "train_start": float(train[0][objective]),
                        "train_final": float(train[-1][objective]),
                        "validation_start": float(validation[0][objective]),
                        "validation_final": float(validation[-1][objective]),
                        "train_relative_reduction": float(
                            1.0 - train[-1][objective] / train[0][objective]
                        ),
                        "validation_relative_reduction": float(
                            1.0
                            - validation[-1][objective] / validation[0][objective]
                        ),
                        "final_generalization_gap": float(
                            validation[-1][objective] - train[-1][objective]
                        ),
                    }
                )
    return rows, history


def support_rows(
    suite_dir: Path,
    runs: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    source = read_csv(suite_dir / "evaluation" / "prototype_signatures.csv")
    counts: dict[tuple[str, int, str, str], list[int]] = defaultdict(list)
    for item in source:
        counts[
            (
                item["row"],
                int(item["seed"]),
                item["modality"],
                item["split"],
            )
        ].append(int(item["count"]))

    rows: list[dict[str, Any]] = []
    possible = {
        ("eeg", "train"): 1800,
        ("eeg", "validation"): 500,
        ("fnirs", "train"): 1800,
        ("fnirs", "validation"): 500,
    }
    for row in ROWS:
        for seed in SEEDS:
            for modality in MODALITIES:
                hard = runs[(row, seed)]["representations"][modality]["hard_id"]
                for split, token_key in (
                    ("train", "train_token_count"),
                    ("validation", "validation_token_count"),
                ):
                    array = np.asarray(
                        counts[(row, seed, modality, split)], dtype=np.int64
                    )
                    positive = array[array > 0]
                    token_count = int(hard[token_key])
                    rows.append(
                        {
                            "row": row,
                            "seed": seed,
                            "modality": modality,
                            "split": split,
                            "valid_target_tokens": token_count,
                            "possible_target_tokens": possible[(modality, split)],
                            "valid_target_fraction": (
                                token_count / possible[(modality, split)]
                            ),
                            "active_signature_codes": int(positive.size),
                            "median_tokens_per_active_code": float(
                                np.median(positive)
                            ),
                            "active_codes_with_at_most_2_tokens_fraction": float(
                                np.mean(positive <= 2)
                            ),
                        }
                    )
    return rows


def quantizer_rows(
    decision: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    health = decision["quantizer_health"]["runs"]
    for row in ROWS:
        for seed in SEEDS:
            for modality in MODALITIES:
                payload = health[f"{row}_seed{seed}"]["modalities"][modality]
                rows.append(
                    {
                        "row": row,
                        "seed": seed,
                        "modality": modality,
                        "effective_codes": float(payload["effective_codes"]),
                        "active_codes": int(payload["epoch_active_codes"]),
                        "nearest_neighbor_cosine": float(
                            payload["nearest_neighbor_cosine"]
                        ),
                        "health_passed": bool(payload["passed"]),
                    }
                )
    return rows


def aggregate(
    rows: Sequence[Mapping[str, Any]],
    filters: Mapping[str, Any],
    field: str,
) -> np.ndarray:
    return np.asarray(
        [
            float(row[field])
            for row in rows
            if all(row[key] == value for key, value in filters.items())
        ],
        dtype=np.float64,
    )


def figure_primary(
    output_dir: Path,
    representation: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    subject_rows: Sequence[Mapping[str, Any]],
    subject_summary: Mapping[str, Any],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    x = np.arange(len(ROWS), dtype=float)
    offsets = (-0.08, 0.0, 0.08)
    for panel, modality in enumerate(MODALITIES):
        ax = axes.flat[panel]
        for seed_index, seed in enumerate(SEEDS):
            observed = [
                aggregate(
                    representation,
                    {
                        "row": row,
                        "seed": seed,
                        "modality": modality,
                        "representation": "hard_id",
                    },
                    "mean_r2",
                )[0]
                for row in ROWS
            ]
            null = [
                aggregate(
                    representation,
                    {
                        "row": row,
                        "seed": seed,
                        "modality": modality,
                        "representation": "hard_id",
                    },
                    "null_q95",
                )[0]
                for row in ROWS
            ]
            ax.plot(
                x + offsets[seed_index],
                observed,
                color=COLORS[modality],
                marker=("o", "s", "^")[seed_index],
                linewidth=0.9,
                alpha=0.75,
                label=f"Observed, seed {str(seed)[-2:]}",
            )
            ax.scatter(
                x + offsets[seed_index],
                null,
                color="#666666",
                marker="x",
                s=20,
                linewidth=0.8,
                label="Permutation q95" if seed_index == 0 else None,
            )
        ax.axhline(0, color="#999999", linewidth=0.7)
        ax.set_xticks(x, ROWS)
        ax.set_ylabel("Held-out hard-token mean $R^2$")
        ax.set_title(f"{modality.upper()} hard-token endpoint")
        if panel == 0:
            ax.legend(frameon=False, ncol=2, loc="lower left")
        panel_label(ax, chr(ord("A") + panel))

    ax = axes[1, 0]
    for candidate, marker in zip(SEMANTIC_ROWS, ("o", "s")):
        values = [
            decision["required_comparisons"][candidate][
                "seed_matched_pooled_endpoint_delta"
            ][str(seed)]
            for seed in SEEDS
        ]
        ax.plot(
            np.arange(3),
            values,
            marker=marker,
            color=COLORS[candidate],
            label=f"{candidate} - T0",
        )
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_xticks(np.arange(3), [str(seed)[-2:] for seed in SEEDS])
    ax.set_xlabel("Matched seed suffix")
    ax.set_ylabel("Two-modality pooled $\\Delta R^2$")
    ax.set_title("Registered seed-matched effect")
    ax.legend(frameon=False)
    panel_label(ax, "C")

    ax = axes[1, 1]
    rng = np.random.default_rng(20260724)
    for index, candidate in enumerate(SEMANTIC_ROWS):
        values = [
            float(row["modality_balanced_delta"])
            for row in subject_rows
            if row["comparison"] == f"{candidate}-T0"
        ]
        jitter = rng.uniform(-0.07, 0.07, size=len(values))
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            color=COLORS[candidate],
            edgecolor="white",
            linewidth=0.5,
            s=28,
            alpha=0.85,
        )
        summary = subject_summary[candidate]["modality_balanced"]
        ax.errorbar(
            index,
            summary["mean"],
            yerr=[
                [summary["mean"] - summary["ci95_low"]],
                [summary["ci95_high"] - summary["mean"]],
            ],
            fmt="D",
            color="#111111",
            capsize=3,
            markersize=4,
            linewidth=1.0,
        )
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_xticks((0, 1), ("T1 - T0", "T2 - T0"))
    ax.set_ylabel("Subject-clustered, modality-balanced $\\Delta R^2$")
    ax.set_title("Sensitivity analysis (5 subjects)")
    panel_label(ax, "D")
    fig.suptitle("E2 primary endpoint re-audit", fontsize=11, y=1.01)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig01_primary_endpoint_reaudit")


def figure_representation_support(
    output_dir: Path,
    representation: Sequence[Mapping[str, Any]],
    coordinates: Sequence[Mapping[str, Any]],
    support: Sequence[Mapping[str, Any]],
    e0_summary: Mapping[str, Any],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0))

    ax = axes[0, 0]
    heat = np.zeros((6, 4), dtype=float)
    labels: list[str] = []
    for index, (row, modality) in enumerate(
        itertools.product(ROWS, MODALITIES)
    ):
        labels.append(f"{row} {modality.upper()}")
        for column, rep in enumerate(REPRESENTATIONS):
            heat[index, column] = aggregate(
                representation,
                {"row": row, "modality": modality, "representation": rep},
                "mean_r2",
            ).mean()
    limit = float(np.max(np.abs(heat)))
    image = ax.imshow(heat, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_yticks(np.arange(6), labels)
    ax.set_xticks(
        np.arange(4),
        ("Hard ID", "Continuous", "Posterior", "Codebook"),
        rotation=25,
        ha="right",
    )
    ax.set_title("Representation ladder (seed mean)")
    fig.colorbar(image, ax=ax, shrink=0.75, label="Held-out mean $R^2$")
    panel_label(ax, "A")

    ax = axes[0, 1]
    coordinate_order = (
        "r_mean",
        "r_slope",
        "delta_hbo_mean",
        "delta_hb_mean",
        "delta_hbo_slope",
        "delta_hb_slope",
    )
    delta_heat = np.zeros((2, len(coordinate_order)), dtype=float)
    for i, candidate in enumerate(SEMANTIC_ROWS):
        for j, coordinate in enumerate(coordinate_order):
            delta_heat[i, j] = aggregate(
                coordinates,
                {"comparison": f"{candidate}-T0", "coordinate": coordinate},
                "delta_r2",
            ).mean()
    limit = float(np.max(np.abs(delta_heat)))
    image = ax.imshow(
        delta_heat, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto"
    )
    for i in range(delta_heat.shape[0]):
        for j in range(delta_heat.shape[1]):
            ax.text(
                j,
                i,
                f"{delta_heat[i, j]:+.2f}",
                ha="center",
                va="center",
                fontsize=6,
            )
    ax.set_yticks((0, 1), ("T1 - T0", "T2 - T0"))
    ax.set_xticks(
        np.arange(len(coordinate_order)),
        (
            "r mean",
            "r slope",
            "HbO mean",
            "HbR mean",
            "HbO slope",
            "HbR slope",
        ),
        rotation=32,
        ha="right",
    )
    ax.set_title("Coordinate-specific semantic effect")
    fig.colorbar(image, ax=ax, shrink=0.75, label="Mean $\\Delta R^2$")
    panel_label(ax, "B")

    ax = axes[1, 0]
    positions = np.arange(4)
    values = []
    for modality in MODALITIES:
        for split in ("train", "validation"):
            values.append(
                aggregate(
                    support,
                    {"row": "T1", "modality": modality, "split": split},
                    "valid_target_fraction",
                ).mean()
            )
    bars = ax.bar(
        positions,
        values,
        color=[
            COLORS["eeg"],
            COLORS["eeg"],
            COLORS["fnirs"],
            COLORS["fnirs"],
        ],
        alpha=0.80,
    )
    bars[1].set_hatch("//")
    bars[3].set_hatch("//")
    ax.set_xticks(
        positions,
        ("EEG train", "EEG val", "fNIRS train", "fNIRS val"),
        rotation=20,
        ha="right",
    )
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Valid target-token fraction")
    ax.set_title("Usable target support after signal masks")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.03,
            f"{value:.1%}",
            ha="center",
            fontsize=7,
        )
    panel_label(ax, "C")

    ax = axes[1, 1]
    target_space = {
        item["modality"]: float(item["validation_global_r2"])
        for item in e0_summary["vocabulary"]
    }
    realized = {
        modality: max(
            aggregate(
                representation,
                {
                    "row": row,
                    "modality": modality,
                    "representation": "hard_id",
                },
                "mean_r2",
            ).mean()
            for row in ROWS
        )
        for modality in MODALITIES
    }
    x = np.arange(2)
    width = 0.34
    ax.bar(
        x - width / 2,
        [target_space[m] for m in MODALITIES],
        width,
        label="E0 target-space K=128",
        color="#56B4E9",
    )
    ax.bar(
        x + width / 2,
        [realized[m] for m in MODALITIES],
        width,
        label="Best realized E2 hard ID",
        color="#E69F00",
    )
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_xticks(x, ("EEG", "fNIRS"))
    ax.set_ylabel("Diagnostic $R^2$")
    ax.set_title("Target quantizability vs tokenizer realization")
    ax.legend(frameon=False)
    panel_label(ax, "D")
    fig.suptitle("Where the physiological constraint is lost", fontsize=11, y=1.01)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig02_representation_and_support")


def figure_training(
    output_dir: Path,
    history: Mapping[tuple[str, int, str], Sequence[Mapping[str, Any]]],
) -> None:
    objectives = (
        ("eeg_state", "EEG state loss"),
        ("eeg_prototype", "EEG prototype loss"),
        ("fnirs_state", "fNIRS state loss"),
        ("fnirs_prototype", "fNIRS prototype loss"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
    for panel, (objective, title) in enumerate(objectives):
        ax = axes.flat[panel]
        for row in SEMANTIC_ROWS:
            for split, linestyle in (("train", "-"), ("validation", "--")):
                arrays = np.asarray(
                    [
                        [float(item[objective]) for item in history[(row, seed, split)]]
                        for seed in SEEDS
                    ],
                    dtype=np.float64,
                )
                epochs = np.arange(arrays.shape[1])
                mean = arrays.mean(axis=0)
                low = arrays.min(axis=0)
                high = arrays.max(axis=0)
                ax.plot(
                    epochs,
                    mean,
                    color=COLORS[row],
                    linestyle=linestyle,
                    linewidth=1.3,
                    label=f"{row} {split}",
                )
                ax.fill_between(
                    epochs,
                    low,
                    high,
                    color=COLORS[row],
                    alpha=0.10,
                    linewidth=0,
                )
        ax.set_title(title)
        ax.set_ylabel("Unweighted objective loss")
        ax.set_xlabel("Epoch")
        if panel == 0:
            ax.legend(frameon=False, ncol=2)
        panel_label(ax, chr(ord("A") + panel))
    fig.suptitle(
        "Semantic loss dynamics (line = 3-seed mean, band = seed range)",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, output_dir, "fig03_semantic_training_dynamics")


def figure_calibration_health(
    output_dir: Path,
    calibration_rows: Sequence[Mapping[str, str]],
    quantizer: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    representation: Sequence[Mapping[str, Any]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7))

    ax = axes[0, 0]
    objectives = (
        "eeg_state",
        "eeg_prototype",
        "fnirs_state",
        "fnirs_prototype",
    )
    objective_colors = ("#CC79A7", "#0072B2", "#009E73", "#E69F00")
    for objective, color in zip(objectives, objective_colors):
        selected = [
            row for row in calibration_rows if row["objective"] == objective
        ]
        x = np.asarray([float(row["weight"]) for row in selected])
        y = np.asarray([float(row["median_ratio"]) for row in selected])
        order = np.argsort(x)
        ax.plot(x[order], y[order], marker="o", color=color, label=objective)
    ax.axhspan(0.1, 10, color="#999999", alpha=0.13, label="Admission band")
    ax.axvline(0.005, color="#111111", linestyle=":", label="Selected 0.005")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Semantic weight")
    ax.set_ylabel("Gradient norm ratio vs reconstruction")
    ax.set_title("Training-gradient calibration")
    ax.legend(frameon=False, ncol=2)
    panel_label(ax, "A")

    ax = axes[0, 1]
    x = np.arange(len(ROWS), dtype=float)
    width = 0.34
    for offset, modality in ((-width / 2, "eeg"), (width / 2, "fnirs")):
        means = [
            aggregate(
                quantizer, {"row": row, "modality": modality}, "effective_codes"
            ).mean()
            for row in ROWS
        ]
        errors = [
            aggregate(
                quantizer, {"row": row, "modality": modality}, "effective_codes"
            ).std(ddof=1)
            for row in ROWS
        ]
        ax.bar(
            x + offset,
            means,
            width,
            yerr=errors,
            capsize=2,
            color=COLORS[modality],
            label=modality.upper(),
        )
    ax.set_xticks(x, ROWS)
    ax.set_ylabel("Effective codes (mean ± SD, 3 seeds)")
    ax.set_title("Quantizer occupancy remains healthy")
    ax.legend(frameon=False)
    panel_label(ax, "B")

    ax = axes[1, 0]
    for modality, marker in (("eeg", "o"), ("fnirs", "s")):
        values = [
            decision["prototype_stability"][row][modality][
                "mean_matched_cosine"
            ]
            for row in ROWS
        ]
        ax.plot(
            x,
            values,
            marker=marker,
            color=COLORS[modality],
            label=modality.upper(),
        )
    ax.set_xticks(x, ROWS)
    ax.set_ylim(0.90, 0.95)
    ax.set_ylabel("Mean matched prototype cosine")
    ax.set_title("Cross-seed prototype stability")
    ax.legend(frameon=False)
    panel_label(ax, "C")

    ax = axes[1, 1]
    for row in ROWS:
        for modality, marker in (("eeg", "o"), ("fnirs", "s")):
            effective = aggregate(
                quantizer, {"row": row, "modality": modality}, "effective_codes"
            )
            hard = aggregate(
                representation,
                {
                    "row": row,
                    "modality": modality,
                    "representation": "hard_id",
                },
                "mean_r2",
            )
            ax.scatter(
                effective,
                hard,
                color=COLORS[row],
                marker=marker,
                s=28,
                alpha=0.85,
                label=f"{row} {modality.upper()}",
            )
    ax.axhline(0, color="#777777", linewidth=0.7)
    ax.set_xlabel("Effective codes")
    ax.set_ylabel("Hard-token held-out mean $R^2$")
    ax.set_title("More occupied codes do not imply semantics")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, ncol=2, fontsize=6)
    panel_label(ax, "D")
    fig.suptitle("Optimization and quantizer diagnostics", fontsize=11, y=1.01)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig04_calibration_and_quantizer_health")


def summarize_for_report(
    representation: Sequence[Mapping[str, Any]],
    coordinates: Sequence[Mapping[str, Any]],
    support: Sequence[Mapping[str, Any]],
    training: Sequence[Mapping[str, Any]],
    history: Mapping[tuple[str, int, str], Sequence[Mapping[str, Any]]],
    subject_summary: Mapping[str, Any],
    decision: Mapping[str, Any],
    integrity: Sequence[Mapping[str, Any]],
    e0_summary: Mapping[str, Any],
) -> dict[str, Any]:
    hard: dict[str, Any] = {}
    for row in ROWS:
        hard[row] = {}
        for modality in MODALITIES:
            values = aggregate(
                representation,
                {
                    "row": row,
                    "modality": modality,
                    "representation": "hard_id",
                },
                "mean_r2",
            )
            margins = aggregate(
                representation,
                {
                    "row": row,
                    "modality": modality,
                    "representation": "hard_id",
                },
                "null_margin",
            )
            hard[row][modality] = {
                "mean_r2": float(values.mean()),
                "sd_r2": float(values.std(ddof=1)),
                "mean_null_margin": float(margins.mean()),
                "minimum_null_margin": float(margins.min()),
                "maximum_null_margin": float(margins.max()),
                "above_null_count": int((margins > 0).sum()),
            }

    coordinate_summary: dict[str, Any] = {}
    for candidate in SEMANTIC_ROWS:
        coordinate_summary[candidate] = {}
        for coordinate in sorted(
            {
                row["coordinate"]
                for row in coordinates
                if row["comparison"] == f"{candidate}-T0"
            }
        ):
            values = aggregate(
                coordinates,
                {"comparison": f"{candidate}-T0", "coordinate": coordinate},
                "delta_r2",
            )
            coordinate_summary[candidate][coordinate] = {
                "mean_delta": float(values.mean()),
                "positive_seed_count": int((values > 0).sum()),
            }

    training_summary: dict[str, Any] = {}
    for row in SEMANTIC_ROWS:
        training_summary[row] = {}
        for objective in (
            "eeg_state",
            "eeg_prototype",
            "fnirs_state",
            "fnirs_prototype",
        ):
            selected = [
                item
                for item in training
                if item["row"] == row and item["objective"] == objective
            ]
            training_summary[row][objective] = {
                key: float(np.mean([float(item[key]) for item in selected]))
                for key in (
                    "train_relative_reduction",
                    "validation_relative_reduction",
                    "final_generalization_gap",
                )
            }

    support_summary: dict[str, Any] = {}
    for modality in MODALITIES:
        support_summary[modality] = {}
        for split in ("train", "validation"):
            selected = [
                item
                for item in support
                if item["row"] == "T1"
                and item["modality"] == modality
                and item["split"] == split
            ]
            support_summary[modality][split] = {
                key: float(np.mean([float(item[key]) for item in selected]))
                for key in (
                    "valid_target_tokens",
                    "possible_target_tokens",
                    "valid_target_fraction",
                    "active_signature_codes",
                    "median_tokens_per_active_code",
                    "active_codes_with_at_most_2_tokens_fraction",
                )
            }

    sample_coverage: dict[str, Any] = {}
    for split in ("train", "validation"):
        values = np.asarray(
            [
                float(item["auxiliary_target_coverage"])
                for row in SEMANTIC_ROWS
                for seed in SEEDS
                for item in history[(row, seed, split)]
            ],
            dtype=np.float64,
        )
        sample_coverage[split] = {
            "mean": float(values.mean()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }

    return {
        "schema": "physiology_semantic_e2_comprehensive_review_v1",
        "registered_decision": decision["decision"],
        "decision_unchanged": True,
        "integrity_all_passed": all(bool(item["passed"]) for item in integrity),
        "hard_token_summary": hard,
        "coordinate_effects": coordinate_summary,
        "subject_cluster_sensitivity": subject_summary,
        "training_generalization": training_summary,
        "target_support": support_summary,
        "sample_level_auxiliary_target_coverage": sample_coverage,
        "registered_required_comparisons": decision["required_comparisons"],
        "registered_optional_t2_vs_t1": decision["optional_t2_vs_t1"],
        "prototype_stability": decision["prototype_stability"],
        "e0_inherited_constraints": {
            "validation": e0_summary["validation"],
            "physical_observation": e0_summary["physical_observation"],
            "posterior_calibration_pass": e0_summary["posterior_calibration"][
                "posterior_calibration_pass"
            ],
            "target_space_vocabulary": e0_summary["vocabulary"],
        },
        "statistical_audit": {
            "registered_run_endpoint": (
                "token-pooled within each modality, then equal modality mean"
            ),
            "registered_subject_bootstrap": (
                "coordinate-weighted (2 EEG + 4 fNIRS) and independently "
                "resampled within each seed"
            ),
            "added_sensitivity": (
                "subjects clustered across fixed seeds; both coordinate-weighted "
                "and equal-modality estimands; exact 5^5 bootstrap"
            ),
            "interpretation": (
                "The retain-T0 decision is robust, but the original bootstrap "
                "intervals should not be interpreted as subject-clustered "
                "uncertainty for the registered run-level estimand."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-dir",
        type=Path,
        default=Path(
            "experiments/runs/physiology_semantic_tokenizer/"
            "e2_semantic_objectives/"
            "20260723_e2_v4_semantic_objective_suite_v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <suite-dir>/comprehensive_review_20260724",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_dir = args.suite_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else suite_dir / "comprehensive_review_20260724"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    decoding = load_json(suite_dir / "evaluation" / "state_decoding.json")
    runs = {
        (str(run["row"]), int(run["seed"])): run for run in decoding["runs"]
    }
    decision = load_json(suite_dir / "decision" / "decision.json")
    suite_manifest = load_json(suite_dir / "suite_manifest.json")
    e0_summary = load_json(
        Path(
            "experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/"
            "20260723_adaptive_teacher_e0_v3_line_clean_v4_revalidation_v1/"
            "summary.json"
        )
    )

    integrity = audit_integrity(suite_dir, runs, suite_manifest)
    representation = representation_rows(runs)
    coordinates = coordinate_rows(runs)
    subjects, subject_summary = subject_sensitivity(runs)
    training, history = training_rows(suite_dir)
    support = support_rows(suite_dir, runs)
    quantizer = quantizer_rows(decision)
    calibration = read_csv(
        Path(str(suite_manifest["semantic_weight_calibration_path"])).parent
        / "gradient_ratios.csv"
    )

    tables = output_dir / "tables"
    figures = output_dir / "figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    write_csv(tables / "integrity_checks.csv", integrity)
    write_csv(tables / "representation_metrics.csv", representation)
    write_csv(tables / "coordinate_effects.csv", coordinates)
    write_csv(tables / "subject_cluster_sensitivity.csv", subjects)
    write_csv(tables / "training_generalization.csv", training)
    write_csv(tables / "target_support_diagnostics.csv", support)
    write_csv(tables / "quantizer_health.csv", quantizer)

    summary = summarize_for_report(
        representation,
        coordinates,
        support,
        training,
        history,
        subject_summary,
        decision,
        integrity,
        e0_summary,
    )
    (output_dir / "review_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    figure_primary(figures, representation, decision, subjects, subject_summary)
    figure_representation_support(
        figures, representation, coordinates, support, e0_summary
    )
    figure_training(figures, history)
    figure_calibration_health(
        figures, calibration, quantizer, decision, representation
    )

    print(json.dumps(
        {
            "output_dir": str(output_dir),
            "registered_decision": decision["decision"],
            "integrity_all_passed": summary["integrity_all_passed"],
            "figure_count": len(list(figures.glob("*.png"))),
            "table_count": len(list(tables.glob("*.csv"))),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
