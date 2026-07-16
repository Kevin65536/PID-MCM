#!/usr/bin/env python3
"""Plot two Croce folds with the best held-out fNIRS variance recovery.

The parent formal-v3 evaluator uses a scalar shared-state SMC model.  It does
not estimate Croce's five physical states directly.  This post-hoc diagnostic
therefore keeps the saved joint neural driver, EEG-PC1 observation, and fNIRS
reconstruction unchanged, and integrates Croce's canonical hemodynamic ODE to
obtain explicitly labelled companion trajectories for the vasoactive signal
``s`` and relative blood flow ``f = 1 + delta_f``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.integrate import solve_ivp


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import evaluate_shared_neural_driver_unified as evaluator


DEFAULT_RUN_DIR = REPO_ROOT / (
    "experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/"
    "20260715_shared_neural_driver_unified_formal_v3"
)

MODEL_PARAMS = {
    "epsilon": 1.0,
    "kas": 0.41,
    "kaf": 0.65,
    "tau0": 2.0,
    "alpha": 0.32,
    "e0": 0.34,
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in values:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_best_variance_folds(fold_metrics_path: Path, count: int = 2) -> list[dict[str, Any]]:
    """Rank Croce-joint held-out folds by multiplicative distance from variance ratio 1.

    Positive PCC is required only to reject direction-inverted examples.  R2 is
    deliberately not part of the ranking because this diagnostic asks about
    variance recovery specifically; it is nevertheless retained and displayed.
    """

    candidates: list[dict[str, Any]] = []
    with fold_metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["validation"] != "leave_one_trial":
                continue
            if row["model"] != "croce_joint" or row["hrf_mode"] != "canonical":
                continue
            variance_ratio = float(row["variance_ratio"])
            pcc = float(row["pcc"])
            if not (math.isfinite(variance_ratio) and variance_ratio > 0 and math.isfinite(pcc) and pcc > 0):
                continue
            parsed: dict[str, Any] = dict(row)
            for field in (
                "mse", "r2", "pcc", "amplitude_ratio", "variance_ratio",
                "peak_to_peak_ratio", "mean_bias", "baseline_bias", "poststimulus_bias",
                "trend_direction_agreement", "poststimulus_slope_truth",
                "poststimulus_slope_estimate", "poststimulus_slope_sign_agreement",
                "affine_oracle_r2",
            ):
                parsed[field] = float(parsed[field])
            parsed["heldout_trial"] = int(parsed["heldout_trial"])
            parsed["variance_log_distance"] = abs(math.log(variance_ratio))
            candidates.append(parsed)
    candidates.sort(
        key=lambda row: (
            row["variance_log_distance"],
            -row["pcc"],
            -row["r2"],
            row["condition_id"],
            row["subject"],
            row["heldout_trial"],
        )
    )
    if len(candidates) < count:
        raise RuntimeError(f"only {len(candidates)} eligible Croce-joint folds")
    selected = candidates[:count]
    for rank, row in enumerate(selected, start=1):
        row["sample_rank"] = rank
        row["selection_rule"] = "min_abs_log_variance_ratio_with_positive_pcc"
    return selected


def _load_saved_trajectories(
    path: Path,
    selected: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int], dict[str, np.ndarray]]:
    keys = {
        (str(row["condition_id"]), str(row["subject"]), int(row["heldout_trial"]))
        for row in selected
    }
    buffers: dict[tuple[str, str, int], dict[str, list[float]]] = {
        key: {"time_s": [], "truth": [], "estimate": [], "driver": []} for key in keys
    }
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["validation"] != "leave_one_trial":
                continue
            if row["model"] != "croce_joint" or row["hrf_mode"] != "canonical":
                continue
            key = (row["condition_id"], row["subject"], int(row["heldout_trial"]))
            if key not in buffers:
                continue
            for field in ("time_s", "truth", "estimate", "driver"):
                buffers[key][field].append(float(row[field]))
    output: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
    for key, values in buffers.items():
        lengths = {len(value) for value in values.values()}
        if lengths != {200}:
            raise RuntimeError(f"unexpected saved trajectory lengths for {key}: {lengths}")
        output[key] = {name: np.asarray(value, dtype=np.float64) for name, value in values.items()}
    return output


def _replay_fold(
    grouped: Mapping[str, Mapping[str, Sequence[evaluator.Trial]]],
    config: Mapping[str, Any],
    row: Mapping[str, Any],
    saved: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    condition_id = str(row["condition_id"])
    subject = str(row["subject"])
    heldout = int(row["heldout_trial"])
    subjects = grouped[condition_id]
    subject_order = sorted(subjects)
    subject_index = subject_order.index(subject)
    trials = list(subjects[subject])
    train_trials = [trial for index, trial in enumerate(trials) if index != heldout]
    test_trial = trials[heldout]
    data_cfg = config["data"]
    analysis = config["analysis"]
    selected_indices, selected_names, _ = evaluator._select_active_hbo(
        train_trials,
        baseline_duration_s=float(data_cfg["baseline_duration_s"]),
        task_duration_s=float(data_cfg["task_duration_s"]),
        count=int(analysis["fnirs_active_hbo_channels"]),
    )
    expected_names = tuple(str(row["selected_channels"]).split("|"))
    if selected_names != expected_names:
        raise RuntimeError(
            f"selected-channel replay mismatch for {condition_id}/{subject}/{heldout}: "
            f"{selected_names} != {expected_names}"
        )
    train_targets = evaluator._targets(train_trials, selected_indices)
    model, fit = evaluator._fit_croce_model(
        [trial.eeg for trial in train_trials],
        train_targets,
        particles=int(analysis["croce2017"]["n_particles"]),
        resample_threshold=float(analysis["croce2017"]["resample_threshold"]),
        hrf_duration_s=float(analysis["croce2017"]["hrf_duration_s"]),
        seed=int(analysis["seed"]) + subject_index * 1000 + heldout,
    )
    eeg_pc1 = evaluator._apply_croce_eeg(test_trial.eeg, fit)
    truth = evaluator._targets([test_trial], selected_indices)[0]
    fold_seed = int(analysis["seed"]) + subject_index * 1000 + heldout
    np.random.seed(fold_seed)
    result = model.filter(
        eeg_pc1[:, None],
        ((truth - fit["fnirs_mean"]) / fit["fnirs_std"])[:, None],
        return_particles=False,
    )
    estimate = result.fnirs_reconstructed[:, 0] * fit["fnirs_std"] + fit["fnirs_mean"]
    driver = result.state_mean[:, 0]
    checks = {
        "truth_max_abs_error": float(np.max(np.abs(truth - saved["truth"]))),
        "estimate_max_abs_error": float(np.max(np.abs(estimate - saved["estimate"]))),
        "driver_max_abs_error": float(np.max(np.abs(driver - saved["driver"]))),
    }
    if max(checks.values()) > 1e-9:
        raise RuntimeError(f"formal-v3 replay mismatch for {condition_id}/{subject}/{heldout}: {checks}")
    return {
        "trial": test_trial,
        "selected_indices": selected_indices,
        "selected_names": selected_names,
        "selected_channels": np.asarray(test_trial.fnirs[:, selected_indices], dtype=np.float64),
        "eeg_pc1": np.asarray(eeg_pc1, dtype=np.float64),
        "fit": fit,
        "fold_seed": fold_seed,
        "replay_checks": checks,
    }


def croce_companion_states(time_s: np.ndarray, driver: np.ndarray) -> dict[str, np.ndarray]:
    """Integrate canonical Croce hemodynamics with the saved scalar driver as forcing."""

    time_s = np.asarray(time_s, dtype=np.float64)
    driver = np.asarray(driver, dtype=np.float64)
    if time_s.ndim != 1 or driver.shape != time_s.shape or np.any(np.diff(time_s) <= 0):
        raise ValueError("time_s and driver must be aligned increasing 1D arrays")

    epsilon = MODEL_PARAMS["epsilon"]
    kas = MODEL_PARAMS["kas"]
    kaf = MODEL_PARAMS["kaf"]
    tau0 = MODEL_PARAMS["tau0"]
    alpha = MODEL_PARAMS["alpha"]
    e0 = MODEL_PARAMS["e0"]

    def derivative(time: float, state: np.ndarray) -> np.ndarray:
        s, delta_f, delta_hbo, delta_hb = state
        r = float(np.interp(time, time_s, driver))
        flow = max(1.0 + float(delta_f), 1e-4)
        hbo = max(1.0 + float(delta_hbo), 1e-4)
        hb = max(1.0 + float(delta_hb), 1e-4)
        extraction = (1.0 - (1.0 - e0) ** (1.0 / flow)) / e0
        return np.asarray(
            [
                epsilon * r - kas * s - kaf * delta_f,
                s,
                (flow - hbo ** (1.0 / alpha)) / tau0,
                (flow * extraction - hb * hbo ** ((1.0 / alpha) - 1.0)) / tau0,
            ],
            dtype=np.float64,
        )

    solution = solve_ivp(
        derivative,
        (float(time_s[0]), float(time_s[-1])),
        np.zeros(4, dtype=np.float64),
        t_eval=time_s,
        max_step=0.02,
        rtol=1e-8,
        atol=1e-10,
    )
    if not solution.success or solution.y.shape != (4, len(time_s)):
        raise RuntimeError(f"Croce companion integration failed: {solution.message}")
    return {
        "vasoactive_signal_s": solution.y[0],
        "delta_f": solution.y[1],
        "normalized_blood_flow_f": 1.0 + solution.y[1],
        "delta_hbo": solution.y[2],
        "delta_hb": solution.y[3],
    }


def _plot(
    selected: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    figures_dir: Path,
) -> list[Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    colors = {
        "driver": "#000000",
        "eeg": "#0072B2",
        "vaso": "#009E73",
        "flow": "#CC79A7",
        "truth": "#222222",
        "estimate": "#D55E00",
        "channel": "#9E9E9E",
        "stimulus": "#E69F00",
    }
    fig, axes = plt.subplots(5, 2, figsize=(12.2, 13.0), sharex=True, constrained_layout=True)
    panel_labels = iter("ABCDEFGHIJ")
    for column, (metric, sample) in enumerate(zip(selected, samples)):
        time = sample["time_s"]
        title = (
            f"Rank {metric['sample_rank']}: {metric['condition_id']} / {metric['subject']} / "
            f"trial {metric['heldout_trial']}\n"
            f"variance ratio={metric['variance_ratio']:.3f}, R²={metric['r2']:.3f}, "
            f"PCC={metric['pcc']:.3f}"
        )
        axes[0, column].set_title(title, loc="left", pad=8)
        for row_index in range(5):
            axis = axes[row_index, column]
            axis.axvspan(0.0, 10.0, color=colors["stimulus"], alpha=0.08, linewidth=0)
            axis.axvline(0.0, color="#666666", linestyle="--", linewidth=0.8)
            axis.axhline(0.0 if row_index != 3 else 1.0, color="#B0B0B0", linewidth=0.6)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.65)
            axis.text(
                -0.095,
                1.04,
                next(panel_labels),
                transform=axis.transAxes,
                fontsize=11,
                fontweight="bold",
                va="bottom",
            )

        axes[0, column].plot(time, sample["driver"], color=colors["driver"], linewidth=1.35)
        axes[0, column].set_ylabel("Joint neural driver r (AU)")

        axes[1, column].plot(time, sample["eeg_pc1"], color=colors["eeg"], linewidth=1.05)
        axes[1, column].set_ylabel("EEG power PC1 (fold SD)")

        axes[2, column].plot(
            time, sample["vasoactive_signal_s"], color=colors["vaso"], linewidth=1.35
        )
        axes[2, column].set_ylabel("Vasoactive signal s (model AU)")

        axes[3, column].plot(
            time, sample["normalized_blood_flow_f"], color=colors["flow"], linewidth=1.35
        )
        axes[3, column].set_ylabel("Model-relative blood flow f")

        for channel_index, channel_name in enumerate(sample["selected_names"]):
            axes[4, column].plot(
                time,
                sample["selected_channels"][:, channel_index],
                color=colors["channel"],
                linewidth=0.7,
                alpha=0.55,
                label="selected HbO channels" if channel_index == 0 else None,
            )
        axes[4, column].plot(
            time, sample["truth"], color=colors["truth"], linewidth=1.8, label="observed HbO mean"
        )
        axes[4, column].plot(
            time,
            sample["estimate"],
            color=colors["estimate"],
            linewidth=1.6,
            linestyle="--",
            label="formal-v3 reconstructed HbO",
        )
        axes[4, column].set_ylabel("fNIRS HbO (canonical robust SD)")
        axes[4, column].set_xlabel("Event-relative time (s)")
        axes[4, column].legend(loc="best", frameon=False, ncol=1)
        for axis in axes[:, column]:
            axis.set_xlim(float(time[0]), float(time[-1]))

    fig.suptitle(
        "Croce-joint held-out examples with the closest fNIRS variance recovery\n"
        "s and f are deterministic Croce-dynamics companions, not formal-v3 posterior states",
        fontsize=12,
        fontweight="bold",
    )
    figures_dir.mkdir(parents=True, exist_ok=True)
    stem = figures_dir / "best_variance_full_trajectory_comparison"
    outputs = [stem.with_suffix(suffix) for suffix in (".svg", ".pdf", ".png")]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    fig.savefig(outputs[2], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outputs


def _summary_markdown(selected: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Best-variance Croce full-trajectory comparison",
        "",
        "Two `croce_joint` leave-one-trial folds were ranked by the multiplicative distance "
        "`abs(log(variance_ratio))` from ideal variance ratio 1. A positive PCC was required "
        "to exclude direction-inverted examples; R² was not used for selection.",
        "",
        "| Rank | Condition | Subject | Trial | Event index | Selected HbO channels | Variance ratio | R² | PCC |",
        "| ---: | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in selected:
        lines.append(
            f"| {row['sample_rank']} | {row['condition_id']} | {row['subject']} | "
            f"{row['heldout_trial']} | {row['event_index']} | {row['selected_channels']} | "
            f"{row['variance_ratio']:.4f} | {row['r2']:.4f} | {row['pcc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Signal provenance",
            "",
            "- Neural driver, observed fNIRS mean, and reconstructed fNIRS are exact trajectories "
            "from the formal-v3 Croce-joint folds.",
            "- EEG is the exact held-out fold-standardized broadband-power PC1 observation supplied "
            "to the scalar SMC filter.",
            "- Vasoactive signal `s` and relative blood flow `f = 1 + delta_f` are deterministic "
            "companion trajectories obtained by integrating the canonical Croce dynamics with the "
            "saved joint driver. They are not latent posterior states from formal-v3, whose Croce "
            "implementation has only one scalar state plus an HRF observation model.",
            "- The formal-v3 driver is standardized rather than physically calibrated. Consequently, "
            "the magnitudes of `s` and `f` are model-coordinate responses and must not be read as "
            "absolute vasodilation or measured cerebral blood flow.",
            "- Thin gray fNIRS curves are the three training-fold-selected HbO channels; the black "
            "curve is their mean and the orange dashed curve is the model reconstruction.",
            "",
            "## Interpretation",
            "",
            "These are the closest variance-ratio examples, but both still have negative held-out "
            "R². The figure therefore supports only the narrow statement that output variance scale "
            "was relatively well matched; it does not show successful pointwise fNIRS recovery.",
            "",
        ]
    )
    return "\n".join(lines)


def run(run_dir: Path) -> list[Path]:
    run_dir = run_dir.resolve()
    config_path = run_dir / "config.yaml"
    fold_metrics_path = run_dir / "fold_metrics.csv"
    trajectories_path = run_dir / "trajectories.csv"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selected = select_best_variance_folds(fold_metrics_path)
    saved = _load_saved_trajectories(trajectories_path, selected)
    grouped, _ = evaluator._load_trials(config)

    samples: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    replay_checks: list[dict[str, Any]] = []
    for metric in selected:
        key = (metric["condition_id"], metric["subject"], metric["heldout_trial"])
        stored = saved[key]
        replay = _replay_fold(grouped, config, metric, stored)
        companion = croce_companion_states(stored["time_s"], stored["driver"])
        metric["event_index"] = int(replay["trial"].event_index)
        metric["record_id"] = str(replay["trial"].record_id)
        metric["fold_seed"] = int(replay["fold_seed"])
        selection_rows.append(metric)
        replay_checks.append(
            {
                "sample_rank": metric["sample_rank"],
                "condition_id": metric["condition_id"],
                "subject": metric["subject"],
                "heldout_trial": metric["heldout_trial"],
                **replay["replay_checks"],
            }
        )
        sample = {
            "time_s": stored["time_s"],
            "truth": stored["truth"],
            "estimate": stored["estimate"],
            "driver": stored["driver"],
            "eeg_pc1": replay["eeg_pc1"],
            "selected_channels": replay["selected_channels"],
            "selected_names": replay["selected_names"],
            **companion,
        }
        samples.append(sample)
        for index, time in enumerate(stored["time_s"]):
            trajectory_rows.append(
                {
                    "sample_rank": metric["sample_rank"],
                    "condition_id": metric["condition_id"],
                    "subject": metric["subject"],
                    "heldout_trial": metric["heldout_trial"],
                    "event_index": metric["event_index"],
                    "time_s": float(time),
                    "neural_driver_joint": float(stored["driver"][index]),
                    "eeg_power_pc1_observation": float(replay["eeg_pc1"][index]),
                    "vasoactive_signal_s_companion": float(companion["vasoactive_signal_s"][index]),
                    "normalized_blood_flow_f_companion": float(companion["normalized_blood_flow_f"][index]),
                    "delta_hbo_companion": float(companion["delta_hbo"][index]),
                    "delta_hb_companion": float(companion["delta_hb"][index]),
                    "observed_hbo_mean": float(stored["truth"][index]),
                    "reconstructed_hbo_formal_v3": float(stored["estimate"][index]),
                }
            )
            for channel_index, channel_name in enumerate(replay["selected_names"]):
                channel_rows.append(
                    {
                        "sample_rank": metric["sample_rank"],
                        "condition_id": metric["condition_id"],
                        "subject": metric["subject"],
                        "heldout_trial": metric["heldout_trial"],
                        "event_index": metric["event_index"],
                        "time_s": float(time),
                        "channel_name": channel_name,
                        "observed_hbo": float(replay["selected_channels"][index, channel_index]),
                    }
                )

    selection_path = run_dir / "best_variance_samples.csv"
    trajectories_output = run_dir / "best_variance_full_trajectories.csv"
    channels_output = run_dir / "best_variance_channel_trajectories.csv"
    summary_path = run_dir / "best_variance_full_trajectory_summary.md"
    manifest_path = run_dir / "best_variance_full_trajectory_manifest.json"
    _write_csv(selection_path, selection_rows)
    _write_csv(trajectories_output, trajectory_rows)
    _write_csv(channels_output, channel_rows)
    figures = _plot(selected, samples, run_dir / "figures")
    summary_path.write_text(_summary_markdown(selected), encoding="utf-8")

    sources = [config_path, fold_metrics_path, trajectories_path, Path(evaluator.__file__), Path(__file__)]
    outputs = [selection_path, trajectories_output, channels_output, summary_path, *figures]
    manifest = {
        "schema": "best_variance_full_trajectory_comparison_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parent_run": str(run_dir.relative_to(REPO_ROOT)),
        "selection_rule": {
            "validation": "leave_one_trial",
            "model": "croce_joint",
            "hrf_mode": "canonical",
            "eligibility": "finite positive variance_ratio and PCC > 0",
            "ranking": "ascending abs(log(variance_ratio))",
            "count": 2,
        },
        "state_provenance": {
            "neural_driver_joint": "exact saved formal-v3 trajectory",
            "eeg_power_pc1_observation": "exact fold replay of formal-v3 model input",
            "observed_and_reconstructed_hbo": "exact saved formal-v3 trajectories",
            "vasoactive_signal_s_and_normalized_blood_flow_f": (
                "deterministic canonical Croce-dynamics companion replay; not formal-v3 posterior states"
            ),
            "croce_model_params": MODEL_PARAMS,
            "initial_companion_state": [0.0, 0.0, 0.0, 0.0],
        },
        "selected_samples": selection_rows,
        "replay_checks": replay_checks,
        "input_hashes": [{"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path)} for path in sources],
        "output_hashes": [{"path": str(path.relative_to(run_dir)), "sha256": _sha256(path)} for path in outputs],
        "artifacts": [str(path.relative_to(run_dir)) for path in outputs],
        "claim_boundary": [
            "variance-ratio sample selection only",
            "negative R2 examples are not successful pointwise fNIRS recovery",
            "s and f are deterministic companion states, not posterior estimates from formal-v3",
            "s and f magnitudes are not calibrated physical vasodilation or cerebral blood flow",
        ],
    }
    _write_json(manifest_path, manifest)

    parent_manifest_path = run_dir / "manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    artifact_names = [str(path.relative_to(run_dir)) for path in [*outputs, manifest_path]]
    parent_manifest["artifacts"] = list(dict.fromkeys([*parent_manifest.get("artifacts", []), *artifact_names]))
    parent_manifest.setdefault("posthoc_analyses", {})["best_variance_full_trajectories"] = {
        "manifest": str(manifest_path.relative_to(run_dir)),
        "selection_rule": "min abs(log(variance_ratio)) among positive-PCC Croce-joint held-out folds",
        "selected_samples": [
            {
                "condition_id": row["condition_id"],
                "subject": row["subject"],
                "heldout_trial": row["heldout_trial"],
                "variance_ratio": row["variance_ratio"],
                "r2": row["r2"],
                "pcc": row["pcc"],
            }
            for row in selected
        ],
    }
    parent_manifest["manifest_updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(parent_manifest_path, parent_manifest)
    return [*outputs, manifest_path, parent_manifest_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    for output_path in run(parse_args().run_dir):
        print(output_path)
