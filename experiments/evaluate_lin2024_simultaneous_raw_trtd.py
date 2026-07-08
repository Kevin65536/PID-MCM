#!/usr/bin/env python3
"""Run a Lin-2024-style TRTD + subject-specific HRF diagnostic on Simultaneous EEG&NIRS.

This script is the raw-record counterpart of
``evaluate_lin2024_raw_session_trtd.py`` for the Simultaneous EEG&NIRS
cognitive dataset. It reads one subject/task directly from the original
``cnt_*.mat`` and ``mrk_*.mat`` files, not from cropped cache windows.

Default run: subject VP001, word-generation task, target class ``WG``.
"""

from __future__ import annotations

import argparse
import subprocess
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from evaluate_lin2024_raw_session_trtd import (
    TrialSet,
    _aggregate,
    _canonical_params,
    _convolve_same,
    _double_gamma,
    _eeg_channel_mask,
    _eeg_tensor,
    _epoch,
    _evaluate_baselines,
    _evaluate_loso,
    _metrics,
    _plot_outputs,
    _sha256,
    _sos_bandpass,
    _summary_markdown,
    _write_csv,
    _write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "lin2024_simultaneous_raw_trtd_v1"


@dataclass(frozen=True)
class SimultaneousRaw:
    subject: int
    task: str
    eeg: np.ndarray
    eeg_fs: float
    eeg_labels: list[str]
    eeg_marker_time_ms: np.ndarray
    eeg_marker_y: np.ndarray
    eeg_marker_desc: np.ndarray
    eeg_class_names: list[str]
    oxy: np.ndarray
    deoxy: np.ndarray
    fnirs_fs: float
    fnirs_labels: list[str]
    fnirs_marker_time_ms: np.ndarray
    fnirs_marker_y: np.ndarray
    fnirs_marker_desc: np.ndarray
    fnirs_class_names: list[str]
    fnirs_unit: str


@dataclass(frozen=True)
class PreparedSimultaneous:
    trials: TrialSet
    deoxy_average: np.ndarray
    eeg_channel_labels: list[str]
    target_class: str
    target_index: int
    alignment: dict[str, Any]


def _mat_struct(path: Path) -> Any:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    key = next(key for key in mat if not key.startswith("__"))
    return mat[key]


def _labels(raw: Any) -> list[str]:
    return [str(value) for value in np.asarray(raw).ravel().tolist()]


def _marker_y(raw: Any, n_events: int) -> np.ndarray:
    y = np.asarray(raw, dtype=np.float64)
    if y.ndim == 1:
        if y.shape[0] == n_events:
            labels = list(dict.fromkeys(int(value) for value in y.tolist()))
            out = np.zeros((len(labels), n_events), dtype=np.float64)
            mapping = {value: index for index, value in enumerate(labels)}
            for column, value in enumerate(y.tolist()):
                out[mapping[int(value)], column] = 1.0
            return out
        return np.ones((1, n_events), dtype=np.float64)
    return y


def _load_raw(data_root: Path, subject: int, task: str) -> SimultaneousRaw:
    eeg_dir = data_root / f"VP{subject:03d}-EEG"
    fnirs_dir = data_root / f"VP{subject:03d}-NIRS"
    eeg_cnt = _mat_struct(eeg_dir / f"cnt_{task}.mat")
    eeg_mrk = _mat_struct(eeg_dir / f"mrk_{task}.mat")
    fnirs_cnt = _mat_struct(fnirs_dir / f"cnt_{task}.mat")
    fnirs_mrk = _mat_struct(fnirs_dir / f"mrk_{task}.mat")
    oxy = fnirs_cnt.oxy
    deoxy = fnirs_cnt.deoxy
    eeg_times = np.asarray(eeg_mrk.time, dtype=np.float64)
    fnirs_times = np.asarray(fnirs_mrk.time, dtype=np.float64)
    return SimultaneousRaw(
        subject=subject,
        task=task,
        eeg=np.asarray(eeg_cnt.x, dtype=np.float64),
        eeg_fs=float(eeg_cnt.fs),
        eeg_labels=_labels(eeg_cnt.clab),
        eeg_marker_time_ms=eeg_times,
        eeg_marker_y=_marker_y(eeg_mrk.y, len(eeg_times)),
        eeg_marker_desc=np.asarray(eeg_mrk.event.desc, dtype=int),
        eeg_class_names=_labels(eeg_mrk.className),
        oxy=np.asarray(oxy.x, dtype=np.float64),
        deoxy=np.asarray(deoxy.x, dtype=np.float64),
        fnirs_fs=float(oxy.fs),
        fnirs_labels=_labels(oxy.clab),
        fnirs_marker_time_ms=fnirs_times,
        fnirs_marker_y=_marker_y(fnirs_mrk.y, len(fnirs_times)),
        fnirs_marker_desc=np.asarray(fnirs_mrk.event.desc, dtype=int),
        fnirs_class_names=_labels(fnirs_mrk.className),
        fnirs_unit=str(getattr(oxy, "yUnit", "")),
    )


def _alignment_summary(raw: SimultaneousRaw) -> dict[str, Any]:
    common = min(len(raw.eeg_marker_time_ms), len(raw.fnirs_marker_time_ms))
    residual = raw.fnirs_marker_time_ms[:common] - raw.eeg_marker_time_ms[:common]
    labels_match = bool(
        np.array_equal(
            np.argmax(raw.eeg_marker_y[:, :common], axis=0),
            np.argmax(raw.fnirs_marker_y[:, :common], axis=0),
        )
    )
    jumps = np.where(np.abs(np.diff(residual)) > 20_000.0)[0] + 1
    starts = [0, *[int(value) for value in jumps]]
    ends = [*[int(value) for value in jumps], common]
    blocks = []
    for start, end in zip(starts, ends):
        block = residual[start:end]
        blocks.append(
            {
                "start_index": int(start),
                "end_index": int(end - 1),
                "count": int(end - start),
                "offset_mean_ms": float(np.mean(block)),
                "offset_std_ms": float(np.std(block)),
            }
        )
    return {
        "num_eeg_events": int(len(raw.eeg_marker_time_ms)),
        "num_fnirs_events": int(len(raw.fnirs_marker_time_ms)),
        "num_common_events": int(common),
        "label_sequence_match": labels_match,
        "residual_mean_ms": float(np.mean(residual)) if common else None,
        "residual_std_ms": float(np.std(residual)) if common else None,
        "offset_blocks": blocks,
    }


def _class_index(class_names: Sequence[str], target_class: str) -> int:
    lowered = [name.lower() for name in class_names]
    target = target_class.lower()
    if target not in lowered:
        raise ValueError(f"target class {target_class!r} not found in {class_names}")
    return lowered.index(target)


def _stimulus_from_class(time_ms: np.ndarray, y: np.ndarray, fs: float, n: int, target_index: int, duration_s: float) -> np.ndarray:
    stimulus = np.zeros(n, dtype=np.float64)
    duration = int(round(duration_s * fs))
    for onset_ms, is_target in zip(time_ms, y[target_index]):
        if float(is_target) <= 0.5:
            continue
        start = int(round(float(onset_ms) * fs / 1000.0))
        end = min(n, start + duration)
        if 0 <= start < n:
            stimulus[start:end] = 1.0
    return stimulus


def _prepare_trials(
    raw: SimultaneousRaw,
    target_class: str,
    epoch_start_s: float,
    epoch_end_s: float,
    task_duration_s: float,
) -> PreparedSimultaneous:
    if raw.eeg_class_names != raw.fnirs_class_names:
        raise ValueError(f"EEG/NIRS class names differ: {raw.eeg_class_names} vs {raw.fnirs_class_names}")
    target_idx = _class_index(raw.eeg_class_names, target_class)

    eeg_mask = _eeg_channel_mask(raw.eeg_labels)
    eeg_labels = [label for label, keep in zip(raw.eeg_labels, eeg_mask) if keep]
    eeg = _sos_bandpass(raw.eeg[:, eeg_mask], raw.eeg_fs, 1.0, 40.0)
    oxy = _sos_bandpass(raw.oxy, raw.fnirs_fs, 0.01, 0.2, order=3)
    deoxy = _sos_bandpass(raw.deoxy, raw.fnirs_fs, 0.01, 0.2, order=3)

    continuous_stimulus = _stimulus_from_class(
        raw.fnirs_marker_time_ms,
        raw.fnirs_marker_y,
        raw.fnirs_fs,
        len(oxy),
        target_idx,
        task_duration_s,
    )
    design_signal = _convolve_same(continuous_stimulus, _double_gamma(_canonical_params(), raw.fnirs_fs), raw.fnirs_fs)
    design = np.column_stack((np.ones(len(design_signal)), design_signal))
    beta = np.linalg.lstsq(design, oxy, rcond=None)[0]
    prediction = design @ beta
    residual = oxy - prediction
    dof = max(len(oxy) - design.shape[1], 1)
    sigma2 = np.sum(residual**2, axis=0) / dof
    cov_design = np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.maximum(sigma2 * cov_design[1, 1], 1e-12))
    t_values = beta[1] / se
    selected = np.argsort(t_values)[-3:][::-1]

    eeg_epochs = []
    oxy_epochs = []
    deoxy_epochs = []
    stimulus_epochs = []
    tensors = []
    trial_indices = []
    trial_onsets = []
    common = min(len(raw.eeg_marker_time_ms), len(raw.fnirs_marker_time_ms))
    for trial_idx in range(common):
        if raw.eeg_marker_y[target_idx, trial_idx] <= 0.5 or raw.fnirs_marker_y[target_idx, trial_idx] <= 0.5:
            continue
        eeg_epoch = _epoch(eeg, raw.eeg_fs, raw.eeg_marker_time_ms[trial_idx], epoch_start_s, epoch_end_s)
        oxy_epoch = _epoch(oxy, raw.fnirs_fs, raw.fnirs_marker_time_ms[trial_idx], epoch_start_s, epoch_end_s)
        deoxy_epoch = _epoch(deoxy, raw.fnirs_fs, raw.fnirs_marker_time_ms[trial_idx], epoch_start_s, epoch_end_s)
        if eeg_epoch is None or oxy_epoch is None or deoxy_epoch is None:
            continue
        baseline_n = int(round(abs(epoch_start_s) * raw.fnirs_fs))
        oxy_epoch = oxy_epoch - oxy_epoch[:baseline_n].mean(axis=0, keepdims=True)
        deoxy_epoch = deoxy_epoch - deoxy_epoch[:baseline_n].mean(axis=0, keepdims=True)
        stimulus = np.zeros(len(oxy_epoch), dtype=np.float64)
        onset_index = int(round(abs(epoch_start_s) * raw.fnirs_fs))
        stimulus[onset_index : onset_index + int(round(task_duration_s * raw.fnirs_fs))] = 1.0
        eeg_epochs.append(eeg_epoch)
        oxy_epochs.append(oxy_epoch)
        deoxy_epochs.append(deoxy_epoch)
        stimulus_epochs.append(stimulus)
        tensors.append(_eeg_tensor(eeg_epoch, raw.eeg_fs, len(oxy_epoch)))
        trial_indices.append(trial_idx)
        trial_onsets.append(float(raw.fnirs_marker_time_ms[trial_idx]) / 1000.0)

    trials = TrialSet(
        eeg_epochs=np.stack(eeg_epochs),
        eeg_tensors=np.stack(tensors),
        hbo_epochs=np.stack(oxy_epochs),
        hbo_average=np.stack(oxy_epochs)[:, :, selected].mean(axis=2),
        stimulus_epochs=np.stack(stimulus_epochs),
        selected_channels=[int(value) for value in selected],
        selected_channel_labels=[raw.fnirs_labels[int(value)] for value in selected],
        active_channel_t=t_values,
        trial_indices=trial_indices,
        trial_onsets_s=trial_onsets,
    )
    return PreparedSimultaneous(
        trials=trials,
        deoxy_average=np.stack(deoxy_epochs)[:, :, selected].mean(axis=2),
        eeg_channel_labels=eeg_labels,
        target_class=target_class,
        target_index=target_idx,
        alignment=_alignment_summary(raw),
    )


def _plot_extra_outputs(
    prepared: PreparedSimultaneous,
    fit_artifacts: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    run_dir: Path,
    fs: float,
    epoch_start_s: float,
) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    trials = prepared.trials
    predictions = fit_artifacts["predictions"]
    if predictions:
        pred = predictions[0]
        heldout = int(pred["heldout"])
        t_fnirs = np.arange(len(pred["truth"])) / fs + epoch_start_s
        t_eeg = np.arange(trials.eeg_epochs[heldout].shape[0]) / 200.0 + epoch_start_s
        eeg_epoch = trials.eeg_epochs[heldout]
        channel_variance = np.var(eeg_epoch, axis=0)
        eeg_indices = np.argsort(channel_variance)[-4:][::-1]
        fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=False)
        for offset, idx in enumerate(eeg_indices):
            trace = eeg_epoch[:, idx]
            trace = (trace - np.mean(trace)) / max(float(np.std(trace)), 1e-12)
            axes[0].plot(t_eeg, trace + offset * 4.0, linewidth=0.8, label=prepared.eeg_channel_labels[int(idx)])
        axes[0].set_ylabel("EEG z + offset")
        axes[0].legend(loc="upper right", ncol=4, fontsize=8)
        axes[0].grid(alpha=0.2)
        axes[1].plot(t_fnirs, pred["component"], color="#0072B2", label="TRTD EEG component")
        axes[1].plot(t_fnirs, pred["driver"], color="#E69F00", label="HRF driver")
        axes[1].set_ylabel("driver (a.u.)")
        axes[1].legend(loc="upper right")
        axes[1].grid(alpha=0.2)
        axes[2].plot(t_fnirs, pred["truth"], color="#009E73", label="observed oxy/HbO")
        axes[2].plot(t_fnirs, pred["prediction"], color="#D55E00", linestyle="--", label="recovered oxy/HbO")
        axes[2].plot(t_fnirs, prepared.deoxy_average[heldout], color="#CC79A7", alpha=0.8, label="deoxy/HbR")
        axes[2].set_ylabel("mmol/L baseline-corrected")
        axes[2].legend(loc="upper right")
        axes[2].grid(alpha=0.2)
        axes[3].plot(t_fnirs, pred["truth"] - pred["prediction"], color="#111827", label="oxy residual")
        axes[3].axhline(0, color="#6b7280", linewidth=0.8)
        axes[3].set_xlabel("seconds from WG onset")
        axes[3].set_ylabel("mmol/L")
        axes[3].legend(loc="upper right")
        axes[3].grid(alpha=0.2)
        fig.suptitle(f"Simultaneous VP001/WG held-out trial {heldout}: EEG, fNIRS, recovered fNIRS")
        fig.tight_layout()
        for suffix, dpi in (("svg", None), ("png", 300)):
            path = run_dir / "figures" / f"simultaneous_waveform_overlay.{suffix}"
            fig.savefig(path, dpi=dpi)
            artifacts.append({"path": str(path.relative_to(run_dir)), "sha256": _sha256(path)})
        plt.close(fig)

    optimized = [row for row in rows if row.get("model") == "TRTD" and row.get("validation") == "leave_one_trial" and row.get("hrf_mode") == "optimized"]
    if optimized:
        x = np.arange(len(optimized))
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
        axes[0, 0].plot(x, [float(row["hrf_ttp"]) for row in optimized], marker="o", color="#0072B2")
        axes[0, 0].set_ylabel("time to peak (s)")
        axes[0, 1].plot(x, [float(row["hrf_ttu"]) for row in optimized], marker="o", color="#D55E00")
        axes[0, 1].set_ylabel("time to undershoot (s)")
        axes[1, 0].plot(x, [float(row["hrf_c"]) for row in optimized], marker="o", color="#009E73")
        axes[1, 0].set_ylabel("undershoot ratio c")
        axes[1, 1].plot(x, [float(row["amplitude_ratio"]) for row in optimized], marker="o", color="#CC79A7")
        axes[1, 1].set_ylabel("amplitude ratio")
        for axis in axes.ravel():
            axis.grid(alpha=0.25)
            axis.set_xlabel("held-out WG fold")
        fig.suptitle("Optimized HRF and prediction-scale parameters by fold")
        fig.tight_layout()
        for suffix, dpi in (("svg", None), ("png", 300)):
            path = run_dir / "figures" / f"hrf_parameter_folds.{suffix}"
            fig.savefig(path, dpi=dpi)
            artifacts.append({"path": str(path.relative_to(run_dir)), "sha256": _sha256(path)})
        plt.close(fig)

    order = np.argsort(prepared.trials.active_channel_t)[::-1]
    top = order[: min(18, len(order))]
    colors = ["#D55E00" if int(idx) in set(prepared.trials.selected_channels) else "#0072B2" for idx in top]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(np.arange(len(top)), prepared.trials.active_channel_t[top], color=colors)
    ax.set_xticks(np.arange(len(top)), [prepared.trials.selected_channel_labels[prepared.trials.selected_channels.index(int(idx))] if int(idx) in set(prepared.trials.selected_channels) else str(idx) for idx in top], rotation=45, ha="right")
    ax.set_ylabel("canonical GLM t value")
    ax.set_title("Top fNIRS active-channel scores; selected channels in orange")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for suffix, dpi in (("svg", None), ("png", 300)):
        path = run_dir / "figures" / f"active_channel_scores.{suffix}"
        fig.savefig(path, dpi=dpi)
        artifacts.append({"path": str(path.relative_to(run_dir)), "sha256": _sha256(path)})
    plt.close(fig)
    return artifacts


def _sim_summary_markdown(summary: Sequence[Mapping[str, Any]], prepared: PreparedSimultaneous) -> str:
    text = _summary_markdown(summary, prepared.trials.selected_channel_labels)
    return text.replace(
        "# Lin 2024 raw-session TRTD diagnostic",
        "# Lin 2024 Simultaneous EEG&NIRS raw TRTD diagnostic",
    ).replace(
        "This run reads one continuous raw session and does not use Croce cache windows.",
        "This run reads one continuous Simultaneous EEG&NIRS subject/task record and does not use cache windows.",
    )


def run(args: argparse.Namespace) -> Path:
    data_root = (REPO_ROOT / args.data_root).resolve()
    raw = _load_raw(data_root, args.subject, args.task)
    run_dir = Path(args.output_dir).resolve() if args.output_dir else (
        REPO_ROOT / "experiments" / "runs" / "physiology_semantic_tokenizer" / "e0_teacher_validity" /
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_lin2024_simultaneous_raw_trtd_vp{args.subject:03d}_{args.task}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir()
    (run_dir / "figure_data").mkdir()

    prepared = _prepare_trials(raw, args.target_class, args.epoch_start_s, args.epoch_end_s, args.task_duration_s)
    fold_rows, fit_artifacts = _evaluate_loso(prepared.trials, args.rank, args.cp_iterations, args.seed, raw.fnirs_fs)
    baseline_rows = _evaluate_baselines(prepared.trials)
    all_rows = fold_rows + baseline_rows
    summary_rows = _aggregate(all_rows)
    figures = _plot_outputs(prepared.trials, fit_artifacts, all_rows, run_dir, raw.fnirs_fs)
    figures.extend(_plot_extra_outputs(prepared, fit_artifacts, all_rows, run_dir, raw.fnirs_fs, args.epoch_start_s))

    _write_csv(run_dir / "fold_metrics.csv", all_rows)
    _write_csv(run_dir / "metrics.csv", summary_rows)
    _write_csv(run_dir / "active_channel_t_values.csv", [
        {
            "channel_index": idx,
            "channel_label": label,
            "t_value": float(prepared.trials.active_channel_t[idx]),
            "selected": int(idx in set(prepared.trials.selected_channels)),
        }
        for idx, label in enumerate(raw.fnirs_labels)
    ])
    _write_json(run_dir / "summary.json", {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_diagnostic",
        "subject": f"VP{args.subject:03d}",
        "task": args.task,
        "target_class": args.target_class,
        "input_files": {
            "eeg_cnt": str((data_root / f"VP{args.subject:03d}-EEG" / f"cnt_{args.task}.mat").relative_to(REPO_ROOT)),
            "eeg_mrk": str((data_root / f"VP{args.subject:03d}-EEG" / f"mrk_{args.task}.mat").relative_to(REPO_ROOT)),
            "fnirs_cnt": str((data_root / f"VP{args.subject:03d}-NIRS" / f"cnt_{args.task}.mat").relative_to(REPO_ROOT)),
            "fnirs_mrk": str((data_root / f"VP{args.subject:03d}-NIRS" / f"mrk_{args.task}.mat").relative_to(REPO_ROOT)),
        },
        "paper_alignment": {
            "followed": [
                "continuous raw cnt/mrk inputs",
                "20s epochs from -5 to 15s",
                "EEG 1-40Hz filtering",
                "0.5Hz time-frequency representation",
                "shared spatial/frequency tensor factors with trial-specific temporal factors",
                "TRCA temporal component filter",
                "fNIRS oxy/deoxy 0.01-0.2Hz filtering",
                "GLM active-channel selection",
                "subject/task-specific double-gamma HRF",
                "leave-one-trial validation",
            ],
            "dataset_limited": [
                "uses Simultaneous EEG&NIRS word generation rather than Lin finger tapping",
                "uses provided oxy/deoxy concentration in mmol/L rather than HOMER reprocessing",
                "no short-distance fNIRS regressors are available",
                "no BSS-CCA muscle-artifact removal was applied",
            ],
        },
        "alignment": prepared.alignment,
        "fnirs_unit": raw.fnirs_unit,
        "selected_channels": prepared.trials.selected_channel_labels,
        "trial_indices": prepared.trials.trial_indices,
        "trial_onsets_s": prepared.trials.trial_onsets_s,
        "summary_rows": summary_rows,
        "figures": figures,
        "interpretation": {
            "status": "diagnostic_only",
            "upper_bound_scope": "single subject/task, same-record leave-one-trial and in-sample fit",
        },
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip(),
    })
    _write_json(run_dir / "environment.json", {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "git_status_porcelain": subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.splitlines(),
    })
    (run_dir / "summary.md").write_text(_sim_summary_markdown(summary_rows, prepared), encoding="utf-8")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/Simultaneous EEG&NIRS")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--task", default="wg")
    parser.add_argument("--target-class", default="WG")
    parser.add_argument("--epoch-start-s", type=float, default=-5.0)
    parser.add_argument("--epoch-end-s", type=float, default=15.0)
    parser.add_argument("--task-duration-s", type=float, default=10.0)
    parser.add_argument("--rank", type=int, default=6)
    parser.add_argument("--cp-iterations", type=int, default=35)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--output-dir")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
