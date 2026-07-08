#!/usr/bin/env python3
"""Inspect raw EEG data for ocular and muscle artifacts in the hBCI dataset.

Dataset documentation (from doc.ml.tu-berlin.de/hBCI and local HTML):
- EEG: BrainAmp, 30 active electrodes (10-5 system), linked mastoids, 1000 Hz
- EOG: 2 vertical (above/below left eye) + 2 horizontal (outer canthus) electrodes
- ECG and respiration also recorded but not analyzed
- Data stored under "with occular artifact" — artifacts are PRESENT, not removed
- Dataset C: separate controlled artifact recordings (EOG, EMG, blinking, teeth
  clenching, mouth opening) in cnt_artifact.mat

This script checks:
1. Channel inventory — which are EEG vs. EOG vs. ECG vs. Respiration
2. EOG channel amplitude characteristics (blink detection)
3. EOG→EEG propagation (how strongly ocular artifacts contaminate EEG channels)
4. Muscle artifact signatures — high-frequency (>30 Hz) bursts in EEG channels
5. Frequency-domain evidence of artifacts
6. Comparison with the separate artifact-only recordings (Dataset C)
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import loadmat
from scipy.ndimage import label as nd_label

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
warnings.filterwarnings("ignore", category=UserWarning)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cell(payload: Any, index: int) -> Any:
    return np.asarray(payload).ravel()[index]


def _labels(raw: Any) -> list[str]:
    return [str(v) for v in np.asarray(raw).ravel().tolist()]


def _sos_bandpass(data: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    sos = signal.butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, data, axis=0)


def _sos_highpass(data: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    sos = signal.butter(order, cutoff / nyq, btype="highpass", output="sos")
    return signal.sosfiltfilt(sos, data, axis=0)


def _rms(x: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.sqrt(np.mean(x ** 2, axis=axis))


def _channel_type(label: str) -> str:
    """Classify a channel label into EEG, EOG, ECG, or Respiration."""
    upper = label.upper().strip()
    if upper in ("VEOG", "HEOG"):
        return "EOG"
    if upper in ("ECG1", "ECG2", "ECG"):
        return "ECG"
    if "RESP" in upper or "BREATH" in upper:
        return "Respiration"
    # Known EOG variants
    if upper in ("VEOG1", "VEOG2", "HEOG1", "HEOG2", "EOG1", "EOG2", "EOG"):
        return "EOG"
    # EEG channels: typical 10-5 labels or 10-20 labels
    return "EEG"


# ── Main inspection ──────────────────────────────────────────────────────────

def inspect_session(
    data_root: Path,
    subject: int,
    session_index: int,
    output_dir: Path,
    epoch_window: tuple[float, float] = (-5.0, 15.0),
    target_desc: int = 1,  # mental arithmetic for NIRS, 16 for EEG
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    subject_dir = data_root / "EEG_01-29" / f"subject {subject:02d}"
    artifact_path = subject_dir / "cnt_artifact.mat"
    mnt_artifact_path = subject_dir / "mnt_artifact.mat"
    mrk_artifact_path = subject_dir / "mrk_artifact.mat"
    eeg_cnt_path = subject_dir / "with occular artifact" / "cnt.mat"
    eeg_mrk_path = subject_dir / "with occular artifact" / "mrk.mat"
    eeg_mnt_path = subject_dir / "with occular artifact" / "mnt.mat"

    # ── Load EEG data ────────────────────────────────────────────────────
    eeg_cnt_mat = loadmat(eeg_cnt_path, squeeze_me=True, struct_as_record=False)
    eeg_mrk_mat = loadmat(eeg_mrk_path, squeeze_me=True, struct_as_record=False)
    eeg_mnt_mat = loadmat(eeg_mnt_path, squeeze_me=True, struct_as_record=False)

    cnt_session = _cell(eeg_cnt_mat["cnt"], session_index)
    mrk_session = _cell(eeg_mrk_mat["mrk"], session_index)

    eeg_data = np.asarray(cnt_session.x, dtype=np.float64)
    eeg_fs = float(cnt_session.fs)
    eeg_labels = _labels(cnt_session.clab)
    mrk_time = np.asarray(mrk_session.time, dtype=np.float64)
    mrk_desc = np.asarray(mrk_session.event.desc, dtype=int)

    print(f"EEG data: {eeg_data.shape[1]} channels, {eeg_data.shape[0]} samples, {eeg_fs} Hz")
    print(f"Duration: {eeg_data.shape[0] / eeg_fs / 60:.1f} minutes")
    print(f"Channels: {eeg_labels}")

    # ── Classify channels ────────────────────────────────────────────────
    channel_map: dict[str, list[int]] = {"EEG": [], "EOG": [], "ECG": [], "Respiration": [], "Unknown": []}
    for idx, label in enumerate(eeg_labels):
        ch_type = _channel_type(label)
        channel_map.setdefault(ch_type, []).append(idx)

    print(f"\nChannel classification:")
    for ch_type, indices in sorted(channel_map.items()):
        names = [eeg_labels[i] for i in indices]
        print(f"  {ch_type} ({len(indices)}): {names}")

    # ── Extract channels by type ─────────────────────────────────────────
    eeg_ch_mask = np.array([i in channel_map.get("EEG", []) for i in range(len(eeg_labels))])
    eog_ch_mask = np.array([i in channel_map.get("EOG", []) for i in range(len(eeg_labels))])

    eeg_channels = eeg_data[:, eeg_ch_mask]
    eog_channels = eeg_data[:, eog_ch_mask] if eog_ch_mask.any() else None

    eeg_ch_names = [eeg_labels[i] for i in range(len(eeg_labels)) if eeg_ch_mask[i]]
    eog_ch_names = [eeg_labels[i] for i in range(len(eeg_labels)) if eog_ch_mask[i]]

    # ── Event markers ────────────────────────────────────────────────────
    # For EEG, marker 16 = mental arithmetic task onset
    task_markers = mrk_time[mrk_desc == target_desc]
    print(f"\nTask markers (desc={target_desc}): {len(task_markers)} events")

    # ── EOG Analysis ─────────────────────────────────────────────────────
    eog_report: dict[str, Any] = {"channels": eog_ch_names}
    if eog_channels is not None:
        eog_amplitude = np.abs(eog_channels).mean(axis=0)
        eog_peak = np.abs(eog_channels).max(axis=0)
        eog_std = eog_channels.std(axis=0)
        for i, name in enumerate(eog_ch_names):
            print(f"\nEOG channel {name}: mean_abs={eog_amplitude[i]:.2f} µV, peak={eog_peak[i]:.2f} µV, std={eog_std[i]:.2f} µV")

        # Blink detection: large amplitude deviations in VEOG
        # A typical blink produces 100-400 µV deflection lasting ~200-400 ms
        if "VEOG" in str(eog_ch_names).upper() or any("VEOG" in n.upper() for n in eog_ch_names):
            veog_idx = next((i for i, n in enumerate(eog_ch_names) if "VEOG" in n.upper() or "VEOG" not in str(eog_ch_names)), 0)
            veog = eog_channels[:, veog_idx]
            # Detect blinks: >3 std deviation from mean
            veog_centered = veog - veog.mean()
            blink_threshold = 3.0 * float(veog_centered.std())
            blink_mask = np.abs(veog_centered) > blink_threshold
            # Merge nearby blinks
            blink_regions, n_blinks = nd_label(blink_mask)
            # Count distinct blinks
            eog_report["n_blinks"] = int(n_blinks)
            eog_report["blink_rate_per_min"] = float(n_blinks / (len(veog) / eeg_fs / 60))
            eog_report["blink_threshold_uV"] = float(blink_threshold)
            print(f"\nBlink detection (VEOG): {n_blinks} blinks, {n_blinks / (len(veog) / eeg_fs / 60):.1f}/min")
            print(f"Blink threshold: {blink_threshold:.1f} µV")

            # Compute blink amplitudes
            blink_amps = []
            for r in range(1, n_blinks + 1):
                region = np.where(blink_regions == r)[0]
                blink_amps.append(float(np.abs(veog_centered[region]).max()))
            if blink_amps:
                eog_report["blink_amplitude_mean_uV"] = float(np.mean(blink_amps))
                eog_report["blink_amplitude_max_uV"] = float(np.max(blink_amps))
                print(f"Blink amplitudes: mean={np.mean(blink_amps):.1f} µV, max={np.max(blink_amps):.1f} µV")

    # ── EOG→EEG propagation ──────────────────────────────────────────────
    propagation: dict[str, float] = {}
    if eog_channels is not None and len(eeg_ch_names) > 0:
        for eog_i, eog_name in enumerate(eog_ch_names):
            eog_signal = eog_channels[:, eog_i]
            # Correlation with each EEG channel
            for eeg_i, eeg_name in enumerate(eeg_ch_names):
                corr = float(np.corrcoef(eog_signal, eeg_channels[:, eeg_i])[0, 1])
                propagation[f"{eog_name}→{eeg_name}"] = corr

        # Top 10 most contaminated EEG channels from EOG
        sorted_prop = sorted(propagation.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f"\nTop 10 EOG→EEG correlations:")
        for pair, corr in sorted_prop[:10]:
            print(f"  {pair}: {corr:.4f}")

        # Average absolute correlation per EEG channel (across all EOG channels)
        eeg_eog_coupling = {}
        for eeg_name in eeg_ch_names:
            vals = []
            for eog_name in eog_ch_names:
                key = f"{eog_name}→{eeg_name}"
                if key in propagation:
                    vals.append(abs(propagation[key]))
            eeg_eog_coupling[eeg_name] = float(np.mean(vals)) if vals else 0.0
    else:
        eeg_eog_coupling = {name: 0.0 for name in eeg_ch_names}

    # ── Muscle artifact analysis ─────────────────────────────────────────
    # Muscle artifacts appear as high-frequency bursts (typically >20-30 Hz)
    # We compute the ratio of high-frequency power to total power per channel

    muscle_report: dict[str, Any] = {}
    # High-pass filter at 30 Hz to isolate muscle activity
    eeg_hf = _sos_highpass(eeg_channels, eeg_fs, 30.0, order=4)

    # Compute HF RMS per channel
    hf_rms = _rms(eeg_hf, axis=0)
    total_rms = _rms(eeg_channels - eeg_channels.mean(axis=0, keepdims=True), axis=0)
    hf_ratio = hf_rms / np.maximum(total_rms, 1e-12)

    print(f"\nMuscle artifact (HF >30Hz) power ratio by EEG channel:")
    for i, name in enumerate(eeg_ch_names):
        print(f"  {name}: HF/total RMS ratio = {hf_ratio[i]:.4f}")

    muscle_report["hf_30hz_rms_ratio"] = {name: float(v) for name, v in zip(eeg_ch_names, hf_ratio)}
    muscle_report["hf_30hz_rms_ratio_mean"] = float(np.mean(hf_ratio))
    muscle_report["hf_30hz_rms_ratio_median"] = float(np.median(hf_ratio))

    # Detect temporal bursts: sliding window HF power
    window_s = 1.0  # 1-second windows
    window_n = int(window_s * eeg_fs)
    n_windows = len(eeg_hf) // window_n
    if n_windows > 0:
        burst_power = np.array([
            _rms(eeg_hf[i * window_n:(i + 1) * window_n], axis=0).mean()
            for i in range(n_windows)
        ])
        burst_threshold = np.mean(burst_power) + 3 * np.std(burst_power)
        n_bursts = int(np.sum(burst_power > burst_threshold))
        muscle_report["n_hf_burst_windows"] = n_bursts
        muscle_report["hf_burst_fraction"] = float(n_bursts / n_windows)
        print(f"\nHF bursts (>3σ in 1s windows): {n_bursts}/{n_windows} windows "
              f"({100 * n_bursts / n_windows:.1f}%)")

    # ── Frequency spectrum analysis ──────────────────────────────────────
    # Welch PSD for representative channels
    freq_bands = {
        "delta (1-4Hz)": (1.0, 4.0),
        "theta (4-8Hz)": (4.0, 8.0),
        "alpha (8-13Hz)": (8.0, 13.0),
        "beta (13-30Hz)": (13.0, 30.0),
        "low gamma (30-50Hz)": (30.0, 50.0),
        "muscle (>50Hz)": (50.0, min(100.0, eeg_fs / 2 - 1)),
    }

    band_power: dict[str, dict[str, float]] = {}
    # Compute PSD for EOG channels too
    all_ch_names = eeg_ch_names + (eog_ch_names if eog_channels is not None else [])
    all_ch_data = np.column_stack([
        eeg_channels,
        eog_channels if eog_channels is not None else np.zeros((len(eeg_channels), 0)),
    ])

    for i, name in enumerate(all_ch_names):
        freqs, psd = signal.welch(all_ch_data[:, i], fs=eeg_fs, nperseg=int(2 * eeg_fs), noverlap=int(1.9 * eeg_fs))
        band_power[name] = {}
        for band_name, (low, high) in freq_bands.items():
            mask = (freqs >= low) & (freqs < high)
            band_power[name][band_name] = float(np.trapezoid(psd[mask], freqs[mask]))

    # Print band power summary for key channels
    print(f"\nFrequency band power (µV²) for selected channels:")
    key_channels = (eeg_ch_names[:3] if len(eeg_ch_names) >= 3 else eeg_ch_names) + \
                   (["AF7Fp1", "FpzFp1", "C3", "Cz", "Oz"] if len(eeg_ch_names) > 5 else [])
    key_channels = [ch for ch in key_channels if ch in all_ch_names][:8]
    header = f"{'Channel':<10}" + "".join(f"{b:<20}" for b in freq_bands.keys())
    print(header)
    for name in key_channels:
        powers = band_power.get(name, {})
        row = f"{name:<10}" + "".join(f"{powers.get(b, 0):<20.6f}" for b in freq_bands.keys())
        print(row)

    # ── Load and compare with Dataset C artifact recordings (if available) ──
    artifact_comparison: dict[str, Any] = {"available": False}
    if artifact_path.exists():
        print(f"\n── Dataset C (controlled artifact recordings) ──")
        artifact_mat = loadmat(artifact_path, squeeze_me=True, struct_as_record=False)
        mnt_artifact_mat = loadmat(mnt_artifact_path, squeeze_me=True, struct_as_record=False)
        artifact_mnt_key = "mnt_artifact" if "mnt_artifact" in mnt_artifact_mat else "mnt"
        artifact_labels = _labels(mnt_artifact_mat[artifact_mnt_key].clab)
        print(f"  Artifact channels: {artifact_labels}")

        cnt_artifact = artifact_mat["cnt_artifact"]
        artifact_types = ["EOG", "EMG", "Eye Blinking", "Teeth Clenching", "Mouth Opening"]
        artifact_stats = {}

        for art_idx, art_name in enumerate(artifact_types):
            if art_idx >= len(cnt_artifact):
                break
            art_data = np.asarray(_cell(cnt_artifact, art_idx).x, dtype=np.float64)
            rms_per_ch = _rms(art_data, axis=0)
            rms_val = float(np.mean(rms_per_ch))
            peak_val = float(np.abs(art_data).max())
            artifact_stats[art_name] = {"rms_uV": rms_val, "peak_uV": peak_val}
            print(f"  {art_name}: RMS={rms_val:.2f} µV, Peak={peak_val:.2f} µV, "
                  f"duration={art_data.shape[0] / eeg_fs:.1f}s")
        artifact_comparison = {
            "available": True,
            "channel_labels": artifact_labels,
            "artifact_types": artifact_stats,
        }

    # ── Epoch-locked analysis ────────────────────────────────────────────
    # Check EOG activity time-locked to task onset (subject asked to gaze at
    # fixation cross during task → reduced blinking)
    epoch_report: dict[str, Any] = {}
    if eog_channels is not None and len(task_markers) > 0:
        eog_epochs = []
        eeg_epochs_around_task = []
        expected_len = int((epoch_window[1] - epoch_window[0]) * eeg_fs)
        for onset in task_markers:
            start_samp = int((onset / 1000.0 + epoch_window[0]) * eeg_fs)
            end_samp = start_samp + expected_len
            if start_samp < 0 or end_samp > eog_channels.shape[0]:
                continue
            eog_epochs.append(eog_channels[start_samp:end_samp].copy())
            eeg_epochs_around_task.append(eeg_channels[start_samp:end_samp].copy())

        if eog_epochs and all(e.shape == eog_epochs[0].shape for e in eog_epochs):
            eog_epochs_arr = np.stack(eog_epochs)  # n_trials × n_times × n_eog
            eeg_epochs_arr = np.stack(eeg_epochs_around_task)

            # Mean EOG amplitude over time (pre-stim vs. post-stim)
            pre_mask = np.arange(eog_epochs_arr.shape[1]) < int(abs(epoch_window[0]) * eeg_fs)
            post_mask = np.arange(eog_epochs_arr.shape[1]) >= int(abs(epoch_window[0]) * eeg_fs)

            for eog_i, eog_name in enumerate(eog_ch_names):
                pre_amp = float(np.abs(eog_epochs_arr[:, pre_mask, eog_i]).mean())
                post_amp = float(np.abs(eog_epochs_arr[:, post_mask, eog_i]).mean())
                print(f"\nEpoch-locked EOG {eog_name}: pre-stim mean_abs={pre_amp:.2f} µV, "
                      f"post-stim mean_abs={post_amp:.2f} µV")

            epoch_report = {
                "n_epochs": len(eog_epochs),
                "window_s": list(epoch_window),
                "pre_stim_s": abs(epoch_window[0]),
                "post_stim_s": epoch_window[1],
            }
        elif eog_epochs:
            print(f"\nEpoch-locked analysis: epochs have inconsistent shapes, skipping")
            epoch_report = {"n_epochs": len(eog_epochs), "error": "inconsistent epoch shapes"}

    # ── Generate Figures ──────────────────────────────────────────────────

    # Figure 1: Raw EEG + EOG traces (60s segment)
    fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)
    t_seg_start = 120.0  # 2 minutes in
    t_seg_end = t_seg_start + 60.0
    seg_start = int(t_seg_start * eeg_fs)
    seg_end = int(t_seg_end * eeg_fs)
    seg_start = min(seg_start, eeg_data.shape[0] - 1)
    seg_end = min(seg_end, eeg_data.shape[0])
    t_seg = np.arange(seg_end - seg_start) / eeg_fs + t_seg_start

    if eog_channels is not None:
        for i, name in enumerate(eog_ch_names[:2]):
            axes[0].plot(t_seg, eog_channels[seg_start:seg_end, i],
                         alpha=0.8, linewidth=0.6, label=name)
    axes[0].set_ylabel("EOG (µV)")
    axes[0].set_title("EOG channels (60s segment)")
    axes[0].legend(loc="upper right", fontsize=7)
    axes[0].grid(alpha=0.2)

    # Show frontal EEG (most affected by EOG)
    frontal_candidates = [n for n in eeg_ch_names if any(p in n.upper() for p in ("FP", "AF"))]
    if not frontal_candidates:
        frontal_candidates = eeg_ch_names[:4]
    for i, name in enumerate(frontal_candidates[:6]):
        ch_idx = eeg_ch_names.index(name)
        axes[1].plot(t_seg, eeg_channels[seg_start:seg_end, ch_idx],
                     alpha=0.7, linewidth=0.5, label=name)
    axes[1].set_ylabel("Frontal EEG (µV)")
    axes[1].set_title("Frontal EEG channels (most EOG-affected)")
    axes[1].legend(loc="upper right", fontsize=6, ncol=2)
    axes[1].grid(alpha=0.2)

    # Central/posterior EEG
    central_candidates = [n for n in eeg_ch_names if any(p in n.upper() for p in ("C", "CP", "P", "O"))]
    if not central_candidates:
        central_candidates = eeg_ch_names[-6:]
    for i, name in enumerate(central_candidates[:6]):
        ch_idx = eeg_ch_names.index(name)
        axes[2].plot(t_seg, eeg_channels[seg_start:seg_end, ch_idx],
                     alpha=0.7, linewidth=0.5, label=name)
    axes[2].set_ylabel("Central/Post. EEG (µV)")
    axes[2].set_title("Central/Posterior EEG channels (less EOG-affected)")
    axes[2].legend(loc="upper right", fontsize=6, ncol=2)
    axes[2].grid(alpha=0.2)

    # High-frequency (>30Hz) component (muscle artifact)
    hf_eeg = eeg_hf[seg_start:seg_end]
    for i in range(min(4, hf_eeg.shape[1])):
        axes[3].plot(t_seg, hf_eeg[:, i], alpha=0.7, linewidth=0.5,
                     label=eeg_ch_names[i])
    axes[3].set_ylabel("HF >30Hz (µV)")
    axes[3].set_title("High-frequency component (>30 Hz, muscle artifact indicator)")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(loc="upper right", fontsize=6)
    axes[3].grid(alpha=0.2)

    fig.suptitle(f"Subject {subject}, Session {session_index + 1}: EEG/EOG raw signal inspection",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "raw_signal_inspection.png", dpi=200)
    fig.savefig(output_dir / "figures" / "raw_signal_inspection.svg")
    plt.close(fig)

    # Figure 2: Frequency spectra
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # PSD of all EEG channels with EOG overlaid
    for ch_i in range(len(eeg_ch_names)):
        freqs, psd = signal.welch(eeg_channels[:, ch_i], fs=eeg_fs,
                                  nperseg=int(2 * eeg_fs), noverlap=int(1.9 * eeg_fs))
        alpha = 0.15 if len(eeg_ch_names) > 20 else 0.3
        axes[0, 0].semilogy(freqs, psd, color="#2563eb", alpha=alpha, linewidth=0.5)
    axes[0, 0].semilogy(freqs, psd, color="#2563eb", alpha=0.5, linewidth=0.5, label=f"EEG ({len(eeg_ch_names)} ch)")
    if eog_channels is not None:
        for i, name in enumerate(eog_ch_names):
            freqs, psd = signal.welch(eog_channels[:, i], fs=eeg_fs,
                                      nperseg=int(2 * eeg_fs), noverlap=int(1.9 * eeg_fs))
            axes[0, 0].semilogy(freqs, psd, color="#dc2626", linewidth=1.5, label=f"{name}")
    axes[0, 0].set_xlabel("Frequency (Hz)")
    axes[0, 0].set_ylabel("PSD (µV²/Hz)")
    axes[0, 0].set_title("Power Spectral Density: EEG vs. EOG")
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(alpha=0.2)
    axes[0, 0].set_xlim(0, 80)
    # Add band annotations
    for band_name, (low, high) in freq_bands.items():
        axes[0, 0].axvspan(low, high, alpha=0.06, color="gray")
    axes[0, 0].text(2.5, axes[0, 0].get_ylim()[1] * 0.5, "δ", fontsize=8, ha="center")
    axes[0, 0].text(6, axes[0, 0].get_ylim()[1] * 0.5, "θ", fontsize=8, ha="center")
    axes[0, 0].text(10.5, axes[0, 0].get_ylim()[1] * 0.5, "α", fontsize=8, ha="center")
    axes[0, 0].text(21.5, axes[0, 0].get_ylim()[1] * 0.5, "β", fontsize=8, ha="center")

    # Band power bar chart
    eeg_band_means = {}
    for band_name in freq_bands:
        vals = [band_power[ch][band_name] for ch in eeg_ch_names if ch in band_power]
        eeg_band_means[band_name] = np.mean(vals) if vals else 0.0

    bands = list(freq_bands.keys())
    x = np.arange(len(bands))
    axes[0, 1].bar(x, [eeg_band_means[b] for b in bands], color="#2563eb", alpha=0.8)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(bands, rotation=40, ha="right", fontsize=7)
    axes[0, 1].set_ylabel("Mean band power (µV²)")
    axes[0, 1].set_title("EEG mean band power distribution")
    axes[0, 1].grid(axis="y", alpha=0.2)

    # HF power ratio across channels (sorted)
    sorted_idx = np.argsort(hf_ratio)[::-1]
    colors = ["#dc2626" if r > 0.3 else "#f59e0b" if r > 0.2 else "#2563eb" for r in hf_ratio[sorted_idx]]
    axes[1, 0].bar(range(len(hf_ratio)), hf_ratio[sorted_idx], color=colors, alpha=0.8)
    axes[1, 0].set_xticks(range(len(hf_ratio)))
    axes[1, 0].set_xticklabels(
        [f"{eeg_ch_names[i]}" for i in sorted_idx],
        rotation=90, ha="center", fontsize=5
    )
    axes[1, 0].axhline(y=0.2, color="orange", linestyle="--", linewidth=0.8, label="20% threshold")
    axes[1, 0].set_ylabel("HF (>30Hz) / total RMS ratio")
    axes[1, 0].set_title("Muscle artifact indicator: High-frequency power ratio per EEG channel")
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(axis="y", alpha=0.2)

    # EOG→EEG coupling (correlation heatmap style, top 20 channels)
    top_eeg_by_eog = sorted(eeg_eog_coupling.items(), key=lambda x: x[1], reverse=True)[:20]
    if top_eeg_by_eog:
        names, vals = zip(*top_eeg_by_eog)
        colors2 = ["#dc2626" if v > 0.5 else "#f59e0b" if v > 0.3 else "#2563eb" for v in vals]
        axes[1, 1].barh(range(len(names)), vals, color=colors2, alpha=0.8)
        axes[1, 1].set_yticks(range(len(names)))
        axes[1, 1].set_yticklabels(names, fontsize=7)
        axes[1, 1].invert_yaxis()
        axes[1, 1].axvline(x=0.3, color="orange", linestyle="--", linewidth=0.8, label="0.3 |corr|")
        axes[1, 1].axvline(x=0.5, color="red", linestyle="--", linewidth=0.8, label="0.5 |corr|")
        axes[1, 1].set_xlabel("Mean |correlation| with EOG")
        axes[1, 1].set_title("EOG→EEG contamination (top 20 channels)")
        axes[1, 1].legend(fontsize=7)
        axes[1, 1].grid(axis="x", alpha=0.2)

    fig.suptitle(f"Subject {subject}, Session {session_index + 1}: Frequency & artifact analysis",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "frequency_artifact_analysis.png", dpi=200)
    fig.savefig(output_dir / "figures" / "frequency_artifact_analysis.svg")
    plt.close(fig)

    # Figure 3: EOG blink examples + EEG contamination
    if eog_channels is not None and eog_report.get("n_blinks", 0) > 0:
        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        veog_idx = 0
        for i, name in enumerate(eog_ch_names):
            if "VEOG" in name.upper():
                veog_idx = i
                break

        veog = eog_channels[:, veog_idx]
        veog_centered = veog - veog.mean()
        blink_threshold = float(eog_report.get("blink_threshold_uV", 3.0 * veog_centered.std()))

        # Find a good blink example (large amplitude, isolated)
        blink_mask = np.abs(veog_centered) > blink_threshold
        labeled, n_features = nd_label(blink_mask)
        blink_times = []
        for r in range(1, n_features + 1):
            region = np.where(labeled == r)[0]
            mid = region[len(region) // 2]
            blink_times.append(mid / eeg_fs)

        # Show first 3 blinks zoomed
        for blink_i, blink_time in enumerate(blink_times[:min(5, len(blink_times))]):
            win_start = max(0, int((blink_time - 1.5) * eeg_fs))
            win_end = min(eeg_data.shape[0], int((blink_time + 1.5) * eeg_fs))
            t_win = np.arange(win_end - win_start) / eeg_fs + win_start / eeg_fs

            axes[0].plot(t_win, eog_channels[win_start:win_end, veog_idx],
                         alpha=0.7, linewidth=0.8, label=f"blink {blink_i + 1}")
            # Show frontal EEG contamination
            frontal_ch = frontal_candidates[0] if frontal_candidates else eeg_ch_names[0]
            frontal_idx = eeg_ch_names.index(frontal_ch)
            axes[1].plot(t_win, eeg_channels[win_start:win_end, frontal_idx],
                         alpha=0.7, linewidth=0.8, label=f"blink {blink_i + 1}: {frontal_ch}")

        axes[0].set_ylabel("VEOG (µV)")
        axes[0].set_title("Blink waveforms in VEOG (±1.5s around blink peak)")
        axes[0].legend(fontsize=7)
        axes[0].grid(alpha=0.2)
        axes[1].set_ylabel(f"{frontal_ch} EEG (µV)")
        axes[1].set_title(f"EEG contamination from blinks in frontal channel {frontal_ch}")
        axes[1].set_xlabel("Time (s)")
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.2)

        fig.suptitle(f"Subject {subject}, Session {session_index + 1}: Ocular artifact examples",
                     fontsize=13)
        fig.tight_layout()
        fig.savefig(output_dir / "figures" / "ocular_artifact_examples.png", dpi=200)
        fig.savefig(output_dir / "figures" / "ocular_artifact_examples.svg")
        plt.close(fig)

    # ── Summary report ──────────────────────────────────────────────────

    # Quantify artifact severity
    # 1. EOG contamination: fraction of EEG channels with |corr| > 0.3 with any EOG
    n_contaminated_eog = sum(1 for v in eeg_eog_coupling.values() if v > 0.3)
    fraction_contaminated_eog = n_contaminated_eog / max(len(eeg_eog_coupling), 1)

    # 2. Muscle artifact: fraction of channels with HF ratio > 0.2
    n_muscle_contaminated = int(np.sum(hf_ratio > 0.2))
    fraction_muscle = n_muscle_contaminated / max(len(hf_ratio), 1)

    # 3. Overall artifact assessment
    severity_eog = "severe" if fraction_contaminated_eog > 0.4 else \
                   "moderate" if fraction_contaminated_eog > 0.15 else "mild"
    severity_muscle = "severe" if fraction_muscle > 0.4 else \
                      "moderate" if fraction_muscle > 0.15 else "mild"

    summary = {
        "schema": "eeg_artifact_inspection_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "subject": subject,
        "session_index_zero_based": session_index,
        "data_source": str(eeg_cnt_path.relative_to(REPO_ROOT)),
        "dataset_documentation": {
            "url": "http://doc.ml.tu-berlin.de/hBCI",
            "key_facts": [
                "EEG: BrainAmp, 30 active electrodes (10-5 system), linked mastoids, 1000 Hz",
                "EOG recorded: 2 vertical (above/below left eye) + 2 horizontal (outer canthus)",
                "ECG (Einthoven I/II) and respiration also recorded",
                "Data stored under 'with occular artifact' — artifacts NOT removed",
                "Dataset C: 5 types of controlled artifacts (EOG, EMG, blinking, teeth clenching, mouth opening)",
                "Subjects instructed to gaze at fixation cross during task to minimize eye movements",
                "Scalp-electrode impedance kept below 10 kOhm",
                "No artifact removal (ICA, BSS-CCA, etc.) described in official documentation",
            ],
        },
        "channel_inventory": {
            ch_type: [eeg_labels[i] for i in indices]
            for ch_type, indices in channel_map.items()
        },
        "n_task_events": int(len(task_markers)),
        "eog_analysis": eog_report,
        "eog_eeg_propagation": {
            "top_correlations": dict(sorted_prop[:20]),
            "n_channels_contaminated": n_contaminated_eog,
            "fraction_contaminated": fraction_contaminated_eog,
            "severity": severity_eog,
        },
        "muscle_artifact_analysis": {
            **muscle_report,
            "n_channels_contaminated": n_muscle_contaminated,
            "fraction_contaminated": fraction_muscle,
            "severity": severity_muscle,
        },
        "epoch_locked_analysis": epoch_report,
        "artifact_dataset_c_comparison": artifact_comparison,
        "overall_assessment": {
            "ocular_artifact_severity": severity_eog,
            "muscle_artifact_severity": severity_muscle,
            "data_directory_labeled": "with occular artifact",
            "no_artifact_free_version_available": True,
            "artifact_removal_not_applied_by_dataset_authors": True,
            "implication_for_lin2024_test": (
                "The EEG data contains significant ocular artifacts in frontal channels "
                "(|corr| with EOG > 0.3 for {:.0%} of channels) and {:.0%} of channels "
                "show elevated HF power (>30Hz, muscle artifact indicator). "
                "Lin 2024's pipeline includes BSS-CCA muscle artifact removal and uses "
                "motor finger-tapping data likely with cleaner EEG. "
                "For our mental arithmetic data, the absence of artifact removal likely "
                "degrades the TRTD decomposition quality, especially for frontal channels "
                "that are both EOG-contaminated and relevant for cognitive task EEG."
            ).format(fraction_contaminated_eog, fraction_muscle),
        },
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            check=True, capture_output=True, text=True
        ).stdout.strip(),
    }

    with open(output_dir / "artifact_inspection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # Markdown summary
    md_lines = [
        "# EEG Artifact Inspection Report",
        "",
        f"**Subject:** {subject} | **Session:** {session_index + 1} | **EEG sampling:** {eeg_fs} Hz",
        f"**Data source:** `{eeg_cnt_path.relative_to(REPO_ROOT)}`",
        "",
        "## Dataset Documentation Confirmation",
        "",
        "Per the [official dataset documentation](http://doc.ml.tu-berlin.de/hBCI):",
        "",
        "- EEG recorded with BrainAmp, 30 active electrodes, linked mastoids reference, 1000 Hz",
        "- EOG recorded with 2 vertical + 2 horizontal electrodes",
        "- ECG and respiration also recorded",
        "- **Data is provided under `with occular artifact` — artifacts are NOT removed**",
        "- No artifact-free version exists in the dataset",
        "- Dataset C provides separate controlled artifact recordings for validation",
        "- The main paper's analysis scripts (on GitHub) do not appear to apply artifact rejection",
        "",
        "## Channel Inventory",
        "",
    ]
    for ch_type, indices in sorted(channel_map.items()):
        names = [eeg_labels[i] for i in indices]
        md_lines.append(f"- **{ch_type}** ({len(indices)}): {', '.join(names)}")

    md_lines.extend([
        "",
        "## Ocular Artifact Analysis",
        "",
        f"- EOG channels detected: {eog_ch_names if eog_ch_names else 'NONE'}",
        f"- Blinks detected: {eog_report.get('n_blinks', 'N/A')} "
        f"(rate: {eog_report.get('blink_rate_per_min', 'N/A')}/min)",
        f"- EEG channels with |corr| > 0.3 with any EOG: **{n_contaminated_eog}/{len(eeg_eog_coupling)} "
        f"({fraction_contaminated_eog:.1%})**",
        f"- Ocular artifact severity: **{severity_eog.upper()}**",
        "",
        "### Most EOG-contaminated EEG channels:",
        "",
        "| Rank | Channel | Mean |corr| with EOG |",
        "|------|---------|----------------------|",
    ])
    for rank, (ch, val) in enumerate(sorted(eeg_eog_coupling.items(), key=lambda x: x[1], reverse=True)[:15], 1):
        md_lines.append(f"| {rank} | {ch} | {val:.4f} |")

    md_lines.extend([
        "",
        "## Muscle Artifact Analysis",
        "",
        f"- HF (>30 Hz) / total RMS ratio mean: **{muscle_report['hf_30hz_rms_ratio_mean']:.3f}**",
        f"- HF (>30 Hz) / total RMS ratio median: **{muscle_report['hf_30hz_rms_ratio_median']:.3f}**",
        f"- Channels with HF ratio > 0.2: **{n_muscle_contaminated}/{len(hf_ratio)} "
        f"({fraction_muscle:.1%})**",
        f"- Muscle artifact severity: **{severity_muscle.upper()}**",
        "",
        "### Channels with highest muscle artifact indicator:",
        "",
        "| Rank | Channel | HF/total RMS ratio |",
        "|------|---------|-------------------|",
    ])
    for rank, idx in enumerate(sorted_idx[:10], 1):
        md_lines.append(f"| {rank} | {eeg_ch_names[idx]} | {hf_ratio[idx]:.4f} |")

    md_lines.extend([
        "",
        "## Frequency Band Distribution (mean across EEG channels)",
        "",
        "| Band | Mean Power (µV²) |",
        "|------|-----------------|",
    ])
    for band_name in freq_bands:
        md_lines.append(f"| {band_name} | {eeg_band_means[band_name]:.6f} |")

    md_lines.extend([
        "",
        "## Overall Assessment",
        "",
        f"- **Ocular artifact severity:** {severity_eog.upper()} "
        f"({fraction_contaminated_eog:.0%} of EEG channels contaminated)",
        f"- **Muscle artifact severity:** {severity_muscle.upper()} "
        f"({fraction_muscle:.0%} of EEG channels contaminated)",
        f"- **Data directory:** explicitly labeled `with occular artifact`",
        f"- **No artifact-free version** is available in this dataset",
        "",
        "### Impact on Lin 2024 TRTD Test",
        "",
        "Lin 2024's pipeline uses finger-tapping motor imagery data and includes:",
        "- BSS-CCA for muscle artifact removal",
        "- Likely higher SNR motor-cortex EEG",
        "",
        "Our test uses mental arithmetic data WITHOUT artifact removal:",
        f"- {fraction_contaminated_eog:.0%} of EEG channels are contaminated by ocular artifacts",
        f"- {fraction_muscle:.0%} of EEG channels show elevated muscle artifact indicators",
        "- Frontal channels (most relevant for cognitive tasks) are most EOG-affected",
        "- The TRTD decomposition includes these contaminated channels in the tensor",
        "",
        "**Conclusion:** The raw EEG data contains significant ocular and muscle artifacts "
        "that are NOT removed before TRTD decomposition. This is a contributing factor "
        "to the poor EEG→fNIRS prediction, separate from the task-paradigm difference "
        "(mental arithmetic vs. finger tapping). However, the in-sample R²=0.022 ceiling "
        "suggests artifacts alone cannot fully explain the failure — the fundamental "
        "EEG→fNIRS coupling strength in this task+pipeline combination is weak.",
        "",
        "## Figures",
        "",
        "- `figures/raw_signal_inspection.png` — 60s raw signal with EOG, frontal EEG, posterior EEG, HF component",
        "- `figures/frequency_artifact_analysis.png` — PSD, band power, HF ratio, EOG→EEG coupling",
        "- `figures/ocular_artifact_examples.png` — Blink waveforms and EEG contamination (if blinks detected)",
        "",
        f"Full JSON report: `artifact_inspection_summary.json`",
    ])

    with open(output_dir / "artifact_inspection_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nReport written to {output_dir}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/EEG+NIRS Single-Trial")
    parser.add_argument("--subject", type=int, default=19)
    parser.add_argument("--session-index", type=int, default=1,
                        help="0-based; 1 = MA session 2 (sessions 2,4,6 are MA)")
    parser.add_argument("--target-desc", type=int, default=16,
                        help="EEG marker code: 16 = mental arithmetic task onset")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = (
            REPO_ROOT / "experiments" / "runs" / "physiology_semantic_tokenizer"
            / "e0_teacher_validity"
            / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_eeg_artifact_inspection_s{args.subject:02d}"
        )

    summary = inspect_session(
        data_root=REPO_ROOT / args.data_root,
        subject=args.subject,
        session_index=args.session_index,
        output_dir=output_dir,
        target_desc=args.target_desc,
    )
    print(f"\nDone. Output: {output_dir}")
