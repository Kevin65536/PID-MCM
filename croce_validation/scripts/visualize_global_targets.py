#!/usr/bin/env python3
"""Spatial heatmaps for assembled global source/observation targets.

For each modality (EEG, fNIRS optical channel 0, fNIRS optical channel 1),
plots per-channel summary values on the scalp layout, comparing:

  Original  = source + obs  (reconstructed raw signal)
  Source    = physiological prediction
  Obs       = residual (observation noise)

This verifies that source targets carry spatial structure while observation
targets are noise-like and spatially decorrelated.

Usage:
    python croce_validation/scripts/visualize_global_targets.py \\
        --global-cache croce_validation/cache/pf_full/subject_1/global_targets.npz \\
        --subject-id 1 --use-artifact-eeg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.channel_adjacency import build_channel_adjacency
from src.data.eeg_fnirs_dataset import MultiModalEEGfNIRSDataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Spatial heatmaps for global source/observation targets."
    )
    p.add_argument("--global-cache", required=True, type=Path,
                   help="Path to global_targets.npz from assemble_global_targets.py")
    p.add_argument("--data-root", default="data/EEG+NIRS Single-Trial")
    p.add_argument("--subject-id", type=int, default=1)
    p.add_argument("--use-artifact-eeg", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory (default: same as global-cache)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _head_outline(ax: plt.Axes, radius: float = 0.52) -> None:
    """Draw a schematic head outline + nose."""
    circle = Circle((0, 0), radius, fill=False, edgecolor="#444444",
                     linewidth=1.0, zorder=0)
    ax.add_patch(circle)
    # Nose triangle
    nose_x = [0.0, -0.04, 0.04]
    nose_y = [radius + 0.02, radius + 0.06, radius + 0.06]
    ax.fill(nose_x, nose_y, facecolor="#444444", edgecolor="none", zorder=0)


def _plot_scalp_scatter(
    ax: plt.Axes,
    positions: np.ndarray,
    values: np.ndarray,
    title: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "RdBu_r",
    label: str = "",
) -> Tuple[float, float]:
    """Scatter-based topomap: coloured circles at channel positions."""
    vmin = vmin if vmin is not None else float(np.nanmin(values))
    vmax = vmax if vmax is not None else float(np.nanmax(values))
    vrange = max(abs(vmin), abs(vmax))
    vmin_sym = -vrange if cmap == "RdBu_r" else vmin
    vmax_sym = vrange if cmap == "RdBu_r" else vmax

    sc = ax.scatter(
        positions[:, 0], positions[:, 1],
        c=values, cmap=cmap, s=80, edgecolors="#333333",
        linewidths=0.5, zorder=5, vmin=vmin_sym, vmax=vmax_sym,
    )
    _head_outline(ax)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, fontweight="bold")
    if label:
        ax.set_xlabel(label, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    return vmin_sym, vmax_sym


# ---------------------------------------------------------------------------
# Main plotting
# ---------------------------------------------------------------------------

def plot_spatial_heatmaps(
    global_cache_path: Path,
    adjacency: Any,
    output_dir: Path,
) -> None:
    """Generate spatial heatmap figures for all events in the global cache."""

    data = dict(np.load(global_cache_path, allow_pickle=True))
    all_keys = sorted(data.keys())

    # Discover events
    events: Dict[str, Dict[str, np.ndarray]] = {}
    for key in all_keys:
        if "/" not in key:
            continue
        event_part, field = key.split("/", 1)
        events.setdefault(event_part, {})[field] = np.asarray(data[key])

    if not events:
        raise ValueError("No events found in global cache")

    # Unique fNIRS spatial positions for the topomap
    fnirs_names_full = list(adjacency.fnirs_channel_names)
    # First 36 are lowWL, next 36 are highWL — both share the same positions
    n_spatial = len(fnirs_names_full) // 2
    fnirs_spatial_positions = np.asarray(
        adjacency.fnirs_channel_positions_2d[:n_spatial], dtype=np.float64
    )

    eeg_positions = np.asarray(adjacency.eeg_positions_2d, dtype=np.float64)
    eeg_names = list(adjacency.eeg_channel_names)

    for event_part, fields in sorted(events.items()):
        source_eeg = fields.get("source_eeg")
        obs_eeg = fields.get("obs_eeg")
        source_fnirs0 = fields.get("source_fnirs_optical_channel_0")
        obs_fnirs0 = fields.get("obs_fnirs_optical_channel_0")
        source_fnirs1 = fields.get("source_fnirs_optical_channel_1")
        obs_fnirs1 = fields.get("obs_fnirs_optical_channel_1")

        if source_eeg is None:
            continue

        # Reconstruct original = source + obs
        orig_eeg = source_eeg + obs_eeg
        orig_fnirs0 = source_fnirs0 + obs_fnirs0
        orig_fnirs1 = source_fnirs1 + obs_fnirs1

        # ---- Temporal summaries per channel ----
        eeg_orig_mean = np.mean(orig_eeg, axis=0)       # (30,)
        eeg_src_mean = np.mean(source_eeg, axis=0)
        eeg_obs_mean = np.mean(obs_eeg, axis=0)

        fnirs0_orig_mean = np.mean(orig_fnirs0, axis=0)  # (36,)
        fnirs0_src_mean = np.mean(source_fnirs0, axis=0)
        fnirs0_obs_mean = np.mean(obs_fnirs0, axis=0)

        fnirs1_orig_mean = np.mean(orig_fnirs1, axis=0)
        fnirs1_src_mean = np.mean(source_fnirs1, axis=0)
        fnirs1_obs_mean = np.mean(obs_fnirs1, axis=0)

        # ---- Figure: EEG + 2 fNIRS channels, 3 columns each ----
        fig, axes = plt.subplots(3, 3, figsize=(15, 13))

        # EEG: shared symmetric range for orig/src/obs (all have comparable magnitude)
        eeg_vmax = max(
            float(np.nanmax(np.abs(eeg_orig_mean))),
            float(np.nanmax(np.abs(eeg_src_mean))),
            float(np.nanmax(np.abs(eeg_obs_mean))),
            1.0,
        )
        _plot_scalp_scatter(axes[0, 0], eeg_positions, eeg_orig_mean,
                            "EEG — Original (source + obs)", vmin=-eeg_vmax, vmax=eeg_vmax,
                            label="μV")
        _plot_scalp_scatter(axes[0, 1], eeg_positions, eeg_src_mean,
                            "EEG — Source (physiological)", vmin=-eeg_vmax, vmax=eeg_vmax,
                            label="μV")
        _plot_scalp_scatter(axes[0, 2], eeg_positions, eeg_obs_mean,
                            "EEG — Observation (residual)", vmin=-eeg_vmax, vmax=eeg_vmax,
                            label="μV")

        # fNIRS: source ≈ original (same scale), obs is 1000× smaller (separate scale)
        for row_idx, (fnirs_src_mean, fnirs_obs_mean, fnirs_orig_mean, label) in enumerate([
            (fnirs0_src_mean, fnirs0_obs_mean, fnirs0_orig_mean, "fNIRS highWL"),
            (fnirs1_src_mean, fnirs1_obs_mean, fnirs1_orig_mean, "fNIRS lowWL"),
        ]):
            r = row_idx + 1
            fnirs_src_vmax = max(float(np.nanmax(np.abs(fnirs_orig_mean))),
                                 float(np.nanmax(np.abs(fnirs_src_mean))), 0.01)
            fnirs_obs_vmax = max(float(np.nanmax(np.abs(fnirs_obs_mean))), 1e-6)

            _plot_scalp_scatter(axes[r, 0], fnirs_spatial_positions, fnirs_orig_mean,
                                f"{label} — Original", vmin=-fnirs_src_vmax, vmax=fnirs_src_vmax,
                                label="V")
            _plot_scalp_scatter(axes[r, 1], fnirs_spatial_positions, fnirs_src_mean,
                                f"{label} — Source", vmin=-fnirs_src_vmax, vmax=fnirs_src_vmax,
                                label="V")
            _plot_scalp_scatter(axes[r, 2], fnirs_spatial_positions, fnirs_obs_mean,
                                f"{label} — Observation", vmin=-fnirs_obs_vmax, vmax=fnirs_obs_vmax,
                                label="V", cmap="RdBu_r")

        fig.suptitle(
            f"Spatial Heatmaps — Temporal Mean per Channel\n"
            f"Event: {event_part}",
            fontsize=13, fontweight="bold", y=0.998,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

        out_path = output_dir / f"spatial_heatmap_{event_part}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

        # ---- Optional: spatial variance heatmaps ----
        eeg_orig_std = np.std(orig_eeg, axis=0)
        eeg_src_std = np.std(source_eeg, axis=0)
        eeg_obs_std = np.std(obs_eeg, axis=0)

        fnirs0_orig_std = np.std(orig_fnirs0, axis=0)
        fnirs0_src_std = np.std(source_fnirs0, axis=0)
        fnirs0_obs_std = np.std(obs_fnirs0, axis=0)

        fig2, axes2 = plt.subplots(2, 3, figsize=(15, 9))

        eeg_vmax_std = max(
            float(np.nanmax(eeg_orig_std)), float(np.nanmax(eeg_src_std)),
            float(np.nanmax(eeg_obs_std)), 1.0,
        )
        _plot_scalp_scatter(axes2[0, 0], eeg_positions, eeg_orig_std,
                            "EEG — Original std", cmap="YlOrRd",
                            vmin=0, vmax=eeg_vmax_std, label="μV")
        _plot_scalp_scatter(axes2[0, 1], eeg_positions, eeg_src_std,
                            "EEG — Source std", cmap="YlOrRd",
                            vmin=0, vmax=eeg_vmax_std, label="μV")
        _plot_scalp_scatter(axes2[0, 2], eeg_positions, eeg_obs_std,
                            "EEG — Observation std", cmap="YlOrRd",
                            vmin=0, vmax=eeg_vmax_std, label="μV")

        fnirs0_vmax_std = max(
            float(np.nanmax(fnirs0_orig_std)), float(np.nanmax(fnirs0_src_std)),
            float(np.nanmax(fnirs0_obs_std)), 0.01,
        )
        _plot_scalp_scatter(axes2[1, 0], fnirs_spatial_positions, fnirs0_orig_std,
                            "fNIRS highWL — Original std", cmap="YlOrRd",
                            vmin=0, vmax=fnirs0_vmax_std, label="V")
        _plot_scalp_scatter(axes2[1, 1], fnirs_spatial_positions, fnirs0_src_std,
                            "fNIRS highWL — Source std", cmap="YlOrRd",
                            vmin=0, vmax=fnirs0_vmax_std, label="V")
        _plot_scalp_scatter(axes2[1, 2], fnirs_spatial_positions, fnirs0_obs_std,
                            "fNIRS highWL — Observation std", cmap="YlOrRd",
                            vmin=0, vmax=fnirs0_vmax_std, label="V")

        fig2.suptitle(
            f"Spatial Variability (std) per Channel\n"
            f"Event: {event_part}",
            fontsize=13, fontweight="bold", y=0.998,
        )
        fig2.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

        out_path2 = output_dir / f"spatial_variance_{event_part}.png"
        fig2.savefig(out_path2, dpi=180, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved: {out_path2}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cache_path = Path(args.global_cache)
    if not cache_path.exists():
        raise FileNotFoundError(f"Global cache not found: {cache_path}")

    output_dir = args.output_dir or cache_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = MultiModalEEGfNIRSDataset(
        data_root=args.data_root,
        subject_ids=[int(args.subject_id)],
        task="motor_imagery",
        window_duration_s=2.5,
        normalize=False,
        normalization_mode="none",
        use_artifact_data=bool(args.use_artifact_eeg),
        exclude_eog=True,
        hbo_only=False,
        hbr_only=False,
    )
    adjacency = build_channel_adjacency(
        "eeg_fnirs_single_trial",
        args.data_root,
        ds.get_eeg_channel_names(),
        ds.get_fnirs_channel_names(),
        reference_subject_id=int(args.subject_id),
        use_artifact_data=bool(args.use_artifact_eeg),
    )

    print(f"Plotting spatial heatmaps for {cache_path} ...")
    plot_spatial_heatmaps(cache_path, adjacency, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
