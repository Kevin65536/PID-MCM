#!/usr/bin/env python3
"""Audit the E1 model-view scale on registered training subjects only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.physiology_semantic_local import UnifiedPhysiologyLocalViewDataset


QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        f"q{int(100 * quantile):02d}": float(np.quantile(array, quantile))
        for quantile in QUANTILES
    }


def _audit(config_path: Path) -> dict[str, Any]:
    raw_config = config_path.read_bytes()
    config = yaml.safe_load(raw_config)
    data = config["data"]
    train_subjects = list(data["split"]["train_subject_keys"])
    protected_subjects = set(data["split"].get("test_subject_keys", []))
    if protected_subjects.intersection(train_subjects):
        raise RuntimeError("Training subject set overlaps the protected test set")
    dataset = UnifiedPhysiologyLocalViewDataset(
        cache_root=data["cache_root"],
        dataset_ids=data["dataset_ids"],
        subject_keys=train_subjects,
        task_namespaces=data["task_namespaces"],
        window_duration_s=float(data.get("window", {}).get("duration_s", 20.0)),
        local_eeg_channels=int(data.get("local_view", {}).get("eeg_channels", 6)),
        reject_unknown_labels=bool(data.get("reject_unknown_labels", True)),
        allow_cross_coordinate_systems=bool(
            data.get("local_view", {}).get("allow_cross_coordinate_systems", False)
        ),
    )
    measurements: dict[str, dict[str, list[float]]] = {
        modality: {
            "channel_mean": [],
            "absolute_channel_mean": [],
            "channel_std": [],
            "within_window_std_ratio": [],
        }
        for modality in ("eeg", "fnirs")
    }
    empty_valid_windows = {modality: 0 for modality in ("eeg", "fnirs")}
    for sample in dataset:
        for modality in ("eeg", "fnirs"):
            signal = sample[modality].numpy()
            token_mask = sample["token_valid_mask"][modality].numpy().astype(bool)
            if signal.shape[-1] % token_mask.size:
                raise RuntimeError("Token-valid mask does not divide the signal length")
            sample_mask = np.repeat(token_mask, signal.shape[-1] // token_mask.size)
            if not sample_mask.any():
                empty_valid_windows[modality] += 1
                continue
            valid = signal[:, sample_mask]
            means = np.mean(valid, axis=1, dtype=np.float64)
            stds = np.std(valid, axis=1, dtype=np.float64)
            positive = stds[stds > np.finfo(np.float32).eps]
            ratio = float(positive.max() / positive.min()) if positive.size else float("inf")
            measurements[modality]["channel_mean"].extend(means.tolist())
            measurements[modality]["absolute_channel_mean"].extend(np.abs(means).tolist())
            measurements[modality]["channel_std"].extend(stds.tolist())
            measurements[modality]["within_window_std_ratio"].append(ratio)

    modality_results = {}
    for modality, metrics in measurements.items():
        stds = np.asarray(metrics["channel_std"], dtype=np.float64)
        modality_results[modality] = {
            key: _summary(values) for key, values in metrics.items()
        }
        modality_results[modality]["channel_std_outside_0p5_2_fraction"] = float(
            np.mean((stds < 0.5) | (stds > 2.0))
        )
    return {
        "schema": "e1_training_input_normalization_audit_v1",
        "config": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "source": "registered training subjects only",
        "protected_test_opened": False,
        "sample_count": len(dataset),
        "subject_count": len(train_subjects),
        "empty_valid_windows": empty_valid_windows,
        "canonical_input_contract": "full-record per-channel median/MAD robust standardization",
        "archived_comparator_contract": "per-crop source/observation component-wise mean/std standardization",
        "modalities": modality_results,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# E1 training-input normalization audit",
        "",
        "Training subjects only; the protected test split was not opened.",
        "",
        f"- Samples: `{payload['sample_count']}` from `{payload['subject_count']}` subjects",
        f"- Current contract: `{payload['canonical_input_contract']}`",
        f"- Archived comparator: `{payload['archived_comparator_contract']}`",
        "",
        "| modality | channel std q05/q50/q95 | abs channel mean q50/q95 | within-window std ratio q50/q95 | std outside [0.5,2] |",
        "|---|---:|---:|---:|---:|",
    ]
    for modality in ("eeg", "fnirs"):
        result = payload["modalities"][modality]
        std = result["channel_std"]
        mean = result["absolute_channel_mean"]
        ratio = result["within_window_std_ratio"]
        lines.append(
            "| {modality} | {s05:.3f}/{s50:.3f}/{s95:.3f} | {m50:.3f}/{m95:.3f} | "
            "{r50:.3f}/{r95:.3f} | {outside:.3%} |".format(
                modality=modality,
                s05=std["q05"], s50=std["q50"], s95=std["q95"],
                m50=mean["q50"], m95=mean["q95"],
                r50=ratio["q50"], r95=ratio["q95"],
                outside=result["channel_std_outside_0p5_2_fraction"],
            )
        )
    lines.extend(
        [
            "",
            "The current canonicalization is real input normalization, but it is not the archived branch-wise crop normalization. These statistics diagnose the residual window-level scale variation; they do not by themselves authorize a per-window transform, which could remove physiologically meaningful amplitude information.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = _audit(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.md").write_text(
        _markdown(payload), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
