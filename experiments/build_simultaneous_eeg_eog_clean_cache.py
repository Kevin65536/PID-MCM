#!/usr/bin/env python3
"""Materialize and audit conservative HEOG/VEOG repair for Simultaneous EEG."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.eeg_artifact_preprocessing import (
    clean_single_trial_eeg,
)
from src.data.unified_physiology import (
    SIMULTANEOUS_EEG_EOG_CLEAN_SCHEMA_V1,
    _simultaneous_eeg,
    preprocess_eeg_record_with_quality,
    simultaneous_eeg_eog_cleaning_config,
)


CACHE_SCHEMA = "simultaneous_eeg_eog_cache_v1"
TASKS = ("nback", "dsr", "wg")
NEAR_EYE = frozenset({"Fp1", "Fp2", "AFz", "AFF5h", "AFF6h"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        default="data/cache/physiology_semantic_clean_v1/simultaneous_eeg_eog_clean_v1",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/runs/physiology_semantic_tokenizer/data_quality_audit/"
        "simultaneous_eog_clean_20260718",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _worker(subject: str, cache_root_text: str) -> dict[str, Any]:
    cache_root = Path(cache_root_text)
    rows = []
    config = simultaneous_eeg_eog_cleaning_config()
    for task in TASKS:
        base_record_id = f"cnt_{task}"
        native = _simultaneous_eeg(
            REPO_ROOT,
            SimpleNamespace(canonical_subject_id=subject, base_record_id=base_record_id),
        )
        canonical, state, quality = preprocess_eeg_record_with_quality(
            native,
            signal_branch=SIMULTANEOUS_EEG_EOG_CLEAN_SCHEMA_V1,
            artifact_config=config,
        )
        cleaning = state["artifact_cleaning"]
        source_stat = native.source_path.stat()
        join_key = f"simultaneous_eeg_nirs|{subject}|{base_record_id}"
        output_path = cache_root / subject / f"{base_record_id}.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(".npz.tmp")
        with temporary_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                schema=np.asarray(CACHE_SCHEMA),
                signal_branch=np.asarray(SIMULTANEOUS_EEG_EOG_CLEAN_SCHEMA_V1),
                join_key=np.asarray(join_key),
                source_path=np.asarray(str(native.source_path.relative_to(REPO_ROOT))),
                source_size_bytes=np.asarray(source_stat.st_size, dtype=np.int64),
                source_mtime_ns=np.asarray(source_stat.st_mtime_ns, dtype=np.int64),
                eeg=canonical,
                artifact_mask=np.asarray(quality["artifact_mask"], dtype=bool),
                bad_channel_mask=np.asarray(quality["bad_channel_mask"], dtype=bool),
                channel_names=np.asarray(native.channel_names),
                preprocessing_state_json=np.asarray(json.dumps(state, ensure_ascii=False)),
            )
        temporary_path.replace(output_path)

        before = np.asarray(cleaning["eog_correlation_before"], dtype=float)
        after = np.asarray(cleaning["eog_correlation_after"], dtype=float)
        removed = np.asarray(
            cleaning["eog_regression"]["removed_variance_fraction"], dtype=float
        )
        near = np.asarray([name in NEAR_EYE for name in native.channel_names], dtype=bool)
        posterior = np.asarray([
            name.startswith(("P", "O")) for name in native.channel_names
        ], dtype=bool)
        preservation = cleaning["information_preservation"]
        rows.append({
            "join_key": join_key,
            "subject": subject,
            "task": task,
            "source_path": str(native.source_path.relative_to(REPO_ROOT)),
            "cache_path": str(output_path.relative_to(REPO_ROOT)),
            "sample_count": int(canonical.shape[0]),
            "scalp_eeg_channel_count": int(canonical.shape[1]),
            "auxiliary_eog_channel_count": len(native.auxiliary_channel_names),
            "auxiliary_eog_channel_names": list(native.auxiliary_channel_names),
            "output_contains_eog": any("EOG" in name.upper() for name in native.channel_names),
            "artifact_fraction": float(np.mean(quality["artifact_mask"])),
            "bad_channel_count": int(np.count_nonzero(quality["bad_channel_mask"])),
            "median_eog_correlation_before": float(np.median(before)),
            "median_eog_correlation_after": float(np.median(after)),
            "near_eye_eog_correlation_before": float(np.median(before[near])),
            "near_eye_eog_correlation_after": float(np.median(after[near])),
            "posterior_eog_correlation_before": float(np.median(before[posterior])),
            "posterior_eog_correlation_after": float(np.median(after[posterior])),
            "median_removed_variance_fraction": float(np.median(removed)),
            "maximum_removed_variance_fraction": float(np.max(removed)),
            "median_preserved_waveform_correlation": float(
                preservation["median_waveform_correlation"]
            ),
            "minimum_preserved_waveform_correlation": float(
                preservation["minimum_waveform_correlation"]
            ),
            "median_high_frequency_variance_ratio": float(
                preservation["median_high_frequency_variance_ratio"]
            ),
        })
    return {"subject": subject, "rows": rows}


def _plot(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs = []

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    before_near = [row["near_eye_eog_correlation_before"] for row in rows]
    after_near = [row["near_eye_eog_correlation_after"] for row in rows]
    before_post = [row["posterior_eog_correlation_before"] for row in rows]
    after_post = [row["posterior_eog_correlation_after"] for row in rows]
    axes[0].boxplot([before_near, after_near, before_post, after_post], tick_labels=[
        "near-eye\nbefore", "near-eye\nafter", "posterior\nbefore", "posterior\nafter"
    ])
    axes[0].set_ylabel("max |correlation| with HEOG/VEOG")
    axes[0].set_title("Ocular coupling before and after repair")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].scatter(
        [row["median_removed_variance_fraction"] for row in rows],
        [row["median_preserved_waveform_correlation"] for row in rows],
        c=[TASKS.index(row["task"]) for row in rows], cmap="viridis", alpha=0.8,
    )
    axes[1].set_xlabel("median removed variance fraction")
    axes[1].set_ylabel("waveform correlation outside ocular mask")
    axes[1].set_title("Information-preservation audit")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        path = figures / f"simultaneous_eog_repair_audit.{suffix}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        outputs.append(str(path.relative_to(output_dir)))
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    cache_root = (REPO_ROOT / args.cache_root).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output_dir}")
    cache_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    subjects = sorted(path.name.split("-")[0] for path in (
        REPO_ROOT / "data/Simultaneous EEG&NIRS"
    ).glob("VP*-EEG"))
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_worker, subject, str(cache_root)): subject for subject in subjects}
        for future in as_completed(futures):
            result = future.result()
            rows.extend(result["rows"])
            print(f"completed {result['subject']}", flush=True)
    rows.sort(key=lambda row: (row["subject"], row["task"]))

    scalar_keys = [key for key, value in rows[0].items() if not isinstance(value, list)]
    with (output_dir / "record_qc.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows({key: row[key] for key in scalar_keys} for row in rows)
    with (output_dir / "record_qc.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    code_sha256 = {
        "builder": _sha256(Path(__file__)),
        "cleaner": _sha256(Path(clean_single_trial_eeg.__code__.co_filename)),
    }
    manifest = {
        "schema": CACHE_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "signal_branch": SIMULTANEOUS_EEG_EOG_CLEAN_SCHEMA_V1,
        "cleaning_config": simultaneous_eeg_eog_cleaning_config().to_dict(),
        "record_count": len(rows),
        "subject_count": len(subjects),
        "code_sha256": code_sha256,
        "records": [{
            key: row[key] for key in (
                "join_key", "subject", "task", "source_path", "cache_path",
                "sample_count", "scalp_eeg_channel_count", "artifact_fraction",
            )
        } for row in rows],
    }
    (cache_root / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    figure_paths = _plot(rows, output_dir)
    metrics = {
        key: _quantiles([float(row[key]) for row in rows])
        for key in (
            "artifact_fraction",
            "median_eog_correlation_before",
            "median_eog_correlation_after",
            "near_eye_eog_correlation_before",
            "near_eye_eog_correlation_after",
            "posterior_eog_correlation_before",
            "posterior_eog_correlation_after",
            "median_removed_variance_fraction",
            "maximum_removed_variance_fraction",
            "median_preserved_waveform_correlation",
            "minimum_preserved_waveform_correlation",
            "median_high_frequency_variance_ratio",
        )
    }
    summary = {
        "schema": "simultaneous_eeg_eog_clean_audit_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(rows),
        "subject_count": len(subjects),
        "task_count": len(TASKS),
        "all_outputs_have_28_scalp_channels": all(
            row["scalp_eeg_channel_count"] == 28 for row in rows
        ),
        "all_outputs_exclude_eog": all(not row["output_contains_eog"] for row in rows),
        "all_inputs_retain_heog_veog_as_auxiliary": all(
            set(row["auxiliary_eog_channel_names"]) == {"HEOG", "VEOG"} for row in rows
        ),
        "bad_channel_intervention_count": sum(row["bad_channel_count"] for row in rows),
        "metrics": metrics,
        "figures": figure_paths,
        "cache_manifest": str((cache_root / "cache_manifest.json").relative_to(REPO_ROOT)),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = [
        "# Simultaneous EEG HEOG/VEOG repair audit",
        "",
        f"- Records: **{len(rows)}** ({len(subjects)} subjects x {len(TASKS)} tasks)",
        f"- Output contract: **28 scalp EEG channels**, HEOG/VEOG auxiliary-only and excluded: "
        f"**{summary['all_outputs_exclude_eog']}**",
        "- Intervention boundary: robust low-frequency EOG regression only; bad-channel "
        "interpolation and muscle-band attenuation are disabled in this branch.",
        f"- Median EOG correlation: `{metrics['median_eog_correlation_before']['median']:.4f}` -> "
        f"`{metrics['median_eog_correlation_after']['median']:.4f}`.",
        f"- Near-eye median EOG correlation: `{metrics['near_eye_eog_correlation_before']['median']:.4f}` -> "
        f"`{metrics['near_eye_eog_correlation_after']['median']:.4f}`.",
        f"- Waveform correlation outside detected ocular intervals (median): "
        f"`{metrics['median_preserved_waveform_correlation']['median']:.4f}`.",
        f"- 15-45 Hz variance ratio after/before (median): "
        f"`{metrics['median_high_frequency_variance_ratio']['median']:.4f}`.",
        "",
        "These checks support the software/data preprocessing contract and show measured "
        "artifact reduction with bounded information change. They do not establish that all "
        "ocular activity is removed or that downstream scientific validity improves.",
        "",
        "![repair audit](figures/simultaneous_eog_repair_audit.png)",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
