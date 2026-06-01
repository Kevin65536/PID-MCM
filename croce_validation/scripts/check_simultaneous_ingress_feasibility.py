#!/usr/bin/env python3
"""Inspect Simultaneous EEG&NIRS feasibility for Croce cache ingress.

This script answers three questions for the current local copy of the
Simultaneous EEG&NIRS dataset:

1. How do EEG and fNIRS markers align across tasks?
2. Which tasks support Single-Trial-like multimodal sample construction?
3. Does the current local copy still contain direct vendor-format optical
   wavelength files, or only MATLAB oxy/deoxy exports?

Outputs:
  - summary.json
  - summary.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.eeg_fnirs_dataset import load_mat_struct
from src.data.simultaneous_eeg_nirs_dataset import (  # noqa: E402
    SimultaneousCognitiveLoader,
    SimultaneousMultiModalDataset,
)


SUPPORTED_TASKS = ("nback", "wg", "dsr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Simultaneous EEG&NIRS feasibility for Croce cache ingress.",
    )
    parser.add_argument(
        "--data-root",
        default="data/Simultaneous EEG&NIRS",
        help="Path to the Simultaneous EEG&NIRS dataset root.",
    )
    parser.add_argument("--subject-id", type=int, default=1, help="Subject id to inspect.")
    parser.add_argument(
        "--tasks",
        default=",".join(SUPPORTED_TASKS),
        help="Comma-separated task list. Supported: nback,wg,dsr",
    )
    parser.add_argument(
        "--window-duration-s",
        type=float,
        default=10.0,
        help="Window duration used for multimodal sample feasibility checks.",
    )
    parser.add_argument("--output-dir", default="", help="Optional explicit output directory.")
    return parser.parse_args()


def parse_tasks(spec: str) -> List[str]:
    tasks = [item.strip().lower() for item in str(spec).split(",") if item.strip()]
    if not tasks:
        return list(SUPPORTED_TASKS)

    invalid = [task for task in tasks if task not in SUPPORTED_TASKS]
    if invalid:
        raise ValueError(f"Unsupported tasks: {invalid}")
    return tasks


def resolve_output_dir(output_dir: str) -> Path:
    if output_dir:
        resolved = Path(output_dir)
        if not resolved.is_absolute():
            resolved = PROJECT_ROOT / resolved
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved = PROJECT_ROOT / "croce_validation" / "results" / f"simultaneous_ingress_feasibility_{timestamp}"
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def count_recursive_files(root: Path, patterns: Mapping[str, str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for key, pattern in patterns.items():
        counts[key] = sum(1 for _ in root.rglob(pattern))
    return counts


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def load_cnt_struct(data_root: Path, subject_id: int, modality: str, task: str) -> Any:
    suffix = "EEG" if modality == "eeg" else "NIRS"
    path = data_root / f"VP{subject_id:03d}-{suffix}" / f"cnt_{task}.mat"
    mat = load_mat_struct(str(path))
    key = next(name for name in mat if not name.startswith("__"))
    return mat[key]


def summarize_marker_info(marker_info: Mapping[str, Any]) -> Dict[str, Any]:
    event_desc = marker_info.get("event_desc")
    counts: List[Dict[str, Any]] = []
    if event_desc is not None:
        counter = Counter(int(value) for value in np.asarray(event_desc).reshape(-1).tolist())
        counts = [
            {"event_desc": int(event_desc_value), "count": int(counter[event_desc_value])}
            for event_desc_value in sorted(counter)
        ]

    return {
        "num_events": int(len(marker_info.get("time", []))),
        "class_names": [str(name) for name in marker_info.get("className", [])],
        "event_desc_counts": counts,
        "first_event_times_ms": [
            float(value) for value in np.asarray(marker_info.get("time", []), dtype=np.float64)[:10].tolist()
        ],
    }


def summarize_multimodal_samples(
    data_root: str,
    subject_id: int,
    task: str,
    window_duration_s: float,
) -> Dict[str, Any]:
    if task == "dsr":
        return {
            "built": False,
            "task_supports_single_trial_like_multimodal": False,
            "note": "DSR is deprecated in this repository and is excluded from training-ready multimodal loading.",
        }

    dataset = SimultaneousMultiModalDataset(
        data_root=data_root,
        subject_ids=[subject_id],
        task=task,
        window_duration_s=window_duration_s,
        normalize=False,
        normalization_mode="none",
        eeg_preprocessing={"bandpass": [0.5, 45.0]},
        fnirs_preprocessing={"lowpass": 0.2},
        exclude_eog=True,
        hbo_only=True,
        hbr_only=False,
        fnirs_signal="oxy",
        segmentation_mode="auto",
    )

    summary: Dict[str, Any] = {
        "built": True,
        "segmentation_mode": str(dataset.segmentation_mode),
        "num_samples": int(len(dataset)),
        "num_eeg_channels": int(dataset.get_num_eeg_channels()),
        "num_fnirs_channels": int(dataset.get_num_fnirs_channels()),
        "eeg_sample_rate_hz": float(dataset.get_eeg_sample_rate()),
        "fnirs_sample_rate_hz": float(dataset.get_fnirs_sample_rate()),
        "task_supports_single_trial_like_multimodal": bool(dataset.segmentation_mode == "trial"),
    }

    if len(dataset) > 0:
        sample = dataset[0]
        summary.update(
            {
                "first_sample_eeg_shape": list(sample["eeg"].shape),
                "first_sample_fnirs_shape": list(sample["fnirs"].shape),
                "first_sample_label": int(sample["label"].item()),
            }
        )

    if dataset.segmentation_mode == "trial":
        summary["note"] = "Current loader can build trial-level paired EEG-fNIRS windows for this task."
    else:
        summary["note"] = "Current loader can only build session-level paired EEG-fNIRS windows for this task, not Single-Trial-style trial windows."
    return summary


def summarize_task(
    data_root: Path,
    subject_id: int,
    task: str,
    window_duration_s: float,
) -> Dict[str, Any]:
    eeg_loader = SimultaneousCognitiveLoader(
        str(data_root),
        task=task,
        subject_ids=[subject_id],
        modality="eeg",
        allow_deprecated=True,
    )
    fnirs_loader = SimultaneousCognitiveLoader(
        str(data_root),
        task=task,
        subject_ids=[subject_id],
        modality="fnirs",
        fnirs_signal="oxy",
        allow_deprecated=True,
    )
    both_loader = SimultaneousCognitiveLoader(
        str(data_root),
        task=task,
        subject_ids=[subject_id],
        modality="both",
        allow_deprecated=True,
    )

    eeg_data, eeg_markers, eeg_info = eeg_loader.load_subject_data(subject_id, "eeg")
    fnirs_data, fnirs_markers, fnirs_info = fnirs_loader.load_subject_data(subject_id, "fnirs")
    cnt_struct = load_cnt_struct(data_root, subject_id, "fnirs", task)

    session_alignment = both_loader.align_session_markers(subject_id)
    multimodal_summary = summarize_multimodal_samples(str(data_root), subject_id, task, window_duration_s)

    return {
        "task": task,
        "eeg": {
            "shape": list(np.asarray(eeg_data).shape),
            "sample_rate_hz": float(eeg_info["fs"]),
            "num_channels": int(np.asarray(eeg_data).shape[1]),
            "first_channels": [str(name) for name in eeg_info["clab"][:5]],
            "markers": summarize_marker_info(eeg_markers),
        },
        "fnirs": {
            "oxy_shape": list(np.asarray(fnirs_data).shape),
            "sample_rate_hz": float(fnirs_info["fs"]),
            "num_channels": int(np.asarray(fnirs_data).shape[1]),
            "channel_examples": [str(name) for name in fnirs_info["clab"][:5]],
            "available_signals": ["oxy", "deoxy"],
            "units": {
                "oxy": str(getattr(cnt_struct.oxy, "yUnit", "unknown")),
                "deoxy": str(getattr(cnt_struct.deoxy, "yUnit", "unknown")),
            },
            "markers": summarize_marker_info(fnirs_markers),
        },
        "session_alignment": to_builtin(session_alignment),
        "multimodal_sample_feasibility": multimodal_summary,
    }


def build_overall_assessment(raw_file_counts: Mapping[str, int], task_summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    has_vendor_raw = bool(
        raw_file_counts.get("brainvision_vhdr", 0)
        or raw_file_counts.get("brainvision_vmrk", 0)
        or raw_file_counts.get("brainvision_eeg", 0)
        or raw_file_counts.get("nirx_wl1", 0)
        or raw_file_counts.get("nirx_wl2", 0)
        or raw_file_counts.get("nirx_hdr", 0)
    )

    wg_summary = next((summary for summary in task_summaries if summary["task"] == "wg"), None)
    nback_summary = next((summary for summary in task_summaries if summary["task"] == "nback"), None)

    return {
        "current_copy_has_vendor_specific_raw_files": has_vendor_raw,
        "requires_forward_projection_to_optical_space": not has_vendor_raw,
        "current_copy_can_directly_emit_760_850_optical_cache": bool(has_vendor_raw),
        "best_first_task_for_single_trial_like_multimodal_cache": (
            "wg"
            if wg_summary is not None and wg_summary["multimodal_sample_feasibility"].get("task_supports_single_trial_like_multimodal", False)
            else None
        ),
        "best_first_task_for_session_level_smoke_cache": (
            "nback" if nback_summary is not None else None
        ),
        "notes": [
            "Current local Simultaneous copy stores MATLAB oxy/deoxy exports and task/session markers.",
            "If raw NIRx wl1/wl2 files are unavailable locally, wavelength-aligned cache generation must use an explicit forward projection from oxy/deoxy into the 760/850 optical contract.",
            "WG is the strongest candidate for Single-Trial-like multimodal cache experiments because EEG and fNIRS both expose 60 aligned task markers in the current MATLAB export.",
            "n-back aligns cleanly at the session level but does not expose trial-level fNIRS markers in the current MATLAB export.",
        ],
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Simultaneous EEG&NIRS Ingress Feasibility")
    lines.append("")
    lines.append(f"- Subject: {summary['subject_id']}")
    lines.append(f"- Data root: {summary['data_root']}")
    lines.append("")
    lines.append("## Raw File Availability")
    lines.append("")
    for key, value in summary["raw_file_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Overall Assessment")
    lines.append("")
    assessment = summary["overall_assessment"]
    lines.append(f"- Current copy has vendor-specific raw files: {assessment['current_copy_has_vendor_specific_raw_files']}")
    lines.append(f"- Direct 760/850 optical cache possible from current copy: {assessment['current_copy_can_directly_emit_760_850_optical_cache']}")
    lines.append(f"- Requires forward projection to optical space: {assessment['requires_forward_projection_to_optical_space']}")
    lines.append(f"- Best first task for Single-Trial-like multimodal cache: {assessment['best_first_task_for_single_trial_like_multimodal_cache']}")
    lines.append(f"- Best first task for session-level smoke cache: {assessment['best_first_task_for_session_level_smoke_cache']}")
    for note in assessment["notes"]:
        lines.append(f"- Note: {note}")
    lines.append("")

    for task_summary in summary["tasks"]:
        lines.append(f"## Task {task_summary['task']}")
        lines.append("")
        eeg_summary = task_summary["eeg"]
        fnirs_summary = task_summary["fnirs"]
        sample_summary = task_summary["multimodal_sample_feasibility"]
        align_summary = task_summary["session_alignment"]
        lines.append(f"- EEG shape: {eeg_summary['shape']} @ {eeg_summary['sample_rate_hz']} Hz")
        lines.append(f"- fNIRS oxy shape: {fnirs_summary['oxy_shape']} @ {fnirs_summary['sample_rate_hz']} Hz")
        lines.append(f"- fNIRS units: oxy={fnirs_summary['units']['oxy']}, deoxy={fnirs_summary['units']['deoxy']}")
        lines.append(f"- EEG marker count: {eeg_summary['markers']['num_events']}")
        lines.append(f"- fNIRS marker count: {fnirs_summary['markers']['num_events']}")
        lines.append(f"- Session labels match: {align_summary['label_sequence_match']}")
        lines.append(f"- Offset pattern: {align_summary['offset_pattern']['case']}")
        lines.append(f"- Stable offset blocks: {align_summary['offset_pattern']['num_blocks']}")
        lines.append(f"- Multimodal segmentation mode: {sample_summary.get('segmentation_mode', 'n/a')}")
        lines.append(f"- Multimodal samples built: {sample_summary.get('num_samples', 0)}")
        lines.append(f"- Single-Trial-like multimodal supported: {sample_summary['task_supports_single_trial_like_multimodal']}")
        lines.append(f"- Note: {sample_summary['note']}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    tasks = parse_tasks(args.tasks)
    output_dir = resolve_output_dir(args.output_dir)

    raw_file_counts = count_recursive_files(
        data_root,
        {
            "brainvision_vhdr": "*.vhdr",
            "brainvision_vmrk": "*.vmrk",
            "brainvision_eeg": "*.eeg",
            "nirx_wl1": "*.wl1",
            "nirx_wl2": "*.wl2",
            "nirx_hdr": "*.hdr",
            "matlab_cnt": "cnt_*.mat",
            "matlab_mrk": "mrk_*.mat",
            "matlab_mnt": "mnt_*.mat",
        },
    )

    task_summaries = [
        summarize_task(data_root, int(args.subject_id), task, float(args.window_duration_s))
        for task in tasks
    ]
    summary = {
        "generated_at": datetime.now().isoformat(),
        "subject_id": int(args.subject_id),
        "data_root": str(data_root),
        "raw_file_counts": raw_file_counts,
        "tasks": task_summaries,
        "overall_assessment": build_overall_assessment(raw_file_counts, task_summaries),
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(to_builtin(summary), indent=2, ensure_ascii=True) + "\n")
    markdown_path = output_dir / "summary.md"
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")

    print(f"[SimultaneousIngress] Saved summary: {summary_path}")
    print(f"[SimultaneousIngress] Saved markdown: {markdown_path}")
    print(json.dumps(to_builtin(summary["overall_assessment"]), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()