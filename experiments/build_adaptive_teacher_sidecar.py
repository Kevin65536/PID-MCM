#!/usr/bin/env python3
"""Build the E2 adaptive-teacher patch sidecar from admitted E0 trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.physiology_semantic_targets import (  # noqa: E402
    TARGET_ARRAY_SCHEMA,
    TARGET_SIDECAR_SCHEMA,
    target_sample_key,
)
from src.data.unified_physiology import (  # noqa: E402
    UnifiedPhysiologyWindowDataset,
)


TARGET_FAMILY = "adaptive_multimodal_consensus_proxy"
TARGET_VERSION = "adaptive_ssm_gauge_corrected_patch_v1"
STATE_NAMES = (
    "vasodilation_s",
    "flow_delta",
    "hbo_state",
    "hbr_state",
    "shared_driver",
)
EEG_TARGET_NAMES = (
    "r_mean", "r_slope", "r_logvar", "s_mean", "s_slope", "s_logvar",
)
FNIRS_TARGET_NAMES = (
    "delta_f_mean", "delta_hbo_mean", "delta_hb_mean",
    "delta_f_slope", "delta_hbo_slope", "delta_hb_slope",
    "delta_f_logvar", "delta_hbo_logvar", "delta_hb_logvar",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def patch_targets(
    states: np.ndarray,
    state_std: np.ndarray,
    *,
    patch_size: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool gauge-corrected 10 Hz states onto the registered 2 s grid."""

    states = np.asarray(states, dtype=np.float64)
    state_std = np.asarray(state_std, dtype=np.float64)
    if states.shape != state_std.shape or states.ndim != 2 or states.shape[1] != 5:
        raise ValueError("states/state_std must share shape [time,5]")
    if states.shape[0] % patch_size:
        raise ValueError("teacher trajectory does not align to the two-second patch grid")
    patch = states.reshape(-1, patch_size, 5)
    std_patch = state_std.reshape(-1, patch_size, 5)
    time = np.linspace(-1.0, 1.0, patch_size, dtype=np.float64)
    denominator = max(float(np.dot(time, time)), 1e-12)
    means = np.mean(patch, axis=1)
    slopes = np.einsum("ptc,t->pc", patch, time) / denominator
    mean_variance = np.mean(np.square(std_patch), axis=1)
    slope_weights = np.abs(time / denominator)
    slope_variance = np.square(np.einsum("t,ptc->pc", slope_weights, std_patch))

    eeg = np.zeros((len(patch), 6), dtype=np.float64)
    eeg_uncertainty = np.ones_like(eeg)
    eeg[:, [0, 1, 3, 4]] = np.column_stack(
        (means[:, 4], slopes[:, 4], means[:, 0], slopes[:, 0])
    )
    eeg_uncertainty[:, [0, 1, 3, 4]] = np.column_stack(
        (
            mean_variance[:, 4], slope_variance[:, 4],
            mean_variance[:, 0], slope_variance[:, 0],
        )
    )

    fnirs = np.zeros((len(patch), 9), dtype=np.float64)
    fnirs_uncertainty = np.ones_like(fnirs)
    fnirs[:, :6] = np.column_stack((means[:, 1:4], slopes[:, 1:4]))
    fnirs_uncertainty[:, :6] = np.column_stack(
        (mean_variance[:, 1:4], slope_variance[:, 1:4])
    )
    return eeg, eeg_uncertainty, fnirs, fnirs_uncertainty


def _event_lookup(
    source_config: Mapping[str, Any],
) -> dict[tuple[str, str, int], tuple[str, str, int, UnifiedPhysiologyWindowDataset, int]]:
    data_cfg = source_config["data"]
    lookup: dict[tuple[str, str, int], tuple[str, str, int]] = {}
    for condition in data_cfg["conditions"]:
        dataset = UnifiedPhysiologyWindowDataset(
            cache_root=data_cfg["cache_root"],
            dataset_ids=(condition["dataset_id"],),
            window_duration_s=float(data_cfg["window_duration_s"]),
            window_offset_s=float(data_cfg["window_offset_s"]),
            eeg_signal_branch=str(condition["eeg_signal_branch"]),
        )
        allowed_subjects = {str(value) for value in condition["subjects"]}
        selected: dict[str, list[Any]] = defaultdict(list)
        for dataset_index, ref in enumerate(dataset.windows):
            if ref.record.canonical_subject_id not in allowed_subjects:
                continue
            if ref.record.base_record_id != condition["record_id"]:
                continue
            if str(ref.event.get("label")) != str(condition["target_label"]):
                continue
            subject = str(ref.record.canonical_subject_id)
            if len(selected[subject]) < int(condition["max_trials_per_subject"]):
                selected[subject].append((ref, dataset_index))
        for subject, refs in selected.items():
            for heldout, (ref, dataset_index) in enumerate(refs):
                lookup[(str(condition["condition_id"]), subject, heldout)] = (
                    str(ref.record.dataset_id),
                    str(ref.record.base_record_id),
                    int(ref.event.get("event_index", heldout)),
                    dataset,
                    int(dataset_index),
                )
    return lookup


def _standardize_targets(
    eeg: np.ndarray,
    eeg_uncertainty: np.ndarray,
    fnirs: np.ndarray,
    fnirs_uncertainty: np.ndarray,
    train_samples: np.ndarray,
) -> dict[str, Any]:
    coordinate_indices = {"eeg": (0, 1, 3, 4), "fnirs": (0, 1, 2, 3, 4, 5)}
    payload: dict[str, Any] = {}
    for modality, target, uncertainty in (
        ("eeg", eeg, eeg_uncertainty),
        ("fnirs", fnirs, fnirs_uncertainty),
    ):
        indices = np.asarray(coordinate_indices[modality], dtype=int)
        selected = target[train_samples][..., indices].reshape(-1, len(indices))
        mean = np.mean(selected, axis=0)
        scale = np.maximum(np.std(selected, axis=0), 1e-6)
        target[..., indices] = (target[..., indices] - mean) / scale
        uncertainty[..., indices] = uncertainty[..., indices] / np.square(scale)
        payload[modality] = {
            "coordinate_names": [
                (EEG_TARGET_NAMES if modality == "eeg" else FNIRS_TARGET_NAMES)[index]
                for index in indices
            ],
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "fit_sample_count": int(selected.shape[0]),
        }
    return payload


def build_sidecar(
    source_run: Path,
    e2_config_path: Path,
    output_dir: Path,
    decision_path: Path,
) -> Path:
    source_config_path = source_run / "base_model" / "config.yaml"
    trajectories_path = source_run / "base_model" / "trajectories.csv"
    fits_path = source_run / "base_model" / "fit_parameters.csv"
    target_contract_path = source_run / "target_contract.json"
    for path in (source_config_path, trajectories_path, fits_path, target_contract_path, decision_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    e2_config = yaml.safe_load(e2_config_path.read_text(encoding="utf-8"))
    event_lookup = _event_lookup(source_config)

    trajectory_groups: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(trajectories_path):
        if row["model"] != "adaptive_joint" or row["spatial_mode"] != "local":
            continue
        key = (row["condition_id"], row["subject"], int(row["heldout_trial"]))
        trajectory_groups[key].append(row)
    fit_lookup = {
        (row["condition_id"], row["subject"], int(row["heldout_trial"])): row
        for row in _read_csv(fits_path)
        if row["spatial_mode"] == "local"
    }
    if set(trajectory_groups) != set(fit_lookup):
        raise RuntimeError("Adaptive trajectory and fitted-channel fold identities differ")

    records: list[dict[str, Any]] = []
    excluded_channel_rows: list[dict[str, Any]] = []
    source_subjects: set[str] = set()
    for fold_key in sorted(trajectory_groups):
        rows = sorted(trajectory_groups[fold_key], key=lambda row: float(row["time_s"]))
        if len(rows) != 200:
            raise ValueError(f"Expected 200 target points for {fold_key}, got {len(rows)}")
        states = np.column_stack([
            np.asarray([float(row[f"target_{name}"]) for row in rows])
            for name in STATE_NAMES
        ])
        state_std = np.column_stack([
            np.asarray([float(row[f"target_{name}_std"]) for row in rows])
            for name in STATE_NAMES
        ])
        eeg, eeg_uncertainty, fnirs, fnirs_uncertainty = patch_targets(states, state_std)
        dataset_id, record_id, event_index, measured_dataset, measured_index = event_lookup[fold_key]
        _, subject, _ = fold_key
        fit = fit_lookup[fold_key]
        subject_key = f"{dataset_id}|{subject}"
        source_subjects.add(subject_key)
        selected_eeg_channels = fit["selected_eeg_channels"].split("|")
        selected_fnirs_channels = fit["selected_fnirs_channels"].split("|")
        measured = measured_dataset[measured_index]
        eeg_lookup = {str(name): index for index, name in enumerate(measured["channel_names"]["eeg"])}
        fnirs_lookup = {str(name): index for index, name in enumerate(measured["channel_names"]["fnirs"])}
        missing_eeg = [name for name in selected_eeg_channels if name not in eeg_lookup]
        missing_fnirs = [name for name in selected_fnirs_channels if name not in fnirs_lookup]
        bad_eeg = [] if missing_eeg else [
            name for name in selected_eeg_channels
            if bool(np.asarray(measured["bad_channel_mask"]["eeg"], dtype=bool)[eeg_lookup[name]])
        ]
        bad_fnirs = [] if missing_fnirs else [
            name for name in selected_fnirs_channels
            if bool(np.asarray(measured["bad_channel_mask"]["fnirs"], dtype=bool)[fnirs_lookup[name]])
        ]
        if missing_eeg or missing_fnirs or bad_eeg or bad_fnirs:
            excluded_channel_rows.append({
                "sample_key": target_sample_key(dataset_id, subject, record_id, event_index),
                "subject_key": subject_key,
                "missing_eeg_channels": missing_eeg,
                "missing_fnirs_channels": missing_fnirs,
                "bad_eeg_channels": bad_eeg,
                "bad_fnirs_channels": bad_fnirs,
                "reason": "source_teacher_view_not_admitted_by_current_measured_channel_contract",
            })
            continue
        records.append({
            "sample_key": target_sample_key(dataset_id, subject, record_id, event_index),
            "subject_key": subject_key,
            "eeg": eeg,
            "eeg_uncertainty": eeg_uncertainty,
            "fnirs": fnirs,
            "fnirs_uncertainty": fnirs_uncertainty,
            "selected_eeg_channels": selected_eeg_channels,
            "selected_fnirs_channels": selected_fnirs_channels,
        })

    split_cfg = e2_config["data"]["split"]
    train_subjects = {str(value) for value in split_cfg["train_subject_keys"]}
    validation_subjects = {str(value) for value in split_cfg["val_subject_keys"]}
    protected_subjects = {str(value) for value in split_cfg["test_subject_keys"]}
    admitted_subjects = {record["subject_key"] for record in records}
    if source_subjects & protected_subjects:
        raise RuntimeError("Source trajectories unexpectedly contain protected-test subjects")
    expected_development = train_subjects | validation_subjects
    if source_subjects != expected_development:
        raise RuntimeError(
            f"Target sidecar subject coverage differs from E2 development split: "
            f"missing={sorted(expected_development - source_subjects)}, "
            f"unexpected={sorted(source_subjects - expected_development)}"
        )
    if not records:
        raise RuntimeError("No adaptive targets survive the measured-channel admission audit")

    eeg = np.stack([record["eeg"] for record in records])
    eeg_uncertainty = np.stack([record["eeg_uncertainty"] for record in records])
    fnirs = np.stack([record["fnirs"] for record in records])
    fnirs_uncertainty = np.stack([record["fnirs_uncertainty"] for record in records])
    train_samples = np.asarray([record["subject_key"] in train_subjects for record in records])
    standardization = _standardize_targets(
        eeg, eeg_uncertainty, fnirs, fnirs_uncertainty, train_samples
    )
    valid = np.ones(eeg.shape[:2], dtype=bool)
    invalid = np.zeros_like(valid)
    sample_keys = [record["sample_key"] for record in records]

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "targets.npz"
    np.savez_compressed(
        arrays_path,
        schema=np.asarray(TARGET_ARRAY_SCHEMA),
        sample_key=np.asarray(sample_keys, dtype=np.str_),
        selected_eeg_channels=np.asarray(
            [record["selected_eeg_channels"] for record in records], dtype=np.str_
        ),
        selected_fnirs_channels=np.asarray(
            [record["selected_fnirs_channels"] for record in records], dtype=np.str_
        ),
        eeg_target=eeg.astype(np.float32),
        eeg_uncertainty=eeg_uncertainty.astype(np.float32),
        fnirs_target=fnirs.astype(np.float32),
        fnirs_uncertainty=fnirs_uncertainty.astype(np.float32),
        eeg_local_valid_mask=valid,
        eeg_prototype_valid_mask=valid,
        eeg_context_valid_mask=invalid,
        eeg_coupling_valid_mask=invalid,
        fnirs_local_valid_mask=valid,
        fnirs_prototype_valid_mask=valid,
        fnirs_context_valid_mask=invalid,
        fnirs_coupling_valid_mask=invalid,
    )
    sample_order_sha = hashlib.sha256("\n".join(sample_keys).encode("utf-8")).hexdigest()
    manifest = {
        "schema": TARGET_SIDECAR_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_family": TARGET_FAMILY,
        "target_version": TARGET_VERSION,
        "target_identity": "physiology_shaped_multimodal_consensus_proxy",
        "arrays_file": arrays_path.name,
        "arrays_sha256": _sha256(arrays_path),
        "sample_count": len(records),
        "sample_order_sha256": sample_order_sha,
        "patch_duration_s": 2.0,
        "token_count": int(eeg.shape[1]),
        "eeg_target_names": list(EEG_TARGET_NAMES),
        "fnirs_target_names": list(FNIRS_TARGET_NAMES),
        "standardization": {
            "policy": "coordinatewise_train_subject_only_v1",
            **standardization,
        },
        "source": {
            "run": str(source_run),
            "trajectory_sha256": _sha256(trajectories_path),
            "fit_parameters_sha256": _sha256(fits_path),
            "target_contract_sha256": _sha256(target_contract_path),
            "architecture_decision": str(decision_path),
            "architecture_decision_sha256": _sha256(decision_path),
        },
        "source_development_subject_keys": sorted(source_subjects),
        "admitted_development_subject_keys": sorted(admitted_subjects),
        "protected_subject_keys": sorted(protected_subjects),
        "protected_test_included": False,
        "entry_support": {
            "local": True,
            "prototype": True,
            "context": False,
            "coupling": False,
        },
        "uncertainty_weighting_admitted": False,
        "measured_channel_audit": {
            "policy": "exclude_target_when_source_teacher_view_uses_missing_or_bad_measured_channel",
            "source_sample_count": len(records) + len(excluded_channel_rows),
            "admitted_sample_count": len(records),
            "excluded_sample_count": len(excluded_channel_rows),
            "admitted_fraction": len(records) / (len(records) + len(excluded_channel_rows)),
            "excluded_samples": excluded_channel_rows,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    split_sha = hashlib.sha256(
        json.dumps(split_cfg, sort_keys=True).encode("utf-8")
    ).hexdigest()
    gate = {
        "schema": "physiology_semantic_target_family_gate_v1",
        "gate": "E0_OPTIONAL_TARGET_FAMILY_DEVELOPMENT",
        "status": "development_passed_protected_test_closed",
        "target_family_development_passed": True,
        "target_family": TARGET_FAMILY,
        "target_version": TARGET_VERSION,
        "data_contract": e2_config["data"]["contract"],
        "cache_root": e2_config["data"]["cache_root"],
        "split_sha256": split_sha,
        "sidecar_manifest_sha256": _sha256(manifest_path),
        "admissible_coordinates_by_entry": {
            "eeg": {
                "local": ["r_mean", "r_slope", "s_mean", "s_slope"],
                "prototype": ["r_mean", "r_slope", "s_mean", "s_slope"],
                "context": [],
                "coupling": [],
            },
            "fnirs": {
                "local": [
                    "delta_hbo_mean", "delta_hb_mean",
                    "delta_hbo_slope", "delta_hb_slope",
                ],
                "prototype": [
                    "delta_hbo_mean", "delta_hb_mean",
                    "delta_hbo_slope", "delta_hb_slope",
                ],
                "context": [],
                "coupling": [],
            },
        },
        "required_coordinates": {
            "eeg": ["r_mean", "r_slope"],
            "fnirs": [
                "delta_hbo_mean", "delta_hb_mean",
                "delta_hbo_slope", "delta_hb_slope",
            ],
        },
        "optional_coordinates": {"eeg": ["s_mean", "s_slope"], "fnirs": []},
        "uncertainty_weighting_admitted": False,
        "measured_channel_admitted_sample_count": len(records),
        "measured_channel_excluded_sample_count": len(excluded_channel_rows),
        "requires_e0_channel_aware_revalidation_before_formal_e2": bool(excluded_channel_rows),
        "promotion_scope": "development_only",
        "protected_test_opened": False,
    }
    (output_dir / "target_family_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--e2-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--decision",
        default="docs/physiology_semantic_tokenizer/analysis/20260719_PHYSICAL_TEACHER_GRADIENT_ENTRY_DECISION.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_sidecar(
        Path(args.source_run).resolve(),
        Path(args.e2_config).resolve(),
        Path(args.output_dir).resolve(),
        Path(args.decision).resolve(),
    )
    print(json.dumps({"output_dir": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
