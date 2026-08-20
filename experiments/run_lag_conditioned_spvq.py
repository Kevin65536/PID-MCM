#!/usr/bin/env python3
"""Run the exploratory B0/M1/N1 LC-SPVQ development generation.

The executable owns the reviewed data/split/preparation contract. Model training is
dispatched through reusable LC-SPVQ components. Only measured smoke training is
currently authorized; full training fails closed until fit-selection-only lag-weight
and checkpoint selection are implemented and reviewed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.lag_conditioned_native_features import (
    MaskedStandardizer,
    NativeFeatureTargets,
    apply_masked_standardizer,
    extract_eeg_native_targets,
    extract_fnirs_native_targets,
    fit_masked_standardizer,
)
from src.data.lag_conditioned_dataset import (
    CANONICAL_PROTECTED_SUBJECTS,
    TASK_SPECS,
    LagConditionedSampleIndex,
    LagConditionedTaskDataset,
    make_group_derangement,
)
from src.data.unified_physiology import UnifiedPhysiologyWindowDataset
from src.losses.lag_conditioned import (
    native_feature_prediction_loss,
    raw_patch_reconstruction_loss,
    weighted_pretraining_loss,
)
from src.metrics.codebook_health import compute_codebook_health
from src.metrics.lag_conditioned_downstream import evaluate_logit_ablations
from src.tokenizers.lag_conditioned_baseline import B0ContinuousSharedPrivate
from src.tokenizers.lag_conditioned_shared_private_vq import (
    LCSPVQModel,
    LagAwareContinuousMatchingLoss,
    RawFeatureDecoder,
)


SCHEMA = "lag_conditioned_spvq_suite_v1"
PREPARATION_SCHEMA = "lag_conditioned_spvq_preparation_v1"
VARIANT_ORDER = ("B0", "M1", "N1")
FIRST_ROUND_TASKS = ("motor_imagery", "word_generation")
POSITIVE_LAGS_SECONDS = (0, 2, 4, 6, 8, 10)
CANONICAL_SUBJECT_SPLITS: Mapping[str, Mapping[str, tuple[str, ...]]] = (
    MappingProxyType(
        {
            "fit_parameter_subjects": MappingProxyType(
                {
                    "eeg_fnirs_single_trial": tuple(
                        f"subject_{index:02d}" for index in range(1, 16)
                    ),
                    "simultaneous_eeg_nirs": tuple(
                        f"VP{index:03d}" for index in range(1, 16)
                    ),
                }
            ),
            "fit_selection_subjects": MappingProxyType(
                {
                    "eeg_fnirs_single_trial": tuple(
                        f"subject_{index:02d}" for index in range(16, 19)
                    ),
                    "simultaneous_eeg_nirs": tuple(
                        f"VP{index:03d}" for index in range(16, 19)
                    ),
                }
            ),
            "development_apply_subjects": MappingProxyType(
                {
                    "eeg_fnirs_single_trial": tuple(
                        f"subject_{index:02d}" for index in range(19, 24)
                    ),
                    "simultaneous_eeg_nirs": tuple(
                        f"VP{index:03d}" for index in range(19, 24)
                    ),
                }
            ),
            "protected_or_unused": CANONICAL_PROTECTED_SUBJECTS,
        }
    )
)
_PREPARATION_CAPABILITY = object()


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
        json.dumps(
            _jsonable(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = list(rows)
    if not values:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in values:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [{key: _jsonable(row.get(key, "")) for key in fields} for row in values]
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_payload() -> dict[str, str]:
    def call(*args: str) -> str:
        result = subprocess.run(
            args,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    return {
        "commit": call("git", "rev-parse", "HEAD"),
        "status_short": call("git", "status", "--short"),
    }


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def _assert_disjoint_subject_splits(config: Mapping[str, Any]) -> None:
    split = config["data_split"]
    roles = (
        "fit_parameter_subjects",
        "fit_selection_subjects",
        "development_apply_subjects",
        "protected_or_unused",
    )
    for dataset_id in ("eeg_fnirs_single_trial", "simultaneous_eeg_nirs"):
        resolved = {
            role: {str(value) for value in split[role][dataset_id]} for role in roles
        }
        if any(not values for values in resolved.values()):
            raise ValueError(f"{dataset_id} contains an empty subject split")
        required_protected = set(CANONICAL_PROTECTED_SUBJECTS[dataset_id])
        if resolved["protected_or_unused"] != required_protected:
            raise PermissionError(
                f"{dataset_id} protected_or_unused must equal the canonical closed set "
                f"{sorted(required_protected)}"
            )
        for first_index, first in enumerate(roles):
            for second in roles[first_index + 1 :]:
                overlap = resolved[first] & resolved[second]
                if overlap:
                    raise PermissionError(
                        f"{dataset_id} subject overlap {first}/{second}: {sorted(overlap)}"
                    )
        for role in roles[:-1]:
            required = set(CANONICAL_SUBJECT_SPLITS[role][dataset_id])
            if resolved[role] != required:
                raise PermissionError(
                    f"{dataset_id} {role} must equal the canonical reviewed split "
                    f"{sorted(required)}"
                )


def validate_config(config: Mapping[str, Any]) -> None:
    """Fail closed when the reviewed first-generation contract drifts."""

    if config["experiment"].get("schema") != "lag_conditioned_spvq_config_v1":
        raise ValueError("LC-SPVQ configuration schema mismatch")
    if config["experiment"].get("analysis_intent") != "exploratory":
        raise ValueError("first LC-SPVQ generation must remain exploratory")
    if config["experiment"].get("old_continuous_2_of_16_verdict_mutable") is not False:
        raise ValueError("the completed continuous 2/16 verdict is immutable")
    if config["source"].get("protected_open") is not False:
        raise PermissionError("LC-SPVQ requires protected_open=false")
    if config["output"].get("protected_open") is not False:
        raise PermissionError("LC-SPVQ output contract requires protected_open=false")
    if config["source"].get("artifact_mask_is_validity") is not False:
        raise ValueError("artifact annotations cannot become the validity mask")
    if config["data_split"].get("development_is_new_independent_holdout") is not False:
        raise ValueError("reused development subjects cannot be called independent")
    _assert_disjoint_subject_splits(config)

    source = config["source"]
    canonical_cache = (REPO_ROOT / "data/cache/physiology_semantic_clean_v1").resolve()
    if _resolve(source["cache_root"]).resolve() != canonical_cache:
        raise PermissionError("LC-SPVQ requires the reviewed clean-cache root")
    if source.get("eeg_signal_branch") != "single_trial_eeg_artifact_clean_v4":
        raise ValueError("LC-SPVQ requires the reviewed EEG artifact-clean branch")
    if float(source["window_duration_s"]) != 20.0:
        raise ValueError("LC-SPVQ requires 20-second windows")
    if float(source["window_offset_s"]) != -5.0:
        raise ValueError("LC-SPVQ requires the registered -5-second window offset")
    if float(source["patch_duration_s"]) != 2.0:
        raise ValueError("LC-SPVQ requires 2-second patches")
    if float(source["eeg_sample_rate_hz"]) != 200.0:
        raise ValueError("LC-SPVQ requires 200 Hz EEG")
    if float(source["fnirs_sample_rate_hz"]) != 10.0:
        raise ValueError("LC-SPVQ requires 10 Hz fNIRS")
    if source.get("support_policy") != "recorded_and_analysis_valid_intersection":
        raise ValueError("LC-SPVQ support policy drifted")
    if tuple(map(int, config["objective"]["lag_seconds"])) != POSITIVE_LAGS_SECONDS:
        raise ValueError("positive lag bank drifted")
    if config["objective"].get("hard_negative") != (
        "same_subject_condition_nonidentity_same_token_time"
    ):
        raise ValueError("registered hard-negative policy drifted")
    if config["objective"].get("ssm_target_primary_loss") is not False:
        raise ValueError("SSM target must not be a primary LC-SPVQ loss")
    if config["objective"].get("token_cooccurrence_loss") is not False:
        raise ValueError("held-out coupling maps cannot be optimized directly")
    if config["objective"].get("raw_shared_gradient") != "stopped":
        raise ValueError("raw reconstruction must stop at the shared branch")

    task_rows = {str(row["task_id"]): row for row in config["tasks"]}
    if set(task_rows) != set(TASK_SPECS):
        raise ValueError("LC-SPVQ task set differs from the four-task contract")
    for task_id, spec in TASK_SPECS.items():
        row = task_rows[task_id]
        expected = {
            "dataset_id": spec.dataset_id,
            "namespace": spec.namespace,
            "class_names": list(spec.class_names),
            "record_ids": list(spec.record_ids),
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(f"{task_id} requires {key}={value!r}")
    first_round = tuple(row["task_id"] for row in config["tasks"] if row["first_round"])
    if first_round != FIRST_ROUND_TASKS:
        raise ValueError("first-round task order must be motor imagery then word generation")

    if tuple(config["variants"]["first_round_order"]) != VARIANT_ORDER:
        raise ValueError("first-round variant order must be B0/M1/N1")
    for variant in VARIANT_ORDER:
        if variant not in config["variants"]:
            raise ValueError(f"missing variant {variant}")
    if config["variants"]["B0"].get("vector_quantization") is not False:
        raise ValueError("B0 must remain continuous")
    for variant in ("M1", "N1"):
        if config["variants"][variant].get("vector_quantization") is not True:
            raise ValueError(f"{variant} requires VQ")
        if config["variants"][variant].get("lag_objective") is not True:
            raise ValueError(f"{variant} requires the lag objective")
    for variant in ("B0", "M1"):
        if config["variants"][variant].get("positive_pairing") != "matched":
            raise ValueError(f"{variant} must use matched positive pairs")
    if config["variants"]["N1"].get("positive_pairing") != "same_subject_condition_deranged":
        raise ValueError("N1 must use the registered trial derangement")

    quantizer = config["quantizer"]
    if int(quantizer["eeg_codebook_size"]) != 16 or int(
        quantizer["fnirs_codebook_size"]
    ) != 16:
        raise ValueError("primary LC-SPVQ codebooks must both use K=16")
    if quantizer.get("independent_codebooks") is not True:
        raise ValueError("EEG and fNIRS codebooks must be independent")
    if int(quantizer["embedding_dim"]) != int(config["model"]["shared_dim"]):
        raise ValueError("codebook and shared dimensions differ")
    if int(config["head"]["coupling_rank"]) != 8:
        raise ValueError("primary coupling-head rank must remain 8")
    if config["head"].get("shared_encoder_frozen") is not True:
        raise ValueError("shared encoders must be frozen during task-head fitting")
    if config["head"].get("codebook_frozen") is not True:
        raise ValueError("codebooks must be frozen during task-head fitting")
    if tuple(map(int, config["head"]["lag_seconds"])) != POSITIVE_LAGS_SECONDS:
        raise ValueError("coupling-head lag bank drifted")
    if float(config["head"]["ablation_auxiliary_cross_entropy_weight"]) < 0.0:
        raise ValueError("ablation auxiliary loss weight must be non-negative")
    if int(config["model"]["eeg_shared_history_tokens"]) != 1:
        raise ValueError("EEG shared history must be one preceding patch")
    if int(config["model"]["fnirs_shared_history_tokens"]) != 2:
        raise ValueError("fNIRS shared history must be two preceding patches")
    if quantizer.get("initialization") != "fit_parameter_continuous_latents_kmeans":
        raise ValueError("VQ initialization must use all fit-parameter continuous latents")
    if tuple(map(str, config["smoke"]["tasks"])) != FIRST_ROUND_TASKS:
        raise ValueError("smoke tasks must exercise both first-round tasks")
    if tuple(map(str, config["smoke"]["variants"])) != VARIANT_ORDER:
        raise ValueError("smoke variants must exercise B0/M1/N1")
    statistics = config["statistics"]
    if tuple(map(str, statistics["q0_controls"])) != (
        "fnirs_token_history",
        "relative_token_time",
        "task_condition",
    ):
        raise ValueError("q0 control contract drifted")
    if float(statistics["q0_q1_probe_l2"]) != 1.0:
        raise ValueError("q0/q1 probe L2 contract drifted")
    if float(statistics["q0_q1_train_target_label_smoothing"]) != 0.05:
        raise ValueError("q0/q1 train-target smoothing contract drifted")
    if float(statistics["q0_q1_evaluation_target_smoothing"]) != 0.0:
        raise ValueError("q0/q1 evaluation targets must remain unsmoothed")
    if int(statistics["q0_q1_max_iter"]) != 5000:
        raise ValueError("q0/q1 convergence iteration cap drifted")
    if float(statistics["q0_q1_tolerance"]) != 1e-6:
        raise ValueError("q0/q1 convergence tolerance drifted")
    if statistics.get("q0_q1_convergence_required") is not True:
        raise ValueError("q0/q1 fits must fail closed on non-convergence")


def _validate_bound_config(config: Mapping[str, Any], config_path: Path) -> Path:
    """Bind direct runner APIs to a validated, repository-local YAML source."""
    validate_config(config)
    path = Path(config_path).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise PermissionError("LC-SPVQ config must be repository-local") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    left = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":"))
    right = json.dumps(_jsonable(loaded), sort_keys=True, separators=(",", ":"))
    if left != right:
        raise ValueError("in-memory config differs from the bound config file")
    return path


def _require_smoke_training(smoke: bool) -> None:
    if not bool(smoke):
        raise RuntimeError(
            "full training is fail-closed until fit-selection-only lag-weight and "
            "checkpoint selection are implemented and reviewed"
        )


@dataclass(frozen=True)
class ChannelStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    count: np.ndarray

    def __post_init__(self) -> None:
        if self.mean.ndim != 1 or self.scale.shape != self.mean.shape:
            raise ValueError("channel standardizer must contain matching vectors")
        if self.count.shape != self.mean.shape:
            raise ValueError("channel support count shape mismatch")
        if np.any(~np.isfinite(self.mean)) or np.any(~np.isfinite(self.scale)):
            raise ValueError("channel standardizer contains non-finite values")
        if np.any(self.scale <= 0.0) or np.any(self.count <= 0):
            raise ValueError("channel standardizer lacks positive scale/support")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "lag_conditioned_channel_standardizer_v1",
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "count": self.count.tolist(),
            "fit_scope": "fit_parameter_subjects_only",
        }


@dataclass
class PreparedPartition:
    role: str
    eeg: np.ndarray
    fnirs: np.ndarray
    eeg_point_mask: np.ndarray
    fnirs_point_mask: np.ndarray
    eeg_token_mask: np.ndarray
    fnirs_token_mask: np.ndarray
    eeg_channel_mask: np.ndarray
    fnirs_channel_mask: np.ndarray
    target: np.ndarray
    sample_id: np.ndarray
    subject: np.ndarray
    condition: np.ndarray
    record_id: np.ndarray
    eeg_event_time_ms: np.ndarray
    fnirs_event_time_ms: np.ndarray
    eeg_channel_names: tuple[str, ...]
    fnirs_channel_names: tuple[str, ...]
    fnirs_component_roles: tuple[str, ...]
    eeg_native: NativeFeatureTargets
    fnirs_native: NativeFeatureTargets
    donor_index: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.sample_id)
        arrays = (
            self.eeg,
            self.fnirs,
            self.eeg_point_mask,
            self.fnirs_point_mask,
            self.eeg_token_mask,
            self.fnirs_token_mask,
            self.eeg_channel_mask,
            self.fnirs_channel_mask,
            self.target,
            self.subject,
            self.condition,
            self.record_id,
            self.eeg_event_time_ms,
            self.fnirs_event_time_ms,
            self.donor_index,
        )
        if any(len(value) != count for value in arrays):
            raise ValueError("prepared partition arrays differ in sample count")
        if self.eeg_native.values.shape[:2] != self.eeg_token_mask.shape:
            raise ValueError("EEG native target/token axes differ")
        if self.fnirs_native.values.shape[:2] != self.fnirs_token_mask.shape:
            raise ValueError("fNIRS native target/token axes differ")
        if len(set(map(str, self.sample_id))) != count:
            raise ValueError("prepared partition sample IDs must be unique")
        donor = np.asarray(self.donor_index)
        if donor.shape != (count,) or not np.issubdtype(donor.dtype, np.integer):
            raise ValueError("prepared donor_index must be an integer sample vector")
        if np.any(donor < 0) or np.any(donor >= count):
            raise ValueError("prepared donor_index is out of range")
        if np.any(donor == np.arange(count)):
            raise ValueError("prepared donor_index contains identity pairs")
        if len(set(donor.astype(int).tolist())) != count:
            raise ValueError("prepared donor_index must be a permutation")
        if np.any(np.asarray(self.subject).astype(str) != np.asarray(self.subject).astype(str)[donor]):
            raise ValueError("prepared donor_index changes subject")
        if np.any(
            np.asarray(self.condition).astype(str)
            != np.asarray(self.condition).astype(str)[donor]
        ):
            raise ValueError("prepared donor_index changes condition")
        eeg_time = np.asarray(self.eeg_event_time_ms, dtype=np.float64)
        fnirs_time = np.asarray(self.fnirs_event_time_ms, dtype=np.float64)
        if np.any(~np.isfinite(eeg_time)) or np.any(~np.isfinite(fnirs_time)):
            raise ValueError("prepared event timing must be finite")
        same_record = np.asarray(self.record_id).astype(str) == np.asarray(
            self.record_id
        ).astype(str)[donor]
        eeg_overlap = np.abs(eeg_time - eeg_time[donor]) < 20_000.0
        fnirs_overlap = np.abs(fnirs_time - fnirs_time[donor]) < 20_000.0
        if np.any(same_record & (eeg_overlap | fnirs_overlap)):
            raise ValueError("prepared donor_index contains overlapping 20 s windows")

    def index_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "role": self.role,
                "partition_index": index,
                "sample_id": self.sample_id[index],
                "subject": self.subject[index],
                "condition": self.condition[index],
                "record_id": self.record_id[index],
                "eeg_event_time_ms": float(self.eeg_event_time_ms[index]),
                "fnirs_event_time_ms": float(self.fnirs_event_time_ms[index]),
                "target": int(self.target[index]),
                "donor_index": int(self.donor_index[index]),
                "donor_sample_id": self.sample_id[int(self.donor_index[index])],
                "same_subject_donor": bool(
                    self.subject[index] == self.subject[int(self.donor_index[index])]
                ),
                "same_condition_donor": bool(
                    self.condition[index] == self.condition[int(self.donor_index[index])]
                ),
                "identity_donor": bool(index == int(self.donor_index[index])),
                "donor_eeg_event_time_ms": float(
                    self.eeg_event_time_ms[int(self.donor_index[index])]
                ),
                "donor_fnirs_event_time_ms": float(
                    self.fnirs_event_time_ms[int(self.donor_index[index])]
                ),
                "overlapping_window_donor": False,
            }
            for index in range(len(self.sample_id))
        ]


def _partition_derangement_nonoverlap_verified(
    partition: PreparedPartition,
) -> bool:
    donor = np.asarray(partition.donor_index, dtype=np.int64)
    record = np.asarray(partition.record_id).astype(str)
    eeg_time = np.asarray(partition.eeg_event_time_ms, dtype=np.float64)
    fnirs_time = np.asarray(partition.fnirs_event_time_ms, dtype=np.float64)
    same_record = record == record[donor]
    overlap = (np.abs(eeg_time - eeg_time[donor]) < 20_000.0) | (
        np.abs(fnirs_time - fnirs_time[donor]) < 20_000.0
    )
    return bool(not np.any(same_record & overlap))


class PreparedTorchDataset(Dataset):
    """Tensor view over one normalized, native-target-prepared partition."""

    def __init__(self, partition: PreparedPartition) -> None:
        self.partition = partition

    def __len__(self) -> int:
        return len(self.partition.sample_id)

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        donor = int(self.partition.donor_index[index])
        return {
            "index": torch.tensor(index, dtype=torch.long),
            "donor_index": torch.tensor(donor, dtype=torch.long),
            "eeg": torch.from_numpy(self.partition.eeg[index]),
            "fnirs": torch.from_numpy(self.partition.fnirs[index]),
            "donor_eeg": torch.from_numpy(self.partition.eeg[donor]),
            "donor_fnirs": torch.from_numpy(self.partition.fnirs[donor]),
            "eeg_point_valid_mask": torch.from_numpy(
                self.partition.eeg_point_mask[index]
            ),
            "fnirs_point_valid_mask": torch.from_numpy(
                self.partition.fnirs_point_mask[index]
            ),
            "donor_eeg_point_valid_mask": torch.from_numpy(
                self.partition.eeg_point_mask[donor]
            ),
            "donor_fnirs_point_valid_mask": torch.from_numpy(
                self.partition.fnirs_point_mask[donor]
            ),
            "eeg_token_valid_mask": torch.from_numpy(
                self.partition.eeg_token_mask[index]
            ),
            "fnirs_token_valid_mask": torch.from_numpy(
                self.partition.fnirs_token_mask[index]
            ),
            "donor_eeg_token_valid_mask": torch.from_numpy(
                self.partition.eeg_token_mask[donor]
            ),
            "donor_fnirs_token_valid_mask": torch.from_numpy(
                self.partition.fnirs_token_mask[donor]
            ),
            "eeg_channel_valid_mask": torch.from_numpy(
                self.partition.eeg_channel_mask[index]
            ),
            "fnirs_channel_valid_mask": torch.from_numpy(
                self.partition.fnirs_channel_mask[index]
            ),
            "donor_eeg_channel_valid_mask": torch.from_numpy(
                self.partition.eeg_channel_mask[donor]
            ),
            "donor_fnirs_channel_valid_mask": torch.from_numpy(
                self.partition.fnirs_channel_mask[donor]
            ),
            "eeg_native": torch.from_numpy(
                self.partition.eeg_native.values[index]
            ),
            "fnirs_native": torch.from_numpy(
                self.partition.fnirs_native.values[index]
            ),
            "donor_fnirs_native": torch.from_numpy(
                self.partition.fnirs_native.values[donor]
            ),
            "eeg_native_valid_mask": torch.from_numpy(
                self.partition.eeg_native.valid_mask[index]
            ),
            "fnirs_native_valid_mask": torch.from_numpy(
                self.partition.fnirs_native.valid_mask[index]
            ),
            "donor_fnirs_native_valid_mask": torch.from_numpy(
                self.partition.fnirs_native.valid_mask[donor]
            ),
            "target": torch.tensor(
                int(self.partition.target[index]), dtype=torch.long
            ),
            "sample_id": str(self.partition.sample_id[index]),
            "donor_sample_id": str(self.partition.sample_id[donor]),
            "subject": str(self.partition.subject[index]),
            "condition": str(self.partition.condition[index]),
            "record_id": str(self.partition.record_id[index]),
            "eeg_event_time_ms": torch.tensor(
                float(self.partition.eeg_event_time_ms[index]), dtype=torch.float64
            ),
            "fnirs_event_time_ms": torch.tensor(
                float(self.partition.fnirs_event_time_ms[index]), dtype=torch.float64
            ),
        }


def make_prepared_loader(
    partition: PreparedPartition,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        PreparedTorchDataset(partition),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=int(num_workers),
        drop_last=False,
    )


@dataclass(frozen=True)
class PreparedTask:
    task_id: str
    dataset_id: str
    parameter: PreparedPartition
    selection: PreparedPartition
    development: PreparedPartition
    eeg_standardizer: ChannelStandardizer
    fnirs_standardizer: ChannelStandardizer
    eeg_native_standardizer: MaskedStandardizer
    fnirs_native_standardizer: MaskedStandardizer
    protected_metadata_indexed: bool
    measured_access_count: int
    protected_measured_access_count: int
    _preparation_capability: object = field(repr=False, compare=False)
    governance_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "governance_digest", _prepared_governance_digest(self))


def _prepared_governance_digest(prepared: PreparedTask) -> str:
    partitions = []
    for partition in (prepared.parameter, prepared.selection, prepared.development):
        partitions.append(
            {
                "role": partition.role,
                "sample_id": np.asarray(partition.sample_id).astype(str).tolist(),
                "subject": np.asarray(partition.subject).astype(str).tolist(),
                "condition": np.asarray(partition.condition).astype(str).tolist(),
                "record_id": np.asarray(partition.record_id).astype(str).tolist(),
                "eeg_event_time_ms": np.asarray(
                    partition.eeg_event_time_ms, dtype=np.float64
                ).tolist(),
                "fnirs_event_time_ms": np.asarray(
                    partition.fnirs_event_time_ms, dtype=np.float64
                ).tolist(),
                "donor_index": np.asarray(
                    partition.donor_index, dtype=np.int64
                ).tolist(),
            }
        )
    payload = {
        "schema": "lc_spvq_prepared_governance_v1",
        "task_id": prepared.task_id,
        "dataset_id": prepared.dataset_id,
        "protected_metadata_indexed": prepared.protected_metadata_indexed,
        "measured_access_count": int(prepared.measured_access_count),
        "protected_measured_access_count": int(
            prepared.protected_measured_access_count
        ),
        "partitions": partitions,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_prepared_governance(prepared: PreparedTask) -> None:
    if prepared._preparation_capability is not _PREPARATION_CAPABILITY:
        raise PermissionError("prepared task lacks the opaque preparation capability")
    if prepared.governance_digest != _prepared_governance_digest(prepared):
        raise PermissionError("prepared task governance metadata changed after preparation")
    if prepared.protected_metadata_indexed is not True:
        raise RuntimeError("prepared task did not index the reviewed protected metadata boundary")
    if int(prepared.measured_access_count) <= 0:
        raise RuntimeError("prepared task records no measured sample access")
    if int(prepared.protected_measured_access_count) != 0:
        raise PermissionError("prepared task records protected measured access")
    if prepared.task_id not in TASK_SPECS:
        raise ValueError("prepared task identity is unknown")
    spec = TASK_SPECS[prepared.task_id]
    if prepared.dataset_id != spec.dataset_id:
        raise ValueError("prepared task dataset identity drifted")
    expected_roles = (
        (prepared.parameter, "fit_parameter", "fit_parameter_subjects"),
        (prepared.selection, "fit_selection", "fit_selection_subjects"),
        (prepared.development, "development_apply", "development_apply_subjects"),
    )
    total_samples = 0
    protected = set(CANONICAL_PROTECTED_SUBJECTS[spec.dataset_id])
    for partition, role, split_role in expected_roles:
        if partition.role != role:
            raise ValueError(f"prepared partition role drifted: {partition.role!r}")
        subjects = set(np.asarray(partition.subject).astype(str).tolist())
        allowed = set(CANONICAL_SUBJECT_SPLITS[split_role][spec.dataset_id])
        if not subjects or not subjects.issubset(allowed):
            raise PermissionError(f"prepared {role} subjects leave the reviewed split")
        if subjects & protected:
            raise PermissionError(f"prepared {role} includes a protected subject")
        for sample_id, subject, record_id in zip(
            np.asarray(partition.sample_id).astype(str),
            np.asarray(partition.subject).astype(str),
            np.asarray(partition.record_id).astype(str),
            strict=True,
        ):
            expected_prefix = f"{spec.dataset_id}|{subject}|{record_id}|"
            if not sample_id.startswith(expected_prefix):
                raise ValueError("prepared canonical sample identity drifted")
            if record_id not in spec.record_ids:
                raise ValueError("prepared record identity leaves the task contract")
        total_samples += len(partition.sample_id)
    if int(prepared.measured_access_count) != total_samples:
        raise RuntimeError("prepared measured-access count differs from sample support")


def _balanced_indices(
    rows: Sequence[LagConditionedSampleIndex],
    *,
    max_per_subject_class: int | None,
) -> np.ndarray:
    if max_per_subject_class is None:
        return np.arange(len(rows), dtype=np.int64)
    limit = int(max_per_subject_class)
    if limit < 2:
        raise ValueError("smoke selection needs at least two trials per donor group")
    counts: dict[tuple[str, str], int] = {}
    selected: list[int] = []
    for index, row in enumerate(rows):
        key = (row.subject, row.condition)
        count = counts.get(key, 0)
        if count >= limit:
            continue
        selected.append(index)
        counts[key] = count + 1
    if not selected:
        raise RuntimeError("balanced smoke selector admitted no samples")
    if any(value < 2 for value in counts.values()):
        raise RuntimeError("balanced smoke selector lacks derangement support")
    return np.asarray(selected, dtype=np.int64)


def _selected_rows(
    dataset: LagConditionedTaskDataset, indices: np.ndarray
) -> tuple[LagConditionedSampleIndex, ...]:
    return tuple(dataset.rows[int(index)] for index in indices)


def _native_targets_chunked(
    eeg: np.ndarray,
    fnirs: np.ndarray,
    eeg_token_mask: np.ndarray,
    fnirs_token_mask: np.ndarray,
    *,
    eeg_channel_valid_mask: np.ndarray | None = None,
    fnirs_channel_valid_mask: np.ndarray | None = None,
    eeg_channel_names: Sequence[str],
    fnirs_channel_names: Sequence[str],
    fnirs_component_roles: Sequence[str],
    chunk_size: int = 16,
) -> tuple[NativeFeatureTargets, NativeFeatureTargets]:
    eeg_values: list[np.ndarray] = []
    eeg_masks: list[np.ndarray] = []
    fnirs_values: list[np.ndarray] = []
    fnirs_masks: list[np.ndarray] = []
    eeg_names: tuple[str, ...] | None = None
    fnirs_names: tuple[str, ...] | None = None
    for start in range(0, len(eeg), int(chunk_size)):
        stop = min(start + int(chunk_size), len(eeg))
        eeg_part = extract_eeg_native_targets(
            eeg[start:stop],
            eeg_token_mask[start:stop],
            channel_valid_mask=(
                None
                if eeg_channel_valid_mask is None
                else eeg_channel_valid_mask[start:stop]
            ),
            channel_names=eeg_channel_names,
        )
        fnirs_part = extract_fnirs_native_targets(
            fnirs[start:stop],
            fnirs_token_mask[start:stop],
            component_roles=fnirs_component_roles,
            channel_valid_mask=(
                None
                if fnirs_channel_valid_mask is None
                else fnirs_channel_valid_mask[start:stop]
            ),
            channel_names=fnirs_channel_names,
        )
        if eeg_names is None:
            eeg_names = eeg_part.feature_names
            fnirs_names = fnirs_part.feature_names
        elif eeg_names != eeg_part.feature_names or fnirs_names != fnirs_part.feature_names:
            raise RuntimeError("native feature names drifted across chunks")
        eeg_values.append(eeg_part.values)
        eeg_masks.append(eeg_part.valid_mask)
        fnirs_values.append(fnirs_part.values)
        fnirs_masks.append(fnirs_part.valid_mask)
    if eeg_names is None or fnirs_names is None:
        raise RuntimeError("native feature extraction received an empty partition")
    return (
        NativeFeatureTargets(
            values=np.concatenate(eeg_values),
            valid_mask=np.concatenate(eeg_masks),
            feature_names=eeg_names,
        ),
        NativeFeatureTargets(
            values=np.concatenate(fnirs_values),
            valid_mask=np.concatenate(fnirs_masks),
            feature_names=fnirs_names,
        ),
    )


def prepare_partition(
    dataset: LagConditionedTaskDataset,
    *,
    role: str,
    max_per_subject_class: int | None,
    derangement_seed: int,
) -> PreparedPartition:
    if max_per_subject_class is None:
        raise RuntimeError(
            "full preparation remains blocked until fit-selection-only lag-weight "
            "and checkpoint selection is implemented"
        )
    dataset.validate_governance_contract()
    required_protected = set(CANONICAL_PROTECTED_SUBJECTS[dataset.spec.dataset_id])
    if set(dataset.forbidden_subjects) != required_protected:
        raise PermissionError("prepared partition lacks the canonical protected boundary")
    indices = _balanced_indices(
        dataset.rows, max_per_subject_class=max_per_subject_class
    )
    rows = _selected_rows(dataset, indices)
    donor = make_group_derangement(rows, seed=int(derangement_seed))
    loaded = [dataset._tensor_sample(int(index)) for index in indices]
    first = loaded[0]
    eeg_names = tuple(first["eeg_channel_names"])
    fnirs_names = tuple(first["fnirs_channel_names"])
    fnirs_roles = tuple(first["fnirs_component_roles"])
    for sample in loaded[1:]:
        if tuple(sample["eeg_channel_names"]) != eeg_names:
            raise RuntimeError("EEG channel signature drifted inside task partition")
        if tuple(sample["fnirs_channel_names"]) != fnirs_names:
            raise RuntimeError("fNIRS channel signature drifted inside task partition")
        if tuple(sample["fnirs_component_roles"]) != fnirs_roles:
            raise RuntimeError("fNIRS component roles drifted inside task partition")

    def stack(key: str) -> np.ndarray:
        return np.stack([sample[key].numpy() for sample in loaded])

    eeg = stack("eeg").astype(np.float32, copy=False)
    fnirs = stack("fnirs").astype(np.float32, copy=False)
    eeg_token = stack("eeg_token_valid_mask").astype(bool, copy=False)
    fnirs_token = stack("fnirs_token_valid_mask").astype(bool, copy=False)
    eeg_channel = stack("eeg_channel_valid_mask").astype(bool, copy=False)
    fnirs_channel = stack("fnirs_channel_valid_mask").astype(bool, copy=False)
    eeg_native, fnirs_native = _native_targets_chunked(
        eeg,
        fnirs,
        eeg_token,
        fnirs_token,
        eeg_channel_valid_mask=eeg_channel,
        fnirs_channel_valid_mask=fnirs_channel,
        eeg_channel_names=eeg_names,
        fnirs_channel_names=fnirs_names,
        fnirs_component_roles=fnirs_roles,
    )
    return PreparedPartition(
        role=str(role),
        eeg=eeg,
        fnirs=fnirs,
        eeg_point_mask=stack("eeg_point_valid_mask").astype(bool, copy=False),
        fnirs_point_mask=stack("fnirs_point_valid_mask").astype(bool, copy=False),
        eeg_token_mask=eeg_token,
        fnirs_token_mask=fnirs_token,
        eeg_channel_mask=eeg_channel,
        fnirs_channel_mask=fnirs_channel,
        target=stack("target").astype(np.int64, copy=False),
        sample_id=np.asarray([row.sample_id for row in rows], dtype=str),
        subject=np.asarray([row.subject for row in rows], dtype=str),
        condition=np.asarray([row.condition for row in rows], dtype=str),
        record_id=np.asarray([row.record_id for row in rows], dtype=str),
        eeg_event_time_ms=np.asarray(
            [row.event_time_ms for row in rows], dtype=np.float64
        ),
        fnirs_event_time_ms=np.asarray(
            [row.fnirs_event_time_ms for row in rows], dtype=np.float64
        ),
        eeg_channel_names=eeg_names,
        fnirs_channel_names=fnirs_names,
        fnirs_component_roles=fnirs_roles,
        eeg_native=eeg_native,
        fnirs_native=fnirs_native,
        donor_index=donor,
    )


def fit_channel_standardizer(
    signal: np.ndarray,
    point_mask: np.ndarray,
    channel_mask: np.ndarray,
) -> ChannelStandardizer:
    values = np.asarray(signal, dtype=np.float64)
    point = np.asarray(point_mask, dtype=bool)
    channel = np.asarray(channel_mask, dtype=bool)
    if values.ndim != 3:
        raise ValueError("channel standardization requires [sample, channel, time]")
    if point.shape != (values.shape[0], values.shape[2]):
        raise ValueError("point mask shape differs from signal")
    if channel.shape != values.shape[:2]:
        raise ValueError("channel mask shape differs from signal")
    valid = channel[:, :, None] & point[:, None, :] & np.isfinite(values)
    count = valid.sum(axis=(0, 2)).astype(np.int64)
    if np.any(count <= 0):
        raise ValueError("one or more channels lack fit support")
    total = np.where(valid, values, 0.0).sum(axis=(0, 2))
    square = np.where(valid, np.square(values), 0.0).sum(axis=(0, 2))
    mean = total / count
    variance = np.maximum(square / count - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    if np.any(scale < 1e-6) or np.any(~np.isfinite(scale)):
        raise ValueError("one or more channels have degenerate fit scale")
    return ChannelStandardizer(
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        count=count,
    )


def apply_channel_standardizer(
    signal: np.ndarray,
    point_mask: np.ndarray,
    channel_mask: np.ndarray,
    standardizer: ChannelStandardizer,
) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32)
    if values.shape[1] != len(standardizer.mean):
        raise ValueError("channel standardizer dimension differs from signal")
    valid = (
        np.asarray(channel_mask, dtype=bool)[:, :, None]
        & np.asarray(point_mask, dtype=bool)[:, None, :]
        & np.isfinite(values)
    )
    transformed = (
        values - standardizer.mean[None, :, None]
    ) / standardizer.scale[None, :, None]
    return np.where(valid, transformed, 0.0).astype(np.float32)


def _subject_role_values(
    config: Mapping[str, Any],
    *,
    dataset_id: str,
    role: str,
    smoke: bool,
) -> list[str]:
    values = list(map(str, config["data_split"][role][dataset_id]))
    if not smoke:
        return values
    key = {
        "fit_parameter_subjects": "parameter_subjects_per_dataset",
        "fit_selection_subjects": "selection_subjects_per_dataset",
        "development_apply_subjects": "development_subjects_per_dataset",
    }[role]
    return values[: int(config["smoke"][key])]


def prepare_task(
    config: Mapping[str, Any],
    task_id: str,
    *,
    smoke: bool,
    derangement_seed: int,
) -> PreparedTask:
    validate_config(config)
    _require_smoke_training(smoke)
    spec = TASK_SPECS[str(task_id)]
    forbidden = config["data_split"]["protected_or_unused"][spec.dataset_id]
    cache_root = _resolve(str(config["source"]["cache_root"]))
    base = UnifiedPhysiologyWindowDataset(
        cache_root=cache_root,
        dataset_ids=(spec.dataset_id,),
        window_duration_s=float(config["source"]["window_duration_s"]),
        window_offset_s=float(config["source"]["window_offset_s"]),
        eeg_signal_branch=str(config["source"]["eeg_signal_branch"]),
        require_eeg_artifact_cache=spec.dataset_id == "eeg_fnirs_single_trial",
    )
    role_datasets = {
        role: LagConditionedTaskDataset(
            task_id=task_id,
            admitted_subjects=_subject_role_values(
                config,
                dataset_id=spec.dataset_id,
                role=role,
                smoke=smoke,
            ),
            forbidden_subjects=forbidden,
            cache_root=cache_root,
            eeg_signal_branch=str(config["source"]["eeg_signal_branch"]),
            base_dataset=base,
        )
        for role in (
            "fit_parameter_subjects",
            "fit_selection_subjects",
            "development_apply_subjects",
        )
    }
    limit = int(config["smoke"]["samples_per_subject_class"]) if smoke else None
    parameter = prepare_partition(
        role_datasets["fit_parameter_subjects"],
        role="fit_parameter",
        max_per_subject_class=limit,
        derangement_seed=derangement_seed,
    )
    selection = prepare_partition(
        role_datasets["fit_selection_subjects"],
        role="fit_selection",
        max_per_subject_class=limit,
        derangement_seed=derangement_seed,
    )
    development = prepare_partition(
        role_datasets["development_apply_subjects"],
        role="development_apply",
        max_per_subject_class=limit,
        derangement_seed=derangement_seed,
    )
    signatures = {
        (
            partition.eeg_channel_names,
            partition.fnirs_channel_names,
            partition.fnirs_component_roles,
        )
        for partition in (parameter, selection, development)
    }
    if len(signatures) != 1:
        raise RuntimeError("channel/component signature drifted across subject partitions")

    eeg_stats = fit_channel_standardizer(
        parameter.eeg, parameter.eeg_point_mask, parameter.eeg_channel_mask
    )
    fnirs_stats = fit_channel_standardizer(
        parameter.fnirs, parameter.fnirs_point_mask, parameter.fnirs_channel_mask
    )
    eeg_native_stats = fit_masked_standardizer(parameter.eeg_native)
    fnirs_native_stats = fit_masked_standardizer(parameter.fnirs_native)
    protected_metadata_subjects = {
        str(ref.record.canonical_subject_id) for ref in base.windows
    }.intersection(set(map(str, forbidden)))
    protected_metadata_indexed = protected_metadata_subjects == set(map(str, forbidden))
    measured_access_count = sum(
        int(dataset.measured_access_count) for dataset in role_datasets.values()
    )
    protected_measured_access_count = sum(
        int(dataset.protected_measured_access_count) for dataset in role_datasets.values()
    )
    if not protected_metadata_indexed:
        raise RuntimeError("unified loader did not index the complete protected metadata boundary")
    if protected_measured_access_count != 0:
        raise PermissionError("protected measured access occurred during preparation")
    for partition in (parameter, selection, development):
        partition.eeg = apply_channel_standardizer(
            partition.eeg,
            partition.eeg_point_mask,
            partition.eeg_channel_mask,
            eeg_stats,
        )
        partition.fnirs = apply_channel_standardizer(
            partition.fnirs,
            partition.fnirs_point_mask,
            partition.fnirs_channel_mask,
            fnirs_stats,
        )
        partition.eeg_native = apply_masked_standardizer(
            partition.eeg_native, eeg_native_stats
        )
        partition.fnirs_native = apply_masked_standardizer(
            partition.fnirs_native, fnirs_native_stats
        )
    return PreparedTask(
        task_id=task_id,
        dataset_id=spec.dataset_id,
        parameter=parameter,
        selection=selection,
        development=development,
        eeg_standardizer=eeg_stats,
        fnirs_standardizer=fnirs_stats,
        eeg_native_standardizer=eeg_native_stats,
        fnirs_native_standardizer=fnirs_native_stats,
        protected_metadata_indexed=protected_metadata_indexed,
        measured_access_count=measured_access_count,
        protected_measured_access_count=protected_measured_access_count,
        _preparation_capability=_PREPARATION_CAPABILITY,
    )


def save_preparation(prepared: PreparedTask, output: Path) -> dict[str, Any]:
    _validate_prepared_governance(prepared)
    rows = (
        prepared.parameter.index_rows()
        + prepared.selection.index_rows()
        + prepared.development.index_rows()
    )
    _write_csv(output / "source_index.csv", rows)
    stats = {
        "schema": PREPARATION_SCHEMA,
        "task_id": prepared.task_id,
        "dataset_id": prepared.dataset_id,
        "input": {
            "eeg": prepared.eeg_standardizer.to_dict(),
            "fnirs": prepared.fnirs_standardizer.to_dict(),
        },
        "native": {
            "eeg": prepared.eeg_native_standardizer.to_dict(),
            "fnirs": prepared.fnirs_native_standardizer.to_dict(),
            "eeg_feature_names": list(prepared.parameter.eeg_native.feature_names),
            "fnirs_feature_names": list(prepared.parameter.fnirs_native.feature_names),
        },
        "channel_names": {
            "eeg": list(prepared.parameter.eeg_channel_names),
            "fnirs": list(prepared.parameter.fnirs_channel_names),
            "fnirs_component_roles": list(prepared.parameter.fnirs_component_roles),
        },
        "partition_sample_counts": {
            "fit_parameter": len(prepared.parameter.sample_id),
            "fit_selection": len(prepared.selection.sample_id),
            "development_apply": len(prepared.development.sample_id),
        },
        "protected_open": prepared.protected_measured_access_count != 0,
        "protected_metadata_indexed_by_unified_loader": prepared.protected_metadata_indexed,
        "prepared_governance_digest": prepared.governance_digest,
        "derangement_nonoverlap_verified": all(
            _partition_derangement_nonoverlap_verified(partition)
            for partition in (
                prepared.parameter,
                prepared.selection,
                prepared.development,
            )
        ),
        "measured_sample_access_count": prepared.measured_access_count,
        "protected_measured_access_count": prepared.protected_measured_access_count,
    }
    _write_json(output / "preparation.json", stats)
    return stats


def make_same_group_time_negative_mask(
    subjects: Sequence[str],
    conditions: Sequence[str],
    *,
    token_count: int,
    query_trial_ids: Sequence[Any] | None = None,
    target_trial_ids: Sequence[Any] | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Admit only same-subject/condition, other-trial, same-time negatives."""

    subject = tuple(map(str, subjects))
    condition = tuple(map(str, conditions))
    if len(subject) != len(condition) or not subject:
        raise ValueError("subjects and conditions must be matching non-empty vectors")
    query_ids = tuple(range(len(subject))) if query_trial_ids is None else tuple(query_trial_ids)
    target_ids = tuple(range(len(subject))) if target_trial_ids is None else tuple(target_trial_ids)
    if len(query_ids) != len(subject) or len(target_ids) != len(subject):
        raise ValueError("trial ID vectors must match the metadata batch")
    if int(token_count) <= 0:
        raise ValueError("token_count must be positive")
    batch = len(subject)
    output = torch.zeros(
        batch,
        int(token_count),
        batch,
        int(token_count),
        dtype=torch.bool,
        device=device,
    )
    for query in range(batch):
        for target in range(batch):
            if (
                query_ids[query] != target_ids[target]
                and subject[query] == subject[target]
                and condition[query] == condition[target]
            ):
                diagonal = torch.eye(
                    int(token_count), dtype=torch.bool, device=device
                )
                output[query, :, target, :] = diagonal
    return output


def make_aligned_donor_time_negative_mask(
    *,
    batch_size: int,
    token_count: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Pair each query only with its registered donor at the same token time."""
    if int(batch_size) <= 0 or int(token_count) <= 0:
        raise ValueError("batch_size and token_count must be positive")
    batch_eye = torch.eye(int(batch_size), dtype=torch.bool, device=device)
    time_eye = torch.eye(int(token_count), dtype=torch.bool, device=device)
    return batch_eye[:, None, :, None] & time_eye[None, :, None, :]


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: (value.to(device, non_blocking=False) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }


def _b0_model(
    prepared: PreparedTask,
    config: Mapping[str, Any],
) -> B0ContinuousSharedPrivate:
    model = config["model"]
    return B0ContinuousSharedPrivate(
        eeg_channels=len(prepared.parameter.eeg_channel_names),
        fnirs_channels=len(prepared.parameter.fnirs_channel_names),
        eeg_native_dim=len(prepared.parameter.eeg_native.feature_names),
        fnirs_native_dim=len(prepared.parameter.fnirs_native.feature_names),
        class_count=len(TASK_SPECS[prepared.task_id].class_names),
        eeg_patch_samples=TASK_SPECS[prepared.task_id].eeg_patch_samples,
        fnirs_patch_samples=TASK_SPECS[prepared.task_id].fnirs_patch_samples,
        num_tokens=int(model["num_tokens"]),
        shared_dim=int(model["shared_dim"]),
        eeg_private_dim=int(model["eeg_private_dim"]),
        fnirs_private_dim=int(model["fnirs_private_dim"]),
        encoder_depth=int(model["encoder_depth"]),
        encoder_num_heads=int(model["encoder_num_heads"]),
        encoder_feedforward_dim=int(model["encoder_feedforward_dim"]),
        native_decoder_hidden_dim=int(model["native_decoder_hidden_dim"]),
        raw_decoder_hidden_dim=int(model["raw_decoder_hidden_dim"]),
        dropout=float(model["dropout"]),
    )


def _lc_spvq_model(
    prepared: PreparedTask,
    config: Mapping[str, Any],
) -> LCSPVQModel:
    model = config["model"]
    quantizer = config["quantizer"]
    lag_tokens = tuple(
        int(round(float(seconds) / float(config["source"]["patch_duration_s"])))
        for seconds in config["head"]["lag_seconds"]
    )
    raw_decoders = {
        "eeg": RawFeatureDecoder(
            shared_dim=int(model["shared_dim"]),
            private_dim=int(model["eeg_private_dim"]),
            output_channels=len(prepared.parameter.eeg_channel_names),
            patch_samples=TASK_SPECS[prepared.task_id].eeg_patch_samples,
            hidden_dim=int(model["raw_decoder_hidden_dim"]),
        ),
        "fnirs": RawFeatureDecoder(
            shared_dim=int(model["shared_dim"]),
            private_dim=int(model["fnirs_private_dim"]),
            output_channels=len(prepared.parameter.fnirs_channel_names),
            patch_samples=TASK_SPECS[prepared.task_id].fnirs_patch_samples,
            hidden_dim=int(model["raw_decoder_hidden_dim"]),
        ),
    }
    return LCSPVQModel(
        eeg_channels=len(prepared.parameter.eeg_channel_names),
        fnirs_channels=len(prepared.parameter.fnirs_channel_names),
        eeg_patch_samples=TASK_SPECS[prepared.task_id].eeg_patch_samples,
        fnirs_patch_samples=TASK_SPECS[prepared.task_id].fnirs_patch_samples,
        num_tokens=int(model["num_tokens"]),
        shared_dim=int(model["shared_dim"]),
        eeg_private_dim=int(model["eeg_private_dim"]),
        fnirs_private_dim=int(model["fnirs_private_dim"]),
        codebook_size=int(quantizer["eeg_codebook_size"]),
        eeg_shared_history_patches=int(model["eeg_shared_history_tokens"]) + 1,
        fnirs_shared_history_patches=int(model["fnirs_shared_history_tokens"]) + 1,
        encoder_depth=int(model["encoder_depth"]),
        encoder_num_heads=int(model["encoder_num_heads"]),
        encoder_feedforward_dim=int(model["encoder_feedforward_dim"]),
        private_encoder_depth=int(model["encoder_depth"]),
        private_encoder_num_heads=int(model["encoder_num_heads"]),
        private_encoder_feedforward_dim=int(model["encoder_feedforward_dim"]),
        dropout=float(model["dropout"]),
        projection_dim=int(model["projection_dim"]),
        coupling_rank=int(config["head"]["coupling_rank"]),
        allowed_lags=lag_tokens,
        num_classes=len(TASK_SPECS[prepared.task_id].class_names),
        native_decoder_hidden_dim=int(model["native_decoder_hidden_dim"]),
        raw_decoders=raw_decoders,
        eeg_native_feature_dim=len(prepared.parameter.eeg_native.feature_names),
        fnirs_native_feature_dim=len(prepared.parameter.fnirs_native.feature_names),
        quantizer_kwargs={
            "decay": float(quantizer["ema_decay"]),
            "eps": float(quantizer["eps"]),
            "commitment_cost": float(quantizer["commitment_cost"]),
            "temperature": float(quantizer["posterior_temperature_start"]),
            "assignment": str(quantizer["assignment"]),
            "normalize_latents": bool(quantizer["normalize_latents"]),
            "kmeans_init": True,
            "kmeans_iters": int(quantizer["kmeans_iterations"]),
            "revive_dead_codes": bool(quantizer["dead_code_revival"]),
        },
    )


def _lag_matching_module(config: Mapping[str, Any]) -> LagAwareContinuousMatchingLoss:
    patch_duration = float(config["source"]["patch_duration_s"])
    lag_tokens = tuple(
        int(round(float(seconds) / patch_duration))
        for seconds in config["objective"]["lag_seconds"]
    )
    return LagAwareContinuousMatchingLoss(
        positive_lag_weights={lag: 1.0 for lag in lag_tokens},
        temperature=float(config["objective"]["lag_loss_temperature"]),
        bidirectional=True,
        target_stop_gradient=bool(config["objective"]["target_stop_gradient"]),
        learnable_lag_mixture=True,
    )


def _b0_pretraining_losses(
    output: Mapping[str, Any],
    batch: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses = {
        "eeg_native": native_feature_prediction_loss(
            output["eeg_native"],
            batch["eeg_native"],
            batch["eeg_native_valid_mask"],
        ),
        "fnirs_native": native_feature_prediction_loss(
            output["fnirs_native"],
            batch["fnirs_native"],
            batch["fnirs_native_valid_mask"],
        ),
        "eeg_raw": raw_patch_reconstruction_loss(
            output["eeg_raw"],
            batch["eeg"],
            point_valid_mask=batch["eeg_point_valid_mask"],
            channel_valid_mask=batch["eeg_channel_valid_mask"],
        ),
        "fnirs_raw": raw_patch_reconstruction_loss(
            output["fnirs_raw"],
            batch["fnirs"],
            point_valid_mask=batch["fnirs_point_valid_mask"],
            channel_valid_mask=batch["fnirs_channel_valid_mask"],
        ),
    }
    objective = config["objective"]
    weights = {
        "eeg_native": float(objective["native_loss_weight"])
        * float(objective["native_modality_weight"]),
        "fnirs_native": float(objective["native_loss_weight"])
        * float(objective["native_modality_weight"]),
        "eeg_raw": float(objective["raw_loss_weight"])
        * float(objective["raw_modality_weight"]),
        "fnirs_raw": float(objective["raw_loss_weight"])
        * float(objective["raw_modality_weight"]),
    }
    total, _ = weighted_pretraining_loss(losses, weights)
    return total, losses


def _forward_lc_spvq(
    model: LCSPVQModel,
    batch: Mapping[str, Any],
    *,
    variant: str,
) -> Mapping[str, Any]:
    if variant not in {"M1", "N1"}:
        raise ValueError("LC-SPVQ forward variant must be M1 or N1")
    if variant == "N1":
        fnirs = batch["donor_fnirs"]
        fnirs_mask = batch["donor_fnirs_token_valid_mask"]
    else:
        fnirs = batch["fnirs"]
        fnirs_mask = batch["fnirs_token_valid_mask"]
    return model(
        batch["eeg"],
        fnirs,
        batch["eeg_token_valid_mask"],
        fnirs_mask,
    )


def _lc_spvq_pretraining_losses(
    model: LCSPVQModel,
    lag_module: LagAwareContinuousMatchingLoss,
    output: Mapping[str, Any],
    batch: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    variant: str,
    include_commitment: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], Mapping[str, Any]]:
    if variant == "N1":
        fnirs_target = batch["donor_fnirs"]
        fnirs_point_mask = batch["donor_fnirs_point_valid_mask"]
        fnirs_channel_mask = batch["donor_fnirs_channel_valid_mask"]
        fnirs_native = batch["donor_fnirs_native"]
        fnirs_native_mask = batch["donor_fnirs_native_valid_mask"]
        negative_fnirs = batch["fnirs"]
        negative_fnirs_mask = batch["fnirs_token_valid_mask"]
    else:
        fnirs_target = batch["fnirs"]
        fnirs_point_mask = batch["fnirs_point_valid_mask"]
        fnirs_channel_mask = batch["fnirs_channel_valid_mask"]
        fnirs_native = batch["fnirs_native"]
        fnirs_native_mask = batch["fnirs_native_valid_mask"]
        negative_fnirs = batch["donor_fnirs"]
        negative_fnirs_mask = batch["donor_fnirs_token_valid_mask"]

    negative_fnirs_pre = model.fnirs_shared_encoder(
        negative_fnirs, negative_fnirs_mask
    )
    negative_fnirs_projection = model.fnirs_projection_head(negative_fnirs_pre)
    negative_eeg_pre = model.eeg_shared_encoder(
        batch["donor_eeg"], batch["donor_eeg_token_valid_mask"]
    )
    negative_eeg_projection = model.eeg_projection_head(negative_eeg_pre)
    batch_size, token_count = output["eeg_projection"].shape[:2]
    if output["fnirs_projection"].shape[:2] != (batch_size, token_count):
        raise RuntimeError("registered hard-negative timing requires matching token axes")
    subjects = tuple(map(str, batch["subject"]))
    conditions = tuple(map(str, batch["condition"]))
    query_physical_trial_ids = batch["index"].detach().cpu().tolist()
    target_physical_trial_ids = (
        batch["donor_index"].detach().cpu().tolist()
        if variant == "N1"
        else query_physical_trial_ids
    )
    in_batch_negative_mask = make_same_group_time_negative_mask(
        subjects,
        conditions,
        token_count=token_count,
        query_trial_ids=query_physical_trial_ids,
        target_trial_ids=target_physical_trial_ids,
        device=output["eeg_projection"].device,
    )
    aligned_donor_mask = make_aligned_donor_time_negative_mask(
        batch_size=batch_size,
        token_count=token_count,
        device=output["eeg_projection"].device,
    )
    pair_slot_ids = torch.arange(
        batch_size, device=output["eeg_projection"].device, dtype=torch.long
    )
    subject_lookup = {value: index for index, value in enumerate(sorted(set(subjects)))}
    subject_ids = torch.as_tensor(
        [subject_lookup[value] for value in subjects],
        device=output["eeg_projection"].device,
        dtype=torch.long,
    )
    relative_time = torch.arange(
        token_count, device=output["eeg_projection"].device, dtype=torch.long
    ).expand(batch_size, token_count)
    lag_details = lag_module(
        output["eeg_projection"],
        output["fnirs_projection"],
        query_valid_mask=batch["eeg_token_valid_mask"],
        target_valid_mask=(
            batch["donor_fnirs_token_valid_mask"]
            if variant == "N1"
            else batch["fnirs_token_valid_mask"]
        ),
        query_trial_ids=pair_slot_ids,
        target_trial_ids=pair_slot_ids,
        query_subject_ids=subject_ids,
        target_subject_ids=subject_ids,
        query_relative_time=relative_time,
        target_relative_time=relative_time,
        negative_mask=in_batch_negative_mask,
        deranged_target=negative_fnirs_projection,
        deranged_target_negative_mask=aligned_donor_mask,
        deranged_target_valid_mask=negative_fnirs_mask,
        deranged_query=negative_eeg_projection,
        deranged_query_valid_mask=batch["donor_eeg_token_valid_mask"],
        deranged_query_negative_mask=aligned_donor_mask,
        return_details=True,
    )
    losses: dict[str, torch.Tensor] = {
        "eeg_native": native_feature_prediction_loss(
            output["eeg_native_target_prediction"],
            batch["eeg_native"],
            batch["eeg_native_valid_mask"],
        ),
        "fnirs_native": native_feature_prediction_loss(
            output["fnirs_native_target_prediction"],
            fnirs_native,
            fnirs_native_mask,
        ),
        "eeg_raw": raw_patch_reconstruction_loss(
            output["eeg_raw"],
            batch["eeg"],
            point_valid_mask=batch["eeg_point_valid_mask"],
            channel_valid_mask=batch["eeg_channel_valid_mask"],
        ),
        "fnirs_raw": raw_patch_reconstruction_loss(
            output["fnirs_raw"],
            fnirs_target,
            point_valid_mask=fnirs_point_mask,
            channel_valid_mask=fnirs_channel_mask,
        ),
        "lag": lag_details["loss"],
    }
    objective = config["objective"]
    weights = {
        "eeg_native": float(objective["native_loss_weight"])
        * float(objective["native_modality_weight"]),
        "fnirs_native": float(objective["native_loss_weight"])
        * float(objective["native_modality_weight"]),
        "eeg_raw": float(objective["raw_loss_weight"])
        * float(objective["raw_modality_weight"]),
        "fnirs_raw": float(objective["raw_loss_weight"])
        * float(objective["raw_modality_weight"]),
        # Smoke uses the first preregistered candidate; a full run must select
        # this candidate on fit-selection without consulting development.
        "lag": float(objective["lag_loss_weight_candidates"][0]),
    }
    if include_commitment:
        losses["commitment"] = output["commitment_loss"]
        weights["commitment"] = 1.0
    total, _ = weighted_pretraining_loss(losses, weights)
    return total, losses, lag_details


@torch.no_grad()
def _initialize_codebooks_from_fit_parameter(
    model: LCSPVQModel,
    partition: PreparedPartition,
    *,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Initialize both independent K16 codebooks from all admitted fit latents."""

    loader = make_prepared_loader(
        partition,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        seed=int(seed),
        num_workers=int(config["training"]["num_workers"]),
    )
    eeg_latents: list[torch.Tensor] = []
    fnirs_latents: list[torch.Tensor] = []
    model.eval()
    for raw_batch in loader:
        batch = _batch_to_device(raw_batch, device)
        eeg_mask = batch["eeg_token_valid_mask"].bool()
        fnirs_mask = batch["fnirs_token_valid_mask"].bool()
        eeg_pre = model.eeg_shared_encoder(batch["eeg"], eeg_mask)
        fnirs_pre = model.fnirs_shared_encoder(batch["fnirs"], fnirs_mask)
        eeg_latents.append(eeg_pre[eeg_mask])
        fnirs_latents.append(fnirs_pre[fnirs_mask])
    eeg_fit = torch.cat(eeg_latents, dim=0)
    fnirs_fit = torch.cat(fnirs_latents, dim=0)
    model.eeg_quantizer.initialized.fill_(False)
    model.fnirs_quantizer.initialized.fill_(False)
    model.eeg_quantizer._kmeans_initialize(eeg_fit)
    model.fnirs_quantizer._kmeans_initialize(fnirs_fit)
    return {
        "scope": "fit_parameter_all_admitted_continuous_latents",
        "eeg_latent_count": int(eeg_fit.shape[0]),
        "fnirs_latent_count": int(fnirs_fit.shape[0]),
        "eeg_initialized": bool(model.eeg_quantizer.initialized.item()),
        "fnirs_initialized": bool(model.fnirs_quantizer.initialized.item()),
    }


def _forward_b0(
    model: B0ContinuousSharedPrivate,
    batch: Mapping[str, Any],
) -> Mapping[str, Any]:
    return model(
        batch["eeg"],
        batch["fnirs"],
        batch["eeg_token_valid_mask"],
        batch["fnirs_token_valid_mask"],
    )


def _fixed_steps(smoke: bool, config: Mapping[str, Any], loader: DataLoader, stage: str) -> int:
    if smoke:
        key = {
            "pretrain": "pretrain_optimizer_steps",
            "vq": "vq_optimizer_steps",
            "head": "head_optimizer_steps",
        }[stage]
        return int(config["smoke"][key])
    epochs = {
        "pretrain": int(config["training"]["pretrain_epochs"]),
        "vq": int(config["training"]["vq_epochs"]),
        "head": int(config["training"]["head_epochs"]),
    }[stage]
    return epochs * len(loader)


def _iterate_loader(loader: DataLoader) -> Iterable[Mapping[str, Any]]:
    while True:
        for batch in loader:
            yield batch


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _evaluate_b0(
    model: B0ContinuousSharedPrivate,
    partition: PreparedPartition,
    *,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    loader = make_prepared_loader(
        partition,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        seed=int(seed),
        num_workers=int(config["training"]["num_workers"]),
    )
    names = ("shared_marginal_only", "private_only", "combined")
    logits = {name: [] for name in names}
    targets: list[np.ndarray] = []
    subjects: list[str] = []
    sample_ids: list[str] = []
    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch = _batch_to_device(raw_batch, device)
            output = _forward_b0(model, batch)
            for name in names:
                logits[name].append(
                    output["logits"][name].detach().cpu().numpy().astype(np.float32)
                )
            targets.append(batch["target"].detach().cpu().numpy())
            subjects.extend(map(str, raw_batch["subject"]))
            sample_ids.extend(map(str, raw_batch["sample_id"]))
    target_array = np.concatenate(targets)
    logit_arrays = {name: np.concatenate(values) for name, values in logits.items()}
    metrics = evaluate_logit_ablations(
        target_array,
        subjects,
        logit_arrays,
        class_names=TASK_SPECS[model_task_id(model, partition)].class_names,
    )
    arrays = {
        "sample_id": np.asarray(sample_ids, dtype=str),
        "subject": np.asarray(subjects, dtype=str),
        "target": target_array.astype(np.int64),
        **{f"{name}_logits": values for name, values in logit_arrays.items()},
    }
    return dict(metrics), arrays


def model_task_id(
    model: B0ContinuousSharedPrivate,
    partition: PreparedPartition,
) -> str:
    # Task identity is intentionally inferred from the frozen class set in the
    # owning PreparedTask by callers. This helper exists only to fail loudly if
    # an internal call omits that explicit identity.
    task_id = getattr(model, "_lc_spvq_task_id", None)
    if task_id not in TASK_SPECS:
        raise RuntimeError("B0 model lacks its explicit LC-SPVQ task identity")
    return str(task_id)


def train_b0_variant(
    prepared: PreparedTask,
    config: Mapping[str, Any],
    *,
    seed: int,
    device: torch.device,
    output_dir: Path,
    smoke: bool,
) -> dict[str, Any]:
    """Train the registered continuous baseline and its frozen-encoder head."""

    validate_config(config)
    _require_smoke_training(smoke)
    _validate_prepared_governance(prepared)
    _set_seed(int(seed))
    model = _b0_model(prepared, config).to(device)
    model._lc_spvq_task_id = prepared.task_id
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(False)
    train_loader = make_prepared_loader(
        prepared.parameter,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        seed=int(seed),
        num_workers=int(config["training"]["num_workers"]),
    )
    pretrain_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("classifier.")
    ]
    optimizer = torch.optim.AdamW(
        pretrain_parameters,
        lr=float(config["training"]["learning_rate"]),
        betas=tuple(map(float, config["training"]["betas"])),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    history: list[dict[str, Any]] = []
    iterator = iter(_iterate_loader(train_loader))
    model.train()
    for step in range(_fixed_steps(smoke, config, train_loader, "pretrain")):
        batch = _batch_to_device(next(iterator), device)
        optimizer.zero_grad(set_to_none=True)
        output = _forward_b0(model, batch)
        loss, components = _b0_pretraining_losses(output, batch, config)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            pretrain_parameters, float(config["training"]["grad_clip_norm"])
        )
        optimizer.step()
        history.append(
            {
                "stage": "continuous_pretrain",
                "step": step,
                "total_loss": float(loss.detach().cpu()),
                "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
                **{
                    f"{name}_loss": float(value.detach().cpu())
                    for name, value in components.items()
                },
            }
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    head_optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=float(config["training"]["head_learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    auxiliary = float(config["head"]["ablation_auxiliary_cross_entropy_weight"])
    iterator = iter(_iterate_loader(train_loader))
    model.train()
    for step in range(_fixed_steps(smoke, config, train_loader, "head")):
        batch = _batch_to_device(next(iterator), device)
        head_optimizer.zero_grad(set_to_none=True)
        output = _forward_b0(model, batch)
        logits = output["logits"]
        head_loss = torch.nn.functional.cross_entropy(
            logits["combined"], batch["target"]
        ) + auxiliary * (
            torch.nn.functional.cross_entropy(
                logits["shared_marginal_only"], batch["target"]
            )
            + torch.nn.functional.cross_entropy(
                logits["private_only"], batch["target"]
            )
        )
        head_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.classifier.parameters(), float(config["training"]["grad_clip_norm"])
        )
        head_optimizer.step()
        history.append(
            {
                "stage": "task_head",
                "step": step,
                "total_loss": float(head_loss.detach().cpu()),
                "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
            }
        )

    selection_metrics, _ = _evaluate_b0(
        model,
        prepared.selection,
        config=config,
        device=device,
        seed=seed,
    )
    development_metrics, predictions = _evaluate_b0(
        model,
        prepared.development,
        config=config,
        device=device,
        seed=seed,
    )
    _atomic_torch_save(
        {
            "schema": "lc_spvq_b0_checkpoint_v1",
            "task_id": prepared.task_id,
            "variant": "B0",
            "seed": int(seed),
            "model_state": model.state_dict(),
            "class_names": list(TASK_SPECS[prepared.task_id].class_names),
            "protected_open": prepared.protected_measured_access_count != 0,
        },
        output_dir / "checkpoint.pt",
    )
    np.savez_compressed(
        output_dir / "development_predictions.npz",
        schema=np.asarray("lc_spvq_b0_development_predictions_v1"),
        **predictions,
    )
    _write_csv(output_dir / "loss_curves.csv", history)
    result = {
        "schema": "lc_spvq_variant_result_v1",
        "task_id": prepared.task_id,
        "variant": "B0",
        "seed": int(seed),
        "status": "completed",
        "pretrain_steps": sum(row["stage"] == "continuous_pretrain" for row in history),
        "head_steps": sum(row["stage"] == "task_head" for row in history),
        "selection_metrics": selection_metrics,
        "development_metrics": development_metrics,
        "development_is_new_independent_holdout": False,
        "protected_open": prepared.protected_measured_access_count != 0,
        "protected_measured_access_count": prepared.protected_measured_access_count,
    }
    _write_json(output_dir / "result.json", result)
    return result


def _evaluate_lc_spvq(
    model: LCSPVQModel,
    partition: PreparedPartition,
    *,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    loader = make_prepared_loader(
        partition,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        seed=int(seed),
        num_workers=int(config["training"]["num_workers"]),
    )
    logit_names = {
        "coupling_only": "coupling_only_logits",
        "shared_marginal_only": "shared_marginal_only_logits",
        "private_only": "private_only_logits",
        "coupling_plus_private": "combined_logits",
    }
    logits = {name: [] for name in logit_names}
    exports: dict[str, list[np.ndarray]] = {
        "eeg_pre_vq": [],
        "fnirs_pre_vq": [],
        "eeg_posterior": [],
        "fnirs_posterior": [],
        "eeg_hard_id": [],
        "fnirs_hard_id": [],
        "eeg_expected_embedding": [],
        "fnirs_expected_embedding": [],
        "eeg_token_valid_mask": [],
        "fnirs_token_valid_mask": [],
    }
    targets: list[np.ndarray] = []
    subjects: list[str] = []
    conditions: list[str] = []
    sample_ids: list[str] = []
    record_ids: list[str] = []
    eeg_event_times: list[np.ndarray] = []
    fnirs_event_times: list[np.ndarray] = []
    donor_indices: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch = _batch_to_device(raw_batch, device)
            output = model(
                batch["eeg"],
                batch["fnirs"],
                batch["eeg_token_valid_mask"],
                batch["fnirs_token_valid_mask"],
            )
            for public_name, output_name in logit_names.items():
                logits[public_name].append(
                    output[output_name].detach().cpu().numpy().astype(np.float32)
                )
            values = {
                "eeg_pre_vq": output["eeg_pre_vq"],
                "fnirs_pre_vq": output["fnirs_pre_vq"],
                "eeg_posterior": output["eeg_shared_posterior"],
                "fnirs_posterior": output["fnirs_shared_posterior"],
                "eeg_hard_id": output["eeg_shared_hard_ids"],
                "fnirs_hard_id": output["fnirs_shared_hard_ids"],
                "eeg_expected_embedding": output["eeg_expected_embedding"],
                "fnirs_expected_embedding": output["fnirs_expected_embedding"],
                "eeg_token_valid_mask": output["eeg_token_valid_mask"],
                "fnirs_token_valid_mask": output["fnirs_token_valid_mask"],
            }
            for name, value in values.items():
                exports[name].append(value.detach().cpu().numpy())
            targets.append(batch["target"].detach().cpu().numpy())
            donor_indices.append(batch["donor_index"].detach().cpu().numpy())
            subjects.extend(map(str, raw_batch["subject"]))
            conditions.extend(map(str, raw_batch["condition"]))
            sample_ids.extend(map(str, raw_batch["sample_id"]))
            record_ids.extend(map(str, raw_batch["record_id"]))
            eeg_event_times.append(
                raw_batch["eeg_event_time_ms"].detach().cpu().numpy()
            )
            fnirs_event_times.append(
                raw_batch["fnirs_event_time_ms"].detach().cpu().numpy()
            )
    target_array = np.concatenate(targets).astype(np.int64)
    logit_arrays = {name: np.concatenate(value) for name, value in logits.items()}
    metrics = evaluate_logit_ablations(
        target_array,
        subjects,
        logit_arrays,
        class_names=TASK_SPECS[model_task_id_vq(model)].class_names,
    )
    arrays = {
        "sample_id": np.asarray(sample_ids, dtype=str),
        "subject": np.asarray(subjects, dtype=str),
        "condition": np.asarray(conditions, dtype=str),
        "record_id": np.asarray(record_ids, dtype=str),
        "eeg_event_time_ms": np.concatenate(eeg_event_times).astype(np.float64),
        "fnirs_event_time_ms": np.concatenate(fnirs_event_times).astype(np.float64),
        "target": target_array,
        "donor_index": np.concatenate(donor_indices).astype(np.int64),
        **{name: np.concatenate(value) for name, value in exports.items()},
        **{f"{name}_logits": value for name, value in logit_arrays.items()},
    }
    health = {}
    for modality in ("eeg", "fnirs"):
        ids = torch.from_numpy(arrays[f"{modality}_hard_id"]).long()
        mask = torch.from_numpy(arrays[f"{modality}_token_valid_mask"]).bool()
        health[modality] = compute_codebook_health(
            ids[mask], int(config["quantizer"][f"{modality}_codebook_size"]),
            include_distribution=True,
            top_k=min(10, int(config["quantizer"][f"{modality}_codebook_size"])),
        )
    return dict(metrics), arrays, health


def model_task_id_vq(model: LCSPVQModel) -> str:
    task_id = getattr(model, "_lc_spvq_task_id", None)
    if task_id not in TASK_SPECS:
        raise RuntimeError("LC-SPVQ model lacks its explicit task identity")
    return str(task_id)


def train_lc_spvq_variant(
    prepared: PreparedTask,
    config: Mapping[str, Any],
    *,
    variant: str,
    seed: int,
    device: torch.device,
    output_dir: Path,
    smoke: bool,
) -> dict[str, Any]:
    """Run continuous warm start, full-fit K-means VQ, then frozen head."""

    validate_config(config)
    _require_smoke_training(smoke)
    _validate_prepared_governance(prepared)
    if variant not in {"M1", "N1"}:
        raise ValueError("LC-SPVQ train variant must be M1 or N1")
    _set_seed(int(seed))
    model = _lc_spvq_model(prepared, config).to(device)
    model._lc_spvq_task_id = prepared.task_id
    lag_module = _lag_matching_module(config).to(device)
    head_modules = (
        model.coupling_head,
        model.shared_marginal_classifier,
        model.private_classifier,
    )
    for module in head_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    model.set_quantization_strength(
        float(config["quantizer"]["quantization_strength_start"])
    )
    train_loader = make_prepared_loader(
        prepared.parameter,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        seed=int(seed),
        num_workers=int(config["training"]["num_workers"]),
    )
    pretrain_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ] + list(lag_module.parameters())
    optimizer = torch.optim.AdamW(
        pretrain_parameters,
        lr=float(config["training"]["learning_rate"]),
        betas=tuple(map(float, config["training"]["betas"])),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    history: list[dict[str, Any]] = []
    iterator = iter(_iterate_loader(train_loader))
    model.train()
    model.eeg_quantizer.eval()
    model.fnirs_quantizer.eval()
    lag_module.train()
    for step in range(_fixed_steps(smoke, config, train_loader, "pretrain")):
        batch = _batch_to_device(next(iterator), device)
        optimizer.zero_grad(set_to_none=True)
        output = _forward_lc_spvq(model, batch, variant=variant)
        loss, components, lag_details = _lc_spvq_pretraining_losses(
            model,
            lag_module,
            output,
            batch,
            config,
            variant=variant,
            include_commitment=False,
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            pretrain_parameters, float(config["training"]["grad_clip_norm"])
        )
        optimizer.step()
        history.append(
            {
                "stage": "continuous_pretrain",
                "step": step,
                "total_loss": float(loss.detach().cpu()),
                "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
                "lag_weights": ";".join(
                    f"{float(value):.8f}"
                    for value in lag_details["lag_weights"].detach().cpu()
                ),
                **{
                    f"{name}_loss": float(value.detach().cpu())
                    for name, value in components.items()
                },
            }
        )

    kmeans = _initialize_codebooks_from_fit_parameter(
        model,
        prepared.parameter,
        config=config,
        device=device,
        seed=seed,
    )
    model.train()
    model.eeg_quantizer.train()
    model.fnirs_quantizer.train()
    iterator = iter(_iterate_loader(train_loader))
    vq_steps = _fixed_steps(smoke, config, train_loader, "vq")
    for step in range(vq_steps):
        denominator = max(vq_steps - 1, 1)
        fraction = step / denominator
        strength = float(config["quantizer"]["quantization_strength_start"]) + fraction * (
            float(config["quantizer"]["quantization_strength_end"])
            - float(config["quantizer"]["quantization_strength_start"])
        )
        temperature = float(config["quantizer"]["posterior_temperature_start"]) + fraction * (
            float(config["quantizer"]["posterior_temperature_end"])
            - float(config["quantizer"]["posterior_temperature_start"])
        )
        model.set_quantization_strength(strength)
        model.set_posterior_temperature(temperature)
        batch = _batch_to_device(next(iterator), device)
        optimizer.zero_grad(set_to_none=True)
        output = _forward_lc_spvq(model, batch, variant=variant)
        loss, components, lag_details = _lc_spvq_pretraining_losses(
            model,
            lag_module,
            output,
            batch,
            config,
            variant=variant,
            include_commitment=True,
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            pretrain_parameters, float(config["training"]["grad_clip_norm"])
        )
        optimizer.step()
        history.append(
            {
                "stage": "vq_anneal",
                "step": step,
                "total_loss": float(loss.detach().cpu()),
                "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
                "quantization_strength": strength,
                "posterior_temperature": temperature,
                "lag_weights": ";".join(
                    f"{float(value):.8f}"
                    for value in lag_details["lag_weights"].detach().cpu()
                ),
                **{
                    f"{name}_loss": float(value.detach().cpu())
                    for name, value in components.items()
                },
            }
        )

    # Primary endpoint freezes every representation parameter and both EMA
    # codebooks; only the registered ablation heads are fitted.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head_parameters = []
    for module in head_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            head_parameters.append(parameter)
    for parameter in lag_module.parameters():
        parameter.requires_grad_(False)
    head_optimizer = torch.optim.AdamW(
        head_parameters,
        lr=float(config["training"]["head_learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    auxiliary = float(config["head"]["ablation_auxiliary_cross_entropy_weight"])
    iterator = iter(_iterate_loader(train_loader))
    model.eval()
    for module in head_modules:
        module.train()
    for step in range(_fixed_steps(smoke, config, train_loader, "head")):
        batch = _batch_to_device(next(iterator), device)
        head_optimizer.zero_grad(set_to_none=True)
        output = _forward_lc_spvq(model, batch, variant=variant)
        head_loss = torch.nn.functional.cross_entropy(
            output["combined_logits"], batch["target"]
        ) + auxiliary * (
            torch.nn.functional.cross_entropy(
                output["coupling_only_logits"], batch["target"]
            )
            + torch.nn.functional.cross_entropy(
                output["shared_marginal_only_logits"], batch["target"]
            )
            + torch.nn.functional.cross_entropy(
                output["private_only_logits"], batch["target"]
            )
        )
        head_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            head_parameters, float(config["training"]["grad_clip_norm"])
        )
        head_optimizer.step()
        history.append(
            {
                "stage": "task_head",
                "step": step,
                "total_loss": float(head_loss.detach().cpu()),
                "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
            }
        )

    split_metrics = {}
    split_health = {}
    split_arrays = {}
    for role, partition in (
        ("fit_parameter", prepared.parameter),
        ("fit_selection", prepared.selection),
        ("development_apply", prepared.development),
    ):
        metrics, arrays, health = _evaluate_lc_spvq(
            model, partition, config=config, device=device, seed=seed
        )
        split_metrics[role] = metrics
        split_health[role] = health
        for name, value in arrays.items():
            split_arrays[f"{role}__{name}"] = value
    np.savez_compressed(
        output_dir / "token_exports.npz",
        schema=np.asarray("lc_spvq_token_exports_v2"),
        task_id=np.asarray(prepared.task_id),
        variant=np.asarray(variant),
        seed=np.asarray(seed, dtype=np.int64),
        development_is_new_independent_holdout=np.asarray(False),
        protected_open=np.asarray(prepared.protected_measured_access_count != 0),
        derangement_nonoverlap_verified=np.asarray(
            all(
                _partition_derangement_nonoverlap_verified(partition)
                for partition in (
                    prepared.parameter,
                    prepared.selection,
                    prepared.development,
                )
            )
        ),
        registered_hard_negative_policy=np.asarray(
            "same_subject_condition_nonidentity_same_token_time"
        ),
        **split_arrays,
    )
    _atomic_torch_save(
        {
            "schema": "lc_spvq_checkpoint_v1",
            "task_id": prepared.task_id,
            "variant": variant,
            "seed": int(seed),
            "model_state": model.state_dict(),
            "lag_objective_state": lag_module.state_dict(),
            "class_names": list(TASK_SPECS[prepared.task_id].class_names),
            "allowed_lag_tokens": list(model.allowed_lags),
            "protected_open": prepared.protected_measured_access_count != 0,
        },
        output_dir / "checkpoint.pt",
    )
    _write_csv(output_dir / "loss_curves.csv", history)
    result = {
        "schema": "lc_spvq_variant_result_v1",
        "task_id": prepared.task_id,
        "variant": variant,
        "seed": int(seed),
        "status": "completed",
        "positive_pairing": config["variants"][variant]["positive_pairing"],
        "kmeans_initialization": kmeans,
        "lag_tokens": list(model.allowed_lags),
        "lag_weights": [
            float(value)
            for value in lag_module.lag_mixture_weights.detach().cpu().tolist()
        ],
        "quantization_strength_final": model.get_quantization_strength(),
        "posterior_temperature_final": model.get_posterior_temperature(),
        "fit_selection_metrics": split_metrics["fit_selection"],
        "development_metrics": split_metrics["development_apply"],
        "codebook_health": split_health,
        "development_is_new_independent_holdout": False,
        "protected_open": prepared.protected_measured_access_count != 0,
        "protected_measured_access_count": prepared.protected_measured_access_count,
    }
    _write_json(output_dir / "result.json", result)
    return result


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if path.name == "manifest.json" and path.parent == root:
            continue
        output.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return output


def _runtime_provenance_inputs(
    config: Mapping[str, Any],
    config_path: Path,
    *,
    include_models: bool,
) -> list[dict[str, Any]]:
    runtime_paths = [
        Path(__file__).resolve(),
        REPO_ROOT / "src/data/unified_physiology.py",
        REPO_ROOT / "src/data/clean_physiology_cache.py",
        REPO_ROOT / "src/data/eeg_artifact_preprocessing.py",
        REPO_ROOT / "src/data/lag_conditioned_dataset.py",
        REPO_ROOT / "src/analysis/physiological_patch_features.py",
        REPO_ROOT / "src/analysis/lag_conditioned_native_features.py",
    ]
    if include_models:
        runtime_paths.extend(
            [
                REPO_ROOT / "src/tokenizers/continuous_shared_private.py",
                REPO_ROOT / "src/tokenizers/lag_conditioned_baseline.py",
                REPO_ROOT / "src/tokenizers/lag_conditioned_shared_private_vq.py",
                REPO_ROOT / "src/tokenizers/ema_vector_quantizer.py",
                REPO_ROOT / "src/losses/lag_conditioned.py",
                REPO_ROOT / "src/metrics/codebook_health.py",
                REPO_ROOT / "src/metrics/lag_conditioned_downstream.py",
            ]
        )
    cache_root = _resolve(config["source"]["cache_root"])
    cache_manifests = [
        cache_root / "cache_manifest.json",
        cache_root / "eeg_artifact_clean_v4/cache_manifest.json",
        cache_root / "simultaneous_eeg_eog_clean_v1/cache_manifest.json",
    ]
    entries = [(Path(config_path).resolve(), "configuration")]
    entries.extend((path, "runtime_module") for path in runtime_paths)
    entries.extend((path, "cache_manifest") for path in cache_manifests)
    output: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path, kind in entries:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.is_file():
            raise FileNotFoundError(f"provenance input is missing: {resolved}")
        try:
            display = str(resolved.relative_to(REPO_ROOT))
        except ValueError:
            display = str(resolved)
        output.append(
            {
                "path": display,
                "sha256": _sha256(resolved),
                "source_kind": kind,
            }
        )
    return output


def run_preparation_only(
    config: Mapping[str, Any],
    config_path: Path,
    target: Path,
    *,
    smoke: bool,
) -> Path:
    config_path = _validate_bound_config(config, config_path)
    _require_smoke_training(smoke)
    if target.exists():
        raise FileExistsError(f"refusing overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        shutil.copy2(config_path, staging / "config.yaml")
        tasks = (
            list(map(str, config["smoke"]["tasks"]))
            if smoke
            else [row["task_id"] for row in config["tasks"] if row["first_round"]]
        )
        summaries = []
        for task_index, task_id in enumerate(tasks):
            task_dir = staging / "tasks" / task_id
            task_dir.mkdir(parents=True)
            prepared = prepare_task(
                config,
                task_id,
                smoke=smoke,
                derangement_seed=int(config["training"]["seeds"][0]),
            )
            summaries.append(save_preparation(prepared, task_dir))
            print(f"prepared {task_id} ({task_index + 1}/{len(tasks)})", flush=True)
        manifest = {
            "schema": SCHEMA,
            "status": "preparation_completed",
            "mode": "smoke" if smoke else "first_round_preparation",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment": config["experiment"],
            "task_count": len(tasks),
            "tasks": tasks,
            "variants": [],
            "model_training_started": False,
            "protected_open": any(
                int(row["protected_measured_access_count"]) != 0 for row in summaries
            ),
            "protected_metadata_indexed_by_unified_loader": all(
                bool(row["protected_metadata_indexed_by_unified_loader"])
                for row in summaries
            ),
            "protected_measured_access": any(
                int(row["protected_measured_access_count"]) != 0 for row in summaries
            ),
            "protected_sample_getitem_calls": sum(
                int(row["protected_measured_access_count"]) for row in summaries
            ),
            "measured_sample_access_count": sum(
                int(row["measured_sample_access_count"]) for row in summaries
            ),
            "derangement_nonoverlap_verified": all(
                bool(row["derangement_nonoverlap_verified"]) for row in summaries
            ),
            "development_is_new_independent_holdout": False,
            "old_continuous_verdict_modified": False,
            "preparation": summaries,
            "git": _git_payload(),
            "inputs": _runtime_provenance_inputs(
                config, config_path, include_models=False
            ),
            "artifacts": _artifact_inventory(staging),
        }
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, target)
        return target
    except Exception:
        print(f"failed staging retained at {staging}", file=sys.stderr)
        raise


def run_training_suite(
    config: Mapping[str, Any],
    config_path: Path,
    target: Path,
    *,
    smoke: bool,
    requested_tasks: Sequence[str] | None,
    requested_variants: Sequence[str] | None,
    requested_device: str | None,
) -> Path:
    config_path = _validate_bound_config(config, config_path)
    _require_smoke_training(smoke)
    if target.exists():
        raise FileExistsError(f"refusing overwrite: {target}")
    tasks = (
        list(map(str, requested_tasks))
        if requested_tasks
        else (
            list(map(str, config["smoke"]["tasks"]))
            if smoke
            else [row["task_id"] for row in config["tasks"] if row["first_round"]]
        )
    )
    variants = (
        list(map(str, requested_variants))
        if requested_variants
        else (
            list(map(str, config["smoke"]["variants"]))
            if smoke
            else list(map(str, config["variants"]["first_round_order"]))
        )
    )
    if any(task not in TASK_SPECS for task in tasks):
        raise ValueError(f"unknown requested task in {tasks}")
    if any(variant not in VARIANT_ORDER for variant in variants):
        raise ValueError(f"unknown requested variant in {variants}")
    seeds = list(map(int, config["training"]["seeds"]))
    if smoke:
        seeds = seeds[: int(config["smoke"]["seeds"])]
    device = torch.device(requested_device or str(config["training"]["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        shutil.copy2(config_path, staging / "config.yaml")
        results = []
        preparations = []
        completed = 0
        total = len(tasks) * len(variants) * len(seeds)
        for task_id in tasks:
            task_dir = staging / "tasks" / task_id
            task_dir.mkdir(parents=True)
            prepared = prepare_task(
                config,
                task_id,
                smoke=smoke,
                derangement_seed=seeds[0],
            )
            preparations.append(save_preparation(prepared, task_dir))
            for variant in variants:
                for seed in seeds:
                    cell_dir = task_dir / variant / f"seed_{seed}"
                    cell_dir.mkdir(parents=True)
                    if variant == "B0":
                        result = train_b0_variant(
                            prepared,
                            config,
                            seed=seed,
                            device=device,
                            output_dir=cell_dir,
                            smoke=smoke,
                        )
                    elif variant in {"M1", "N1"}:
                        result = train_lc_spvq_variant(
                            prepared,
                            config,
                            variant=variant,
                            seed=seed,
                            device=device,
                            output_dir=cell_dir,
                            smoke=smoke,
                        )
                    else:  # choices and validation make this unreachable
                        raise RuntimeError(f"unreachable variant {variant}")
                    results.append(result)
                    completed += 1
                    print(
                        f"completed {task_id}/{variant}/seed={seed} "
                        f"({completed}/{total})",
                        flush=True,
                    )
        _write_json(staging / "results.json", {"results": results})
        manifest = {
            "schema": SCHEMA,
            "status": "completed",
            "mode": "smoke" if smoke else "first_round_development",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment": config["experiment"],
            "tasks": tasks,
            "variants": variants,
            "seeds": seeds,
            "cell_count": total,
            "completed_cell_count": completed,
            "failed_cell_count": 0,
            "model_training_started": True,
            "preparation": preparations,
            "protected_open": any(
                int(row["protected_measured_access_count"]) != 0
                for row in preparations
            ),
            "protected_metadata_indexed_by_unified_loader": all(
                bool(row["protected_metadata_indexed_by_unified_loader"])
                for row in preparations
            ),
            "protected_measured_access": any(
                int(row["protected_measured_access_count"]) != 0
                for row in preparations
            ),
            "protected_sample_getitem_calls": sum(
                int(row["protected_measured_access_count"]) for row in preparations
            ),
            "measured_sample_access_count": sum(
                int(row["measured_sample_access_count"]) for row in preparations
            ),
            "derangement_nonoverlap_verified": all(
                bool(row["derangement_nonoverlap_verified"])
                for row in preparations
            ),
            "development_is_new_independent_holdout": False,
            "old_continuous_verdict_modified": False,
            "git": _git_payload(),
            "inputs": _runtime_provenance_inputs(
                config, config_path, include_models=True
            ),
            "artifacts": _artifact_inventory(staging),
        }
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, target)
        return target
    except Exception:
        print(f"failed staging retained at {staging}", file=sys.stderr)
        raise


def run(args: argparse.Namespace) -> Path:
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    _set_seed(int(config["training"]["seeds"][0]))
    target = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _resolve(config["output"]["root"])
        / (
            datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_"
            + str(config["experiment"]["name"])
            + ("_smoke" if args.smoke else "")
        )
    )
    if args.prepare_only:
        return run_preparation_only(
            config, config_path, target, smoke=bool(args.smoke)
        )
    return run_training_suite(
        config,
        config_path,
        target,
        smoke=bool(args.smoke),
        requested_tasks=args.tasks,
        requested_variants=args.variants,
        requested_device=args.device,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "experiments/configs/physiology_semantic_tokenizer/lag_conditioned_spvq.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASK_SPECS))
    parser.add_argument("--variants", nargs="+", choices=VARIANT_ORDER)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
