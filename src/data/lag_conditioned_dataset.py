"""Leakage-aware 20-second task views for lag-conditioned EEG-fNIRS models.

This module defines metadata filtering and measured-sample access for the new
LC-SPVQ development generation.  It deliberately does not reuse the frozen SSM
trajectory targets.  Subject and task filtering are completed before
``UnifiedPhysiologyWindowDataset.__getitem__`` is called, so a forbidden subject
is never dereferenced through this interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.unified_physiology import (
    DEFAULT_ADMISSIBLE_ALIGNMENT_CASES,
    UnifiedPhysiologyWindowDataset,
    canonical_label,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_LC_CACHE_ROOT = (
    _REPO_ROOT / "data/cache/physiology_semantic_clean_v1"
).resolve()
REVIEWED_EEG_SIGNAL_BRANCH = "single_trial_eeg_artifact_clean_v4"


@dataclass(frozen=True)
class LagConditionedTaskSpec:
    """One classification and coupling task on a common 20-second window."""

    task_id: str
    dataset_id: str
    namespace: str
    class_names: tuple[str, ...]
    record_ids: tuple[str, ...]
    window_duration_s: float = 20.0
    window_offset_s: float = -5.0
    eeg_sample_rate_hz: float = 200.0
    fnirs_sample_rate_hz: float = 10.0
    patch_duration_s: float = 2.0

    @property
    def num_tokens(self) -> int:
        ratio = self.window_duration_s / self.patch_duration_s
        if not float(ratio).is_integer():
            raise ValueError("window duration must be divisible by patch duration")
        return int(ratio)

    @property
    def eeg_patch_samples(self) -> int:
        return int(round(self.eeg_sample_rate_hz * self.patch_duration_s))

    @property
    def fnirs_patch_samples(self) -> int:
        return int(round(self.fnirs_sample_rate_hz * self.patch_duration_s))


TASK_SPECS: Mapping[str, LagConditionedTaskSpec] = MappingProxyType({
    "motor_imagery": LagConditionedTaskSpec(
        task_id="motor_imagery",
        dataset_id="eeg_fnirs_single_trial",
        namespace="eeg_fnirs_single_trial:motor_imagery",
        class_names=("LMI", "RMI"),
        record_ids=("session_00", "session_02", "session_04"),
    ),
    "mental_arithmetic": LagConditionedTaskSpec(
        task_id="mental_arithmetic",
        dataset_id="eeg_fnirs_single_trial",
        namespace="eeg_fnirs_single_trial:mental_arithmetic",
        class_names=("MA", "BL"),
        record_ids=("session_01", "session_03", "session_05"),
    ),
    "word_generation": LagConditionedTaskSpec(
        task_id="word_generation",
        dataset_id="simultaneous_eeg_nirs",
        namespace="simultaneous_eeg_nirs:wg",
        class_names=("WG", "BL"),
        record_ids=("cnt_wg",),
    ),
    "n_back": LagConditionedTaskSpec(
        task_id="n_back",
        dataset_id="simultaneous_eeg_nirs",
        namespace="simultaneous_eeg_nirs:nback",
        class_names=("0-back session", "2-back session", "3-back session"),
        record_ids=("cnt_nback",),
    ),
})


CANONICAL_PROTECTED_SUBJECTS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "eeg_fnirs_single_trial": tuple(
            f"subject_{index:02d}" for index in range(24, 30)
        ),
        "simultaneous_eeg_nirs": tuple(
            f"VP{index:03d}" for index in range(24, 27)
        ),
    }
)


@dataclass(frozen=True)
class LagConditionedSampleIndex:
    """Metadata-only pointer to one admitted unified-loader window."""

    base_index: int
    sample_id: str
    dataset_id: str
    task_id: str
    subject: str
    record_id: str
    condition: str
    class_index: int
    event_index: int
    event_time_ms: float
    fnirs_event_time_ms: float


def get_task_spec(task_id: str) -> LagConditionedTaskSpec:
    try:
        return TASK_SPECS[str(task_id)]
    except KeyError as exc:
        raise KeyError(
            f"unknown lag-conditioned task {task_id!r}; expected {sorted(TASK_SPECS)}"
        ) from exc


def _canonical_sample_id(
    *,
    dataset_id: str,
    subject: str,
    record_id: str,
    event_index: int,
    window_offset_s: float,
    window_duration_s: float,
) -> str:
    return (
        f"{dataset_id}|{subject}|{record_id}|event={event_index}|"
        f"offset={window_offset_s:g}|duration={window_duration_s:g}"
    )


def build_admitted_index(
    windows: Sequence[Any],
    spec: LagConditionedTaskSpec,
    *,
    admitted_subjects: Sequence[str],
    forbidden_subjects: Sequence[str] = (),
) -> tuple[LagConditionedSampleIndex, ...]:
    """Filter window metadata without dereferencing measured sample arrays."""

    admitted = {str(value) for value in admitted_subjects}
    forbidden = {str(value) for value in forbidden_subjects}
    if not admitted:
        raise ValueError("admitted_subjects must be non-empty")
    overlap = admitted & forbidden
    if overlap:
        raise PermissionError(
            f"admitted and forbidden subject sets overlap: {sorted(overlap)}"
        )
    class_to_index = {name: index for index, name in enumerate(spec.class_names)}
    rows: list[LagConditionedSampleIndex] = []
    for base_index, ref in enumerate(windows):
        subject = str(ref.record.canonical_subject_id)
        if subject not in admitted:
            continue
        if subject in forbidden:
            raise PermissionError(f"forbidden subject reached admitted index: {subject}")
        record_id = str(ref.record.base_record_id)
        if record_id not in spec.record_ids:
            continue
        label = canonical_label(ref.event, spec.dataset_id)
        if str(label["namespace"]) != spec.namespace:
            continue
        condition = str(label["condition"])
        if condition not in class_to_index:
            continue
        event_index = int(ref.event.get("event_index", -1))
        window_shift_ms = 1000.0 * float(getattr(ref, "window_offset_s", 0.0))
        event_time_ms = float(
            ref.event.get("eeg_time_ms", ref.event.get("onset_ms", float("nan")))
        ) + window_shift_ms
        fnirs_event_time_ms = float(
            ref.event.get("fnirs_time_ms", ref.event.get("onset_ms", float("nan")))
        ) + window_shift_ms
        if not np.isfinite(event_time_ms) or not np.isfinite(fnirs_event_time_ms):
            raise ValueError(f"event {event_index} lacks finite paired timing metadata")
        sample_id = _canonical_sample_id(
            dataset_id=spec.dataset_id,
            subject=subject,
            record_id=record_id,
            event_index=event_index,
            window_offset_s=spec.window_offset_s
            + float(getattr(ref, "window_offset_s", 0.0)),
            window_duration_s=spec.window_duration_s,
        )
        rows.append(
            LagConditionedSampleIndex(
                base_index=int(base_index),
                sample_id=sample_id,
                dataset_id=spec.dataset_id,
                task_id=spec.task_id,
                subject=subject,
                record_id=record_id,
                condition=condition,
                class_index=class_to_index[condition],
                event_index=event_index,
                event_time_ms=event_time_ms,
                fnirs_event_time_ms=fnirs_event_time_ms,
            )
        )
    rows.sort(key=lambda row: row.sample_id)
    if not rows:
        raise RuntimeError(f"no admitted samples for task {spec.task_id}")
    identities = [row.sample_id for row in rows]
    if len(identities) != len(set(identities)):
        raise RuntimeError("lag-conditioned task index contains duplicate sample IDs")
    observed_classes = {row.condition for row in rows}
    if observed_classes != set(spec.class_names):
        raise RuntimeError(
            f"task {spec.task_id} class support differs from contract: "
            f"{sorted(observed_classes)} != {sorted(spec.class_names)}"
        )
    return tuple(rows)


def _windows_overlap(
    first: LagConditionedSampleIndex,
    second: LagConditionedSampleIndex,
) -> bool:
    if first.record_id != second.record_id:
        return False
    if first.task_id != second.task_id:
        raise ValueError("donor candidates must belong to one task")
    duration_ms = 1000.0 * get_task_spec(first.task_id).window_duration_s
    eeg_overlap = (
        abs(float(first.event_time_ms) - float(second.event_time_ms)) < duration_ms
    )
    fnirs_overlap = (
        abs(float(first.fnirs_event_time_ms) - float(second.fnirs_event_time_ms))
        < duration_ms
    )
    return eeg_overlap or fnirs_overlap


def make_group_derangement(
    rows: Sequence[LagConditionedSampleIndex],
    *,
    seed: int,
) -> np.ndarray:
    """Match nonidentity, nonoverlapping donors within subject and condition."""

    groups: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        if not np.isfinite(float(row.event_time_ms)) or not np.isfinite(
            float(row.fnirs_event_time_ms)
        ):
            raise ValueError(f"row {row.sample_id} lacks finite paired event timing")
        groups.setdefault((row.subject, row.condition), []).append(index)
    donor = np.full(len(rows), -1, dtype=np.int64)
    for group, indices in groups.items():
        if len(indices) < 2:
            raise ValueError(f"derangement group has fewer than two samples: {group}")
        target_order = sorted(
            indices,
            key=lambda index: hashlib.sha256(
                f"target|{int(seed)}|{rows[index].sample_id}".encode("utf-8")
            ).hexdigest(),
        )
        candidates = {
            target: sorted(
                [
                    source
                    for source in indices
                    if source != target and not _windows_overlap(rows[target], rows[source])
                ],
                key=lambda source: hashlib.sha256(
                    (
                        f"donor|{int(seed)}|{rows[target].sample_id}|"
                        f"{rows[source].sample_id}"
                    ).encode("utf-8")
                ).hexdigest(),
            )
            for target in target_order
        }
        if any(not values for values in candidates.values()):
            raise ValueError(
                f"derangement group lacks nonoverlapping donor support: {group}"
            )

        assignment: dict[int, int] = {}

        def assign(position: int, used: set[int]) -> bool:
            if position == len(target_order):
                return True
            target = target_order[position]
            for source in candidates[target]:
                if source in used:
                    continue
                assignment[target] = source
                used.add(source)
                if assign(position + 1, used):
                    return True
                used.remove(source)
                assignment.pop(target, None)
            return False

        if not assign(0, set()):
            raise ValueError(
                f"derangement group has no complete nonoverlapping matching: {group}"
            )
        for target, source in assignment.items():
            donor[target] = source
    if np.any(donor < 0) or np.any(donor == np.arange(len(rows))):
        raise RuntimeError("failed to construct a complete non-identity derangement")
    if len(set(donor.tolist())) != len(rows):
        raise RuntimeError("derangement donors are not a permutation")
    for target, source in enumerate(donor):
        if rows[target].subject != rows[int(source)].subject:
            raise RuntimeError("derangement changed subject")
        if rows[target].condition != rows[int(source)].condition:
            raise RuntimeError("derangement changed condition")
        if _windows_overlap(rows[target], rows[int(source)]):
            raise RuntimeError("derangement paired overlapping windows")
    return donor


def _point_support(sample: Mapping[str, Any], modality: str) -> np.ndarray:
    recorded = np.asarray(sample["valid_mask"][modality], dtype=bool)
    analysis = np.asarray(
        sample.get("analysis_valid_mask", {}).get(modality, recorded), dtype=bool
    )
    if recorded.shape != analysis.shape:
        raise ValueError(f"{modality} recorded/analysis mask shape mismatch")
    return recorded & analysis


def _token_support(point_support: np.ndarray, patch_samples: int) -> np.ndarray:
    if point_support.ndim != 1 or len(point_support) % int(patch_samples):
        raise ValueError("point support cannot be partitioned into token patches")
    return point_support.reshape(-1, int(patch_samples)).all(axis=1)


def _resolve_reviewed_cache_root(cache_root: str | Path) -> Path:
    path = Path(cache_root).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    resolved = path.resolve()
    if resolved != CANONICAL_LC_CACHE_ROOT:
        raise PermissionError(
            "lag-conditioned measured access requires the repository canonical cache"
        )
    return resolved


def _validate_reviewed_base_contract(
    base: UnifiedPhysiologyWindowDataset,
    *,
    spec: LagConditionedTaskSpec,
) -> None:
    if not isinstance(base, UnifiedPhysiologyWindowDataset):
        raise TypeError(
            "base_dataset must be a reviewed UnifiedPhysiologyWindowDataset instance"
        )
    base_cache_root = Path(base.cache_root).expanduser()
    eeg_artifact_cache_root = Path(base.eeg_artifact_cache_root).expanduser()
    simultaneous_eeg_cache_root = Path(base.simultaneous_eeg_cache_root).expanduser()
    checks = {
        "cache_root_absolute": base_cache_root.is_absolute(),
        "cache_root": base_cache_root.resolve() == CANONICAL_LC_CACHE_ROOT,
        "eeg_artifact_cache_root_absolute": eeg_artifact_cache_root.is_absolute(),
        "eeg_artifact_cache_root": eeg_artifact_cache_root.resolve()
        == (CANONICAL_LC_CACHE_ROOT / "eeg_artifact_clean_v4").resolve(),
        "simultaneous_eeg_cache_root_absolute": simultaneous_eeg_cache_root.is_absolute(),
        "simultaneous_eeg_cache_root": simultaneous_eeg_cache_root.resolve()
        == (CANONICAL_LC_CACHE_ROOT / "simultaneous_eeg_eog_clean_v1").resolve(),
        "eeg_artifact_config": base.eeg_artifact_config is None,
        "project_root": Path(base.project_root).resolve() == _REPO_ROOT.resolve(),
        "dataset_ids": tuple(base.dataset_ids) == (spec.dataset_id,),
        "window_duration_s": float(base.window_duration_s) == 20.0,
        "window_offset_s": float(base.window_offset_s) == -5.0,
        "eeg_signal_branch": str(base.eeg_signal_branch)
        == REVIEWED_EEG_SIGNAL_BRANCH,
        "require_eeg_artifact_cache": bool(base.require_eeg_artifact_cache)
        == (spec.dataset_id == "eeg_fnirs_single_trial"),
        "require_paired_timestamps": bool(base.require_paired_timestamps),
        "alignment_support": frozenset(base.admissible_alignment_cases or ())
        == DEFAULT_ADMISSIBLE_ALIGNMENT_CASES,
        "event_filter": base.include_event_types is None,
    }
    failed = sorted(name for name, valid in checks.items() if not valid)
    if failed:
        raise PermissionError(
            "base dataset leaves the reviewed LC-SPVQ source contract: "
            + ", ".join(failed)
        )


class LagConditionedTaskDataset(Dataset):
    """Measured paired windows with a frozen subject/task access boundary."""

    def __init__(
        self,
        *,
        task_id: str,
        admitted_subjects: Sequence[str],
        forbidden_subjects: Sequence[str] | None = None,
        cache_root: str | Path = "data/cache/physiology_semantic_clean_v1",
        eeg_signal_branch: str = REVIEWED_EEG_SIGNAL_BRANCH,
        base_dataset: UnifiedPhysiologyWindowDataset | None = None,
    ) -> None:
        self.spec = get_task_spec(task_id)
        self.admitted_subjects = tuple(sorted({str(value) for value in admitted_subjects}))
        required_forbidden = set(CANONICAL_PROTECTED_SUBJECTS[self.spec.dataset_id])
        supplied_forbidden = (
            required_forbidden
            if forbidden_subjects is None
            else {str(value) for value in forbidden_subjects}
        )
        if supplied_forbidden != required_forbidden:
            raise PermissionError(
                "lag-conditioned measured access requires the immutable canonical "
                f"protected set {sorted(required_forbidden)}"
            )
        self.forbidden_subjects = tuple(sorted(required_forbidden))
        if set(self.admitted_subjects) & set(self.forbidden_subjects):
            raise PermissionError("admitted subjects overlap forbidden subjects")
        resolved_cache_root = _resolve_reviewed_cache_root(cache_root)
        if str(eeg_signal_branch) != REVIEWED_EEG_SIGNAL_BRANCH:
            raise PermissionError("lag-conditioned measured access requires the reviewed EEG branch")
        self.base = (
            base_dataset
            if base_dataset is not None
            else UnifiedPhysiologyWindowDataset(
                cache_root=resolved_cache_root,
                dataset_ids=(self.spec.dataset_id,),
                window_duration_s=self.spec.window_duration_s,
                window_offset_s=self.spec.window_offset_s,
                eeg_signal_branch=REVIEWED_EEG_SIGNAL_BRANCH,
                require_eeg_artifact_cache=self.spec.dataset_id
                == "eeg_fnirs_single_trial",
            )
        )
        _validate_reviewed_base_contract(self.base, spec=self.spec)
        self.rows = build_admitted_index(
            self.base.windows,
            self.spec,
            admitted_subjects=self.admitted_subjects,
            forbidden_subjects=self.forbidden_subjects,
        )
        self.derangement: np.ndarray | None = None
        self.measured_access_count = 0
        self.protected_measured_access_count = 0

    def validate_governance_contract(self) -> None:
        _validate_reviewed_base_contract(self.base, spec=self.spec)
        protected = set(CANONICAL_PROTECTED_SUBJECTS[self.spec.dataset_id])
        if set(self.forbidden_subjects) != protected:
            raise PermissionError("dataset protected boundary changed after construction")
        if set(self.admitted_subjects) & protected:
            raise PermissionError("dataset admitted subjects cross the protected boundary")

    def set_derangement(self, *, seed: int) -> None:
        self.derangement = make_group_derangement(self.rows, seed=int(seed))

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, index: int) -> tuple[LagConditionedSampleIndex, Mapping[str, Any]]:
        self.validate_governance_contract()
        row = self.rows[int(index)]
        if row.subject in set(self.forbidden_subjects):
            self.protected_measured_access_count += 1
            raise PermissionError(f"refusing measured access for {row.subject}")
        sample = self.base[row.base_index]
        self.measured_access_count += 1
        if str(sample.get("dataset_id")) != self.spec.dataset_id:
            raise RuntimeError("metadata/sample dataset identity drift")
        expected_sample_branch = (
            REVIEWED_EEG_SIGNAL_BRANCH
            if self.spec.dataset_id == "eeg_fnirs_single_trial"
            else "simultaneous_eeg_eog_clean_v1"
        )
        if str(sample.get("eeg_signal_branch")) != expected_sample_branch:
            raise RuntimeError("metadata/sample EEG branch identity drift")
        sample_rates = sample.get("sample_rate_hz", {})
        if float(sample_rates.get("eeg", float("nan"))) != 200.0 or float(
            sample_rates.get("fnirs", float("nan"))
        ) != 10.0:
            raise RuntimeError("metadata/sample canonical rate drift")
        if str(sample["subject"]) != row.subject:
            raise RuntimeError("metadata/sample subject identity drift")
        if str(sample["record_id"]) != row.record_id:
            raise RuntimeError("metadata/sample record identity drift")
        event = sample.get("event", {})
        if int(event.get("event_index", -1)) != int(row.event_index):
            raise RuntimeError("metadata/sample event identity drift")
        alignment = sample.get("alignment", {})
        expected_eeg_start = row.event_time_ms + self.spec.window_offset_s * 1000.0
        expected_fnirs_start = (
            row.fnirs_event_time_ms + self.spec.window_offset_s * 1000.0
        )
        if not np.isclose(
            float(alignment.get("eeg_time_ms", float("nan"))),
            expected_eeg_start,
            rtol=0.0,
            atol=1e-6,
        ) or not np.isclose(
            float(alignment.get("fnirs_time_ms", float("nan"))),
            expected_fnirs_start,
            rtol=0.0,
            atol=1e-6,
        ):
            raise RuntimeError("metadata/sample event timing drift")
        label = sample["label"]
        if str(label["condition"]) != row.condition:
            raise RuntimeError("metadata/sample condition identity drift")
        return row, sample

    def _tensor_sample(self, index: int) -> dict[str, Any]:
        row, sample = self._load(index)
        eeg = np.asarray(sample["eeg"], dtype=np.float32)
        fnirs = np.asarray(sample["fnirs"], dtype=np.float32)
        eeg_point = _point_support(sample, "eeg")
        fnirs_point = _point_support(sample, "fnirs")
        if eeg.shape[-1] != len(eeg_point) or fnirs.shape[-1] != len(fnirs_point):
            raise ValueError("signal and point-support lengths differ")
        eeg_token = _token_support(eeg_point, self.spec.eeg_patch_samples)
        fnirs_token = _token_support(fnirs_point, self.spec.fnirs_patch_samples)
        if len(eeg_token) != self.spec.num_tokens or len(fnirs_token) != self.spec.num_tokens:
            raise ValueError("token count differs from lag-conditioned task contract")
        eeg_channel_valid = ~np.asarray(sample["bad_channel_mask"]["eeg"], dtype=bool)
        fnirs_channel_valid = ~np.asarray(sample["bad_channel_mask"]["fnirs"], dtype=bool)
        if eeg_channel_valid.shape != (eeg.shape[0],):
            raise ValueError("EEG bad-channel mask shape mismatch")
        if fnirs_channel_valid.shape != (fnirs.shape[0],):
            raise ValueError("fNIRS bad-channel mask shape mismatch")
        return {
            "index": torch.tensor(int(index), dtype=torch.long),
            "eeg": torch.from_numpy(eeg.copy()),
            "fnirs": torch.from_numpy(fnirs.copy()),
            "eeg_point_valid_mask": torch.from_numpy(eeg_point.copy()),
            "fnirs_point_valid_mask": torch.from_numpy(fnirs_point.copy()),
            "eeg_token_valid_mask": torch.from_numpy(eeg_token.copy()),
            "fnirs_token_valid_mask": torch.from_numpy(fnirs_token.copy()),
            "eeg_channel_valid_mask": torch.from_numpy(eeg_channel_valid.copy()),
            "fnirs_channel_valid_mask": torch.from_numpy(fnirs_channel_valid.copy()),
            "target": torch.tensor(row.class_index, dtype=torch.long),
            "sample_id": row.sample_id,
            "subject": row.subject,
            "condition": row.condition,
            "record_id": row.record_id,
            "eeg_channel_names": tuple(map(str, sample["channel_names"]["eeg"])),
            "fnirs_channel_names": tuple(map(str, sample["channel_names"]["fnirs"])),
            "fnirs_component_roles": tuple(
                map(str, sample["component_roles"]["fnirs"])
            ),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        output = self._tensor_sample(int(index))
        if self.derangement is not None:
            donor_index = int(self.derangement[int(index)])
            _, donor = self._load(donor_index)
            output["donor_index"] = torch.tensor(donor_index, dtype=torch.long)
            output["donor_sample_id"] = self.rows[donor_index].sample_id
            output["donor_fnirs"] = torch.from_numpy(
                np.asarray(donor["fnirs"], dtype=np.float32).copy()
            )
            donor_point = _point_support(donor, "fnirs")
            output["donor_fnirs_point_valid_mask"] = torch.from_numpy(
                donor_point.copy()
            )
            output["donor_fnirs_token_valid_mask"] = torch.from_numpy(
                _token_support(donor_point, self.spec.fnirs_patch_samples).copy()
            )
        return output

    def contract_summary(self) -> dict[str, Any]:
        by_class = {
            name: sum(row.condition == name for row in self.rows)
            for name in self.spec.class_names
        }
        return {
            "schema": "lag_conditioned_task_dataset_v1",
            "task_id": self.spec.task_id,
            "dataset_id": self.spec.dataset_id,
            "namespace": self.spec.namespace,
            "class_names": list(self.spec.class_names),
            "record_ids": list(self.spec.record_ids),
            "window_duration_s": self.spec.window_duration_s,
            "window_offset_s": self.spec.window_offset_s,
            "patch_duration_s": self.spec.patch_duration_s,
            "num_tokens": self.spec.num_tokens,
            "sample_count": len(self.rows),
            "subjects": sorted({row.subject for row in self.rows}),
            "class_counts": by_class,
            "forbidden_subjects": list(self.forbidden_subjects),
            "measured_access_count": int(self.measured_access_count),
            "forbidden_measured_access": int(self.protected_measured_access_count),
            "artifact_mask_used_as_validity": False,
            "support_policy": "recorded_and_analysis_valid_intersection",
        }


__all__ = [
    "CANONICAL_LC_CACHE_ROOT",
    "CANONICAL_PROTECTED_SUBJECTS",
    "REVIEWED_EEG_SIGNAL_BRANCH",
    "LagConditionedSampleIndex",
    "LagConditionedTaskDataset",
    "LagConditionedTaskSpec",
    "TASK_SPECS",
    "build_admitted_index",
    "get_task_spec",
    "make_group_derangement",
]
