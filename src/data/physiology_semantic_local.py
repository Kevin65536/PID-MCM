"""Measurement-first local views for the physiology-semantic tokenizer.

The adapter consumes :class:`UnifiedPhysiologyWindowDataset` samples and only
then constructs the existing six-EEG/two-chromophore model view.  It never
reads a Croce cache or substitutes a teacher target for a measured signal.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .unified_physiology import UnifiedPhysiologyWindowDataset, canonical_label


LOCAL_VIEW_SCHEMA = "physiology_semantic_measurement_local_v1"


@dataclass(frozen=True)
class LocalWindowEntry:
    base_index: int
    dataset_id: str
    subject_key: str
    dependency_group_id: str


def _position(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[row.get(axis) for axis in ("x", "y", "z")] for row in rows],
        dtype=np.float64,
    )


def _visual_dependency_group(record_id: str, event: Mapping[str, Any]) -> str:
    record = str(record_id).replace("_Probe1", "").replace("_Probe2", "")
    event_index = event.get("event_index", event.get("epoch_id", -1))
    return f"{record}|event={event_index}"


class UnifiedPhysiologyLocalViewDataset(Dataset):
    """Select one deterministic measured local view from every unified window."""

    def __init__(
        self,
        cache_root: str = "data/cache/physiology_semantic_clean_v1",
        *,
        dataset_ids: Sequence[str] = ("eeg_fnirs_single_trial",),
        subject_keys: Iterable[str] | None = None,
        task_namespaces: Iterable[str] | None = None,
        window_duration_s: float = 20.0,
        local_eeg_channels: int = 6,
        reject_unknown_labels: bool = True,
        allow_cross_coordinate_systems: bool = False,
        base_dataset: UnifiedPhysiologyWindowDataset | None = None,
    ) -> None:
        self.base = base_dataset or UnifiedPhysiologyWindowDataset(
            cache_root=cache_root,
            dataset_ids=dataset_ids,
            window_duration_s=window_duration_s,
        )
        self.local_eeg_channels = int(local_eeg_channels)
        if self.local_eeg_channels != 6:
            raise ValueError("The current local tokenizer contract requires six EEG channels")
        self.allow_cross_coordinate_systems = bool(allow_cross_coordinate_systems)
        requested_subjects = None if subject_keys is None else {str(value) for value in subject_keys}
        requested_tasks = None if task_namespaces is None else {str(value) for value in task_namespaces}

        entries: list[LocalWindowEntry] = []
        for index, ref in enumerate(self.base.windows):
            subject_key = f"{ref.record.dataset_id}|{ref.record.canonical_subject_id}"
            label = canonical_label(ref.event, ref.record.dataset_id)
            if requested_subjects is not None and subject_key not in requested_subjects:
                continue
            if requested_tasks is not None and label["namespace"] not in requested_tasks:
                continue
            if reject_unknown_labels and (
                int(label.get("class_index", -1)) < 0
                or str(label.get("condition", "unknown")).lower() == "unknown"
            ):
                continue
            dependency = (
                _visual_dependency_group(ref.record.base_record_id, ref.event)
                if ref.record.dataset_id == "visual_cognitive_motivation"
                else f"{ref.record.join_key}|event={ref.event.get('event_index', index)}"
            )
            entries.append(
                LocalWindowEntry(
                    base_index=index,
                    dataset_id=ref.record.dataset_id,
                    subject_key=subject_key,
                    dependency_group_id=dependency,
                )
            )
        if not entries:
            raise ValueError("Unified local view produced no admitted windows")
        self.entries = entries
        self.subject_keys = {entry.subject_key for entry in entries}

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def _paired_indices(sample: Mapping[str, Any]) -> list[tuple[int, int]]:
        roles = list(sample["component_roles"]["fnirs"])
        rows = list(sample["channel_geometry"]["fnirs"])
        bad = np.asarray(sample["bad_channel_mask"]["fnirs"], dtype=bool)
        hbr_by_base = {
            str(row.get("base_channel_name")): index
            for index, (row, role) in enumerate(zip(rows, roles))
            if role == "HbR" and not bad[index]
        }
        pairs = []
        for index, (row, role) in enumerate(zip(rows, roles)):
            if role != "HbO" or bad[index]:
                continue
            pair = hbr_by_base.get(str(row.get("base_channel_name")))
            if pair is not None:
                pairs.append((index, pair))
        if not pairs:
            raise ValueError("Measured fNIRS window has no paired HbO/HbR anchors")
        return pairs

    def _select_eeg(
        self,
        sample: Mapping[str, Any],
        fnirs_index: int,
    ) -> np.ndarray:
        eeg_rows = list(sample["channel_geometry"]["eeg"])
        fnirs_rows = list(sample["channel_geometry"]["fnirs"])
        eeg_positions = _position(eeg_rows)
        anchor = _position([fnirs_rows[fnirs_index]])[0]
        eeg_units = {str(row.get("coordinate_units")) for row in eeg_rows}
        fnirs_units = str(fnirs_rows[fnirs_index].get("coordinate_units"))
        if not self.allow_cross_coordinate_systems and (
            len(eeg_units) != 1 or fnirs_units not in eeg_units
        ):
            raise ValueError(
                "Cross-modal local selection requires matching coordinate units; "
                "use an explicitly admitted adapter before enabling cross-system geometry"
            )
        finite = np.all(np.isfinite(eeg_positions), axis=1) & np.isfinite(anchor).all()
        bad = np.asarray(sample["bad_channel_mask"]["eeg"], dtype=bool)
        eligible = finite & ~bad
        distances = np.linalg.norm(eeg_positions - anchor[None, :], axis=1)
        distances[~eligible] = np.inf
        selected = np.argsort(distances)[: self.local_eeg_channels]
        if len(selected) != self.local_eeg_channels or np.any(~np.isfinite(distances[selected])):
            raise ValueError("Fewer than six finite, non-bad EEG channels surround the fNIRS anchor")
        return np.asarray(selected, dtype=np.int64)

    @staticmethod
    def _token_mask(mask: np.ndarray, patch_samples: int) -> torch.Tensor:
        value = np.asarray(mask, dtype=bool)
        if value.size % patch_samples:
            raise ValueError("Validity mask does not align to the tokenizer patch grid")
        return torch.from_numpy(value.reshape(-1, patch_samples).all(axis=1))

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        sample = self.base[entry.base_index]
        pairs = self._paired_indices(sample)
        digest = hashlib.sha256(
            f"{sample['join_key']}|{sample['event'].get('event_index', entry.base_index)}".encode("utf-8")
        ).digest()
        pair_index = int.from_bytes(digest[:8], "little") % len(pairs)
        hbo_index, hbr_index = pairs[pair_index]
        eeg_indices = self._select_eeg(sample, hbo_index)

        eeg_valid = np.asarray(sample["analysis_valid_mask"]["eeg"], dtype=bool)
        fnirs_valid = np.asarray(sample["analysis_valid_mask"]["fnirs"], dtype=bool)
        eeg = np.asarray(sample["eeg"], dtype=np.float32)[eeg_indices].copy()
        fnirs = np.asarray(sample["fnirs"], dtype=np.float32)[[hbo_index, hbr_index]].copy()
        eeg[:, ~eeg_valid] = 0.0
        fnirs[:, ~fnirs_valid] = 0.0
        label = sample["label"]
        anchor_name = str(sample["channel_geometry"]["fnirs"][hbo_index].get("base_channel_name"))
        sample_id = (
            f"{sample['dataset_id']}|{sample['subject']}|{sample['record_id']}|"
            f"event={sample['event'].get('event_index', entry.base_index)}|anchor={anchor_name}"
        )
        return {
            "schema": LOCAL_VIEW_SCHEMA,
            "eeg": torch.from_numpy(np.ascontiguousarray(eeg)),
            "fnirs": torch.from_numpy(np.ascontiguousarray(fnirs)),
            "token_valid_mask": {
                "eeg": self._token_mask(eeg_valid, 400),
                "fnirs": self._token_mask(fnirs_valid, 20),
            },
            "subject_key": entry.subject_key,
            "dataset_id": sample["dataset_id"],
            "subject": str(sample["subject"]),
            "record_id": str(sample["record_id"]),
            "task_namespace": str(label["namespace"]),
            "label": torch.tensor(int(label["class_index"]), dtype=torch.long),
            "anchor": anchor_name,
            "selected_eeg_channels": [sample["channel_names"]["eeg"][int(i)] for i in eeg_indices],
            "dependency_group_id": entry.dependency_group_id,
            "sample_id": sample_id,
        }


__all__ = [
    "LOCAL_VIEW_SCHEMA",
    "LocalWindowEntry",
    "UnifiedPhysiologyLocalViewDataset",
]
