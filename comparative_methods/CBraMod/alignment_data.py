"""CBraMod-owned public data boundary for adapter-alignment v2 auditing."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Sampler


REPO_ROOT = Path(__file__).resolve().parents[2]
EFRM_ROOT = REPO_ROOT / "comparative_methods/EFRM-PyTorch"
for import_path in (REPO_ROOT, EFRM_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from comparative_methods.audit_public_preflight import (
    EXPECTED_REGISTRY_SHA256,
    public_json,
    registry_manifest,
    sha256_file,
    strict_public_entry,
)
from efrm_pytorch.tasks import EFRMUnifiedTaskDataset, TASK_SPECS


CONFIG_SCHEMA = "cbramod_adapter_alignment_v2"
SUPPORTED_TASKS = (
    "motor_imagery",
    "mental_arithmetic",
    "wg",
    "nback",
    "dsr",
    "visual",
)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = resolve_repo_path(path)
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"expected {CONFIG_SCHEMA} config: {config_path}")
    if value.get("method_id") != "cbramod" or value.get("mode") != "public_audit_only":
        raise PermissionError("CBraMod alignment config must remain a public-only CBraMod audit")
    if value.get("protected_test_default") != "locked":
        raise PermissionError("protected test must default to locked")
    if value.get("registry", {}).get("registry_sha256") != EXPECTED_REGISTRY_SHA256:
        raise ValueError("method-neutral registry fingerprint drifted")
    if int(value["data"]["panel_size"]) != 16:
        raise ValueError("support-matched CBraMod audit requires a 16-channel panel")
    if float(value["data"]["eeg_sample_rate_hz"]) != 200.0:
        raise ValueError("CBraMod alignment audit requires canonical 200 Hz EEG")
    for task in SUPPORTED_TASKS:
        task_config = value.get("tasks", {}).get(task)
        if not isinstance(task_config, dict) or task_config.get("supported") is not True:
            raise ValueError(f"supported CBraMod task is missing from config: {task}")
        panel = tuple(str(name) for name in task_config.get("panel", ()))
        if len(panel) != 16 or len(set(panel)) != 16:
            raise ValueError(f"task {task} must declare 16 unique measured channels")
        if not math.isclose(float(task_config["duration_s"]), TASK_SPECS[task].input_duration_s):
            raise ValueError(f"task duration differs from the canonical contract: {task}")
    refed = value.get("tasks", {}).get("refed_regression", {})
    if refed.get("supported") is not False or not refed.get("unsupported_reason_code"):
        raise ValueError("REFED must retain an explicit unsupported disposition")
    return value, config_path


@dataclass(frozen=True)
class PublicInventory:
    task: str
    panel: tuple[str, ...]
    duration_s: float
    dataset: EFRMUnifiedTaskDataset
    indices: tuple[int, ...]
    sample_ids: tuple[str, ...]
    split_rows: tuple[Mapping[str, Any], ...]

    @property
    def split_fingerprint(self) -> str:
        return stable_hash(list(self.split_rows))

    @property
    def sample_inventory_sha256(self) -> str:
        return stable_hash(sorted(self.sample_ids))


def sample_id(dataset: EFRMUnifiedTaskDataset, index: int) -> str:
    row = dataset.lightweight_metadata(int(index))
    return (
        f"{row['join_key']}|event={int(row['event_index'])}"
        f"|offset_ms={int(round(float(row['window_offset_s']) * 1000.0))}"
    )


def load_public_inventory(
    config: Mapping[str, Any], *, task: str
) -> PublicInventory:
    if task not in SUPPORTED_TASKS:
        raise KeyError(f"task is not supported by the CBraMod v2 audit: {task}")
    registry_path = resolve_repo_path(config["registry"]["manifest"])
    registry = registry_manifest(registry_path)
    dataset = EFRMUnifiedTaskDataset(
        TASK_SPECS[task], cache_root=str(resolve_repo_path(config["data"]["cache_root"]))
    )
    all_indices: set[int] = set()
    split_rows: list[dict[str, Any]] = []
    for outer_fold in config["registry"]["outer_folds"]:
        entry = strict_public_entry(registry, task=task, outer_fold=int(outer_fold))
        public_path = Path(str(entry["public_path"])).resolve()
        manifest = public_json(public_path)
        digest = sha256_file(public_path)
        if digest != str(entry["public_sha256"]):
            raise RuntimeError(f"public split hash drifted: {public_path}")
        train, validation = dataset.validate_shared_public_split(public_path)
        if set(train).intersection(validation):
            raise RuntimeError(f"public split overlaps for {task}/outer{outer_fold}")
        if len(train) != int(entry["train_sample_count"]):
            raise RuntimeError(f"public train count drifted for {task}/outer{outer_fold}")
        if len(validation) != int(entry["validation_sample_count"]):
            raise RuntimeError(f"public validation count drifted for {task}/outer{outer_fold}")
        all_indices.update(int(index) for index in train)
        all_indices.update(int(index) for index in validation)
        split_rows.append(
            {
                "outer_fold": int(outer_fold),
                "public_manifest_sha256": digest,
                "split_sha256": str(manifest["split_sha256"]),
                "metadata_sha256": str(manifest["metadata_sha256"]),
                "train_sample_count": len(train),
                "validation_sample_count": len(validation),
            }
        )
    indices = tuple(sorted(all_indices))
    identifiers = tuple(sample_id(dataset, index) for index in indices)
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError(f"canonical public inventory contains duplicate IDs for {task}")
    task_config = config["tasks"][task]
    return PublicInventory(
        task=task,
        panel=tuple(str(name) for name in task_config["panel"]),
        duration_s=float(task_config["duration_s"]),
        dataset=dataset,
        indices=indices,
        sample_ids=identifiers,
        split_rows=tuple(split_rows),
    )


class CBraModPublicView(Dataset[dict[str, Any]]):
    """Deliver only the frozen real EEG panel and complete recorded support."""

    def __init__(self, inventory: PublicInventory, *, sample_rate_hz: float) -> None:
        self.inventory = inventory
        self.dataset = inventory.dataset
        self.panel = inventory.panel
        self.sample_rate_hz = float(sample_rate_hz)
        self.required_samples = int(round(inventory.duration_s * self.sample_rate_hz))

    def __len__(self) -> int:
        return len(self.dataset)

    def _release_other_records(self, current_join_key: str) -> None:
        cache = getattr(self.dataset.base, "_record_cache", None)
        if isinstance(cache, dict):
            for key in tuple(cache):
                if key != current_join_key:
                    cache.pop(key, None)

    def __getitem__(self, index: int) -> dict[str, Any]:
        task_index = int(index)
        source = self.dataset.base[self.dataset.indices[task_index]]
        join_key = str(source["join_key"])
        names = tuple(str(name) for name in source["channel_names"]["eeg"])
        lookup = {name: position for position, name in enumerate(names)}
        missing = [name for name in self.panel if name not in lookup]
        if missing:
            raise ValueError(f"CBraMod panel absent for {join_key}: {missing}")
        selected = np.asarray([lookup[name] for name in self.panel], dtype=np.int64)
        rate = float(source["sample_rate_hz"]["eeg"])
        if not math.isclose(rate, self.sample_rate_hz):
            raise ValueError(f"unexpected EEG rate for {join_key}: {rate}")
        eeg = np.asarray(source["eeg"], dtype=np.float32)[selected, : self.required_samples]
        recorded = np.asarray(source["valid_mask"]["eeg"], dtype=bool)[: self.required_samples]
        bad = np.asarray(source["bad_channel_mask"]["eeg"], dtype=bool)[selected]
        if eeg.shape != (len(self.panel), self.required_samples):
            raise ValueError(f"EEG window is too short for {join_key}: {eeg.shape}")
        if recorded.shape != (self.required_samples,) or not bool(recorded.all()):
            raise ValueError(f"EEG window contains unrecorded/padded support: {join_key}")
        if bool(bad.any()):
            rejected = [name for name, flag in zip(self.panel, bad, strict=True) if flag]
            raise ValueError(f"CBraMod panel contains bad measured channels for {join_key}: {rejected}")
        if not bool(np.isfinite(eeg).all()):
            raise ValueError(f"EEG window contains non-finite values: {join_key}")
        geometry = {
            str(row["channel_name"]): row for row in source["channel_geometry"]["eeg"]
        }
        missing_geometry = [
            name
            for name in self.panel
            if name not in geometry or not bool(geometry[name].get("position_available"))
        ]
        if missing_geometry:
            raise ValueError(f"CBraMod panel lacks registered geometry: {missing_geometry}")
        identifier = sample_id(self.dataset, task_index)
        self._release_other_records(join_key)
        return {
            "eeg": torch.from_numpy(np.ascontiguousarray(eeg)),
            "dataset_index": torch.tensor(task_index, dtype=torch.long),
            "sample_id": identifier,
            "join_key": join_key,
            "recorded_support_count": torch.tensor(int(recorded.sum()), dtype=torch.long),
        }


class RecordGroupedBatchSampler(Sampler[list[int]]):
    """Batch public samples within physical records to bound loader memory."""

    def __init__(
        self,
        dataset: EFRMUnifiedTaskDataset,
        indices: Sequence[int],
        *,
        batch_size: int,
        seed: int,
    ) -> None:
        groups: dict[str, list[int]] = defaultdict(list)
        for index in indices:
            key = str(dataset.lightweight_metadata(int(index))["join_key"])
            groups[key].append(int(index))
        if batch_size <= 0 or not groups:
            raise ValueError("record-grouped sampler requires data and a positive batch size")
        self.groups = groups
        self.batch_size = int(batch_size)
        self.seed = int(seed)

    def __iter__(self) -> Iterator[list[int]]:
        keys = sorted(self.groups)
        random.Random(self.seed).shuffle(keys)
        for key in keys:
            indices = sorted(self.groups[key])
            for start in range(0, len(indices), self.batch_size):
                yield indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return sum(math.ceil(len(values) / self.batch_size) for values in self.groups.values())


def make_loader(
    view: CBraModPublicView,
    *,
    batch_size: int,
    workers: int,
) -> DataLoader:
    sampler = RecordGroupedBatchSampler(
        view.dataset, view.inventory.indices, batch_size=batch_size, seed=42
    )
    kwargs: dict[str, Any] = {
        "batch_sampler": sampler,
        "num_workers": int(workers),
        "pin_memory": True,
    }
    if workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(view, **kwargs)


def data_branch_fingerprints(config: Mapping[str, Any]) -> dict[str, str]:
    cache_root = resolve_repo_path(config["data"]["cache_root"])
    paths = {
        "unified_loader": REPO_ROOT / "src/data/unified_physiology.py",
        "task_adapter": EFRM_ROOT / "efrm_pytorch/tasks.py",
        "cache_manifest": cache_root / "cache_manifest.json",
        "event_manifest": cache_root / "event_index/event_manifest.json",
        "geometry_manifest": cache_root / "channel_geometry/geometry_manifest.json",
        "single_trial_eeg_branch": cache_root / "eeg_artifact_clean_v4/cache_manifest.json",
        "simultaneous_eeg_branch": cache_root
        / "simultaneous_eeg_eog_clean_v1/cache_manifest.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"CBraMod data branch evidence is missing: {missing}")
    return {name: sha256_file(path) for name, path in paths.items()}
