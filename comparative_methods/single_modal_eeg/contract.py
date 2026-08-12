"""Frozen public data contract for single-modal EEG feature extraction."""

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


SCHEMA = "single_modal_eeg_public_performance_v1"
SUPPORTED_METHODS = ("biot", "cbramod", "reve")


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = resolve_repo_path(path)
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA} config: {config_path}")
    if value.get("mode") != "public_development_only":
        raise PermissionError("single-modal performance config must remain public-development only")
    if value.get("protected_test_default") != "locked":
        raise PermissionError("protected test must default to locked")
    if value.get("registry", {}).get("registry_sha256") != EXPECTED_REGISTRY_SHA256:
        raise ValueError("configured method-neutral registry fingerprint drifted")
    methods = value.get("methods", {})
    if set(methods) != set(SUPPORTED_METHODS):
        raise ValueError(f"configured methods must be exactly {SUPPORTED_METHODS}")
    panel_size = int(value["data"]["panel_size"])
    if panel_size != 16:
        raise ValueError("v1 comparison contract requires a 16-channel panel")
    for task, task_config in value.get("tasks", {}).items():
        panel = tuple(str(name) for name in task_config.get("panel", ()))
        if len(panel) != panel_size or len(set(panel)) != panel_size:
            raise ValueError(f"task {task} does not define {panel_size} unique channels")
        if task not in TASK_SPECS:
            raise ValueError(f"unknown shared task in config: {task}")
    return value, config_path


@dataclass(frozen=True)
class PublicTaskContract:
    config: Mapping[str, Any]
    config_path: Path
    config_sha256: str
    task: str
    outer_fold: int
    panel: tuple[str, ...]
    duration_s: float
    public_manifest_path: Path
    public_manifest_sha256: str
    public_split_sha256: str
    metadata_sha256: str
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    dataset: EFRMUnifiedTaskDataset

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(self.dataset.spec.class_names)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "task": self.task,
            "dataset_id": self.dataset.spec.dataset_id,
            "outer_fold": self.outer_fold,
            "panel": list(self.panel),
            "duration_s": self.duration_s,
            "class_names": list(self.class_names),
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
            "public_manifest_path": str(self.public_manifest_path),
            "public_manifest_sha256": self.public_manifest_sha256,
            "public_split_sha256": self.public_split_sha256,
            "metadata_sha256": self.metadata_sha256,
            "train_sample_count": len(self.train_indices),
            "validation_sample_count": len(self.validation_indices),
            "protected_test_opened": False,
        }


def load_public_contract(
    config_path: str | Path,
    *,
    task: str,
    outer_fold: int,
) -> PublicTaskContract:
    config, resolved_config_path = load_config(config_path)
    if task not in config["tasks"]:
        raise KeyError(f"task {task!r} is absent from the frozen config")
    task_config = config["tasks"][task]
    if not bool(task_config.get("supported")):
        raise RuntimeError(
            f"task {task!r} is frozen unsupported: {task_config.get('reason', 'unspecified')}"
        )
    if TASK_SPECS[task].task_type != "classification":
        raise RuntimeError("v1 runner admits classification tasks only")
    if not 0 <= int(outer_fold) < 5:
        raise ValueError("outer_fold must be in [0, 4]")

    registry_path = resolve_repo_path(config["registry"]["manifest"])
    registry = registry_manifest(registry_path)
    entry = strict_public_entry(registry, task=task, outer_fold=int(outer_fold))
    public_path = Path(str(entry["public_path"])).resolve()
    manifest = public_json(public_path)
    public_digest = sha256_file(public_path)
    if public_digest != str(entry["public_sha256"]):
        raise RuntimeError(f"public split hash drifted: {public_path}")

    dataset = EFRMUnifiedTaskDataset(
        TASK_SPECS[task], cache_root=str(resolve_repo_path(config["data"]["cache_root"]))
    )
    train, validation = dataset.validate_shared_public_split(public_path)
    if set(train).intersection(validation):
        raise RuntimeError("public train and validation indices overlap")
    if len(train) != int(entry["train_sample_count"]):
        raise RuntimeError("public train count differs from method-neutral registry")
    if len(validation) != int(entry["validation_sample_count"]):
        raise RuntimeError("public validation count differs from method-neutral registry")
    duration_s = float(task_config["duration_s"])
    if not math.isclose(duration_s, float(dataset.spec.input_duration_s)):
        raise ValueError("task duration differs from shared task contract")
    return PublicTaskContract(
        config=config,
        config_path=resolved_config_path,
        config_sha256=sha256_file(resolved_config_path),
        task=task,
        outer_fold=int(outer_fold),
        panel=tuple(str(name) for name in task_config["panel"]),
        duration_s=duration_s,
        public_manifest_path=public_path,
        public_manifest_sha256=public_digest,
        public_split_sha256=str(manifest["split_sha256"]),
        metadata_sha256=str(manifest["metadata_sha256"]),
        train_indices=tuple(int(index) for index in train),
        validation_indices=tuple(int(index) for index in validation),
        dataset=dataset,
    )


def balanced_subset(
    contract: PublicTaskContract,
    indices: Sequence[int],
    *,
    samples_per_class: int,
) -> tuple[int, ...]:
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        condition = str(contract.dataset.lightweight_metadata(int(index))["condition"])
        grouped[condition].append(int(index))
    expected = set(contract.class_names)
    if set(grouped) != expected:
        raise RuntimeError(
            f"public subset classes differ from task contract: {sorted(grouped)} != {sorted(expected)}"
        )
    selected: list[int] = []
    for condition in contract.class_names:
        candidates = grouped[condition]
        if len(candidates) < samples_per_class:
            raise RuntimeError(
                f"class {condition} has {len(candidates)} samples, needs {samples_per_class}"
            )
        selected.extend(candidates[:samples_per_class])
    return tuple(selected)


class EEGTaskView(Dataset[dict[str, Any]]):
    """Select the frozen real-channel panel and fail closed on invalid support."""

    def __init__(self, contract: PublicTaskContract) -> None:
        self.contract = contract
        self.dataset = contract.dataset
        self.panel = contract.panel
        self.required_samples = int(
            round(
                contract.duration_s
                * float(contract.config["data"]["eeg_sample_rate_hz"])
            )
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def _release_other_records(self, current_join_key: str) -> None:
        cache = getattr(self.dataset.base, "_record_cache", None)
        if not isinstance(cache, dict):
            return
        for key in tuple(cache):
            if key != current_join_key:
                cache.pop(key, None)

    def __getitem__(self, index: int) -> dict[str, Any]:
        task_index = int(index)
        metadata = self.dataset.lightweight_metadata(task_index)
        source = self.dataset.base[self.dataset.indices[task_index]]
        join_key = str(source["join_key"])
        names = tuple(str(name) for name in source["channel_names"]["eeg"])
        lookup = {name: position for position, name in enumerate(names)}
        missing = [name for name in self.panel if name not in lookup]
        if missing:
            raise ValueError(f"frozen EEG panel is absent for {join_key}: {missing}")
        selected = np.asarray([lookup[name] for name in self.panel], dtype=np.int64)
        rate = float(source["sample_rate_hz"]["eeg"])
        if not math.isclose(rate, float(self.contract.config["data"]["eeg_sample_rate_hz"])):
            raise ValueError(f"unexpected EEG rate for {join_key}: {rate}")
        eeg = np.asarray(source["eeg"], dtype=np.float32)[selected, : self.required_samples]
        time_valid = np.asarray(source["analysis_valid_mask"]["eeg"], dtype=bool)[
            : self.required_samples
        ]
        bad = np.asarray(source["bad_channel_mask"]["eeg"], dtype=bool)[selected]
        if eeg.shape != (len(self.panel), self.required_samples):
            raise ValueError(f"EEG task window is too short for {join_key}: {eeg.shape}")
        if time_valid.shape != (self.required_samples,) or not bool(time_valid.all()):
            raise ValueError(f"EEG task window has invalid or padded time support: {join_key}")
        if bool(bad.any()):
            rejected = [name for name, flag in zip(self.panel, bad, strict=True) if flag]
            raise ValueError(f"frozen EEG panel contains bad channels for {join_key}: {rejected}")
        if not bool(np.isfinite(eeg).all()):
            raise ValueError(f"EEG task window contains non-finite values: {join_key}")
        geometry = source["channel_geometry"]["eeg"]
        geometry_lookup = {str(row["channel_name"]): row for row in geometry}
        missing_geometry = [
            name
            for name in self.panel
            if name not in geometry_lookup or not geometry_lookup[name].get("position_available")
        ]
        if missing_geometry:
            raise ValueError(f"frozen EEG panel lacks registered geometry: {missing_geometry}")
        condition = str(source["label"]["condition"])
        if condition not in self.dataset.class_to_index:
            raise ValueError(f"condition {condition!r} is not in the frozen task classes")
        self._release_other_records(join_key)
        sample_id = (
            f"{join_key}|event={int(metadata['event_index'])}"
            f"|offset_ms={int(round(float(metadata['window_offset_s']) * 1000.0))}"
        )
        return {
            "eeg": torch.from_numpy(np.ascontiguousarray(eeg)),
            "target": torch.tensor(self.dataset.class_to_index[condition], dtype=torch.long),
            "dataset_index": torch.tensor(task_index, dtype=torch.long),
            "subject": str(metadata["subject"]),
            "sample_id": sample_id,
            "join_key": join_key,
        }


class RecordGroupedBatchSampler(Sampler[list[int]]):
    """Keep each feature batch within one cached physical record."""

    def __init__(
        self,
        dataset: EFRMUnifiedTaskDataset,
        indices: Sequence[int],
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        groups: dict[str, list[int]] = defaultdict(list)
        for index in indices:
            key = str(dataset.lightweight_metadata(int(index))["join_key"])
            groups[key].append(int(index))
        if not groups:
            raise RuntimeError("feature sampler received no public indices")
        self.groups = groups
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        keys = sorted(self.groups)
        if self.shuffle:
            rng.shuffle(keys)
        for key in keys:
            indices = list(self.groups[key])
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return sum(
            math.ceil(len(indices) / self.batch_size) for indices in self.groups.values()
        )


def make_feature_loader(
    view: EEGTaskView,
    indices: Sequence[int],
    *,
    batch_size: int,
    workers: int,
    seed: int,
) -> DataLoader:
    sampler = RecordGroupedBatchSampler(
        view.dataset,
        indices,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    kwargs: dict[str, Any] = {
        "batch_sampler": sampler,
        "num_workers": int(workers),
        "pin_memory": True,
    }
    if workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(view, **kwargs)


def data_branch_fingerprints(contract: PublicTaskContract) -> dict[str, str]:
    cache_root = resolve_repo_path(contract.config["data"]["cache_root"])
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
        raise FileNotFoundError(f"comparison data branch evidence is missing: {missing}")
    return {name: sha256_file(path) for name, path in paths.items()}
