"""Public-only synchronized data boundary for BrainFusion adapter alignment v2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
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


CONFIG_SCHEMA = "brainfusion_adapter_alignment_v2"
METHOD_ID = "brainfusion_nvc_csp_stacking_reimplementation"
SUPPORTED_TASKS = (
    "motor_imagery",
    "mental_arithmetic",
    "wg",
    "nback",
    "visual",
)
UNSUPPORTED_TASKS = ("dsr", "refed_regression")


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _unique_strings(values: Any, *, field: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{field} must contain unique names")
    return result


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = resolve_repo_path(path)
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"expected {CONFIG_SCHEMA} config: {config_path}")
    if value.get("method_id") != METHOD_ID or value.get("mode") != "public_audit_only":
        raise PermissionError("BrainFusion alignment config must remain public-only")
    if value.get("protected_test_default") != "locked":
        raise PermissionError("protected test must default to locked")
    if value.get("registry", {}).get("registry_sha256") != EXPECTED_REGISTRY_SHA256:
        raise ValueError("method-neutral registry fingerprint drifted")
    data = value.get("data", {})
    if float(data.get("eeg_sample_rate_hz", -1)) != 200.0:
        raise ValueError("BrainFusion canonical EEG rate must remain 200 Hz")
    if float(data.get("fnirs_sample_rate_hz", -1)) != 10.0:
        raise ValueError("BrainFusion canonical fNIRS rate must remain 10 Hz")
    if data.get("required_modalities") != ["eeg", "fnirs_hbo", "fnirs_hbr"]:
        raise ValueError("BrainFusion requires synchronized EEG, HbO, and HbR")
    observation = value.get("observation_budget", {})
    if observation.get("alignment_profile") != "support_matched_direct":
        raise ValueError("supported BrainFusion tasks must be support matched")
    if float(observation.get("extra_pre_anchor_context_s", -1)) != 0.0:
        raise ValueError("BrainFusion may not read pre-anchor context")
    if float(observation.get("extra_post_interval_context_s", -1)) != 0.0:
        raise ValueError("BrainFusion may not read post-interval context")
    if observation.get("hrf_is_method_local_transform_not_permission_for_extra_input") is not True:
        raise ValueError("HRF input boundary must be explicit")

    inventories = value.get("channel_inventories", {})
    for dataset_id, inventory in inventories.items():
        if not isinstance(inventory, dict):
            raise ValueError(f"invalid channel inventory: {dataset_id}")
        _unique_strings(inventory.get("eeg", ()), field=f"{dataset_id}.eeg")
        _unique_strings(
            inventory.get("fnirs_locations", ()), field=f"{dataset_id}.fnirs_locations"
        )
    for task in SUPPORTED_TASKS:
        task_config = value.get("tasks", {}).get(task, {})
        if task_config.get("supported") is not True:
            raise ValueError(f"supported BrainFusion task is missing: {task}")
        if task_config.get("dataset_id") != TASK_SPECS[task].dataset_id:
            raise ValueError(f"task dataset drifted: {task}")
        if not math.isclose(
            float(task_config.get("duration_s", -1)), TASK_SPECS[task].input_duration_s
        ):
            raise ValueError(f"task duration drifted: {task}")
        if task_config["dataset_id"] not in inventories:
            raise ValueError(f"task has no frozen channel inventory: {task}")
    for task in UNSUPPORTED_TASKS:
        task_config = value.get("tasks", {}).get(task, {})
        if task_config.get("supported") is not False or not task_config.get(
            "unsupported_reason_code"
        ):
            raise ValueError(f"unsupported task lacks a reason code: {task}")
    if float(value["tasks"]["dsr"]["duration_s"]) >= float(
        observation["minimum_supported_nvc_interval_s"]
    ):
        raise ValueError("DSR unsupported disposition no longer matches its duration")
    return value, config_path


@dataclass(frozen=True)
class PublicInventory:
    task: str
    dataset: EFRMUnifiedTaskDataset
    indices: tuple[int, ...]
    sample_ids: tuple[str, ...]
    split_rows: tuple[Mapping[str, Any], ...]
    duration_s: float
    eeg_channels: tuple[str, ...]
    fnirs_locations: tuple[str, ...]

    @property
    def split_fingerprint(self) -> str:
        return stable_hash(list(self.split_rows))

    @property
    def sample_inventory_sha256(self) -> str:
        return stable_hash(sorted(self.sample_ids))

    @property
    def measured_channel_identity_sha256(self) -> str:
        return stable_hash(
            {
                "eeg": sorted(self.eeg_channels),
                "fnirs_hbo": sorted(self.fnirs_locations),
                "fnirs_hbr": sorted(self.fnirs_locations),
            }
        )


def sample_id(dataset: EFRMUnifiedTaskDataset, index: int) -> str:
    row = dataset.lightweight_metadata(int(index))
    return (
        f"{row['join_key']}|event={int(row['event_index'])}"
        f"|offset_ms={int(round(float(row['window_offset_s']) * 1000.0))}"
    )


def load_public_inventory(config: Mapping[str, Any], *, task: str) -> PublicInventory:
    if task not in SUPPORTED_TASKS:
        raise KeyError(f"task is not supported by the BrainFusion v2 adapter: {task}")
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
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError(f"canonical public inventory contains duplicate IDs: {task}")
    task_config = config["tasks"][task]
    channels = config["channel_inventories"][task_config["dataset_id"]]
    return PublicInventory(
        task=task,
        dataset=dataset,
        indices=indices,
        sample_ids=identifiers,
        split_rows=tuple(split_rows),
        duration_s=float(task_config["duration_s"]),
        eeg_channels=_unique_strings(channels["eeg"], field=f"{task}.eeg"),
        fnirs_locations=_unique_strings(
            channels["fnirs_locations"], field=f"{task}.fnirs_locations"
        ),
    )


def _component_base(name: str) -> str:
    return re.sub(r"_(?:HbO|HbR)$", "", str(name), flags=re.IGNORECASE)


def data_branch_fingerprints(config: Mapping[str, Any]) -> dict[str, str]:
    cache_root = resolve_repo_path(config["data"]["cache_root"])
    paths = {
        "unified_loader": REPO_ROOT / "src/data/unified_physiology.py",
        "measurement_adapter": REPO_ROOT / "src/data/physiology_measurement_adapter.py",
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
        raise FileNotFoundError(f"BrainFusion data-branch evidence is missing: {missing}")
    return {name: sha256_file(path) for name, path in paths.items()}


class BrainFusionPublicView(Dataset[dict[str, Any]]):
    """Deliver a fixed real synchronized inventory without padding or copying."""

    def __init__(
        self,
        inventory: PublicInventory,
        *,
        eeg_sample_rate_hz: float = 200.0,
        fnirs_sample_rate_hz: float = 10.0,
    ) -> None:
        self.inventory = inventory
        self.dataset = inventory.dataset
        self.eeg_sample_rate_hz = float(eeg_sample_rate_hz)
        self.fnirs_sample_rate_hz = float(fnirs_sample_rate_hz)
        self.eeg_samples = int(round(inventory.duration_s * self.eeg_sample_rate_hz))
        self.fnirs_samples = int(round(inventory.duration_s * self.fnirs_sample_rate_hz))

    def __len__(self) -> int:
        return len(self.dataset)

    def _release_other_records(self, current_join_key: str) -> None:
        cache = getattr(self.dataset.base, "_record_cache", None)
        if isinstance(cache, dict):
            for key in tuple(cache):
                if key != current_join_key:
                    cache.pop(key, None)

    @staticmethod
    def _full_support(source: Mapping[str, Any], *, modality: str, samples: int) -> None:
        recorded = np.asarray(source["valid_mask"][modality], dtype=bool)[:samples]
        analysis = np.asarray(source["analysis_valid_mask"][modality], dtype=bool)[:samples]
        if recorded.shape != (samples,) or not bool(recorded.all()):
            raise ValueError(f"{modality} contains unrecorded/padded support")
        if analysis.shape != (samples,) or not bool(analysis.all()):
            raise ValueError(f"{modality} contains analysis-invalid support")

    def __getitem__(self, index: int) -> dict[str, Any]:
        task_index = int(index)
        source = self.dataset.base[self.dataset.indices[task_index]]
        join_key = str(source["join_key"])
        eeg_rate = float(source["sample_rate_hz"]["eeg"])
        fnirs_rate = float(source["sample_rate_hz"]["fnirs"])
        if not math.isclose(eeg_rate, self.eeg_sample_rate_hz):
            raise ValueError(f"unexpected EEG rate for {join_key}: {eeg_rate}")
        if not math.isclose(fnirs_rate, self.fnirs_sample_rate_hz):
            raise ValueError(f"unexpected fNIRS rate for {join_key}: {fnirs_rate}")
        self._full_support(source, modality="eeg", samples=self.eeg_samples)
        self._full_support(source, modality="fnirs", samples=self.fnirs_samples)

        eeg_names = tuple(str(name) for name in source["channel_names"]["eeg"])
        if len(eeg_names) != len(set(eeg_names)):
            raise ValueError(f"duplicate EEG channel identity for {join_key}")
        eeg_lookup = {name: position for position, name in enumerate(eeg_names)}
        missing_eeg = [name for name in self.inventory.eeg_channels if name not in eeg_lookup]
        if missing_eeg:
            raise ValueError(f"frozen EEG inventory absent for {join_key}: {missing_eeg}")
        eeg_indices = np.asarray(
            [eeg_lookup[name] for name in self.inventory.eeg_channels], dtype=np.int64
        )
        eeg_bad = np.asarray(source["bad_channel_mask"]["eeg"], dtype=bool)[eeg_indices]
        if bool(eeg_bad.any()):
            rejected = [
                name
                for name, flag in zip(self.inventory.eeg_channels, eeg_bad, strict=True)
                if flag
            ]
            raise ValueError(f"frozen EEG inventory contains bad channels: {rejected}")

        fnirs_names = tuple(str(name) for name in source["channel_names"]["fnirs"])
        roles = tuple(str(role) for role in source["component_roles"]["fnirs"])
        if len(fnirs_names) != len(set(fnirs_names)) or len(fnirs_names) != len(roles):
            raise ValueError(f"invalid fNIRS channel identity for {join_key}")
        by_role: dict[str, dict[str, int]] = {"HbO": {}, "HbR": {}}
        for position, (name, role) in enumerate(zip(fnirs_names, roles, strict=True)):
            if role in by_role:
                base = _component_base(name)
                if base in by_role[role]:
                    raise ValueError(f"duplicate {role} location {base} for {join_key}")
                by_role[role][base] = position
        missing_fnirs = [
            location
            for location in self.inventory.fnirs_locations
            if location not in by_role["HbO"] or location not in by_role["HbR"]
        ]
        if missing_fnirs:
            raise ValueError(f"frozen paired fNIRS inventory absent: {missing_fnirs}")
        hbo_indices = np.asarray(
            [by_role["HbO"][name] for name in self.inventory.fnirs_locations], dtype=np.int64
        )
        hbr_indices = np.asarray(
            [by_role["HbR"][name] for name in self.inventory.fnirs_locations], dtype=np.int64
        )
        fnirs_bad = np.asarray(source["bad_channel_mask"]["fnirs"], dtype=bool)
        bad_locations = [
            name
            for name, hbo, hbr in zip(
                self.inventory.fnirs_locations, hbo_indices, hbr_indices, strict=True
            )
            if fnirs_bad[hbo] or fnirs_bad[hbr]
        ]
        if bad_locations:
            raise ValueError(f"frozen paired fNIRS inventory contains bad channels: {bad_locations}")

        eeg = np.asarray(source["eeg"], dtype=np.float32)[
            eeg_indices, : self.eeg_samples
        ]
        fnirs = np.asarray(source["fnirs"], dtype=np.float32)
        hbo = fnirs[hbo_indices, : self.fnirs_samples]
        hbr = fnirs[hbr_indices, : self.fnirs_samples]
        expected = (
            (len(self.inventory.eeg_channels), self.eeg_samples),
            (len(self.inventory.fnirs_locations), self.fnirs_samples),
        )
        if eeg.shape != expected[0] or hbo.shape != expected[1] or hbr.shape != expected[1]:
            raise ValueError(f"BrainFusion window is too short for {join_key}")
        if not bool(np.isfinite(eeg).all() and np.isfinite(hbo).all() and np.isfinite(hbr).all()):
            raise ValueError(f"BrainFusion input contains non-finite values: {join_key}")

        identifier = sample_id(self.dataset, task_index)
        self._release_other_records(join_key)
        return {
            "eeg": torch.from_numpy(np.ascontiguousarray(eeg)),
            "hbo": torch.from_numpy(np.ascontiguousarray(hbo)),
            "hbr": torch.from_numpy(np.ascontiguousarray(hbr)),
            "dataset_index": torch.tensor(task_index, dtype=torch.long),
            "sample_id": identifier,
            "join_key": join_key,
            "eeg_channel_names": self.inventory.eeg_channels,
            "fnirs_location_names": self.inventory.fnirs_locations,
            "recorded_support_count": {
                "eeg": self.eeg_samples,
                "fnirs_hbo": self.fnirs_samples,
                "fnirs_hbr": self.fnirs_samples,
            },
        }
