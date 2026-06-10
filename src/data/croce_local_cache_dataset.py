"""Dataset adapter for Croce local source/observation target caches."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


HIGHWL_SOURCE_FIELD = "source_fnirs_optical_channel_0"
HIGHWL_OBSERVATION_FIELD = "obs_fnirs_optical_channel_0"
LOWWL_SOURCE_FIELD = "source_fnirs_optical_channel_1"
LOWWL_OBSERVATION_FIELD = "obs_fnirs_optical_channel_1"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sanitize_anchor(anchor: str) -> str:
    return str(anchor).replace(" ", "_").replace("-", "_")


def _subject_from_path(path: Path) -> Optional[int]:
    for part in path.parts:
        match = re.fullmatch(r"subject[_-]?0*(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def _infer_label_name(manifest: Mapping[str, Any], job: Mapping[str, Any]) -> str:
    for key in ("event_label_name_fnirs", "event_label_name_eeg", "bundle_segment_label", "segment_label_name"):
        value = job.get(key)
        if value:
            return str(value)

    config = manifest.get("config", {}) if isinstance(manifest.get("config"), Mapping) else {}
    for key in ("bundle_segment_label", "segment_label_name", "task"):
        value = config.get(key)
        if value:
            return str(value)

    bundle_path = str(config.get("bundle_path", ""))
    match = re.search(r"event_window_\d+_(.+?)\.npz$", bundle_path)
    if match:
        return match.group(1)
    return "unknown"


def _as_pair_labels(value: Any) -> Sequence[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item) for item in value)
    return ()


def _is_subject_dir_name(value: str) -> bool:
    return re.fullmatch(r"subject[_-]?0*\d+", str(value)) is not None


def _infer_subject_only_label(root: Path, manifest_path: Path, task: str) -> str:
    try:
        relative_parts = manifest_path.parent.relative_to(root).parts
    except ValueError:
        relative_parts = manifest_path.parent.parts

    for part in relative_parts:
        if not _is_subject_dir_name(part):
            return str(part)
    if not _is_subject_dir_name(root.name):
        return str(root.name)
    return str(task or "unknown")


def _infer_subject_only_prefix_metadata(prefix: str) -> tuple[str, Optional[int]]:
    parts = str(prefix).split("/")
    anchor = parts[0] if parts else str(prefix)
    event_idx: Optional[int] = None
    for part in parts[1:]:
        match = re.fullmatch(r"event_(\d+)", part)
        if match:
            event_idx = int(match.group(1))
            break
    return anchor, event_idx


def _as_channel_time(array: np.ndarray, *, expected_channels: int, field_name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{field_name} must be 2D [T, C], got shape {arr.shape}")
    if arr.shape[1] != expected_channels:
        raise ValueError(f"{field_name} expected {expected_channels} channels, got shape {arr.shape}")
    return arr


@dataclass(frozen=True)
class CroceCacheEntry:
    cache_path: Path
    prefix: str
    subject_id: int
    source_name: str
    task: str
    anchor: str
    event_idx: Optional[int]
    label_name: str
    pair_mode: str = "unknown"
    pair_labels: Sequence[str] = ()
    label_id: int = -1
    deterministic_fnirs_start: Optional[int] = None


class CroceLocalCacheDataset(Dataset):
    """Read highWL-only local anchor windows from generated Croce caches.

    The fNIRS tensor has one channel: a single spatial anchor's high wavelength
    optical measurement-space signal. The cached lowWL component is deliberately
    left unread for this training phase.
    """

    def __init__(
        self,
        *,
        cache_sources: Sequence[Mapping[str, Any] | str],
        subject_ids: Optional[Iterable[int]] = None,
        split: str = "train",
        crop_duration_s: float = 20.0,
        eeg_sample_rate_hz: float = 200.0,
        fnirs_sample_rate_hz: float = 10.0,
        train_random_crop: bool = True,
        eval_event_offsets_s: Sequence[float] = (-10.0, 0.0, 20.0),
        seed: int = 42,
        cache_npz_handles: bool = True,
        max_npz_cache_size: int = 128,
        normalization: Optional[Mapping[str, Any]] = None,
    ):
        if crop_duration_s <= 0:
            raise ValueError("crop_duration_s must be positive")
        self.subject_ids = {int(value) for value in subject_ids} if subject_ids is not None else None
        self.split = str(split)
        self.crop_duration_s = float(crop_duration_s)
        self.eeg_sample_rate_hz = float(eeg_sample_rate_hz)
        self.fnirs_sample_rate_hz = float(fnirs_sample_rate_hz)
        self.eeg_per_fnirs = int(round(self.eeg_sample_rate_hz / self.fnirs_sample_rate_hz))
        if self.eeg_per_fnirs < 1 or not np.isclose(self.eeg_per_fnirs * self.fnirs_sample_rate_hz, self.eeg_sample_rate_hz):
            raise ValueError("EEG sample rate must be an integer multiple of fNIRS sample rate")
        self.eeg_crop_samples = int(round(self.crop_duration_s * self.eeg_sample_rate_hz))
        self.fnirs_crop_samples = int(round(self.crop_duration_s * self.fnirs_sample_rate_hz))
        self.train_random_crop = bool(train_random_crop)
        self.eval_event_offsets_s = tuple(float(value) for value in eval_event_offsets_s)
        self.rng = np.random.default_rng(int(seed))
        self.cache_npz_handles = bool(cache_npz_handles)
        self.max_npz_cache_size = max(int(max_npz_cache_size), 1)
        self.normalization_cfg = dict(normalization or {})
        self.normalization_enabled = bool(self.normalization_cfg.get("enabled", False))
        self.normalization_eps = float(self.normalization_cfg.get("eps", 1e-6))
        self.normalization_estimator = str(self.normalization_cfg.get("estimator", "std"))
        self.normalization_center = str(self.normalization_cfg.get("center", "mean"))
        self.eeg_normalization_mode = self._resolve_modality_normalization_mode("eeg")
        self.fnirs_normalization_mode = self._resolve_modality_normalization_mode("fnirs")
        self._npz_cache: Dict[Path, Any] = {}
        self.cache_sources = [self._normalize_source(source, index) for index, source in enumerate(cache_sources)]

        discovered = self._discover_entries()
        label_to_id = {label: index for index, label in enumerate(sorted({entry.label_name for entry in discovered}))}
        self.entries: List[CroceCacheEntry] = [
            CroceCacheEntry(**{**entry.__dict__, "label_id": label_to_id[entry.label_name]})
            for entry in discovered
        ]
        self.label_to_id = label_to_id
        if not self.entries:
            raise ValueError(
                f"No Croce cache samples found for split={self.split!r} and subjects={sorted(self.subject_ids or [])}."
            )

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_npz_cache"] = {}
        return state

    def close(self) -> None:
        for handle in self._npz_cache.values():
            close = getattr(handle, "close", None)
            if callable(close):
                close()
        self._npz_cache.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _normalize_source(source: Mapping[str, Any] | str, index: int) -> Dict[str, Any]:
        if isinstance(source, Mapping):
            root = source.get("root") or source.get("cache_root") or source.get("path")
            if root is None:
                raise ValueError("Each Croce cache source must define root/cache_root/path.")
            payload = dict(source)
            payload["root"] = str(root)
            payload.setdefault("name", f"cache_source_{index}")
            return payload
        return {"root": str(source), "name": f"cache_source_{index}"}

    def _resolve_modality_normalization_mode(self, modality: str) -> str:
        if not self.normalization_enabled:
            return "none"
        modality_cfg = self.normalization_cfg.get(modality, {})
        if not isinstance(modality_cfg, Mapping):
            modality_cfg = {}
        mode = modality_cfg.get(
            "branch_scales",
            modality_cfg.get("mode", self.normalization_cfg.get("branch_scales", "separate")),
        )
        mode = str(mode)
        if mode not in {"none", "separate"}:
            raise ValueError(
                f"Croce {modality} normalization mode {mode!r} is unsupported; "
                "supported modes are 'none' and 'separate'."
            )
        return mode

    def _discover_entries(self) -> List[CroceCacheEntry]:
        entries: List[CroceCacheEntry] = []
        for source in self.cache_sources:
            root = Path(str(source["root"]))
            source_name = str(source.get("name", root.name))
            task_override = source.get("task")
            if not root.exists():
                raise FileNotFoundError(f"Croce cache root not found: {root}")
            for manifest_path in sorted(root.rglob("cache_manifest.json")):
                manifest = _load_json(manifest_path)
                config = manifest.get("config", {}) if isinstance(manifest.get("config"), Mapping) else {}
                subject_id = int(config.get("subject_id") or _subject_from_path(manifest_path) or -1)
                if subject_id < 0:
                    continue
                if self.subject_ids is not None and subject_id not in self.subject_ids:
                    continue
                cache_file = manifest.get("cache_file")
                if not cache_file:
                    continue
                cache_path = manifest_path.parent / str(cache_file)
                if not cache_path.exists():
                    continue
                task = str(task_override or config.get("task") or config.get("bundle_task") or root.name)
                pair_mode = str(
                    source.get("pair_mode")
                    or config.get("pair_mode")
                    or manifest.get("pair_mode")
                    or "wavelength"
                )
                pair_labels = _as_pair_labels(
                    source.get("pair_labels")
                    or config.get("pair_labels")
                    or manifest.get("pair_labels")
                    or ("highWL", "lowWL")
                )
                jobs = manifest.get("per_job_results", [])
                if not isinstance(jobs, Sequence) or isinstance(jobs, (str, bytes)) or not jobs:
                    entries.extend(
                        self._discover_subject_only_entries(
                            source=source,
                            source_name=source_name,
                            root=root,
                            manifest_path=manifest_path,
                            cache_path=cache_path,
                            manifest=manifest,
                            task=task,
                            subject_id=subject_id,
                            pair_mode=pair_mode,
                            pair_labels=pair_labels,
                        )
                    )
                    continue

                for job in jobs:
                    if not isinstance(job, Mapping):
                        continue
                    anchor = str(job.get("anchor", ""))
                    if not anchor:
                        continue
                    prefix = _sanitize_anchor(anchor)
                    event_idx = job.get("event_idx")
                    if event_idx is not None:
                        event_idx = int(event_idx)
                        prefix = f"{prefix}/event_{event_idx:03d}"
                    label_name = _infer_label_name(manifest, job)
                    if self.split == "train":
                        entries.append(
                            CroceCacheEntry(
                                cache_path=cache_path,
                                prefix=prefix,
                                subject_id=subject_id,
                                source_name=source_name,
                                task=task,
                                anchor=anchor,
                                event_idx=event_idx,
                                label_name=label_name,
                                pair_mode=pair_mode,
                                pair_labels=pair_labels,
                            )
                        )
                    else:
                        for start in self._deterministic_starts(job):
                            entries.append(
                                CroceCacheEntry(
                                    cache_path=cache_path,
                                    prefix=prefix,
                                    subject_id=subject_id,
                                    source_name=source_name,
                                    task=task,
                                    anchor=anchor,
                                    event_idx=event_idx,
                                    label_name=label_name,
                                    pair_mode=pair_mode,
                                    pair_labels=pair_labels,
                                    deterministic_fnirs_start=start,
                                )
                            )
        return entries

    def _discover_subject_only_entries(
        self,
        *,
        source: Mapping[str, Any],
        source_name: str,
        root: Path,
        manifest_path: Path,
        cache_path: Path,
        manifest: Mapping[str, Any],
        task: str,
        subject_id: int,
        pair_mode: str,
        pair_labels: Sequence[str],
    ) -> List[CroceCacheEntry]:
        label_name = str(source.get("label_name") or _infer_subject_only_label(root, manifest_path, task))
        config = manifest.get("config", {}) if isinstance(manifest.get("config"), Mapping) else {}
        event_window_pre_s = source.get("event_window_pre_s", config.get("event_window_pre_s"))
        discovered: List[CroceCacheEntry] = []

        with np.load(cache_path, allow_pickle=False) as npz:
            prefixes = sorted(
                key[: -len("/source_eeg")]
                for key in npz.files
                if key.endswith("/source_eeg")
            )
            for prefix in prefixes:
                fnirs_key = f"{prefix}/{HIGHWL_SOURCE_FIELD}"
                if fnirs_key not in npz:
                    continue
                anchor, event_idx = _infer_subject_only_prefix_metadata(prefix)
                job = {"num_fnirs_steps": int(npz[fnirs_key].shape[0])}
                if event_window_pre_s is not None:
                    job["event_window_pre_s"] = float(event_window_pre_s)
                if self.split == "train":
                    starts = [None]
                else:
                    starts = self._deterministic_starts(job)
                for start in starts:
                    discovered.append(
                        CroceCacheEntry(
                            cache_path=cache_path,
                            prefix=prefix,
                            subject_id=subject_id,
                            source_name=source_name,
                            task=task,
                            anchor=anchor,
                            event_idx=event_idx,
                            label_name=label_name,
                            pair_mode=pair_mode,
                            pair_labels=pair_labels,
                            deterministic_fnirs_start=start,
                        )
                    )
        return discovered

    def _deterministic_starts(self, job: Mapping[str, Any]) -> List[int]:
        total_fnirs = int(job.get("num_fnirs_steps", 500))
        max_start = max(total_fnirs - self.fnirs_crop_samples, 0)
        pre_s = job.get("event_window_pre_s")
        starts: List[int] = []
        for offset_s in self.eval_event_offsets_s:
            if pre_s is not None:
                start_s = float(pre_s) + float(offset_s)
            else:
                start_s = max(float(offset_s), 0.0)
            start = int(round(start_s * self.fnirs_sample_rate_hz))
            starts.append(min(max(start, 0), max_start))
        return sorted(set(starts)) or [0]

    def __len__(self) -> int:
        return len(self.entries)

    def _crop_start(self, total_fnirs: int, entry: CroceCacheEntry) -> int:
        max_start = max(int(total_fnirs) - self.fnirs_crop_samples, 0)
        if entry.deterministic_fnirs_start is not None:
            return min(max(int(entry.deterministic_fnirs_start), 0), max_start)
        if self.train_random_crop and max_start > 0:
            return int(self.rng.integers(0, max_start + 1))
        return 0

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.entries[index]
        if self.cache_npz_handles:
            npz = self._open_npz(entry.cache_path)
            eeg_source, eeg_obs, fnirs_source, fnirs_obs = self._read_entry_arrays(entry, npz)
        else:
            with np.load(entry.cache_path, allow_pickle=False) as npz:
                eeg_source, eeg_obs, fnirs_source, fnirs_obs = self._read_entry_arrays(entry, npz)

        total_fnirs = min(fnirs_source.shape[0], fnirs_obs.shape[0])
        fnirs_start = self._crop_start(total_fnirs, entry)
        fnirs_end = fnirs_start + self.fnirs_crop_samples
        eeg_start = fnirs_start * self.eeg_per_fnirs
        eeg_end = eeg_start + self.eeg_crop_samples

        eeg_source_crop = eeg_source[eeg_start:eeg_end]
        eeg_obs_crop = eeg_obs[eeg_start:eeg_end]
        fnirs_source_crop = fnirs_source[fnirs_start:fnirs_end]
        fnirs_obs_crop = fnirs_obs[fnirs_start:fnirs_end]
        if eeg_source_crop.shape != (self.eeg_crop_samples, 6):
            raise ValueError(f"Unexpected EEG crop shape {eeg_source_crop.shape} for {entry.cache_path}:{entry.prefix}")
        if fnirs_source_crop.shape != (self.fnirs_crop_samples, 1):
            raise ValueError(f"Unexpected fNIRS crop shape {fnirs_source_crop.shape} for {entry.cache_path}:{entry.prefix}")

        (
            eeg_source_model,
            eeg_obs_model,
            eeg_normalization_meta,
        ) = self._normalize_source_observation_pair(
            eeg_source_crop,
            eeg_obs_crop,
            modality="eeg",
            mode=self.eeg_normalization_mode,
        )
        (
            fnirs_source_model,
            fnirs_obs_model,
            fnirs_normalization_meta,
        ) = self._normalize_source_observation_pair(
            fnirs_source_crop,
            fnirs_obs_crop,
            modality="fnirs",
            mode=self.fnirs_normalization_mode,
        )

        eeg_raw = eeg_source_model + eeg_obs_model
        fnirs_raw = fnirs_source_model + fnirs_obs_model

        def tensor_tc_to_ct(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(array.T.astype(np.float32, copy=False)))

        return {
            "eeg": tensor_tc_to_ct(eeg_raw),
            "fnirs": tensor_tc_to_ct(fnirs_raw),
            "targets": {
                "eeg_source": tensor_tc_to_ct(eeg_source_model),
                "eeg_observation": tensor_tc_to_ct(eeg_obs_model),
                "fnirs_source": tensor_tc_to_ct(fnirs_source_model),
                "fnirs_observation": tensor_tc_to_ct(fnirs_obs_model),
            },
            "normalization": {
                "enabled": torch.tensor(1 if self.normalization_enabled else 0, dtype=torch.long),
                **self._tensorize_normalization_meta(eeg_normalization_meta, "eeg"),
                **self._tensorize_normalization_meta(fnirs_normalization_meta, "fnirs"),
            },
            "subject": torch.tensor(entry.subject_id, dtype=torch.long),
            "subject_id": torch.tensor(entry.subject_id, dtype=torch.long),
            "label": torch.tensor(entry.label_id, dtype=torch.long),
            "anchor": entry.anchor,
            "source_name": entry.source_name,
            "source_task": entry.task,
            "event_idx": torch.tensor(-1 if entry.event_idx is None else entry.event_idx, dtype=torch.long),
            "crop_start_fnirs": torch.tensor(fnirs_start, dtype=torch.long),
            "fnirs_component": "highWL",
        }

    def _component_normalization_stats(self, array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.normalization_center == "none":
            offset = np.zeros((1, array.shape[1]), dtype=np.float32)
        elif self.normalization_center == "mean":
            offset = np.mean(array, axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
        else:
            raise ValueError("Croce normalization center must be 'mean' or 'none'.")

        if self.normalization_estimator != "std":
            raise ValueError("Croce normalization estimator currently supports only 'std'.")
        scale = np.std(array - offset, axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
        scale = np.where(scale >= self.normalization_eps, scale, 1.0).astype(np.float32)
        return offset, scale

    def _normalize_component(self, array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        offset, scale = self._component_normalization_stats(array)
        normalized = (array.astype(np.float32, copy=False) - offset) / scale
        return normalized.astype(np.float32, copy=False), offset.reshape(-1), scale.reshape(-1)

    def _normalize_source_observation_pair(
        self,
        source: np.ndarray,
        observation: np.ndarray,
        *,
        modality: str,
        mode: str,
    ) -> tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray | str]]:
        if not self.normalization_enabled or mode == "none":
            channels = source.shape[1]
            meta: Dict[str, np.ndarray | str] = {
                "mode": "none",
                "source_offset": np.zeros(channels, dtype=np.float32),
                "source_scale": np.ones(channels, dtype=np.float32),
                "observation_offset": np.zeros(channels, dtype=np.float32),
                "observation_scale": np.ones(channels, dtype=np.float32),
            }
            return source, observation, meta

        if mode != "separate":
            raise ValueError(f"Unsupported {modality} normalization mode: {mode}")

        source_normalized, source_offset, source_scale = self._normalize_component(source)
        observation_normalized, observation_offset, observation_scale = self._normalize_component(observation)
        meta = {
            "mode": "separate",
            "source_offset": source_offset,
            "source_scale": source_scale,
            "observation_offset": observation_offset,
            "observation_scale": observation_scale,
        }
        return source_normalized, observation_normalized, meta

    @staticmethod
    def _tensorize_normalization_meta(meta: Mapping[str, Any], prefix: str) -> Dict[str, torch.Tensor]:
        return {
            f"{prefix}_source_offset": torch.from_numpy(np.asarray(meta["source_offset"], dtype=np.float32)),
            f"{prefix}_source_scale": torch.from_numpy(np.asarray(meta["source_scale"], dtype=np.float32)),
            f"{prefix}_observation_offset": torch.from_numpy(np.asarray(meta["observation_offset"], dtype=np.float32)),
            f"{prefix}_observation_scale": torch.from_numpy(np.asarray(meta["observation_scale"], dtype=np.float32)),
        }

    def _open_npz(self, path: Path):
        cached = self._npz_cache.get(path)
        if cached is not None:
            return cached

        if len(self._npz_cache) >= self.max_npz_cache_size:
            evict_path, evict_handle = next(iter(self._npz_cache.items()))
            close = getattr(evict_handle, "close", None)
            if callable(close):
                close()
            del self._npz_cache[evict_path]

        handle = np.load(path, allow_pickle=False)
        self._npz_cache[path] = handle
        return handle

    def _read_entry_arrays(self, entry: CroceCacheEntry, npz: Any):
        def read(field: str, expected_channels: int) -> np.ndarray:
            key = f"{entry.prefix}/{field}"
            if key not in npz:
                raise KeyError(f"Missing Croce cache field {key!r} in {entry.cache_path}")
            return _as_channel_time(npz[key], expected_channels=expected_channels, field_name=key)

        return (
            read("source_eeg", 6),
            read("obs_eeg", 6),
            read(HIGHWL_SOURCE_FIELD, 1),
            read(HIGHWL_OBSERVATION_FIELD, 1),
        )

    def get_num_eeg_channels(self) -> int:
        return 6

    def get_num_fnirs_channels(self) -> int:
        return 1

    def get_eeg_channel_names(self) -> List[str]:
        return [f"local_eeg_{index}" for index in range(6)]

    def get_fnirs_channel_names(self) -> List[str]:
        return ["highWL"]

    def get_eeg_sample_rate(self) -> float:
        return self.eeg_sample_rate_hz

    def get_fnirs_sample_rate(self) -> float:
        return self.fnirs_sample_rate_hz

    def get_gate0_metadata(self) -> Dict[str, Any]:
        pair_modes = sorted({entry.pair_mode for entry in self.entries})
        pair_label_sets = sorted({tuple(entry.pair_labels) for entry in self.entries})
        pair_mode = pair_modes[0] if len(pair_modes) == 1 else "mixed"
        pair_labels = list(pair_label_sets[0]) if len(pair_label_sets) == 1 else []
        return {
            "available": True,
            "cache_contract": "croce_local_anchor_highwl_v1",
            "selected_fnirs_component": "highWL",
            "ignored_fnirs_components": ["lowWL"],
            "pair_mode": pair_mode,
            "pair_labels": pair_labels,
            "cache_pair_modes": pair_modes,
            "cache_pair_label_sets": [list(values) for values in pair_label_sets],
            "fnirs_target_semantics": "optical_measurement_space",
            "fnirs_spatial_anchors": 1,
            "fnirs_optical_components": 1,
            "eeg_channels": 6,
            "fnirs_channels": 1,
            "crop_duration_s": self.crop_duration_s,
            "eeg_seq_length": self.eeg_crop_samples,
            "fnirs_seq_length": self.fnirs_crop_samples,
            "num_samples": len(self.entries),
            "label_to_id": dict(self.label_to_id),
            "normalization_enabled": self.normalization_enabled,
            "normalization_mode": str(self.normalization_cfg.get("mode", "none")),
            "eeg_normalization_mode": self.eeg_normalization_mode,
            "fnirs_normalization_mode": self.fnirs_normalization_mode,
        }

    def describe_sources(self) -> List[Dict[str, Any]]:
        counts: Dict[str, int] = {}
        tasks: Dict[str, str] = {}
        for entry in self.entries:
            counts[entry.source_name] = counts.get(entry.source_name, 0) + 1
            tasks.setdefault(entry.source_name, entry.task)
        return [
            {
                "name": name,
                "dataset": "croce_local_cache",
                "task": tasks.get(name),
                "length": count,
            }
            for name, count in sorted(counts.items())
        ]


__all__ = ["CroceLocalCacheDataset"]
