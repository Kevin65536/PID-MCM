#!/usr/bin/env python
"""Export deterministic event-relative windows for coupling identifiability audits."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.analysis import patch_features_torch
from src.data import create_configured_multimodal_dataloaders
from src.tokenizers import create_tokenizer
from src.utils import load_checkpoint_file


def _to_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _strings(value: Any, count: int) -> np.ndarray:
    if isinstance(value, (list, tuple)):
        values = [str(item) for item in value]
    else:
        values = [str(value)] * count
    return np.asarray(values[:count], dtype=str)


def _denormalize_component(
    value: torch.Tensor,
    normalization: Dict[str, torch.Tensor],
    modality: str,
    component: str,
) -> torch.Tensor:
    offset = normalization[f"{modality}_{component}_offset"].to(value.device).unsqueeze(-1)
    scale = normalization[f"{modality}_{component}_scale"].to(value.device).unsqueeze(-1)
    return value * scale + offset


def _append(store: Dict[str, List[np.ndarray]], key: str, value: Any, dtype=None) -> None:
    array = _to_numpy(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    store.setdefault(key, []).append(array)


def _flush_shard(
    store: Dict[str, List[np.ndarray]],
    output_dir: Path,
    shard_dir: Path,
    shard_index: int,
) -> tuple[str, int, Dict[str, List[int]]]:
    arrays = {key: np.concatenate(chunks, axis=0) for key, chunks in store.items()}
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"part_{shard_index:05d}.npz"
    np.savez(shard_path, **arrays)
    return (
        str(shard_path.relative_to(output_dir)),
        int(arrays["subject_id"].shape[0]),
        {key: list(value.shape[1:]) for key, value in arrays.items()},
    )


def _rss_gib() -> float:
    with open("/proc/self/statm", encoding="utf-8") as handle:
        resident_pages = int(handle.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 ** 3)


@torch.inference_mode()
def export_split(
    model,
    loader,
    device: torch.device,
    output_dir: Path,
    split: str,
    max_batches: int | None,
    shard_batches: int,
) -> Dict[str, Any]:
    store: Dict[str, List[np.ndarray]] = {}
    batches = 0
    samples = 0
    shard_index = 0
    shard_paths: List[str] = []
    array_shapes: Dict[str, List[int]] = {}
    started_at = time.monotonic()
    shard_dir = output_dir / "shards" / split
    if shard_dir.exists():
        for stale_path in shard_dir.glob("part_*.npz"):
            stale_path.unlink()
    for batch in loader:
        if max_batches is not None and batches >= max_batches:
            break
        eeg = batch["eeg"].to(device, non_blocking=True)
        fnirs = batch["fnirs"].to(device, non_blocking=True)
        targets = {
            key: value.to(device, non_blocking=True)
            for key, value in batch["targets"].items()
        }
        normalization = batch["normalization"]

        latents = model.encode_modalities(eeg, fnirs)
        token_arrays = {}
        for name in ("eeg_source", "fnirs_source"):
            _, indices, _ = model.quantize(latents[name], modality=name)
            token_arrays[name] = indices

        eeg_source_physical = _denormalize_component(targets["eeg_source"], normalization, "eeg", "source")
        eeg_obs_physical = _denormalize_component(targets["eeg_observation"], normalization, "eeg", "observation")
        fnirs_source_physical = _denormalize_component(targets["fnirs_source"], normalization, "fnirs", "source")
        fnirs_obs_physical = _denormalize_component(targets["fnirs_observation"], normalization, "fnirs", "observation")

        eeg_features = patch_features_torch(
            torch.cat([
                eeg_source_physical + eeg_obs_physical,
                eeg_source_physical,
                eeg_obs_physical,
            ]),
            sample_rate_hz=200.0,
            patch_size=400,
            eeg=True,
        ).chunk(3)
        fnirs_features = patch_features_torch(
            torch.cat([
                fnirs_source_physical + fnirs_obs_physical,
                fnirs_source_physical,
                fnirs_obs_physical,
            ]),
            sample_rate_hz=10.0,
            patch_size=20,
            eeg=False,
        ).chunk(3)
        for key, features in zip(
            ("eeg_raw_features", "eeg_source_features", "eeg_observation_features"),
            eeg_features,
        ):
            _append(store, key, features, np.float32)
        for key, features in zip(
            ("fnirs_raw_features", "fnirs_source_features", "fnirs_observation_features"),
            fnirs_features,
        ):
            _append(store, key, features, np.float32)

        _append(store, "eeg_source_tokens", token_arrays["eeg_source"], np.int16)
        _append(store, "fnirs_source_tokens", token_arrays["fnirs_source"], np.int16)
        _append(store, "eeg_source_latent", latents["eeg_source"], np.float16)
        _append(store, "fnirs_source_latent", latents["fnirs_source"], np.float16)
        for key in (
            "subject_id", "event_idx", "crop_start_fnirs", "event_window_pre_s",
            "crop_start_s", "event_relative_start_s", "token_relative_position",
            "token_event_time_s",
        ):
            _append(store, key, batch[key])
        count = int(eeg.shape[0])
        for key in ("cache_entry_id", "label_name", "anchor", "source_name", "source_task"):
            _append(store, key, _strings(batch[key], count))
        batches += 1
        samples += count
        if batches % shard_batches == 0:
            relative_path, shard_samples, array_shapes = _flush_shard(
                store, output_dir, shard_dir, shard_index,
            )
            shard_paths.append(relative_path)
            shard_index += 1
            store.clear()
            elapsed = max(time.monotonic() - started_at, 1e-6)
            print(
                f"export progress: {split} batches={batches} samples={samples} "
                f"rate={samples / elapsed:.1f} samples/s rss={_rss_gib():.2f} GiB "
                f"last_shard={shard_samples}",
                flush=True,
            )

    if store:
        relative_path, _, array_shapes = _flush_shard(store, output_dir, shard_dir, shard_index)
        shard_paths.append(relative_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = {
        "schema_version": "coupling_identifiability_sharded_split_v1",
        "split": split,
        "batches": batches,
        "samples": samples,
        "shards": shard_paths,
        "array_tail_shapes": array_shapes,
    }
    manifest_path = output_dir / f"{split}.manifest.json"
    manifest_path.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    return {
        "path": str(manifest_path),
        "batches": batches,
        "samples": samples,
        "shards": len(shard_paths),
        "array_tail_shapes": array_shapes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--shard-batches", type=int, default=16)
    parser.add_argument(
        "--clear-entry-filters",
        action="store_true",
        help="Ignore data.entry_filters stored in the tokenizer checkpoint during audit export",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    device = torch.device(args.device)
    checkpoint = load_checkpoint_file(checkpoint_path, device="cpu")
    config = checkpoint["config"]
    if args.clear_entry_filters:
        config.get("data", {}).pop("entry_filters", None)
    config["training"]["batch_size"] = int(args.batch_size)
    config["data"]["num_workers"] = int(args.num_workers)
    config["data"].setdefault("crop", {})["train_random"] = False
    config["data"]["crop"]["force_deterministic_all_splits"] = True
    config["data"]["crop"]["eval_event_offsets_s"] = [float(value) for value in range(-10, 22, 2)]
    config["data"].setdefault("dataloader", {})["drop_last"] = False
    # Each split is traversed once. Keeping three worker pools alive only retains
    # their NPZ caches and pinned buffers for the remainder of the export.
    config["data"]["dataloader"]["persistent_workers"] = False

    model = create_tokenizer(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    loaders = create_configured_multimodal_dataloaders(config)

    summaries = {}
    for split, loader in loaders.items():
        summaries[split] = export_split(
            model,
            loader,
            device,
            output_dir,
            split,
            args.max_batches,
            args.shard_batches,
        )
    manifest = {
        "schema_version": "coupling_identifiability_export_v1",
        "label": args.label,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "event_offsets_s": list(range(-10, 22, 2)),
        "summaries": summaries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
