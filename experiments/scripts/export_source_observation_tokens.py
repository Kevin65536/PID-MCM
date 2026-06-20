#!/usr/bin/env python
"""Export source/observation tokenizer outputs as token-sequence datasets."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import yaml

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import create_configured_multimodal_dataloaders
from src.tokenizers import create_tokenizer
from src.utils import load_checkpoint_file, require_standard_training_launcher


DEFAULT_TOKENIZER_RUN_DIR = (
    "experiments/runs/source_observation/croce_local/highwl_v2/"
    "s2_croce_local_highwl_v2_branch_norm_coupling0_lr2e4_compile_20260610_161921"
)
DEFAULT_OUTPUT_ROOT = "experiments/runs/downstream/source_observation_tokens"


def load_yaml_optional(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = project_root / "experiments" / "configs" / path
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {resolved}")
    return payload


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def resolve_checkpoint(run_dir: Path, checkpoint: str) -> Path:
    candidate = Path(checkpoint)
    if candidate.is_absolute():
        return candidate
    run_relative = run_dir / candidate
    if run_relative.exists():
        return run_relative
    if len(candidate.parts) == 1:
        return run_dir / "checkpoints" / checkpoint
    return resolve_path(candidate)


def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def git_commit() -> Optional[str]:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def list_from_batch(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if torch.is_tensor(value):
        return [str(item) for item in value.detach().cpu().tolist()]
    return [str(value)]


def append_tensor(store: Dict[str, List[np.ndarray]], key: str, value: Any) -> None:
    store.setdefault(key, []).append(as_numpy(value))


def concatenate_store(store: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    merged: Dict[str, np.ndarray] = {}
    for key, chunks in store.items():
        if not chunks:
            continue
        first = chunks[0]
        if first.ndim == 0:
            merged[key] = np.asarray(chunks)
        else:
            merged[key] = np.concatenate(chunks, axis=0)
    return merged


def save_npz(path: Path, arrays: Dict[str, np.ndarray], *, compress: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def prepare_export_config(
    tokenizer_config: Dict[str, Any],
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    train_random_crop: bool,
    clear_entry_filters: bool = False,
) -> Dict[str, Any]:
    config = copy.deepcopy(tokenizer_config)
    config.setdefault("training", {})["batch_size"] = int(batch_size)
    data_cfg = config.setdefault("data", {})
    if clear_entry_filters:
        data_cfg.pop("entry_filters", None)
    data_cfg["seed"] = int(seed)
    data_cfg["num_workers"] = int(num_workers)
    data_cfg.setdefault("crop", {})["train_random"] = bool(train_random_crop)
    dataloader_cfg = data_cfg.setdefault("dataloader", {})
    dataloader_cfg["drop_last"] = False
    if num_workers <= 0:
        dataloader_cfg["persistent_workers"] = False
        dataloader_cfg.pop("prefetch_factor", None)
    return config


def move_targets_to_device(targets: Any, device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
    if not isinstance(targets, dict):
        return None
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else torch.as_tensor(value, device=device)
        for key, value in targets.items()
    }


@torch.no_grad()
def export_split(
    *,
    split_name: str,
    dataloader: Iterable[Dict[str, Any]],
    model: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    compress: bool,
    include_latents: bool,
    max_batches: Optional[int],
    max_samples: Optional[int],
) -> Dict[str, Any]:
    tensor_store: Dict[str, List[np.ndarray]] = {}
    string_store: Dict[str, List[str]] = {
        "anchor": [],
        "cache_entry_id": [],
        "label_name": [],
        "source_name": [],
        "source_task": [],
        "fnirs_component": [],
    }
    batches = 0
    samples = 0

    for batch in dataloader:
        if max_batches is not None and batches >= max_batches:
            break

        eeg = batch["eeg"].to(device, non_blocking=True)
        fnirs = batch["fnirs"].to(device, non_blocking=True)
        targets = move_targets_to_device(batch.get("targets"), device)
        outputs = model(eeg, fnirs, targets=targets)

        current = int(eeg.shape[0])
        take = current
        if max_samples is not None:
            remaining = max(int(max_samples) - samples, 0)
            if remaining <= 0:
                break
            take = min(take, remaining)

        token_keys = [
            "eeg_source_indices",
            "fnirs_source_indices",
            "eeg_observation_indices",
            "fnirs_observation_indices",
        ]
        for key in token_keys:
            append_tensor(tensor_store, key.replace("_indices", "_tokens"), outputs[key][:take])

        append_tensor(tensor_store, "eeg_tokens", outputs["eeg_indices"][:take])
        append_tensor(tensor_store, "fnirs_tokens", outputs["fnirs_indices"][:take])
        append_tensor(tensor_store, "label", batch["label"][:take])
        append_tensor(tensor_store, "subject_id", batch["subject_id"][:take])
        append_tensor(tensor_store, "event_idx", batch["event_idx"][:take])
        append_tensor(tensor_store, "crop_start_fnirs", batch["crop_start_fnirs"][:take])
        for key in [
            "event_window_pre_s",
            "crop_start_s",
            "event_relative_start_s",
            "token_relative_position",
            "token_event_time_s",
        ]:
            if key in batch:
                append_tensor(tensor_store, key, batch[key][:take])

        if include_latents:
            for key in [
                "eeg_source_z_q",
                "fnirs_source_z_q",
                "eeg_observation_z_q",
                "fnirs_observation_z_q",
            ]:
                append_tensor(tensor_store, key, outputs[key][:take])

        for key in string_store:
            string_store[key].extend(list_from_batch(batch.get(key))[:take])

        batches += 1
        samples += take
        if max_samples is not None and samples >= max_samples:
            break

    arrays = concatenate_store(tensor_store)
    for key, values in string_store.items():
        arrays[key] = np.asarray(values, dtype=str)

    if "eeg_source_tokens" in arrays:
        arrays["eeg_attention_mask"] = np.ones_like(arrays["eeg_source_tokens"], dtype=np.int64)
        arrays["fnirs_attention_mask"] = np.ones_like(arrays["fnirs_source_tokens"], dtype=np.int64)
        token_count = int(arrays["eeg_source_tokens"].shape[1])
        arrays["token_time_s"] = (
            np.arange(token_count, dtype=np.float32) + 0.5
        ) * 2.0
        arrays["token_duration_s"] = np.asarray([2.0], dtype=np.float32)

    token_path = output_dir / "tokens" / f"{split_name}_tokens.npz"
    save_npz(token_path, arrays, compress=compress)

    summary = {
        "split": split_name,
        "batches_exported": batches,
        "samples_exported": samples,
        "path": str(token_path.relative_to(project_root)),
        "arrays": {key: list(value.shape) for key, value in arrays.items()},
    }
    for key in [
        "eeg_source_tokens",
        "fnirs_source_tokens",
        "eeg_observation_tokens",
        "fnirs_observation_tokens",
    ]:
        if key in arrays and arrays[key].size:
            summary[f"{key}_unique"] = int(np.unique(arrays[key]).size)

    write_json(output_dir / "summaries" / f"{split_name}_summary.json", summary)
    return summary


def build_model_from_checkpoint(checkpoint: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise KeyError("Checkpoint does not include a usable 'config' mapping")
    model = create_tokenizer(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Croce source/observation tokenizer token sequences")
    parser.add_argument(
        "--config",
        default="downstream/source_observation_token_export_coupling0.yaml",
        help="Export config path relative to experiments/configs",
    )
    parser.add_argument("--tokenizer-run-dir", default=None, help="Tokenizer run directory")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint filename/path")
    parser.add_argument("--output-root", default=None, help="Root for token export runs")
    parser.add_argument("--run-name", default=None, help="Optional export run name")
    parser.add_argument("--splits", default=None, help="Comma-separated split list")
    parser.add_argument("--batch-size", type=int, default=None, help="Export batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker count")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional per-split batch cap")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional per-split sample cap")
    parser.add_argument("--device", default=None, help="Device for tokenizer inference")
    parser.add_argument("--include-latents", action="store_true", help="Also save quantized latent vectors")
    parser.add_argument("--no-compress", action="store_true", help="Use np.savez instead of np.savez_compressed")
    parser.add_argument("--train-random-crop", action="store_true", help="Keep random train crops enabled")
    parser.add_argument(
        "--clear-entry-filters",
        action="store_true",
        help="Ignore data.entry_filters stored in the tokenizer checkpoint during export",
    )
    parser.add_argument("--seed", type=int, default=None, help="Export RNG seed")
    return parser.parse_args()


def main() -> None:
    require_standard_training_launcher("source-observation-token-export")
    args = parse_args()
    export_cfg = load_yaml_optional(args.config)

    tokenizer_cfg = export_cfg.get("tokenizer", {})
    output_cfg = export_cfg.get("output", {})
    data_cfg = export_cfg.get("data", {})
    experiment_cfg = export_cfg.get("experiment", {})

    tokenizer_run_dir = resolve_path(
        args.tokenizer_run_dir
        or tokenizer_cfg.get("run_dir")
        or DEFAULT_TOKENIZER_RUN_DIR
    )
    checkpoint_path = resolve_checkpoint(
        tokenizer_run_dir,
        args.checkpoint or tokenizer_cfg.get("checkpoint", "best_model.pt"),
    )
    output_root = resolve_path(args.output_root or output_cfg.get("root", DEFAULT_OUTPUT_ROOT))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{timestamp}_{experiment_cfg.get('name', 'source_observation_token_export')}"
    run_dir = output_root / run_name

    device_name = args.device or export_cfg.get("device", "cpu")
    device = torch.device(device_name)

    checkpoint = load_checkpoint_file(checkpoint_path, device=device)
    tokenizer_config = checkpoint.get("config")
    if not isinstance(tokenizer_config, dict):
        raise KeyError(f"Checkpoint missing config: {checkpoint_path}")

    batch_size = int(args.batch_size or data_cfg.get("batch_size", tokenizer_config.get("training", {}).get("batch_size", 64)))
    num_workers = int(args.num_workers if args.num_workers is not None else data_cfg.get("num_workers", 0))
    seed = int(args.seed if args.seed is not None else data_cfg.get("seed", 42))
    train_random_crop = bool(args.train_random_crop or data_cfg.get("train_random_crop", False))
    export_data_config = prepare_export_config(
        tokenizer_config,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        train_random_crop=train_random_crop,
        clear_entry_filters=bool(args.clear_entry_filters or data_cfg.get("clear_entry_filters", False)),
    )
    if isinstance(data_cfg.get("config_overrides"), dict):
        export_data_config = deep_update(export_data_config, data_cfg["config_overrides"])

    model = build_model_from_checkpoint(checkpoint, device)
    dataloaders = create_configured_multimodal_dataloaders(export_data_config)

    split_list = args.splits or data_cfg.get("splits", "train,val,test")
    if isinstance(split_list, str):
        splits = [split.strip() for split in split_list.split(",") if split.strip()]
    else:
        splits = list(split_list)

    run_dir.mkdir(parents=True, exist_ok=False)
    write_yaml(run_dir / "config.yaml", {
        "export_config": export_cfg,
        "effective_data_config": export_data_config,
        "cli": vars(args),
    })

    started_at = datetime.now().isoformat()
    summaries = []
    for split in splits:
        if split not in dataloaders:
            raise KeyError(f"Unknown split {split!r}. Available: {list(dataloaders)}")
        summary = export_split(
            split_name=split,
            dataloader=dataloaders[split],
            model=model,
            device=device,
            output_dir=run_dir,
            compress=not bool(args.no_compress or output_cfg.get("compress") is False),
            include_latents=bool(args.include_latents or output_cfg.get("include_latents", False)),
            max_batches=args.max_batches if args.max_batches is not None else data_cfg.get("max_batches_per_split"),
            max_samples=args.max_samples if args.max_samples is not None else data_cfg.get("max_samples_per_split"),
        )
        summaries.append(summary)

    manifest = {
        "started_at": started_at,
        "completed_at": datetime.now().isoformat(),
        "git_commit": git_commit(),
        "tokenizer_run_dir": str(tokenizer_run_dir.relative_to(project_root) if tokenizer_run_dir.is_relative_to(project_root) else tokenizer_run_dir),
        "checkpoint_path": str(checkpoint_path.relative_to(project_root) if checkpoint_path.is_relative_to(project_root) else checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_best_epoch": int(checkpoint.get("best_epoch", checkpoint.get("epoch", -1))),
        "checkpoint_monitor_metric": checkpoint.get("monitor_metric"),
        "checkpoint_monitor_value": checkpoint.get("monitor_value"),
        "run_dir": str(run_dir.relative_to(project_root)),
        "device": str(device),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "train_random_crop": train_random_crop,
        "clear_entry_filters": bool(args.clear_entry_filters or data_cfg.get("clear_entry_filters", False)),
        "token_semantics": {
            "token_duration_s": 2.0,
            "tokens_per_20s_window": int(tokenizer_config.get("model", {}).get("eeg", {}).get("seq_length", 4000))
            // int(tokenizer_config.get("model", {}).get("eeg", {}).get("patch_size", 400)),
            "eeg_source_vocab_size": int(tokenizer_config.get("model", {}).get("source", {}).get("codebook_size", 32)),
            "fnirs_source_vocab_size": int(tokenizer_config.get("model", {}).get("source", {}).get("codebook_size", 32)),
            "eeg_observation_vocab_size": int(tokenizer_config.get("model", {}).get("eeg_observation", {}).get("codebook_size", 64)),
            "fnirs_observation_vocab_size": int(tokenizer_config.get("model", {}).get("fnirs_observation", {}).get("codebook_size", 32)),
        },
        "splits": summaries,
    }
    write_json(run_dir / "manifest.json", manifest)
    print(f"Export complete: {run_dir}")
    for summary in summaries:
        print(f"  {summary['split']}: {summary['samples_exported']} samples -> {summary['path']}")


if __name__ == "__main__":
    main()
