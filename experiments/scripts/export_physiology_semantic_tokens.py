#!/usr/bin/env python3
"""Export versioned physiology-semantic representations from a frozen checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.factory import create_configured_multimodal_dataloaders
from src.teachers.physical_state_teacher import PhysicalStateTeacher
from src.tokenizers.registry import create_tokenizer
import src.tokenizers  # noqa: F401


EXPORT_SCHEMA = "physiology_semantic_export_v2"


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def build_export_batch(
    outputs: Mapping[str, Any],
    teacher: Any | None,
    batch: Mapping[str, Any],
    top_k: int | None = None,
) -> Dict[str, np.ndarray]:
    payload: Dict[str, np.ndarray] = {}
    for modality in ("eeg", "fnirs"):
        output = outputs[modality]
        payload[f"{modality}_hard_ids"] = _numpy(output.quantizer.hard_ids)
        payload[f"{modality}_semantic_latent"] = _numpy(output.semantic_latent)
        payload[f"{modality}_codebook_embedding"] = _numpy(output.quantizer.quantized)
        if top_k is None:
            payload[f"{modality}_posterior"] = _numpy(output.quantizer.posterior)
        else:
            probabilities, indices = output.quantizer.posterior.topk(top_k, dim=-1)
            payload[f"{modality}_posterior_topk_indices"] = _numpy(indices)
            payload[f"{modality}_posterior_topk_probabilities"] = _numpy(probabilities)
        payload[f"{modality}_expected_embedding"] = _numpy(output.quantizer.expected_embedding)
        payload[f"{modality}_residual"] = _numpy(output.residual)
        token_masks = batch.get("token_valid_mask", {})
        if modality in token_masks:
            payload[f"{modality}_token_valid_mask"] = _numpy(token_masks[modality].bool())
    if teacher is not None:
        payload["teacher_full_summary"] = _numpy(teacher.full_summary)
        payload["teacher_full_uncertainty"] = _numpy(teacher.full_uncertainty)
        payload["teacher_valid_mask"] = _numpy(teacher.valid_mask)
        payload["teacher_context_valid_mask"] = _numpy(teacher.context_valid_mask)
        for modality in ("eeg", "fnirs"):
            payload[f"{modality}_target"] = _numpy(getattr(teacher, f"{modality}_target"))
            payload[f"{modality}_target_uncertainty"] = _numpy(
                getattr(teacher, f"{modality}_uncertainty")
            )
            for entry, mask in teacher.entry_masks[modality].items():
                payload[f"{modality}_{entry}_target_valid_mask"] = _numpy(mask)
    for key in ("subject_id", "label", "crop_start_s", "has_auxiliary_target"):
        if key in batch:
            payload[key] = _numpy(batch[key])
    string_keys = (
        "sample_id", "target_sample_key", "subject_key", "dataset_id", "subject",
        "record_id", "task_namespace", "cache_entry_id", "source_name", "source_task",
        "anchor", "label_name",
    )
    for key in string_keys:
        if key not in batch:
            continue
        value = batch[key]
        if isinstance(value, str):
            value = [value]
        payload[key] = np.asarray(value, dtype=np.str_)
    return payload


def _concatenate(chunks: Iterable[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    chunks = list(chunks)
    if not chunks:
        raise ValueError("No batches were exported")
    keys = set(chunks[0])
    if any(set(chunk) != keys for chunk in chunks[1:]):
        raise ValueError("Export batch schemas differ")
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in sorted(keys)}


def run(args: argparse.Namespace) -> Path:
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if config is None:
        if not args.config:
            raise ValueError("Checkpoint has no embedded config; --config is required")
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model = create_tokenizer(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    target_cfg = config.get("data", {}).get("auxiliary_target", {}) or {}
    teacher_adapter = PhysicalStateTeacher(
        target_family=str(target_cfg.get("family")),
        target_version=str(target_cfg.get("version")),
    )
    dataloader = create_configured_multimodal_dataloaders(config)[args.split]

    chunks = []
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            if args.max_batches is not None and index >= args.max_batches:
                break
            outputs = model(
                batch["eeg"], batch["fnirs"], token_valid_masks=batch.get("token_valid_mask")
            )
            teacher = teacher_adapter(batch["teacher"]) if "teacher" in batch else None
            chunks.append(build_export_batch(outputs, teacher, batch, top_k=args.top_k))
    payload = _concatenate(chunks)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    sample_key_name = "sample_id" if "sample_id" in payload else "cache_entry_id"
    sample_hash = hashlib.sha256(
        "\n".join(payload[sample_key_name].tolist()).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema": EXPORT_SCHEMA,
        "split": args.split,
        "sample_count": int(payload["eeg_hard_ids"].shape[0]),
        "sample_order_sha256": sample_hash,
        "sample_key_array": sample_key_name,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint": str(checkpoint_path),
        "top_k": args.top_k,
        "arrays": {key: list(value.shape) for key, value in payload.items()},
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "samples": manifest["sample_count"]}, sort_keys=True))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
