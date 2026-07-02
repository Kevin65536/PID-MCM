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


EXPORT_SCHEMA = "physiology_semantic_export_v1"


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def build_export_batch(
    outputs: Mapping[str, Any],
    teacher: Any,
    batch: Mapping[str, Any],
    top_k: int | None = None,
) -> Dict[str, np.ndarray]:
    payload: Dict[str, np.ndarray] = {}
    for modality in ("eeg", "fnirs"):
        output = outputs[modality]
        payload[f"{modality}_hard_ids"] = _numpy(output.quantizer.hard_ids)
        if top_k is None:
            payload[f"{modality}_posterior"] = _numpy(output.quantizer.posterior)
        else:
            probabilities, indices = output.quantizer.posterior.topk(top_k, dim=-1)
            payload[f"{modality}_posterior_topk_indices"] = _numpy(indices)
            payload[f"{modality}_posterior_topk_probabilities"] = _numpy(probabilities)
        payload[f"{modality}_expected_embedding"] = _numpy(output.quantizer.expected_embedding)
        payload[f"{modality}_residual"] = _numpy(output.residual)
    payload["teacher_full_summary"] = _numpy(teacher.full_summary)
    payload["teacher_full_uncertainty"] = _numpy(teacher.full_uncertainty)
    payload["teacher_valid_mask"] = _numpy(teacher.valid_mask)
    for key in ("subject_id", "label", "crop_start_s"):
        payload[key] = _numpy(batch[key])
    for key in ("cache_entry_id", "source_name", "source_task", "anchor", "label_name"):
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
    teacher_adapter = PhysicalStateTeacher()
    dataloader = create_configured_multimodal_dataloaders(config)[args.split]

    chunks = []
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            if args.max_batches is not None and index >= args.max_batches:
                break
            outputs = model(batch["eeg"], batch["fnirs"])
            teacher = teacher_adapter(batch["teacher"])
            chunks.append(build_export_batch(outputs, teacher, batch, top_k=args.top_k))
    payload = _concatenate(chunks)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    sample_hash = hashlib.sha256("\n".join(payload["cache_entry_id"].tolist()).encode("utf-8")).hexdigest()
    manifest = {
        "schema": EXPORT_SCHEMA,
        "split": args.split,
        "sample_count": int(payload["eeg_hard_ids"].shape[0]),
        "sample_order_sha256": sample_hash,
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
