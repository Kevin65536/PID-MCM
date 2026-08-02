#!/usr/bin/env python3
"""Run retained public-only A5/A6 NormWear adapter smoke checks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from comparative_methods.NormWear.adapters.normwear import (
    NormWearFrozenEncoder,
    NormWearLinearProbe,
    load_verified_normwear_encoder,
)
from comparative_methods.NormWear.alignment_data import (
    NormWearPublicView,
    load_config,
    load_public_inventory,
)


METHOD_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = METHOD_ROOT / "configs/alignment_v2.yaml"
DEFAULT_OUTPUT = METHOD_ROOT / "evidence/adapter_smoke_v2/summary.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    if "protected" in {part.lower() for part in resolved.parts}:
        raise PermissionError(f"refusing protected NormWear smoke path: {resolved}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _public_item(config: Mapping[str, Any], task: str) -> tuple[Any, Any]:
    inventory = load_public_inventory(config, task=task)
    item = NormWearPublicView(inventory)[inventory.indices[0]]
    return inventory, item


def _forward_public(
    adapter: NormWearFrozenEncoder,
    item: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, float, float]:
    args = [item[name].unsqueeze(0).to(device) for name in ("eeg", "hbo", "hbr")]
    kwargs = {
        "eeg_sampling_rate_hz": 200.0,
        "fnirs_sampling_rate_hz": 10.0,
        "eeg_channel_names": item["eeg_channel_names"],
        "fnirs_location_names": item["fnirs_location_names"],
    }
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        first = adapter(*args, **kwargs)
        second = adapter(*args, **kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_mib = torch.cuda.max_memory_allocated(device) / 2**20
    else:
        peak_mib = 0.0
    elapsed = time.perf_counter() - started
    if not torch.equal(first, second):
        raise RuntimeError("NormWear public adapter replay is not bitwise deterministic")
    if not bool(torch.isfinite(first).all()) or float(first.std()) <= 1e-8:
        raise RuntimeError("NormWear public adapter feature is invalid")
    return first, elapsed / 2.0, peak_mib


def run(
    *,
    device: torch.device,
    config_path: Path = DEFAULT_CONFIG,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config, _ = load_config(config_path)
    execution = config["adapter"]["execution"]
    chunk_size = int(execution["channel_attention_chunk_size"])
    thresholds = execution["chunked_numerical_acceptance"]
    backbone, upstream_module, metadata = load_verified_normwear_encoder(device=device)
    chunked = NormWearFrozenEncoder(
        backbone, upstream_module, channel_chunk_size=chunk_size
    )

    generator = torch.Generator(device=device).manual_seed(1907)
    eeg = torch.randn(1, 1, 400, generator=generator, device=device)
    hbo = torch.randn(1, 1, 20, generator=generator, device=device)
    hbr = torch.randn(1, 1, 20, generator=generator, device=device)
    kwargs = {
        "eeg_sampling_rate_hz": 200.0,
        "fnirs_sampling_rate_hz": 10.0,
        "eeg_channel_names": ("E1",),
        "fnirs_location_names": ("L1",),
    }
    with torch.inference_mode():
        model_input, delivered_names = chunked.prepare_model_input(eeg, hbo, hbr, **kwargs)
        cwt = chunked.calculate_cwt(model_input)
        upstream = backbone.get_signal_embedding(cwt, hidden_out=False, device=device)
        unchunked = NormWearFrozenEncoder(
            backbone, upstream_module, channel_chunk_size=len(delivered_names)
        ).encode_cwt(cwt)
        forced_chunks = NormWearFrozenEncoder(
            backbone, upstream_module, channel_chunk_size=1
        ).encode_cwt(cwt)
    if not torch.equal(unchunked, upstream):
        raise RuntimeError("NormWear unchunked adapter is not bitwise identical to upstream")
    difference = (forced_chunks - upstream).abs()
    upstream_features = upstream.mean(dim=2).flatten(start_dim=1)
    chunked_features = forced_chunks.mean(dim=2).flatten(start_dim=1)
    feature_difference = (chunked_features - upstream_features).abs()
    feature_cosine = torch.nn.functional.cosine_similarity(
        chunked_features, upstream_features
    )
    metrics = {
        "maximum_token_absolute_difference": float(difference.max()),
        "mean_token_absolute_difference": float(difference.mean()),
        "maximum_pooled_feature_absolute_difference": float(feature_difference.max()),
        "pooled_feature_cosine_similarity": float(feature_cosine.min()),
    }
    if metrics["maximum_token_absolute_difference"] > float(
        thresholds["maximum_token_absolute_difference"]
    ):
        raise RuntimeError("NormWear chunked token maximum error exceeds the frozen bound")
    if metrics["mean_token_absolute_difference"] > float(
        thresholds["maximum_mean_token_absolute_difference"]
    ):
        raise RuntimeError("NormWear chunked token mean error exceeds the frozen bound")
    if metrics["maximum_pooled_feature_absolute_difference"] > float(
        thresholds["maximum_pooled_feature_absolute_difference"]
    ):
        raise RuntimeError("NormWear chunked feature error exceeds the frozen bound")
    if metrics["pooled_feature_cosine_similarity"] < float(
        thresholds["minimum_pooled_feature_cosine_similarity"]
    ):
        raise RuntimeError("NormWear chunked feature cosine falls below the frozen bound")

    public_reports: dict[str, Any] = {}
    retained_features: dict[str, torch.Tensor] = {}
    for task in ("dsr", "motor_imagery"):
        inventory, item = _public_item(config, task)
        features, seconds, peak_mib = _forward_public(chunked, item, device=device)
        retained_features[task] = features
        duration = float(config["tasks"][task]["duration_s"])
        model_samples = int(round(duration * 65.0))
        cwt_samples = model_samples - 1 if model_samples % 2 == 0 else model_samples - 2
        time_patches = (cwt_samples - 9) // 9 + 1
        public_reports[task] = {
            "sample_id": str(item["sample_id"]),
            "real_channel_count": len(inventory.delivered_channel_names),
            "model_input_samples": model_samples,
            "expected_cwt_time_samples": cwt_samples,
            "expected_token_count_per_channel": time_patches * 13 + 1,
            "feature_shape": list(features.shape),
            "feature_standard_deviation": float(features.std()),
            "bitwise_replay_exact": True,
            "mean_forward_seconds": seconds,
            "peak_allocated_mib": peak_mib,
        }

    dsr_feature = retained_features["dsr"].clone()
    probe = NormWearLinearProbe(
        chunked, channel_count=dsr_feature.shape[1] // 768, output_dim=2
    ).to(device)
    before = probe.head.weight.detach().clone()
    optimizer = torch.optim.AdamW(probe.head.parameters(), lr=1e-3)
    logits = probe.head(dsr_feature)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1], device=device))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if torch.equal(before, probe.head.weight.detach()):
        raise RuntimeError("NormWear smoke linear head did not update")
    if any(parameter.grad is not None for parameter in backbone.parameters()):
        raise RuntimeError("NormWear frozen encoder received gradients")

    report = {
        "schema": "normwear_adapter_smoke_v2",
        "status": "pass",
        "completed_at": utc_now(),
        "method_id": "normwear_eeg_fnirs_adapted",
        "device": str(device),
        "checkpoint": {
            "sha256": metadata.sha256,
            "encoder_entry_count": metadata.encoder_entry_count,
            "excluded_decoder_entry_count": metadata.excluded_decoder_entry_count,
            "encoder_parameter_count": metadata.encoder_parameter_count,
        },
        "execution": {
            "dtype": str(next(backbone.parameters()).dtype),
            "channel_attention_chunk_size": chunk_size,
            "upstream_unchunked_bitwise_exact": True,
            "chunked_numerical_metrics": metrics,
            "chunked_numerical_thresholds": thresholds,
        },
        "public_smoke": public_reports,
        "linear_probe": {
            "trainable_parameters": ["head.weight", "head.bias"],
            "head_updated": True,
            "encoder_gradient_count": 0,
        },
        "protected_test_opened": False,
    }
    write_json(output_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = run(
        device=torch.device(args.device),
        config_path=args.config,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
