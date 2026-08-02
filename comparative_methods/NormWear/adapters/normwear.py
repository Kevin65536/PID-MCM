"""Verified frozen NormWear encoder and explicit EEG/HbO/HbR adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy import signal
import torch
from torch import nn
import yaml


METHOD_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA256 = "36d0bca18356ccfc8e8916058bf838f26f1212a646f5780b487ad78581a92561"
CHECKPOINT_SIZE_BYTES = 544_579_503
UPSTREAM_REVISION = "07517fcb13def8c89cb586128359cec02f86ec8d"
MODEL_SAMPLE_RATE_HZ = 65.0


@dataclass(frozen=True)
class NormWearCheckpointMetadata:
    artifact_id: str
    path: Path
    sha256: str
    size_bytes: int
    upstream_revision: str
    checkpoint_entry_count: int
    encoder_entry_count: int
    excluded_decoder_entry_count: int
    encoder_parameter_count: int
    embedding_dim_per_channel: int
    pretrained_position_grid: tuple[int, int]
    patch_kernel: tuple[int, int]
    representation_layer: str
    pooling: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(method_root: Path) -> dict[str, Any]:
    value = yaml.safe_load(
        (method_root / "sources/method_manifest.yaml").read_text(encoding="utf-8")
    )
    if not isinstance(value, dict):
        raise ValueError("NormWear method manifest must be a mapping")
    return value


def _upstream_module(method_root: Path) -> Any:
    upstream = (method_root / "upstream").resolve()
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    module = importlib.import_module("modules.normwear")
    expected = upstream / "modules/normwear.py"
    if Path(module.__file__).resolve() != expected:
        raise ImportError(f"unexpected modules.normwear import: {module.__file__}")
    return module


def load_verified_normwear_encoder(
    *,
    device: torch.device | str = "cpu",
    method_root: Path = METHOD_ROOT,
) -> tuple[nn.Module, Any, NormWearCheckpointMetadata]:
    """Hash-check and load only the encoder tensors from the official checkpoint."""

    manifest = _manifest(method_root)
    if manifest.get("method_id") != "normwear_eeg_fnirs_adapted":
        raise ValueError("NormWear manifest method identity drifted")
    upstream = manifest.get("upstream", {})
    if upstream.get("revision") != UPSTREAM_REVISION:
        raise ValueError("NormWear upstream revision drifted")
    artifacts = manifest.get("checkpoint", {}).get("artifacts", [])
    matches = [item for item in artifacts if item.get("artifact_id") == "normwear_pretrain"]
    if len(matches) != 1:
        raise KeyError("expected exactly one NormWear backbone artifact")
    artifact: Mapping[str, Any] = matches[0]
    checkpoint = method_root / str(artifact["local_path"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing NormWear checkpoint: {checkpoint}")
    size = checkpoint.stat().st_size
    if size != int(artifact["size_bytes"]) or size != CHECKPOINT_SIZE_BYTES:
        raise ValueError("NormWear checkpoint size drifted")
    digest = _sha256(checkpoint)
    if digest != str(artifact["sha256"]) or digest != CHECKPOINT_SHA256:
        raise ValueError("NormWear checkpoint SHA-256 drifted")

    upstream_module = _upstream_module(method_root)
    backbone = upstream_module.NormWear(
        img_size=(387, 65),
        patch_size=(9, 5),
        mask_scheme="random",
        mask_prob=0.8,
        use_cwt=True,
        nvar=4,
        comb_freq=False,
        is_pretrain=False,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if type(state).__name__ != "OrderedDict" or len(state) != 261:
        raise ValueError("unexpected NormWear checkpoint container")
    expected = backbone.state_dict()
    missing = sorted(set(expected) - set(state))
    extras = sorted(set(state) - set(expected))
    allowed_extra_prefixes = (
        "decoder_",
        "mask_token",
        "spatial_recon.",
        "temporal_recon.",
    )
    unexpected_extras = [
        key for key in extras if not key.startswith(allowed_extra_prefixes)
    ]
    if missing or unexpected_extras or len(extras) != 39:
        raise ValueError(
            "NormWear encoder/decoder checkpoint boundary drifted: "
            f"missing={missing}, unexpected={unexpected_extras}, extras={len(extras)}"
        )
    encoder_state = {key: state[key] for key in expected}
    result = backbone.load_state_dict(encoder_state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("NormWear encoder checkpoint failed strict loading")
    del encoder_state, state
    backbone.requires_grad_(False)
    backbone.eval().to(device)
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise RuntimeError("NormWear encoder was not frozen")
    metadata = NormWearCheckpointMetadata(
        artifact_id="normwear_pretrain",
        path=checkpoint.resolve(),
        sha256=digest,
        size_bytes=size,
        upstream_revision=UPSTREAM_REVISION,
        checkpoint_entry_count=261,
        encoder_entry_count=len(expected),
        excluded_decoder_entry_count=len(extras),
        encoder_parameter_count=sum(parameter.numel() for parameter in backbone.parameters()),
        embedding_dim_per_channel=768,
        pretrained_position_grid=(43, 13),
        patch_kernel=(9, 5),
        representation_layer="final_encoder_layer_norm_tokens",
        pooling="mean_all_tokens_per_channel_then_concatenate_real_channels",
    )
    return backbone, upstream_module, metadata


def resample_polyphase(
    values: torch.Tensor,
    *,
    source_rate_hz: float,
    target_rate_hz: float = MODEL_SAMPLE_RATE_HZ,
) -> torch.Tensor:
    """Deterministically anti-alias and resample a `[B,C,T]` tensor on CPU."""

    if values.ndim != 3:
        raise ValueError(f"resampling input must have shape [B,C,T], got {tuple(values.shape)}")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("resampling input contains non-finite values")
    source = float(source_rate_hz)
    target = float(target_rate_hz)
    if source == 200.0 and target == 65.0:
        up, down = 13, 40
    elif source == 10.0 and target == 65.0:
        up, down = 13, 2
    else:
        raise ValueError(f"unsupported frozen NormWear rate conversion: {source:g}->{target:g}")
    original_device = values.device
    array = np.ascontiguousarray(values.detach().cpu().numpy(), dtype=np.float32)
    transformed = signal.resample_poly(
        array,
        up,
        down,
        axis=-1,
        window=("kaiser", 5.0),
        padtype="constant",
    )
    expected = math.ceil(values.shape[-1] * up / down)
    if transformed.shape != (*values.shape[:-1], expected):
        raise RuntimeError(
            f"unexpected NormWear resampling shape: {transformed.shape} != "
            f"{(*values.shape[:-1], expected)}"
        )
    if not np.isfinite(transformed).all():
        raise ValueError("NormWear resampling produced non-finite values")
    return torch.from_numpy(np.ascontiguousarray(transformed, dtype=np.float32)).to(
        original_device
    )


def _require_all_true(
    mask: torch.Tensor | None, *, expected_shape: tuple[int, ...], field: str
) -> None:
    if mask is None:
        return
    if tuple(mask.shape) != expected_shape:
        raise ValueError(f"{field} shape must be {expected_shape}, got {tuple(mask.shape)}")
    if mask.dtype != torch.bool or not bool(mask.all()):
        raise ValueError(f"NormWear refuses missing or padded {field}")


class NormWearFrozenEncoder(nn.Module):
    """Frozen variable-channel execution with bounded-memory channel chunking."""

    def __init__(
        self,
        backbone: nn.Module,
        upstream_module: Any,
        *,
        channel_chunk_size: int = 16,
    ) -> None:
        super().__init__()
        if channel_chunk_size < 1:
            raise ValueError("channel_chunk_size must be positive")
        if not hasattr(upstream_module, "cwt_wrap"):
            raise TypeError("pinned NormWear optimized CWT is unavailable")
        if len(getattr(backbone, "encoder_blocks", ())) != 12:
            raise ValueError("NormWear adapter requires the 12-layer official encoder")
        if tuple(backbone.patch_embed.patch_size) != (9, 5):
            raise ValueError("NormWear patch projection drifted")
        self.backbone = backbone
        self.upstream_module = upstream_module
        self.channel_chunk_size = int(channel_chunk_size)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "NormWearFrozenEncoder":
        super().train(mode)
        self.backbone.eval()
        return self

    @staticmethod
    def _validate_names(
        eeg_channels: int,
        fnirs_locations: int,
        eeg_channel_names: Sequence[str],
        fnirs_location_names: Sequence[str],
    ) -> tuple[str, ...]:
        eeg_names = tuple(str(name) for name in eeg_channel_names)
        fnirs_names = tuple(str(name) for name in fnirs_location_names)
        if len(eeg_names) != eeg_channels or len(eeg_names) != len(set(eeg_names)):
            raise ValueError("NormWear EEG channel identities are missing or duplicated")
        if len(fnirs_names) != fnirs_locations or len(fnirs_names) != len(set(fnirs_names)):
            raise ValueError("NormWear fNIRS location identities are missing or duplicated")
        return (
            *(f"eeg:{name}" for name in eeg_names),
            *(f"fnirs_hbo:{name}" for name in fnirs_names),
            *(f"fnirs_hbr:{name}" for name in fnirs_names),
        )

    def prepare_model_input(
        self,
        eeg: torch.Tensor,
        hbo: torch.Tensor,
        hbr: torch.Tensor,
        *,
        eeg_sampling_rate_hz: float,
        fnirs_sampling_rate_hz: float,
        eeg_channel_names: Sequence[str],
        fnirs_location_names: Sequence[str],
        channel_valid: torch.Tensor | None = None,
        eeg_sample_valid: torch.Tensor | None = None,
        fnirs_sample_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[str, ...]]:
        for name, value in (("eeg", eeg), ("hbo", hbo), ("hbr", hbr)):
            if value.ndim != 3:
                raise ValueError(
                    f"NormWear {name} input must have shape [B,C,T], got {tuple(value.shape)}"
                )
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"NormWear {name} input must be finite floating point")
        if hbo.shape != hbr.shape:
            raise ValueError("NormWear requires paired HbO/HbR shapes")
        if eeg.shape[0] != hbo.shape[0] or eeg.shape[0] < 1:
            raise ValueError("NormWear modality batch dimensions must match and be nonempty")
        delivered_names = self._validate_names(
            eeg.shape[1], hbo.shape[1], eeg_channel_names, fnirs_location_names
        )
        _require_all_true(
            channel_valid,
            expected_shape=(eeg.shape[0], len(delivered_names)),
            field="channel support",
        )
        _require_all_true(
            eeg_sample_valid,
            expected_shape=(eeg.shape[0], eeg.shape[-1]),
            field="EEG sample support",
        )
        _require_all_true(
            fnirs_sample_valid,
            expected_shape=(hbo.shape[0], hbo.shape[-1]),
            field="fNIRS sample support",
        )
        eeg_65 = resample_polyphase(
            eeg, source_rate_hz=eeg_sampling_rate_hz, target_rate_hz=MODEL_SAMPLE_RATE_HZ
        )
        hbo_65 = resample_polyphase(
            hbo, source_rate_hz=fnirs_sampling_rate_hz, target_rate_hz=MODEL_SAMPLE_RATE_HZ
        )
        hbr_65 = resample_polyphase(
            hbr, source_rate_hz=fnirs_sampling_rate_hz, target_rate_hz=MODEL_SAMPLE_RATE_HZ
        )
        if eeg_65.shape[-1] != hbo_65.shape[-1] or hbo_65.shape != hbr_65.shape:
            raise ValueError("NormWear synchronized modalities did not resample to equal support")
        combined = torch.cat((eeg_65, hbo_65, hbr_65), dim=1)
        if combined.shape[1] != len(delivered_names):
            raise RuntimeError("NormWear delivered channel identity count drifted")
        return combined, delivered_names

    def calculate_cwt(self, model_input: torch.Tensor) -> torch.Tensor:
        if model_input.ndim != 3 or model_input.shape[-1] < 11:
            raise ValueError("NormWear model input is too short for the frozen CWT")
        batch, channels, samples = model_input.shape
        transformed = self.upstream_module.cwt_wrap(
            model_input.reshape(batch * channels, samples), 0.1, 64
        )
        if transformed.ndim != 4 or transformed.shape[1] != 3 or transformed.shape[-1] != 65:
            raise RuntimeError(f"unexpected NormWear CWT shape: {tuple(transformed.shape)}")
        return transformed.reshape(
            batch, channels, 3, transformed.shape[-2], transformed.shape[-1]
        )

    def encode_cwt(self, cwt: torch.Tensor) -> torch.Tensor:
        if cwt.ndim != 5 or cwt.shape[2] != 3 or cwt.shape[-1] != 65:
            raise ValueError(f"NormWear CWT input shape is invalid: {tuple(cwt.shape)}")
        batch, channels, planes, times, scales = cwt.shape
        backbone = self.backbone
        flattened = cwt.reshape(batch * channels, planes, times, scales)
        tokens = backbone.patch_embed(flattened)
        position = backbone.pos_adjust((times, scales), device=cwt.device)
        tokens = tokens + position[:, 1:, :]
        cls = (backbone.cls_token + position[:, :1, :]).expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)

        for block in backbone.encoder_blocks:
            independent = torch.cat(
                [
                    block.variate_encoder(tokens[start : start + self.channel_chunk_size])
                    for start in range(0, tokens.shape[0], self.channel_chunk_size)
                ],
                dim=0,
            )
            if block.curr_layer % block.fuse_frequency == 0 and not block.no_fusion:
                token_count, embedding_dim = independent.shape[1:]
                structured = independent.reshape(
                    batch, channels, token_count, embedding_dim
                )
                patch_tokens = structured[:, :, 1:, :]
                cls_tokens = (
                    structured.mean(dim=2)
                    if block.mean_fuse
                    else structured[:, :, 0, :]
                )
                fused = block.cls_fusion(cls_tokens).unsqueeze(2)
                tokens = torch.cat((fused, patch_tokens), dim=2).reshape(
                    batch * channels, token_count, embedding_dim
                )
            else:
                tokens = independent
        tokens = backbone.norm(tokens)
        return tokens.reshape(batch, channels, tokens.shape[1], tokens.shape[2])

    def forward(
        self,
        eeg: torch.Tensor,
        hbo: torch.Tensor,
        hbr: torch.Tensor,
        *,
        eeg_sampling_rate_hz: float,
        fnirs_sampling_rate_hz: float,
        eeg_channel_names: Sequence[str],
        fnirs_location_names: Sequence[str],
        channel_valid: torch.Tensor | None = None,
        eeg_sample_valid: torch.Tensor | None = None,
        fnirs_sample_valid: torch.Tensor | None = None,
        return_tokens: bool = False,
    ) -> torch.Tensor:
        with torch.no_grad():
            model_input, _ = self.prepare_model_input(
                eeg,
                hbo,
                hbr,
                eeg_sampling_rate_hz=eeg_sampling_rate_hz,
                fnirs_sampling_rate_hz=fnirs_sampling_rate_hz,
                eeg_channel_names=eeg_channel_names,
                fnirs_location_names=fnirs_location_names,
                channel_valid=channel_valid,
                eeg_sample_valid=eeg_sample_valid,
                fnirs_sample_valid=fnirs_sample_valid,
            )
            cwt = self.calculate_cwt(model_input)
            tokens = self.encode_cwt(cwt)
            if return_tokens:
                return tokens
            features = tokens.mean(dim=2).flatten(start_dim=1)
            if not bool(torch.isfinite(features).all()):
                raise RuntimeError("NormWear encoder produced non-finite features")
            return features


class NormWearLinearProbe(nn.Module):
    """Trainable linear head over an otherwise frozen NormWear representation."""

    def __init__(
        self,
        frozen_encoder: NormWearFrozenEncoder,
        *,
        channel_count: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        if channel_count < 1 or output_dim < 1:
            raise ValueError("NormWear probe dimensions must be positive")
        self.frozen_encoder = frozen_encoder
        self.head = nn.Linear(channel_count * 768, output_dim)

    def train(self, mode: bool = True) -> "NormWearLinearProbe":
        super().train(mode)
        self.frozen_encoder.eval()
        return self

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.head(self.frozen_encoder(*args, **kwargs))
