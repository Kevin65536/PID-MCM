"""Audited BIOT checkpoint loading and frozen-encoder probe wrappers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import nn
import yaml


METHOD_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = METHOD_ROOT / "sources/method_manifest.yaml"
UPSTREAM_MODEL_PATH = METHOD_ROOT / "upstream/model/biot.py"
LOCAL_AVAILABILITY = {"downloaded", "source_bundled"}


@dataclass(frozen=True)
class BIOTCheckpointMetadata:
    """Verified identity needed to reproduce a loaded BIOT encoder."""

    artifact_id: str
    path: Path
    sha256: str
    size_bytes: int
    source_revision: str
    n_channels: int
    n_fft: int
    hop_length: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(method_root: Path) -> dict[str, Any]:
    with (method_root / "sources/method_manifest.yaml").open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("BIOT method manifest must be a mapping")
    return value


def _artifact(manifest: Mapping[str, Any], artifact_id: str) -> Mapping[str, Any]:
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("BIOT manifest checkpoint entry must be a mapping")
    artifacts = checkpoint.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("BIOT manifest checkpoint.artifacts must be a list")
    matches = [item for item in artifacts if item.get("artifact_id") == artifact_id]
    if len(matches) != 1:
        raise KeyError(f"expected one BIOT artifact {artifact_id!r}, found {len(matches)}")
    return matches[0]


def _load_upstream_module(model_path: Path) -> ModuleType:
    if not model_path.is_file():
        raise FileNotFoundError(f"missing pinned BIOT upstream model: {model_path}")
    spec = importlib.util.spec_from_file_location("_pinned_biot_upstream_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import pinned BIOT upstream model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_verified_biot_encoder(
    artifact_id: str,
    *,
    device: torch.device | str = "cpu",
    method_root: Path = METHOD_ROOT,
) -> tuple[nn.Module, BIOTCheckpointMetadata]:
    """Hash, safely deserialize, and strictly load one official BIOT encoder."""

    manifest = _manifest(method_root)
    artifact = _artifact(manifest, artifact_id)
    availability = str(artifact.get("availability", ""))
    if availability not in LOCAL_AVAILABILITY:
        raise PermissionError(
            f"BIOT artifact {artifact_id!r} is not declared locally available: {availability!r}"
        )

    path = method_root / str(artifact["local_path"])
    if not path.is_file():
        raise FileNotFoundError(f"missing BIOT checkpoint: {path}")
    expected_size = int(artifact["size_bytes"])
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"BIOT checkpoint size mismatch for {path}: "
            f"{path.stat().st_size} != {expected_size}"
        )
    expected_hash = str(artifact["sha256"])
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"BIOT checkpoint SHA-256 mismatch for {path}: {actual_hash} != {expected_hash}"
        )

    # These upstream files are PyTorch pickle containers. weights_only=True is
    # deliberate: no checkpoint-defined object is allowed to execute here.
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not state:
        raise TypeError(f"BIOT checkpoint {path} did not contain a state mapping")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise TypeError(f"BIOT checkpoint {path} contains non-tensor state")

    channel_tokens = state.get("channel_tokens.weight")
    patch_projection = state.get("patch_embedding.projection.weight")
    if not torch.is_tensor(channel_tokens) or channel_tokens.ndim != 2:
        raise ValueError("BIOT checkpoint has no valid channel token table")
    if not torch.is_tensor(patch_projection) or patch_projection.ndim != 2:
        raise ValueError("BIOT checkpoint has no valid frequency projection")
    n_channels = int(channel_tokens.shape[0])
    n_fft = 2 * (int(patch_projection.shape[1]) - 1)
    hop_length = 100
    if n_channels not in {16, 18} or n_fft != 200:
        raise ValueError(
            f"unexpected BIOT checkpoint shape: n_channels={n_channels}, n_fft={n_fft}"
        )

    module = _load_upstream_module(method_root / "upstream/model/biot.py")
    encoder = module.BIOTEncoder(
        emb_size=256,
        heads=8,
        depth=4,
        n_channels=n_channels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    encoder.load_state_dict(state, strict=True)
    encoder.requires_grad_(False)
    encoder.eval()
    encoder.to(device)

    metadata = BIOTCheckpointMetadata(
        artifact_id=artifact_id,
        path=path.resolve(),
        sha256=actual_hash,
        size_bytes=expected_size,
        source_revision=str(artifact["source_revision"]),
        n_channels=n_channels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return encoder, metadata


class BIOTFrozenEncoder(nn.Module):
    """Conservative adapter that refuses fabricated or padded EEG support."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        sampling_rate_hz: float = 200.0,
        max_sequence_tokens: int = 1024,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.max_sequence_tokens = int(max_sequence_tokens)
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    @property
    def channel_capacity(self) -> int:
        return int(self.encoder.channel_tokens.num_embeddings)

    def train(self, mode: bool = True) -> "BIOTFrozenEncoder":
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(
        self,
        eeg: torch.Tensor,
        *,
        sampling_rate_hz: float,
        channel_names: Sequence[str],
        channel_valid: torch.Tensor | None = None,
        sample_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if eeg.ndim != 3:
            raise ValueError(f"BIOT EEG input must have shape [B,C,T], got {tuple(eeg.shape)}")
        batch, channels, samples = eeg.shape
        if batch < 1 or channels < 1:
            raise ValueError("BIOT EEG input must contain at least one batch item and channel")
        if float(sampling_rate_hz) != self.sampling_rate_hz:
            raise ValueError(
                f"BIOT requires {self.sampling_rate_hz:g} Hz EEG, got {sampling_rate_hz:g} Hz"
            )
        names = tuple(str(name) for name in channel_names)
        if len(names) != channels or len(set(names)) != channels:
            raise ValueError("BIOT channel_names must uniquely identify every measured channel")
        if channels > self.channel_capacity:
            raise ValueError(
                f"BIOT checkpoint capacity is {self.channel_capacity} channels, got {channels}"
            )
        if channel_valid is not None:
            if channel_valid.shape != (batch, channels) or not bool(channel_valid.all()):
                raise ValueError("BIOT primary adapter does not admit missing or padded channels")
        if sample_valid is not None:
            if sample_valid.shape != (batch, samples) or not bool(sample_valid.all()):
                raise ValueError("BIOT primary adapter does not admit missing or padded samples")
        if not bool(torch.isfinite(eeg).all()):
            raise ValueError("BIOT EEG input contains non-finite values")
        if samples < int(self.encoder.n_fft):
            raise ValueError(f"BIOT EEG input requires at least {self.encoder.n_fft} samples")
        frames = 1 + (samples - int(self.encoder.n_fft)) // int(self.encoder.hop_length)
        sequence_tokens = channels * frames
        if sequence_tokens > self.max_sequence_tokens:
            raise ValueError(
                f"BIOT sequence has {sequence_tokens} tokens, above {self.max_sequence_tokens}"
            )

        self.encoder.eval()
        with torch.no_grad():
            embedding = self.encoder(eeg)
        if embedding.shape != (batch, 256) or not bool(torch.isfinite(embedding).all()):
            raise RuntimeError(f"BIOT encoder returned invalid embedding shape {tuple(embedding.shape)}")
        return embedding


class BIOTLinearProbe(nn.Module):
    """Frozen BIOT representation followed by a trainable linear head."""

    def __init__(self, frozen_encoder: BIOTFrozenEncoder, output_dim: int) -> None:
        super().__init__()
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        self.frozen_encoder = frozen_encoder
        self.head = nn.Linear(256, int(output_dim))

    def forward(self, eeg: torch.Tensor, **adapter_kwargs: Any) -> torch.Tensor:
        return self.head(self.frozen_encoder(eeg, **adapter_kwargs))
