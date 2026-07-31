"""Audited CBraMod checkpoint loading and source-faithful probe wrappers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn
import yaml


METHOD_ROOT = Path(__file__).resolve().parents[1]
LOCAL_AVAILABILITY = {"downloaded", "source_bundled"}


@dataclass(frozen=True)
class CBraModCheckpointMetadata:
    artifact_id: str
    path: Path
    sha256: str
    size_bytes: int
    source_revision: str
    patch_samples: int
    embedding_dim: int
    representation_layer: str


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
        raise ValueError("CBraMod method manifest must be a mapping")
    return value


def _artifact(manifest: Mapping[str, Any], artifact_id: str) -> Mapping[str, Any]:
    artifacts = manifest.get("checkpoint", {}).get("artifacts", [])
    matches = [item for item in artifacts if item.get("artifact_id") == artifact_id]
    if len(matches) != 1:
        raise KeyError(f"expected one CBraMod artifact {artifact_id!r}, found {len(matches)}")
    return matches[0]


def _load_upstream_class(method_root: Path) -> type[nn.Module]:
    upstream = (method_root / "upstream").resolve()
    expected_model = upstream / "models/cbramod.py"
    if not expected_model.is_file():
        raise FileNotFoundError(f"missing pinned CBraMod upstream model: {expected_model}")

    loaded_models = sys.modules.get("models")
    if loaded_models is not None:
        loaded_path = Path(str(getattr(loaded_models, "__file__", ""))).resolve()
        if upstream not in loaded_path.parents:
            raise ImportError(
                "a different top-level 'models' package is already loaded; "
                "run CBraMod in its isolated comparison process"
            )
    sys.path.insert(0, str(upstream))
    try:
        module = importlib.import_module("models.cbramod")
    finally:
        if sys.path[0] == str(upstream):
            sys.path.pop(0)
    if Path(module.__file__).resolve() != expected_model:
        raise ImportError(f"resolved unexpected CBraMod module: {module.__file__}")
    return module.CBraMod


def load_verified_cbramod_encoder(
    artifact_id: str = "cbramod_pretrained",
    *,
    device: torch.device | str = "cpu",
    method_root: Path = METHOD_ROOT,
) -> tuple[nn.Module, CBraModCheckpointMetadata]:
    """Hash, safely deserialize, and strictly load the official CBraMod encoder."""

    manifest = _manifest(method_root)
    artifact = _artifact(manifest, artifact_id)
    availability = str(artifact.get("availability", ""))
    if availability not in LOCAL_AVAILABILITY:
        raise PermissionError(
            f"CBraMod artifact {artifact_id!r} is not locally available: {availability!r}"
        )
    path = method_root / str(artifact["local_path"])
    if not path.is_file():
        raise FileNotFoundError(f"missing CBraMod checkpoint: {path}")
    expected_size = int(artifact["size_bytes"])
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"CBraMod checkpoint size mismatch: {path.stat().st_size} != {expected_size}"
        )
    expected_hash = str(artifact["sha256"])
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(f"CBraMod checkpoint SHA-256 mismatch: {actual_hash} != {expected_hash}")

    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not state:
        raise TypeError(f"CBraMod checkpoint {path} did not contain a state mapping")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise TypeError(f"CBraMod checkpoint {path} contains non-tensor state")

    encoder_class = _load_upstream_class(method_root)
    encoder = encoder_class(
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=800,
        seq_len=30,
        n_layer=12,
        nhead=8,
    )
    encoder.load_state_dict(state, strict=True)
    # The official quick example and every released downstream wrapper remove
    # the pretraining reconstruction projection *after* strict checkpoint
    # loading. Downstream features are the encoder latent tokens, not the
    # reconstruction values produced by proj_out.
    encoder.proj_out = nn.Identity()
    encoder.requires_grad_(False)
    encoder.eval()
    encoder.to(device)
    metadata = CBraModCheckpointMetadata(
        artifact_id=artifact_id,
        path=path.resolve(),
        sha256=actual_hash,
        size_bytes=expected_size,
        source_revision=str(artifact["source_revision"]),
        patch_samples=200,
        embedding_dim=200,
        representation_layer="encoder_latent_before_pretraining_proj_out",
    )
    return encoder, metadata


class CBraModFrozenEncoder(nn.Module):
    """200 Hz EEG adapter using the official latent-token average-pooling route."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        sampling_rate_hz: float = 200.0,
        patch_samples: int = 200,
        token_pooling: str = "official_avgpooling_patch_reps",
    ) -> None:
        super().__init__()
        if token_pooling != "official_avgpooling_patch_reps":
            raise ValueError("only official_avgpooling_patch_reps is implemented")
        if not isinstance(getattr(encoder, "proj_out", None), nn.Identity):
            raise ValueError(
                "CBraMod downstream encoder must expose latent tokens with proj_out=Identity"
            )
        self.encoder = encoder
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.patch_samples = int(patch_samples)
        self.token_pooling = token_pooling
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True) -> "CBraModFrozenEncoder":
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
            raise ValueError(f"CBraMod EEG input must have shape [B,C,T], got {tuple(eeg.shape)}")
        batch, channels, samples = eeg.shape
        if batch < 1 or channels < 1:
            raise ValueError("CBraMod EEG input must contain at least one item and channel")
        if float(sampling_rate_hz) != self.sampling_rate_hz:
            raise ValueError(
                f"CBraMod requires {self.sampling_rate_hz:g} Hz EEG, got {sampling_rate_hz:g} Hz"
            )
        names = tuple(str(name) for name in channel_names)
        if len(names) != channels or len(set(names)) != channels:
            raise ValueError("CBraMod channel_names must uniquely identify every measured channel")
        if samples < self.patch_samples or samples % self.patch_samples:
            raise ValueError(
                f"CBraMod requires a positive multiple of {self.patch_samples} samples, got {samples}"
            )
        if channel_valid is not None:
            if channel_valid.shape != (batch, channels) or not bool(channel_valid.all()):
                raise ValueError("CBraMod primary adapter does not admit missing or padded channels")
        if sample_valid is not None:
            if sample_valid.shape != (batch, samples) or not bool(sample_valid.all()):
                raise ValueError("CBraMod primary adapter does not admit missing or padded samples")
        if not bool(torch.isfinite(eeg).all()):
            raise ValueError("CBraMod EEG input contains non-finite values")

        patches = eeg.reshape(batch, channels, samples // self.patch_samples, self.patch_samples)
        self.encoder.eval()
        with torch.no_grad():
            tokens = self.encoder(patches)
            expected_tokens = (batch, channels, samples // self.patch_samples, 200)
            if tokens.shape != expected_tokens or not bool(torch.isfinite(tokens).all()):
                raise RuntimeError(
                    f"CBraMod encoder returned invalid latent tokens {tuple(tokens.shape)}"
                )
            embedding = tokens.mean(dim=(1, 2))
        if embedding.shape != (batch, 200) or not bool(torch.isfinite(embedding).all()):
            raise RuntimeError(
                f"CBraMod encoder returned invalid embedding shape {tuple(embedding.shape)}"
            )
        return embedding


class CBraModLinearProbe(nn.Module):
    def __init__(self, frozen_encoder: CBraModFrozenEncoder, output_dim: int) -> None:
        super().__init__()
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        self.frozen_encoder = frozen_encoder
        self.head = nn.Linear(200, int(output_dim))

    def forward(self, eeg: torch.Tensor, **adapter_kwargs: Any) -> torch.Tensor:
        return self.head(self.frozen_encoder(eeg, **adapter_kwargs))
