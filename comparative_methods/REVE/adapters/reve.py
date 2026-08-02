"""Audited REVE loading and coordinate-aware frozen-probe wrappers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModel
import yaml


METHOD_ROOT = Path(__file__).resolve().parents[1]
LOCAL_AVAILABILITY = {"downloaded", "source_bundled"}


@dataclass(frozen=True)
class REVECheckpointMetadata:
    artifact_id: str
    path: Path
    sha256: str
    size_bytes: int
    source_revision: str
    position_artifact_id: str
    position_path: Path
    position_sha256: str
    position_size_bytes: int
    position_source_revision: str
    upstream_code_revision: str
    patch_samples: int
    patch_overlap: int
    embedding_dim: int
    position_bank_size: int
    representation_layer: str
    pooling: str


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
        raise ValueError("REVE method manifest must be a mapping")
    return value


def _artifact(manifest: Mapping[str, Any], artifact_id: str) -> Mapping[str, Any]:
    artifacts = manifest.get("checkpoint", {}).get("artifacts", [])
    matches = [item for item in artifacts if item.get("artifact_id") == artifact_id]
    if len(matches) != 1:
        raise KeyError(f"expected one REVE artifact {artifact_id!r}, found {len(matches)}")
    return matches[0]


def _verify_artifact(
    method_root: Path,
    artifact: Mapping[str, Any],
) -> tuple[Path, str, int]:
    availability = str(artifact.get("availability", ""))
    if availability not in LOCAL_AVAILABILITY:
        raise PermissionError(
            f"REVE artifact {artifact.get('artifact_id')!r} is not locally available: "
            f"{availability!r}"
        )
    path = method_root / str(artifact["local_path"])
    if not path.is_file():
        raise FileNotFoundError(f"missing REVE artifact: {path}")
    expected_size = int(artifact["size_bytes"])
    if path.stat().st_size != expected_size:
        raise ValueError(f"REVE artifact size mismatch: {path.stat().st_size} != {expected_size}")
    expected_hash = str(artifact["sha256"])
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(f"REVE artifact SHA-256 mismatch: {actual_hash} != {expected_hash}")
    for trusted_file in artifact.get("trusted_code_files", []):
        trusted_path = method_root / str(trusted_file["local_path"])
        if not trusted_path.is_file():
            raise FileNotFoundError(f"missing trusted REVE snapshot file: {trusted_path}")
        trusted_size = int(trusted_file["size_bytes"])
        if trusted_path.stat().st_size != trusted_size:
            raise ValueError(
                f"REVE trusted file size mismatch for {trusted_path}: "
                f"{trusted_path.stat().st_size} != {trusted_size}"
            )
        trusted_hash = str(trusted_file["sha256"])
        actual_trusted_hash = _sha256(trusted_path)
        if actual_trusted_hash != trusted_hash:
            raise ValueError(
                f"REVE trusted file SHA-256 mismatch for {trusted_path}: "
                f"{actual_trusted_hash} != {trusted_hash}"
            )
    return path, actual_hash, expected_size


def load_verified_reve_base(
    *,
    device: torch.device | str = "cpu",
    method_root: Path = METHOD_ROOT,
) -> tuple[nn.Module, nn.Module, REVECheckpointMetadata]:
    """Verify and locally load REVE-base plus the official position bank."""

    manifest = _manifest(method_root)
    encoder_artifact = _artifact(manifest, "reve_base")
    position_artifact = _artifact(manifest, "reve_positions")
    encoder_path, encoder_hash, encoder_size = _verify_artifact(method_root, encoder_artifact)
    position_path, position_hash, position_size = _verify_artifact(
        method_root, position_artifact
    )

    # trust_remote_code is intentionally restricted to the already downloaded,
    # pinned local snapshot whose model tensor was verified above.
    encoder = AutoModel.from_pretrained(
        encoder_path.parent,
        trust_remote_code=True,
        local_files_only=True,
    )
    position_bank = AutoModel.from_pretrained(
        position_path.parent,
        trust_remote_code=True,
        local_files_only=True,
    )
    if not hasattr(encoder, "attention_pooling"):
        raise TypeError("local REVE snapshot has no attention_pooling method")
    if not hasattr(position_bank, "mapping") or not hasattr(position_bank, "embedding"):
        raise TypeError("local REVE position snapshot has no auditable mapping")
    if int(encoder.config.embed_dim) != 512 or int(encoder.config.patch_size) != 200:
        raise ValueError("unexpected REVE-base architecture config")
    if int(encoder.config.patch_overlap) != 20:
        raise ValueError("unexpected REVE-base patch overlap")
    if not isinstance(getattr(encoder, "final_layer", None), nn.Identity):
        raise ValueError("REVE-base output must be the final transformer latent tokens")
    query = getattr(encoder, "cls_query_token", None)
    if not isinstance(query, nn.Parameter) or query.shape != (1, 1, 512):
        raise ValueError("REVE-base pretrained attention-pooling query is unavailable")
    if not bool(torch.isfinite(query).all()):
        raise ValueError("REVE-base pretrained attention-pooling query is non-finite")

    mapping = position_bank.mapping
    positions = position_bank.embedding
    if not isinstance(mapping, dict) or len(mapping) != 543:
        raise ValueError("unexpected REVE official position-bank mapping")
    if set(mapping.values()) != set(range(543)):
        raise ValueError("REVE official position-bank indices are not one-to-one")
    if positions.shape != (543, 3) or not bool(torch.isfinite(positions).all()):
        raise ValueError("unexpected REVE official position-bank tensor")

    upstream = manifest.get("upstream", {})
    upstream_revision = str(upstream.get("revision", ""))
    if len(upstream_revision) != 40:
        raise ValueError("REVE upstream source revision is not pinned to a Git commit")

    encoder.requires_grad_(False)
    position_bank.requires_grad_(False)
    encoder.eval().to(device)
    position_bank.eval().to(device)
    metadata = REVECheckpointMetadata(
        artifact_id="reve_base",
        path=encoder_path.resolve(),
        sha256=encoder_hash,
        size_bytes=encoder_size,
        source_revision=str(encoder_artifact["source_revision"]),
        position_artifact_id="reve_positions",
        position_path=position_path.resolve(),
        position_sha256=position_hash,
        position_size_bytes=position_size,
        position_source_revision=str(position_artifact["source_revision"]),
        upstream_code_revision=upstream_revision,
        patch_samples=int(encoder.config.patch_size),
        patch_overlap=int(encoder.config.patch_overlap),
        embedding_dim=int(encoder.config.embed_dim),
        position_bank_size=len(mapping),
        representation_layer="final_transformer_latent_tokens_after_identity_final_layer",
        pooling="frozen_pretrained_cls_query_attention_pooling",
    )
    return encoder, position_bank, metadata


class REVEFrozenEncoder(nn.Module):
    """Coordinate-aware adapter that refuses unknown or fabricated positions."""

    def __init__(
        self,
        encoder: nn.Module,
        position_bank: nn.Module,
        *,
        sampling_rate_hz: float = 200.0,
        pooling: str = "frozen_pretrained_cls_query_attention_pooling",
    ) -> None:
        super().__init__()
        if pooling != "frozen_pretrained_cls_query_attention_pooling":
            raise ValueError("only frozen_pretrained_cls_query_attention_pooling is implemented")
        if not isinstance(getattr(encoder, "final_layer", None), nn.Identity):
            raise ValueError("REVE adapter requires final transformer latent tokens")
        query = getattr(encoder, "cls_query_token", None)
        expected_dim = int(encoder.config.embed_dim)
        if not isinstance(query, nn.Parameter) or query.shape != (1, 1, expected_dim):
            raise ValueError("REVE adapter requires the pretrained attention-pooling query")
        self.encoder = encoder
        self.position_bank = position_bank
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.pooling = pooling
        self.encoder.requires_grad_(False)
        self.position_bank.requires_grad_(False)
        self.encoder.eval()
        self.position_bank.eval()

    def train(self, mode: bool = True) -> "REVEFrozenEncoder":
        super().train(mode)
        self.encoder.eval()
        self.position_bank.eval()
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
            raise ValueError(f"REVE EEG input must have shape [B,C,T], got {tuple(eeg.shape)}")
        batch, channels, samples = eeg.shape
        if batch < 1 or channels < 1:
            raise ValueError("REVE EEG input must contain at least one item and channel")
        if float(sampling_rate_hz) != self.sampling_rate_hz:
            raise ValueError(
                f"REVE requires {self.sampling_rate_hz:g} Hz EEG, got {sampling_rate_hz:g} Hz"
            )
        names = tuple(str(name) for name in channel_names)
        if len(names) != channels or len(set(names)) != channels:
            raise ValueError("REVE channel_names must uniquely identify every measured channel")
        unknown = [name for name in names if name not in self.position_bank.mapping]
        if unknown:
            raise ValueError(f"REVE position bank has no coordinates for channels: {unknown}")
        if samples < int(self.encoder.config.patch_size):
            raise ValueError(
                f"REVE requires at least {self.encoder.config.patch_size} samples, got {samples}"
            )
        if channel_valid is not None:
            if channel_valid.shape != (batch, channels) or not bool(channel_valid.all()):
                raise ValueError("REVE primary adapter does not admit missing or padded channels")
        if sample_valid is not None:
            if sample_valid.shape != (batch, samples) or not bool(sample_valid.all()):
                raise ValueError("REVE primary adapter does not admit missing or padded samples")
        if not bool(torch.isfinite(eeg).all()):
            raise ValueError("REVE EEG input contains non-finite values")

        indices = torch.tensor(
            [self.position_bank.mapping[name] for name in names],
            dtype=torch.long,
            device=self.position_bank.embedding.device,
        )
        positions = self.position_bank.embedding[indices]
        positions = positions.unsqueeze(0).expand(batch, -1, -1).to(eeg)
        if positions.shape != (batch, channels, 3) or not bool(torch.isfinite(positions).all()):
            raise RuntimeError("REVE position bank returned invalid coordinates")
        self.encoder.eval()
        with torch.no_grad():
            tokens = self.encoder(eeg=eeg, pos=positions)
            patch_samples = int(self.encoder.config.patch_size)
            patch_step = patch_samples - int(self.encoder.config.patch_overlap)
            patch_count = 1 + (samples - patch_samples) // patch_step
            expected_tokens = (
                batch,
                channels,
                patch_count,
                int(self.encoder.config.embed_dim),
            )
            if tokens.shape != expected_tokens or not bool(torch.isfinite(tokens).all()):
                raise RuntimeError(
                    f"REVE encoder returned invalid latent tokens {tuple(tokens.shape)}"
                )
            embedding = self.encoder.attention_pooling(tokens)
        expected_dim = int(self.encoder.config.embed_dim)
        if embedding.shape != (batch, expected_dim) or not bool(torch.isfinite(embedding).all()):
            raise RuntimeError(f"REVE encoder returned invalid embedding shape {tuple(embedding.shape)}")
        return embedding


class REVELinearProbe(nn.Module):
    def __init__(self, frozen_encoder: REVEFrozenEncoder, output_dim: int) -> None:
        super().__init__()
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        self.frozen_encoder = frozen_encoder
        self.head = nn.Linear(int(frozen_encoder.encoder.config.embed_dim), int(output_dim))

    def forward(self, eeg: torch.Tensor, **adapter_kwargs: Any) -> torch.Tensor:
        return self.head(self.frozen_encoder(eeg, **adapter_kwargs))
