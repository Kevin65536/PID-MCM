"""Audited adapter exports for the NormWear EEG-fNIRS adaptation."""

from .normwear import (
    NormWearCheckpointMetadata,
    NormWearFrozenEncoder,
    NormWearLinearProbe,
    load_verified_normwear_encoder,
    resample_polyphase,
)

__all__ = [
    "NormWearCheckpointMetadata",
    "NormWearFrozenEncoder",
    "NormWearLinearProbe",
    "load_verified_normwear_encoder",
    "resample_polyphase",
]
