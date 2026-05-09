"""Loss functions used across tokenizer and downstream training."""

from .alignment import AlignmentLoss
from .classification import LabelSmoothingCrossEntropy
from .multimodal_tokenizer import (
    align_pair,
    batch_usage_entropy_loss,
    coupling_kl_loss,
    orthogonality_loss,
    symmetric_kl_from_logits,
)
from .reconstruction import (
    compute_band_power_loss,
    compute_multi_stft_loss,
    compute_smoothness_loss,
    compute_stft_loss,
)

__all__ = [
    'AlignmentLoss',
    'LabelSmoothingCrossEntropy',
    'align_pair',
    'batch_usage_entropy_loss',
    'compute_band_power_loss',
    'compute_multi_stft_loss',
    'compute_smoothness_loss',
    'compute_stft_loss',
    'coupling_kl_loss',
    'orthogonality_loss',
    'symmetric_kl_from_logits',
]
