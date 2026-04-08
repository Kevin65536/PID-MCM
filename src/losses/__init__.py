"""Loss functions used across tokenizer and downstream training."""

from .alignment import AlignmentLoss
from .classification import LabelSmoothingCrossEntropy
from .multimodal_tokenizer import (
    align_pair,
    batch_usage_entropy_loss,
    compute_factorized_shared_alignment_losses,
    compute_shared_alignment_losses,
    coupling_kl_loss,
    orthogonality_loss,
    smooth_signal,
    symmetric_hard_assignment_ce,
    symmetric_kl_from_logits,
    symmetric_prob_kl,
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
    'compute_factorized_shared_alignment_losses',
    'compute_multi_stft_loss',
    'compute_shared_alignment_losses',
    'compute_smoothness_loss',
    'compute_stft_loss',
    'coupling_kl_loss',
    'orthogonality_loss',
    'smooth_signal',
    'symmetric_hard_assignment_ce',
    'symmetric_kl_from_logits',
    'symmetric_prob_kl',
]
