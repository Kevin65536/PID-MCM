"""Loss functions used across tokenizer and downstream training."""

from .alignment import AlignmentLoss
from .classification import LabelSmoothingCrossEntropy
from .multimodal_tokenizer import (
    batch_usage_entropy_loss,
    coupling_eeg_neighbor_smoothness_loss,
    coupling_joint_probabilities,
    coupling_lag_focus_loss,
    orthogonality_loss,
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
    'batch_usage_entropy_loss',
    'coupling_eeg_neighbor_smoothness_loss',
    'coupling_joint_probabilities',
    'coupling_lag_focus_loss',
    'compute_band_power_loss',
    'compute_multi_stft_loss',
    'compute_smoothness_loss',
    'compute_stft_loss',
    'orthogonality_loss',
]
