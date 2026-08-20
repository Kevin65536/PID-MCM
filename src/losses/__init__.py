"""Loss functions used across tokenizer and downstream training."""

from .alignment import AlignmentLoss
from .classification import LabelSmoothingCrossEntropy
from .reconstruction import (
    compute_band_power_loss,
    compute_multi_stft_loss,
    compute_smoothness_loss,
    compute_stft_loss,
)
from .physiology_semantic import PhysiologySemanticLoss
from .lag_conditioned import (
    masked_mean_loss,
    masked_mse,
    native_feature_prediction_loss,
    raw_patch_reconstruction_loss,
    weighted_pretraining_loss,
)
from .ssm_observation import (
    masked_huber_loss,
    ssm_observation_objective,
    uncertainty_weighted_huber_loss,
)

__all__ = [
    'AlignmentLoss',
    'LabelSmoothingCrossEntropy',
    'compute_band_power_loss',
    'compute_multi_stft_loss',
    'compute_smoothness_loss',
    'compute_stft_loss',
    'PhysiologySemanticLoss',
    'masked_mean_loss',
    'masked_mse',
    'native_feature_prediction_loss',
    'raw_patch_reconstruction_loss',
    'weighted_pretraining_loss',
    'masked_huber_loss',
    'ssm_observation_objective',
    'uncertainty_weighted_huber_loss',
]
