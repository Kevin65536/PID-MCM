"""Losses module for PID-MCM."""

from .pid_losses import (
    AlignmentLoss,
    OrthogonalityLoss,
    SynergyLoss,
    ReconstructionLoss,
    PIDTotalLoss,
)
from .contrastive_losses import (
    InfoNCELoss,
    TokenAlignmentLoss,
    IndexMatchingLoss,
    SharedCodebookLoss,
)

__all__ = [
    # PID losses
    'AlignmentLoss',
    'OrthogonalityLoss',
    'SynergyLoss',
    'ReconstructionLoss',
    'PIDTotalLoss',
    # Contrastive / shared-codebook losses
    'InfoNCELoss',
    'TokenAlignmentLoss',
    'IndexMatchingLoss',
    'SharedCodebookLoss',
]
