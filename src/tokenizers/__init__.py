"""
Tokenizers module for PID-MCM.
Contains VQ-VAE, FSQ, and other quantization methods.
"""

from .base import BaseTokenizer
from .fsq import FSQTokenizer
from .vqvae import VQVAETokenizer
from .patch_vqvae import PatchVQVAETokenizer
from .freq_patch_vqvae import FreqDomainPatchVQVAE, FreqDomainPatchVQVAE_V2
from .neurorvq import (
    NeuroRVQTokenizer,
    NeuroRVQTokenizer_V2,
    NormEMAVectorQuantizer,
    ResidualVectorQuantization,
    MultiScaleTemporalEncoder,
)

__all__ = [
    'BaseTokenizer', 
    'FSQTokenizer', 
    'VQVAETokenizer', 
    'PatchVQVAETokenizer',
    'FreqDomainPatchVQVAE',
    'FreqDomainPatchVQVAE_V2',
    'NeuroRVQTokenizer',
    'NeuroRVQTokenizer_V2',
    'NormEMAVectorQuantizer',
    'ResidualVectorQuantization',
    'MultiScaleTemporalEncoder',
]
