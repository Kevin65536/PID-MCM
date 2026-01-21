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
from .registry import (
    register_tokenizer,
    get_tokenizer_class,
    create_tokenizer,
    list_tokenizers,
    StandardizedOutput,
)

# Register all tokenizers
from .registry import _TOKENIZER_REGISTRY
_TOKENIZER_REGISTRY['patch_vqvae'] = PatchVQVAETokenizer
_TOKENIZER_REGISTRY['time_patch_vqvae'] = PatchVQVAETokenizer  # alias
_TOKENIZER_REGISTRY['freq_patch_vqvae'] = FreqDomainPatchVQVAE
_TOKENIZER_REGISTRY['freq_patch_vqvae_v2'] = FreqDomainPatchVQVAE_V2
_TOKENIZER_REGISTRY['neurorvq'] = NeuroRVQTokenizer
_TOKENIZER_REGISTRY['neurorvq_v2'] = NeuroRVQTokenizer_V2

__all__ = [
    # Base
    'BaseTokenizer', 
    # Quantizers
    'FSQTokenizer', 
    'VQVAETokenizer', 
    # Tokenizers
    'PatchVQVAETokenizer',
    'FreqDomainPatchVQVAE',
    'FreqDomainPatchVQVAE_V2',
    'NeuroRVQTokenizer',
    'NeuroRVQTokenizer_V2',
    # NeuroRVQ components
    'NormEMAVectorQuantizer',
    'ResidualVectorQuantization',
    'MultiScaleTemporalEncoder',
    # Registry
    'register_tokenizer',
    'get_tokenizer_class',
    'create_tokenizer',
    'list_tokenizers',
    'StandardizedOutput',
]
