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
    NeuroRVQTokenizerFNIRS,
    NormEMAVectorQuantizer,
    ResidualVectorQuantization,
    MultiScaleTemporalEncoder,
    MultiScaleTemporalEncoderFNIRS,
)
from .labram_vqnsp import (
    LaBraMVQNSP,
    LaBraMVQNSP_EEG,
    LaBraMVQNSP_fNIRS,
)
from .shared_labram_vqnsp import SharedLaBraMVQNSP
from .factorized_labram_vqnsp import FactorizedLaBraMVQNSP
from .codebook_focus_factorized_labram_vqnsp import (
    CodebookFocusedFactorizedLaBraMVQNSP,
    OverfitFactorizedLaBraMVQNSP,
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
_TOKENIZER_REGISTRY['vqvae'] = VQVAETokenizer
_TOKENIZER_REGISTRY['fsq'] = FSQTokenizer
_TOKENIZER_REGISTRY['patch_vqvae'] = PatchVQVAETokenizer
_TOKENIZER_REGISTRY['time_patch_vqvae'] = PatchVQVAETokenizer  # alias
_TOKENIZER_REGISTRY['freq_patch_vqvae'] = FreqDomainPatchVQVAE
_TOKENIZER_REGISTRY['freq_patch_vqvae_v2'] = FreqDomainPatchVQVAE_V2
_TOKENIZER_REGISTRY['neurorvq'] = NeuroRVQTokenizer
_TOKENIZER_REGISTRY['neurorvq_fnirs'] = NeuroRVQTokenizerFNIRS
_TOKENIZER_REGISTRY['labram_vqnsp'] = LaBraMVQNSP
_TOKENIZER_REGISTRY['labram_vqnsp_eeg'] = LaBraMVQNSP_EEG
_TOKENIZER_REGISTRY['labram_vqnsp_fnirs'] = LaBraMVQNSP_fNIRS
_TOKENIZER_REGISTRY['shared_labram_vqnsp'] = SharedLaBraMVQNSP
_TOKENIZER_REGISTRY['factorized_labram_vqnsp'] = FactorizedLaBraMVQNSP
_TOKENIZER_REGISTRY['codebook_focus_factorized_labram_vqnsp'] = CodebookFocusedFactorizedLaBraMVQNSP
_TOKENIZER_REGISTRY['overfit_factorized_labram_vqnsp'] = OverfitFactorizedLaBraMVQNSP

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
    'NeuroRVQTokenizerFNIRS',
    # LaBraM VQNSP
    'LaBraMVQNSP',
    'LaBraMVQNSP_EEG',
    'LaBraMVQNSP_fNIRS',
    'SharedLaBraMVQNSP',
    'FactorizedLaBraMVQNSP',
    'CodebookFocusedFactorizedLaBraMVQNSP',
    'OverfitFactorizedLaBraMVQNSP',
    # NeuroRVQ components
    'NormEMAVectorQuantizer',
    'ResidualVectorQuantization',
    'MultiScaleTemporalEncoder',
    'MultiScaleTemporalEncoderFNIRS',
    # Registry
    'register_tokenizer',
    'get_tokenizer_class',
    'create_tokenizer',
    'list_tokenizers',
    'StandardizedOutput',
]
