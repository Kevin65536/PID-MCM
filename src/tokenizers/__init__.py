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
from .ema_vector_quantizer import EMAVectorQuantizer, QuantizerOutput
from .physiology_semantic_tokenizer import PhysiologySemanticTokenizer
from .shared_driver_semantic_vq import (
    FullWindowModalityEncoder,
    SharedDriverContinuousModel,
    SharedDriverTrajectoryDecoder,
    TemporalPatchStem,
)
from .continuous_shared_private import ContinuousSharedPrivateModel
from .lag_conditioned_baseline import B0ContinuousSharedPrivate
from .lag_conditioned_shared_private_vq import (
    LCSPVQModel,
    LagAwareContinuousMatchingLoss,
    LocalCausalPatchEncoder,
    LowRankLagCouplingHead,
)
from .ssm_observation_shared_private import (
    CausalFIRTransferHead,
    ContinuousDecomposedTaskHead,
    ContinuousLagInteractionHead,
    SSMObservationSharedPrivateModel,
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
_TOKENIZER_REGISTRY['physiology_semantic'] = PhysiologySemanticTokenizer

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
    'EMAVectorQuantizer',
    'QuantizerOutput',
    'PhysiologySemanticTokenizer',
    'FullWindowModalityEncoder',
    'SharedDriverContinuousModel',
    'SharedDriverTrajectoryDecoder',
    'TemporalPatchStem',
    'ContinuousSharedPrivateModel',
    'B0ContinuousSharedPrivate',
    'LCSPVQModel',
    'LagAwareContinuousMatchingLoss',
    'LocalCausalPatchEncoder',
    'LowRankLagCouplingHead',
    'CausalFIRTransferHead',
    'ContinuousDecomposedTaskHead',
    'ContinuousLagInteractionHead',
    'SSMObservationSharedPrivateModel',
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
