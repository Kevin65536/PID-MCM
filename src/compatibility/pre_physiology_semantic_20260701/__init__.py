"""Pre-redesign source/observation compatibility surface.

Importing :mod:`src.tokenizers` does not register this model. Historical tools
must opt in by calling :func:`register_legacy_tokenizers`.
"""

from src.tokenizers.registry import _TOKENIZER_REGISTRY

from .source_observation_tokenizer import SourceObservationLaBraMVQNSP


def register_legacy_tokenizers() -> None:
    """Register checkpoint-compatible tokenizer names for explicit legacy use."""

    _TOKENIZER_REGISTRY['source_observation_labram_vqnsp'] = SourceObservationLaBraMVQNSP
    _TOKENIZER_REGISTRY['factorized_labram_vqnsp'] = SourceObservationLaBraMVQNSP


__all__ = ['SourceObservationLaBraMVQNSP', 'register_legacy_tokenizers']
