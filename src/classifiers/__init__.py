"""
Classifiers for downstream tasks on tokenized representations.
"""

from .simple_classifier import SimpleTokenClassifier, TokenClassifierHead
from .end_to_end import EndToEndClassifier

__all__ = [
    'SimpleTokenClassifier',
    'TokenClassifierHead', 
    'EndToEndClassifier',
]
