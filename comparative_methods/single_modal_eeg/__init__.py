"""Shared leakage-bounded runners for EEG foundation-model comparisons."""

from .contract import EEGTaskView, load_public_contract

__all__ = ["EEGTaskView", "load_public_contract"]
