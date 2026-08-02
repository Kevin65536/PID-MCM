"""GPU components for the BrainFusion independent reimplementation."""

from .nvc import (
    NVCConfig,
    brainfusion_cpu_nvc_reference_avg_raw,
    brainfusion_gpu_nvc_avg_raw,
    brainfusion_nvc_contribution_timeseries,
)
from .features import BrainFusionFeaturePipeline, CSPConfig, NVCPairSelector, TorchCSP
from .stacking import FoldLocalStackingClassifier, StackingConfig
from .pipeline import BrainFusionFoldPipeline

__all__ = [
    "NVCConfig",
    "brainfusion_cpu_nvc_reference_avg_raw",
    "brainfusion_gpu_nvc_avg_raw",
    "brainfusion_nvc_contribution_timeseries",
    "BrainFusionFeaturePipeline",
    "CSPConfig",
    "NVCPairSelector",
    "TorchCSP",
    "FoldLocalStackingClassifier",
    "StackingConfig",
    "BrainFusionFoldPipeline",
]
