"""Visualization utilities for tokenizer and classifier experiments.

The functions from ``token_physiology_plots`` are the standard, provenance-
aware figure surface for Token Physiology Atlas outputs. ``TokenizerVisualizer``
is retained for historical training dashboards.
"""

from .tokenizer_plots import TokenizerVisualizer, visualize_tokenizer_run
from .classifier_plots import ClassifierVisualizer, visualize_classifier_run
from .tensorboard_logger import TensorBoardLogger
from .gradient_diagnostics import (
    plot_gradient_conflict_dashboard,
    plot_gradient_influence_dashboard,
)
from .token_physiology_plots import (
    FigureArtifacts,
    INSUFFICIENT_SUPPORT_COLOR,
    MISSING_COLOR,
    SUPPORTED_COLOR,
    build_figure_manifest,
    plot_codebook_embedding_colored,
    plot_token_feature_heatmap,
    plot_token_feature_profile_ci,
    plot_token_support,
    save_figure_atomic,
)

__all__ = [
    "FigureArtifacts",
    "INSUFFICIENT_SUPPORT_COLOR",
    "MISSING_COLOR",
    "SUPPORTED_COLOR",
    "build_figure_manifest",
    "plot_codebook_embedding_colored",
    "plot_token_feature_heatmap",
    "plot_token_feature_profile_ci",
    "plot_token_support",
    "save_figure_atomic",
    "TokenizerVisualizer",
    "visualize_tokenizer_run",
    "ClassifierVisualizer",
    "visualize_classifier_run",
    "TensorBoardLogger",
    "plot_gradient_conflict_dashboard",
    "plot_gradient_influence_dashboard",
]
