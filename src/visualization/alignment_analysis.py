from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable

import torch

from .factorized_alignment_analysis import analyze_factorized_alignment
from .shared_alignment_analysis import analyze_shared_alignment

AlignmentAnalyzer = Callable[..., Dict[str, object]]

ALIGNMENT_ANALYZERS: Dict[str, AlignmentAnalyzer] = {
    'shared_alignment': analyze_shared_alignment,
    'factorized_alignment': analyze_factorized_alignment,
}


def resolve_alignment_analysis_type(model, analysis_type: str | None = None) -> str:
    if analysis_type is not None:
        return analysis_type
    return getattr(model, 'get_analysis_type', lambda: 'shared_alignment')()


def get_alignment_analyzer(analysis_type: str) -> AlignmentAnalyzer:
    try:
        return ALIGNMENT_ANALYZERS[analysis_type]
    except KeyError as exc:
        known_types = ', '.join(sorted(ALIGNMENT_ANALYZERS))
        raise ValueError(f'Unknown alignment analysis type: {analysis_type}. Known types: {known_types}') from exc


def analyze_alignment(
    model,
    dataloaders: Dict[str, object],
    config: Dict[str, object],
    output_dir: Path,
    device: torch.device,
    splits: Iterable[str] = ('val', 'test'),
    analysis_type: str | None = None,
) -> Dict[str, object]:
    resolved_type = resolve_alignment_analysis_type(model, analysis_type)
    analyzer = get_alignment_analyzer(resolved_type)
    return analyzer(
        model=model,
        dataloaders=dataloaders,
        config=config,
        output_dir=output_dir,
        device=device,
        splits=splits,
    )


__all__ = [
    'ALIGNMENT_ANALYZERS',
    'analyze_alignment',
    'get_alignment_analyzer',
    'resolve_alignment_analysis_type',
]