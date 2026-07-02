from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable

import torch

from .source_observation_analysis import generate_source_observation_scorecard

AlignmentAnalyzer = Callable[..., Dict[str, object]]


def _analyze_source_observation_alignment(**kwargs) -> Dict[str, object]:
    scorecard = generate_source_observation_scorecard(**kwargs)
    return {
        'analysis_type': 'source_observation_alignment',
        'artifact_root': scorecard['artifact_root'],
        'primary_split': scorecard['primary_split'],
        'gates': {
            name: details
            for name, details in scorecard.get('gates', {}).items()
            if name in {'gate1', 'gate2', 'gate3'}
        },
        'splits': {
            split_name: {
                'gates': {
                    name: details
                    for name, details in split_payload.get('gates', {}).items()
                    if name in {'gate1', 'gate2', 'gate3'}
                }
            }
            for split_name, split_payload in scorecard.get('splits', {}).items()
            if split_payload.get('available', False)
        },
    }


ALIGNMENT_ANALYZERS: Dict[str, AlignmentAnalyzer] = {
    'source_observation_alignment': _analyze_source_observation_alignment,
}


def resolve_alignment_analysis_type(model, analysis_type: str | None = None) -> str:
    if analysis_type is not None:
        return analysis_type
    return getattr(model, 'get_analysis_type', lambda: 'source_observation_alignment')()


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