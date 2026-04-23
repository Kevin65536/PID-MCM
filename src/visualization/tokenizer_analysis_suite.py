from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import torch

from .alignment_analysis import analyze_alignment
from .semantic_space_analysis import analyze_semantic_space


def generate_tokenizer_analysis_suite(
    *,
    model,
    dataloaders,
    config: Dict[str, object],
    run_dir: Path,
    device: torch.device,
    output_dir: Optional[Path] = None,
    splits: Iterable[str] = ('val', 'test'),
    analysis_type: Optional[str] = None,
    max_batches: Optional[int] = None,
    max_feature_samples: int = 20000,
    max_probe_samples: Optional[int] = None,
    augmentation_probe_batches: Optional[int] = None,
    probe_seed: Optional[int] = None,
) -> Dict[str, object]:
    suite_root = Path(output_dir) if output_dir is not None else Path(run_dir) / 'analysis' / 'tokenizer_report'

    alignment_results = analyze_alignment(
        model=model,
        dataloaders=dataloaders,
        config=config,
        output_dir=suite_root,
        device=device,
        splits=splits,
        analysis_type=analysis_type,
    )
    semantic_results = analyze_semantic_space(
        model=model,
        dataloaders=dataloaders,
        config=config,
        output_dir=suite_root,
        device=device,
        splits=splits,
        run_dir=Path(run_dir),
        max_batches=max_batches,
        max_feature_samples=max_feature_samples,
        max_probe_samples=max_probe_samples,
        augmentation_probe_batches=augmentation_probe_batches,
        probe_seed=probe_seed,
    )

    return {
        'output_dir': str(suite_root),
        'alignment': alignment_results,
        'semantic': semantic_results,
    }


__all__ = ['generate_tokenizer_analysis_suite']