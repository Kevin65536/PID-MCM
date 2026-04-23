from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable


ANALYSIS_SUITE_VERSION = 'v2'


def prepare_analysis_layout(
    suite_root: Path,
    analysis_name: str,
    splits: Iterable[str],
    *,
    metadata: Dict[str, object] | None = None,
) -> Dict[str, object]:
    suite_root = Path(suite_root)
    analysis_root = suite_root / analysis_name
    metrics_root = analysis_root / 'metrics'
    figures_root = analysis_root / 'figures'
    splits_root = analysis_root / 'splits'

    suite_root.mkdir(parents=True, exist_ok=True)
    analysis_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    splits_root.mkdir(parents=True, exist_ok=True)

    split_layouts: Dict[str, Dict[str, Path]] = {}
    for split_name in splits:
        split_root = splits_root / split_name
        split_metrics = split_root / 'metrics'
        split_figures = split_root / 'figures'
        split_metrics.mkdir(parents=True, exist_ok=True)
        split_figures.mkdir(parents=True, exist_ok=True)
        split_layouts[str(split_name)] = {
            'root': split_root,
            'metrics': split_metrics,
            'figures': split_figures,
        }

    _update_suite_manifest(
        suite_root=suite_root,
        analysis_name=analysis_name,
        analysis_root=analysis_root,
        splits=list(split_layouts),
        metadata=metadata or {},
    )

    return {
        'suite_root': suite_root,
        'analysis_root': analysis_root,
        'metrics_root': metrics_root,
        'figures_root': figures_root,
        'splits_root': splits_root,
        'splits': split_layouts,
    }


def _update_suite_manifest(
    suite_root: Path,
    analysis_name: str,
    analysis_root: Path,
    splits: list[str],
    metadata: Dict[str, object],
) -> None:
    manifest_path = suite_root / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            'suite_version': ANALYSIS_SUITE_VERSION,
            'analyses': {},
        }

    analyses = manifest.setdefault('analyses', {})
    analyses[analysis_name] = {
        'root': str(analysis_root.relative_to(suite_root)),
        'splits': splits,
        'metadata': metadata,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')


__all__ = ['ANALYSIS_SUITE_VERSION', 'prepare_analysis_layout']