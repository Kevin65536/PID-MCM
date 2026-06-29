#!/usr/bin/env python
"""Reselect schedule-complete alignment checkpoints from completed tokenizer runs."""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
import yaml

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from experiments.scripts.train_source_observation_tokenizer import (
    alignment_checkpoint_min_epoch,
    cross_modal_alignment_enabled,
)


def _number(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) else None


def _checkpoint_candidates(run_dir: Path) -> Iterable[Path]:
    checkpoint_dir = run_dir / 'checkpoints'
    names = ['best_hard_primary.pt', 'final_model.pt']
    paths = [checkpoint_dir / name for name in names]
    paths.extend(sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt')))
    return [path for path in paths if path.exists()]


def select_alignment_checkpoint(
    run_dir: Path,
    *,
    max_primary_relative_degradation: float = 0.05,
    output_name: str = 'best_alignment_eligible.pt',
) -> Dict[str, Any]:
    config = yaml.safe_load((run_dir / 'config.yaml').read_text(encoding='utf-8'))
    if not cross_modal_alignment_enabled(config):
        return {'run': run_dir.name, 'status': 'no_alignment_objective'}

    metrics = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
    epoch_metrics = {int(item['epoch']): item for item in metrics.get('epochs', [])}
    primary_values = [
        _number(item.get('val_primary_loss')) for item in epoch_metrics.values()
    ]
    primary_values = [value for value in primary_values if value is not None]
    if not primary_values:
        return {'run': run_dir.name, 'status': 'missing_primary_metrics'}

    best_primary = min(primary_values)
    primary_limit = best_primary * (1.0 + max(max_primary_relative_degradation, 0.0))
    min_epoch = alignment_checkpoint_min_epoch(config)
    candidates = []
    seen_epochs = set()
    for checkpoint_path in _checkpoint_candidates(run_dir):
        payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
        epoch = int(payload.get('epoch', -1))
        if epoch in seen_epochs or epoch < min_epoch or epoch not in epoch_metrics:
            continue
        seen_epochs.add(epoch)
        values = epoch_metrics[epoch]
        alignment = _number(values.get('val_cross_modal_alignment_unscaled_loss'))
        primary = _number(values.get('val_primary_loss'))
        if alignment is None or primary is None or primary > primary_limit:
            continue
        candidates.append({
            'checkpoint': checkpoint_path,
            'epoch': epoch,
            'alignment_unscaled_loss': alignment,
            'primary_loss': primary,
        })

    if not candidates:
        return {
            'run': run_dir.name,
            'status': 'no_eligible_checkpoint',
            'min_epoch': min_epoch,
            'best_primary_loss': best_primary,
            'primary_loss_limit': primary_limit,
        }

    selected = min(candidates, key=lambda item: (item['alignment_unscaled_loss'], item['primary_loss']))
    destination = run_dir / 'checkpoints' / output_name
    shutil.copy2(selected['checkpoint'], destination)
    result = {
        'run': run_dir.name,
        'status': 'selected',
        'selection_metric': 'val_cross_modal_alignment_unscaled_loss',
        'min_epoch': min_epoch,
        'max_primary_relative_degradation': max_primary_relative_degradation,
        'best_primary_loss': best_primary,
        'primary_loss_limit': primary_limit,
        'selected_epoch': selected['epoch'],
        'selected_alignment_unscaled_loss': selected['alignment_unscaled_loss'],
        'selected_primary_loss': selected['primary_loss'],
        'source_checkpoint': selected['checkpoint'].name,
        'output_checkpoint': destination.name,
        'eligible_candidates': [
            {
                **{key: value for key, value in item.items() if key != 'checkpoint'},
                'checkpoint': item['checkpoint'].name,
            }
            for item in sorted(candidates, key=lambda item: item['epoch'])
        ],
    }
    (run_dir / 'alignment_checkpoint_selection.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8'
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-dir', required=True)
    parser.add_argument('--stage', default='screening')
    parser.add_argument('--conditions', nargs='*', default=None)
    parser.add_argument('--max-primary-relative-degradation', type=float, default=0.05)
    args = parser.parse_args()

    stage_dir = Path(args.suite_dir) / args.stage
    conditions = {item.upper() for item in args.conditions} if args.conditions else None
    results = []
    for run_dir in sorted(stage_dir.glob('k128*')):
        parts = run_dir.name.split('_')
        condition = next((part.upper() for part in parts if part.lower().startswith('z')), '')
        if conditions is not None and condition not in conditions:
            continue
        results.append(select_alignment_checkpoint(
            run_dir,
            max_primary_relative_degradation=args.max_primary_relative_degradation,
        ))

    summary = {
        'schema_version': 'alignment_checkpoint_reselection_v1',
        'stage': args.stage,
        'results': results,
    }
    output_path = Path(args.suite_dir) / f'{args.stage}_alignment_checkpoint_reselection.json'
    output_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
