#!/usr/bin/env python
"""
Aggregate downstream run results with unified metrics schema.

This script scans run directories and computes:
- accuracy / balanced_accuracy / macro_f1 mean+-std
- class recall balance (gap between per-class recalls)
- per-subject accuracy mean+-std
"""

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Any


PROJECT_ROOT = Path(__file__).parent.parent.parent


def _safe_mean(xs: List[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def _safe_std(xs: List[float]) -> float:
    return float(pstdev(xs)) if len(xs) > 1 else 0.0


def _class_recall_gap(confusion_matrix: List[List[int]]) -> float:
    if not confusion_matrix or len(confusion_matrix) < 2:
        return 0.0

    recalls = []
    for i, row in enumerate(confusion_matrix):
        tp = row[i]
        total = sum(row)
        recalls.append((tp / total) if total > 0 else 0.0)

    return float(max(recalls) - min(recalls)) if recalls else 0.0


def _load_result(path: Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def aggregate(run_dirs: List[Path]) -> Dict[str, Any]:
    rows = []

    for run_dir in run_dirs:
        result_path = run_dir / 'results.json'
        if not result_path.exists():
            continue

        payload = _load_result(result_path)
        test = payload.get('test_metrics', {})
        subj_acc = payload.get('subject_accuracies') or {}

        subject_values = [float(v) for v in subj_acc.values()]
        macro_f1 = float(test.get('f1', 0.0))
        row = {
            'run_dir': str(run_dir),
            'experiment': payload.get('experiment', run_dir.name),
            'accuracy': float(test.get('accuracy', 0.0)),
            'balanced_accuracy': float(test.get('balanced_accuracy', 0.0)),
            'macro_f1': macro_f1,
            'class_recall_gap': _class_recall_gap(test.get('confusion_matrix', [])),
            'per_subject_acc_mean': _safe_mean(subject_values),
            'per_subject_acc_std': _safe_std(subject_values),
        }
        rows.append(row)

    summary = {
        'n_runs': len(rows),
        'metrics_schema': {
            'accuracy': 'mean+-std across runs',
            'balanced_accuracy': 'mean+-std across runs',
            'macro_f1': 'mean+-std across runs',
            'class_recall_gap': 'mean+-std across runs; lower is better',
            'per_subject_acc_mean': 'mean+-std of run-level per-subject mean accuracy',
            'per_subject_acc_std': 'mean+-std of run-level per-subject std accuracy',
        },
        'aggregate': {
            'accuracy_mean': _safe_mean([r['accuracy'] for r in rows]),
            'accuracy_std': _safe_std([r['accuracy'] for r in rows]),
            'balanced_accuracy_mean': _safe_mean([r['balanced_accuracy'] for r in rows]),
            'balanced_accuracy_std': _safe_std([r['balanced_accuracy'] for r in rows]),
            'macro_f1_mean': _safe_mean([r['macro_f1'] for r in rows]),
            'macro_f1_std': _safe_std([r['macro_f1'] for r in rows]),
            'class_recall_gap_mean': _safe_mean([r['class_recall_gap'] for r in rows]),
            'class_recall_gap_std': _safe_std([r['class_recall_gap'] for r in rows]),
            'per_subject_acc_mean_mean': _safe_mean([r['per_subject_acc_mean'] for r in rows]),
            'per_subject_acc_mean_std': _safe_std([r['per_subject_acc_mean'] for r in rows]),
            'per_subject_acc_std_mean': _safe_mean([r['per_subject_acc_std'] for r in rows]),
            'per_subject_acc_std_std': _safe_std([r['per_subject_acc_std'] for r in rows]),
        },
        'runs': rows,
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Aggregate downstream run results')
    parser.add_argument(
        '--runs-root',
        type=str,
        default='experiments/runs',
        help='Directory containing run folders',
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*',
        help='Glob pattern to select run folders, e.g. "foundation_v0_*"',
    )
    parser.add_argument(
        '--out',
        type=str,
        default='experiments/runs/downstream_aggregate_summary.json',
        help='Output JSON path',
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = PROJECT_ROOT / runs_root

    run_dirs = sorted([p for p in runs_root.glob(args.pattern) if p.is_dir()])
    summary = aggregate(run_dirs)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f'Aggregated {summary["n_runs"]} runs -> {out_path}')


if __name__ == '__main__':
    main()
