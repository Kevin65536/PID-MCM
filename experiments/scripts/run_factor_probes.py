#!/usr/bin/env python
"""
Run Phase-A factor probes on exported run embeddings.

Expected run artifact:
- experiments/runs/<run_name>/probes/train_embeddings.npz
- experiments/runs/<run_name>/probes/val_embeddings.npz
- experiments/runs/<run_name>/probes/test_embeddings.npz

Each NPZ should contain:
- embedding: [N, D]
- label: [N]
- subject_id: [N] (optional for subject probe)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).parent.parent.parent


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    payload = {k: data[k] for k in data.files}
    data.close()
    return payload


def _fit_and_eval_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
) -> Dict[str, float]:
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=3000)),
    ])
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_eval)

    return {
        'accuracy': float(accuracy_score(y_eval, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_eval, y_pred)),
        'macro_f1': float(f1_score(y_eval, y_pred, average='macro', zero_division=0)),
    }


def _cross_validated_probe(
    x: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> Dict[str, Any]:
    unique_labels, counts = np.unique(y, return_counts=True)
    if unique_labels.shape[0] < 2:
        return {
            'status': 'skipped',
            'reason': 'Need at least 2 classes for probe.',
        }

    max_splits = int(counts.min())
    if max_splits < 2:
        return {
            'status': 'skipped',
            'reason': 'Not enough samples per class for cross-validation.',
        }

    actual_splits = min(n_splits, max_splits)
    splitter = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)

    fold_metrics: List[Dict[str, float]] = []
    for train_idx, test_idx in splitter.split(x, y):
        metrics = _fit_and_eval_classifier(
            x[train_idx],
            y[train_idx],
            x[test_idx],
            y[test_idx],
        )
        fold_metrics.append(metrics)

    return {
        'status': 'ok',
        'n_splits': actual_splits,
        'accuracy_mean': float(np.mean([m['accuracy'] for m in fold_metrics])),
        'accuracy_std': float(np.std([m['accuracy'] for m in fold_metrics])),
        'balanced_accuracy_mean': float(np.mean([m['balanced_accuracy'] for m in fold_metrics])),
        'balanced_accuracy_std': float(np.std([m['balanced_accuracy'] for m in fold_metrics])),
        'macro_f1_mean': float(np.mean([m['macro_f1'] for m in fold_metrics])),
        'macro_f1_std': float(np.std([m['macro_f1'] for m in fold_metrics])),
        'fold_metrics': fold_metrics,
    }


def run_probes(run_dir: Path) -> Dict[str, Any]:
    probe_dir = run_dir / 'probes'
    train_path = probe_dir / 'train_embeddings.npz'
    val_path = probe_dir / 'val_embeddings.npz'
    test_path = probe_dir / 'test_embeddings.npz'

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f'Missing probe data in {probe_dir}. Please run train_downstream.py with logging.export_probe_data=true.'
        )

    train = _load_npz(train_path)
    test = _load_npz(test_path)
    val = _load_npz(val_path) if val_path.exists() else None

    x_train = train['embedding']
    y_train_task = train['label']
    x_test = test['embedding']
    y_test_task = test['label']

    task_probe = _fit_and_eval_classifier(x_train, y_train_task, x_test, y_test_task)

    probes = {
        'task_probe': {
            'name': 'Probe(Zt_proxy -> task)',
            **task_probe,
        },
        'task_probe_train_cv': {
            'name': 'Probe(Zt_proxy -> task) within-train CV',
            **_cross_validated_probe(x_train, y_train_task),
        }
    }

    if 'subject_id' in train:
        probes['subject_probe_train_cv'] = {
            'name': 'Probe(Zt_proxy -> subject) within-train CV',
            **_cross_validated_probe(x_train, train['subject_id']),
        }

    if 'subject_id' in train and 'subject_id' in test:
        y_train_subj = train['subject_id']
        y_test_subj = test['subject_id']
        if set(np.unique(y_test_subj)).issubset(set(np.unique(y_train_subj))):
            subj_probe = _fit_and_eval_classifier(x_train, y_train_subj, x_test, y_test_subj)
            probes['subject_probe'] = {
                'name': 'Probe(Zt_proxy -> subject)',
                **subj_probe,
            }
        else:
            probes['subject_probe'] = {
                'name': 'Probe(Zt_proxy -> subject)',
                'status': 'incompatible',
                'reason': 'Eval subject IDs are unseen in train split under cross-subject protocol.',
            }

    if val is not None:
        x_val = val['embedding']
        y_val_task = val['label']
        probes['task_probe_val'] = {
            'name': 'Probe(Zt_proxy -> task) on val',
            **_fit_and_eval_classifier(x_train, y_train_task, x_val, y_val_task),
        }

        if 'subject_id' in train and 'subject_id' in val:
            if set(np.unique(val['subject_id'])).issubset(set(np.unique(train['subject_id']))):
                probes['subject_probe_val'] = {
                    'name': 'Probe(Zt_proxy -> subject) on val',
                    **_fit_and_eval_classifier(x_train, train['subject_id'], x_val, val['subject_id']),
                }
            else:
                probes['subject_probe_val'] = {
                    'name': 'Probe(Zt_proxy -> subject) on val',
                    'status': 'incompatible',
                    'reason': 'Val subject IDs are unseen in train split under cross-subject protocol.',
                }

    report = {
        'run_dir': str(run_dir),
        'probe_dir': str(probe_dir),
        'n_train': int(x_train.shape[0]),
        'n_test': int(x_test.shape[0]),
        'embedding_dim': int(x_train.shape[1]),
        'probes': probes,
    }

    out_file = probe_dir / 'factor_probe_report.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Run factor probes on exported embeddings')
    parser.add_argument(
        '--run-dir',
        type=str,
        required=True,
        help='Path to a run directory under experiments/runs/',
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir

    report = run_probes(run_dir)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
