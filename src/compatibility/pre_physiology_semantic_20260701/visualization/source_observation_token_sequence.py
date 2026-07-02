from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


TOKEN_SEQUENCE_SCHEMA_VERSION = 'source_observation_token_sequence_v1'

BRANCHES: Tuple[Tuple[str, str], ...] = (
    ('eeg_source_tokens', 'EEG source'),
    ('fnirs_source_tokens', 'fNIRS source'),
    ('eeg_observation_tokens', 'EEG observation'),
    ('fnirs_observation_tokens', 'fNIRS observation'),
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fieldnames})


def _save_figure(figure, path: Path, *, dpi: int = 160) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)
    return str(path)


def _load_json_optional(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def _load_split_npz(run_dir: Path, split: str) -> Dict[str, np.ndarray]:
    path = run_dir / 'tokens' / f'{split}_tokens.npz'
    if not path.exists():
        raise FileNotFoundError(f'Token file not found: {path}')
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _available_splits(run_dir: Path, requested: Optional[Iterable[str]] = None) -> List[str]:
    if requested:
        return [str(split) for split in requested]
    token_dir = run_dir / 'tokens'
    if not token_dir.exists():
        raise FileNotFoundError(f'Token directory not found: {token_dir}')
    splits = []
    for path in sorted(token_dir.glob('*_tokens.npz')):
        suffix = '_tokens.npz'
        if path.name.endswith(suffix):
            splits.append(path.name[:-len(suffix)])
    if not splits:
        raise FileNotFoundError(f'No *_tokens.npz files found under {token_dir}')
    preferred_order = {'train': 0, 'val': 1, 'test': 2}
    return sorted(splits, key=lambda split: (preferred_order.get(split, 100), split))


def _infer_vocab_size(tokens: np.ndarray, manifest: Mapping[str, Any], branch_key: str) -> int:
    token_semantics = manifest.get('token_semantics', {}) if isinstance(manifest.get('token_semantics'), Mapping) else {}
    semantic_key = branch_key.replace('_tokens', '_vocab_size')
    if semantic_key in token_semantics:
        return int(token_semantics[semantic_key])
    if tokens.size == 0:
        return 1
    return int(np.max(tokens)) + 1


def _distribution(tokens: np.ndarray, vocab_size: int) -> np.ndarray:
    flat = tokens.reshape(-1).astype(np.int64, copy=False)
    counts = np.bincount(flat, minlength=max(vocab_size, 1)).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return counts
    return counts / total


def _entropy(probabilities: np.ndarray) -> float:
    nonzero = probabilities[probabilities > 0]
    if nonzero.size == 0:
        return 0.0
    return float(-(nonzero * np.log2(nonzero)).sum())


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    size = max(left.shape[0], right.shape[0])
    left_pad = np.pad(left, (0, size - left.shape[0]))
    right_pad = np.pad(right, (0, size - right.shape[0]))
    midpoint = 0.5 * (left_pad + right_pad)
    return float(_kl_divergence(left_pad, midpoint) * 0.5 + _kl_divergence(right_pad, midpoint) * 0.5)


def _kl_divergence(left: np.ndarray, right: np.ndarray) -> float:
    mask = left > 0
    if not mask.any():
        return 0.0
    return float((left[mask] * np.log2(left[mask] / np.clip(right[mask], 1e-12, None))).sum())


def _transition_matrix(tokens: np.ndarray, vocab_size: int) -> np.ndarray:
    matrix = np.zeros((vocab_size, vocab_size), dtype=np.int64)
    if tokens.ndim != 2 or tokens.shape[1] < 2:
        return matrix
    starts = tokens[:, :-1].reshape(-1).astype(np.int64, copy=False)
    ends = tokens[:, 1:].reshape(-1).astype(np.int64, copy=False)
    valid = (starts >= 0) & (starts < vocab_size) & (ends >= 0) & (ends < vocab_size)
    np.add.at(matrix, (starts[valid], ends[valid]), 1)
    return matrix


def _joint_matrix(left: np.ndarray, right: np.ndarray, left_vocab: int, right_vocab: int, lag: int = 0) -> np.ndarray:
    if lag > 0:
        left_aligned = left[:, :-lag]
        right_aligned = right[:, lag:]
    elif lag < 0:
        left_aligned = left[:, -lag:]
        right_aligned = right[:, :lag]
    else:
        left_aligned = left
        right_aligned = right
    matrix = np.zeros((left_vocab, right_vocab), dtype=np.int64)
    if left_aligned.size == 0 or right_aligned.size == 0:
        return matrix
    left_flat = left_aligned.reshape(-1).astype(np.int64, copy=False)
    right_flat = right_aligned.reshape(-1).astype(np.int64, copy=False)
    valid = (left_flat >= 0) & (left_flat < left_vocab) & (right_flat >= 0) & (right_flat < right_vocab)
    np.add.at(matrix, (left_flat[valid], right_flat[valid]), 1)
    return matrix


def _mutual_information(joint_counts: np.ndarray) -> float:
    total = joint_counts.sum()
    if total <= 0:
        return 0.0
    joint = joint_counts.astype(np.float64) / float(total)
    left = joint.sum(axis=1, keepdims=True)
    right = joint.sum(axis=0, keepdims=True)
    expected = left @ right
    mask = joint > 0
    return float((joint[mask] * np.log2(joint[mask] / np.clip(expected[mask], 1e-12, None))).sum())


def _normalized_matrix(matrix: np.ndarray) -> np.ndarray:
    total = matrix.sum()
    if total <= 0:
        return matrix.astype(np.float64)
    return matrix.astype(np.float64) / float(total)


def _safe_text_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).astype(str)


def _sort_indices(data: Mapping[str, np.ndarray]) -> np.ndarray:
    count = int(next(iter(data.values())).shape[0])
    task = _safe_text_array(data.get('source_task', np.asarray([''] * count)))
    subject = np.asarray(data.get('subject_id', np.full(count, -1))).astype(np.int64, copy=False)
    label = np.asarray(data.get('label', np.full(count, -1))).astype(np.int64, copy=False)
    event = np.asarray(data.get('event_idx', np.full(count, -1))).astype(np.int64, copy=False)
    return np.lexsort((event, label, subject, task))


def _sample_sorted_indices(data: Mapping[str, np.ndarray], max_samples: int) -> np.ndarray:
    sorted_indices = _sort_indices(data)
    if sorted_indices.shape[0] <= max_samples:
        return sorted_indices
    positions = np.linspace(0, sorted_indices.shape[0] - 1, num=max_samples, dtype=np.int64)
    return sorted_indices[positions]


def _build_sequence_features(split_arrays: Mapping[str, Mapping[str, np.ndarray]], vocab_sizes: Mapping[str, int]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    features = []
    metadata: Dict[str, List[np.ndarray]] = {
        'split': [],
        'source_task': [],
        'subject_id': [],
        'label': [],
    }
    for split, data in split_arrays.items():
        count = data['eeg_source_tokens'].shape[0]
        split_features = []
        for branch_key, _ in BRANCHES:
            vocab_size = vocab_sizes[branch_key]
            branch_features = np.zeros((count, vocab_size), dtype=np.float32)
            tokens = data[branch_key].astype(np.int64, copy=False)
            for sample_idx in range(count):
                counts = np.bincount(tokens[sample_idx], minlength=vocab_size).astype(np.float32)
                branch_features[sample_idx] = counts / max(float(counts.sum()), 1.0)
            split_features.append(branch_features)
        features.append(np.concatenate(split_features, axis=1))
        metadata['split'].append(np.asarray([split] * count, dtype=str))
        metadata['source_task'].append(_safe_text_array(data.get('source_task', np.asarray([''] * count))))
        metadata['subject_id'].append(np.asarray(data.get('subject_id', np.full(count, -1))).astype(np.int64, copy=False))
        metadata['label'].append(np.asarray(data.get('label', np.full(count, -1))).astype(np.int64, copy=False))

    if not features:
        return np.empty((0, 0), dtype=np.float32), {key: np.asarray([]) for key in metadata}
    merged_metadata = {key: np.concatenate(values, axis=0) for key, values in metadata.items()}
    return np.concatenate(features, axis=0), merged_metadata


def _pca_2d(features: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    if features.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32), [0.0, 0.0]
    centered = features.astype(np.float64) - features.mean(axis=0, keepdims=True)
    if centered.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32), [0.0, 0.0]
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2].T
    coords = centered @ components
    if coords.shape[1] == 1:
        coords = np.concatenate([coords, np.zeros((coords.shape[0], 1))], axis=1)
    variances = singular_values ** 2 / max(centered.shape[0] - 1, 1)
    total_variance = float(variances.sum())
    ratios = (variances[:2] / total_variance).tolist() if total_variance > 0 else [0.0, 0.0]
    while len(ratios) < 2:
        ratios.append(0.0)
    return coords[:, :2].astype(np.float32), [float(ratios[0]), float(ratios[1])]


def _plot_token_heatmaps(
    split_arrays: Mapping[str, Mapping[str, np.ndarray]],
    vocab_sizes: Mapping[str, int],
    figure_dir: Path,
    *,
    max_samples: int,
    dpi: int,
) -> str:
    if not HAS_MATPLOTLIB:
        return ''
    splits = list(split_arrays)
    figure, axes = plt.subplots(
        len(splits),
        len(BRANCHES),
        figsize=(4.0 * len(BRANCHES), max(2.8, 2.4 * len(splits))),
        squeeze=False,
    )
    for row_idx, split in enumerate(splits):
        data = split_arrays[split]
        indices = _sample_sorted_indices(data, max_samples)
        for col_idx, (branch_key, branch_name) in enumerate(BRANCHES):
            ax = axes[row_idx][col_idx]
            tokens = data[branch_key][indices]
            image = ax.imshow(
                tokens,
                aspect='auto',
                interpolation='nearest',
                vmin=0,
                vmax=max(vocab_sizes[branch_key] - 1, 1),
                cmap='viridis',
            )
            ax.set_title(f'{split}: {branch_name}')
            ax.set_xlabel('2s token index')
            ax.set_ylabel('sample')
            ax.set_xticks(range(tokens.shape[1]))
            figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return _save_figure(figure, figure_dir / 'token_heatmaps_by_split.png', dpi=dpi)


def _plot_code_usage(
    split_distributions: Mapping[str, Mapping[str, np.ndarray]],
    figure_dir: Path,
    *,
    dpi: int,
) -> str:
    if not HAS_MATPLOTLIB:
        return ''
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False)
    axes_flat = axes.reshape(-1)
    for ax, (branch_key, branch_name) in zip(axes_flat, BRANCHES):
        for split, branch_distributions in split_distributions.items():
            probs = branch_distributions[branch_key]
            ax.plot(np.arange(probs.shape[0]), probs, marker='o', markersize=2, linewidth=1.2, label=split)
        ax.set_title(branch_name)
        ax.set_xlabel('code id')
        ax.set_ylabel('frequency')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
    return _save_figure(figure, figure_dir / 'code_usage_histograms.png', dpi=dpi)


def _plot_transition_matrices(
    transitions: Mapping[str, Mapping[str, np.ndarray]],
    figure_dir: Path,
    *,
    dpi: int,
) -> str:
    if not HAS_MATPLOTLIB:
        return ''
    split = 'train' if 'train' in transitions else next(iter(transitions))
    figure, axes = plt.subplots(2, 2, figsize=(10, 9), squeeze=False)
    for ax, (branch_key, branch_name) in zip(axes.reshape(-1), BRANCHES):
        matrix = _normalized_matrix(transitions[split][branch_key])
        image = ax.imshow(matrix, aspect='auto', interpolation='nearest', cmap='magma')
        ax.set_title(f'{split}: {branch_name}')
        ax.set_xlabel('next code')
        ax.set_ylabel('current code')
        figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return _save_figure(figure, figure_dir / 'transition_matrices.png', dpi=dpi)


def _plot_cross_modal_alignment(
    split_arrays: Mapping[str, Mapping[str, np.ndarray]],
    vocab_sizes: Mapping[str, int],
    figure_dir: Path,
    *,
    dpi: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    if not HAS_MATPLOTLIB:
        return '', []
    split = 'train' if 'train' in split_arrays else next(iter(split_arrays))
    data = split_arrays[split]
    lags = [0, 1, 2]
    figure, axes = plt.subplots(1, len(lags), figsize=(4.2 * len(lags), 3.8), squeeze=False)
    metrics = []
    for ax, lag in zip(axes.reshape(-1), lags):
        matrix = _joint_matrix(
            data['eeg_source_tokens'],
            data['fnirs_source_tokens'],
            vocab_sizes['eeg_source_tokens'],
            vocab_sizes['fnirs_source_tokens'],
            lag=lag,
        )
        metrics.append({
            'split': split,
            'lag_tokens': lag,
            'lag_seconds': lag * 2.0,
            'mutual_information_bits': _mutual_information(matrix),
            'pair_count': int(matrix.sum()),
        })
        image = ax.imshow(_normalized_matrix(matrix), aspect='auto', interpolation='nearest', cmap='cividis')
        ax.set_title(f'EEG src -> fNIRS src lag {lag}')
        ax.set_xlabel('fNIRS source code')
        ax.set_ylabel('EEG source code')
        figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return _save_figure(figure, figure_dir / 'cross_modal_alignment.png', dpi=dpi), metrics


def _plot_divergence(
    divergence_rows: Sequence[Mapping[str, Any]],
    figure_dir: Path,
    *,
    dpi: int,
) -> str:
    if not HAS_MATPLOTLIB:
        return ''
    pairs = ['eeg_vs_fnirs_source', 'eeg_source_vs_observation', 'fnirs_source_vs_observation']
    splits = sorted({str(row['split']) for row in divergence_rows})
    values = {
        pair: [next(float(row['js_divergence_bits']) for row in divergence_rows if row['split'] == split and row['pair'] == pair) for split in splits]
        for pair in pairs
    }
    x = np.arange(len(splits))
    width = 0.24
    figure, ax = plt.subplots(figsize=(9, 4.5))
    for idx, pair in enumerate(pairs):
        ax.bar(x + (idx - 1) * width, values[pair], width=width, label=pair.replace('_', ' '))
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel('JS divergence (bits)')
    ax.set_title('Branch token-distribution divergence')
    ax.legend(fontsize=8)
    return _save_figure(figure, figure_dir / 'source_observation_divergence.png', dpi=dpi)


def _plot_pca(
    features: np.ndarray,
    metadata: Mapping[str, np.ndarray],
    figure_dir: Path,
    *,
    dpi: int,
) -> Tuple[str, List[float]]:
    if not HAS_MATPLOTLIB:
        return '', [0.0, 0.0]
    coords, variance_ratio = _pca_2d(features)
    if coords.shape[0] == 0:
        return '', variance_ratio

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    ax_task, ax_subject = axes.reshape(-1)
    tasks = _safe_text_array(metadata['source_task'])
    unique_tasks = sorted(set(tasks.tolist()))
    cmap = plt.get_cmap('tab10')
    for idx, task in enumerate(unique_tasks):
        mask = tasks == task
        ax_task.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.75, color=cmap(idx % 10), label=task)
    ax_task.set_title('Sequence histogram PCA by task')
    ax_task.set_xlabel(f'PC1 ({variance_ratio[0] * 100:.1f}%)')
    ax_task.set_ylabel(f'PC2 ({variance_ratio[1] * 100:.1f}%)')
    ax_task.legend(fontsize=8, loc='best')

    subjects = np.asarray(metadata['subject_id'], dtype=np.float32)
    scatter = ax_subject.scatter(coords[:, 0], coords[:, 1], c=subjects, s=18, alpha=0.75, cmap='viridis')
    ax_subject.set_title('Sequence histogram PCA by subject')
    ax_subject.set_xlabel(f'PC1 ({variance_ratio[0] * 100:.1f}%)')
    ax_subject.set_ylabel(f'PC2 ({variance_ratio[1] * 100:.1f}%)')
    figure.colorbar(scatter, ax=ax_subject, fraction=0.046, pad=0.04, label='subject_id')
    return _save_figure(figure, figure_dir / 'task_subject_pca.png', dpi=dpi), variance_ratio


def _build_usage_rows(
    split_arrays: Mapping[str, Mapping[str, np.ndarray]],
    manifest: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, np.ndarray]], Dict[str, int]]:
    rows = []
    distributions: Dict[str, Dict[str, np.ndarray]] = {}
    vocab_sizes: Dict[str, int] = {}
    for branch_key, _ in BRANCHES:
        branch_vocab = max(_infer_vocab_size(data[branch_key], manifest, branch_key) for data in split_arrays.values())
        vocab_sizes[branch_key] = branch_vocab

    for split, data in split_arrays.items():
        distributions[split] = {}
        for branch_key, branch_name in BRANCHES:
            vocab_size = vocab_sizes[branch_key]
            probs = _distribution(data[branch_key], vocab_size)
            distributions[split][branch_key] = probs
            counts = np.bincount(data[branch_key].reshape(-1).astype(np.int64), minlength=vocab_size)
            nonzero = np.flatnonzero(counts)
            top_codes = sorted(
                ((int(code), int(counts[code])) for code in nonzero),
                key=lambda item: item[1],
                reverse=True,
            )[:5]
            rows.append({
                'split': split,
                'branch': branch_key,
                'branch_name': branch_name,
                'vocab_size': vocab_size,
                'token_count': int(counts.sum()),
                'unique_codes': int(nonzero.shape[0]),
                'utilization': float(nonzero.shape[0] / max(vocab_size, 1)),
                'entropy_bits': _entropy(probs),
                'top_codes': ';'.join(f'{code}:{count}' for code, count in top_codes),
            })
    return rows, distributions, vocab_sizes


def _build_transition_rows(
    transitions: Mapping[str, Mapping[str, np.ndarray]],
) -> List[Dict[str, Any]]:
    rows = []
    for split, branch_matrices in transitions.items():
        for branch_key, matrix in branch_matrices.items():
            total = int(matrix.sum())
            nonzero = np.argwhere(matrix > 0)
            top_pairs = sorted(
                ((int(start), int(end), int(matrix[start, end])) for start, end in nonzero),
                key=lambda item: item[2],
                reverse=True,
            )[:10]
            rows.append({
                'split': split,
                'branch': branch_key,
                'transition_count': total,
                'unique_transitions': int(nonzero.shape[0]),
                'top_transitions': ';'.join(f'{start}->{end}:{count}' for start, end, count in top_pairs),
            })
    return rows


def _build_divergence_rows(
    split_distributions: Mapping[str, Mapping[str, np.ndarray]],
) -> List[Dict[str, Any]]:
    rows = []
    pairs = (
        ('eeg_vs_fnirs_source', 'eeg_source_tokens', 'fnirs_source_tokens'),
        ('eeg_source_vs_observation', 'eeg_source_tokens', 'eeg_observation_tokens'),
        ('fnirs_source_vs_observation', 'fnirs_source_tokens', 'fnirs_observation_tokens'),
    )
    for split, distributions in split_distributions.items():
        for pair_name, left_key, right_key in pairs:
            rows.append({
                'split': split,
                'pair': pair_name,
                'left_branch': left_key,
                'right_branch': right_key,
                'js_divergence_bits': _js_divergence(distributions[left_key], distributions[right_key]),
            })
    return rows


def _build_split_rows(split_arrays: Mapping[str, Mapping[str, np.ndarray]]) -> List[Dict[str, Any]]:
    rows = []
    for split, data in split_arrays.items():
        tasks = Counter(_safe_text_array(data.get('source_task', np.asarray([]))).tolist())
        subjects = Counter(np.asarray(data.get('subject_id', np.asarray([]))).astype(int).tolist())
        rows.append({
            'split': split,
            'samples': int(data['eeg_source_tokens'].shape[0]),
            'tasks': ';'.join(f'{key}:{value}' for key, value in sorted(tasks.items())),
            'subjects': ';'.join(f'{key}:{value}' for key, value in sorted(subjects.items())),
        })
    return rows


def analyze_source_observation_token_sequences(
    run_dir: str | Path,
    *,
    output_dir: Optional[str | Path] = None,
    splits: Optional[Iterable[str]] = None,
    max_heatmap_samples: int = 160,
    dpi: int = 160,
) -> Dict[str, Any]:
    """Analyze exported source/observation token sequence datasets."""
    run_path = Path(run_dir)
    analysis_dir = Path(output_dir) if output_dir is not None else run_path / 'analysis' / 'token_sequence'
    figure_dir = analysis_dir / 'figures'
    table_dir = analysis_dir / 'tables'
    split_names = _available_splits(run_path, splits)
    split_arrays = {split: _load_split_npz(run_path, split) for split in split_names}
    export_manifest = _load_json_optional(run_path / 'manifest.json')

    usage_rows, split_distributions, vocab_sizes = _build_usage_rows(split_arrays, export_manifest)
    transitions = {
        split: {
            branch_key: _transition_matrix(data[branch_key], vocab_sizes[branch_key])
            for branch_key, _ in BRANCHES
        }
        for split, data in split_arrays.items()
    }
    transition_rows = _build_transition_rows(transitions)
    divergence_rows = _build_divergence_rows(split_distributions)
    split_rows = _build_split_rows(split_arrays)
    features, feature_metadata = _build_sequence_features(split_arrays, vocab_sizes)

    table_paths = {
        'code_usage_summary': table_dir / 'code_usage_summary.csv',
        'transition_summary': table_dir / 'transition_summary.csv',
        'split_task_subject_counts': table_dir / 'split_task_subject_counts.csv',
        'divergence_summary': table_dir / 'divergence_summary.csv',
    }
    _write_csv(
        table_paths['code_usage_summary'],
        usage_rows,
        ['split', 'branch', 'branch_name', 'vocab_size', 'token_count', 'unique_codes', 'utilization', 'entropy_bits', 'top_codes'],
    )
    _write_csv(
        table_paths['transition_summary'],
        transition_rows,
        ['split', 'branch', 'transition_count', 'unique_transitions', 'top_transitions'],
    )
    _write_csv(
        table_paths['split_task_subject_counts'],
        split_rows,
        ['split', 'samples', 'tasks', 'subjects'],
    )
    _write_csv(
        table_paths['divergence_summary'],
        divergence_rows,
        ['split', 'pair', 'left_branch', 'right_branch', 'js_divergence_bits'],
    )

    figure_paths: Dict[str, str] = {}
    if HAS_MATPLOTLIB:
        figure_paths['token_heatmaps_by_split'] = _plot_token_heatmaps(
            split_arrays,
            vocab_sizes,
            figure_dir,
            max_samples=int(max_heatmap_samples),
            dpi=dpi,
        )
        figure_paths['code_usage_histograms'] = _plot_code_usage(split_distributions, figure_dir, dpi=dpi)
        figure_paths['transition_matrices'] = _plot_transition_matrices(transitions, figure_dir, dpi=dpi)
        figure_paths['cross_modal_alignment'], alignment_rows = _plot_cross_modal_alignment(
            split_arrays,
            vocab_sizes,
            figure_dir,
            dpi=dpi,
        )
        figure_paths['source_observation_divergence'] = _plot_divergence(divergence_rows, figure_dir, dpi=dpi)
        figure_paths['task_subject_pca'], pca_variance_ratio = _plot_pca(features, feature_metadata, figure_dir, dpi=dpi)
    else:
        alignment_rows = []
        pca_variance_ratio = [0.0, 0.0]

    alignment_table = table_dir / 'cross_modal_alignment_summary.csv'
    _write_csv(
        alignment_table,
        alignment_rows,
        ['split', 'lag_tokens', 'lag_seconds', 'mutual_information_bits', 'pair_count'],
    )
    table_paths['cross_modal_alignment_summary'] = alignment_table

    analysis_manifest = {
        'schema_version': TOKEN_SEQUENCE_SCHEMA_VERSION,
        'created_at': datetime.now().isoformat(),
        'run_dir': str(run_path),
        'analysis_dir': str(analysis_dir),
        'splits': split_rows,
        'vocab_sizes': {key: int(value) for key, value in vocab_sizes.items()},
        'figures': {key: str(value) for key, value in figure_paths.items() if value},
        'tables': {key: str(value) for key, value in table_paths.items()},
        'pca_variance_ratio': pca_variance_ratio,
        'matplotlib_available': HAS_MATPLOTLIB,
    }
    _write_json(analysis_dir / 'manifest.json', analysis_manifest)
    return analysis_manifest
