#!/usr/bin/env python
"""
Generate comprehensive plots from existing UMAP experiment runs.

Usage:
    # Generate plots for all existing finetune runs (from saved history/results)
    python generate_plots.py retroactive

    # Generate comparison across all finetune runs
    python generate_plots.py compare

    # Both
    python generate_plots.py all
"""

import sys
import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import umap_plots


RUNS_DIR = SCRIPT_DIR / 'runs'


def retroactive_pretrain(run_dir: Path):
    """Generate history-based pretrain plots from a completed run."""
    plot_dir = run_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)

    history_path = run_dir / 'history.json'
    config_path = run_dir / 'config.json'

    if not history_path.exists():
        print(f"  skip (no history.json)")
        return

    with open(history_path) as f:
        history = json.load(f)

    # Check if this is a pretrain run (has loss_total)
    if 'loss_total' not in history.get('train', {}):
        return

    print(f"  → pretrain loss decomposition")
    umap_plots.plot_pretrain_loss_decomposition(history, plot_dir)
    print(f"  → pretrain loss correlation")
    umap_plots.plot_pretrain_loss_correlation(history, plot_dir)

    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"  → LR schedule")
        umap_plots.plot_lr_schedule(cfg, plot_dir)


def retroactive_finetune(run_dir: Path):
    """Generate history-based finetune plots from a completed run."""
    plot_dir = run_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)

    history_path = run_dir / 'history.json'
    results_path = run_dir / 'results.json'

    if not history_path.exists():
        print(f"  skip (no history.json)")
        return

    with open(history_path) as f:
        history = json.load(f)

    # Check if this is a finetune run (has 'accuracy')
    if 'accuracy' not in history.get('train', {}):
        return

    results = {}
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

    class_names = ['Left MI', 'Right MI']
    title = run_dir.name
    subj_accs = results.get('subject_accs', {})

    # We can reconstruct preds/labels from the classification report
    # but we don't have them directly saved. Generate what we can.
    print(f"  → classification dashboard (from history)")
    umap_plots.plot_classification_dashboard(
        history, np.array([]), np.array([]), subj_accs, class_names, plot_dir, title
    )


def compare_all_finetune():
    """Generate comparison plots across all finetune v2 runs."""
    results = {}
    for run_dir in sorted(RUNS_DIR.iterdir()):
        results_path = run_dir / 'results.json'
        if not results_path.exists():
            continue
        with open(results_path) as f:
            r = json.load(f)
        # Only include finetune runs with test accuracy
        if 'test_accuracy' not in r:
            continue
        # Skip non-v2 pretrained runs from batch 2 (they had the bug)
        name = run_dir.name
        if name.startswith('U2-') and 'pt' in name.lower() and 'v2' not in name.lower():
            continue
        results[name] = r

    if len(results) < 2:
        print(f"Need at least 2 finetune runs for comparison, found {len(results)}")
        return

    compare_dir = RUNS_DIR / '_comparison'
    compare_dir.mkdir(exist_ok=True)

    print(f"\nGenerating comparison plots for {len(results)} runs: {list(results.keys())}")
    umap_plots.plot_experiment_comparison(results, compare_dir)
    print("  → experiment_comparison.png")
    umap_plots.plot_modality_robustness(results, compare_dir)
    print("  → modality_robustness.png")


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('retroactive', 'compare', 'all'):
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1]

    if mode in ('retroactive', 'all'):
        print("=== Retroactive plot generation ===")
        for run_dir in sorted(RUNS_DIR.iterdir()):
            if not run_dir.is_dir() or run_dir.name.startswith('_'):
                continue
            print(f"\n[{run_dir.name}]")
            try:
                retroactive_pretrain(run_dir)
                retroactive_finetune(run_dir)
            except Exception as e:
                print(f"  ERROR: {e}")

    if mode in ('compare', 'all'):
        print("\n=== Comparison plots ===")
        compare_all_finetune()

    print("\nDone.")
