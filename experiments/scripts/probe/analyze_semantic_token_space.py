#!/usr/bin/env python

import argparse
import json
import sys
from pathlib import Path

import yaml

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from experiments.scripts.train_source_observation_tokenizer import create_multimodal_dataloaders, setup_device
from src.tokenizers import create_tokenizer
from src.utils import load_checkpoint_file
from src.visualization import generate_source_observation_scorecard


def main():
    parser = argparse.ArgumentParser(description='Re-run Gate 4 related scorecard metrics for a completed tokenizer run')
    parser.add_argument('--run-dir', required=True, help='Run directory containing config.yaml and checkpoints')
    parser.add_argument('--checkpoint', default='best_model.pt', help='Checkpoint filename inside run_dir/checkpoints')
    parser.add_argument('--output-dir', default=None, help='Optional output directory for semantic analysis results')
    parser.add_argument('--splits', nargs='+', default=['val', 'test'], help='Splits to analyze')
    parser.add_argument('--device', default=None, help='Optional device override, e.g. cpu or cuda')
    parser.add_argument('--max-batches', type=int, default=None, help='Optional limit for quick smoke analysis')
    parser.add_argument('--max-feature-samples', type=int, default=20000, help='Maximum latent-token pairs per branch for state-quality metrics')
    parser.add_argument('--max-probe-samples', type=int, default=None, help='Optional cap for per-sample probe features used by P2 metrics')
    parser.add_argument('--augmentation-probe-batches', type=int, default=None, help='Optional cap for augmentation-consistency probe batches')
    parser.add_argument('--probe-seed', type=int, default=None, help='Optional random seed override for lightweight probes')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config_path = run_dir / 'config.yaml'
    checkpoint_path = run_dir / 'checkpoints' / args.checkpoint
    if not config_path.exists():
        raise FileNotFoundError(f'Config not found: {config_path}')
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    config = yaml.safe_load(config_path.read_text())
    if args.device is not None:
        config.setdefault('experiment', {})['device'] = args.device
    device = setup_device(config)

    model = create_tokenizer(config).to(device)
    checkpoint = load_checkpoint_file(checkpoint_path, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataloaders = create_multimodal_dataloaders(config)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / 'analysis'
    results = generate_source_observation_scorecard(
        model=model,
        dataloaders=dataloaders,
        config=config,
        output_dir=output_dir,
        device=device,
        splits=args.splits,
        run_dir=run_dir,
    )
    gate4 = results.get('gates', {}).get('gate4', {})
    print(
        json.dumps(
            {
                'output_dir': str(results.get('artifact_root', output_dir)),
                'primary_split': results.get('primary_split'),
                'gate4_status': gate4.get('status'),
                'splits': list(results.get('splits', {}).keys()),
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()