#!/usr/bin/env python

import argparse
import json
import sys
from pathlib import Path

import yaml

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.scripts.train_source_observation_tokenizer import create_multimodal_dataloaders, setup_device
from src.tokenizers import create_tokenizer
from src.utils import load_checkpoint_file
from src.visualization import analyze_alignment


def main():
    parser = argparse.ArgumentParser(description='Re-run standardized alignment analysis for a completed multimodal tokenizer run')
    parser.add_argument('--run-dir', required=True, help='Run directory containing config.yaml and checkpoints')
    parser.add_argument('--checkpoint', default='best_model.pt', help='Checkpoint filename inside run_dir/checkpoints')
    parser.add_argument('--output-dir', default=None, help='Optional output directory for analysis results')
    parser.add_argument('--splits', nargs='+', default=['val', 'test'], help='Splits to analyze')
    parser.add_argument('--device', default=None, help='Optional device override, e.g. cpu or cuda')
    parser.add_argument(
        '--analysis-type',
        choices=['source_observation_alignment'],
        default='source_observation_alignment',
        help='Alignment rerun now only supports the source/observation scorecard path.',
    )
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
    analysis_type = args.analysis_type or getattr(model, 'get_analysis_type', lambda: 'source_observation_alignment')()
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / 'analysis'
    results = analyze_alignment(
        model=model,
        dataloaders=dataloaders,
        config=config,
        output_dir=output_dir,
        device=device,
        splits=args.splits,
        analysis_type=analysis_type,
    )
    print(
        json.dumps(
            {
                'output_dir': str(results.get('artifact_root', output_dir)),
                'analysis_type': analysis_type,
                'splits': list(results.get('splits', {}).keys()),
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()