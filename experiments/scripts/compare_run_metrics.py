#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.utils.run_metrics_comparison import (
    DEFAULT_TRACK_METRICS,
    collect_run_summaries,
    prepare_report_directory,
    render_markdown_table,
    resolve_run_dirs,
    sort_run_summaries,
    write_report_bundle,
    write_csv_report,
    write_json_report,
    write_markdown_report,
)


DEFAULT_TABLE_COLUMNS = [
    'run_name',
    'learning_rate',
    'source_codebook_size',
    'observation_codebook_sizes',
    'best_val_loss',
    'best_epoch',
    'gate1_status',
    'gate1_min_active_ratio',
    'gate1_min_perplexity_ratio',
    'gate1_max_top_5_coverage',
    'gate1_primary_bottleneck',
    'trajectory_pattern',
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Batch compare completed experiment runs by saved metrics and Gate summaries')
    parser.add_argument('--runs-root', default='experiments/runs', help='Directory containing run subdirectories')
    parser.add_argument('--run', dest='runs', action='append', default=[], help='Run name or absolute run directory path')
    parser.add_argument('--glob', dest='patterns', action='append', default=[], help='Glob pattern under runs-root, e.g. gate1_health_*')
    parser.add_argument('--split', default='test', help='Analysis split to read from analysis/split_<split>.json')
    parser.add_argument('--baseline', default=None, help='Optional baseline run name/path used to compute delta columns')
    parser.add_argument('--track', nargs='*', default=None, help='Additional metrics to capture from best/last epochs')
    parser.add_argument('--sort-by', default='best_val_loss', help='Summary field used for sorting')
    parser.add_argument('--descending', action='store_true', help='Sort in descending order')
    parser.add_argument('--limit', type=int, default=None, help='Optional number of rows to print after sorting')
    parser.add_argument('--table-columns', nargs='*', default=None, help='Override printed table columns')
    parser.add_argument('--report-root', default='experiments/comparison_reports', help='Root directory used to create a uniquely named report folder for each invocation')
    parser.add_argument('--report-name', default=None, help='Optional human-readable name used in the generated report directory name')
    parser.add_argument('--no-plots', action='store_true', help='Skip PNG visualization generation inside the report directory')
    parser.add_argument('--output-csv', default=None, help='Optional CSV output path')
    parser.add_argument('--output-json', default=None, help='Optional JSON output path')
    parser.add_argument('--output-md', default=None, help='Optional Markdown output path')
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    run_dirs = resolve_run_dirs(args.runs_root, run_names=args.runs, patterns=args.patterns)
    if not run_dirs:
        parser.error('No run directories matched the provided --run/--glob filters.')

    track_metrics = args.track if args.track else DEFAULT_TRACK_METRICS
    rows = collect_run_summaries(
        run_dirs,
        split=args.split,
        track_metrics=track_metrics,
        baseline=args.baseline,
    )
    rows = sort_run_summaries(rows, sort_by=args.sort_by, descending=args.descending)
    if args.limit is not None:
        rows = rows[:args.limit]

    table_columns = args.table_columns or list(DEFAULT_TABLE_COLUMNS)
    if args.baseline:
        table_columns.extend([
            'improvement_effectiveness',
            'delta_best_val_loss',
            'delta_gate1_min_active_ratio',
            'delta_gate1_max_top_5_coverage',
        ])

    report_dir = prepare_report_directory(
        args.report_root,
        run_dirs,
        report_name=args.report_name,
    )
    report_bundle = write_report_bundle(
        report_dir,
        rows,
        table_columns,
        metadata={
            'split': args.split,
            'baseline': args.baseline,
            'sort_by': args.sort_by,
            'descending': args.descending,
            'track_metrics': track_metrics,
            'selected_runs': [str(path) for path in run_dirs],
        },
        include_visualizations=not args.no_plots,
    )

    print(render_markdown_table(rows, table_columns))
    print(f'\nCompared runs: {len(rows)}')
    print(f'Report directory: {report_bundle["report_dir"]}')
    print(f'Analysis JSON: {report_bundle["analysis_json"]}')
    print(f'Markdown report: {report_bundle["report_markdown"]}')

    if report_bundle['figures']:
        print('Figures:')
        for figure_name, figure_path in report_bundle['figures'].items():
            print(f'  {figure_name}: {figure_path}')

    if args.output_csv:
        write_csv_report(args.output_csv, rows)
        print(f'CSV report: {args.output_csv}')
    if args.output_json:
        write_json_report(args.output_json, rows)
        print(f'JSON report: {args.output_json}')
    if args.output_md:
        write_markdown_report(args.output_md, rows, table_columns)
        print(f'Markdown report: {args.output_md}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())