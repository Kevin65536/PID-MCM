import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.utils.run_metrics_comparison import collect_run_summaries, prepare_report_directory, resolve_run_dirs, write_report_bundle


class RunMetricsComparisonTests(unittest.TestCase):
    def test_collect_run_summaries_extracts_gate_and_baseline_deltas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_root = Path(tmpdir) / 'runs'
            baseline_dir = self._create_run(
                runs_root / 'baseline_run',
                best_epoch=2,
                best_val_loss=3.20,
                last_val_loss=3.35,
                source_k=32,
                gate1_metrics={
                    'eeg_source': self._codebook(32, 16, 10.8, 0.58, 10),
                    'fnirs_source': self._codebook(32, 11, 8.4, 0.73, 14),
                    'eeg_observation': self._codebook(32, 14, 11.0, 0.54, 9),
                    'fnirs_observation': self._codebook(32, 13, 10.4, 0.57, 10),
                },
            )
            candidate_dir = self._create_run(
                runs_root / 'candidate_run',
                best_epoch=3,
                best_val_loss=3.05,
                last_val_loss=3.11,
                source_k=24,
                gate1_metrics={
                    'eeg_source': self._codebook(24, 12, 8.6, 0.61, 7),
                    'fnirs_source': self._codebook(24, 8, 5.1, 0.88, 16),
                    'eeg_observation': self._codebook(32, 14, 10.7, 0.55, 8),
                    'fnirs_observation': self._codebook(32, 15, 11.8, 0.52, 7),
                },
            )

            run_dirs = resolve_run_dirs(runs_root, run_names=['baseline_run', 'candidate_run'])
            self.assertEqual(run_dirs, [baseline_dir.resolve(), candidate_dir.resolve()])

            rows = collect_run_summaries(run_dirs, baseline='baseline_run')
            row_by_name = {row['run_name']: row for row in rows}

            candidate = row_by_name['candidate_run']
            self.assertEqual(candidate['source_codebook_size'], 24)
            self.assertEqual(candidate['observation_codebook_sizes'], '32/32')
            self.assertAlmostEqual(candidate['best_val_loss'], 3.05)
            self.assertAlmostEqual(candidate['last_val_loss'], 3.11)
            self.assertEqual(candidate['gate1_min_active_codebook'], 'fnirs_source')
            self.assertAlmostEqual(candidate['gate1_min_active_ratio'], 8 / 24)
            self.assertAlmostEqual(candidate['gate1_min_perplexity_ratio'], 5.1 / 24)
            self.assertAlmostEqual(candidate['gate1_max_top_5_coverage'], 0.88)
            self.assertIn('fnirs_source.top_5_coverage=0.8800', candidate['gate1_primary_bottleneck'])
            self.assertAlmostEqual(candidate['delta_best_val_loss'], -0.15)
            self.assertAlmostEqual(candidate['delta_gate1_max_top_5_coverage'], 0.15)
            self.assertIn(candidate['trajectory_pattern'], {'steady_improvement', 'late_instability', 'observation_starved'})
            self.assertIn(candidate['improvement_effectiveness'], {'effective', 'mostly_effective', 'mixed_tradeoff', 'limited_or_negative'})

    def test_write_report_bundle_creates_named_directory_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_root = Path(tmpdir) / 'runs'
            self._create_run(
                runs_root / 'baseline_run',
                best_epoch=2,
                best_val_loss=3.20,
                last_val_loss=3.35,
                source_k=32,
                gate1_metrics={
                    'eeg_source': self._codebook(32, 16, 10.8, 0.58, 10),
                    'fnirs_source': self._codebook(32, 11, 8.4, 0.73, 14),
                    'eeg_observation': self._codebook(32, 14, 11.0, 0.54, 9),
                    'fnirs_observation': self._codebook(32, 13, 10.4, 0.57, 10),
                },
            )
            self._create_run(
                runs_root / 'candidate_run',
                best_epoch=3,
                best_val_loss=3.05,
                last_val_loss=3.11,
                source_k=24,
                gate1_metrics={
                    'eeg_source': self._codebook(24, 12, 8.6, 0.61, 7),
                    'fnirs_source': self._codebook(24, 8, 5.1, 0.88, 16),
                    'eeg_observation': self._codebook(32, 14, 10.7, 0.55, 8),
                    'fnirs_observation': self._codebook(32, 15, 11.8, 0.52, 7),
                },
            )

            rows = collect_run_summaries(resolve_run_dirs(runs_root), baseline='baseline_run')
            report_dir = prepare_report_directory(Path(tmpdir) / 'comparison_reports', resolve_run_dirs(runs_root), report_name='Gate1 Health Iteration 05')
            bundle = write_report_bundle(
                report_dir,
                rows,
                columns=['run_name', 'best_val_loss', 'gate1_max_top_5_coverage'],
                metadata={'baseline': 'baseline_run', 'split': 'test', 'sort_by': 'best_val_loss'},
            )

            self.assertTrue(bundle['report_dir'].name.endswith('gate1_health_iteration_05'))
            self.assertTrue(bundle['summary_csv'].exists())
            self.assertTrue(bundle['summary_json'].exists())
            self.assertTrue(bundle['analysis_json'].exists())
            self.assertTrue(bundle['report_markdown'].exists())
            self.assertTrue(bundle['metadata_json'].exists())
            report_text = bundle['report_markdown'].read_text(encoding='utf-8')
            self.assertIn('## Pattern Analysis', report_text)
            self.assertIn('### Grouped Runs', report_text)
            self.assertIn('## Visualizations', report_text)
            self.assertIn('figures/best_val_loss_ranking.png', report_text)
            self.assertIn('figures/trajectory_patterns.png', report_text)
            self.assertTrue((bundle['report_dir'] / 'figures' / 'best_val_loss_ranking.png').exists())
            self.assertTrue((bundle['report_dir'] / 'figures' / 'gate1_health_overview.png').exists())
            self.assertTrue((bundle['report_dir'] / 'figures' / 'stability_overview.png').exists())
            self.assertTrue((bundle['report_dir'] / 'figures' / 'trajectory_patterns.png').exists())
            self.assertTrue((bundle['report_dir'] / 'figures' / 'branch_perplexity_trajectories.png').exists())

    def test_resolve_run_dirs_recurses_namespaced_runs_and_skips_archive_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_root = Path(tmpdir) / 'runs'
            top_level = self._create_run(
                runs_root / 'baseline_run',
                best_epoch=2,
                best_val_loss=3.20,
                last_val_loss=3.35,
                source_k=32,
                gate1_metrics={},
            )
            nested = self._create_run(
                runs_root / 'source_observation' / 'croce_local' / 'highwl_v1' / 'candidate_run',
                best_epoch=3,
                best_val_loss=3.05,
                last_val_loss=3.11,
                source_k=32,
                gate1_metrics={},
            )
            archived = self._create_run(
                runs_root / 'archive' / 'old_phase' / 'archived_run',
                best_epoch=1,
                best_val_loss=4.00,
                last_val_loss=4.00,
                source_k=32,
                gate1_metrics={},
            )

            run_dirs = resolve_run_dirs(runs_root)
            self.assertEqual(run_dirs, [top_level.resolve(), nested.resolve()])
            self.assertNotIn(archived.resolve(), run_dirs)

            pattern_dirs = resolve_run_dirs(runs_root, patterns=['candidate_*'])
            self.assertEqual(pattern_dirs, [nested.resolve()])

    def _create_run(
        self,
        run_dir: Path,
        *,
        best_epoch: int,
        best_val_loss: float,
        last_val_loss: float,
        source_k: int,
        gate1_metrics: dict[str, dict[str, object]],
    ) -> Path:
        (run_dir / 'analysis').mkdir(parents=True)

        metrics_payload = {
            'started_at': '2026-05-10T00:00:00',
            'completed_at': '2026-05-10T00:30:00',
            'epochs': [
                {
                    'epoch': 1,
                    'train_loss': 4.0,
                    'val_loss': 3.8,
                    'metrics': {
                        'val_utilization': 1.0,
                        'val_perplexity': 3.0,
                    },
                },
                {
                    'epoch': best_epoch,
                    'train_loss': 3.3,
                    'val_loss': best_val_loss,
                    'metrics': {
                        'val_utilization': 1.0,
                        'val_perplexity': 5.0,
                    },
                },
                {
                    'epoch': best_epoch + 1,
                    'train_loss': 3.4,
                    'val_loss': last_val_loss,
                    'metrics': {
                        'val_utilization': 0.25,
                        'val_perplexity': 4.0,
                    },
                },
            ],
            'final_metrics': {
                'best_epoch': best_epoch,
                'best_monitor': best_val_loss,
                'val_loss': best_val_loss,
                'val_perplexity': 5.0,
                'val_utilization': 1.0,
            },
        }
        (run_dir / 'metrics.json').write_text(json.dumps(metrics_payload, indent=2), encoding='utf-8')

        final_summary = {
            'promotion_verdict': 'blocked_gate1',
            'gate_verdicts': {
                'gate1': 'fail',
                'gate2': 'fail',
                'gate3': 'fail',
                'gate4': 'fail',
            },
            'coupling_lag_policy': 'all_valid_lags',
        }
        (run_dir / 'final_summary.json').write_text(json.dumps(final_summary, indent=2), encoding='utf-8')

        split_payload = {
            'gates': {
                'gate1': {
                    'status': 'fail',
                    'metrics': {
                        'codebooks': gate1_metrics,
                    },
                },
            },
        }
        (run_dir / 'analysis' / 'split_test.json').write_text(json.dumps(split_payload, indent=2), encoding='utf-8')

        config = {
            'experiment': {
                'name': run_dir.name,
                'description': f'description for {run_dir.name}',
            },
            'training': {
                'learning_rate': 0.0001,
            },
            'model': {
                'source': {'codebook_size': source_k},
                'eeg_observation': {'codebook_size': 32},
                'fnirs_observation': {'codebook_size': 32},
                'quantizer': {
                    'beta': 0.5,
                    'decay': 0.95,
                },
            },
            'loss': {
                'codebook': {'balance_weight': 0.08},
                'coupling': {'weight': 0.0},
            },
        }
        (run_dir / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        return run_dir

    def _codebook(self, codebook_size: int, active_codes: int, perplexity: float, top_5_coverage: float, dead_code_count: int) -> dict[str, object]:
        return {
            'codebook_size': codebook_size,
            'active_codes': active_codes,
            'active_code_ratio': active_codes / codebook_size,
            'dead_code_count': dead_code_count,
            'perplexity': perplexity,
            'top_5_coverage': top_5_coverage,
            'passes_thresholds': False,
        }


if __name__ == '__main__':
    unittest.main()
