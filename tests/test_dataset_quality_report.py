"""Unit tests for dataset_quality_report module."""

from __future__ import annotations

import base64
import json
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np

from src.data.dataset_quality_report import (
    IMPLEMENTED_DATASET_IDS,
    PLANNED_DATASET_IDS,
    ChannelAmplitudeStats,
    CrossDatasetComparison,
    DatasetQualityReporter,
    DatasetQualitySnapshot,
    _compute_amplitude_stats,
    _figure_to_base64,
    _figure_to_base64_from_fig,
    _human_size,
    _status_badge,
    _unit_family_label,
)
from src.data.fnirs_standardization import FNIRSMeasurementContract


class TestAmplitudeStats(unittest.TestCase):
    """Tests for _compute_amplitude_stats."""

    def test_sine_wave_stats(self):
        """Amplitude stats on a pure sine wave should have near-zero mean and known std."""
        t = np.linspace(0, 10, 1000, endpoint=False)
        signal = np.sin(2 * np.pi * 1.0 * t)[None, :]  # 1 channel, 1000 samples
        stats = _compute_amplitude_stats(signal)

        global_stats = stats.global_
        self.assertAlmostEqual(global_stats['mean'], 0.0, delta=0.01)
        self.assertAlmostEqual(global_stats['std'], np.sqrt(0.5), delta=0.05)
        self.assertAlmostEqual(global_stats['min'], -1.0, delta=0.05)
        self.assertAlmostEqual(global_stats['max'], 1.0, delta=0.05)

    def test_dc_signal(self):
        """A constant DC signal should have zero std."""
        signal = np.ones((3, 200), dtype=np.float64) * 5.0
        stats = _compute_amplitude_stats(signal)

        self.assertAlmostEqual(stats.global_['mean'], 5.0)
        self.assertAlmostEqual(stats.global_['std'], 0.0)
        self.assertEqual(stats.global_['min'], 5.0)
        self.assertEqual(stats.global_['max'], 5.0)

    def test_multi_channel(self):
        """Multi-channel stats should have per-channel entries."""
        rng = np.random.RandomState(42)
        signal = rng.randn(4, 500).astype(np.float64)
        stats = _compute_amplitude_stats(signal)

        self.assertEqual(len(stats.per_channel), 4)
        self.assertIn('ch_0', stats.per_channel)
        self.assertIn('ch_3', stats.per_channel)
        for ch_stats in stats.per_channel.values():
            for key in ('min', 'max', 'mean', 'std', 'median', 'skew', 'kurtosis'):
                self.assertIn(key, ch_stats)

    def test_transposed_input(self):
        """Stats should work with [time, channels] input (auto-detect)."""
        rng = np.random.RandomState(42)
        signal_time_first = rng.randn(1000, 2).astype(np.float64)
        stats = _compute_amplitude_stats(signal_time_first)
        self.assertEqual(len(stats.per_channel), 2)

    def test_with_nans(self):
        """Non-finite values should be filtered out."""
        signal = np.array([[1.0, 2.0, np.nan, 4.0, np.inf, -np.inf, 5.0]], dtype=np.float64)
        stats = _compute_amplitude_stats(signal)
        self.assertAlmostEqual(stats.global_['min'], 1.0)
        self.assertAlmostEqual(stats.global_['max'], 5.0)
        self.assertAlmostEqual(stats.global_['mean'], 3.0)


class TestHelpers(unittest.TestCase):
    """Tests for module-level helper functions."""

    def test_unit_family_label(self):
        contract = FNIRSMeasurementContract(
            dataset_id='test',
            signal_key='wavelength_pair',
            measurement_family='optical_intensity',
            native_unit='V',
            channel_roles=('lowWL', 'highWL'),
        )
        label = _unit_family_label(contract)
        self.assertIn('Optical Intensity', label)
        self.assertIn('V', label)

    def test_unit_family_label_absorbance(self):
        contract = FNIRSMeasurementContract(
            dataset_id='test2',
            signal_key='absorbance',
            measurement_family='absorbance',
            native_unit='unreported_absorbance',
            channel_roles=('Abs780',),
        )
        label = _unit_family_label(contract)
        self.assertIn('Absorbance', label)

    def test_status_badge_implemented(self):
        badge = _status_badge('implemented')
        self.assertIn('Implemented', badge)
        self.assertIn('2E8B57', badge)

    def test_status_badge_planned(self):
        badge = _status_badge('planned')
        self.assertIn('Planned', badge)
        self.assertIn('D97706', badge)

    def test_figure_to_base64(self):
        with TemporaryDirectory() as tmp:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            path = Path(tmp) / 'test.png'
            fig.savefig(path, dpi=72)
            plt.close(fig)

            encoded = _figure_to_base64(path)
            self.assertTrue(encoded.startswith('data:image/png;base64,'))
            # Decode and verify it's a valid PNG
            b64_part = encoded.split(',', 1)[1]
            decoded = base64.b64decode(b64_part)
            self.assertTrue(decoded[:8], b'\x89PNG\r\n\x1a\n')  # PNG header

    def test_figure_to_base64_from_fig(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        encoded = _figure_to_base64_from_fig(fig)
        self.assertTrue(encoded.startswith('data:image/png;base64,'))

    def test_human_size(self):
        self.assertEqual(_human_size(0), '0.0 B')
        self.assertEqual(_human_size(512), '512.0 B')
        self.assertIn('KB', _human_size(2048))
        self.assertIn('MB', _human_size(5 * 1024 * 1024))


class TestCrossDatasetComparison(unittest.TestCase):
    """Tests for CrossDatasetComparison construction."""

    def test_empty_tables(self):
        comp = CrossDatasetComparison()
        self.assertEqual(comp.unit_family_table, [])
        self.assertEqual(comp.sampling_rate_table, [])
        self.assertEqual(comp.channel_count_table, [])

    def test_populated_tables(self):
        snap = DatasetQualitySnapshot(
            dataset_id='test_ds',
            display_name='Test Dataset',
            loader_status='implemented',
            sync_strategy='test_sync',
            eeg_sample_rate_hz=200.0,
            fnirs_sample_rate_hz=10.0,
            eeg_channels=30,
            fnirs_channels=36,
        )
        reporter = DatasetQualityReporter(output_dir=Path('/tmp/dummy'))
        comp = reporter.compute_cross_dataset_comparisons([snap])

        self.assertEqual(len(comp.sampling_rate_table), 1)
        self.assertEqual(comp.sampling_rate_table[0]['eeg_sample_rate_hz'], 200.0)
        self.assertEqual(comp.sampling_rate_table[0]['ratio'], 20.0)

        self.assertEqual(len(comp.channel_count_table), 1)
        self.assertEqual(comp.channel_count_table[0]['eeg_channels'], 30)


class TestSnapshotDataclass(unittest.TestCase):
    """Tests for DatasetQualitySnapshot."""

    def test_minimal_snapshot(self):
        snap = DatasetQualitySnapshot(
            dataset_id='test_ds',
            display_name='Test',
            loader_status='planned',
            sync_strategy='none',
        )
        d = snap.to_dict()
        self.assertEqual(d['dataset_id'], 'test_ds')
        self.assertEqual(d['loader_status'], 'planned')

    def test_full_snapshot_to_dict(self):
        snap = DatasetQualitySnapshot(
            dataset_id='test_ds',
            display_name='Test',
            loader_status='implemented',
            sync_strategy='markers',
            eeg_sample_rate_hz=200.0,
            fnirs_sample_rate_hz=10.0,
            eeg_channels=30,
            fnirs_channels=36,
            eeg_amplitude_stats={'global': {'mean': 0.1, 'std': 1.0}},
            issues=['test issue'],
        )
        d = snap.to_dict()
        self.assertEqual(d['eeg_sample_rate_hz'], 200.0)
        self.assertEqual(d['issues'], ['test issue'])
        self.assertEqual(d['eeg_amplitude_stats']['global']['mean'], 0.1)

    def test_croce_cache_is_not_counted_as_dataset(self):
        self.assertNotIn('croce_local_cache', IMPLEMENTED_DATASET_IDS)

    def test_all_four_raw_datasets_use_unified_loader(self):
        self.assertEqual(PLANNED_DATASET_IDS, set())
        self.assertIn('refed', IMPLEMENTED_DATASET_IDS)
        self.assertIn('visual_cognitive_motivation', IMPLEMENTED_DATASET_IDS)


class TestHTMLOutput(unittest.TestCase):
    """Tests for HTML report generation."""

    def setUp(self):
        self.tmp_dir = TemporaryDirectory()
        self.output_dir = Path(self.tmp_dir.name)
        self.figures_dir = self.output_dir / 'figures'
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _make_dummy_figure(self, path: Path) -> None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'test', ha='center')
        fig.savefig(path, dpi=72)
        plt.close(fig)

    def test_html_report_generates_valid_file(self):
        snap = DatasetQualitySnapshot(
            dataset_id='test_ds',
            display_name='Test Dataset',
            loader_status='implemented',
            sync_strategy='test_sync',
            eeg_sample_rate_hz=200.0,
            fnirs_sample_rate_hz=10.0,
            eeg_channels=30,
            fnirs_channels=36,
            eeg_amplitude_stats={'global': {'min': -1.0, 'max': 1.0, 'mean': 0.0, 'std': 0.5,
                                             'median': 0.0, 'skew': 0.0, 'kurtosis': 0.0}},
        )
        comp = CrossDatasetComparison()
        reporter = DatasetQualityReporter(output_dir=self.output_dir)

        html_path = reporter.build_html_report([snap], comp)
        self.assertTrue(html_path.exists())
        content = html_path.read_text(encoding='utf-8')
        self.assertIn('<!DOCTYPE html>', content)
        self.assertIn('<html', content)
        self.assertIn('Test Dataset', content)

    def test_html_report_handles_planned_dataset(self):
        snap = DatasetQualitySnapshot(
            dataset_id='refed',
            display_name='REFED',
            loader_status='planned',
            sync_strategy='continuous_annotation_alignment',
            issues=['Loader not yet implemented.'],
        )
        comp = CrossDatasetComparison()
        reporter = DatasetQualityReporter(output_dir=self.output_dir)

        html_path = reporter.build_html_report([snap], comp)
        content = html_path.read_text(encoding='utf-8')
        self.assertIn('Planned', content)
        self.assertIn('REFED', content)

    def test_html_report_with_embedded_figure(self):
        ds_dir = self.figures_dir / 'test_ds'
        ds_dir.mkdir(parents=True, exist_ok=True)
        fig_path = ds_dir / 'waveform_eeg.png'
        self._make_dummy_figure(fig_path)

        snap = DatasetQualitySnapshot(
            dataset_id='test_ds',
            display_name='Test',
            loader_status='implemented',
            sync_strategy='test',
            waveform_figures={'waveform_eeg': fig_path},
            fnirs_contracts={
                'wavelength_pair': {
                    'measurement_family': 'optical_intensity',
                    'native_unit': 'V',
                    'channel_roles': ['lowWL', 'highWL'],
                    'canonical_semantics': 'dimensionless',
                },
            },
        )
        comp = CrossDatasetComparison()
        reporter = DatasetQualityReporter(output_dir=self.output_dir, embed_images=True)

        html_path = reporter.build_html_report([snap], comp)
        content = html_path.read_text(encoding='utf-8')
        self.assertIn('data:image/png;base64,', content)

    def test_markdown_summary(self):
        snap = DatasetQualitySnapshot(
            dataset_id='test_ds',
            display_name='Test',
            loader_status='implemented',
            sync_strategy='test',
            eeg_channels=30,
            fnirs_channels=36,
            eeg_sample_rate_hz=200.0,
            fnirs_sample_rate_hz=10.0,
            native_units={'wavelength_pair': 'V'},
            eeg_amplitude_stats={'global': {'min': -1, 'max': 1, 'mean': 0, 'std': 0.5}},
            issues=['test warning'],
        )
        comp = CrossDatasetComparison()
        reporter = DatasetQualityReporter(output_dir=self.output_dir)

        md_path = reporter.build_markdown_summary([snap], comp)
        self.assertTrue(md_path.exists())
        content = md_path.read_text(encoding='utf-8')
        self.assertIn('# Dataset Quality Audit Report', content)
        self.assertIn('Test', content)
        self.assertIn('30 channels', content)
        self.assertIn('test warning', content)

    def test_error_snapshot(self):
        reporter = DatasetQualityReporter(output_dir=self.output_dir)
        snap = reporter._error_snapshot('nonexistent_ds', 'test error')
        self.assertEqual(snap.dataset_id, 'nonexistent_ds')
        self.assertIn('test error', snap.issues[0])


if __name__ == '__main__':
    unittest.main()
