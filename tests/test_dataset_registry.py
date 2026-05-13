import unittest
from unittest.mock import patch

import numpy as np

from src.data.factory import create_configured_multimodal_dataloaders
from src.data.registry import load_experiment_config, normalize_data_config
from src.data.simultaneous_eeg_nirs_dataset import classify_alignment_pattern, detect_offset_blocks
from src.data.validation import build_dataset_validation_plan


class DatasetRegistryTests(unittest.TestCase):
    def test_infers_single_trial_from_root(self):
        normalized = normalize_data_config({
            'data_root': 'data/EEG+NIRS Single-Trial',
            'modality': 'eeg',
        })
        self.assertEqual(normalized['dataset'], 'eeg_fnirs_single_trial')
        self.assertEqual(normalized['dataset_registry']['sync_strategy'], 'shared_parallel_port_markers')

    def test_load_experiment_config_normalizes_shared_config(self):
        config = load_experiment_config('source_observation/phase1/default.yaml')
        self.assertEqual(config['data']['dataset'], 'eeg_fnirs_single_trial')
        self.assertEqual(config['data']['data_root'], 'data/EEG+NIRS Single-Trial')
        self.assertIn('dataset_registry', config['data'])

    def test_load_experiment_config_exposes_modality_specific_preprocessing(self):
        config = load_experiment_config('source_observation/phase1/default.yaml')
        self.assertEqual(config['data']['eeg_preprocessing']['bandpass'], [0.5, 45])
        self.assertEqual(config['data']['fnirs_preprocessing']['lowpass'], 0.1)

    def test_multimodal_factory_projects_legacy_preprocessing_by_modality(self):
        config = {
            'data': {
                'dataset': 'eeg_fnirs_single_trial',
                'data_root': 'data/EEG+NIRS Single-Trial',
                'task': 'motor_imagery',
                'window': {'duration_s': 10.0, 'offset_ms': 0.0},
                'split': {
                    'train_subjects': [1],
                    'val_subjects': [2],
                    'test_subjects': [3],
                },
                'preprocessing': {
                    'bandpass': [0.5, 45],
                    'lowpass': 0.1,
                    'resample_rate': 200,
                },
                'exclude_eog': True,
                'hbo_only': True,
                'hbr_only': False,
                'num_workers': 0,
            },
            'training': {'batch_size': 2},
        }

        with patch('src.data.factory.create_single_trial_dataloaders', return_value={'train': None, 'val': None, 'test': None}) as mocked:
            create_configured_multimodal_dataloaders(config)

        kwargs = mocked.call_args.kwargs
        self.assertEqual(kwargs['eeg_preprocessing']['bandpass'], [0.5, 45])
        self.assertEqual(kwargs['fnirs_preprocessing']['lowpass'], 0.1)
        self.assertEqual(kwargs['fnirs_preprocessing']['resample_rate'], 200)
        self.assertNotIn('bandpass', kwargs['fnirs_preprocessing'])

    def test_load_experiment_config_resolves_downstream_base(self):
        config = load_experiment_config('downstream/mi_multimodal_token.yaml')
        self.assertEqual(config['data']['dataset'], 'eeg_fnirs_single_trial')
        self.assertEqual(config['data']['data_root'], 'data/EEG+NIRS Single-Trial')
        self.assertEqual(config['data']['task'], 'motor_imagery')

    def test_normalize_data_config_accepts_multi_source(self):
        normalized = normalize_data_config({
            'modality': 'both',
            'sources': [
                {
                    'dataset': 'eeg_fnirs_single_trial',
                    'data_root': 'data/EEG+NIRS Single-Trial',
                    'task': 'motor_imagery',
                },
                {
                    'dataset': 'simultaneous_eeg_nirs',
                    'data_root': 'data/Simultaneous EEG&NIRS',
                    'task': 'nback',
                },
            ],
        })
        self.assertEqual(normalized['dataset'], 'multi_source')
        self.assertEqual(normalized['dataset_registry']['sync_strategy'], 'source_defined')

    def test_refed_plan_uses_annotation_alignment(self):
        plan = build_dataset_validation_plan('refed')
        self.assertEqual(plan['sync_strategy'], 'continuous_annotation_alignment')
        check_ids = {check['check_id'] for check in plan['checks']}
        self.assertIn('record-duration-alignment', check_ids)
        self.assertIn('annotation-resample-check', check_ids)
        self.assertIn('global-visual-alignment', check_ids)
        self.assertIn('local-visual-alignment', check_ids)

    def test_visual_plan_uses_reconstruction_checks(self):
        plan = build_dataset_validation_plan('visual_cognitive_motivation')
        check_ids = {check['check_id'] for check in plan['checks']}
        self.assertIn('cross-device-event-reconstruction', check_ids)
        self.assertIn('label-join-consistency', check_ids)

    def test_classifies_stable_fixed_offset_pattern(self):
        residual_ms = np.asarray([1000.0, 1015.0, 995.0, 1005.0])
        blocks = detect_offset_blocks(residual_ms, jump_threshold_ms=20_000.0)
        pattern = classify_alignment_pattern(residual_ms, blocks)
        self.assertEqual(pattern['case'], 'stable_fixed_offset')

    def test_classifies_piecewise_constant_offset_pattern(self):
        residual_ms = np.asarray([1000.0, 1010.0, 995.0, 52000.0, 52015.0, 51990.0])
        blocks = detect_offset_blocks(residual_ms, jump_threshold_ms=20_000.0)
        pattern = classify_alignment_pattern(residual_ms, blocks)
        self.assertEqual(pattern['case'], 'piecewise_constant_offset')
        self.assertEqual(pattern['num_blocks'], 2)


if __name__ == '__main__':
    unittest.main()