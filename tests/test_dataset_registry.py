import unittest

from src.data.registry import load_experiment_config, normalize_data_config
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
        config = load_experiment_config('phase0plus/shared_labram_lag_warmstart_eeg_fnirs_30s_2s_cb512.yaml')
        self.assertEqual(config['data']['dataset'], 'eeg_fnirs_single_trial')
        self.assertEqual(config['data']['data_root'], 'data/EEG+NIRS Single-Trial')
        self.assertIn('dataset_registry', config['data'])

    def test_load_experiment_config_resolves_downstream_base(self):
        config = load_experiment_config('downstream/mi_multimodal_token.yaml')
        self.assertEqual(config['data']['dataset'], 'eeg_fnirs_single_trial')
        self.assertEqual(config['data']['data_root'], 'data/EEG+NIRS Single-Trial')
        self.assertEqual(config['data']['task'], 'motor_imagery')

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


if __name__ == '__main__':
    unittest.main()