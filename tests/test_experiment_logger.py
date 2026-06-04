import tempfile
import unittest
from pathlib import Path

import yaml

from src.utils.logger import ExperimentLogger


class ExperimentLoggerTests(unittest.TestCase):
    def test_run_group_places_run_under_namespaced_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            experiments_dir = Path(tmpdir) / 'experiments'
            config_dir = experiments_dir / 'configs'
            config_dir.mkdir(parents=True)
            config_path = config_dir / 'test.yaml'
            config_path.write_text(
                yaml.safe_dump(
                    {
                        'experiment': {
                            'name': 'test_exp',
                            'run_group': 'source_observation/croce_local/highwl_v1',
                        },
                        'data': {
                            'dataset': 'croce_local_cache',
                            'modality': 'both',
                        },
                    },
                    sort_keys=False,
                ),
                encoding='utf-8',
            )

            logger = ExperimentLogger(
                config_path='test.yaml',
                experiments_dir=str(experiments_dir),
                run_name='run_a',
            )

            expected = (
                experiments_dir
                / 'runs'
                / 'source_observation'
                / 'croce_local'
                / 'highwl_v1'
                / 'run_a'
            )
            self.assertEqual(logger.run_dir, expected)
            self.assertTrue((expected / 'config.yaml').exists())

    def test_run_group_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            experiments_dir = Path(tmpdir) / 'experiments'
            config_dir = experiments_dir / 'configs'
            config_dir.mkdir(parents=True)
            for filename, run_group in (
                ('bad_posix.yaml', '../outside'),
                ('bad_windows.yaml', '..\\outside'),
            ):
                config_path = config_dir / filename
                config_path.write_text(
                    yaml.safe_dump(
                        {
                            'experiment': {
                                'name': 'bad_exp',
                                'run_group': run_group,
                            },
                            'data': {
                                'dataset': 'croce_local_cache',
                                'modality': 'both',
                            },
                        },
                        sort_keys=False,
                    ),
                    encoding='utf-8',
                )

                with self.assertRaises(ValueError):
                    ExperimentLogger(
                        config_path=filename,
                        experiments_dir=str(experiments_dir),
                        run_name='run_a',
                    )


if __name__ == '__main__':
    unittest.main()
