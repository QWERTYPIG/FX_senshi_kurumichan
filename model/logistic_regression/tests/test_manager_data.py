import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import load_history
import train


class ManagerDataTests(unittest.TestCase):
    def test_environment_directories(self):
        env = dict(os.environ, MANAGER_BASIC_DATA_DIR='/basic', MANAGER_DATA_DIR='/recent')
        result = subprocess.run([sys.executable, '-c',
                                 'import common; print(common.BASIC_DATA_DIR); print(common.DATA_DIR)'],
                                cwd=Path(train.__file__).parent, env=env,
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.splitlines(), ['/basic', '/recent'])

    def test_training_routes_manager_defaults_and_cli_overrides(self):
        frame = pd.DataFrame({'date': pd.to_datetime(['2025-12-30', '2026-01-02']),
                              'stock_source_date': pd.to_datetime(['2025-12-29', '2025-12-30']),
                              'target': [.01, -.01]})
        cases = [([], Path('/basic'), Path('/recent'), Path('/basic'), Path('/recent')),
                 (['--basic-data-dir', '/custom-basic', '--data-dir', '/custom-recent'],
                  Path('/basic'), Path('/recent'), Path('/custom-basic'), Path('/custom-recent')),
                 ([], None, Path('/complete'), Path('/complete'), None),
                 ([], Path('/same'), Path('/same'), Path('/same'), None)]
        for extra, basic, recent, expected_history, expected_update in cases:
            with self.subTest(extra=extra, basic=basic), tempfile.TemporaryDirectory() as out:
                argv = ['train.py', '--artifacts-dir', out, '--penalties', '.1', '--half-lives', '0', *extra]
                with patch.object(sys, 'argv', argv), patch('train.BASIC_DATA_DIR', basic), \
                     patch('train.DATA_DIR', recent), \
                     patch('train.read_template', return_value=pd.DataFrame({'id': ['2330']})), \
                     patch('train.build_data', return_value=frame) as build, \
                     patch('train.encode', return_value=[]), patch('train.fit', return_value={}), \
                     patch('train.predict', side_effect=lambda model, val: (val.target.to_numpy(), None)), \
                     contextlib.redirect_stdout(io.StringIO()):
                    train.main()
                args, kwargs = build.call_args
                self.assertEqual(args[0], expected_history)
                self.assertEqual(kwargs['update_dir'], expected_update)
                self.assertEqual(args[1].columns[-1], '2026-05-31')

    def test_merge_overlap_and_cutoff_without_changing_sources(self):
        with tempfile.TemporaryDirectory() as root:
            basic, recent = Path(root) / 'basic', Path(root) / 'recent'
            for relative in [Path('market/2330.csv'), Path('global/global_data.csv')]:
                for directory, dates, values in [
                    (basic, ['2026-05-29', '2026-08-31'], [100, 101]),
                    (recent, ['2026-08-31', '2026-09-01'], [102, 103])]:
                    path = directory / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({'date': dates, 'value': values}).to_csv(path, index=False)
                originals = [(directory / relative).read_bytes() for directory in [basic, recent]]
                early = load_history(basic, recent, relative, pd.Timestamp('2026-05-31'))
                self.assertEqual(early.value.tolist(), [100])
                later = load_history(basic, recent, relative, pd.Timestamp('2026-09-01'))
                self.assertEqual(later.value.tolist(), [100, 102, 103])
                self.assertEqual(originals, [(directory / relative).read_bytes() for directory in [basic, recent]])


if __name__ == '__main__':
    unittest.main()
