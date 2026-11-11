import sys
import unittest
import tempfile
import json
import contextlib
import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import fit, predict, sample_weights
import train


class RecencyTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'date': pd.to_datetime([
            '2025-01-01', '2025-01-01', '2025-01-03', '2025-01-06']),
            'target': [.01, -.02, .01, -.01], 'signal': [1., -1., 1., -1.]})

    def test_disabled_matches_original_weights(self):
        r = self.frame.target.abs().to_numpy()
        np.testing.assert_array_equal(sample_weights(self.frame), r / r.mean())

    def test_half_life_and_same_date_weights(self):
        w = sample_weights(self.frame, 2)
        self.assertAlmostEqual(w.mean(), 1)
        self.assertAlmostEqual(w[0] / w[-1], .5)
        self.assertAlmostEqual(w[1] / w[0], 2)

    def test_order_and_calendar_gap_invariance(self):
        shuffled = self.frame.iloc[[3, 0, 2, 1]]
        expected = pd.Series(sample_weights(self.frame, 2), index=self.frame.index)
        np.testing.assert_allclose(sample_weights(shuffled, 2), expected.loc[shuffled.index])
        moved = self.frame.copy()
        moved['date'] = moved.date + pd.Timedelta(days=300)
        np.testing.assert_array_equal(sample_weights(moved, 2), sample_weights(self.frame, 2))

    def test_invalid_weights(self):
        for h in [-1, np.inf, np.nan]:
            with self.assertRaises(ValueError):
                sample_weights(self.frame, h)
        with self.assertRaises(ValueError):
            sample_weights(self.frame.assign(target=0), 2)

    def test_recent_regime_changes_classifier(self):
        frame = pd.DataFrame({'date': pd.bdate_range('2025-01-01', periods=40),
                              'target': [-.01] * 30 + [.01] * 10, 'signal': 0.})
        old = fit(frame, ['signal'], .1)
        recent = fit(frame, ['signal'], .1, half_life=3)
        self.assertLess(predict(old, frame)[1][-1], .5)
        self.assertGreater(predict(recent, frame)[1][-1], .5)
        np.testing.assert_array_equal(old['magnitude_beta'], recent['magnitude_beta'])

    def test_training_split_cutoff_and_refit(self):
        frame = self.frame.assign(stock_source_date=self.frame.date - pd.Timedelta(days=1))
        calls = []
        def checked_fit(rows, features, penalty, half_life):
            calls.append((rows.date.max(), half_life))
            return {'half_life': half_life}
        with tempfile.TemporaryDirectory() as out:
            argv = ['train.py', '--artifacts-dir', out, '--validation-start', '2025-01-03',
                    '--train-end', '2025-01-03', '--penalties', '.1', '--half-lives', '0', '2']
            with patch.object(sys, 'argv', argv), patch('train.read_template', return_value=pd.DataFrame({'id':['x']})), \
                 patch('train.build_data', return_value=frame), patch('train.encode', return_value=['signal']), \
                 patch('train.fit', side_effect=checked_fit), \
                 patch('train.predict', side_effect=lambda model, val: (val.signal.to_numpy(), None)), \
                 contextlib.redirect_stdout(io.StringIO()):
                train.main()
            self.assertEqual(calls, [(pd.Timestamp('2025-01-01'), 0),
                                     (pd.Timestamp('2025-01-01'), 2),
                                     (pd.Timestamp('2025-01-03'), 0)])
            result = json.loads((Path(out) / 'metrics.json').read_text())
            self.assertEqual(result['validation']['rows'], 1)
            self.assertEqual(result['training_rows'], 3)
            self.assertEqual(result['validation']['half_life'], 0)


if __name__ == '__main__':
    unittest.main()
