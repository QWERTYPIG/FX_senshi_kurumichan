"""Protect chronological member selection and averaging semantics."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import chronological_blends, rank_on_past


class BlendingTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "date": pd.bdate_range("2026-01-01", periods=16).strftime("%Y-%m-%d"),
            "id": "1111", "fold": np.repeat([1, 2, 3, 4], 4),
            "actual_return": [0.01, -0.02, 0.03, -0.01] * 4,
        })
        self.predictions = {
            1: np.repeat(0.09, 16),
            2: np.repeat(-0.01, 16),
            3: np.repeat(-0.01, 16),
        }

    def test_future_labels_and_predictions_cannot_change_past_ranking(self):
        expected = rank_on_past(self.frame, self.predictions, before_fold=3)
        changed = self.frame.copy()
        changed.loc[changed["fold"] >= 3, "actual_return"] *= -100
        predictions = {trial: values.copy() for trial, values in self.predictions.items()}
        for values in predictions.values():
            values[changed["fold"] >= 3] *= -100
        self.assertEqual(rank_on_past(changed, predictions, before_fold=3), expected)

    def test_average_returns_precedes_direction_threshold(self):
        records, outputs = chronological_blends(self.frame, self.predictions, [1, 3], warmup_folds=2)
        evaluated = self.frame["fold"] >= 3
        np.testing.assert_allclose(outputs[3][evaluated], (0.09 - 0.01 - 0.01) / 3)
        self.assertTrue((outputs[3][evaluated] > 0).all())  # Majority vote would be negative.
        self.assertTrue(np.isnan(outputs[3][~evaluated]).all())
        self.assertEqual(records[3]["metrics"]["rows"], 8)
        self.assertEqual(len(records[3]["folds"][0]["members"]), 3)

    def test_changing_later_actuals_cannot_change_earlier_blend(self):
        _, original = chronological_blends(self.frame, self.predictions, [1, 3], warmup_folds=2)
        changed = self.frame.copy()
        changed.loc[changed["fold"] >= 3, "actual_return"] *= -100
        _, result = chronological_blends(changed, self.predictions, [1, 3], warmup_folds=2)
        fold3 = self.frame["fold"] == 3
        np.testing.assert_allclose(original[1][fold3], result[1][fold3])


if __name__ == "__main__":
    unittest.main()
