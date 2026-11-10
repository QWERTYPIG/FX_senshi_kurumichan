import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation import select_iterations, walk_forward_splits


class WalkForwardTests(unittest.TestCase):
    def test_order_expanding_history_and_disjoint_outer_folds(self):
        dates = pd.bdate_range("2026-01-01", periods=50)
        splits, test_start = walk_forward_splits(
            dates, folds=3, valid_days=5, test_days=7, early_stop_days=4, min_train_days=10)
        self.assertEqual(test_start, dates[-7])
        outer_dates = []
        for index, split in enumerate(splits):
            self.assertLess(split["inner_train_end"], split["early_stop_start"])
            self.assertLessEqual(split["early_stop_start"], split["train_end"])
            self.assertLess(split["train_end"], split["valid_start"])
            self.assertLessEqual(split["valid_end"], test_start)
            held_out = dates[(dates >= split["valid_start"]) & (dates <= split["valid_end"])]
            self.assertEqual(len(held_out), 5)
            outer_dates.extend(held_out)
            if index:
                self.assertGreater(split["train_end"], splits[index - 1]["train_end"])
                self.assertLess(splits[index - 1]["valid_end"], split["valid_start"])
        self.assertEqual(len(set(outer_dates)), 15)
        self.assertLess(max(outer_dates), test_start)

    def test_duplicate_stock_dates_do_not_change_split(self):
        dates = pd.bdate_range("2026-01-01", periods=50)
        params = dict(folds=2, valid_days=5, test_days=5, early_stop_days=3, min_train_days=10)
        self.assertEqual(walk_forward_splits(dates, **params),
                         walk_forward_splits(list(dates) * 100, **params))

    def test_insufficient_history_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Need at least"):
            walk_forward_splits(pd.bdate_range("2026-01-01", periods=20),
                                folds=3, valid_days=5, test_days=5, early_stop_days=3, min_train_days=10)

    def test_iteration_selection_uses_median(self):
        self.assertEqual(select_iterations([3, 4, 100]), 4)
        self.assertEqual(select_iterations([3, 4]), 3)


if __name__ == "__main__":
    unittest.main()
