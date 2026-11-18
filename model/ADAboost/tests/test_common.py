import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import evaluate, read_template, walk_forward_splits
from train import select_threshold


class CommonTests(unittest.TestCase):
    def test_template_keeps_string_ids_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "template.csv"
            pd.DataFrame({"id": ["001", "002"], "2026-11-19": [0, 0]}).to_csv(path, index=False)
            result = read_template(path)
            self.assertEqual(result["id"].tolist(), ["001", "002"])
            pd.DataFrame({"id": ["001", "001"], "2026-11-19": [0, 0]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "unique"):
                read_template(path)

    def test_walk_forward_is_ordered_and_test_is_untouched(self):
        dates = pd.bdate_range("2026-01-01", periods=30)
        splits, test_start = walk_forward_splits(
            list(dates) * 3, folds=2, valid_days=4, test_days=5, min_train_days=10,
        )
        self.assertEqual(test_start, dates[-5])
        self.assertLess(splits[0]["train_end"], splits[0]["valid_start"])
        self.assertLess(splits[0]["valid_end"], splits[1]["valid_start"])
        self.assertLess(splits[1]["valid_end"], test_start)

    def test_official_score(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "intraday_return": [0.02, -0.01],
        })
        result = evaluate(frame, np.array([1, 0]))
        self.assertAlmostEqual(result["score"], 0.015)
        self.assertEqual(result["accuracy"], 1.0)

    def test_threshold_selection_matches_inference_and_keeps_ties_together(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-01"] * 4),
            "intraday_return": [0.04, 0.03, -0.02, -0.01],
        })
        probabilities = np.array([0.8, 0.6, 0.6, 0.2])
        threshold, score, up_rate = select_threshold(frame, probabilities)
        prediction = (probabilities >= threshold).astype(int)
        self.assertEqual(threshold, 0.6)
        self.assertEqual(prediction.tolist(), [1, 1, 1, 0])
        self.assertAlmostEqual(score, evaluate(frame, prediction)["score"])
        self.assertEqual(up_rate, 0.75)


if __name__ == "__main__":
    unittest.main()
