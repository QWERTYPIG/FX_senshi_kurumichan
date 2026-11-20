"""Verify that the ZIP preserves baseline features and uses the intended lags."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FeatureExperimentTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        # features.py computes ROOT from its location, so retain enough parents.
        source = self.root / "source/model/experiment"
        source.mkdir(parents=True)
        with ZipFile(ROOT / "lightgbm_ab_experiment.zip") as archive:
            (source / "features.py").write_bytes(archive.read("features.py"))
        self.candidate = load_module("candidate_features", source / "features.py")
        self.baseline = load_module("baseline_features", ROOT / "model/lightgbm_baseline/features.py")
        self.dates = pd.bdate_range("2026-01-01", periods=15)
        self.raw = pd.DataFrame({
            "date": self.dates, "etl:adj_open": np.arange(100, 115),
            "etl:adj_close": np.arange(102, 117), "etl:adj_high": np.arange(103, 118),
            "etl:adj_low": np.arange(99, 114), "price:成交股數": np.arange(1000, 1015),
        })
        (self.root / "market").mkdir()
        self.path = self.root / "market/1111.csv"
        self.raw.to_csv(self.path, index=False)

    def test_baseline_features_are_identical(self):
        old = self.baseline.build_dataset(self.root, ["1111"])
        new = self.candidate.build_dataset(self.root, ["1111"])
        self.assertEqual(self.candidate.FEATURE_SETS["A"], self.baseline.FEATURES)
        pd.testing.assert_frame_equal(old[self.baseline.FEATURES], new[self.baseline.FEATURES])

    def test_lags_reference_previous_valid_trading_days(self):
        frame = self.candidate.build_dataset(self.root, ["1111"])
        row = frame.iloc[5]
        for lag in (1, 2, 3):
            source_index = 5 - lag
            actual = self.raw.iloc[source_index]
            self.assertAlmostEqual(row[f"intraday_lag{lag}"], actual["etl:adj_close"] / actual["etl:adj_open"] - 1)
            self.assertAlmostEqual(row[f"return_lag{lag}"],
                                   actual["etl:adj_close"] / self.raw.iloc[source_index - 1]["etl:adj_close"] - 1)

    def test_same_day_and_future_changes_cannot_change_past_features(self):
        before = self.candidate.build_dataset(self.root, ["1111"])
        cutoff = self.dates[8]
        self.raw.loc[self.raw["date"] >= cutoff, "etl:adj_close"] *= 10
        self.raw.to_csv(self.path, index=False)
        after = self.candidate.build_dataset(self.root, ["1111"])
        features = self.candidate.FEATURE_SETS["B"]
        pd.testing.assert_frame_equal(before.loc[before["date"] <= cutoff, features],
                                      after.loc[after["date"] <= cutoff, features])


if __name__ == "__main__":
    unittest.main()
