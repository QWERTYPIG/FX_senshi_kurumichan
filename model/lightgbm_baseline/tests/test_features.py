"""Behavior checks for the temporal boundary and per-stock isolation."""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import FEATURES, build_dataset


class CausalFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "market").mkdir()
        self.dates = pd.bdate_range("2026-09-01", periods=15)
        for stock, scale in [("1111", 1), ("2222", 10)]:
            pd.DataFrame({
                "date": self.dates,
                "etl:adj_open": [scale * (100 + i) for i in range(15)],
                "etl:adj_close": [scale * (101 + i) for i in range(15)],
                "etl:adj_high": [scale * (102 + i) for i in range(15)],
                "etl:adj_low": [scale * (99 + i) for i in range(15)],
                "price:成交股數": [1000 + i for i in range(15)],
                "y": [0] * 15,
            }).to_csv(self.root / "market" / f"{stock}.csv", index=False)
        self.ids = ["1111", "2222"]

    def test_current_and_future_rows_cannot_change_target_features(self):
        before = build_dataset(self.root, self.ids)
        cutoff = self.dates[10]
        path = self.root / "market/1111.csv"
        raw = pd.read_csv(path)
        raw.loc[pd.to_datetime(raw["date"]) >= cutoff, "etl:adj_close"] *= 100
        raw["y"] = 1
        raw.to_csv(path, index=False)
        after = build_dataset(self.root, self.ids)
        pd.testing.assert_frame_equal(
            before.loc[before["date"] <= cutoff, FEATURES],
            after.loc[after["date"] <= cutoff, FEATURES],
        )
        valid = before["feature_date"].notna()
        self.assertTrue((before.loc[valid, "feature_date"] < before.loc[valid, "date"]).all())
        self.assertTrue(before.groupby("stock_id", observed=True).head(1)["feature_date"].isna().all())

    def test_future_template_uses_last_observed_day(self):
        next_date = self.dates[-1] + pd.offsets.BDay(1)
        result = build_dataset(self.root, self.ids, dates=[next_date])
        self.assertEqual(len(result), 2)
        self.assertTrue((result["feature_date"] == self.dates[-1]).all())
        self.assertTrue(result["intraday_lag1"].notna().all())

    def test_no_history_does_not_borrow_future_data(self):
        result = build_dataset(self.root, self.ids, dates=[self.dates[0]])
        self.assertTrue(result["feature_date"].isna().all())
        self.assertTrue(result["intraday_lag1"].isna().all())


if __name__ == "__main__":
    unittest.main()
