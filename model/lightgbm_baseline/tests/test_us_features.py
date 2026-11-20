"""US signals must be available before Taiwan's forecast, including holidays."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from us_features import SYMBOLS, US_FEATURES, add_us_features, load_us_session_features, us_coverage


class USFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "prices").mkdir()
        self.dates = pd.bdate_range("2026-01-01", "2026-03-13")
        self.raw = pd.DataFrame({
            "date": self.dates, "Open": np.arange(len(self.dates)) + 99,
            "High": np.arange(len(self.dates)) + 102, "Low": np.arange(len(self.dates)) + 98,
            "Close": np.arange(len(self.dates)) + 100, "Adj_Close": np.arange(len(self.dates)) + 100,
            "Volume": np.arange(len(self.dates)) + 1000,
        })
        for symbol in SYMBOLS:
            self.raw.assign(stock_id=symbol).to_csv(self.root / "prices" / f"{symbol}.csv", index=False)

    def query(self, date, previous):
        return pd.DataFrame({"date": pd.to_datetime([date]), "feature_date": pd.to_datetime([previous])})

    def test_ny_same_day_and_future_rows_do_not_affect_taiwan_morning(self):
        query = self.query("2026-03-10", "2026-03-09")
        before = add_us_features(query, self.root)
        path = self.root / "prices/SPY.csv"
        raw = pd.read_csv(path)
        raw.loc[raw["date"] >= "2026-03-10", ["Close", "Adj_Close"]] *= 100
        raw.to_csv(path, index=False)
        after = add_us_features(query, self.root)
        pd.testing.assert_frame_equal(before[US_FEATURES], after[US_FEATURES])
        monday = self.raw.loc[self.raw["date"] == "2026-03-09", "Adj_Close"].iloc[0]
        friday = self.raw.loc[self.raw["date"] == "2026-03-06", "Adj_Close"].iloc[0]
        self.assertAlmostEqual(before.iloc[0]["us_spy_ret1"], monday / friday - 1)

    def test_monday_uses_friday_and_accumulated_move_handles_taiwan_gap(self):
        result = add_us_features(self.query("2026-03-09", "2026-03-05"), self.root)
        friday = self.raw.loc[self.raw["date"] == "2026-03-06", "Adj_Close"].iloc[0]
        wednesday = self.raw.loc[self.raw["date"] == "2026-03-04", "Adj_Close"].iloc[0]
        self.assertAlmostEqual(result.iloc[0]["us_spy_since_tw_close"], friday / wednesday - 1)
        self.assertEqual(result.iloc[0]["us_age_days"], 3)

    def test_publication_date_is_correct_across_dst(self):
        source = load_us_session_features(self.root, "SPY")
        for day, available in [("2026-03-06", "2026-03-07 08:10"), ("2026-03-09", "2026-03-10 08:10")]:
            actual = source.loc[source["source_date"] == pd.Timestamp(day), "available_at"].iloc[0]
            self.assertEqual(actual, pd.Timestamp(available, tz="Asia/Taipei"))

    def test_coverage_requires_warmup_and_rejects_stale_prices(self):
        first = add_us_features(self.query("2026-01-05", "2026-01-02"), self.root)
        self.assertFalse(us_coverage(first).iloc[0])
        normal = add_us_features(self.query("2026-03-10", "2026-03-09"), self.root)
        self.assertTrue(us_coverage(normal).iloc[0])
        stale = add_us_features(self.query("2026-04-01", "2026-03-31"), self.root)
        self.assertFalse(us_coverage(stale).iloc[0])

    def test_adjusted_close_prevents_raw_stock_split_jump(self):
        path = self.root / "prices/SPY.csv"
        raw = pd.read_csv(path)
        raw["Close"] = raw["Close"].astype(float)
        raw.loc[raw.index >= 30, "Close"] /= 2
        raw.to_csv(path, index=False)
        result = load_us_session_features(self.root, "SPY")
        self.assertLess(abs(result.iloc[30]["ret1"]), 0.02)


if __name__ == "__main__":
    unittest.main()
