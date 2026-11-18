import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import FEATURES, build_dataset, discover_stock_ids, load_stock_history


def market_frame(dates, scale=1.0):
    size = len(dates)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "etl:adj_open": [scale * (100 + index) for index in range(size)],
        "etl:adj_close": [scale * (101 + index + (index % 2)) for index in range(size)],
        "etl:adj_high": [scale * (103 + index) for index in range(size)],
        "etl:adj_low": [scale * (98 + index) for index in range(size)],
        "price:成交股數": [1000 + index * 10 for index in range(size)],
        "RSI": [50 + index / 10 for index in range(size)],
        "SMA": [scale * (100 + index) for index in range(size)],
    })


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "market").mkdir()
        self.dates = pd.bdate_range("2026-08-03", periods=30)
        for stock, scale in (("1111", 1.0), ("2222", 2.0)):
            market_frame(self.dates, scale).to_csv(self.root / "market" / f"{stock}.csv", index=False)

    def test_target_date_uses_only_strictly_earlier_features(self):
        before = build_dataset(self.root, ["1111", "2222"])
        cutoff = self.dates[20]
        path = self.root / "market/1111.csv"
        changed = pd.read_csv(path)
        changed.loc[pd.to_datetime(changed["date"]) >= cutoff, "etl:adj_close"] *= 100
        changed.to_csv(path, index=False)
        after = build_dataset(self.root, ["1111", "2222"])
        columns = ["stock_id", *FEATURES]
        pd.testing.assert_frame_equal(
            before.loc[before["date"] <= cutoff, columns].reset_index(drop=True),
            after.loc[after["date"] <= cutoff, columns].reset_index(drop=True),
        )
        usable = before["feature_date"].notna()
        self.assertTrue((before.loc[usable, "feature_date"] < before.loc[usable, "date"]).all())

    def test_future_date_uses_last_observation(self):
        next_date = self.dates[-1] + pd.offsets.BDay(1)
        result = build_dataset(self.root, ["1111", "2222"], dates=[next_date])
        self.assertEqual(len(result), 2)
        self.assertTrue((result["feature_date"] == self.dates[-1]).all())
        self.assertTrue(result["intraday_lag1"].notna().all())

    def test_live_overlap_replaces_history_without_modifying_source(self):
        history = self.root / "history"
        (history / "market").mkdir(parents=True)
        full = pd.read_csv(self.root / "market/1111.csv")
        full.to_csv(history / "market/1111.csv", index=False)
        live = full.iloc[-2:].copy()
        live.loc[live.index[0], "etl:adj_close"] = 999
        live.to_csv(self.root / "market/1111.csv", index=False)
        merged = load_stock_history(self.root / "market/1111.csv", history / "market/1111.csv")
        self.assertEqual(len(merged), 30)
        self.assertEqual(merged.iloc[-2]["etl:adj_close"], 999)
        pd.testing.assert_frame_equal(pd.read_csv(history / "market/1111.csv"), full)

    def test_missing_data_error_lists_paths_and_remedies(self):
        missing_live = self.root / "missing-live"
        missing_history = self.root / "missing-history"
        with self.assertRaises(FileNotFoundError) as raised:
            discover_stock_ids(missing_live, missing_history)
        message = str(raised.exception)
        self.assertIn(str((missing_live / "market").resolve()), message)
        self.assertIn(str((missing_history / "market").resolve()), message)
        self.assertIn("--history-dir", message)
        self.assertIn("data_access_key", message)


if __name__ == "__main__":
    unittest.main()
