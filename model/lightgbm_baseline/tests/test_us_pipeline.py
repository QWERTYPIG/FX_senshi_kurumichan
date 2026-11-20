"""Train/load the US feature schema without changing the baseline interface."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

MODEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL))
from features import FEATURES
from us_features import SYMBOLS, US_FEATURES


class USPipelineTests(unittest.TestCase):
    def test_us_model_predicts_and_rejects_stale_inputs(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            live, history, us, artifacts = [root / part for part in ("live", "history", "us", "artifacts")]
            for directory in (live / "market", history / "market", us / "prices"):
                directory.mkdir(parents=True)
            dates = pd.bdate_range("2026-08-03", periods=36)
            prices = np.arange(len(dates), dtype=float) + 100
            raw = pd.DataFrame({"date": dates, "etl:adj_open": prices,
                                "etl:adj_close": prices + np.where(np.arange(len(dates)) % 2, 1, -1),
                                "etl:adj_high": prices + 2, "etl:adj_low": prices - 2,
                                "price:成交股數": prices * 100})
            for symbol in ("1111", "2222"):
                raw.iloc[:30].to_csv(history / "market" / f"{symbol}.csv", index=False)
                raw.iloc[30:].to_csv(live / "market" / f"{symbol}.csv", index=False)
            us_dates = pd.bdate_range(dates[0] - pd.Timedelta(days=70), dates[-1])
            values = np.arange(len(us_dates), dtype=float) + 100
            for symbol in SYMBOLS:
                pd.DataFrame({"date": us_dates, "stock_id": symbol, "Open": values - 1,
                              "High": values + 1, "Low": values - 2, "Close": values,
                              "Adj_Close": values, "Volume": values * 100}).to_csv(us / "prices" / f"{symbol}.csv", index=False)
            template = root / "template.csv"
            pd.DataFrame({"id": ["2222"], "2026-10-01": [0]}).to_csv(template, index=False)
            common = ["--data-dir", str(live), "--history-dir", str(history),
                      "--us-data-dir", str(us), "--artifacts-dir", str(artifacts)]
            trained = subprocess.run([sys.executable, str(MODEL / "train.py"), *common,
                                      "--feature-set", "us", "--stock-template", str(template),
                                      "--cv-folds", "2", "--valid-days", "3", "--test-days", "3",
                                      "--early-stop-days", "3", "--min-train-days", "10"], capture_output=True, text=True)
            self.assertEqual(trained.returncode, 0, trained.stdout + trained.stderr)
            metadata = json.loads((artifacts / "metadata.json").read_text())
            self.assertEqual(metadata["features"], FEATURES + US_FEATURES)
            self.assertEqual(metadata["feature_set"], "us")
            output = root / "prediction.csv"
            next_date = (dates[-1] + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
            command = [sys.executable, str(MODEL / "predict.py"), *common, "--output", str(output)]
            result = subprocess.run([*command, "--date", next_date], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            prediction = pd.read_csv(output, dtype={"id": str})
            self.assertEqual(prediction["id"].tolist(), ["2222"])
            self.assertTrue(prediction[next_date].isin([0, 1]).all())
            original = output.read_bytes()
            stale_date = (dates[-1] + pd.Timedelta(days=9)).strftime("%Y-%m-%d")
            rejected = subprocess.run([*command, "--date", stale_date], capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("US history", rejected.stderr)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
