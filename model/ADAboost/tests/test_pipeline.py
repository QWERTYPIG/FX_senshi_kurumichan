import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

MODEL_DIR = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_small_training_and_binary_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            history = root / "history"
            live = root / "live"
            artifacts = root / "artifacts"
            (history / "market").mkdir(parents=True)
            (live / "market").mkdir(parents=True)
            dates = pd.bdate_range("2026-07-01", periods=36)
            for stock_index, stock in enumerate(("1111", "2222", "3333")):
                open_price = [100 + stock_index * 5 + index for index in range(len(dates))]
                rows = pd.DataFrame({
                    "date": dates.strftime("%Y-%m-%d"),
                    "etl:adj_open": open_price,
                    "etl:adj_close": [value + (1 if (index + stock_index) % 2 else -1)
                                      for index, value in enumerate(open_price)],
                    "etl:adj_high": [value + 2 for value in open_price],
                    "etl:adj_low": [value - 2 for value in open_price],
                    "price:成交股數": [1000 + index * 10 + stock_index for index in range(len(dates))],
                })
                rows.iloc[:30].to_csv(history / "market" / f"{stock}.csv", index=False)
                rows.iloc[30:].to_csv(live / "market" / f"{stock}.csv", index=False)
            template = root / "template.csv"
            pd.DataFrame({"id": ["3333", "1111"], "2026-09-01": [0, 0]}).to_csv(template, index=False)

            train = subprocess.run([
                sys.executable, str(MODEL_DIR / "train.py"),
                "--data-dir", str(live), "--history-dir", str(history),
                "--stock-template", str(template), "--artifacts-dir", str(artifacts),
                "--cv-folds", "2", "--valid-days", "3", "--test-days", "3",
                "--min-train-days", "10", "--candidate", "1,5,0.5",
            ], capture_output=True, text=True)
            self.assertEqual(train.returncode, 0, train.stdout + train.stderr)
            metadata = json.loads((artifacts / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["prediction_stock_ids"], ["3333", "1111"])
            self.assertEqual(metadata["trained_through"], str(dates[-1].date()))
            self.assertTrue(pd.notna(metadata["threshold"]))
            metrics = json.loads((artifacts / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["test"]["rows"], 6)

            output = root / "prediction.csv"
            next_date = (dates[-1] + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
            prediction = subprocess.run([
                sys.executable, str(MODEL_DIR / "predict.py"),
                "--data-dir", str(live), "--history-dir", str(history),
                "--artifacts-dir", str(artifacts), "--date", next_date,
                "--output", str(output),
            ], capture_output=True, text=True)
            self.assertEqual(prediction.returncode, 0, prediction.stdout + prediction.stderr)
            result = pd.read_csv(output, dtype={"id": str})
            self.assertEqual(result["id"].tolist(), ["3333", "1111"])
            self.assertEqual(result.columns.tolist(), ["id", next_date])
            self.assertTrue(result[next_date].isin([0, 1]).all())


if __name__ == "__main__":
    unittest.main()
