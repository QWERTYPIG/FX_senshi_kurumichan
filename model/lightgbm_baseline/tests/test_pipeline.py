"""Exercise full-universe training, target-only output and fresh data on retraining."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

MODEL_DIR = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_retraining_reads_updates_and_prediction_keeps_target_subset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            history, live, artifacts = root / "history", root / "live", root / "artifacts"
            for source in (history, live):
                (source / "market").mkdir(parents=True)
            dates = pd.bdate_range("2026-08-03", periods=36)
            originals = {}
            for stock in ["1111", "2222", "3333"]:
                rows = pd.DataFrame({
                    "date": dates.strftime("%Y-%m-%d"),
                    "etl:adj_open": [100 + i for i in range(36)],
                    "etl:adj_close": [100 + i + (1 if i % 2 else -1) for i in range(36)],
                    "etl:adj_high": [102 + i for i in range(36)],
                    "etl:adj_low": [98 + i for i in range(36)],
                    "price:成交股數": [1000 + i for i in range(36)],
                })
                rows.iloc[:30].to_csv(history / "market" / f"{stock}.csv", index=False)
                rows.iloc[30:35].to_csv(live / "market" / f"{stock}.csv", index=False)
                originals[stock] = rows
            template = root / "template.csv"
            pd.DataFrame({"id": ["3333", "1111"], "2026-10-01": 0}).to_csv(template, index=False)
            common = ["--data-dir", str(live), "--history-dir", str(history),
                      "--artifacts-dir", str(artifacts)]

            def run(script, *args):
                result = subprocess.run([sys.executable, str(MODEL_DIR / script), *common, *args],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            def train():
                run("train.py", "--stock-template", str(template), "--valid-days", "3", "--test-days", "3",
                    "--cv-folds", "2", "--early-stop-days", "3", "--min-train-days", "10")
                return json.loads((artifacts / "metadata.json").read_text())

            first = train()
            self.assertEqual(first["stock_ids"], ["1111", "2222", "3333"])
            self.assertEqual(first["prediction_stock_ids"], ["3333", "1111"])
            self.assertEqual(first["trained_through"], str(dates[34].date()))
            for stock, rows in originals.items():
                rows.iloc[30:].to_csv(live / "market" / f"{stock}.csv", index=False)
            second = train()
            self.assertEqual(second["trained_through"], str(dates[35].date()))
            self.assertEqual(second["training_rows"], first["training_rows"] + 3)
            metrics = json.loads((artifacts / "metrics.json").read_text())
            self.assertEqual(metrics["test"]["rows"], 6)  # 3 dates x 2 target stocks
            self.assertEqual(metrics["validation"]["rows"], 12)
            self.assertEqual(len(metrics["cross_validation"]["folds"]), 2)
            oof = pd.read_csv(artifacts / "validation_predictions.csv")
            self.assertFalse(oof.duplicated(["date", "id"]).any())
            self.assertLess(oof["date"].max(), metrics["test"]["start"])
            output = root / "prediction.csv"
            next_date = (dates[-1] + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
            run("predict.py", "--date", next_date, "--output", str(output))
            prediction = pd.read_csv(output, dtype={"id": str})
            self.assertEqual(prediction["id"].tolist(), ["3333", "1111"])
            self.assertEqual(prediction.columns.tolist(), ["id", next_date])
            self.assertTrue(prediction[next_date].isin([0, 1]).all())
            # Change only held-out test prices: CV scores and selected rounds must
            # remain identical, even though test and deployment results may change.
            for stock, rows in originals.items():
                changed = rows.iloc[30:].copy()
                changed.loc[changed.index[-3:], "etl:adj_close"] += 10
                changed.to_csv(live / "market" / f"{stock}.csv", index=False)
            third = train()
            changed_metrics = json.loads((artifacts / "metrics.json").read_text())
            self.assertEqual(third["params"]["n_estimators"], second["params"]["n_estimators"])
            self.assertEqual(changed_metrics["cross_validation"], metrics["cross_validation"])
            self.assertEqual(changed_metrics["validation"], metrics["validation"])
            self.assertNotEqual(changed_metrics["test"]["mse"], metrics["test"]["mse"])


if __name__ == "__main__":
    unittest.main()
