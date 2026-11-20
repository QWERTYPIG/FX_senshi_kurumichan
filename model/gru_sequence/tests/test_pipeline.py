import csv
import json
import math
import os
import random
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
MARKET_HEADER = (
    "date", "etl:adj_close", "etl:adj_open", "etl:adj_high", "etl:adj_low",
    "price:成交筆數", "price:成交金額", "price:成交股數",
    "foreign_investors_shareholding:全體外資及陸資持股比率",
    "institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)",
    "institutional_investors_trading_summary:投信買賣超股數",
    "institutional_investors_trading_summary:自營商買賣超股數(自行買賣)",
    "institutional_investors_trading_summary:自營商買賣超股數(避險)",
    "RSI", "SMA", "MACD", "ATR", "BBANDS_upper", "BBANDS_mid", "BBANDS_lower", "y",
)


def market_row(date, base=100):
    return [date, base + 2, base, base + 3, base - 1, 1000, 27500, 10000,
            20, -70, -15, 10, 0, 55, base - 7, -2, 4, base + 1, base - 7, base - 15, 0]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle).writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.reader(handle))


class GRUSequencePipelineTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.history = self.root / "basic_data"
        self.update = self.root / "data"
        self.template = self.root / "template.csv"
        self.output = self.root / "prediction.csv"
        self.artifacts = self.root / "artifacts"
        self.config_path = self.root / "tiny.json"
        self.stocks = ["0123", "1101", "2330"]
        self.targets = ["2330", "0123"]
        self.config = {"window_length": 3, "hidden_size": 4, "gru_layers": 1,
                       "head_hidden_size": 4, "learning_rate": 0.01, "epochs": 2, "seed": 7}
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        for data_dir in (self.history, self.update):
            (data_dir / "market").mkdir(parents=True)
        (self.history / "info.json").write_text(json.dumps({
            "Top100@Y": self.stocks,
            "Top50@Y": self.targets,
        }), encoding="utf-8")
        for i, stock in enumerate(self.stocks):
            write_csv(self.market_file(self.history, stock), [MARKET_HEADER, *[
                market_row(f"2026-05-{day:02d}", 100 + day * (i + 1)) for day in range(18, 23)
            ]])
            write_csv(self.market_file(self.update, stock), [MARKET_HEADER, *[
                market_row(f"2026-05-{day:02d}", 102 + day * (i + 1)) for day in (22, 25)
            ]])

    def market_file(self, data_dir, stock):
        return data_dir / "market" / f"{stock}.csv"

    def run_command(self, index, *args):
        entry = (MODEL_DIR / "usage.txt").read_text(encoding="utf-8").splitlines()[index]
        interpreter, *command = shlex.split(entry)
        self.assertEqual(interpreter, "python")
        result = subprocess.run(
            [sys.executable, *command, "--artifacts-dir", str(self.artifacts), *args],
            cwd=MODEL_DIR, env={**os.environ, "MANAGER_ROOT": str(self.root),
                                "MANAGER_BASIC_DATA_DIR": str(self.history),
                                "MANAGER_DATA_DIR": str(self.update),
                                "MANAGER_TEMPLATE": str(self.template),
                                "MANAGER_OUTPUT": str(self.output)},
            capture_output=True, text=True, timeout=20, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def train(self):
        self.run_command(1, "--config", str(self.config_path))
        return json.loads((self.artifacts / "metadata.json").read_text(encoding="utf-8"))

    def predict(self, *dates):
        write_csv(self.template, [["id", *dates], *[[stock, *([""] * len(dates))]
                                                   for stock in self.targets]])
        self.run_command(0)
        return read_csv(self.output)

    def details(self):
        with self.output.with_name("prediction_detail.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        indexed = {(row["id"], row["date"]): row for row in rows}
        self.assertEqual(len(indexed), len(rows))
        return indexed

    def test_stock_rows_on_or_after_prediction_date_cannot_change_its_scores(self):
        self.train()
        day, later = "2026-05-26", "2026-05-28"
        for stock in self.stocks:
            path = self.market_file(self.update, stock)
            write_csv(path, [*read_csv(path), market_row(day), market_row("2026-05-27")])
        header, *rows = self.predict(day, later)
        details = self.details()
        self.assertCountEqual(details, [(stock, date) for stock in self.targets for date in (day, later)])
        for stock, *directions in rows:
            for date, direction in zip(header[1:], directions):
                score = float(details[stock, date]["f"])
                self.assertTrue(math.isfinite(score))
                self.assertEqual(direction, str(int(score > 0)))
            self.assertEqual(details[stock, day]["latest_market_date"], "2026-05-25")
            self.assertEqual(details[stock, later]["latest_market_date"], "2026-05-27")

        rng = random.Random(12)
        for data_dir in (self.history, self.update):
            for stock in self.stocks:
                path = self.market_file(data_dir, stock)
                header, *rows = read_csv(path)
                for row in rows:
                    if row[0] >= day:
                        row[1:] = [rng.uniform(1, 1000) for _ in row[1:]]
                write_csv(path, [header, *rows])
        self.predict(day, later)
        updated = self.details()
        for stock in self.targets:
            self.assertEqual(updated[stock, day], details[stock, day])
        self.assertTrue(any(updated[stock, later]["f"] != details[stock, later]["f"]
                            for stock in self.targets))

    def test_retraining_expands_to_latest_usable_label_in_update_data(self):
        paths = [self.market_file(data_dir, stock) for data_dir in (self.history, self.update)
                 for stock in self.stocks]
        original = {path: path.read_bytes() for path in paths}
        self.assertEqual(self.train()["last_training_label_date"], "2026-05-25")
        self.assertEqual({path: path.read_bytes() for path in paths}, original)

        day = "2026-05-26"
        for stock in self.stocks:
            update_path = self.market_file(self.update, stock)
            history_path = self.market_file(self.history, stock)
            # A valid historical row is superseded by the update, even if the update is invalid.
            write_csv(history_path, [*read_csv(history_path), market_row(day)])
            latest = market_row(day)
            latest[2] = 0  # adjusted open
            write_csv(update_path, [*read_csv(update_path), latest])
        self.assertEqual(self.train()["last_training_label_date"], "2026-05-25")

        for stock in self.stocks:
            path = self.market_file(self.update, stock)
            write_csv(path, [*read_csv(path)[:-1], market_row(day)])
        expanded = {path: path.read_bytes() for path in paths}
        self.assertEqual(self.train()["last_training_label_date"], day)
        self.predict("2026-05-27")
        self.assertEqual({row["latest_market_date"] for row in self.details().values()}, {day})
        self.assertEqual({path: path.read_bytes() for path in paths}, expanded)

    def test_fixed_config_reproduces_scores_and_override_changes_training(self):
        date = "2026-05-26"
        self.assertEqual(self.train()["config"], self.config)
        self.predict(date)
        expected = self.details()
        self.train()
        self.predict(date)
        self.assertEqual(self.details(), expected)

        self.config["epochs"] = 1
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.assertEqual(self.train()["config"], self.config)
        self.predict(date)
        self.assertNotEqual(self.details(), expected)

    def test_training_uses_intraday_return_not_y(self):
        date, last_label = "2026-05-26", "2026-05-25"
        self.train()
        self.predict(date)
        positive = self.details()

        for stock in self.stocks:
            path = self.market_file(self.update, stock)
            header, *rows = read_csv(path)
            rows[-1][-1] = 1  # y does not enter the loss or the input
            write_csv(path, [header, *rows])
        self.train()
        self.predict(date)
        self.assertEqual(self.details(), positive)

        originals = {}
        for stock in self.stocks:
            path = self.market_file(self.update, stock)
            header, *rows = read_csv(path)
            originals[path] = [header, *[row.copy() for row in rows]]
            self.assertEqual(rows[-1][0], last_label)
            rows[-1][2] = float(rows[-1][1]) * 1.5  # open above close gives a negative return
            rows[-1][3] = rows[-1][2]
            write_csv(path, [header, *rows])
        self.train()
        for path, rows in originals.items():
            write_csv(path, rows)  # compare both models on exactly the same inference inputs
        self.predict(date)
        for key, row in self.details().items():
            self.assertLess(float(row["f"]), float(positive[key]["f"]))

    def test_training_on_universe_writes_template_order_for_both_phases(self):
        metadata = self.train()
        self.assertCountEqual(metadata["stock_ids"], self.stocks)
        for dates in (("2026-05-27", "2026-05-26"), ("2026-05-26",)):
            with self.subTest(dates=dates):
                self.output.write_text("obsolete output\n", encoding="utf-8")
                header, *rows = self.predict(*dates)
                self.assertEqual(header, ["id", *dates])
                self.assertEqual([row[0] for row in rows], self.targets)
                for row in rows:
                    self.assertEqual(len(row), len(header))
                    self.assertTrue(all(value in ("0", "1") for value in row[1:]))


if __name__ == "__main__":
    unittest.main()
