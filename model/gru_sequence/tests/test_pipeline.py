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
from datetime import date, timedelta
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

GLOBAL_HEADER = ("date", "^VIX", "^IXIC", "^GSPC", "^TWII", "^KS11", "TX_OI")


def global_row(day):
    return [f"2026-05-{day:02d}", 15 + day / 3, 10_000 + 37 * day, 4000 + day**2,
            20_000 + 51 * day, 2000 + 13 * day, 10_000 + 100 * day]


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
                       "head_hidden_size": 4, "learning_rate": 0.01, "epochs": 2, "seed": 7,
                       "attention": True, "embedding": True, "embedding_size": 4, "input_noise_std": 0.1}
        self.write_config()
        for data_dir in (self.history, self.update):
            (data_dir / "market").mkdir(parents=True)
            (data_dir / "global").mkdir()
        write_csv(self.global_file(self.history), [GLOBAL_HEADER, *map(global_row, range(18, 23)),
                                                   global_row(26)])
        write_csv(self.global_file(self.update), [GLOBAL_HEADER, *map(global_row, (22, 25, 26, 27))])
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

    def global_file(self, data_dir):
        return data_dir / "global" / "global_data.csv"

    def edit_global(self, data_dir, day, **values):
        path = self.global_file(data_dir)
        header, *rows = read_csv(path)
        row, = (row for row in rows if row[0] == day)
        for col, value in values.items():
            row[header.index(col)] = value
        write_csv(path, [header, *rows])

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
        return result

    def write_config(self):
        settings = {**self.config, "deployment_mode": "standalone"}
        self.config_path.write_text(json.dumps(settings), encoding="utf-8")

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

    def test_same_date_us_session_informs_score_from_update_row(self):
        day = "2026-05-26"
        self.train()
        self.predict(day)
        before = self.details()
        self.assertEqual({row["latest_gspc_date"] for row in before.values()}, {day})

        self.edit_global(self.history, day, **{"^GSPC": 80_000})
        self.predict(day)
        self.assertEqual(self.details(), before)

        self.edit_global(self.update, day, **{"^GSPC": 700})
        self.predict(day)
        self.assertTrue(any(row["f"] != before[key]["f"] for key, row in self.details().items()))

    def test_global_window_uses_lagged_non_us_close_and_us_cutoff(self):
        day = "2026-05-26"
        self.train()
        self.predict(day)
        before = self.details()

        for column, value, source in (("^TWII", 2000, "latest_twii_date"),
                                      ("^KS11", 1000, "latest_ks11_date"),
                                      ("TX_OI", 3000, "latest_tx_oi_date")):
            with self.subTest(column=column):
                # May 25's close is available before predicting May 26.
                self.edit_global(self.update, "2026-05-25", **{column: value})
                self.predict(day)
                after = self.details()
                self.assertTrue(any(row["f"] != before[key]["f"] for key, row in after.items()))
                self.assertEqual({row[source] for row in after.values()}, {"2026-05-25"})
            self.edit_global(self.update, "2026-05-25", **{column: global_row(25)[GLOBAL_HEADER.index(column)]})

        # Move only the May 22 US observation to Saturday: this misses Friday's
        # cutoff but keeps the same values for Monday and for the today vector.
        self.edit_global(self.update, "2026-05-22", **{"^GSPC": ""})
        saturday = ["2026-05-23", "", "", global_row(22)[3], "", "", ""]
        path = self.global_file(self.update)
        write_csv(path, [*read_csv(path), saturday])
        self.predict(day)
        self.assertTrue(any(row["f"] != before[key]["f"] for key, row in self.details().items()))

    def test_us_holiday_matches_carried_levels_and_traces_actual_source_dates(self):
        day = "2026-05-26"
        # Remove one series at an earlier cutoff, then remove the entire global row
        # at today's cutoff in both directories. Taiwan still has a prediction date.
        self.edit_global(self.update, "2026-05-25", **{"^IXIC": ""})
        for data_dir in (self.history, self.update):
            path = self.global_file(data_dir)
            write_csv(path, [row for row in read_csv(path) if row[0] != day])
        self.train()
        self.assertEqual(self.predict(day)[0], ["id", day])
        before = self.details()
        for row in before.values():
            self.assertTrue(math.isfinite(float(row["f"])))
            self.assertEqual({key: row[key] for key in (
                "latest_gspc_date", "latest_ixic_date", "latest_vix_date",
                "latest_twii_date", "latest_ks11_date", "latest_tx_oi_date",
            )}, {
                "latest_gspc_date": "2026-05-25",
                "latest_ixic_date": "2026-05-22",
                "latest_vix_date": "2026-05-25",
                "latest_twii_date": "2026-05-25",
                "latest_ks11_date": "2026-05-25",
                "latest_tx_oi_date": "2026-05-25",
            })

        self.edit_global(self.update, "2026-05-25", **{"^IXIC": global_row(22)[2]})
        path = self.global_file(self.update)
        repeated = [day, global_row(25)[1], global_row(22)[2], global_row(25)[3], "", "", ""]
        write_csv(path, [*read_csv(path), repeated])
        self.predict(day)
        for key, row in self.details().items():
            self.assertEqual(row["f"], before[key]["f"])
            self.assertEqual(row["latest_ixic_date"], day)

    def test_unavailable_stock_and_global_rows_cannot_change_scores(self):
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
            path = self.global_file(data_dir)
            header, *rows = read_csv(path)
            for row in rows:
                columns = ("^TWII", "^KS11", "TX_OI") if row[0] == day else header[1:]
                if row[0] >= day:
                    for column in columns:
                        row[header.index(column)] = rng.uniform(1, 1000)
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
        metadata = self.train()
        self.assertEqual(metadata["last_training_label_date"], "2026-05-25")
        self.assertEqual(metadata["data_date_range"], ["2026-05-18", "2026-05-25"])
        self.assertEqual(metadata["global_data_date_range"], ["2026-05-18", "2026-05-27"])
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
        self.write_config()
        self.assertEqual(self.train()["config"], self.config)
        self.predict(date)
        self.assertNotEqual(self.details(), expected)

    def test_default_input_noise_changes_training_but_inference_is_repeatable(self):
        day = "2026-05-26"
        self.config.pop("input_noise_std")
        self.write_config()
        self.train()
        self.predict(day)
        default = self.details()
        self.predict(day)
        self.assertEqual(self.details(), default)

        self.config["input_noise_std"] = 0.1
        self.write_config()
        self.train()
        self.predict(day)
        self.assertEqual(self.details(), default)

        self.config["input_noise_std"] = 0
        self.write_config()
        self.train()
        self.predict(day)
        self.assertTrue(any(row["f"] != default[key]["f"] for key, row in self.details().items()))

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

    def test_manager_defaults_train_five_seeds_with_the_frozen_selection(self):
        frozen = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
        start = date(2026, 3, 1)
        days = frozen["window_length"] + 1
        rows = [market_row((start + timedelta(days=offset)).isoformat()) for offset in range(days)]
        for stock in self.stocks:
            write_csv(self.market_file(self.history, stock), [MARKET_HEADER, *rows])
            write_csv(self.market_file(self.update, stock), [MARKET_HEADER])
        result = self.run_command(1)
        self.assertNotIn("provisional GRU defaults", result.stderr)
        metadata = json.loads((self.artifacts / "metadata.json").read_text())
        self.assertEqual(metadata["configuration_status"], "selected_default")
        self.assertEqual(metadata["epoch_source"], "validation")
        self.assertEqual(metadata["epoch_selection"], frozen["epoch_selection"])
        self.assertEqual(metadata["config"]["epochs"], frozen["epochs"])
        self.assertEqual(metadata["deployment_seeds"], frozen["deployment_seeds"])
        self.assertEqual(len(set(metadata["deployment_seeds"])), 5)
        prediction_date = (start + timedelta(days=days)).isoformat()
        self.assertEqual(self.predict(prediction_date)[0], ["id", prediction_date])

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
