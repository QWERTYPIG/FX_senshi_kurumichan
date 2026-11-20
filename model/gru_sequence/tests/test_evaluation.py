import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_pipeline import GLOBAL_HEADER, MARKET_HEADER, MODEL_DIR, market_row, read_csv, write_csv


class GRUEvaluationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.history = self.root / "basic_data"
        self.artifacts = self.root / "evaluation"
        self.config = self.root / "tiny.json"
        self.targets = ["2330", "0123"]
        self.stocks = ["0123", "1101", "2330"]
        self.days = [f"2026-05-{day}" for day in (18, 19, 20, 21, 22, 25, 26, 27, 28, 29)]
        self.test_days = ["2026-06-01", "2026-06-02"]
        self.settings = {
            "window_length": 3, "hidden_size": 4, "gru_layers": 1, "head_hidden_size": 4,
            "learning_rate": 0.01, "epochs": 2, "seed": 7,
            "evaluation": {
                "folds": [{"start": self.days[6], "end": self.days[7]},
                          {"start": self.days[8], "end": self.days[9]}],
                "early_stopping_days": 2, "patience": 1, "final_test_start": self.test_days[0],
            },
        }
        self.config.write_text(json.dumps(self.settings), encoding="utf-8")
        (self.history / "market").mkdir(parents=True)
        (self.history / "global").mkdir()
        (self.history / "info.json").write_text(json.dumps({"Top100@Y": self.stocks, "Top50@Y": self.targets}),
                                                encoding="utf-8")
        write_csv(self.history / "global" / "global_data.csv", [GLOBAL_HEADER])
        self.write_prices()

    def write_prices(self, test_closes=None):
        for stock, close in zip(self.stocks, (99, 150, 102)):
            rows = []
            for day in self.days + self.test_days:
                actual = test_closes[stock] if test_closes is not None and day in self.test_days else close
                if day in self.days[4:6]:
                    actual = 100  # First inner holdout is flat, so the earliest epoch wins the tie.
                row = market_row(day)
                row[1], row[3] = actual, max(103, actual)
                rows.append(row)
            write_csv(self.history / "market" / f"{stock}.csv", [MARKET_HEADER, *rows])

    def run_evaluation(self, mode, artifacts=None):
        return self.run_cli("evaluate.py", mode, artifacts=artifacts)

    def run_cli(self, *command, artifacts=None):
        return subprocess.run(
            [sys.executable, *command, "--config", str(self.config), "--artifacts-dir", str(artifacts or self.artifacts)],
            cwd=MODEL_DIR, env={**os.environ, "MANAGER_ROOT": str(self.root),
                                "MANAGER_BASIC_DATA_DIR": str(self.history),
                                "MANAGER_DATA_DIR": str(self.root / "update"),
                                "MANAGER_TEMPLATE": str(self.root / "unused.csv")},
            capture_output=True, text=True, timeout=20, check=False,
        )

    def metrics(self, period):
        return json.loads((self.artifacts / f"{period}_metrics.json").read_text(encoding="utf-8"))

    def predictions(self, period):
        with (self.artifacts / f"{period}_predictions.csv").open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def assert_scores(self, metrics, predictions, dates, target_returns):
        self.assertCountEqual([(row["date"], row["id"]) for row in predictions],
                              [(day, stock) for day in dates for stock in self.targets])
        self.assertEqual(metrics["row_count"], len(dates) * 2)
        baseline = sum(target_returns.values()) / 2
        self.assertAlmostEqual(metrics["always_up"], baseline)
        self.assertAlmostEqual(metrics["always_down"], -baseline)
        signed = 0.0
        for row in predictions:
            direction = int(row["direction"])
            self.assertEqual(direction, int(float(row["f"]) > 0))
            self.assertAlmostEqual(float(row["intraday_return"]), target_returns[row["id"]])
            signed += target_returns[row["id"]] if direction else -target_returns[row["id"]]
        self.assertAlmostEqual(metrics["contest_score"], signed / metrics["row_count"])

    def test_validation_reports_expanding_training_and_target_only_scores(self):
        result = self.run_evaluation("validate")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        metrics = self.metrics("validation")
        self.assertEqual(metrics["universe_stock_ids"], self.stocks)
        self.assertEqual(metrics["target_stock_ids"], self.targets)
        self.assertEqual(len(metrics["folds"]), 2)
        predictions = self.predictions("validation")
        for fold, first in zip(metrics["folds"], (6, 8)):
            self.assertEqual(fold["training_dates"], self.days[3:first])
            self.assertEqual(fold["early_stopping_dates"], self.days[first - 2:first])
            self.assertEqual(fold["validation_dates"], self.days[first:first + 2])
            self.assert_scores(fold, [row for row in predictions if row["fold"] == str(fold["fold"])],
                               self.days[first:first + 2], {"2330": 0.02, "0123": -0.01})
        self.assertEqual(len(predictions), 8)
        first, second = (fold["best_epoch"] for fold in metrics["folds"])
        self.assertEqual((first, second), (1, 2))
        self.assertEqual(metrics["refit_epochs"], 1)  # The median is 1.5; floor rather than round.
        self.assertEqual({path.name for path in self.artifacts.iterdir()},
                         {"validation_metrics.json", "validation_predictions.csv"})

    def test_final_test_is_an_explicit_report_for_frozen_config(self):
        result = self.run_evaluation("validate")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        selection = {path: path.read_bytes() for path in self.artifacts.iterdir()}

        # Final reporting can also run independently, with no selection artifacts present.
        self.artifacts = self.root / "final"
        self.settings["epochs"] = 3
        self.config.write_text(json.dumps(self.settings), encoding="utf-8")
        result = self.run_evaluation("final-test")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        metrics = self.metrics("final_test")
        predictions = self.predictions("final_test")
        self.assertEqual(metrics["training_dates"], self.days[3:])
        self.assertEqual(metrics["test_dates"], self.test_days)
        self.assertEqual(metrics["config"]["epochs"], 3)
        self.assert_scores(metrics, predictions, self.test_days, {"2330": 0.02, "0123": -0.01})
        self.assertEqual({path.name for path in self.artifacts.iterdir()},
                         {"final_test_metrics.json", "final_test_predictions.csv"})

        self.write_prices({"0123": 95, "1101": 70, "2330": 104})
        result = self.run_evaluation("final-test")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already exist", result.stderr)
        self.assertEqual(self.metrics("final_test"), metrics)
        self.assertEqual(self.predictions("final_test"), predictions)
        self.artifacts = self.root / "alternative_final"
        result = self.run_evaluation("final-test")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        changed = self.predictions("final_test")
        self.assert_scores(self.metrics("final_test"), changed, self.test_days, {"2330": 0.04, "0123": -0.05})
        self.assertEqual([row["f"] for row in changed if row["date"] == self.test_days[0]],
                         [row["f"] for row in predictions if row["date"] == self.test_days[0]])

        self.settings["epochs"] = 2
        self.config.write_text(json.dumps(self.settings), encoding="utf-8")
        result = self.run_evaluation("validate", self.root / "evaluation")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual({path: path.read_bytes() for path in selection}, selection)

    def test_outer_validation_labels_do_not_pick_epochs(self):
        result = self.run_evaluation("validate")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        first, _ = self.metrics("validation")["folds"]
        day = first["validation_dates"][0]
        before = [row["f"] for row in self.predictions("validation") if row["date"] == day]
        for stock in self.stocks:
            path = self.history / "market" / f"{stock}.csv"
            header, *rows = read_csv(path)
            for row in rows:
                if row[0] == day:
                    row[1], row[3] = 180, 180
            write_csv(path, [header, *rows])

        result = self.run_evaluation("validate")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        changed, _ = self.metrics("validation")["folds"]
        self.assertNotEqual(changed["always_up"], first["always_up"])
        self.assertEqual(changed["best_epoch"], first["best_epoch"])
        self.assertEqual([row["f"] for row in self.predictions("validation") if row["date"] == day], before)

    def test_labeled_target_without_a_window_still_counts_for_all_scores(self):
        path = self.history / "market/0123.csv"
        header, *rows = read_csv(path)
        write_csv(path, [header, next(row for row in rows if row[0] == self.days[-1])])
        result = self.run_evaluation("validate")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        _, last = self.metrics("validation")["folds"]
        rows = [row for row in self.predictions("validation") if row["fold"] == "2"]
        self.assertCountEqual([(row["date"], row["id"]) for row in rows],
                              [("2026-05-28", "2330"), ("2026-05-29", "2330"), ("2026-05-29", "0123")])
        missing_window, = [row for row in rows if row["id"] == "0123"]
        self.assertEqual(missing_window["direction"], "0")
        self.assertEqual(last["row_count"], 3)
        self.assertAlmostEqual(last["always_up"], 0.01)
        self.assertAlmostEqual(last["always_down"], -0.01)

    def test_weekly_fit_ignores_evaluation_settings_and_uses_latest_label(self):
        self.settings["evaluation"]["folds"] = []
        self.config.write_text(json.dumps(self.settings), encoding="utf-8")
        result = self.run_cli("train.py")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual({path.name for path in self.artifacts.iterdir()}, {"model.pt", "metadata.json"})
        metadata = json.loads((self.artifacts / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["config"]["epochs"], 2)
        self.assertEqual(metadata["last_training_label_date"], self.test_days[-1])
