import csv
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_pipeline import GLOBAL_HEADER, MARKET_HEADER, MODEL_DIR, global_row, market_row, read_csv, write_csv


class PredictionSafeguardsTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        self.history = root / "basic_data"
        self.update = root / "data"
        self.template = root / "template.csv"
        self.output = root / "prediction.csv"
        self.detail = root / "prediction_detail.csv"
        self.artifacts = root / "artifacts"
        self.env = {**os.environ, "MANAGER_ROOT": str(root),
                    "MANAGER_BASIC_DATA_DIR": str(self.history), "MANAGER_DATA_DIR": str(self.update),
                    "MANAGER_TEMPLATE": str(self.template), "MANAGER_OUTPUT": str(self.output)}
        self.targets = ["2330", "1101"]
        (self.history / "market").mkdir(parents=True)
        (self.history / "global").mkdir()
        (self.update / "market").mkdir(parents=True)
        (self.history / "info.json").write_text(json.dumps({
            "Top100@Y": self.targets, "Top50@Y": self.targets,
        }), encoding="utf-8")
        write_csv(self.history / "global/global_data.csv", [GLOBAL_HEADER, *map(global_row, range(19, 26))])
        for stock in self.targets:
            write_csv(self.history / "market" / f"{stock}.csv", [MARKET_HEADER, *[
                market_row(f"2026-05-{day:02d}", 100 + day) for day in (19, 20, 21, 22, 25)
            ]])
        config_path = root / "tiny.json"
        config_path.write_text(json.dumps({
            "window_length": 3, "hidden_size": 4, "gru_layers": 1,
            "head_hidden_size": 4, "learning_rate": 0.01, "epochs": 2, "seed": 7,
        }), encoding="utf-8")
        result = self.run_command(1, "--config", str(config_path))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def run_command(self, index, *args):
        entry = (MODEL_DIR / "usage.txt").read_text(encoding="utf-8").splitlines()[index]
        _, *command = shlex.split(entry)
        return subprocess.run(
            [sys.executable, *command, "--artifacts-dir", str(self.artifacts), *args],
            cwd=MODEL_DIR, env=self.env, capture_output=True, text=True, timeout=20, check=False,
        )

    def predict(self, *days):
        write_csv(self.template, [["id", *days], *[[stock, *[""] * len(days)] for stock in self.targets]])
        return self.run_command(0)

    def test_stale_stock_gets_direction_zero_while_ten_day_old_stock_is_scored(self):
        day = "2026-06-04"
        # Replace the stale stock's last row with an invalid update. Its next valid
        # row is from the prediction date and must not make it appear fresh.
        invalid = market_row("2026-05-25")
        invalid[1] = 0
        write_csv(self.update / "market/2330.csv", [MARKET_HEADER, invalid, market_row(day)])
        result = self.predict(day)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        header, stale, fresh = read_csv(self.output)
        self.assertEqual(header, ["id", day])
        self.assertEqual(stale, ["2330", "0"])
        self.assertEqual(fresh[0], "1101")
        with self.detail.open(encoding="utf-8", newline="") as handle:
            details = {row["id"]: row for row in csv.DictReader(handle)}
        self.assertEqual(details["2330"]["f"], "")
        self.assertEqual(details["2330"]["latest_market_date"], "")
        self.assertEqual(details["1101"]["latest_market_date"], "2026-05-25")
        score = float(details["1101"]["f"])
        self.assertTrue(math.isfinite(score))
        self.assertEqual(fresh[1], str(int(score > 0)))
        self.assertIn("WARNING", result.stderr)
        self.assertIn("2330", result.stderr)
        self.assertIn("2026-05-22", result.stderr)
        self.assertNotIn("1101", result.stderr)

        # Removing the stale peer's valid window leaves the same active cross-section.
        path = self.history / "market/2330.csv"
        write_csv(path, [MARKET_HEADER, market_row("2026-05-22")])
        result = self.predict(day)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with self.detail.open(encoding="utf-8", newline="") as handle:
            self.assertEqual({row["id"]: row for row in csv.DictReader(handle)}, details)

    def test_all_stocks_stale_on_one_date_aborts_without_writing_any_output(self):
        result = self.predict("2026-06-04", "2026-06-05")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("2026-06-05", result.stderr)
        self.assertIn("all universe stocks have stale market data", result.stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.detail.exists())

    def test_dates_at_or_before_last_training_label_are_rejected(self):
        for in_sample_day in ("2026-05-25", "2026-05-22"):
            with self.subTest(day=in_sample_day):
                result = self.predict("2026-05-26", in_sample_day)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f"prediction date {in_sample_day}", result.stderr)
                self.assertIn("last training label date 2026-05-25", result.stderr)
                self.assertFalse(self.output.exists())
                self.assertFalse(self.detail.exists())


if __name__ == "__main__":
    unittest.main()
