import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_pipeline import (
    GLOBAL_HEADER, MARKET_HEADER, MODEL_DIR, global_row, market_row, read_csv, write_csv,
)


class CrossStockTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.history = self.root / "basic_data"
        self.market = self.history / "market"
        self.market.mkdir(parents=True)
        global_dir = self.history / "global"
        global_dir.mkdir()
        write_csv(global_dir / "global_data.csv", [GLOBAL_HEADER, *map(global_row, range(18, 27))])
        self.target, self.peer = "0123", "2330"
        self.prediction_date = "2026-05-26"
        self.days = [f"2026-05-{day:02d}" for day in range(18, 23)] + ["2026-05-25"]
        self.config = {
            "window_length": 3, "hidden_size": 4, "gru_layers": 1,
            "head_hidden_size": 8, "learning_rate": 0.01, "epochs": 2, "seed": 7,
        }
        for stock in (self.target, self.peer):
            write_csv(self.market / f"{stock}.csv", [MARKET_HEADER, *map(market_row, self.days)])
        self.set_universe(self.target, self.peer)

    def set_universe(self, *stocks):
        (self.history / "info.json").write_text(json.dumps({
            "Top100@Y": stocks, "Top50@Y": [self.target],
        }), encoding="utf-8")

    def run_manager(self, index, *args):
        entry = (MODEL_DIR / "usage.txt").read_text(encoding="utf-8").splitlines()[index]
        interpreter, *command = shlex.split(entry)
        self.assertEqual(interpreter, "python")
        result = subprocess.run(
            [sys.executable, *command, "--artifacts-dir", str(self.root / "artifacts"), *args],
            cwd=MODEL_DIR, env={**os.environ, "MANAGER_ROOT": str(self.root),
                                "MANAGER_BASIC_DATA_DIR": str(self.history),
                                "MANAGER_DATA_DIR": str(self.root / "updates"),
                                "MANAGER_TEMPLATE": str(self.root / "template.csv"),
                                "MANAGER_OUTPUT": str(self.root / "prediction.csv")},
            capture_output=True, text=True, timeout=20, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def train(self, **overrides):
        path = self.root / "config.json"
        path.write_text(json.dumps(self.config | overrides), encoding="utf-8")
        self.run_manager(1, "--config", str(path))

    def predict(self, *stocks):
        write_csv(self.root / "template.csv", [["id", self.prediction_date],
                                               *[[stock, ""] for stock in stocks]])
        self.run_manager(0)
        header, *rows = read_csv(self.root / "prediction_detail.csv")
        self.assertEqual([row[header.index("id")] for row in rows], list(stocks))
        scores = [float(row[header.index("f")]) for row in rows]
        self.assertTrue(all(math.isfinite(score) for score in scores))
        return scores

    def test_attention_default_lets_a_non_target_peer_change_target_score(self):
        peer_path = self.market / f"{self.peer}.csv"
        original = read_csv(peer_path)
        changed = [*original[:-1], market_row(self.days[-1], base=1300)]
        for overrides, should_change in (({}, True), ({"attention": False}, False)):
            with self.subTest(config=overrides):
                write_csv(peer_path, original)
                self.train(**overrides)
                before, = self.predict(self.target)
                write_csv(peer_path, changed)
                after, = self.predict(self.target)
                self.assertEqual(after != before, should_change)

    def test_embedding_learns_different_scores_for_identical_inputs(self):
        peer_path = self.market / f"{self.peer}.csv"
        original = read_csv(peer_path)
        opposite_label = original[-1].copy()
        opposite_label[2] = "104"  # adjusted open is above the unchanged close (102)
        opposite_label[3] = "105"
        for attention in (False, True):
            for embedding in (False, True):
                with self.subTest(attention=attention, embedding=embedding):
                    write_csv(peer_path, [*original[:-1], opposite_label])
                    self.train(attention=attention, embedding=embedding, window_length=5, epochs=20)
                    write_csv(peer_path, original)
                    # Reverse the template order relative to the universe.
                    peer_score, target_score = self.predict(self.peer, self.target)
                    if embedding:
                        self.assertLess(peer_score, 0)
                        self.assertGreater(target_score, 0)
                    else:
                        self.assertAlmostEqual(peer_score, target_score, places=6)

    def test_absent_universe_stock_does_not_change_present_stocks_scores(self):
        self.train(embedding=False)
        expected = self.predict(self.target, self.peer)

        absent = "1101"
        self.set_universe(self.target, absent, self.peer)
        path = self.market / f"{absent}.csv"
        history = [market_row(day, base=700) for day in self.days[:3]]
        # There is enough history for a training window, but no corresponding label.
        write_csv(path, [MARKET_HEADER, *history])
        self.train(embedding=False)
        # At prediction the stock is missing a full window.
        write_csv(path, [MARKET_HEADER, *history[1:]])
        self.assertEqual(self.predict(self.target, self.peer), expected)

    def test_joint_model_retraining_reproduces_scores(self):
        self.train()
        expected = self.predict(self.target, self.peer)
        self.train()
        self.assertEqual(self.predict(self.target, self.peer), expected)
