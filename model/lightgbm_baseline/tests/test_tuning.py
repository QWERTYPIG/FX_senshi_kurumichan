"""Parameter search must rank by validation and leave the test out of trials."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_DIR))
from train import validate_overrides
from tune import rank_trials


class TuningTests(unittest.TestCase):
    def test_ranking_ignores_test_and_breaks_ties_deterministically(self):
        trials = [
            {"trial": 1, "cv_score": 0.01, "cv_score_std": 0.02, "cv_mse": 1, "test_score": 999},
            {"trial": 2, "cv_score": 0.02, "cv_score_std": 0.03, "cv_mse": 1, "test_score": -999},
            {"trial": 3, "cv_score": 0.02, "cv_score_std": 0.01, "cv_mse": 1, "test_score": -999},
        ]
        self.assertEqual([trial["trial"] for trial in rank_trials(trials)], [3, 2, 1])

    def test_rejects_unknown_parameters_and_invalid_values(self):
        for values in ({"num_leave": 7}, {"objective": "binary"}, {"num_leaves": 1},
                       {"learning_rate": -1}, {"min_child_samples": 1.5}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_overrides(values)

    def test_search_exports_selected_model_and_future_training_reuses_parameters(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            live, history, artifacts = root / "live", root / "history", root / "artifacts"
            for source in (live, history):
                (source / "market").mkdir(parents=True)
            dates = pd.bdate_range("2026-08-03", periods=36)
            raw = pd.DataFrame({
                "date": dates, "etl:adj_open": [100 + i for i in range(36)],
                "etl:adj_close": [100 + i + (1 if i % 2 else -1) for i in range(36)],
                "etl:adj_high": [102 + i for i in range(36)], "etl:adj_low": [98 + i for i in range(36)],
                "price:成交股數": [1000 + i for i in range(36)],
            })
            for stock in ("1111", "2222"):
                raw.iloc[:30].to_csv(history / "market" / f"{stock}.csv", index=False)
                raw.iloc[30:].to_csv(live / "market" / f"{stock}.csv", index=False)
            template = root / "template.csv"
            pd.DataFrame({"id": ["2222"], "2026-10-01": [0]}).to_csv(template, index=False)
            grid = root / "grid.json"
            grid.write_text(json.dumps([{}, {"num_leaves": 3, "max_depth": 2, "reg_lambda": 10}]))
            common = ["--data-dir", str(live), "--history-dir", str(history),
                      "--stock-template", str(template), "--artifacts-dir", str(artifacts),
                      "--cv-folds", "2", "--valid-days", "3", "--test-days", "3",
                      "--early-stop-days", "3", "--min-train-days", "10"]

            def run(script, *extra):
                result = subprocess.run([sys.executable, str(MODEL_DIR / script), *common, *extra],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            run("tune.py", "--grid", str(grid))
            search = next((artifacts / "search").iterdir())
            ranking = json.loads((search / "search_results.json").read_text())
            self.assertEqual(len(ranking), 2)
            best = json.loads((artifacts / "best_params.json").read_text())
            self.assertEqual(best["params"], ranking[0]["params"])
            self.assertFalse(best["selection"]["test_used_for_selection"])
            for trial in (search / "trial_01", search / "trial_02"):
                self.assertNotIn("test", json.loads((trial / "metrics.json").read_text()))
                self.assertFalse((trial / "model.txt").exists())
            self.assertTrue((artifacts / "model.txt").exists())
            for stock in ("1111", "2222"):
                changed = raw.iloc[30:].copy()
                changed.loc[changed.index[-3:], "etl:adj_close"] += 10
                changed.to_csv(live / "market" / f"{stock}.csv", index=False)
            run("tune.py", "--grid", str(grid))
            second_search = next(path for path in (artifacts / "search").iterdir() if path != search)
            self.assertEqual(json.loads((second_search / "search_results.json").read_text()), ranking)
            run("train.py")
            metadata = json.loads((artifacts / "metadata.json").read_text())
            self.assertEqual(metadata["parameter_source"], str((artifacts / "best_params.json").resolve()))
            for key, value in best["params"].items():
                if key != "n_estimators":
                    self.assertEqual(metadata["params"][key], value)


if __name__ == "__main__":
    unittest.main()
