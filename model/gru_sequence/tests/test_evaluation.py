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
                "early_stopping_days": 2, "patience": 1, "max_epochs": 2, "final_test_start": self.test_days[0],
            },
        }
        self.config.write_text(json.dumps(self.settings), encoding="utf-8")
        self.template = self.root / "template.csv"
        self.output = self.root / "prediction.csv"
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
                    actual = 100  # The first early-stopping split is flat; the earliest epoch wins the tie.
                row = market_row(day)
                row[1], row[3] = actual, max(103, actual)
                rows.append(row)
            write_csv(self.history / "market" / f"{stock}.csv", [MARKET_HEADER, *rows])

    def run_evaluation(self, mode, artifacts=None):
        return self.run_cli("evaluate.py", mode, artifacts=artifacts)

    def run_cli(self, *command, artifacts=None):
        config_args = [] if command[0] == "predict.py" else ["--config", str(self.config)]
        return subprocess.run(
            [sys.executable, *command, *config_args, "--artifacts-dir", str(artifacts or self.artifacts)],
            cwd=MODEL_DIR, env={**os.environ, "MANAGER_ROOT": str(self.root),
                                "MANAGER_BASIC_DATA_DIR": str(self.history),
                                "MANAGER_DATA_DIR": str(self.root / "update"),
                                "MANAGER_TEMPLATE": str(self.template), "MANAGER_OUTPUT": str(self.output)},
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

    def test_validation_epoch_limit_does_not_follow_fixed_refit_epochs(self):
        self.settings["epochs"] = 1
        self.config.write_text(json.dumps(self.settings), encoding="utf-8")
        result = self.run_evaluation("validate")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        metrics = self.metrics("validation")
        self.assertEqual(metrics["config"]["epochs"], 1)
        self.assertEqual(metrics["max_epochs"], 2)
        self.assertEqual([fold["best_epoch"] for fold in metrics["folds"]], [1, 2])

    def test_final_test_is_an_explicit_report_for_frozen_config(self):
        result = self.run_evaluation("validate")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        selection = {path: path.read_bytes() for path in self.artifacts.iterdir()}

        # Final reporting can also run independently, with no selection artifacts present.
        self.artifacts = self.root / "final"
        self.settings.update(epochs=3, deployment_mode="standalone")
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

    def test_final_test_reports_the_frozen_ensemble_not_mean_seed_scores(self):
        seeds = [7, 8, 9, 10, 11]
        self.settings["deployment_seeds"] = seeds
        self.config.write_text(json.dumps(self.settings), encoding="utf-8")
        result = self.run_evaluation("final-test")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        metrics = self.metrics("final_test")
        ensemble = self.predictions("final_test")
        self.assertEqual(metrics["deployment_seeds"], seeds)
        self.assertEqual(metrics["training_dates"], self.days[3:])
        self.assert_scores(metrics, ensemble, self.test_days, {"2330": 0.02, "0123": -0.01})

        self.settings.pop("deployment_seeds")
        self.settings["deployment_mode"] = "standalone"
        runs = []
        for seed in seeds:
            self.settings["seed"] = seed
            self.config.write_text(json.dumps(self.settings), encoding="utf-8")
            self.artifacts = self.root / f"final-{seed}"
            result = self.run_evaluation("final-test")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            runs.append({(row["date"], row["id"]): float(row["f"]) for row in self.predictions("final_test")})
        for row in ensemble:
            scores = [run[row["date"], row["id"]] for run in runs]
            self.assertAlmostEqual(float(row["f"]), sum(scores) / 5, places=7)

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

    def test_search_pairs_ablations_and_freezes_validation_only_selection(self):
        self.settings["search"] = {"window_length": [2, 3], "gru_layers": [1], "hidden_size": [4],
                                   "seeds": [1, 2, 3]}
        self.settings["evaluation"]["max_epochs"] = 6
        self.settings["deployment_seeds"] = [7, 8, 9, 10, 11]
        self.config.write_text(json.dumps(self.settings), encoding="utf-8")
        result = self.run_cli("search.py")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = self.metrics("search")
        trials = {trial["id"]: trial for trial in report["trials"]}
        grid = [trial for trial in trials.values() if trial["stage"] == 1]
        self.assertCountEqual([trial["config"]["window_length"] for trial in grid], [2, 3])
        self.assertTrue(all(trial["config"]["attention"] and trial["config"]["embedding"] for trial in grid))
        self.assertEqual(len(trials), 4)
        winner = trials[report["stage_1_winner"]]
        self.assertEqual(winner["mean_contest_score"], max(t["mean_contest_score"] for t in grid))
        for trial in trials.values():
            self.assertEqual([fold["fold"] for fold in trial["folds"]], [1, 2])
            for fold in trial["folds"]:
                self.assertEqual([run["seed"] for run in fold["runs"]], [1, 2, 3])
                self.assertAlmostEqual(fold["contest_score"], sum(run["contest_score"] for run in fold["runs"]) / 3)
                self.assertAlmostEqual(fold["always_down"], -0.005)
            self.assertAlmostEqual(trial["mean_contest_score"], sum(f["contest_score"] for f in trial["folds"]) / 2)
            for seed in self.settings["search"]["seeds"]:
                path = self.artifacts / trial["id"] / str(seed) / "validation_predictions.csv"
                with path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertCountEqual([(row["date"], row["id"]) for row in rows],
                                      [(day, stock) for day in self.days[6:] for stock in self.targets])
                for fold in trial["folds"]:
                    signs = [float(row["intraday_return"]) * (2 * int(row["direction"]) - 1)
                             for row in rows if row["fold"] == str(fold["fold"])]
                    run = next(run for run in fold["runs"] if run["seed"] == seed)
                    self.assertAlmostEqual(run["contest_score"], sum(signs) / len(signs))
        eligible = []
        self.assertCountEqual([ablation["switch"] for ablation in report["ablations"]], ["attention", "embedding"])
        for ablation in report["ablations"]:
            trial = trials[ablation["trial"]]
            self.assertEqual(ablation["baseline"], winner["id"])
            self.assertEqual(trial["config"], {**winner["config"], ablation["switch"]: False})
            improvements = sum(a["contest_score"] > b["contest_score"] for a, b in zip(trial["folds"], winner["folds"]))
            qualifies = improvements == 2 and trial["mean_contest_score"] > winner["mean_contest_score"]
            self.assertEqual(ablation["improved_fold_count"], improvements)
            self.assertEqual(ablation["gate_eligible"], qualifies)
            self.assertEqual(ablation["adopted"], trial["id"] == report["selected_trial"])
            if qualifies:
                eligible.append(trial)
        selected = max(eligible or [winner], key=lambda trial: trial["mean_contest_score"])
        self.assertEqual(report["selected_trial"], selected["id"])
        frozen = json.loads((self.artifacts / "selected_config.json").read_text())
        self.assertEqual(frozen["deployment_seeds"], [7, 8, 9, 10, 11])
        epoch_selection = frozen["epoch_selection"]
        self.assertEqual(epoch_selection["trial"], report["selected_trial"])
        fold_epochs = []
        for fold, source in zip(selected["folds"], epoch_selection["folds"], strict=True):
            self.assertEqual(source["fold"], fold["fold"])
            self.assertEqual(source["seed_best_epochs"], [
                {"seed": run["seed"], "best_epoch": run["best_epoch"]} for run in fold["runs"]
            ])
            _, expected, _ = sorted(run["best_epoch"] for run in fold["runs"])
            self.assertEqual(source["median_best_epoch"], expected)
            fold_epochs.append(expected)
        self.assertEqual(frozen["epochs"], sum(fold_epochs) // 2)
        pooled = sorted(run["best_epoch"] for fold in selected["folds"] for run in fold["runs"])
        self.assertNotEqual(frozen["epochs"], (pooled[2] + pooled[3]) // 2)
        self.assertEqual(epoch_selection["refit_epochs"], frozen["epochs"])
        self.assertEqual(report["universe_stock_ids"], self.stocks)
        self.assertEqual(report["target_stock_ids"], self.targets)
        self.assertEqual(report["data_date_range"], [self.days[0], self.days[-1]])
        self.assertIn("torch", report["package_versions"])

        snapshot = {path.relative_to(self.artifacts): path.read_bytes()
                    for path in self.artifacts.rglob("*") if path.is_file()}
        self.assertFalse(any("final_test" in str(path) for path in snapshot))
        self.write_prices({"0123": 95, "1101": 70, "2330": 104})
        replay = self.root / "replay"
        result = self.run_cli("search.py", artifacts=replay)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(snapshot, {path.relative_to(replay): path.read_bytes()
                                    for path in replay.rglob("*") if path.is_file()})

    def test_weekly_records_selected_epoch_provenance_even_after_a_fixed_override(self):
        self.settings["search"] = {"window_length": [3], "gru_layers": [1], "hidden_size": [4],
                                   "seeds": [7, 8, 9]}
        self.settings["deployment_seeds"] = [7, 8, 9, 10, 11]
        self.config.write_text(json.dumps(self.settings), encoding="utf-8")
        result = self.run_cli("search.py")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.config = self.artifacts / "selected_config.json"
        frozen = json.loads(self.config.read_text())
        for override in (False, True):
            with self.subTest(override=override):
                settings = {**frozen, "epochs": frozen["epochs"] + int(override)}
                self.config.write_text(json.dumps(settings), encoding="utf-8")
                weekly = self.root / "weekly"
                result = self.run_cli("train.py", artifacts=weekly)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                metadata = json.loads((weekly / "metadata.json").read_text())
                self.assertEqual(metadata["data_date_range"], [self.days[0], self.test_days[-1]])
                self.assertEqual(metadata["epoch_selection"], frozen["epoch_selection"])
                self.assertEqual(metadata["epoch_source"], "fixed_override" if override else "validation")
                self.assertEqual(metadata["config"]["epochs"], settings["epochs"])

    def test_search_adopts_only_the_higher_scoring_variant_when_both_qualify(self):
        self.settings["search"] = {"window_length": [2], "gru_layers": [1], "hidden_size": [4],
                                   "seeds": [1, 17, 38]}
        self.settings["deployment_seeds"] = [7, 8, 9, 10, 11]
        self.config.write_text(json.dumps(self.settings), encoding="utf-8")
        result = self.run_cli("search.py")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = self.metrics("search")
        self.assertEqual([(a["switch"], a["gate_eligible"], a["adopted"]) for a in report["ablations"]],
                         [("attention", True, False), ("embedding", True, True)])
        self.assertEqual(report["selected_trial"], "no-embedding")
        self.assertTrue(report["selected_config"]["attention"])
        self.assertFalse(report["selected_config"]["embedding"])

    def test_weekly_restores_five_seed_models_and_averages_f_before_sign(self):
        seeds = [7, 8, 9, 10, 11]
        self.settings.update(deployment_seeds=seeds, search={"not_a_valid_search_grid": True})
        self.settings["evaluation"]["folds"] = []
        self.config.write_text(json.dumps(self.settings), encoding="utf-8")
        write_csv(self.template, [["id", "2026-06-03"], *[[stock, ""] for stock in self.targets]])
        result = self.run_cli("train.py")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        metadata = json.loads((self.artifacts / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["deployment_seeds"], seeds)
        self.assertEqual(metadata["config"]["epochs"], 2)
        self.assertEqual(metadata["epoch_source"], "fixed")
        self.assertIsNone(metadata["epoch_selection"])
        original_files = {path.relative_to(self.artifacts) for path in self.artifacts.rglob("*")}
        result = self.run_cli("train.py")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual({path.relative_to(self.artifacts) for path in self.artifacts.rglob("*")}, original_files)
        self.assertEqual(metadata["stock_ids"], self.stocks)
        self.assertEqual(metadata["first_training_label_date"], self.days[3])
        self.assertEqual(metadata["last_training_label_date"], self.test_days[-1])
        self.assertIn("torch", metadata["package_versions"])
        self.assertEqual(list(self.artifacts.rglob("search_metrics.json")), [])

        def predict(artifacts):
            result = self.run_cli("predict.py", artifacts=artifacts)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            _, *rows = read_csv(self.output.with_name("prediction_detail.csv"))
            return {row[0]: float(row[2]) for row in rows}

        # The readable sidecar is not part of the prediction snapshot.
        (self.artifacts / "metadata.json").unlink()
        actual = predict(self.artifacts)
        _, *directions = read_csv(self.output)
        self.settings.pop("deployment_seeds")
        self.settings["deployment_mode"] = "standalone"
        single_scores = []
        for seed in seeds:
            self.settings["seed"] = seed
            self.config.write_text(json.dumps(self.settings), encoding="utf-8")
            artifacts = self.root / f"weekly-{seed}"
            result = self.run_cli("train.py", artifacts=artifacts)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            single_scores.append(predict(artifacts))
        self.assertGreater(len({scores[self.targets[0]] for scores in single_scores}), 1)
        for stock, direction in directions:
            expected = sum(scores[stock] for scores in single_scores) / 5
            self.assertAlmostEqual(actual[stock], expected, places=7)
            self.assertEqual(direction, str(int(expected > 0)))

    def test_weekly_requires_five_seeds_unless_standalone_is_explicit(self):
        result = self.run_cli("train.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("deployment_seeds must contain exactly 5 seeds", result.stderr)
        self.assertFalse(self.artifacts.exists())

        self.settings["deployment_mode"] = "standalone"
        self.config.write_text(json.dumps(self.settings), encoding="utf-8")
        result = self.run_cli("train.py")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        metadata = json.loads((self.artifacts / "metadata.json").read_text())
        self.assertEqual(metadata["deployment_mode"], "standalone")
        self.assertEqual(metadata["deployment_seeds"], [self.settings["seed"]])

    def test_weekly_fit_ignores_evaluation_settings_and_uses_latest_label(self):
        self.settings["deployment_mode"] = "standalone"
        self.settings["evaluation"]["folds"] = []
        self.config.write_text(json.dumps(self.settings), encoding="utf-8")
        result = self.run_cli("train.py")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual({path.name for path in self.artifacts.iterdir()}, {"checkpoint.pt", "metadata.json"})
        metadata = json.loads((self.artifacts / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["config"]["epochs"], 2)
        self.assertEqual(metadata["last_training_label_date"], self.test_days[-1])
