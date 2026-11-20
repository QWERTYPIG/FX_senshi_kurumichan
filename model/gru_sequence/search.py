"""Select a frozen GRU configuration using walk-forward validation only."""

import argparse
import json
import math
from dataclasses import asdict, replace
from itertools import product
from operator import itemgetter
from pathlib import Path
from statistics import median

from evaluate import EvaluationPlan, evaluate_validation, refit_epoch_count, write_report
from runtime import MODEL_DIR, SEARCH_SEED_COUNT, Config, ManagerPaths, deployment_seeds, read_seeds

GRID_STAGE, ABLATION_STAGE = 1, 2
GRID_FIELDS = ("window_length", "gru_layers", "hidden_size")
FOLD_FIELDS = ("fold", "start", "end", "validation_dates", "row_count", "always_up", "always_down")


def grid(settings, base):
    axes = []
    for field in GRID_FIELDS:
        values = settings[field]
        if (not isinstance(values, list) or not values
                or any(type(value) is not int or value < 1 for value in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"search.{field} must contain distinct positive integers")
        axes.append(values)
    return [replace(base, attention=True, embedding=True, **dict(zip(GRID_FIELDS, values)))
            for values in product(*axes)]


def run_trial(paths, plan, seeds, directory, trial_id, stage, config):
    folds = {}
    for seed in seeds:
        metrics, predictions = evaluate_validation(paths, replace(config, seed=seed), plan)
        write_report(directory / trial_id / str(seed), "validation", metrics, predictions, mode="x")
        for fold in metrics["folds"]:
            identity = {field: fold[field] for field in FOLD_FIELDS}
            aggregate = folds.setdefault(fold["fold"], {**identity, "runs": []})
            if any(aggregate[field] != identity[field] for field in FOLD_FIELDS):
                raise ValueError(f"{trial_id}: validation fold or scoring rows changed between seeds")
            aggregate["runs"].append({"seed": seed, "best_epoch": fold["best_epoch"],
                                      "contest_score": fold["contest_score"]})
    for fold in folds.values():
        fold["contest_score"] = math.fsum(run["contest_score"] for run in fold["runs"]) / len(seeds)
        fold["median_best_epoch"] = median(run["best_epoch"] for run in fold["runs"])
    trial = {"id": trial_id, "stage": stage, "config": asdict(config), "folds": list(folds.values()),
             "mean_contest_score": math.fsum(fold["contest_score"] for fold in folds.values()) / len(folds),
             "refit_epochs": refit_epoch_count(fold["median_best_epoch"] for fold in folds.values())}
    metadata = {key: metrics[key] for key in ("universe_stock_ids", "target_stock_ids", "data_date_range",
                                               "global_data_date_range", "package_versions")}
    return trial, metadata


def compare_ablation(baseline, trial, switch):
    baseline_folds = {fold["fold"]: fold for fold in baseline["folds"]}
    trial_folds = {fold["fold"]: fold for fold in trial["folds"]}
    if baseline_folds.keys() != trial_folds.keys():
        raise ValueError("ablation did not score the same folds as its baseline")
    deltas = []
    for number, before in baseline_folds.items():
        after = trial_folds[number]
        if (any(before[field] != after[field] for field in FOLD_FIELDS)
                or {run["seed"] for run in before["runs"]} != {run["seed"] for run in after["runs"]}):
            raise ValueError(f"ablation fold {number} does not match the baseline's scoring rows or seeds")
        deltas.append({"fold": number, "contest_score_delta": after["contest_score"] - before["contest_score"]})
    improved = sum(delta["contest_score_delta"] > 0 for delta in deltas)
    mean_delta = trial["mean_contest_score"] - baseline["mean_contest_score"]
    return {"switch": switch, "baseline": baseline["id"], "trial": trial["id"], "fold_deltas": deltas,
            "mean_contest_score_delta": mean_delta, "improved_fold_count": improved,
            "gate_eligible": mean_delta > 0 and 2 * improved > len(deltas)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=MODEL_DIR / "config.json")
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    base = Config.from_settings(settings)
    candidates = grid(settings["search"], base)
    seeds = read_seeds(settings["search"]["seeds"], SEARCH_SEED_COUNT, "search.seeds")
    deployment_seeds(settings, base, allow_standalone=False)
    plan = EvaluationPlan.from_settings(settings["evaluation"])
    paths = ManagerPaths.read()
    directory = args.artifacts_dir
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        parser.error("--artifacts-dir must be empty for a new search")

    trials = []
    for number, config in enumerate(candidates, start=1):
        trial, metadata = run_trial(paths, plan, seeds, directory, f"grid-{number}", GRID_STAGE, config)
        trials.append(trial)
    rank = itemgetter("mean_contest_score")
    winner = max(trials, key=rank)

    eligible, ablations = [], []
    for switch in ("attention", "embedding"):
        config = replace(Config(**winner["config"]), **{switch: False})
        trial, _ = run_trial(paths, plan, seeds, directory, f"no-{switch}", ABLATION_STAGE, config)
        trials.append(trial)
        decision = compare_ablation(winner, trial, switch)
        ablations.append(decision)
        if decision["gate_eligible"]:
            eligible.append(trial)
    selected = max(eligible, key=rank) if eligible else winner
    for decision in ablations:
        decision["adopted"] = decision["trial"] == selected["id"]
    epoch_selection = {
        "trial": selected["id"],
        "refit_epochs": selected["refit_epochs"],
        "folds": [
            {**{key: fold[key] for key in ("fold", "start", "end", "median_best_epoch")},
             "seed_best_epochs": [{"seed": run["seed"], "best_epoch": run["best_epoch"]} for run in fold["runs"]]}
            for fold in selected["folds"]
        ],
    }
    frozen = {**settings, **selected["config"], "epochs": selected["refit_epochs"], "epoch_selection": epoch_selection}
    report = {**metadata, "config": settings, "trials": trials, "stage_1_winner": winner["id"],
              "ablations": ablations, "selected_trial": selected["id"], "selected_config": frozen}
    for name, payload in (("search_metrics.json", report), ("selected_config.json", frozen)):
        with (directory / name).open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
