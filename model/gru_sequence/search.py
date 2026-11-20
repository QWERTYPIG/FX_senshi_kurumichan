"""Select a frozen GRU configuration using walk-forward validation only."""

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import asdict, replace
from itertools import product
from multiprocessing import get_context
from operator import itemgetter
from pathlib import Path
from statistics import median

from evaluate import EvaluationPlan, evaluate_validation, refit_epoch_count, write_report
from runtime import MODEL_DIR, SEARCH_SEED_COUNT, Config, ManagerPaths, deployment_seeds, read_seeds

GRID_STAGE, ABLATION_STAGE = 1, 2
GRID_FIELDS = ("window_length", "gru_layers", "hidden_size")
FOLD_FIELDS = ("fold", "start", "end", "validation_dates", "row_count", "always_up", "always_down",
               "all_market_stale_dates")


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


def run_seed(paths, plan, directory, trial_id, config):
    metrics, predictions = evaluate_validation(paths, config, plan)
    write_report(directory / trial_id / str(config.seed), "validation", metrics, predictions, mode="x")
    return metrics


def run_stage(pool, paths, plan, seeds, directory, stage, candidates):
    jobs = {(trial_id, seed): (paths, plan, directory, trial_id, replace(config, seed=seed))
            for trial_id, config in candidates for seed in seeds}
    if pool is None:
        results = {key: run_seed(*args) for key, args in jobs.items()}
    else:
        futures = {pool.submit(run_seed, *args): key for key, args in jobs.items()}
        try:
            results = {futures[future]: future.result() for future in as_completed(futures)}
        finally:
            for future in futures:
                future.cancel()
    # Keep configured candidate and seed order no matter which worker finishes first.
    return [aggregate_trial(trial_id, stage, config, seeds, results) for trial_id, config in candidates]


def aggregate_trial(trial_id, stage, config, seeds, results):
    folds = {}
    for seed in seeds:
        metrics = results[trial_id, seed]
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
    parser.add_argument("--workers", type=int, default=1, help="number of parallel CPU workers (default: 1)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be a positive integer")
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

    rank = itemgetter("mean_contest_score")
    switches = ("attention", "embedding")
    # Spawn fresh processes instead of forking a process that has imported PyTorch.
    executor = ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) if args.workers > 1 else nullcontext()
    with executor as pool:
        grid_candidates = [(f"grid-{number}", config) for number, config in enumerate(candidates, start=1)]
        grid_results = run_stage(pool, paths, plan, seeds, directory, GRID_STAGE, grid_candidates)
        trials = [trial for trial, _ in grid_results]
        metadata = grid_results[-1][1]
        winner = max(trials, key=rank)
        winning_config = Config(**winner["config"])
        ablation_candidates = [(f"no-{switch}", replace(winning_config, **{switch: False})) for switch in switches]
        ablation_results = run_stage(pool, paths, plan, seeds, directory, ABLATION_STAGE, ablation_candidates)

    eligible, ablations = [], []
    for switch, (trial, _) in zip(switches, ablation_results, strict=True):
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
