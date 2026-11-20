"""Select hyperparameters using walk-forward score, then train one selected model."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import pandas as pd

from features import DATA_DIR, HISTORY_DIR, FEATURES, FEATURE_VERSION, MODEL_DIR, ROOT
from train import BASE_PARAMS, validate_overrides


def rank_trials(trials):
    # No test field is ever consulted. Stable and explicit tie-breaking.
    return sorted(trials, key=lambda trial: (
        -trial["cv_score"], trial["cv_score_std"], trial["cv_mse"], trial["trial"],
    ))


def input_digest(args):
    paths = [args.grid, args.stock_template, MODEL_DIR / "features.py",
             MODEL_DIR / "train.py", MODEL_DIR / "validation.py", MODEL_DIR / "tune.py",
             *args.data_dir.glob("market/*.csv"), *args.history_dir.glob("market/*.csv")]
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.resolve()).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=MODEL_DIR / "parameter_grid.json")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--history-dir", type=Path, default=HISTORY_DIR)
    parser.add_argument("--stock-template", type=Path, default=ROOT / "model/all_down/all_down.csv")
    parser.add_argument("--artifacts-dir", type=Path, default=MODEL_DIR / "artifacts")
    parser.add_argument("--train-end")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--valid-days", type=int, default=20)
    parser.add_argument("--test-days", type=int, default=20)
    parser.add_argument("--early-stop-days", type=int, default=20)
    parser.add_argument("--min-train-days", type=int, default=252)
    args = parser.parse_args()
    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    if not isinstance(grid, list) or not grid:
        parser.error("--grid must contain a nonempty JSON list of parameter objects")
    grid = [validate_overrides(candidate) for candidate in grid]
    # Include the original baseline in every search, for a direct comparison.
    if not any({**BASE_PARAMS, **candidate} == BASE_PARAMS for candidate in grid):
        grid.insert(0, {})
    unique = {}
    for candidate in grid:
        unique.setdefault(tuple(sorted({**BASE_PARAMS, **candidate}.items())), candidate)
    grid = list(unique.values())
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    folder = args.artifacts_dir / "search" / run_id
    folder.mkdir(parents=True)
    before = input_digest(args)
    configuration = {
        "input_sha256": before, "features": FEATURES, "feature_version": FEATURE_VERSION,
        "grid": grid, "arguments": {key: str(value) if isinstance(value, Path) else value
                                    for key, value in vars(args).items()},
        "ranking": "CV fold mean score descending, fold score std ascending, pooled MSE ascending, trial order",
        "test_policy": "No trial evaluates test. Only the selected configuration evaluates it once.",
    }
    (folder / "configuration.json").write_text(json.dumps(configuration, indent=2) + "\n")
    common = [
        "--data-dir", str(args.data_dir.resolve()), "--history-dir", str(args.history_dir.resolve()),
        "--stock-template", str(args.stock_template.resolve()), "--cv-folds", str(args.cv_folds),
        "--valid-days", str(args.valid_days), "--test-days", str(args.test_days),
        "--early-stop-days", str(args.early_stop_days), "--min-train-days", str(args.min_train_days),
    ]
    if args.train_end:
        common += ["--train-end", args.train_end]

    def train(target, parameter_path, validation_only=False):
        command = [sys.executable, "-u", str(MODEL_DIR / "train.py"), *common,
                   "--params", str(parameter_path.resolve()), "--artifacts-dir", str(target.resolve())]
        if validation_only:
            command.append("--validation-only")
        with target.with_suffix(".log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        if input_digest(args) != before:
            raise RuntimeError("Data or code changed during tuning; no model promoted. Rerun on a stable snapshot.")
        return json.loads((target / "metrics.json").read_text())

    trials = []
    for index, overrides in enumerate(grid, 1):
        candidate = folder / f"trial_{index:02d}"
        parameter_path = folder / f"trial_{index:02d}_params.json"
        parameter_path.write_text(json.dumps(overrides, indent=2) + "\n")
        print(f"Trial {index}/{len(grid)}: {overrides}", flush=True)
        metrics = train(candidate, parameter_path, validation_only=True)
        assert "test" not in metrics
        summary = metrics["cross_validation"]["fold_summary"]
        trials.append({
            "trial": index, "params": overrides, "cv_score": summary["score"]["mean"],
            "cv_score_std": summary["score"]["std"], "cv_mse": metrics["validation"]["mse"],
            "cv_accuracy": metrics["validation"]["accuracy"], "best_iteration": metrics["best_iteration"],
            "fold_scores": [fold["validation"]["score"] for fold in metrics["cross_validation"]["folds"]],
        })
        ranking = rank_trials(trials)
        (folder / "search_results.json").write_text(json.dumps(ranking, indent=2) + "\n")
        pd.json_normalize(ranking).to_csv(folder / "search_results.csv", index=False)
        print(f"  CV score={trials[-1]['cv_score']:.9f}, std={trials[-1]['cv_score_std']:.9f}", flush=True)
    best = rank_trials(trials)[0]
    selected = folder / "selected_model"
    selected.mkdir()
    best_path = selected / "best_params.json"
    export = {
        "params": best["params"], "selection": {
            "run": str(folder.resolve()), "input_sha256": before, "trial": best["trial"],
            "trial_count": len(trials), "cv_score": best["cv_score"],
            "rule": configuration["ranking"], "test_used_for_selection": False,
            "note": "Selected CV scores are optimistic after parameter search; use test and future dates for evaluation.",
        },
    }
    best_path.write_text(json.dumps(export, indent=2) + "\n")
    print(f"Selected trial {best['trial']}; fitting/evaluating the selected model only.", flush=True)
    selected_metrics = train(selected, best_path)
    assert selected_metrics["best_iteration"] == best["best_iteration"]
    # Preserve existing artifacts, then publish only a completely trained model.
    backup = folder / "previous_model"
    backup.mkdir()
    for source in selected.iterdir():
        target = args.artifacts_dir / source.name
        if target.is_file():
            shutil.copy2(target, backup / target.name)
        shutil.copy2(source, target)
    (folder / "summary.json").write_text(json.dumps({
        "selected": best, "test": selected_metrics["test"], "artifacts_dir": str(args.artifacts_dir.resolve()),
    }, indent=2) + "\n")
    print(f"Saved selected model and best_params.json to {args.artifacts_dir}. Search records: {folder}", flush=True)


if __name__ == "__main__":
    main()
