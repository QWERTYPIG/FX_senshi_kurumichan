"""Compare equal-weight parameter blends with chronological member selection."""
from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model/lightgbm_baseline"))
from features import FEATURES, FEATURE_VERSION, build_dataset, discover_stock_ids, read_template
from train import BASE_PARAMS, evaluate, prediction_rows
from tune import input_digest


def aligned_predictions(search, trials):
    first, predictions = None, {}
    for trial in trials:
        frame = pd.read_csv(search / f"trial_{trial['trial']:02d}/validation_predictions.csv",
                            dtype={"id": str}).sort_values(["date", "id"]).reset_index(drop=True)
        if frame.duplicated(["date", "id"]).any():
            raise ValueError("Duplicate stock-date rows in cached predictions")
        if first is None:
            first = frame
        else:
            pd.testing.assert_frame_equal(first[["date", "id", "fold", "actual_return"]],
                                          frame[["date", "id", "fold", "actual_return"]])
        predictions[trial["trial"]] = frame["predicted_return"].to_numpy()
    return first, predictions


def rank_on_past(frame, predictions, before_fold):
    """Select using earlier OOF folds only, including deterministic tie breaking."""
    mask = frame["fold"].to_numpy() < before_fold
    if not mask.any():
        raise ValueError("Need earlier validation folds to rank members")
    ranked = []
    for trial, values in predictions.items():
        returns = frame.loc[mask, "actual_return"].to_numpy()
        contribution = (2 * (values[mask] > 0) - 1) * returns
        folds = pd.DataFrame({"fold": frame.loc[mask, "fold"], "score": contribution}).groupby("fold")["score"].mean()
        ranked.append((trial, float(folds.mean()), float(folds.std(ddof=0)),
                       float(np.mean((values[mask] - returns) ** 2))))
    return [item[0] for item in sorted(ranked, key=lambda item: (-item[1], item[2], item[3], item[0]))]


def chronological_blends(frame, predictions, sizes, warmup_folds=2):
    folds = sorted(frame["fold"].unique())
    if len(folds) <= warmup_folds:
        raise ValueError("Need more validation folds than warm-up folds")
    records, outputs = {}, {}
    for size in sizes:
        result = np.full(len(frame), np.nan)
        fold_records = []
        for fold in folds[warmup_folds:]:
            members = rank_on_past(frame, predictions, int(fold))[:size]
            mask = frame["fold"].to_numpy() == fold
            result[mask] = np.mean([predictions[trial][mask] for trial in members], axis=0)
            target = frame.loc[mask].rename(columns={"actual_return": "intraday_return"}).copy()
            target["date"] = pd.to_datetime(target["date"])
            fold_records.append({"fold": int(fold), "members": members,
                                 "metrics": evaluate(target, result[mask])})
        mask = np.isfinite(result)
        target = frame.loc[mask].rename(columns={"actual_return": "intraday_return"}).copy()
        target["date"] = pd.to_datetime(target["date"])
        scores = [record["metrics"]["score"] for record in fold_records]
        records[size] = {
            "metrics": evaluate(target, result[mask]), "folds": fold_records,
            "fold_mean_score": float(np.mean(scores)), "fold_score_std": float(np.std(scores)),
        }
        outputs[size] = result
    return records, outputs


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--search-run", type=Path)
    parser.add_argument("--warmup-folds", type=int, default=2)
    args = parser.parse_args()
    if args.warmup_folds < 1:
        parser.error("--warmup-folds must be positive")
    if args.search_run is None:
        best = json.loads((ROOT / "model/lightgbm_baseline/artifacts/best_params.json").read_text())
        args.search_run = Path(best["selection"]["run"])
    search = args.search_run
    configuration = json.loads((search / "configuration.json").read_text())
    source_args = Namespace(**configuration["arguments"])
    for key in ("grid", "data_dir", "history_dir", "stock_template", "artifacts_dir"):
        setattr(source_args, key, Path(getattr(source_args, key)))
    if input_digest(source_args) != configuration["input_sha256"]:
        raise ValueError("Search inputs/code changed: rerun tune.py on current data before blending")
    trials = json.loads((search / "search_results.json").read_text())
    if len(trials) < 5:
        raise ValueError("Need at least five searched parameter configurations")
    output = Path(__file__).resolve().parent / "results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True)
    sizes = sorted(set([1, 3, 5, len(trials)]))
    (output / "configuration.json").write_text(json.dumps({
        "search_run": str(search.resolve()), "input_sha256": configuration["input_sha256"],
        "sizes": sizes, "warmup_folds": args.warmup_folds, "weights": "equal, average raw predicted returns",
        "selection": "Choose best blend size by chronological validation fold mean score; ties use smaller std then smaller size. "
                     "Compare with chronological top-1. Final-test results never select size or weights.",
        "limitations": "The parameter grid and test dates were explored previously; this is not a fresh blind experiment.",
    }, indent=2) + "\n")
    frame, predictions = aligned_predictions(search, trials)
    records, outputs = chronological_blends(frame, predictions, sizes, args.warmup_folds)
    best_size = min([size for size in sizes if size > 1],
                    key=lambda size: (-records[size]["fold_mean_score"], records[size]["fold_score_std"], size))
    full_ranking = rank_on_past(frame, predictions, int(frame["fold"].max()) + 1)
    members = full_ranking[:best_size]
    selection = {
        "selected_blend_size": best_size, "final_members": members,
        "validation_improved_over_top1": records[best_size]["fold_mean_score"] > records[1]["fold_mean_score"],
        "chronological_validation": records,
    }
    # Save the choice BEFORE loading or evaluating final-test predictions.
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    for size in sizes:
        print(f"Top-{size}: chronological score={records[size]['fold_mean_score']:.9f}, "
              f"std={records[size]['fold_score_std']:.9f}", flush=True)
    print(f"Selected equal-weight Top-{best_size}; fitting members {members}", flush=True)
    details = frame[["fold", "date", "id", "actual_return"]].copy()
    for size, values in outputs.items():
        details[f"predicted_top_{size}"] = values
    details.to_csv(output / "chronological_predictions.csv", index=False)
    correlation = pd.DataFrame(predictions).corr()
    correlation.to_csv(output / "prediction_correlations.csv")
    if len(correlation) > 1:
        mean_correlation = float(correlation.to_numpy()[np.triu_indices(len(correlation), k=1)].mean())
    else:
        mean_correlation = 1.0

    stock_ids = discover_stock_ids(source_args.data_dir, source_args.history_dir)
    targets = read_template(source_args.stock_template)["id"].tolist()
    data = build_dataset(source_args.data_dir, stock_ids, history_dir=source_args.history_dir)
    data = data[data["feature_date"].notna() & data["intraday_return"].notna()].copy()
    # Use the cached effective cutoff even when the original search had no --train-end.
    cached_metadata = json.loads((search / "selected_model/metadata.json").read_text())
    data = data[data["date"] <= pd.Timestamp(cached_metadata["trained_through"])]
    test_start = pd.Timestamp(frame["date"].max())
    training = data[data["date"] <= test_start]
    test = data[(data["date"] > test_start) & data["stock_id"].isin(targets)]
    model_dir = output / "blend_model"
    model_dir.mkdir()
    test_predictions, member_records = [], []
    trial_lookup = {trial["trial"]: trial for trial in trials}
    for trial_id in members:
        candidate = trial_lookup[trial_id]
        params = {**BASE_PARAMS, **candidate["params"], "n_estimators": candidate["best_iteration"]}
        evaluation_model = lgb.LGBMRegressor(**params)
        evaluation_model.fit(training[FEATURES], training["intraday_return"], categorical_feature=["stock_id"])
        test_predictions.append(evaluation_model.predict(test[FEATURES]))
        final = lgb.LGBMRegressor(**params)
        final.fit(data[FEATURES], data["intraday_return"], categorical_feature=["stock_id"])
        filename = f"member_{trial_id:02d}.txt"
        final.booster_.save_model(str(model_dir / filename))
        member_records.append({"trial": trial_id, "model_file": filename,
                               "weight": 1 / best_size, "params": params})
    predicted = np.mean(test_predictions, axis=0)
    blend_test = evaluate(test, predicted)
    prediction_rows(test, predicted).to_csv(output / "blend_test_predictions.csv", index=False)
    # Reference is the same currently selected single model, not its full-history fit.
    single_test_frame = pd.read_csv(search / "selected_model/test_predictions.csv", dtype={"id": str})
    expected = prediction_rows(test, predicted)
    pd.testing.assert_frame_equal(expected[["date", "id"]].assign(date=lambda x: x["date"].astype(str)).reset_index(drop=True),
                                  single_test_frame[["date", "id"]].reset_index(drop=True))
    np.testing.assert_allclose(test["intraday_return"].to_numpy(), single_test_frame["actual_return"].to_numpy())
    reference = evaluate(test, single_test_frame["predicted_return"].to_numpy())
    metadata = {
        "model_type": "equal_weight_blend", "members": member_records,
        "feature_version": FEATURE_VERSION, "features": FEATURES, "stock_ids": stock_ids,
        "prediction_stock_ids": targets, "threshold": 0,
        "trained_through": str(data["date"].max().date()),
        "selection_source": str((output / "selection.json").resolve()),
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    result = {**selection, "test_blend": blend_test, "test_single": reference,
              "mean_member_prediction_correlation": mean_correlation,
              "input_sha256": configuration["input_sha256"],
              "trained_through": metadata["trained_through"],
              "production_changed": False}
    if input_digest(source_args) != configuration["input_sha256"]:
        raise RuntimeError("Data/code changed during blending; results are invalid, rerun the experiment")
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Test single={reference['score']:.9f}, blend={blend_test['score']:.9f}. Results: {output}", flush=True)


if __name__ == "__main__":
    main()
