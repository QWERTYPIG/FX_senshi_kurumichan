"""Walk-forward regression validation, untouched final test, full-history deployment."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import (DATA_DIR, HISTORY_DIR, FEATURES, FEATURE_VERSION, MODEL_DIR, ROOT,
                      build_dataset, discover_stock_ids, read_template)
from validation import select_iterations, walk_forward_splits
from us_features import US_DATA_DIR, US_FEATURES, US_FEATURE_VERSION, add_us_features, us_coverage

BASE_PARAMS = dict(
    objective="regression", n_estimators=500, learning_rate=0.03,
    num_leaves=7, max_depth=4, min_child_samples=50,
    reg_lambda=1.0, random_state=42, n_jobs=4, verbosity=-1,
    deterministic=True, force_col_wise=True,
)
TUNABLE_PARAMS = {
    "n_estimators", "learning_rate", "num_leaves", "max_depth", "min_child_samples",
    "reg_alpha", "reg_lambda", "colsample_bytree", "subsample", "subsample_freq",
}


def validate_overrides(overrides):
    if not isinstance(overrides, dict) or overrides.keys() - TUNABLE_PARAMS:
        raise ValueError(f"Parameter overrides must be an object using only {sorted(TUNABLE_PARAMS)}")
    for name, value in overrides.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        if name in {"n_estimators", "num_leaves", "max_depth", "min_child_samples", "subsample_freq"}:
            if not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if name in {"n_estimators", "learning_rate", "min_child_samples"} and value <= 0:
            raise ValueError(f"{name} must be positive")
        if name == "num_leaves" and value < 2:
            raise ValueError("num_leaves must be at least 2")
        if name in {"reg_alpha", "reg_lambda", "subsample_freq"} and value < 0:
            raise ValueError(f"{name} must not be negative")
        if name in {"colsample_bytree", "subsample"} and not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    return overrides


def evaluate(frame, predicted_returns):
    actual_returns = frame["intraday_return"].to_numpy()

    prediction = (predicted_returns > 0).astype(int)

    contribution = (
        (2 * prediction - 1) * actual_returns
    )

    return {
        "rows": len(frame),
        "dates": int(frame["date"].nunique()),
        "start": str(frame["date"].min().date()),
        "end": str(frame["date"].max().date()),
        "accuracy": float(
            np.mean(prediction == (actual_returns > 0))
        ),
        "score": float(contribution.mean()),
        "mse": float(
            np.mean((predicted_returns - actual_returns) ** 2)
        ),
        "mae": float(
            np.mean(np.abs(predicted_returns - actual_returns))
        ),
        "all_up_score": float(actual_returns.mean()),
        "all_down_score": float(-actual_returns.mean()),
    }


def prediction_rows(frame, predicted_returns, fold=None):
    result = pd.DataFrame({
        "date": frame["date"].to_numpy(), "id": frame["stock_id"].astype(str).to_numpy(),
        "actual_return": frame["intraday_return"].to_numpy(),
        "predicted_return": predicted_returns, "prediction": (predicted_returns > 0).astype(int),
    })
    if fold is not None:
        result.insert(0, "fold", fold)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--history-dir", type=Path, default=HISTORY_DIR)
    parser.add_argument("--stock-template", type=Path, default=ROOT / "model/all_down/all_down.csv")
    parser.add_argument("--artifacts-dir", type=Path, default=MODEL_DIR / "artifacts")
    parser.add_argument("--feature-set", choices=("baseline", "us"), default="baseline")
    parser.add_argument("--us-data-dir", type=Path, default=US_DATA_DIR)
    parser.add_argument("--match-us-coverage", action="store_true", help="Use identical US-covered rows for the baseline comparison")
    parameter_source = parser.add_mutually_exclusive_group()
    parameter_source.add_argument("--params", type=Path, help="JSON parameter overrides or exported best_params.json")
    parameter_source.add_argument("--baseline", action="store_true", help="Ignore saved tuning results and use original parameters")
    parser.add_argument("--validation-only", action="store_true", help="Save CV results without evaluating test or fitting deployment model")
    parser.add_argument("--train-end", help="Last allowed label date, YYYY-MM-DD (inclusive)")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--valid-days", type=int, default=20, help="Trading dates in each outer validation fold")
    parser.add_argument("--test-days", type=int, default=20, help="Untouched final test trading dates")
    parser.add_argument("--early-stop-days", type=int, default=20, help="Inner tuning dates before each outer fold")
    parser.add_argument("--min-train-days", type=int, default=252, help="Minimum inner training dates in the first fold")
    args = parser.parse_args()
    features = FEATURES + US_FEATURES if args.feature_set == "us" else FEATURES
    params_path = args.params
    saved = args.artifacts_dir / "best_params.json"
    if params_path is None and not args.baseline and saved.is_file():
        params_path = saved
    parameter_config = {}
    if params_path is not None:
        parameter_config = json.loads(params_path.read_text(encoding="utf-8"))
        if not isinstance(parameter_config, dict):
            parser.error("--params must contain a JSON object")
    overrides = validate_overrides(parameter_config.get("params", parameter_config))
    params = {**BASE_PARAMS, **overrides}
    prediction_stock_ids = read_template(args.stock_template)["id"].tolist()
    stock_ids = discover_stock_ids(args.data_dir, args.history_dir)
    if not set(prediction_stock_ids).issubset(stock_ids):
        raise ValueError("Some target stocks have no CSV in the historical or live data")
    frame = build_dataset(args.data_dir, stock_ids, history_dir=args.history_dir)
    frame = frame[frame["feature_date"].notna() & frame["intraday_return"].notna()].copy()
    if args.train_end:
        frame = frame[frame["date"] <= pd.Timestamp(args.train_end)]
    if args.feature_set == "us" or args.match_us_coverage:
        frame = add_us_features(frame, args.us_data_dir)
        frame = frame.loc[us_coverage(frame)].copy()
    dates = sorted(frame["date"].unique())
    try:
        splits, test_start = walk_forward_splits(
            dates, folds=args.cv_folds, valid_days=args.valid_days, test_days=args.test_days,
            early_stop_days=args.early_stop_days, min_train_days=args.min_train_days,
        )
    except ValueError as error:
        parser.error(str(error))
    train = frame[frame["date"] < test_start]
    target = frame["stock_id"].isin(prediction_stock_ids)
    test = frame[(frame["date"] >= test_start) & target]
    if test.empty:
        raise ValueError("No target-stock labels in the test period")
    print(f"Training {len(stock_ids)} stocks, evaluating/predicting {len(prediction_stock_ids)} stocks. "
          f"Loaded {len(frame)} rows from {frame['date'].min().date()} "
          f"through {frame['date'].max().date()} (latest available valid labels).")
    folds, oof_frames, oof_returns = [], [], []
    for split in splits:
        inner_train = frame[frame["date"] <= split["inner_train_end"]]
        early_stop = frame[(frame["date"] >= split["early_stop_start"]) &
                           (frame["date"] <= split["train_end"]) & target]
        fold_train = frame[frame["date"] <= split["train_end"]]
        valid = frame[(frame["date"] >= split["valid_start"]) &
                      (frame["date"] <= split["valid_end"]) & target]
        if early_stop.empty or valid.empty:
            raise ValueError(f"Fold {split['fold']} has no target-stock tuning/validation labels")
        tuning = lgb.LGBMRegressor(**params)
        tuning.fit(
            inner_train[features], inner_train["intraday_return"],
            eval_set=[(early_stop[features], early_stop["intraday_return"])],
            eval_metric="l2", categorical_feature=["stock_id"],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        best = max(1, int(tuning.best_iteration_))
        # The outer fold is not used for early stopping. Refit on all earlier rows.
        fold_model = lgb.LGBMRegressor(**{**params, "n_estimators": best})
        fold_model.fit(fold_train[features], fold_train["intraday_return"], categorical_feature=["stock_id"])
        predicted = fold_model.predict(valid[features])
        result = {
            **{key: (str(value.date()) if isinstance(value, pd.Timestamp) else value)
               for key, value in split.items()},
            "train_rows": len(fold_train), "train_dates": int(fold_train["date"].nunique()),
            "train_start": str(fold_train["date"].min().date()), "best_iteration": best,
            "validation": evaluate(valid, predicted),
        }
        folds.append(result)
        oof_frames.append(valid)
        oof_returns.append(predicted)
        print(f"Fold {split['fold']}/{args.cv_folds}: "
              f"{result['valid_start']}..{result['valid_end']}, "
              f"iterations={best}, score={result['validation']['score']:.6f}")
    selected_iterations = select_iterations([fold["best_iteration"] for fold in folds])
    params["n_estimators"] = selected_iterations
    validation_frame = pd.concat(oof_frames, ignore_index=True)
    validation_returns = np.concatenate(oof_returns)
    fold_summary = {
        metric: {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0)),
                 "min": float(np.min(values)), "max": float(np.max(values))}
        for metric in ("accuracy", "score", "mse", "mae")
        for values in [[fold["validation"][metric] for fold in folds]]
    }
    metrics = {
        "training_stock_count": len(stock_ids), "evaluation_stock_count": len(prediction_stock_ids),
        "train_rows": len(train), "train_dates": int(train["date"].nunique()),
        "train_start": str(train["date"].min().date()),
        "train_end": str(train["date"].max().date()),
        "best_iteration": selected_iterations,
        "feature_set": args.feature_set, "feature_count": len(features),
        "cross_validation": {
            "method": "expanding_window", "folds": folds, "fold_summary": fold_summary,
            "valid_days": args.valid_days, "early_stop_days": args.early_stop_days,
            "min_train_days": args.min_train_days, "test_days": args.test_days,
            "iteration_selection": "median of inner early-stopping best iterations, rounded down",
            "positive_score_folds": sum(fold["validation"]["score"] > 0 for fold in folds),
        },
        "validation": evaluate(validation_frame, validation_returns),
        "search_params": {**BASE_PARAMS, **overrides},
        "parameter_source": str(params_path.resolve()) if params_path else "baseline",
    }
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([prediction_rows(f, p, split["fold"])
               for f, p, split in zip(oof_frames, oof_returns, splits)], ignore_index=True).to_csv(
                   args.artifacts_dir / "validation_predictions.csv", index=False)
    pd.json_normalize(folds).to_csv(args.artifacts_dir / "fold_metrics.csv", index=False)
    if args.validation_only:
        (args.artifacts_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return
    # Only evaluate test after the parameters have been fixed.
    model = lgb.LGBMRegressor(**params)
    model.fit(train[features], train["intraday_return"], categorical_feature=["stock_id"])
    metrics["test"] = evaluate(test, model.predict(test[features]))
    # Freeze iteration count, then refit on ALL allowed dates for future inference.
    # The saved model's full-history outputs must not be used to report test scores.
    final = lgb.LGBMRegressor(**params)
    final.fit(frame[features], frame["intraday_return"],
              categorical_feature=["stock_id"])
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    final.booster_.save_model(str(args.artifacts_dir / "model.txt"))
    metadata = {
        "feature_version": FEATURE_VERSION, "features": features, "stock_ids": stock_ids,
        "feature_set": args.feature_set, "us_feature_version": US_FEATURE_VERSION if args.feature_set == "us" else None,
        "prediction_stock_ids": prediction_stock_ids,
        "training_rows": len(frame), "training_dates": len(dates),
        "training_start": str(frame["date"].min().date()),
        "trained_through": str(frame["date"].max().date()),
        "data_sources": {"history_dir": str(args.history_dir.resolve()),
                         "data_dir": str(args.data_dir.resolve()),
                         "overlap_policy": "live row replaces historical row"},
        "threshold": 0, "params": params,
        "us_data_dir": str(args.us_data_dir.resolve()) if args.feature_set == "us" else None,
        "us_availability_assumption": "08:10 Asia/Taipei after NY session close; forecast 08:25" if args.feature_set == "us" else None,
        "validation_method": "expanding_window", "cv_folds": args.cv_folds,
        "parameter_source": metrics["parameter_source"],
        "parameter_selection": parameter_config.get("selection"),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("numpy", "pandas", "lightgbm", "scikit-learn")},
    }
    for filename, content in [("metadata.json", metadata), ("metrics.json", metrics)]:
        (args.artifacts_dir / filename).write_text(
            json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame({"feature": features, "gain": final.booster_.feature_importance("gain")}).sort_values(
        "gain", ascending=False).to_csv(args.artifacts_dir / "feature_importance.csv", index=False)
    prediction_rows(test, model.predict(test[features])).to_csv(args.artifacts_dir / "test_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved deployment model trained through {metadata['trained_through']} to {args.artifacts_dir}")


if __name__ == "__main__":
    main()
