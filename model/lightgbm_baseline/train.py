"""Chronological train/validation/test baseline; refit full history for deployment."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import (DATA_DIR, HISTORY_DIR, FEATURES, FEATURE_VERSION, MODEL_DIR, ROOT,
                      build_dataset, discover_stock_ids, read_template)


def evaluate(frame, predicted_returns):
    actual_returns = frame["intraday_return"].to_numpy()

    prediction = (predicted_returns > 0).astype(int)

    contribution = (
        (2 * prediction - 1) * actual_returns
    )

    return {
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
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--history-dir", type=Path, default=HISTORY_DIR)
    parser.add_argument("--stock-template", type=Path, default=ROOT / "model/all_down/all_down.csv")
    parser.add_argument("--artifacts-dir", type=Path, default=MODEL_DIR / "artifacts")
    parser.add_argument("--train-end", help="Last allowed label date, YYYY-MM-DD (inclusive)")
    parser.add_argument("--valid-days", type=int, default=7)
    parser.add_argument("--test-days", type=int, default=7)
    args = parser.parse_args()
    if min(args.valid_days, args.test_days) < 1:
        parser.error("valid-days and test-days must be positive")
    prediction_stock_ids = read_template(args.stock_template)["id"].tolist()
    stock_ids = discover_stock_ids(args.data_dir, args.history_dir)
    if not set(prediction_stock_ids).issubset(stock_ids):
        raise ValueError("Some target stocks have no CSV in the historical or live data")
    frame = build_dataset(args.data_dir, stock_ids, history_dir=args.history_dir)
    frame = frame[frame["feature_date"].notna() & frame["intraday_return"].notna()].copy()
    if args.train_end:
        frame = frame[frame["date"] <= pd.Timestamp(args.train_end)]
    dates = sorted(frame["date"].unique())
    if len(dates) < args.valid_days + args.test_days + 10:
        raise ValueError("Need at least 10 training dates plus validation and test dates")
    valid_start = dates[-args.test_days - args.valid_days]
    test_start = dates[-args.test_days]
    train = frame[frame["date"] < valid_start]
    target = frame["stock_id"].isin(prediction_stock_ids)
    valid = frame[(frame["date"] >= valid_start) & (frame["date"] < test_start) & target]
    test = frame[(frame["date"] >= test_start) & target]
    if valid.empty or test.empty:
        raise ValueError("No target-stock labels in the validation or test period")
    print(f"Training {len(stock_ids)} stocks, evaluating/predicting {len(prediction_stock_ids)} stocks. "
          f"Loaded {len(frame)} rows from {frame['date'].min().date()} "
          f"through {frame['date'].max().date()} (latest available valid labels).")
    params = dict(
        objective="regression", n_estimators=500, learning_rate=0.03,
        num_leaves=7, max_depth=4, min_child_samples=50,
        reg_lambda=1.0, random_state=42, n_jobs=4, verbosity=-1,
        deterministic=True, force_col_wise=True,
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(
        train[FEATURES], train["intraday_return"],
        eval_set=[(valid[FEATURES], valid["intraday_return"])],
        eval_metric="l2", categorical_feature=["stock_id"],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    metrics = {
        "training_stock_count": len(stock_ids), "evaluation_stock_count": len(prediction_stock_ids),
        "train_rows": len(train), "train_dates": int(train["date"].nunique()),
        "train_start": str(train["date"].min().date()),
        "train_end": str(train["date"].max().date()),
        "best_iteration": int(model.best_iteration_),
        "validation": evaluate(valid, model.predict(valid[FEATURES])),
        "test": evaluate(test, model.predict(test[FEATURES])),
    }
    # Freeze iteration count, then refit on ALL allowed dates for future inference.
    # The saved model's full-history outputs must not be used to report test scores.
    params["n_estimators"] = max(1, model.best_iteration_)
    final = lgb.LGBMRegressor(**params)
    final.fit(frame[FEATURES], frame["intraday_return"],
              categorical_feature=["stock_id"])
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    final.booster_.save_model(str(args.artifacts_dir / "model.txt"))
    metadata = {
        "feature_version": FEATURE_VERSION, "features": FEATURES, "stock_ids": stock_ids,
        "prediction_stock_ids": prediction_stock_ids,
        "training_rows": len(frame), "training_dates": len(dates),
        "training_start": str(frame["date"].min().date()),
        "trained_through": str(frame["date"].max().date()),
        "data_sources": {"history_dir": str(args.history_dir.resolve()),
                         "data_dir": str(args.data_dir.resolve()),
                         "overlap_policy": "live row replaces historical row"},
        "threshold": 0, "params": params,
        "versions": {name: importlib.metadata.version(name)
                     for name in ("numpy", "pandas", "lightgbm", "scikit-learn")},
    }
    for filename, content in [("metadata.json", metadata), ("metrics.json", metrics)]:
        (args.artifacts_dir / filename).write_text(
            json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame({"feature": FEATURES, "gain": final.booster_.feature_importance("gain")}).sort_values(
        "gain", ascending=False).to_csv(args.artifacts_dir / "feature_importance.csv", index=False)
    predicted_returns = model.predict(test[FEATURES])

    pd.DataFrame({
        "date": test["date"],
        "id": test["stock_id"],
        "actual_return": test["intraday_return"],
        "predicted_return": predicted_returns,
        "prediction": (predicted_returns > 0).astype(int),
    }).to_csv(
        args.artifacts_dir / "test_predictions.csv",
        index=False,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved deployment model trained through {metadata['trained_through']} to {args.artifacts_dir}")


if __name__ == "__main__":
    main()
