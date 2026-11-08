"""Chronological train/validation/test baseline; refit full history for deployment."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import DATA_DIR, FEATURES, FEATURE_VERSION, MODEL_DIR, ROOT, build_dataset, read_template


def evaluate(frame, probabilities):
    prediction = (probabilities >= 0.5).astype(int)
    returns = frame["intraday_return"].to_numpy()
    contribution = (2 * prediction - 1) * returns
    daily = pd.DataFrame({"date": frame["date"].to_numpy(), "score": contribution})
    return {
        "rows": len(frame), "dates": int(frame["date"].nunique()),
        "start": str(frame["date"].min().date()), "end": str(frame["date"].max().date()),
        "accuracy": float(np.mean(prediction == (returns > 0))),
        "score": float(contribution.mean()),
        "all_up_score": float(returns.mean()), "all_down_score": float(-returns.mean()),
        "daily_scores": {str(k.date()): float(v) for k, v in daily.groupby("date")["score"].mean().items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--stock-template", type=Path, default=ROOT / "model/all_down/all_down.csv")
    parser.add_argument("--artifacts-dir", type=Path, default=MODEL_DIR / "artifacts")
    parser.add_argument("--train-end", help="Last allowed label date, YYYY-MM-DD (inclusive)")
    parser.add_argument("--valid-days", type=int, default=7)
    parser.add_argument("--test-days", type=int, default=7)
    args = parser.parse_args()
    if min(args.valid_days, args.test_days) < 1:
        parser.error("valid-days and test-days must be positive")
    stock_ids = read_template(args.stock_template)["id"].tolist()
    frame = build_dataset(args.data_dir, stock_ids)
    frame = frame[frame["feature_date"].notna() & frame["intraday_return"].notna()].copy()
    if args.train_end:
        frame = frame[frame["date"] <= pd.Timestamp(args.train_end)]
    dates = sorted(frame["date"].unique())
    if len(dates) < args.valid_days + args.test_days + 10:
        raise ValueError("Need at least 10 training dates plus validation and test dates")
    valid_start = dates[-args.test_days - args.valid_days]
    test_start = dates[-args.test_days]
    train = frame[frame["date"] < valid_start]
    valid = frame[(frame["date"] >= valid_start) & (frame["date"] < test_start)]
    test = frame[frame["date"] >= test_start]
    params = dict(
        objective="binary", n_estimators=500, learning_rate=0.03,
        num_leaves=7, max_depth=3, min_child_samples=50,
        reg_lambda=1.0, random_state=42, n_jobs=4, verbosity=-1,
        deterministic=True, force_col_wise=True,
    )
    model = lgb.LGBMClassifier(**params)
    model.fit(
        train[FEATURES], (train["intraday_return"] > 0).astype(int),
        eval_set=[(valid[FEATURES], (valid["intraday_return"] > 0).astype(int))],
        eval_metric="binary_logloss", categorical_feature=["stock_id"],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    metrics = {
        "train_rows": len(train), "train_dates": int(train["date"].nunique()),
        "train_start": str(train["date"].min().date()),
        "train_end": str(train["date"].max().date()),
        "best_iteration": int(model.best_iteration_),
        "validation": evaluate(valid, model.predict_proba(valid[FEATURES])[:, 1]),
        "test": evaluate(test, model.predict_proba(test[FEATURES])[:, 1]),
    }
    # Freeze iteration count, then refit on ALL allowed dates for future inference.
    # The saved model's full-history outputs must not be used to report test scores.
    params["n_estimators"] = max(1, model.best_iteration_)
    final = lgb.LGBMClassifier(**params)
    final.fit(frame[FEATURES], (frame["intraday_return"] > 0).astype(int),
              categorical_feature=["stock_id"])
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    final.booster_.save_model(str(args.artifacts_dir / "model.txt"))
    metadata = {
        "feature_version": FEATURE_VERSION, "features": FEATURES, "stock_ids": stock_ids,
        "trained_through": str(frame["date"].max().date()),
        "threshold": 0.5, "params": params,
        "versions": {name: importlib.metadata.version(name)
                     for name in ("numpy", "pandas", "lightgbm", "scikit-learn")},
    }
    for filename, content in [("metadata.json", metadata), ("metrics.json", metrics)]:
        (args.artifacts_dir / filename).write_text(
            json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame({"feature": FEATURES, "gain": final.booster_.feature_importance("gain")}).sort_values(
        "gain", ascending=False).to_csv(args.artifacts_dir / "feature_importance.csv", index=False)
    pd.DataFrame({
        "date": test["date"], "id": test["stock_id"],
        "return": test["intraday_return"],
        "probability": model.predict_proba(test[FEATURES])[:, 1],
    }).to_csv(args.artifacts_dir / "test_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved deployment model trained through {metadata['trained_through']} to {args.artifacts_dir}")


if __name__ == "__main__":
    main()
