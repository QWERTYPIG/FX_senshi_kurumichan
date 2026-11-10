"""Write manager-compatible predictions with time and data freshness checks."""
import argparse
import json
import os
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from features import DATA_DIR, HISTORY_DIR, FEATURES, FEATURE_VERSION, MODEL_DIR, TEMPLATE, build_dataset, read_template


def prediction_template(template_path, date, stock_ids):
    if date:
        parsed = pd.to_datetime(date, format="%Y-%m-%d", errors="raise")
        if parsed.strftime("%Y-%m-%d") != date:
            raise ValueError("--date must use YYYY-MM-DD format")
        return pd.DataFrame({"id": stock_ids, date: 0})
    if not template_path.is_file():
        raise ValueError(
            f"No prediction template at {template_path}. "
            "Specify --date YYYY-MM-DD or --template path/to/daily_sample_submission.csv."
        )
    return read_template(template_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--history-dir", type=Path, default=HISTORY_DIR)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--template", type=Path, default=TEMPLATE,
                           help="Default: MANAGER_TEMPLATE or manager_data/template.csv")
    selection.add_argument("--date", help="Predict one explicit date (YYYY-MM-DD) for the target stock list")
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("MANAGER_OUTPUT", "prediction.csv")))
    parser.add_argument("--artifacts-dir", type=Path, default=MODEL_DIR / "artifacts")
    parser.add_argument("--max-history-age-days", type=int, default=10,
                        help="Maximum calendar days between source and target dates")
    args = parser.parse_args()
    metadata_path = args.artifacts_dir / "metadata.json"
    if not metadata_path.exists():
        raise ValueError("No trained model. Run train.py or manager's retrain action first.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["features"] != FEATURES or metadata["feature_version"] != FEATURE_VERSION:
        raise ValueError("Feature implementation changed; retrain the model")
    try:
        template = prediction_template(args.template, args.date, metadata["prediction_stock_ids"])
    except ValueError as error:
        parser.error(str(error))
    stock_ids = template["id"].tolist()
    if stock_ids != metadata["prediction_stock_ids"]:
        raise ValueError("Template stock list/order differs from the saved target list; retrain the model")
    dates = list(template.columns[1:])
    if min(pd.to_datetime(dates)) <= pd.Timestamp(metadata["trained_through"]):
        parser.error(
            f"Template dates {min(dates)} through {max(dates)} must be after "
            f"trained_through={metadata['trained_through']}. "
            "Use --date YYYY-MM-DD for a later prediction date or --template with a daily sample. "
            "For historical evaluation use artifacts/metrics.json and test_predictions.csv."
        )
    frame = build_dataset(args.data_dir, stock_ids, dates=dates, history_dir=args.history_dir,
                          category_ids=metadata["stock_ids"])
    age = (frame["date"] - frame["feature_date"]).dt.days
    if frame["feature_date"].isna().any() or (age > args.max_history_age_days).any():
        raise ValueError("Missing or stale price history; synchronize data before predicting")
    model = lgb.Booster(model_file=str(args.artifacts_dir / "model.txt"))
    frame["prediction"] = (model.predict(frame[FEATURES]) > metadata["threshold"]).astype(int)
    lookup = frame.pivot(index="stock_id", columns="date", values="prediction")
    for date in dates:
        template[date] = lookup.loc[stock_ids, pd.Timestamp(date)].to_numpy()
    # Do not touch an existing prediction until all checks and inference succeed.
    temp = args.output.with_suffix(".csv.tmp")
    template.to_csv(temp, index=False)
    temp.replace(args.output)
    print(f"Wrote {args.output}: {len(stock_ids)} stocks, {len(dates)} dates")


if __name__ == "__main__":
    main()
