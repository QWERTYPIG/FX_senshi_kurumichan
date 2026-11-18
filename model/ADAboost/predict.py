"""Generate a manager-compatible binary prediction.csv with the trained AdaBoost model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import DATA_DIR, HISTORY_DIR, MODEL_DIR, TEMPLATE, read_template
from features import FEATURES, FEATURE_VERSION, build_dataset


def prediction_template(template_path: Path, date: str | None, stock_ids: list[str]) -> pd.DataFrame:
    if date is not None:
        parsed = pd.to_datetime(date, format="%Y-%m-%d", errors="raise")
        if parsed.strftime("%Y-%m-%d") != date:
            raise ValueError("--date must use YYYY-MM-DD format")
        return pd.DataFrame({"id": stock_ids, date: 0})
    if not template_path.is_file():
        raise ValueError(
            f"No prediction template at {template_path}. "
            "Specify --date YYYY-MM-DD or --template path/to/sample_submission.csv."
        )
    return read_template(template_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--history-dir", type=Path, default=HISTORY_DIR)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--template", type=Path, default=TEMPLATE)
    selection.add_argument("--date", help="Explicit single prediction date in YYYY-MM-DD format")
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("MANAGER_OUTPUT", "prediction.csv")))
    parser.add_argument("--artifacts-dir", type=Path, default=MODEL_DIR / "artifacts")
    parser.add_argument("--max-history-age-days", type=int, default=10)
    args = parser.parse_args()

    metadata_path = args.artifacts_dir / "metadata.json"
    model_path = args.artifacts_dir / "model.joblib"
    if not metadata_path.is_file() or not model_path.is_file():
        raise ValueError("No trained AdaBoost artifacts. Run train.py first.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("feature_version") != FEATURE_VERSION or metadata.get("features") != FEATURES:
        raise ValueError("Feature implementation differs from the saved model; retrain")
    if "threshold" not in metadata:
        raise ValueError("Saved model predates validation-selected thresholds; retrain")
    try:
        template = prediction_template(args.template, args.date, metadata["prediction_stock_ids"])
    except ValueError as error:
        parser.error(str(error))
    stock_ids = template["id"].tolist()
    if stock_ids != metadata["prediction_stock_ids"]:
        raise ValueError("Template stock list/order differs from training; retrain with the current stock template")
    dates = list(template.columns[1:])
    parsed_dates = pd.to_datetime(dates)
    if parsed_dates.min() <= pd.Timestamp(metadata["trained_through"]):
        parser.error(
            f"Prediction dates must be after trained_through={metadata['trained_through']}; "
            "use saved test_predictions.csv for historical evaluation"
        )
    if args.max_history_age_days < 1:
        parser.error("--max-history-age-days must be positive")

    frame = build_dataset(
        args.data_dir, stock_ids, dates=dates, history_dir=args.history_dir,
    )
    age = (frame["date"] - frame["feature_date"]).dt.days
    if frame["feature_date"].isna().any() or (age <= 0).any() or (age > args.max_history_age_days).any():
        raise ValueError("Missing or stale prior-day history; synchronize market data before predicting")
    model = joblib.load(model_path)
    frame["up_probability"] = model.predict_proba(frame[FEATURES])[:, 1]
    frame["prediction"] = (frame["up_probability"] >= float(metadata["threshold"])).astype(int)
    if len(frame) != len(stock_ids) * len(dates) or not np.isin(frame["prediction"], [0, 1]).all():
        raise ValueError("Inference did not produce one binary prediction per stock/date")
    lookup = frame.pivot(index="stock_id", columns="date", values="prediction")
    for date in dates:
        template[date] = lookup.loc[stock_ids, pd.Timestamp(date)].to_numpy(dtype=int)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    template.to_csv(temporary, index=False)
    temporary.replace(args.output)
    print(f"Wrote {args.output}: {len(stock_ids)} stocks, {len(dates)} dates")


if __name__ == "__main__":
    main()
