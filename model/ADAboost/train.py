"""Tune, evaluate, and fit a return-weighted AdaBoost direction classifier."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from common import DATA_DIR, HISTORY_DIR, MODEL_DIR, ROOT, evaluate, read_template, walk_forward_splits, write_json
from features import FEATURES, FEATURE_VERSION, build_dataset, discover_stock_ids


DEFAULT_CANDIDATES = [(1, 100, 0.05), (1, 200, 0.05), (2, 100, 0.05), (2, 100, 0.10)]


def parse_candidate(value: str) -> tuple[int, int, float]:
    try:
        depth_text, estimators_text, rate_text = value.split(",")
        candidate = (int(depth_text), int(estimators_text), float(rate_text))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("candidate must be DEPTH,ESTIMATORS,LEARNING_RATE") from error
    if candidate[0] < 1 or candidate[1] < 1 or not np.isfinite(candidate[2]) or candidate[2] <= 0:
        raise argparse.ArgumentTypeError("candidate values must be finite and positive")
    return candidate


def make_model(depth: int, estimators: int, learning_rate: float, random_state: int) -> Pipeline:
    base = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=50, random_state=random_state)
    classifier = AdaBoostClassifier(
        estimator=base,
        n_estimators=estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("classifier", classifier),
    ])


def return_weights(frame: pd.DataFrame) -> np.ndarray:
    weights = np.abs(frame["intraday_return"].to_numpy(dtype=float))
    positive = weights[weights > 0]
    if len(positive) == 0:
        raise ValueError("Training requires at least one non-zero intraday return")
    # A small floor keeps flat days in the classification problem while preserving
    # the official metric's emphasis on high-magnitude observations.
    weights = weights + np.median(positive) * 0.01
    return weights / weights.mean()


def fit_model(frame: pd.DataFrame, candidate: tuple[int, int, float], random_state: int) -> Pipeline:
    labels = (frame["intraday_return"].to_numpy() > 0).astype(int)
    if len(np.unique(labels)) != 2:
        raise ValueError("AdaBoost training data must contain both up and down labels")
    model = make_model(*candidate, random_state=random_state)
    model.fit(frame[FEATURES], labels, classifier__sample_weight=return_weights(frame))
    return model


def predict_up_probability(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(frame[FEATURES])[:, 1]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Model produced invalid up probabilities")
    return probabilities


def predict_binary(model: Pipeline, frame: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    prediction = (predict_up_probability(model, frame) >= threshold).astype(int)
    if not np.isin(prediction, [0, 1]).all():
        raise ValueError("Model produced a non-binary prediction")
    return prediction


def select_threshold(frame: pd.DataFrame, probabilities: np.ndarray) -> tuple[float, float, float]:
    """Maximize official score, flipping equal-probability rows as one group."""
    returns = frame["intraday_return"].to_numpy(dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(returns) == 0 or len(returns) != len(probabilities):
        raise ValueError("Threshold selection requires one probability per non-empty row")
    if not np.isfinite(returns).all() or not np.isfinite(probabilities).all():
        raise ValueError("Threshold selection requires finite values")
    order = np.argsort(-probabilities, kind="stable")
    sorted_probability = probabilities[order]
    sorted_return = returns[order]
    # A threshold just above the largest probability represents predict-all-down.
    best_threshold = float(np.nextafter(sorted_probability[0], np.inf))
    best_score = float(-returns.mean())
    best_up_count = 0
    cumulative_return = np.cumsum(sorted_return)
    group_ends = np.flatnonzero(np.r_[sorted_probability[1:] != sorted_probability[:-1], True])
    for end in group_ends:
        score = float((-returns.sum() + 2 * cumulative_return[end]) / len(returns))
        threshold = float(sorted_probability[end])
        up_count = int(end + 1)
        candidate_key = (score, -abs(threshold - 0.5), -up_count)
        best_key = (best_score, -abs(best_threshold - 0.5), -best_up_count)
        if candidate_key > best_key:
            best_threshold, best_score, best_up_count = threshold, score, up_count
    return best_threshold, best_score, best_up_count / len(returns)


def candidate_dict(candidate: tuple[int, int, float]) -> dict[str, int | float]:
    return {"max_depth": candidate[0], "n_estimators": candidate[1], "learning_rate": candidate[2]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--history-dir", type=Path, default=HISTORY_DIR)
    parser.add_argument("--stock-template", type=Path, default=ROOT / "model/all_down/all_down.csv")
    parser.add_argument("--artifacts-dir", type=Path, default=MODEL_DIR / "artifacts")
    parser.add_argument("--train-end", help="Last allowed label date, YYYY-MM-DD (inclusive)")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--valid-days", type=int, default=20)
    parser.add_argument("--test-days", type=int, default=20)
    parser.add_argument("--min-train-days", type=int, default=252)
    parser.add_argument(
        "--candidate", action="append", type=parse_candidate,
        help="Repeat DEPTH,ESTIMATORS,LEARNING_RATE; defaults to four small grids",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    prediction_stock_ids = read_template(args.stock_template)["id"].tolist()
    try:
        stock_ids = discover_stock_ids(args.data_dir, args.history_dir)
    except FileNotFoundError as error:
        parser.error(str(error))
    missing = sorted(set(prediction_stock_ids) - set(stock_ids))
    if missing:
        raise ValueError(f"Target stocks missing from market data: {missing}")
    frame = build_dataset(args.data_dir, stock_ids, history_dir=args.history_dir)
    frame = frame[frame["feature_date"].notna() & frame["intraday_return"].notna()].copy()
    if args.train_end:
        cutoff = pd.to_datetime(args.train_end, format="%Y-%m-%d", errors="raise")
        frame = frame[frame["date"] <= cutoff]
    if frame.empty:
        raise ValueError("No labeled rows remain after loading and applying --train-end")

    try:
        splits, test_start = walk_forward_splits(
            frame["date"], folds=args.cv_folds, valid_days=args.valid_days,
            test_days=args.test_days, min_train_days=args.min_train_days,
        )
    except ValueError as error:
        parser.error(str(error))
    target_mask = frame["stock_id"].isin(prediction_stock_ids)
    test = frame[(frame["date"] >= test_start) & target_mask]
    if test.empty:
        raise ValueError("No target-stock labels in the final test period")

    candidates = args.candidate or DEFAULT_CANDIDATES
    trials: list[dict[str, object]] = []
    validation_predictions: list[pd.DataFrame] = []
    for candidate_index, candidate in enumerate(candidates):
        candidate_parts: list[pd.DataFrame] = []
        for split in splits:
            train = frame[frame["date"] <= split["train_end"]]
            valid = frame[
                (frame["date"] >= split["valid_start"])
                & (frame["date"] <= split["valid_end"])
                & target_mask
            ]
            if valid.empty:
                raise ValueError(f"Fold {split['fold']} has no target-stock validation rows")
            model = fit_model(train, candidate, args.random_state)
            detail = valid[["date", "stock_id", "intraday_return"]].copy()
            detail["up_probability"] = predict_up_probability(model, valid)
            detail.insert(0, "fold", split["fold"])
            detail.insert(0, "candidate", candidate_index)
            detail["train_end"] = str(split["train_end"].date())
            candidate_parts.append(detail)
        candidate_oof = pd.concat(candidate_parts, ignore_index=True)
        threshold, selected_score, selected_up_rate = select_threshold(
            candidate_oof, candidate_oof["up_probability"].to_numpy()
        )
        candidate_oof["prediction"] = (
            candidate_oof["up_probability"] >= threshold
        ).astype(int)
        validation_predictions.append(candidate_oof)
        fold_metrics = []
        for split in splits:
            fold_frame = candidate_oof[candidate_oof["fold"] == split["fold"]]
            fold_metrics.append({
                "fold": split["fold"],
                "train_end": str(split["train_end"].date()),
                "valid_start": str(split["valid_start"].date()),
                "valid_end": str(split["valid_end"].date()),
                **evaluate(fold_frame, fold_frame["prediction"].to_numpy()),
            })
        fold_scores = [fold["score"] for fold in fold_metrics]
        trial = {
            "candidate": candidate_index,
            **candidate_dict(candidate),
            "threshold": threshold,
            "validation_score": selected_score,
            "validation_up_rate": selected_up_rate,
            "mean_score": float(np.mean(fold_scores)),
            "score_std": float(np.std(fold_scores, ddof=0)),
            "positive_score_folds": int(np.sum(np.asarray(fold_scores) > 0)),
            "folds": fold_metrics,
        }
        trials.append(trial)
        print(json.dumps({key: value for key, value in trial.items() if key != "folds"}), flush=True)

    best_trial = max(trials, key=lambda row: (row["mean_score"], -row["score_std"]))
    selected = candidates[int(best_trial["candidate"])]
    selected_threshold = float(best_trial["threshold"])
    pretest = frame[frame["date"] < test_start]
    test_model = fit_model(pretest, selected, args.random_state)
    test_probability = predict_up_probability(test_model, test)
    test_prediction = (test_probability >= selected_threshold).astype(int)
    test_metrics = evaluate(test, test_prediction)

    deployment_model = fit_model(frame, selected, args.random_state)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(deployment_model, args.artifacts_dir / "model.joblib")
    versions = {
        package: importlib.metadata.version(package)
        for package in ("numpy", "pandas", "scikit-learn", "joblib")
    }
    metadata = {
        "feature_version": FEATURE_VERSION,
        "features": FEATURES,
        "stock_ids": stock_ids,
        "prediction_stock_ids": prediction_stock_ids,
        "trained_through": str(frame["date"].max().date()),
        "training_start": str(frame["date"].min().date()),
        "training_rows": int(len(frame)),
        "training_dates": int(frame["date"].nunique()),
        "selected_params": candidate_dict(selected),
        "threshold": selected_threshold,
        "random_state": args.random_state,
        "versions": versions,
        "data_sources": {
            "history_dir": str(args.history_dir.resolve()),
            "data_dir": str(args.data_dir.resolve()),
            "overlap_policy": "live row replaces historical row",
        },
    }
    metrics = {
        "selection_metric": "official signed-return score with probability threshold selected on walk-forward OOF rows",
        "cross_validation": trials,
        "selected_candidate": int(best_trial["candidate"]),
        "test": test_metrics,
    }
    write_json(args.artifacts_dir / "metadata.json", metadata)
    write_json(args.artifacts_dir / "metrics.json", metrics)
    pd.DataFrame([{key: value for key, value in trial.items() if key != "folds"} for trial in trials]).to_csv(
        args.artifacts_dir / "candidate_metrics.csv", index=False
    )
    pd.concat(validation_predictions, ignore_index=True).to_csv(
        args.artifacts_dir / "validation_predictions.csv", index=False
    )
    test_detail = test[["date", "stock_id", "intraday_return"]].copy()
    test_detail["up_probability"] = test_probability
    test_detail["prediction"] = test_prediction
    test_detail.to_csv(args.artifacts_dir / "test_predictions.csv", index=False)
    print(json.dumps({"selected_params": metadata["selected_params"], "test": test_metrics}, indent=2))
    print(f"Saved deployment model trained through {metadata['trained_through']} to {args.artifacts_dir}")


if __name__ == "__main__":
    main()
