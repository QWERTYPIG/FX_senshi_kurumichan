"""Evaluate one GRU configuration outside the manager's weekly training command."""

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from statistics import median

import torch
from market import MAX_MARKET_AGE, MarketHistory, read_stock_ids
from runtime import MODEL_DIR, Config, ManagerPaths, configure_cpu, iso_date
from train import fit_fixed, initialize_training, smooth_loss, train_epoch


@dataclass(frozen=True)
class Fold:
    start: date
    end: date


@dataclass(frozen=True)
class EvaluationPlan:
    folds: tuple[Fold, ...]
    early_stopping_days: int
    patience: int
    final_test_start: date

    def __post_init__(self):
        for field in ("early_stopping_days", "patience"):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"evaluation.{field} must be a positive integer")
        if not self.folds or any(fold.start > fold.end or fold.end >= self.final_test_start for fold in self.folds):
            raise ValueError("validation folds must be nonempty ranges strictly before the final test")
        if any(right.start - left.end != timedelta(days=1) for left, right in zip(self.folds, self.folds[1:])):
            raise ValueError("validation folds must be ordered, disjoint and contiguous")

    @classmethod
    def from_settings(cls, settings):
        return cls(**{
            **settings,
            "folds": tuple(Fold(**{key: iso_date(value) for key, value in fold.items()}) for fold in settings["folds"]),
            "final_test_start": iso_date(settings["final_test_start"]),
        })


class _TrainingBatches:
    """Reuse normalized labeled batches across epochs and overlapping fold fits, within a memory budget."""

    def __init__(self, history: MarketHistory, config: Config):
        self.history = history
        self.length = config.window_length
        self.cached = {}
        self.remaining_bytes = 256 * 1024**2

    def for_dates(self, dates):
        for day in dates:
            batch = self.cached.get(day)
            if batch is None:
                batch = self.history.batch(day, self.length, labels=True)
                # The sequence dominates memory; include the other tensors and provenance pointers.
                size = batch.sequence.nbytes + batch.today.nbytes + batch.present.nbytes + batch.intraday_return.nbytes
                size += 1024 + 128 * len(batch.present)
                if size <= self.remaining_bytes:
                    self.cached[day] = batch
                    self.remaining_bytes -= size
            yield batch


@dataclass(frozen=True)
class _FoldDates:
    training: list[date]
    inner_fit: list[date]
    early_stopping: list[date]
    usable_early_stopping: list[date]
    validation: list[date]


def _split_dates(calendar, fold, early_stopping_days, labeled):
    training = [day for day in calendar if day < fold.start]
    if len(training) <= early_stopping_days:
        raise ValueError(f"fold starting {fold.start}: not enough trading dates for inner early stopping")
    early = training[-early_stopping_days:]
    dates = _FoldDates(
        training=[day for day in training if day in labeled],
        inner_fit=[day for day in training[:-early_stopping_days] if day in labeled],
        early_stopping=early,
        usable_early_stopping=[day for day in early if day in labeled],
        validation=[day for day in calendar if fold.start <= day <= fold.end],
    )
    if not dates.inner_fit or not dates.usable_early_stopping or not dates.validation:
        raise ValueError(f"fold starting {fold.start}: empty inner fit, early stopping or validation split")
    return dates


@torch.no_grad()
def _early_loss(model, batches):
    model.eval()
    losses = [smooth_loss(model, batch).item() for batch in batches]
    loss = sum(losses) / len(losses)
    if not math.isfinite(loss):
        raise ValueError("nonfinite early stopping loss")
    return loss


def _best_epoch(config, patience, num_stocks, batches, inner_fit_dates, early_labeled_dates):
    model, optimizer = initialize_training(config, num_stocks)
    best_loss = math.inf
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        train_epoch(model, optimizer, batches.for_dates(inner_fit_dates))
        loss = _early_loss(model, batches.for_dates(early_labeled_dates))
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch
        elif epoch - best_epoch >= patience:
            break
    return best_epoch


def _target_positions(history, target_ids):
    positions = {stock: i for i, stock in enumerate(history.stock_ids)}
    if not set(target_ids) <= positions.keys():
        raise ValueError("Top50@Y contains IDs outside Top100@Y")
    return [positions[stock] for stock in target_ids]


@torch.no_grad()
def _score(model, history, config, dates, target_positions):
    model.eval()
    rows = []
    for day in dates:
        realized = history.realized_returns(day, target_positions)
        if not realized:
            continue
        # The prediction mask is causal: realized labels never decide which stocks enter attention.
        not_before = day - MAX_MARKET_AGE
        batch = history.batch(day, config.window_length, market_not_before=not_before)
        if all(last is None or last < not_before for last in batch.latest_market_date):
            raise ValueError(f"{day}: all universe stocks have stale market data")
        scores = model(batch).tolist()
        for i, intraday_return in realized.items():
            f = scores[i]
            if not math.isfinite(f) or not math.isfinite(intraday_return):
                raise ValueError(f"{day}: nonfinite score or realized return for {history.stock_ids[i]}")
            rows.append({"date": day.isoformat(), "id": history.stock_ids[i], "f": f,
                         "direction": int(f > 0), "intraday_return": intraday_return})
    if not rows:
        raise ValueError("no labeled target stock-dates to score")
    always_up = math.fsum(row["intraday_return"] for row in rows) / len(rows)
    contest_score = math.fsum((2 * row["direction"] - 1) * row["intraday_return"] for row in rows) / len(rows)
    return {"row_count": len(rows), "contest_score": contest_score,
            "always_up": always_up, "always_down": -always_up}, rows


def _dates(days):
    return [day.isoformat() for day in days]


def _run_metadata(history, targets, config, final_test_start):
    return {"config": asdict(config), "universe_stock_ids": history.stock_ids, "target_stock_ids": targets,
            "final_test_start": final_test_start.isoformat(), "torch_version": torch.__version__}


def evaluate_validation(paths: ManagerPaths, config: Config, plan: EvaluationPlan):
    """Run one configuration and seed; return validation metrics and OOS rows, never final-test outcomes."""
    configure_cpu()
    history = MarketHistory(paths.history, paths.update, through=plan.folds[-1].end)
    targets = read_stock_ids(paths.history, "Top50@Y")
    target_positions = _target_positions(history, targets)
    labeled = history.labeled_dates(config.window_length)
    splits = [_split_dates(history.dates, fold, plan.early_stopping_days, labeled) for fold in plan.folds]
    batches = _TrainingBatches(history, config)
    num_stocks = len(history.stock_ids)
    folds, predictions = [], []
    for number, (fold, dates) in enumerate(zip(plan.folds, splits), start=1):
        epoch = _best_epoch(config, plan.patience, num_stocks, batches, dates.inner_fit, dates.usable_early_stopping)
        model = fit_fixed(replace(config, epochs=epoch), num_stocks, lambda: batches.for_dates(dates.training))
        scores, rows = _score(model, history, config, dates.validation, target_positions)
        folds.append({"fold": number, "start": fold.start.isoformat(), "end": fold.end.isoformat(),
                      "training_dates": _dates(dates.training), "inner_fit_dates": _dates(dates.inner_fit),
                      "early_stopping_dates": _dates(dates.early_stopping), "validation_dates": _dates(dates.validation),
                      "best_epoch": epoch, **scores})
        predictions.extend({"fold": number, **row} for row in rows)
    refit_epochs = max(1, math.floor(median(fold["best_epoch"] for fold in folds)))
    metrics = _run_metadata(history, targets, config, plan.final_test_start)
    return {**metrics, "early_stopping_days": plan.early_stopping_days, "patience": plan.patience,
            "folds": folds, "refit_epochs": refit_epochs}, predictions


def report_final_test(paths: ManagerPaths, config: Config, final_test_start: date):
    """Fit the frozen configuration without test rows; then score it on the final test period."""
    configure_cpu()
    pre_test = MarketHistory(paths.history, paths.update, through=final_test_start - timedelta(days=1))
    training_dates = sorted(pre_test.labeled_dates(config.window_length))
    if not training_dates:
        raise ValueError("no usable training labels before the final test period")
    targets = read_stock_ids(paths.history, "Top50@Y")
    target_positions = _target_positions(pre_test, targets)
    metadata = _run_metadata(pre_test, targets, config, final_test_start)
    batches = _TrainingBatches(pre_test, config)
    model = fit_fixed(config, len(pre_test.stock_ids), lambda: batches.for_dates(training_dates))
    del batches, pre_test

    history = MarketHistory(paths.history, paths.update)
    if history.stock_ids != metadata["universe_stock_ids"]:
        raise ValueError("the stock universe changed during final-test reporting")
    dates = [day for day in history.dates if day >= final_test_start]
    if not dates:
        raise ValueError(f"no trading dates in the final test period starting {final_test_start}")
    scores, rows = _score(model, history, config, dates, target_positions)
    return {**metadata, "training_dates": _dates(training_dates), "test_dates": _dates(dates), **scores}, rows


def _write_report(directory, period, metrics, predictions, *, mode):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{period}_predictions.csv").open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    with (directory / f"{period}_metrics.json").open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "final-test"))
    parser.add_argument("--config", type=Path, help="model config (required for final-test)")
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "final-test" and args.config is None:
        parser.error("final-test requires --config with frozen model settings")
    config_path = args.config or MODEL_DIR / "config.json"
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    paths, config, evaluation = ManagerPaths.read(), Config.from_settings(settings), settings["evaluation"]
    if args.mode == "validate":
        metrics, predictions = evaluate_validation(paths, config, EvaluationPlan.from_settings(evaluation))
        _write_report(args.artifacts_dir, "validation", metrics, predictions, mode="w")
    else:
        for filename in ("final_test_metrics.json", "final_test_predictions.csv"):
            if (args.artifacts_dir / filename).exists():
                raise SystemExit(f"Final-test artifacts already exist in {args.artifacts_dir}")
        metrics, predictions = report_final_test(paths, config, iso_date(evaluation["final_test_start"]))
        _write_report(args.artifacts_dir, "final_test", metrics, predictions, mode="x")


if __name__ == "__main__":
    main()
