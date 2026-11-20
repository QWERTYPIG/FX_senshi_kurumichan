# Intraday Direction Prediction

Teams predict whether each of 50 Taiwanese stocks closes above its open on a trading date, scored by the signed intraday return of each prediction.

## Prediction target

**Intraday return**:
The adjusted close over the adjusted open of a stock on one trading date, minus one.
_Avoid_: Daily return, return (unqualified)

**Direction**:
The binary prediction for a stock on a prediction date: 1 means up, 0 means down. A flat intraday return counts as down for labeling.
_Avoid_: Signal, label (when meaning the submitted value)

**Contest score**:
The mean over submitted predictions of the intraday return, signed positive when the direction is up and negative when it is down.
_Avoid_: Accuracy, PnL, utility

**Always-down baseline**:
The contest score obtained by predicting down for every stock on every date; the reference any model must beat, because intraday returns drift negative.
_Avoid_: All-zero baseline

**Market component**:
The average intraday return across target stocks on a prediction date; the part of each stock's intraday return shared by all of them.
_Avoid_: Beta, index return

**Relative component**:
A stock's intraday return minus the market component of the same prediction date.
_Avoid_: Alpha, excess return, residual

**Overnight gap**:
A stock's adjusted open on a trading date over its adjusted close on the previous trading date, minus one. It is unknown at the information cutoff.
_Avoid_: Gap return, opening return

## Universe and time

**Target stock**:
One of the 50 stocks whose directions are submitted (`Top50@Y` in the data info file).
_Avoid_: Submission stock

**Universe stock**:
Any of the 100 stocks with market history in the data (`Top100@Y`), a superset of the target stocks.

**Prediction date**:
The trading date whose intraday return a direction refers to.

**Information cutoff**:
The latest moment whose data may inform a prediction: 09:00 Taiwan time on the prediction date, before the market opens. The prediction date's own open price is not yet known.
_Avoid_: As-of date

## Evaluation

**Validation fold**:
A contiguous block of past trading dates used to compare model configurations, with training restricted to dates before it.

**Final test period**:
Trading dates from June 1, 2026 on, scored once per model to report out-of-sample results and never used to choose configurations.
_Avoid_: Holdout, test fold

**Ablation**:
A comparison of two configurations that differ in exactly one choice, scored on the same validation folds.

## Competition

**Phase 1**:
The static backtest over the 84 fixed prediction dates from June 1 to September 30, 2026.
_Avoid_: Static phase test set

**Session**:
One Phase 2 prediction date with its own submission window that closes at the information cutoff.

**Competition day**:
A session inside the scored Phase 2 window (November 16 to December 24, 2026); ranking uses the total contest score over competition days.
