# Recency weighting experiment
> worse result than original logistic regression
## Reverted validation adjustment

Removed the five expanding-history validation windows, their CLI option,
window-specific metrics, tests, and the previous validation document. Restored
the original single chronological split: train on labels before January 1,
2026, validate on January–May 2026, and refit the selected configuration on all
eligible labels through May 31 (actual last label May 29). Dates after that
cutoff do not participate in selection or fitting.

The old `artifacts/chronological_validation` directory is retained only as a
historical experiment; the new run uses `artifacts/recency_weighting`.

## New weighting

For classifier fitting, each row receives:

```
raw_weight = abs(intraday_return) * 2 ** (-age / half_life)
weight = raw_weight / mean(raw_weight)
```

Age is the number of distinct observed training dates between the row's date
and the latest date in that fit. Weekends and gaps do not add decay. All rows
on the same date have the same decay factor. The training subset alone supplies
the date ranks; validation/future dates do not set the decay reference date.
The final refit computes weights again relative to its own latest training date.

`--half-lives` defaults to `0 63 126 252`. Zero disables decay and reproduces
the original absolute-return weights. Mean-one normalization preserves the
overall loss scale relative to L2 regularization. Existing median imputation,
scaling, clipping, features, magnitude ridge fit, and prediction threshold remain
unchanged. In particular, only the logistic loss/gradient use recency weights;
the positive magnitude estimate does not affect binary prediction signs.

The trainer compares all four half-lives with all nine existing L2 penalties
(36 combinations) on the same validation rows using the original official
signed-return score. Validation metrics themselves are not recency weighted.
Exact score ties prefer no decay, then the larger penalty, then the longer
half-life. The winning configuration is refit for deployment.

`regularization_search.csv` and `metrics.json` record every trial and its
half-life. `metadata.json` records `selected_half_life`, `selected_penalty`,
and `validation_strategy: single_split`. The pickle also records its half-life;
prediction requires no new input or feature schema.

## Reproduce

From the model directory:

```sh
python -m unittest discover -s logistic_regression/tests -v
OPENBLAS_NUM_THREADS=1 python logistic_regression/train.py --data-dir ../../FinalProject/data --artifacts-dir logistic_regression/artifacts/recency_weighting
```

Use `--half-lives 0` for the original no-decay search. Use `--half-lives 126`
to test only a 126-date half-life. Tests check exact no-decay weights, half-life
ratios, same-date consistency, ordering and date-offset invariance, invalid
inputs, and a synthetic regime change that reverses the classifier's decision.

The validation period has already been used for model development. Scores here
are selection-set comparisons, not independent evidence of future improvement.
Phase 1 labels are not used to select a half-life or penalty.

## Results from November 9, 2026

All 36 configurations were evaluated on 4,750 stock-days from January 2
through May 29, 2026. Best configuration within each half-life:

| Half-life (observed trading dates) | L2 penalty | Validation score |
| --- | --- | --- |
| 0 (no decay) | 0.1 | 0.0016091863 |
| 63 | 10 | 0.0015498450 |
| 126 | 1 | 0.001853 (rounded) |
| 252 | 1 | 0.0019547854 |

The selected 252-date model improves the selection-set score by approximately
0.0003456, or 3.46 basis points per stock-day, over the best no-decay model.
The always-down baseline is 0.0015498450. The no-decay search reproduces the
original selected penalty and score. Exact trial results are saved under
`logistic_regression/artifacts/recency_weighting/regularization_search.csv`.

Refit the winner on 77,624 eligible rows through May 29, then regenerated
`logistic_regression/prediction.csv` for Phase 1 (50 stocks, 84 dates, June 1–
September 30). There are 576 up predictions and 3,624 down predictions.
Detailed estimates are in `logistic_regression/predictions.csv`.

Prediction history combines `../../FinalProject/data` with
`../../FinalProject/new_data/data` under `/tmp/logistic_phase_one_data`, using
recent records on overlaps. Source files were not modified. The temporary
directory must be recreated from those sources if it has been removed.

```sh
python logistic_regression/predict.py --data-dir /tmp/logistic_phase_one_data --template ../../FinalProject/sample_submission.csv --artifacts-dir logistic_regression/artifacts/recency_weighting --output logistic_regression/prediction.csv
```

Verified template shape/order, binary values, and strictly prior-date feature
sources for every prediction. All six unit/integration tests passed, including
single-split boundaries, cutoff exclusion, no-decay tie preference, and final
refit. `git diff --check` passed. Existing default artifacts remain intact; the
new model requires the explicit artifact path shown above.
