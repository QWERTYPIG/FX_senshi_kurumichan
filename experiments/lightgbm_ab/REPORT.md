# Lag feature A/B evaluation — 2026-11-17

Keep the existing production design. The four additional lag features did not
improve average out-of-sample competition score on either validation horizon.
This does not prove that lag features never help; it rejects this particular
feature addition under the current data, parameters and training procedure.

## Experiment

- **A:** current 14 features. **B:** A plus `intraday_lag2`, `intraday_lag3`,
  `return_lag2`, `return_lag3` (18 total).
- The existing baseline already has 5/10-day return means, 5-day volatility and
  technical indicators. B adds explicit day-by-day lag values rather than
  introducing historical information for the first time.
- Both use 100 training stocks, 50 target stocks, identical valid stock-date rows,
  LightGBM hyperparameters and causal feature transformation. The A primary run
  reproduced the existing production `metrics.json` exactly.
- Source: `basic_data/market/` plus `data/market/`, with live rows overriding overlaps.
- Cutoff **2026-11-06**, the latest valid date in the local files on evaluation day.
  No conclusions cover 2026-11-09 through 2026-11-17; local prices for them are absent.
- Primary: 5 x 20 validation dates (2026-05-14 through 2026-10-06), 5,000 stock-days.
- Longer horizon: 10 x 20 dates (2025-12-08 through 2026-10-06), 9,999 stock-days.
  One stock-day has no valid label and is excluded identically from both variants.
- Inner early stopping: 20 prior dates, MSE, patience 30. Each outer validation
  period is excluded from model fitting and round selection for its own fold.
- Final test: 2026-10-07 through 2026-11-06, 20 dates / 1,000 stock-days.
- Fixed parameters: regression objective, learning rate 0.03, at most 500 rounds,
  7 leaves, depth 4, min child samples 50, L2 regularization 1, seed 42, 4 threads,
  deterministic column-wise training. Final rounds: median inner best iterations.

## Competition score

Scores are mean signed realized returns per stock-day, using return fractions.
B − A is paired on exactly the same stocks and dates.

| Validation horizon | A | B | B − A | B wins / ties / losses |
| --- | ---: | ---: | ---: | ---: |
| 5 folds / 100 validation dates | 0.000435525 | 0.000264780 | -0.000170746 | 1 / 3 / 1 |
| 10 folds / 200 validation dates | 0.000624329 | 0.000521895 | -0.000102435 | 4 / 4 / 2 |

Primary validation accuracy: A **53.24%**,
B **53.02%**. Primary MSE:
A `0.000744988`, B `0.000745561`.
B changed 263 of 5,000 validation directions.

The validation-only acceptance rule, recorded before fitting, required higher mean
score AND a strict majority of fold wins in BOTH horizons. B failed both mean-score
checks. The longer horizon contains the same latest five folds as the primary run;
these are overlapping checks, not independent replications or significance tests.

### Per-fold comparison over the longer horizon

| Fold | Validation dates | A score | B score | B − A |
| --- | --- | ---: | ---: | ---: |
| 1 | 2025-12-08 – 2026-01-06 | 0.001602788 | 0.001117484 | -0.000485304 |
| 2 | 2026-01-07 – 2026-02-03 | 0.000992325 | 0.001001004 | 0.000008679 |
| 3 | 2026-02-04 – 2026-03-13 | -0.000956958 | -0.000847475 | 0.000109483 |
| 4 | 2026-03-16 – 2026-04-14 | 0.001578482 | 0.001774624 | 0.000196142 |
| 5 | 2026-04-15 – 2026-05-13 | 0.000850009 | 0.000850009 | 0.000000000 |
| 6 | 2026-05-14 – 2026-06-10 | -0.001060294 | -0.001060294 | 0.000000000 |
| 7 | 2026-06-11 – 2026-07-09 | 0.002519874 | 0.002519874 | 0.000000000 |
| 8 | 2026-07-13 – 2026-08-07 | 0.002551439 | 0.001586752 | -0.000964687 |
| 9 | 2026-08-10 – 2026-09-04 | -0.002845193 | -0.002845193 | 0.000000000 |
| 10 | 2026-09-07 – 2026-10-06 | 0.001011800 | 0.001122759 | 0.000110959 |

### Final test diagnostic (not used to select the feature set)

| Metric | A | B |
| --- | ---: | ---: |
| Score | 0.000795934 | 0.000875934 |
| Accuracy | 54.10% | 54.20% |
| MSE | 0.000749466 | 0.000749609 |
| MAE | 0.020037837 | 0.020058331 |

B's score improves by `0.000080000` on this 20-day test,
but this small localized improvement does not overturn the lower validation
scores over 100 and 200 dates. Test dates have been examined in previous project
iterations, so they are not a newly blind test. No claim of statistical significance
is made. Feature importance is descriptive and does not establish incremental value.

## Code and artifact review

- Lag 2/3 use `shift(1)`/`shift(2)` on historical source rows, then join to strictly
  later target dates. This correctly gives t−2/t−3 and introduces no same-day lookahead.
- Invalid price rows are filtered before lagging, so a lag means the previous
  **valid observation**, not necessarily the immediately preceding exchange session.
  This convention is shared with production and should be kept in mind for missing data.
- The ZIP resets history defaults to `ROOT/basic_data`, losing the current
  `MANAGER_BASIC_DATA_DIR` override, and changes default artifact paths to
  `artifacts_A`/`artifacts_B`. It should not be blindly copied over production.
- Production code, model artifacts, manager configuration and saved predictions
  were not changed. The ZIP remains intact.
- Three feature tests pass: A parity, expected trading-day lags and invariance to
  same-day/future price modifications. All A/B paired dates, labels and fold
  boundaries match. Input fingerprints were unchanged throughout the experiment.

## Reproduce

```sh
.venv/bin/python experiments/lightgbm_ab/run.py --train-end 2026-11-06
.venv/bin/python -m unittest discover -s experiments/lightgbm_ab -p 'test_*.py' -v
```

Artifacts live in `results/2026-11-06/` (ignored by Git): configuration, source
copies, all four fitted runs, raw logs, paired fold/daily comparisons, selection
and complete JSON results. Reproduction requires the same source data bytes;
future live-data revisions can change historical results even with a fixed cutoff.
Combined ZIP/data/template SHA-256:
`3ad7f59bb2806d2f822b8b84cada040627214626a32c6518e096137c45fd320d`.
