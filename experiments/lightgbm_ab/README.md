# Lag feature A/B evaluation

Evaluate `lightgbm_ab_experiment.zip` without replacing the manager's production model.
`A` uses the existing 14 features. `B` adds the second and third previous valid trading
days' intraday and close-to-close returns (4 more features). The baseline already
contains 5/10-day return means, volatility and technical indicators.

Run from the repository root:

```sh
.venv/bin/python experiments/lightgbm_ab/run.py --train-end 2026-11-06
.venv/bin/python -m unittest discover -s experiments/lightgbm_ab -p 'test_*.py' -v
```

The runner extracts only the reviewed ZIP files into ignored `results/<cutoff>/source/`,
passes explicit source paths, and hashes all stock inputs, the stock template and ZIP.
Each A/B pair uses identical cutoffs, stocks, labels, parameters, chronological splits,
inner early stopping and final 20-day test. A single fixed random seed is sufficient
here because both configurations use deterministic full-row/full-column training.

The primary comparison uses 5 x 20-day outer validation folds. The longer-period
check uses 10 x 20-day folds; its latest 5 folds overlap the primary comparison and
must not be treated as an independent replication. Inner early-stopping periods
are 20 dates and the first training fold requires at least 252 dates.

The decision rule is recorded before fitting: switch to B only if its mean official
validation score improves and it wins a strict majority of all folds in BOTH
comparisons. Ties do not count as wins. This is a conservative engineering acceptance
rule, not a statistical significance test. The final test comparison is disclosed
after recording the CV-based choice. Historical test dates have already been examined
in previous project iterations, so this is a held-out diagnostic, not a fresh blind test.

`results/<cutoff>/` contains:

- `configuration.json`: settings, decision rule and input digest.
- `primary/{A,B}/`, `robustness/{A,B}/`: trained artifacts and raw metrics.
- `primary/paired_folds.csv`, `robustness/paired_folds.csv`: paired fold scores.
- `*/paired_validation_daily.csv`: paired daily contributions (B minus A).
- `paired_test_daily.csv`: final test daily comparison.
- `selection.json`: CV-based decision before test disclosure.
- `comparison.json`: complete results and test diagnostics.

Production source/artifacts are only changed separately if the acceptance rule passes.
