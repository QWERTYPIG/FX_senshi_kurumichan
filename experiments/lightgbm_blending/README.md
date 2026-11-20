# Parameter blending experiment

Use the existing 14 features and cached parameter-search OOF predictions to compare
equal-weight Top-3, Top-5 and all-12 regression blends. Average raw predicted returns
first, then submit 1 if the average is positive, otherwise 0. This differs from
majority-voting the members' binary directions.

```sh
.venv/bin/python experiments/lightgbm_blending/run.py
.venv/bin/python -m unittest discover -s experiments/lightgbm_blending -p 'test_*.py' -v
```

The default search run is read from production `artifacts/best_params.json`. Supply
`--search-run /path/to/search/run` to evaluate another completed search. Cached data,
features and training code must still match the original search input digest; if
not, rerun `model/lightgbm_baseline/tune.py` first. The script does not modify the
production model or submission settings.

## Evaluation

The first two validation folds initialize parameter ranking. Each subsequent fold
ranks members using only earlier folds' average competition score, with the same
standard-deviation / MSE / trial-order tie breakers as parameter search. Thus no
fold selects its own members using its own labels. Compare Top-1 and each equal-weight
blend on exactly the same remaining folds. The best blend size is fixed before
evaluating final test; its final members are selected using all pre-test validation
folds. Only those members are fitted for final-test inference and full-history
experimental model artifacts. The current production single model's stored test
predictions are used as the reference (not its full-history fitted model).

This improves the chronology of member selection, but selecting the best of three
blend families still biases validation scores. The parameter grid, features and test
dates were examined in earlier experiments. Test results are a diagnostic, not a
new blind evaluation or proof of reduced overfitting. Ensembles with highly correlated
members may gain little diversity. No new features or learned continuous weights
are used.

## Artifacts

`results/<run_id>/` (ignored by Git) contains `configuration.json`, `selection.json`
(written before test evaluation), `comparison.json`, chronological per-row predictions,
member prediction correlations, final-test predictions and `blend_model/`.
Each `member_XX.txt` in that folder is a full-history LightGBM regression model;
`metadata.json` records the members, weights, features and training cutoff.
The production single-model `predict.py` does not load this ensemble format.
To deploy it later, add ensemble inference and matching retraining behavior rather
than copying a member over `artifacts/model.txt`.
