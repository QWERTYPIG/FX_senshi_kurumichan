# Return-weighted logistic regression

Manager interface (`usage.txt`): first line `python predict.py`, second line `python train.py --train-end latest --validation-start 2026-09-01`. Uses the manager's Python environment; install `requirements.txt` in that environment. No network access, submission, or changes to source data are performed.

`MANAGER_ROOT`, `MANAGER_BASIC_DATA_DIR`, `MANAGER_DATA_DIR`, and `MANAGER_TEMPLATE` are honored. Prediction writes `prediction.csv` in the command's current directory, retaining the exact template stock/date order and binary values. Supports Phase 1 multi-date templates and Phase 2 daily templates. Optional arguments: `--data-dir`, `--template`, `--date`, `--output`, `--artifacts-dir`, `--max-history-age-days` (default 10 calendar days). Missing artifacts require training first.

The included `artifacts/` contains the model retrained through November 20, 2026, with L2 strength 1 and no recency decay, copied from `FinalProject/logistic_regression/artifacts_2026-11-20/`. It was selected from nine strengths (1e-6 through 100) on September 1–November 20, 2026 validation by the official mean signed-return score: 0.0043326002, versus 0.0025227110 for always down. Direction accuracy is 55.2963%. These are selection-set results, not independent test results. The final refit uses 83,524 rows across the 50 stocks, including the validation period. Prediction dates must be after November 20; this deployed model cannot generate Phase 1 historical predictions. The positive ridge magnitude estimate times `(2 * weighted_probability - 1)` produces signed fractional-return estimates; binary submission values use their signs. `predictions.csv` contains the detailed estimates and input source dates. Load only trusted pickle artifacts.

Features are strictly prior-date stock returns, momentum, volatility, volume, institutional flows, technical indicators, global indicators, weekday, source ages, and stock identity. Median imputation and scaling are fit on training data. Both logistic and ridge coefficients use the selected L2 penalty, excluding intercepts. Invalid nonpositive stock prices are excluded; nonpositive global index/VIX placeholders are treated as missing. Negative TX_OI and institutional flow values are valid. Inputs must contain the same schema as the project data.

Manager retraining uses the latest available valid price labels and validates from September 1, 2026. It jointly searches nine L2 penalties and recency half-lives of 0, 63, 126, and 252 observed trading dates, then refits on all allowed dates. `--train-end latest` resolves the cutoff from merged stock history on each run, excluding nonpositive price placeholders. Prediction rejects template dates on or before the actual last training label date. Running `python train.py` directly without arguments retains the Phase 1 defaults (May 31 cutoff, January 1 validation start) and searches recency half-lives as described below.

Artifacts: `artifacts/model.pkl`, `metadata.json`, `metrics.json`, `regularization_search.csv`, and `run_config.json` (source retraining configuration). Training reads original history through August 2026 from `MANAGER_BASIC_DATA_DIR` and recent data starting September 2026 from `MANAGER_DATA_DIR`. It merges stock and global histories in memory, with recent records taking precedence on overlapping dates, then applies the training cutoff. Source files are never modified. Override these directories with `--basic-data-dir` and `--data-dir`. Without a basic directory, training retains the single-directory behavior and expects complete history in `--data-dir`. Prediction still reads `--data-dir` only; Phase 1 prediction requires combined history. See [manager data directories](docs/manager_data_directories.md).

Standalone examples from the repository root:

```sh
python model/logistic_regression/train.py --basic-data-dir /path/to/original/data --data-dir /path/to/recent/data
python model/logistic_regression/predict.py --data-dir /path/to/data --template /path/to/sample.csv --output model/logistic_regression/prediction.csv
```

The default output is relative to the working directory, as required when the manager executes inside the model directory. Global date fields have no release timestamps; strict prior-date joins do not independently verify publication latency or point-in-time adjusted-price construction.

Training uses the original single chronological split and jointly tests recency half-lives (0, 63, 126, 252 observed trading dates) and L2 penalties. Zero disables decay. See [recency weighting](docs/recency_weighting.md) for implementation, commands, and results.

To retrain using the deployed model’s cutoff, validation split, and no-decay search:

```sh
python model/logistic_regression/train.py --basic-data-dir /path/to/original/data --data-dir /path/to/recent/data --train-end 2026-11-20 --validation-start 2026-09-01 --half-lives 0
```

The manager’s `usage.txt` instead uses `--train-end latest --validation-start 2026-09-01` and retains the full decay search; data directories come from the manager environment.
