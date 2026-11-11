# Return-weighted logistic regression

Manager interface (`usage.txt`): first line `python predict.py`, second line `python train.py`. Uses the manager's Python environment; install `requirements.txt` in that environment. No network access, submission, or changes to source data are performed.

`MANAGER_ROOT`, `MANAGER_BASIC_DATA_DIR`, `MANAGER_DATA_DIR`, and `MANAGER_TEMPLATE` are honored. Prediction writes `prediction.csv` in the command's current directory, retaining the exact template stock/date order and binary values. Supports Phase 1 multi-date templates and Phase 2 daily templates. Optional arguments: `--data-dir`, `--template`, `--date`, `--output`, `--artifacts-dir`, `--max-history-age-days` (default 10 calendar days). Missing artifacts require training first.

The included `artifacts/` contains the existing model trained through May 29, 2026, with L2 strength 0.1. It was selected from nine strengths (1e-6 through 100) on January–May 2026 validation by the official mean signed-return score: 0.0016091863, versus 0.0014524435 at strength 0.01. These are selection-set results. Training uses 77,624 rows across the 50 stocks. The positive ridge magnitude estimate times `(2 * weighted_probability - 1)` produces signed fractional-return estimates; binary submission values use their signs. `predictions.csv` contains the detailed estimates and input source dates. Load only trusted pickle artifacts.

Features are strictly prior-date stock returns, momentum, volatility, volume, institutional flows, technical indicators, global indicators, weekday, source ages, and stock identity. Median imputation and scaling are fit on training data. Both logistic and ridge coefficients use the selected L2 penalty, excluding intercepts. Invalid nonpositive stock prices are excluded; nonpositive global index/VIX placeholders are treated as missing. Negative TX_OI and institutional flow values are valid. Inputs must contain the same schema as the project data.

Default retraining retains the Phase 1 holdout: labels through May 31, 2026 only, validation from January 1, 2026. It tunes the same nine penalties, then refits on all allowed dates. Override with `--train-end YYYY-MM-DD --validation-start YYYY-MM-DD` for later deployment; choose a validation start before the training cutoff. Prediction rejects template dates on or before the actual last training label date. Default retraining therefore never uses June–September labels. Earlier observed Phase 1 rows can supply features for subsequent Phase 1 dates without refitting.

Artifacts: `artifacts/model.pkl`, `metadata.json`, `metrics.json`, `regularization_search.csv`. Training reads original history through August 2026 from `MANAGER_BASIC_DATA_DIR` and recent data starting September 2026 from `MANAGER_DATA_DIR`. It merges stock and global histories in memory, with recent records taking precedence on overlapping dates, then applies the training cutoff. Source files are never modified. Override these directories with `--basic-data-dir` and `--data-dir`. Without a basic directory, training retains the single-directory behavior and expects complete history in `--data-dir`. Prediction still reads `--data-dir` only; Phase 1 prediction requires combined history. See [manager data directories](docs/manager_data_directories.md).

Standalone examples from the repository root:

```sh
python model/logistic_regression/train.py --basic-data-dir /path/to/original/data --data-dir /path/to/recent/data
python model/logistic_regression/predict.py --data-dir /path/to/data --template /path/to/sample.csv --output model/logistic_regression/prediction.csv
```

The default output is relative to the working directory, as required when the manager executes inside the model directory. Global date fields have no release timestamps; strict prior-date joins do not independently verify publication latency or point-in-time adjusted-price construction.

Training uses the original single chronological split and jointly tests recency half-lives (0, 63, 126, 252 observed trading dates) and L2 penalties. Zero disables decay. See [recency weighting](docs/recency_weighting.md) for implementation, commands, and results.
