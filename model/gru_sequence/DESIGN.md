# GRU sequence model

This document records the market-only baseline from [#10](https://github.com/QWERTYPIG/FX_senshi_kurumichan/issues/10), part of [#9](https://github.com/QWERTYPIG/FX_senshi_kurumichan/issues/9). It is a working model, not a selected configuration or a claim of beating always-down.

## Data and information cutoff

- Historical `info.json` supplies the universe (`Top100@Y`). In the provided history it matches all 100 `market/<stock>.csv` files. Use the same stock order for each date. The manager's template, not the universe list, supplies the target stocks, output order and prediction dates.
- Merge stock rows by `(id, date)` in memory. The daily update wins on overlap before validation; an invalid update does not revive the historical row it replaced. Drop rows with nonpositive adjusted OHLC or missing/nonfinite market values. Do not write to either source. An absent update directory is allowed for historical development.
- For each stock and prediction date, take its last `window_length` valid trading rows **strictly before** that date. The competition's 09:00 information cutoff excludes even the prediction date's adjusted open. Missing dates are not fabricated. A single `present` mask marks whether a stock has a full window, and, in training, also whether it has a valid row for the label date. Each date has a fixed universe stock axis even when some stocks are absent.
- Use all 19 market columns in the source schema. Following [ADR 0001](../../docs/adr/0001-feature-changes-require-walk-forward-evidence.md), no column was selected for its perceived usefulness. `date` is an index and `y` is not an input. Divide adjusted OHLC, SMA, MACD, ATR and all Bollinger bands by the adjusted close of the last day in the window, as specified in #9 (Qlib's Alpha360 similarly uses latest-close normalization for price histories). Z-score the remaining channels independently over the stock's window, mapping constant channels to zero. The statistics never use another window or a label day. Global inputs are staged for #11, not rejected by a feature decision.

## Network and training objective

One shared GRU encodes every stock's window, and an MLP maps its final state to a scalar `f`. Sharing lets the model learn common patterns over all universe stocks without using a stock's identity. A recurrent encoder is a starting point for using time-ordered data directly: the Alpha360 benchmark cited in #9 reports stronger results for a GRU than a temporal Transformer, and Fischer & Krauss (2018, *European Journal of Operational Research*) found recurrent models useful on stock direction. This is motivation, not validation on this competition's data.

One batch is one trading date, and only present stocks contribute to the loss. The [competition score](../../docs/final-project-statement.md#evaluation) multiplies a signed direction by the realized intraday return. Replacing the hard sign with `tanh(f)` gives the differentiable loss `-mean(tanh(f) * (adjusted_close / adjusted_open - 1))`. Recompute the return on the label date rather than using `y`, which cannot encode the return's magnitude. At prediction, direction is `1` only when `f > 0`; there is no tuned threshold.

The manager training command uses the exact JSON settings, with fixed epochs, one seed and deterministic single-threaded CPU execution. The checked-in values are provisional. Global inputs, attention and embeddings, prediction safeguards, walk-forward selection and multi-seed refitting are later tickets. In particular, do not treat predictions for dates on or before `last_training_label_date` as out-of-sample.

## Running and artifacts

Install `model/gru_sequence/requirements.txt` in the manager's Python environment (the Linux pin selects the CPU wheel). The manager discovers `usage.txt`. For an isolated run from `model/gru_sequence/`, set the [manager variables](../../docs/manager-model-interface.md#inputs-and-output), then run:

```sh
python train.py --config /path/to/config.json --artifacts-dir /path/to/run
python predict.py --artifacts-dir /path/to/run
```

Without overrides, training reads `config.json` beside the model and both commands use its local `artifacts/` directory. Training saves `model.pt` and `metadata.json` (settings used, universe stock IDs, actual first/last training label dates and PyTorch version). Prediction uses those saved settings, rewrites `MANAGER_OUTPUT`, and writes a sibling `<output-stem>_detail.csv` with `id,date,f,latest_market_date` for each target stock-date. Stocks without a complete window receive direction `0` with blank detail values because no score or source window was used.

The synthetic subprocess tests exercise both manager commands and templates, date causality, deterministic retraining, config overrides, the return objective, update precedence and expansion without modifying the source files. They do not estimate contest performance.
