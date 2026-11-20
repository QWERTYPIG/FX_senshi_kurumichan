# GRU sequence model

This document records the GRU sequence model from [#9](https://github.com/QWERTYPIG/FX_senshi_kurumichan/issues/9). The checked-in settings are provisional. Real-data selection and final-test reporting are tracked in [#16](https://github.com/QWERTYPIG/FX_senshi_kurumichan/issues/16).

## Data and information cutoff

- Historical `info.json` supplies the universe (`Top100@Y`). In the provided history it matches all 100 `market/<stock>.csv` files. Use the same stock order for each date. The manager's template, not the universe list, supplies the target stocks, output order and prediction dates.
- Merge stock rows by `(id, date)` in memory. The daily update wins on overlap before validation; an invalid update does not revive the historical row it replaced. Drop rows with nonpositive adjusted OHLC or missing/nonfinite market values. Do not write to either source. An absent update directory is allowed for historical development.
- For each stock and prediction date, take its last `window_length` valid trading rows **strictly before** that date. The competition's 09:00 information cutoff excludes even the prediction date's adjusted open. Missing dates are not fabricated. A single `present` mask marks whether a stock has a full window, and, in training, also whether it has a valid row for the label date. Each date has a fixed universe stock axis even when some stocks are absent.
- Use all 19 market columns in the source schema. Following [ADR 0001](../../docs/adr/0001-feature-changes-require-walk-forward-evidence.md), no column was selected for its perceived usefulness. `date` is an index and `y` is not an input. Divide adjusted OHLC, SMA, MACD, ATR and all Bollinger bands by the adjusted close of the last day in the window, as specified in #9 (Qlib's Alpha360 similarly uses latest-close normalization for price histories). Z-score the remaining channels independently over the stock's window, mapping constant channels to zero. The statistics never use another window or a label day.

### Global inputs

- Read `global/global_data.csv` from history and, when present, daily updates. Merge by date in memory; the update row wins on overlap, even when a cell is blank. Carry each series independently from its last finite observation (positive for index levels). An absent row or cell does not borrow from the future or drop a Taiwan trading date.
- **Information cutoff (09:00 Taiwan):** for Taiwan trading date `d`, use `^GSPC`, `^IXIC`, and `^VIX` dated **on or before `d`**, and `^TWII`, `^KS11`, and `TX_OI` dated **strictly before `d`**. In the historical data, #9 verified that the row labeled `2024-07-02` holds the S&P 500 close of `5475.09` from the July 1 US session, which closed before the July 2 Taiwan cutoff. The code applies this convention to updates too; verify their timestamp convention against actual daily files before relying on it for live prediction.
- The Taiwan calendar is the union of valid universe-stock trading dates. First carry forward eligible levels, then compute each index's log return as the difference in log levels between consecutive Taiwan cutoffs. An unchanged carried level gives a zero return; during a Taiwan closure, changes in the eligible levels accumulate until the next trading date. With no prior level to compare, use a neutral zero. Z-score `TX_OI` over the observed or carried values within each stock's window; leading gaps don't affect the statistics and map to zero, as do constant windows.
- **Window and today vector:** append all six global channels to each stock row at that row's cutoff. The head gets a separate today vector of US index log returns from the previous Taiwan cutoff to the prediction date's cutoff, even if no market row yet exists for that date. This preserves eligible US observations that arrive after the window's last cutoff.
- **Provenance:** each global trace is the actual CSV row date of the latest aligned level, including a carried level behind a zero return. Take US traces from the today vector and non-US traces from the last row of that stock's window. Leave the trace blank if no level was available.
- **Evidence:** all six series enter under ADR 0001's default-input rule. The timing check establishes availability, not forecast value. The [synthetic command tests](tests/test_pipeline.py) check alignment, causality, holiday carry-forward and provenance. #11 ran no real-data ablation or contest scoring.

## Network and training objective

One shared GRU encodes each stock's market and global window, learning common patterns across the universe. The Alpha360 benchmark cited in #9 reports stronger results for a GRU than a temporal Transformer, and Fischer & Krauss (2018, *European Journal of Operational Research*) found recurrent models useful on stock direction.

- **Cross-stock attention:** Each stock's GRU summary receives a residual update from one self-attention layer over the present universe stocks on that date. MASTER (Li et al., AAAI 2024) motivates aggregating inter-stock information to model the market and relative components together. Zeng et al. (AAAI 2023) motivate keeping the ordered time axis recurrent; the cross-section has no inherent order. The batch's presence mask excludes absent stocks as queries, keys, and values.
- **Stock ID embedding:** An embedding indexed by the saved universe order lets the model learn persistent stock-specific drift. The historical per-stock mean intraday return correlated 0.53 between 2020–2023H1 and 2023H2–2026 (see #9). The MLP takes the summary, embedding and today vector concatenated together and produces scalar `f`.

Attention and embedding are enabled by default and can be disabled separately for ablations. The cited evidence motivates testing them, not adoption without walk-forward results. Retrain checkpoints created before these settings were added.

One batch is one trading date, and only present stocks contribute to the loss. The [competition score](../../docs/final-project-statement.md#evaluation) multiplies a signed direction by the realized intraday return. Replacing the hard sign with `tanh(f)` gives the differentiable loss `-mean(tanh(f) * (adjusted_close / adjusted_open - 1))`. Recompute the return on the label date rather than using `y`, which cannot encode the return's magnitude. At prediction, direction is `1` only when `f > 0`; there is no tuned threshold.

Weekly training uses fixed settings and refits one model per configured deployment seed on all usable labels, including newly available labels. Prediction averages their continuous `f` values before applying the sign rule; it does not vote on per-seed directions. Weekly training reads neither the `search` nor the `evaluation` settings.

## Running and artifacts

Install `model/gru_sequence/requirements.txt` in the manager's Python environment (the Linux pin selects the CPU wheel). The manager discovers `usage.txt`. For an isolated run from `model/gru_sequence/`, set the [manager variables](../../docs/manager-model-interface.md#inputs-and-output), then run:

```sh
python train.py --config /path/to/config.json --artifacts-dir /path/to/run
python predict.py --artifacts-dir /path/to/run
```

Without overrides, training reads the local `config.json`, and both commands use the local `artifacts/` directory. `metadata.json` records the model settings (including fixed epochs), seed-to-checkpoint list, universe stock IDs, actual first/last training label dates, and runtime versions. Weekly training publishes a new ensemble directory before switching metadata to it; previous weight sets remain on disk. A config that omits `deployment_seeds` is a single-seed config for standalone runs. Retrain any market-only checkpoint from #10: the wider GRU and head are not compatible with its weights.

Prediction uses the saved settings and rewrites `MANAGER_OUTPUT`. For a cutoff audit, inspect the sibling `<output-stem>_detail.csv` for mean `f` and the actual latest input dates. Stocks without a complete window receive direction `0` and blank scores and traces.

The synthetic subprocess tests exercise both manager commands and templates, date causality, deterministic retraining, config overrides, the return objective, update precedence and expansion without modifying the source files. They do not estimate contest performance.

## Walk-forward evaluation

[#14](https://github.com/QWERTYPIG/FX_senshi_kurumichan/issues/14) separates validation from final-test reporting. Both use the manager data directories, deterministic single-threaded CPU training, and `Top50@Y` in historical `info.json` to identify target stocks. The submission template is not needed.

**Validate one configuration.** Run `python evaluate.py validate --config CONFIG --artifacts-dir RUN`. The inclusive quarter boundaries in `config.json` define eight validation folds through 2026-05-29. Training expands to include all usable earlier universe-stock labels. Reserve the last 60 trading dates of that training calendar for inner early stopping; the outer fold does not choose epochs. Train up to `epochs` and monitor mean per-date smooth loss, retaining the earliest minimum and stopping after `patience` epochs without improvement. Reinitialize from the same seed and refit on the full fold training window for the chosen epoch count before scoring. `validation_metrics.json` and `validation_predictions.csv` audit the splits, per-fold best epochs, scores, and predictions.

### Select and freeze

Run `python search.py --config CONFIG --artifacts-dir RUN` in a fresh output directory. Search uses only the validation path and limits input history to the end of the last validation fold. Use a stable snapshot of both data directories for the run.

1. **Grid:** Try every configured window length, layer count and hidden size with attention and embedding on. For each trial, average the three seed contest scores within each fold, then average the fold scores with equal weight. An exact grid tie keeps the first candidate in config product order (window, layers, hidden size).
2. **Ablations:** Turn off attention and embedding separately, both against the same grid winner with matching fold and seed IDs. Under [ADR 0001](../../docs/adr/0001-feature-changes-require-walk-forward-evidence.md), a variant qualifies only if its mean contest score increases and strictly more than half the folds improve; tied folds do not count. If both qualify, adopt the higher-scoring tested variant (attention-off first on an exact tie). The combined ablation is untested and cannot be selected. `gate_eligible` and `adopted` record these separate decisions.
3. **Freeze:** Take the floor of the median of all selected trial's seed-fold best epochs, at least 1. `selected_config.json` holds the fixed model settings, that epoch count and the five deployment seeds. It is directly usable with `train.py --config` and does not overwrite the checked-in config.

A completed run has `search_metrics.json` (all fold/seed scores and best epochs, comparisons and decisions, original and selected configs, stock lists, evaluated history range and runtime versions) and the original validation metrics and OOS predictions under `<trial>/<seed>/`. Keep those artifacts with any real-data selection record.

### Report the final test

After selection is frozen, run `python evaluate.py final-test --config FROZEN_CONFIG --artifacts-dir RUN` once for reporting. It refits the same five-seed ensemble with the frozen epoch count, but only on labels before `final_test_start`. It then scores the sign of mean `f` on available final-test labels without further fitting. It reads neither validation artifacts nor fold settings, and refuses to overwrite final-test artifacts in the same output directory. Keep final-test results out of selection.

All three scores (model, always-up, always-down) use the **same labeled target stock-dates** and the official signed intraday-return formula. A labeled target without usable inputs receives direction `0` and still counts. A missing realized return cannot be scored by any of the three. Inference masks depend on input availability and staleness, not on which stocks will have a realized return. A bounded cache reuses labeled batches across epochs and overlapping fits without keeping a copy per fold.

## Prediction safeguards

- **Stale stock:** Compare each universe stock's latest valid market row strictly before the prediction date to that date. More than 10 calendar days old (or no prior valid row) is stale; exactly 10 days is eligible. Warn for each stale stock and mark it absent before inference, so it cannot influence other stocks. For a stale target stock, write direction `0` with a blank score and source dates.
- **All stocks stale:** If every universe stock is stale on any requested date, exit nonzero with that date in the error. Leave both prediction files untouched, even if other dates in the template could be scored.
- **In-sample date:** Require every prediction date to be strictly after the checkpoint's `last_training_label_date`. If a date fails that check, exit nonzero before writing either output.
