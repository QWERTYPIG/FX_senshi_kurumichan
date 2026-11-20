# GRU sequence model

This document records the GRU sequence model from [#9](https://github.com/QWERTYPIG/FX_senshi_kurumichan/issues/9). The checked-in config contains the settings selected on historical data for [#16](https://github.com/QWERTYPIG/FX_senshi_kurumichan/issues/16): a 60-trading-day window, one GRU layer of width 64, attention and embedding enabled, and two epochs. See the [experiment record](EXPERIMENTS.md) for the walk-forward results, final test, and protocol deviation.

## Data and information cutoff

- Historical `info.json` supplies the universe (`Top100@Y`). In the provided history it matches all 100 `market/<stock>.csv` files. Use the same stock order for each date. The manager's template, not the universe list, supplies the target stocks, output order and prediction dates.
- Merge stock rows by `(id, date)` in memory. The daily update wins on overlap before validation; an invalid update does not revive the historical row it replaced. Drop rows with nonpositive adjusted OHLC or missing/nonfinite market values. Do not write to either source. An absent update directory is allowed for historical development.
- For each stock and prediction date, take its last `window_length` valid trading rows **strictly before** that date. The competition's 09:00 information cutoff excludes even the prediction date's adjusted open. Missing dates are not fabricated. A single `present` mask marks whether a stock has a full window, and, in training, also whether it has a valid row for the label date. Each date has a fixed universe stock axis even when some stocks are absent.
- Use all 19 market columns in the source schema. Following [ADR 0001](../../docs/adr/0001-feature-changes-require-walk-forward-evidence.md), no column was selected for its perceived usefulness. `date` is an index and `y` is not an input. Divide adjusted OHLC, SMA, MACD, ATR and all Bollinger bands by the adjusted close of the last day in the window, as specified in #9 (Qlib's Alpha360 similarly uses latest-close normalization for price histories). Z-score the remaining channels independently over the stock's window, mapping constant channels to zero. The statistics never use another window or a label day.

### Global inputs

- Read `global/global_data.csv` from history and, when present, daily updates. Merge by date in memory; the update row wins on overlap, even when a cell is blank. Carry each series independently from its last finite observation (positive for index levels). An absent row or cell does not borrow from the future or drop a Taiwan trading date.
- **Prediction cutoff (09:00 Taiwan on `t`):** `^GSPC`, `^IXIC`, and `^VIX` can use dates through `t`; `^TWII`, `^KS11`, and `TX_OI` must use dates strictly before `t`. In the historical data, #9 verified that the row labeled `2024-07-02` holds the S&P 500 close of `5475.09` from the July 1 US session, which closed before the July 2 Taiwan cutoff. Verify this convention on daily updates before relying on it for live prediction.
- **Completed window rows:** append all six globals through each row's date `d`, inclusive. Non-US values from `d` are eligible because the stock row is already a completed trading day (`d < t`), even though that day's non-US values were not known at 09:00 on `d`. Align to the union of valid universe-stock dates. For each index, take the difference in carried log levels from the preceding Taiwan trading date; a repeated level or no level to compare gives zero. Changes during a Taiwan closure accumulate until the next trading date.
- **Open interest:** `TX_OI` is a count, not an index level. Apply the same within-window z-score policy as the other nonprice inputs. Retain carried values after the first usable observation; placeholders before it are not observations, do not enter the statistics and map to zero. Constant windows map to zero.
- **Today vector:** the head gets a separate US-only vector of log-level changes between the preceding Taiwan trading date and the cutoff on `t`, even if market rows for `t` do not yet exist.
- **Provenance:** each global trace is the actual CSV row date of the latest aligned level, including a carried level behind a zero return. Take US traces from the today vector and non-US traces from the last row of that stock's window. Leave the trace blank if no level was available.
- **Evidence:** all six series enter under ADR 0001's default-input rule. Timing establishes availability, not forecast value. The separate today-vector representation was held fixed and was not tested by the #16 ablations. The [synthetic command tests](tests/test_pipeline.py) check alignment, causality, holiday carry-forward and provenance, not forecast quality. [Verification on daily updates remains pending](EXPERIMENTS.md#daily-update-alignment).

## Network and training objective

One shared GRU encodes each stock's market and global window, learning common patterns across the universe. The Alpha360 benchmark cited in #9 reports stronger results for a GRU than a temporal Transformer, and Fischer & Krauss (2018, *European Journal of Operational Research*) found recurrent models useful on stock direction.

- **Cross-stock attention:** Each stock's GRU summary receives a residual update from one self-attention layer over the present universe stocks on that date. MASTER (Li et al., AAAI 2024) motivates aggregating inter-stock information to model the market and relative components together. Zeng et al. (AAAI 2023) motivate keeping the ordered time axis recurrent; the cross-section has no inherent order. The batch's presence mask excludes absent stocks as queries, keys, and values.
- **Stock ID embedding:** An embedding indexed by the saved universe order lets the model learn persistent stock-specific drift. The historical per-stock mean intraday return correlated 0.53 between 2020–2023H1 and 2023H2–2026 (see #9). The MLP takes the summary, embedding and today vector concatenated together and produces scalar `f`.

Attention and embedding remain enabled: neither removal passed the walk-forward adoption rule in the [historical search](EXPERIMENTS.md#selection). This decision does not establish that either component improves the final-test score.

One batch is one trading date, and only present stocks contribute to the loss. The [competition score](../../docs/final-project-statement.md#evaluation) multiplies a signed direction by the realized intraday return. Replacing the hard sign with `tanh(f)` gives the differentiable loss `-mean(tanh(f) * (adjusted_close / adjusted_open - 1))`. Recompute the return on the label date rather than using `y`, which cannot encode the return's magnitude. During optimization, add zero-mean Gaussian noise with standard deviation `input_noise_std` (default 0.1) after input transforms. Sequence perturbations are independent by cell; the today-vector perturbation is shared across stocks on that date. Each training seed controls the noise stream. Early-stopping loss, validation scores and predictions use clean inputs. At prediction, direction is `1` only when `f > 0`; there is no tuned threshold.

Weekly training uses fixed settings and refits one model per configured deployment seed on all usable labels, including newly available labels. Prediction averages their continuous `f` values before applying the sign rule; it does not vote on per-seed directions. Weekly training reads neither the `search` nor the `evaluation` settings.

## Running and artifacts

Install `model/gru_sequence/requirements.txt` in the manager's Python environment (the Linux pin selects the CPU wheel). The manager discovers `usage.txt`. For an isolated run from `model/gru_sequence/`, set the [manager variables](../../docs/manager-model-interface.md#inputs-and-output), then run:

```sh
python train.py --config /path/to/config.json --artifacts-dir /path/to/run
python predict.py --artifacts-dir /path/to/run
```

Without overrides, training reads `config.json` and both commands use `artifacts/`. Ensemble mode requires five distinct `deployment_seeds`. For an isolated one-seed run, pass a separate config with `"deployment_mode": "standalone"` and `seed` instead of `deployment_seeds`.

Training atomically replaces `checkpoint.pt`, which contains the weights for every seed and their metadata together. Prediction reads only that file; an already-open reader can finish using the old checkpoint. `metadata.json` is a readable copy for inspection, not a second input to prediction. Retrain older split-file checkpoints; clean up legacy `ensemble-*` directories and `model.pt` only after stopping any older readers. Weekly training no longer creates weight directories.

The metadata records the actual config, seeds, universe, package versions, and first/last training label dates. `data_date_range` covers valid merged market rows available to the run, including window history; `global_data_date_range` covers accepted source observations, or is null when none exist. These are available ranges, not assertions that every row was used. `epoch_selection` preserves the selected trial's per-fold and per-seed best epochs, or is null when none were supplied. `epoch_source` distinguishes `validation` (the frozen count), `fixed_override` (a different fixed count with retained provenance), and `fixed` (a count without selection provenance).

Prediction uses the saved settings and rewrites `MANAGER_OUTPUT`. For a cutoff audit, inspect the sibling `<output-stem>_detail.csv` for mean `f` and the actual latest input dates. Stocks without a complete window receive direction `0` and blank scores and traces.

The synthetic subprocess tests exercise both manager commands and templates, date causality, deterministic retraining, config overrides, the return objective, update precedence and expansion without modifying the source files. They do not estimate contest performance.

## Walk-forward evaluation

[#14](https://github.com/QWERTYPIG/FX_senshi_kurumichan/issues/14) separates validation from final-test reporting. Both use the manager data directories, deterministic single-threaded CPU training, and `Top50@Y` in historical `info.json` to identify target stocks. The submission template is not needed.

**Validate one configuration.** Run `python evaluate.py validate --config CONFIG --artifacts-dir RUN`. The inclusive quarter boundaries in `config.json` define eight validation folds through 2026-05-29. Training expands to include all usable earlier universe-stock labels. Reserve the last 60 trading dates of that training calendar for inner early stopping; the outer fold does not choose epochs. Train up to `evaluation.max_epochs` (default 5, independent of the fixed refit `epochs`) and monitor mean per-date smooth loss, retaining the earliest minimum and stopping after `patience` epochs without improvement. Reinitialize from the same seed and refit on the full fold training window for the chosen epoch count before scoring. `validation_metrics.json` and `validation_predictions.csv` audit the splits, per-fold best epochs, scores, and predictions.

### Select and freeze

Run `python search.py --config CONFIG --artifacts-dir RUN` in a fresh output directory. Search uses only the validation path and limits input history to the end of the last validation fold. Use a stable snapshot of both data directories for the run.

Use `--workers 3` to run up to three configuration/seed jobs at once on a machine with that many allocated CPU cores. The default (`--workers 1`) runs serially. Each process is single-threaded for PyTorch and holds its own history and training cache, so choose the worker count based on available RAM as well as CPU. Ablations wait for grid selection; results aggregate in configured order regardless of worker completion order.

1. **Grid:** Try every configured window length, layer count and hidden size with attention and embedding on. For each trial, average the three seed contest scores within each fold, then average the fold scores with equal weight. An exact grid tie keeps the first candidate in config product order (window, layers, hidden size).
2. **Ablations:** Turn off attention and embedding separately, both against the same grid winner with matching fold and seed IDs. Under [ADR 0001](../../docs/adr/0001-feature-changes-require-walk-forward-evidence.md), a variant qualifies only if its mean contest score increases and strictly more than half the folds improve; tied folds do not count. If both qualify, adopt the higher-scoring tested variant (attention-off first on an exact tie). The combined ablation is untested and cannot be selected. `gate_eligible` and `adopted` record these separate decisions.
3. **Freeze:** Take each validation fold's median best epoch across the three seeds, then take the floor of the median across those fold medians (minimum 1). `selected_config.json` preserves those fold medians and raw seed epochs as `epoch_selection`, separately from the fixed `epochs` used for refitting. Pass this file directly to `evaluate.py final-test --config` to avoid scoring the wrong settings. Search does not overwrite `config.json`; copy the frozen values there for weekly training.

A completed run has `search_metrics.json` (all fold/seed scores and best epochs, comparisons and decisions, original and selected configs, stock lists, evaluated history range and runtime versions) and the original validation metrics and out-of-sample predictions under `<trial>/<seed>/`. See [the #16 experiment record](EXPERIMENTS.md) for the real configuration, per-fold contest scores and adoption decisions. An on-by-default feature or a synthetic search winner does not constitute real-data evidence.

### Report the final test

After selection is frozen, run `python evaluate.py final-test --config FROZEN_CONFIG --artifacts-dir RUN` once for reporting. It refits the same five-seed ensemble with the frozen epoch count, but only on labels before `final_test_start`. It then scores the sign of mean `f` on available final-test labels without further fitting. It reads neither validation artifacts nor fold settings, and refuses to overwrite final-test artifacts in the same output directory. Keep final-test results out of selection.

All three scores (model, always-up, always-down) use the **same labeled target stock-dates** and the official signed intraday-return formula. A labeled target without usable inputs receives direction `0` and still counts, including on the first trading day after a long closure. Offline metrics record scored dates when every stock was stale under `all_market_stale_dates`; on those dates every labeled target scores as always-down. The live prediction command still fails on an all-stale date so a broken refresh is visible. A missing realized return cannot be scored by any of the three. Inference masks depend on input availability and staleness, not on which stocks will have a realized return. A bounded cache reuses labeled batches across epochs and overlapping fits without keeping a copy per fold.

## Prediction safeguards

- **Stale stock:** Compare each universe stock's latest valid market row strictly before the prediction date to that date. More than 10 calendar days old (or no prior valid row) is stale; exactly 10 days is eligible. Warn for each stale stock and mark it absent before inference, so it cannot influence other stocks. For a stale target stock, write direction `0` with a blank score and source dates.
- **All stocks stale:** If every universe stock is stale on any requested date, exit nonzero with that date in the error. Leave both prediction files untouched, even if other dates in the template could be scored.
- **In-sample date:** Require every prediction date to be strictly after the checkpoint's `last_training_label_date`. If a date fails that check, exit nonzero before writing either output.
