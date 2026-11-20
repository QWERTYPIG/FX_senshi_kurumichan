# GRU sequence experiments

Real-data selection and final test for [#16](https://github.com/QWERTYPIG/FX_senshi_kurumichan/issues/16). This record also documents the ablation decisions required by [ADR 0001](https://github.com/QWERTYPIG/FX_senshi_kurumichan/blob/14520d68d5bd30ec19170e7c5cb47e0dc77c8f70/docs/adr/0001-feature-changes-require-walk-forward-evidence.md).

## Walk-forward setup

- Historical `basic_data/` snapshot through 2026-08-31; no daily updates. Selection reads only market rows dated 2020-01-02 through 2026-05-29. Training uses the 100 universe stocks; scoring uses only the 50 target stocks with realized intraday returns.
- Linux CPU, Python 3.13.9, PyTorch 2.7.1+cpu. Search seeds: 42, 43, 44. Each expanding-window fold uses the last 60 pre-validation trading dates for early stopping (at most five epochs, patience two), then refits on its full training window. See [the evaluated workflow](DESIGN.md#walk-forward-evaluation).
- Each trial keeps the same inputs, loss, head width (16), embedding width (4 when enabled), learning rate (0.001), and Gaussian input noise (0.1). Grid trials vary lookback, layer count, and hidden width; each ablation changes just one switch from the grid winner.
- Each fold score below averages the three seeds. The mean gives all folds equal weight. Scores are raw contest-score decimals, not percentages. The always-down baseline uses the same stock-dates. On 2025-02-03 and 2026-02-23 all stocks had stale inputs after a market closure; those labeled dates stayed in the folds with direction 0.

| Fold | Validation bounds (inclusive) | Scored stock-dates |
|---|---|---:|
| 1 | 2024-07-01–2024-09-30 | 3,149 |
| 2 | 2024-10-01–2024-12-31 | 3,100 |
| 3 | 2025-01-01–2025-03-31 | 2,750 |
| 4 | 2025-04-01–2025-06-30 | 3,050 |
| 5 | 2025-07-01–2025-09-30 | 3,242 |
| 6 | 2025-10-01–2025-12-31 | 3,098 |
| 7 | 2026-01-01–2026-03-31 | 2,750 |
| 8 | 2026-04-01–2026-05-29 | 2,000 |

## Selection

All grid trials enable attention and embedding. Configuration is shown as `lookback / GRU layers / hidden width`. The two ablations use the `grid-6` configuration with the named component disabled.

| Trial | Configuration | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | Mean | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Always down | — | 0.00034504 | -0.00004015 | 0.00090476 | -0.00052348 | 0.00030630 | 0.00114371 | 0.00145651 | 0.00167817 | 0.00065886 | Baseline |
| grid-1 | 20 / 1 / 32 | 0.00118784 | 0.00009807 | 0.00063671 | -0.00018209 | -0.00041041 | 0.00106067 | 0.00133667 | 0.00092439 | 0.00058148 | Not selected |
| grid-2 | 20 / 1 / 64 | 0.00082537 | 0.00012318 | 0.00074766 | -0.00034995 | -0.00077799 | 0.00087158 | 0.00060699 | 0.00135160 | 0.00042480 | Not selected |
| grid-3 | 20 / 2 / 32 | 0.00034504 | -0.00004015 | 0.00090476 | -0.00052348 | 0.00030630 | 0.00114371 | 0.00145651 | 0.00167817 | 0.00065886 | Not selected |
| grid-4 | 20 / 2 / 64 | 0.00034504 | -0.00004015 | 0.00090476 | -0.00052348 | 0.00030630 | 0.00114371 | 0.00145651 | 0.00167817 | 0.00065886 | Not selected |
| grid-5 | 60 / 1 / 32 | 0.00054943 | 0.00000318 | 0.00086918 | 0.00001230 | -0.00051074 | 0.00114371 | 0.00148957 | 0.00131518 | 0.00060898 | Not selected |
| grid-6 | 60 / 1 / 64 | 0.00034504 | -0.00003664 | 0.00090476 | -0.00052348 | 0.00030630 | 0.00114371 | 0.00145651 | 0.00167817 | 0.00065930 | Selected |
| grid-7 | 60 / 2 / 32 | 0.00054759 | -0.00004252 | 0.00091378 | -0.00052348 | -0.00002766 | 0.00114371 | 0.00150782 | 0.00148219 | 0.00062518 | Not selected |
| grid-8 | 60 / 2 / 64 | 0.00034504 | -0.00004015 | 0.00090476 | -0.00052348 | 0.00030630 | 0.00114371 | 0.00145651 | 0.00167817 | 0.00065886 | Not selected |
| grid-9 | 120 / 1 / 32 | 0.00034504 | -0.00004015 | 0.00091329 | -0.00067820 | 0.00030630 | 0.00114371 | 0.00145651 | 0.00167817 | 0.00064059 | Not selected |
| grid-10 | 120 / 1 / 64 | 0.00034504 | -0.00004015 | 0.00090476 | -0.00052348 | 0.00030630 | 0.00114371 | 0.00145651 | 0.00167817 | 0.00065886 | Not selected |
| grid-11 | 120 / 2 / 32 | 0.00034504 | -0.00004015 | 0.00090476 | -0.00052348 | 0.00030630 | 0.00114371 | 0.00145651 | 0.00167817 | 0.00065886 | Not selected |
| grid-12 | 120 / 2 / 64 | 0.00034504 | -0.00004015 | 0.00090476 | -0.00052348 | 0.00030630 | 0.00114371 | 0.00145651 | 0.00167817 | 0.00065886 | Not selected |
| no-attention | 60 / 1 / 64 | 0.00079658 | -0.00004623 | 0.00096298 | -0.00056064 | -0.00034246 | 0.00111363 | 0.00144352 | 0.00127477 | 0.00058027 | Rejected |
| no-embedding | 60 / 1 / 64 | 0.00034504 | -0.00003828 | 0.00089685 | -0.00042753 | 0.00030630 | 0.00112110 | 0.00153871 | 0.00167817 | 0.00067755 | Rejected |

- **Remove attention:** mean change vs. `grid-6` was −0.00007903; improved in 2/8 folds. Fails both adoption criteria.
- **Remove embedding:** mean change was +0.00001825; improved in 2/8 folds. Fails the strict-majority criterion despite a higher mean.
- **Freeze `grid-6`:** its mean exceeded always-down by only 0.0000004378. Per-fold median best epochs were `5, 5, 3, 2, 1, 2, 2, 2`, yielding **2** refit epochs. [`config.json`](config.json) contains the full frozen setting and epoch provenance. Weekly refits reuse it for deployment seeds 42–46 without running search.

## Final test

The frozen `grid-6` ensemble was fit on usable labels through 2026-05-29 and scored from 2026-06-01 through 2026-08-31. The score includes all **3,200 stock-dates** (64 dates × 50 stocks).

| Predictor | Contest score |
|---|---:|
| `grid-6` (five seeds) | -0.0005479315896581678 |
| Always-down baseline | -0.0005479315896581678 |

Every ensemble output was negative, giving direction 0 on all 3,200 stock-dates. The model **did not improve on always-down** on this period; the validation edge did not carry over.

**Protocol deviation:** `final-01` accidentally scored the old defaults (20-day lookback, width 32, five epochs) before the selected configuration was scored. This was not the planned single final evaluation. The `search-04` winner and its two epochs were already frozen by validation. `final-selected-01` evaluated that original `selected_config.json` once, without choosing or tuning anything using either final-test output. `final-01` is retained for audit and is not reported as the selected model result.

### Artifact provenance

The original artifacts are in the local `gru16-results/` archive (not committed): `search-04/` contains the 42 seed runs, `search_metrics.json` and `selected_config.json`; `final-selected-01/` contains the final-test metrics and predictions. `final-01/` contains the mistaken run.

| Artifact | SHA-256 |
|---|---|
| `search-04/search_metrics.json` | `b7368440b667b91dae8d2d8592abaaaba6e06675286c5bb87458358b4f2efd76` |
| `search-04/selected_config.json` | `2b73d94814092924da2c883ab94c1347252e4559b5c7e781040d81c55b053552` |
| `final-selected-01/final_test_metrics.json` | `3b04972bf05606f798deea90c9eee9cfa47c99347aa391a678a2f97631fe06b9` |
| `final-01/final_test_metrics.json` | `e79ea6fb7025de42f8d596c7249727b3a7ddfbbe23ecef702722ad85919eba1a` |

## Daily-update alignment

**Pending.** No active daily-update directory was used in these experiments. The separate `new_data.zip` has the same global CSV schema and extends the history, but that does not verify what trading session a dated value represents. Before relying on these inputs for live prediction, verify the daily-update source and compare the dated US index and VIX values with the session closes available before the 09:00 Taiwan cutoff; also confirm the non-US date convention has not changed.
