# Manager training data directories

Updated November 11, 2026.

The manager supplies original data through August 2026 in
`MANAGER_BASIC_DATA_DIR` and recent data from September 1 onward in
`MANAGER_DATA_DIR`. Previously, training read only the latter. With the default
May 31 training cutoff, all September records were excluded, causing
`No valid price history: 2330` even when the original history was available
in the separate basic directory.

Training now passes the original directory and recent directory to the existing
history loader. Both stock and global CSVs are concatenated in memory, deduplicated
by date with recent records taking precedence, sorted, and filtered to the cutoff.
No source files are modified. Both directories must contain the expected stock
and global CSVs. Invalid price filtering still applies after loading.

The original training cutoff, validation split, feature calculations, and recency
weighting search are unchanged. In particular, September updates do not add
training labels when using the default May cutoff.

Manager invocation remains `python train.py`; directory defaults are supplied
by the environment. For explicit paths, from the repository root:

```sh
python model/logistic_regression/train.py --basic-data-dir /path/to/original/data --data-dir /path/to/recent/data
```

Without `MANAGER_BASIC_DATA_DIR` or `--basic-data-dir`, training keeps the old
single-directory behavior, treating `--data-dir` as complete history. When both
directories resolve to the same path, it reads that directory only once.
Explicit CLI paths override their respective environment defaults.

Prediction code is unchanged by this training fix. Phase 1 prediction still
requires a combined history directory; later predictions can use recent history
provided it meets the existing freshness and feature requirements.

Tests cover environment defaults, CLI overrides, single-directory compatibility,
same-directory handling, stock/global overlap precedence, cutoff filtering,
and preservation of source CSVs:

```sh
python -m unittest discover -s model/logistic_regression/tests -v
```

Verification: all nine tests passed. Also ran actual training with the two
manager environment variables pointing to the local original/recent datasets,
using no decay and penalty 0.1. It fit 77,624 training rows through May 29 and
reproduced the original validation score 0.0016091863. Smoke-test artifacts were
written under `/tmp/logistic_manager_split_smoke`; deployment artifacts were
not replaced. `git diff --check` passed.
