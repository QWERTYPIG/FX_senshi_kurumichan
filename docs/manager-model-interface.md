# Manager model interface

## Discovery and commands

Place `usage.txt` in `model/<model_name>/`. The manager discovers models by this path; no registration is needed. Its first two lines must be nonempty shell commands:

```text
python predict.py
python train.py
```

The first command generates predictions. The second trains the model; for a model requiring no training, use a successful no-op. Both run with the model directory as the working directory. A train-and-predict run executes training first, then prediction. Commands must exit successfully; their stdout and stderr are captured as logs, not predictions. `python` resolves to the manager's Python environment by default; an absolute interpreter path can select another environment. The manager treats these shell commands as trusted local code.

## Inputs and output

Both commands receive these absolute paths as environment variables:

- `MANAGER_ROOT`: repository root.
- `MANAGER_BASIC_DATA_DIR`: `basic_data/`, historical data.
- `MANAGER_DATA_DIR`: `data/`, daily updates. Merge overlapping historical and update rows in the model.
- `MANAGER_TEMPLATE`: CSV containing the expected stock order and prediction dates. When a Phase 2 session is open, it contains that session's date; otherwise it uses the Phase 1 template.
- `MANAGER_OUTPUT`: `model/<model_name>/prediction.csv`.

Each prediction command must create or update `MANAGER_OUTPUT` on every successful run; an unchanged old file is rejected. If an existing model writes elsewhere, wrap its command and copy the result to `"$MANAGER_OUTPUT"`. If it writes CSV to stdout, redirect stdout to `prediction.csv` and send logs to stderr.

The output CSV must have `id` as its first header, followed by unique dates in `YYYY-MM-DD` format. Include exactly the stock IDs and order in `model/all_down/all_down.csv`, with one row per stock and a `0` or `1` in every date cell. Use `MANAGER_TEMPLATE` to preserve the expected IDs, order, and dates; for Phase 2, output the session's single trading date. The template does not prevent data leakage: use only information available before each prediction date.

The manager validates the CSV and archives a copy after a successful run. `model/all_down/predict.py` is a minimal example of adapting the template to the output contract.
