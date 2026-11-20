# Development setup

## Python

Use the repo-local `.venv` from the repository root. If it is missing, create it with [uv](https://docs.astral.sh/uv/) and Python 3.13:

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Run commands with `.venv/bin/python` or activate with `source .venv/bin/activate`. Check the base environment with `uv pip check --python .venv/bin/python` and `.venv/bin/python -m manager --help`; neither starts the manager. For model-specific dependencies, consult `model/<name>/requirements.txt` if present.

## Data

- **Historical training data:** `basic_data/` contains `market/*.csv`, `global/global_data.csv`, and `info.json`. It is local and ignored by Git.
- **Daily updates:** `data/` is separate from history. When live data is needed, the manager's refresh action downloads updates using `data_access_key` from the local `.env`. Refresh can also run model predictions. Historical model development doesn't require this credential or a live refresh.
