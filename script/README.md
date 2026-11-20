# submit script:
``` python script/submit.py sampleout.csv --phase 1 --submit```

# FinMind US daily prices

```sh
# Initialize 2022-to-present daily history under us_data/
python script/download_us.py
# Update the persisted symbol list with a 7-day overlap
python script/update_us.py
# Choose a different universe
python script/download_us.py --symbols '^IXIC' '^GSPC' NVDA AMD TSM
# Discover every stock ID from USStockInfo
python script/download_us.py --all
```

Optional token: `FINMIND_TOKEN` in the repo `.env`. Progress and symbols persist in `us_data/metadata.json` and `us_data/config.json`. Re-run after interruptions. `--full` refreshes all historical years, `--max-requests` limits each invocation, and `--output-dir` changes the destination. Manager uses the default repo `us_data/` directory and exposes it to models as `MANAGER_US_DATA_DIR`. See the root README for schedule, schema, quotas and data version behavior.
