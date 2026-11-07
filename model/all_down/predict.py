"""Example model: preserve the supplied template and predict zero for every cell."""
import csv
import os
from pathlib import Path

source = Path(os.environ.get('MANAGER_TEMPLATE', Path(__file__).with_name('all_down.csv')))
target = Path(os.environ.get('MANAGER_OUTPUT', 'prediction.csv'))
with source.open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.reader(f))
with target.open('w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(rows[0])
    writer.writerows([row[0], *([0] * (len(rows[0]) - 1))] for row in rows[1:])
print(f'Wrote {target}')
