import os
from pathlib import Path
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('MANAGER_ROOT', MODEL_DIR.parents[1]))
DATA_DIR = Path(os.environ.get('MANAGER_DATA_DIR', ROOT / 'data'))
TEMPLATE = Path(os.environ.get('MANAGER_TEMPLATE', ROOT / 'manager_data/template.csv'))


def read_template(path):
    frame = pd.read_csv(path, encoding='utf-8-sig', dtype={'id': str})
    if frame.empty or frame.columns[0] != 'id' or frame.id.isna().any() or frame.id.duplicated().any() or len(frame.columns) < 2:
        raise ValueError('Template requires unique stock IDs and prediction dates')
    for date in frame.columns[1:]:
        if pd.Timestamp(date).strftime('%Y-%m-%d') != date:
            raise ValueError('Template dates must use YYYY-MM-DD')
    return frame


def encode(panel, stock_ids):
    features = [c for c in panel if c not in ['date', 'stock_source_date', 'global_source_date', 'stock_id', 'target']]
    for sid in stock_ids:
        column = 'stock_' + sid
        panel[column] = (panel.stock_id == sid).astype(float)
        features.append(column)
    return features
