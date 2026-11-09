"""Generate binary prediction.csv for the manager's supplied template."""
import argparse
import json
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from common import DATA_DIR, MODEL_DIR, TEMPLATE, read_template, encode
from features import build_data, predict


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DATA_DIR)
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--template',type=Path,default=TEMPLATE)
    selection.add_argument('--date')
    parser.add_argument('--output',type=Path,default=Path(os.environ.get('MANAGER_OUTPUT','prediction.csv')))
    parser.add_argument('--artifacts-dir',type=Path,default=MODEL_DIR/'artifacts')
    parser.add_argument('--max-history-age-days',type=int,default=10)
    args=parser.parse_args()
    metadata=json.loads((args.artifacts_dir/'metadata.json').read_text())
    if metadata['feature_version'] != 1:
        raise ValueError('Feature version changed; retrain')
    template=(pd.DataFrame({'id':metadata['stock_ids'],pd.Timestamp(args.date).strftime('%Y-%m-%d'):0})
              if args.date else read_template(args.template))
    ids=template.id.tolist()
    if ids != metadata['stock_ids']:
        raise ValueError('Template stock order differs from training')
    dates=pd.to_datetime(template.columns[1:])
    if dates.min() <= pd.Timestamp(metadata['trained_through']):
        raise ValueError('Prediction dates must be after the training cutoff')
    panel=build_data(args.data_dir,template)
    features=encode(panel,ids)
    test=panel[panel.date.isin(dates)].copy()
    for col in ['stock_source_date','global_source_date']:
        age=(test.date-test[col]).dt.days
        if test[col].isna().any() or (age<=0).any() or (age>args.max_history_age_days).any():
            raise ValueError('Missing or stale history; synchronize data before predicting')
    with (args.artifacts_dir/'model.pkl').open('rb') as f:
        model=pickle.load(f)
    if features != model['features']:
        raise ValueError('Feature schema differs; retrain')
    test['predicted_return'],test['weighted_up_probability']=predict(model,test)
    if len(test)!=len(ids)*len(dates) or not np.isfinite(test.predicted_return).all():
        raise ValueError('Missing or nonfinite predictions')
    test['prediction']=(test.predicted_return>0).astype(int)
    table=test.pivot(index='stock_id',columns='date',values='prediction')
    for date in template.columns[1:]:
        template[date]=table.loc[ids,pd.Timestamp(date)].to_numpy()
    test[['stock_id','date','predicted_return','weighted_up_probability','prediction','stock_source_date','global_source_date']].to_csv(MODEL_DIR/'predictions.csv',index=False)
    temp=args.output.with_suffix('.csv.tmp')
    template.to_csv(temp,index=False)
    temp.replace(args.output)
    print(f'Wrote {args.output}: {len(ids)} stocks, {len(dates)} dates')


if __name__=='__main__':
    main()
