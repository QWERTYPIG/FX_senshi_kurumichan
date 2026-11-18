# Return-weighted AdaBoost

本目錄提供可由專案 manager 直接訓練及預測的 AdaBoost 分類模型。模型預測每支股票
從調整後開盤到調整後收盤為漲（1）或跌／持平（0），並將結果寫成官方要求的
`id,YYYY-MM-DD,...` CSV。預測檔會完整保留模板中的 50 支股票順序與日期欄位。

## 安裝與執行

在 repo 根目錄建立環境並安裝本模型鎖定的套件版本：

```sh
python -m venv .venv
.venv/bin/python -m pip install -r model/ADAboost/requirements.txt
```

訓練：

```sh
.venv/bin/python model/ADAboost/train.py
```

使用 manager 提供的模板預測時，`MANAGER_DATA_DIR`、`MANAGER_BASIC_DATA_DIR`、
`MANAGER_TEMPLATE` 與 `MANAGER_OUTPUT` 會自動生效。也可以手動指定日期或模板：

```sh
.venv/bin/python model/ADAboost/predict.py \
  --date 2026-11-19 \
  --output model/ADAboost/prediction.csv

.venv/bin/python model/ADAboost/predict.py \
  --template /path/to/sample_submission.csv \
  --output model/ADAboost/prediction.csv
```

`usage.txt` 的前兩行符合 manager 介面：第一行執行預測，第二行重新訓練。
第一次預測前必須先成功訓練；manager 的每日同步只會執行預測，不會自動重訓。

## 資料讀取

- `MANAGER_BASIC_DATA_DIR`（預設 `basic_data/`）提供長期歷史。
- `MANAGER_DATA_DIR`（預設 `data/`）提供每日更新資料。
- 每次執行都重新掃描兩者的 `market/*.csv`。股票與日期重疊時，以每日更新資料的
  整列為準；來源檔案不會被修改。
- 訓練使用兩個目錄中找到的所有股票；驗證、test 與輸出只使用
  `model/all_down/all_down.csv` 指定的 50 支及其順序。
- `--train-end YYYY-MM-DD` 可限制最後一個訓練標籤日，適合歷史回測。未指定時會
  訓練至資料中最新的有效價格日。

訓練前至少應能看到以下檔案結構；目錄只有名稱、沒有 `*.csv` 仍無法訓練：

```text
basic_data/market/<stock_id>.csv   # 2020 年起的課程歷史資料
data/market/<stock_id>.csv         # manager 下載的近期更新資料
```

可先檢查：

```sh
find basic_data/market data/market -maxdepth 1 -name '*.csv' | head
```

若透過 manager 同步 `data/`，repo 根目錄的 `.env` 必須有課程提供的
`data_access_key`；缺少金鑰時 manager 無法下載資料。`basic_data/` 的長期訓練集仍需
由課程下載內容放入上述結構。資料放在其他位置時，直接指定：

```sh
.venv/bin/python model/ADAboost/train.py \
  --history-dir /path/to/basic_data \
  --data-dir /path/to/data
```

資料中的無效價格（非正數）和負成交量會在 rolling 計算前移除。程式不使用來源中
可能存在的 `y` 欄，而是依 final.pdf 定義重新計算：

```text
intraday_return = adj_close / adj_open - 1
label = 1 if intraday_return > 0 else 0
```

## 防止 look-ahead leakage

每個來源日期先計算「該日收盤後」才完整的特徵，再以 `merge_asof` 且
`allow_exact_matches=False` 合併至預測日。因此日期 `t` 的輸入只可能來自嚴格早於
`t` 的最近有效行情；不會使用 `t` 日的開、高、低、收、成交量或技術指標。
特徵包含：

- 前期盤中報酬、1/2/5/10/20 日收盤報酬、跳空與日內振幅。
- 5/20 日報酬均值與波動度。
- 成交量變化、5/20 日量比與成交金額。
- RSI、SMA、布林通道位置／寬度、MACD、ATR、外資持股比率；來源缺欄時保留 NaN。
- 星期、月份及上一筆有效行情距預測日的天數。

缺失值的中位數與 missing indicators 僅由每次訓練區間估計。推論若沒有先前行情，
或最近行情距預測日超過 10 個日曆日，會直接拒絕輸出，避免默默使用過期資料。

## AdaBoost 與官方目標

模型為 `AdaBoostClassifier`，weak learner 是限制深度的 decision tree。每筆訓練資料
使用 `abs(intraday_return)` 作為主要 sample weight，另加 1% 中位正報酬權重的下限；
因此大幅波動日對 boosting 的影響較大，但平盤資料仍會保留。這比單純最佳化 accuracy
更接近官方分數：

```text
score = mean((2 * prediction - 1) * intraday_return)
```

AdaBoost 的 probability 不一定校準在 0.5 附近，因此決策閾值也視為超參數：每個候選
先產生全部 walk-forward out-of-fold probability，再只用這些 validation rows 選擇使
官方 score 最大的閾值。相同 probability 的資料會整組翻轉，避免 tie 處理與實際
`>= threshold` 推論不一致。final test 完全不參與閾值或模型選擇。

預設比較四組 `max_depth / n_estimators / learning_rate`。可重複指定自訂候選，例如：

```sh
.venv/bin/python model/ADAboost/train.py \
  --candidate 1,100,0.05 \
  --candidate 2,150,0.05
```

## 時間驗證

預設按「唯一交易日」切分，而不是隨機打散股票列：

1. 最後 20 個交易日保留為一次性的 final test。
2. test 之前建立 3 個 expanding-window folds，每折驗證連續 20 個交易日。
3. 每個候選只用各 fold 之前的資料訓練，以合併的 OOF 資料選決策閾值，再依 3 折
   官方 score 平均選擇候選。
4. 選定參數後用 test 前全部資料訓練並只評估 test 一次。
5. 最後以全部允許歷史重訓部署模型；test 成績仍來自重訓前模型。

歷史不足時可調整 `--cv-folds`、`--valid-days`、`--test-days`、`--min-train-days`。
驗證和 test 只衡量目標 50 支，模型訓練則使用資料中的所有股票。

## 產物

訓練後 `artifacts/` 包含：

- `model.joblib`：部署模型（只應載入自己訓練、可信任的 joblib 檔）。
- `metadata.json`：特徵版本、資料日期、股票清單、選定參數及套件版本。
- `metrics.json`：每個候選／fold 的結果與 untouched test 結果。
- `candidate_metrics.csv`：候選參數的彙總，便於報告製表。
- `validation_predictions.csv`、`test_predictions.csv`：樣本外逐筆預測。

預測日期必須晚於 `metadata.json` 的 `trained_through`。歷史區間請使用已保存的
validation/test 預測評估，或另設 `--train-end` 及 `--artifacts-dir` 重訓，不能用最新
部署模型回算後宣稱是樣本外結果。

## 測試

測試完全使用暫存合成資料，不會修改正式資料：

```sh
.venv/bin/python -m unittest discover -s model/ADAboost/tests -v
```

測試涵蓋因果時間邊界、歷史／更新合併、模板驗證、walk-forward 切分，以及一輪小型
端到端訓練和 0/1 CSV 預測。

參考：scikit-learn `AdaBoostClassifier` 與 Freund and Schapire (1997),
“A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting.”
