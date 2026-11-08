# LightGBM baseline

50 支提交股票共用一個二元分類器。標籤依 final.pdf 重新計算為
`adj_close / adj_open - 1 > 0`，不使用資料中的 `y`。

## 安裝與訓練

在 repo 根目錄執行：

```sh
.venv/bin/python -m pip install -r model/lightgbm_baseline/requirements.txt
.venv/bin/python model/lightgbm_baseline/train.py
```

預設讀取 `data/market/` 與 `model/all_down/all_down.csv` 的股票清單。
訓練與預測都支援 `--data-dir`，預設也會遵守 `MANAGER_DATA_DIR`。
模型與結果存於本目錄的 `artifacts/`，已排除於 Git：

- `model.txt`：部署用 LightGBM 模型。
- `metadata.json`：特徵版本、順序、股票清單、資料截止日、超參數、套件版本。
- `metrics.json`：驗證與測試的 accuracy、官方 score、全漲／全跌分數及每日分數。
- `test_predictions.csv`：時間切分測試集的機率與實際報酬。
- `feature_importance.csv`：部署模型的 gain importance。

## 特徵與時間邊界

`features.py` 負責資料轉換，`FEATURES` 定義固定特徵順序：

- 股票 ID（categorical）與預測日星期幾。
- 前一筆交易日的盤中報酬、收盤報酬、振幅、跳空幅度。
- 前一筆成交量相對當時 5 日均量的比值。
- 截至前一筆交易日的 5 日平均報酬、標準差與 10 日平均報酬。
- 前一筆 RSI、收盤價對 SMA 比值、ATR／收盤價、MACD／收盤價。

先逐股票計算截至各歷史日期收盤的特徵，再以嚴格早於預測日的
`merge_asof` 合併。這相當於依交易日 `shift(1)`，也能處理資料中尚未出現的
下一個交易日，不會拿預測當天的開高低收當輸入。缺失指標及不足 rolling
視窗的值保留為 NaN，交由 LightGBM 處理；沒有任何過去資料的訓練列會排除。
全球資料尚未納入，待確認日期與發布時間後再擴充。

## 驗證設計

按唯一交易日期排序：最後 7 日為 test、之前 7 日為 validation，其餘為 train，
同一天的所有股票保持在同一集合。validation 的 binary logloss 用來 early stopping，
閾值固定 0.5；test 不參與訓練或選擇迭代次數。

```
score = mean((2 * prediction - 1) * intraday_return)
```

評估只對有有效標籤的股票日計算，基準也使用相同股票日；這和 manager
要求完整 50 支才公布某日分數的規則不同。`metrics.json` 會記錄實際列數。

評估完成後，用已選定的迭代次數在全部允許日期上重新訓練部署模型，包含
原本 validation/test。歷史測試成績來自重訓前的模型；不能用部署模型回算
同一段資料宣稱樣本外績效。反覆根據 test 調整方案後，該段也不再是獨立測試集。

可指定截止日與切分日數，例如：

```sh
.venv/bin/python model/lightgbm_baseline/train.py --train-end 2026-10-30 --valid-days 5 --test-days 5
```

目前資料約 45 個交易日，時間上的獨立樣本很少；第一版結果主要用於確認流程，
後續應累積資料並增加多段時間回測。

## 預測與 manager

直接執行時，預設讀取 `MANAGER_TEMPLATE`，未設定時讀取 manager 最近產生的
`manager_data/template.csv`，不再使用舊的 Phase 1 模板。這份檔案可能是之前場次
留下的模板，請確認日期；若不存在，程式會提示明確指定日期或模板。

也可以直接指定預測日期（須自行確認是欲預測的交易日）：

```sh
.venv/bin/python model/lightgbm_baseline/predict.py --date 2026-11-09 --output model/lightgbm_baseline/prediction.csv
```

或使用官方當日模板：

```sh
.venv/bin/python model/lightgbm_baseline/predict.py --template /path/to/daily_sample_submission.csv --output model/lightgbm_baseline/prediction.csv
```

`predict.py` 保留模板的股票順序與日期，只輸出 0/1。預設使用
`MANAGER_TEMPLATE`、`MANAGER_OUTPUT`，因此 manager 可透過本目錄的
`usage.txt` 執行「重新訓練＋預測」。第一次使用需先訓練；資料同步只會預測，
不會自動重訓。

所有預測日期都必須晚於模型的 `trained_through`；歷史評估應查看
`test_predictions.csv`，或另訓練截止日更早的模型。最新歷史價格距預測日超過
10 個日曆日會拒絕預測，可用 `--max-history-age-days` 調整。

**repo 預設模板是六月開始的 Phase 1 模板，目前資料從九月開始，不能直接產生
完整、無洩漏的 Phase 1 預測。** 請使用 Phase 2 當日模板，或備妥更早歷史資料。
manager 在沒有開放的 Phase 2 場次時會回退到 Phase 1 模板，此時拒絕預測是預期行為。

## 測試

```sh
.venv/bin/python -m unittest discover -s model/lightgbm_baseline/tests -v
```

涵蓋當日／未來資料修改不影響過去特徵、股票間隔離、下一交易日特徵與無歷史資料。

LightGBM API：[官方文件](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html)。
