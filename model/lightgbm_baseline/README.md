# LightGBM baseline

全部 100 支股票共用一個二元分類器，驗證、測試與輸出預測針對提交的 50 支。
標籤依 final.pdf 重新計算為
`adj_close / adj_open - 1 > 0`，不使用資料中的 `y`。

## 安裝與訓練

在 repo 根目錄執行：

```sh
.venv/bin/python -m pip install -r model/lightgbm_baseline/requirements.txt
.venv/bin/python model/lightgbm_baseline/train.py
```

每次訓練都重新掃描 `basic_data/market/` 與 `data/market/` 的 CSV，使用兩者聯集
中的全部股票。`model/all_down/all_down.csv`（可用 `--stock-template` 指定）只用來
決定驗證與輸出的目標股票清單，不限制訓練股票。

`basic_data/market/` 是不變的歷史資料，`data/market/` 是持續更新的資料；先合併
再排序、計算 rolling 特徵，相同股票與日期由 `data/` 整列覆蓋。更新列若價格無效
會被排除，不回填成舊版價格。程式不修改這兩份來源。`basic_data/data/` 是舊的
下載副本，不會遞迴讀入。

訓練與預測都支援 `--history-dir`（預設 `basic_data/`）、`--data-dir`（預設 `data/`，
也會遵守 `MANAGER_DATA_DIR`）。預設訓練截止日就是本次讀入的最新有效標籤日期，
不使用固定日期或上一次模型的截止日；`--train-end` 僅供手動限制歷史回測。
例如今天資料仍只更新到 11/06，截止日就是 11/06；資料更新到 11/09 後，下次訓練
便會包含 11/09。不同股票缺漏仍依各自有效資料處理。

模型與結果存於本目錄的 `artifacts/`，已排除於 Git：

- `model.txt`：部署用 LightGBM 模型。
- `metadata.json`：特徵版本、順序、100 支訓練股票、50 支目標股票、資料來源、列數、日期範圍、超參數、套件版本。
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
同一天的所有股票保持在同一集合。train 使用全部股票，validation/test 只評估目標
50 支，validation 的 binary logloss 用來 early stopping，
閾值固定 0.5；test 不參與訓練或選擇迭代次數。

```
score = mean((2 * prediction - 1) * intraday_return)
```

評估只對有有效標籤的股票日計算，基準也使用相同股票日；這和 manager
要求完整 50 支才公布某日分數的規則不同。`metrics.json` 會記錄實際列數。

評估完成後，用已選定的迭代次數在全部股票、全部允許日期上重新訓練部署模型，包含
原本 validation/test。歷史測試成績來自重訓前的模型；不能用部署模型回算
同一段資料宣稱樣本外績效。反覆根據 test 調整方案後，該段也不再是獨立測試集。

可指定截止日與切分日數，例如：

```sh
.venv/bin/python model/lightgbm_baseline/train.py --train-end 2026-10-30 --valid-days 5 --test-days 5
```

靜態資料可補回 2020 年起的歷史；預設最後 7 日的測試期間仍很短，
後續可擴大驗證區間並增加多段時間回測。

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

預測只載入模板所需的 50 支股票行情，但 categorical 使用保存的完整訓練股票類別，
維持 100 支訓練與 50 支推論的編碼一致。這次資料合併與股票清單的格式有更新，
舊模型需先重新訓練一次。

所有預測日期都必須晚於模型的 `trained_through`；歷史評估應查看
`test_predictions.csv`，或另訓練截止日更早的模型。最新歷史價格距預測日超過
10 個日曆日會拒絕預測，可用 `--max-history-age-days` 調整。

manager 在沒有開放的 Phase 2 場次時會回退到六月開始的 Phase 1 模板；用最新資料
訓練的模型會拒絕這種歷史預測。若要回測 Phase 1，請另指定 `--artifacts-dir`，
以 `--train-end 2026-05-31` 訓練，再用同一 artifacts 目錄搭配 Phase 1 模板預測。

## 測試

```sh
.venv/bin/python -m unittest discover -s model/lightgbm_baseline/tests -v
```

涵蓋當日／未來資料修改不影響過去特徵、股票間隔離、下一交易日特徵、無歷史資料、
靜態與更新資料合併、更新覆蓋、重新讀取最新日期、完整股票類別與模板選擇。

LightGBM API：[官方文件](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html)。
