# LightGBM baseline

全部 100 支股票共用一個 LightGBM 回歸模型，驗證、測試與輸出預測針對提交的 50 支。
目標依 final.pdf 重新計算為盤中報酬 `adj_close / adj_open - 1`，不使用資料中的 `y`。
預測報酬大於 0 輸出 1，否則輸出 0。

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

訓練與預測都支援 `--history-dir`（預設採用 `MANAGER_BASIC_DATA_DIR`，未設定時為
`basic_data/`）與 `--data-dir`（預設採用 `MANAGER_DATA_DIR`，未設定時為 `data/`）。
`basic_data/` 提供 2020 年起的歷史資料，`data/` 提供每日更新的 2026-09-01 起資料。
命令列參數可覆寫這些預設路徑。預設訓練截止日就是本次讀入的最新有效標籤日期，
不使用固定日期或上一次模型的截止日；`--train-end` 僅供手動限制歷史回測。
例如今天資料仍只更新到 11/06，截止日就是 11/06；資料更新到 11/09 後，下次訓練
便會包含 11/09。不同股票缺漏仍依各自有效資料處理。

模型與結果存於本目錄的 `artifacts/`，已排除於 Git：

- `model.txt`：部署用 LightGBM 模型。
- `metadata.json`：特徵版本、順序、100 支訓練股票、50 支目標股票、資料來源、列數、日期範圍、超參數、套件版本。
- `metrics.json`：逐折驗證、彙總驗證與最終測試的 accuracy、官方 score、MSE、MAE、全漲／全跌分數。
- `fold_metrics.csv`：每折日期、訓練筆數、最佳輪數與驗證指標，方便比較不同時期。
- `validation_predictions.csv`：每折的樣本外預測、實際報酬與折數。
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

使用 expanding-window walk-forward cross-validation：按唯一交易日期排序，
最後 20 個交易日為保留 test，在 test 前連續做 5 折、每折 20 日的 validation。
每折 train 都從最早日期開始、截止在該折 validation 之前；後一折可以使用前一折
已經發生的行情。validation 日期彼此不重疊，同日所有股票都屬於同一時間區間。

```text
Fold 1: [歷史訓練................] [驗證 20 日]
Fold 2: [歷史訓練...........................] [驗證 20 日]
Fold 3: [歷史訓練......................................] [驗證 20 日]
Fold 4: [歷史訓練.................................................] [驗證 20 日]
Fold 5: [歷史訓練............................................................] [驗證 20 日]
Test:   [全部 test 之前的資料................................................................] [test 20 日]
```

每折內還有一段獨立的 early-stopping 區間：從該折 train 的末尾取 20 日，先用
更早的資料訓練並以這 20 日的 MSE (`l2`) 選擇輪數；之後固定輪數，在整個該折
train 重新訓練，再預測外層 validation。**外層 validation 不參與 early stopping**，
避免同一批資料既挑最佳輪數、又被拿來報驗證成績。inner train 至少需要 252 日。
所有訓練使用全部股票，early stopping、validation 與 test 僅評估目標 50 支。

5 折完成後，用各折內層最佳輪數的中位數（小數向下取整、至少 1）作為固定輪數，
在所有 test 之前的資料重新訓練，僅評估一次保留 test。test 不用於 early stopping、
調參或選擇輪數。閾值固定為報酬 0，未針對 validation/test 搜尋閾值。

`metrics.json` 中：

- `validation`：合併所有外層樣本外預測後計算的指標，按股票日筆數加權。
- `cross_validation.folds`：每折的訓練、early stopping、驗證日期、最佳輪數及指標。
- `cross_validation.fold_summary`：各折 accuracy/score/MSE/MAE 的平均、標準差、最小與最大值。
  每折權重相同，標準差使用 `ddof=0`；缺漏股票日較多時，和合併指標的權重不同。
- `cross_validation.positive_score_folds`：官方 score 大於零的折數。
- `best_iteration`：固定中位數規則選出的輪數。
- `test`：最後保留區間的評估。

比較方案時，不只看平均 score，也看每折是否優於全漲／全跌，以及波動和最差折。
訓練期間彼此重疊、市場日期也有相關性，因此折間標準差不是信賴區間，
不能把股票日筆數當作同等數量的獨立觀測。增加折數能涵蓋更多時期，但不保證未來績效。

```
score = mean((2 * prediction - 1) * intraday_return)
```

評估只對有有效標籤的股票日計算，基準也使用相同股票日；這和 manager
要求完整 50 支才公布某日分數的規則不同。`metrics.json` 會記錄實際列數。

評估完成後，用已選定的輪數在全部股票、全部允許日期上重新訓練部署模型，包含
原本 validation/test。歷史測試成績來自重訓前的模型；不能用部署模型回算
同一段資料宣稱樣本外績效。反覆根據 test 調整方案後，該段也不再是獨立測試集。

直接執行 `train.py` 即使用上述預設，也可指定截止日、折數與切分日數，例如：

```sh
.venv/bin/python model/lightgbm_baseline/train.py --cv-folds 5 --valid-days 20 --early-stop-days 20 --test-days 20 --min-train-days 252
```

靜態資料可補回 2020 年起的歷史。預設驗證涵蓋 test 前最近 100 個交易日，並非
平均分散在 2020 年至今；可增加 `--cv-folds` 或 `--valid-days` 檢查更長時期。
`--train-end YYYY-MM-DD` 仍可限制全部資料的截止日。

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
靜態與更新資料合併、更新覆蓋、重新讀取最新日期、完整股票類別、模板選擇、
walk-forward 時間順序與不重複驗證，以及修改 test 價格不影響 CV 與選出的輪數。

LightGBM API：[官方文件](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html)。
