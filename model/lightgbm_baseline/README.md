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

## 參數搜尋

保留目前 14 個特徵，透過相同 walk-forward 切分選擇參數，然後訓練選出的模型：

```sh
.venv/bin/python model/lightgbm_baseline/tune.py
```

預設 `parameter_grid.json` 有 12 組候選，包含原本的參數：

| 參數 | 候選值 |
| --- | --- |
| `num_leaves` / `max_depth` | 7 / 4、15 / 5、31 / 6 |
| `min_child_samples` | 50、200 |
| `reg_lambda` | 1、10 |

`learning_rate=0.03` 固定，最多 500 輪；各折仍在內層以 MSE 做 early stopping，
外層用官方 score 評估。最終輪數取選中候選各折最佳輪數的中位數。

每個候選只產生 validation 指標，不評估 test、也不訓練部署模型。排名先比較
**各折平均 score（高者優先）**；相同時比較折間 score 標準差（低者優先），再比較
合併 validation MSE（低者優先），最後依原候選順序。這個規則與 pooled score 在
缺少股票日、各折筆數不相同時略有差異。搜尋期間會檢查資料與程式雜湊；若變動，
停止且不替換部署模型，避免不同候選使用不同資料。

選定後才用最佳參數重新訓練並評估一次最後的 test，再以全部允許日期訓練部署模型。
成功後自動更新 `artifacts/model.txt` 等檔案；舊檔案備份在該次搜尋的 `previous_model/`。
此操作只更新本機模型，不會透過 manager 提交比賽預測。

產出位於 `artifacts/search/<run_id>/`：

- `configuration.json`：搜尋網格、時間切分、特徵、資料與程式雜湊、排名規則。
- `search_results.csv` / `.json`：所有候選依 validation 排名的結果。
- `trial_XX/` 與 `.log`：每個候選的逐折結果和執行日誌；不含 test 指標。
- `selected_model/`：選中的模型、CV/test 評估與參數。
- `summary.json`：最佳候選與最後 test 結果。

`artifacts/best_params.json` 保存所選超參數及搜尋來源。之後執行 `train.py` 或 manager
的「重新訓練＋預測」會自動沿用這份參數，在最新資料上重新做 CV 選輪數和訓練；
不會每次自動搜尋。想重新選參數時再執行 `tune.py`。

可自行提供候選清單，也支援所有原有時間切分參數：

```sh
.venv/bin/python model/lightgbm_baseline/tune.py --grid /path/to/grid.json --cv-folds 5 --valid-days 20
.venv/bin/python model/lightgbm_baseline/train.py --params /path/to/best_params.json
.venv/bin/python model/lightgbm_baseline/train.py --baseline
```

候選檔是 JSON list，例如 `[{}, {"learning_rate": 0.05, "reg_lambda": 10}]`。
空物件代表原本參數。也可調整 `n_estimators`、`num_leaves`、`max_depth`、
`min_child_samples`、`reg_alpha`、`colsample_bytree`、`subsample` 與 `subsample_freq`。
若清單不含原本參數，搜尋會補入；重複候選會合併。`--baseline` 明確忽略保存的最佳參數。
若使用 `--params` 在另一個 `--artifacts-dir` 訓練，之後也應繼續明確指定該參數檔。

**選出來的最佳 validation 成績帶有選擇偏差**，不能當作完全獨立的樣本外成績。
最後 test 不參與搜尋，但這些歷史日期已在之前的實驗中查看過，仍需持續追蹤
新日期的實際表現。想比較大量網格的泛化能力，應增加另一層時間驗證或預留新的測試期間。

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
`global/global_data.csv` 尚未納入。另可透過下列選項使用 FinMind 美股日線。

## 美股特徵實驗

先使用 repo 的下載程式（檔名是 `download_us.py`）補齊五個標的：

```sh
.venv/bin/python script/download_us.py --symbols SPY QQQ SMH TSM NVDA --start-date 2022-01-01
```

使用 SPY 大盤、QQQ 科技、SMH 半導體、TSM ADR 與 NVDA 的 22 個新特徵：

- 各標的 1 日、5 日還原收盤報酬，以及台股上次有效收盤以來的累積美股報酬。
- SMH 的 20 日波動與相對 20 日均量比。
- QQQ／SMH 相對 SPY、TSM／NVDA 相對 SMH 的 1 日報酬差。
- 最舊美股來源距台股預測日的日曆天數。

`us_features.py` 將 CSV 日期視為紐約交易日，先計算 16:00 ET 收盤時間並處理 DST，
再以對應台北日期 **08:10** 為資料可用時間，以台股當日 **08:25** 為預測 cutoff
做 backward as-of join；不會使用同一日尚未發生的紐約行情。台股連假期間會累計
美股多日變動，而不是固定 shift 一列。報酬使用 `Adj_Close`，避免把股票分割當成跌幅。

**08:10 是回測假設，不是歷史發布時間證明。** [FinMind 文件](https://finmind.github.io/tutor/UnitedStatesMarket/Technical/)
標示日線每天 08:00 更新、實際以 API 為主，CSV 不含逐日發布時間與歷史修訂版本。
因此結果仍受資料延遲、歷史修訂及還原價事後修訂影響。即時預測只使用本機已下載
的列；來源缺失、超過 7 個日曆日或 rolling 暖機不足時會拒絕產生 US 預測。

公平比較與保存實驗模型：

```sh
.venv/bin/python experiments/lightgbm_us/run.py
```

比較三版：原本 2020 年起的 baseline、僅使用美股完整覆蓋日期的 baseline，以及相同
覆蓋日期加入美股特徵的版本。後兩版股票日、驗證日期、參數完全相同，避免把歷史
長度變化誤認為特徵效果。固定沿用 `artifacts/best_params.json` 的參數（若不存在則
使用原參數），不再針對美股特徵搜尋參數。先比較 5 折、再檢查較長 10 折驗證，
最後才評估保留 test；10 折包含原 5 折，不是獨立重複實驗。

結果、參數、資料雜湊、逐折與 test 預測存在 `experiments/lightgbm_us/results/<run_id>/`。
此指令不替換 manager 的預設模型、不提交比賽預測。

也可以單獨訓練、使用保存的美股模型：

```sh
.venv/bin/python model/lightgbm_baseline/train.py --feature-set us --params model/lightgbm_baseline/artifacts/best_params.json --artifacts-dir model/lightgbm_baseline/artifacts/us_experiment
.venv/bin/python model/lightgbm_baseline/predict.py --artifacts-dir model/lightgbm_baseline/artifacts/us_experiment --template /path/to/daily_sample_submission.csv --output model/lightgbm_baseline/prediction.csv
```

`--us-data-dir` 預設採用 `MANAGER_US_DATA_DIR`，未設定時使用 repo 的 `us_data/`。
預測會依模型 metadata 選用 baseline 或 US 特徵。普通 `train.py` 預設仍使用原本
14 個特徵。`usage.txt` 已切換為美股試跑模式：manager 訓練時使用 `--feature-set us`，
沿用 `artifacts/best_params.json`，模型寫入 `artifacts/us/`，預測也讀取同一目錄。
原本 `artifacts/model.txt` 與相關單模型結果保留。需要回到原版時，將 usage.txt
前兩行改回 `python predict.py`、`python train.py`。

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
