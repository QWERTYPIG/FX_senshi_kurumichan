# FX 戰士 Model Manager

本機 Python webserver，整合 `script/auth.py`、`downloads.py`、`state.py`、`submit.py`。只需要 Python 3.10+ 與既有 requests / python-dotenv，沒有前端 build 步驟。

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m manager
```

開啟 <http://127.0.0.1:8000>。`.env` 沿用 `email`、`passwd`、`data_access_key`。啟動時會立即同步一次；必須持續執行服務才能排程。可使用 `--port 8080`、`--command-timeout 14400` 調整連接埠與每條模型指令的逾時秒數。

## 模型介面

將模型統一放在 repo 的 `model/` 目錄，例如 `model/my_model/usage.txt`，**前兩行不可空白**：

```text
python predict.py
python train.py
```

第一行產生預測，第二行重新訓練；「重新訓練＋預測」依序執行第二行、第一行。兩條指令都在該模型目錄執行，透過 shell 支援參數、重導向及虛擬環境；usage.txt 視同受信任的本機程式碼。預設 `python` 使用啟動 manager 的 Python 環境，模型也可寫自己的環境絕對路徑。

Manager 提供以下環境變數（都是絕對路徑）：

| 變數 | 用途 |
| --- | --- |
| `MANAGER_ROOT` | repo 路徑 |
| `MANAGER_DATA_DIR` | 每日更新的 `data/`，包含 2026-09-01 起的資料，含 `market/*.csv`、`global/global_data.csv` |
| `MANAGER_BASIC_DATA_DIR` | `basic_data/`，包含 2020 年起至目前的歷史資料 |
| `MANAGER_US_DATA_DIR` | `us_data/`，FinMind 美股日線，價格存於 `prices/*.csv` |
| `MANAGER_TEMPLATE` | 本次參考模板，包含正確股票順序與日期 |

訓練與預測指令都會收到上述資料目錄。模型可讀取 `os.environ["MANAGER_BASIC_DATA_DIR"]` 取得歷史資料，並以 `os.environ["MANAGER_DATA_DIR"]` 取得每日更新資料。兩者有重疊日期時，由模型合併並處理重複列。

預測結果寫入 current working directory 的 `prediction.csv`，也就是 `model/<model_name>/prediction.csv`，不是 stdout。Manager 驗證後會複製一份到 `manager_data/runs/<run_id>/prediction.csv` 封存，模型目錄保留最新產出的檔案；stdout 與 stderr 只收進執行 Log。若程式輸出 CSV 到 stdout，可寫 `python predict.py > prediction.csv`，並把其他日誌寫到 stderr。

既有程式若固定輸出其他檔名，可將第一行設為 `python predict.py && cp result.csv "$MANAGER_OUTPUT"`。每次都必須成功建立或更新檔案；指令失敗或檔案未更新時，不會拿舊檔冒充新預測。CSV 使用 `id,YYYY-MM-DD,...` 格式，股票 ID 與順序必須等同 `model/all_down/all_down.csv` 的 50 支，每格僅接受 0 或 1。

目前 Phase 2 場次開放時，模板只含該場次日期；其他時間使用原本的 Phase 1 模板。模板只協助保留格式，模型必須自行避免使用未來資料。`model/all_down/` 已附上可直接執行的範例。若官方更改股票清單，請以最新官方 sample 更新 `model/all_down/all_down.csv` 的 ID 與順序。

## LightGBM baseline

`model/lightgbm_baseline/` 合併 `basic_data/market/` 歷史與 `data/market/` 更新資料，用全部 100 支股票訓練報酬回歸模型，驗證及預測針對目標 50 支；同日重複資料以 `data/` 為準。提供基本價量／技術特徵、5 折 walk-forward 驗證與獨立最終測試、官方報酬計分及 manager 預測介面。

```sh
.venv/bin/python -m pip install -r model/lightgbm_baseline/requirements.txt
.venv/bin/python model/lightgbm_baseline/train.py
.venv/bin/python model/lightgbm_baseline/predict.py --template /path/to/daily_sample_submission.csv --output model/lightgbm_baseline/prediction.csv
```

manager 可透過「重新訓練＋預測」使用此模型。每次訓練重新讀檔，預設涵蓋最新有效行情日期，無須調整固定截止日。預測日期必須晚於訓練截止日；Phase 1 歷史回測需另訓練截止日更早的模型。特徵、評估設計、產出檔案與參數詳見 [模型說明](model/lightgbm_baseline/README.md)。

執行 `.venv/bin/python model/lightgbm_baseline/tune.py` 可在維持原有特徵的情況下比較 12 組參數，依 walk-forward validation 的平均 score 選出最佳組合並訓練模型。排名存於 `artifacts/search/<run_id>/search_results.csv`；成功後的 `artifacts/best_params.json` 會由後續 `train.py` 與 manager 重新訓練自動沿用。

## Logistic regression

`model/logistic_regression/` 提供依報酬絕對值加權的 logistic regression，附上原專案訓練至
2026-05-29 的模型，可直接透過 manager 預測；`usage.txt` 同時提供重新訓練指令。

```sh
.venv/bin/python -m pip install -r model/logistic_regression/requirements.txt
.venv/bin/python model/logistic_regression/predict.py --template /path/to/daily_sample_submission.csv --output model/logistic_regression/prediction.csv
```

預設重訓保留 Phase 1 的時間切分，只使用 2026-05-31 以前的標籤；需要包含更近期資料時，
請指定 `--train-end` 與 `--validation-start`。資料目錄必須包含完整歷史，僅有九月以後的更新資料
無法執行預設重訓。特徵、資料需求、日期限制與參數詳見 [模型說明](model/logistic_regression/README.md)。

## 美國日線資料（FinMind）

透過 [FinMind 美股技術面 API](https://finmind.github.io/tutor/UnitedStatesMarket/Technical/) 的 `USStockPrice` 下載 **2022-01-01 至執行當日**的可用日線，存於 repo 根目錄的 `us_data/`。結束日期以紐約日期計算，實際最新交易日取決於 API 已公布的資料。使用既有 `requests` 與 `python-dotenv`，可在 `.env` 設定 `FINMIND_TOKEN=你的token` 以帶入 Bearer token。未設定時會嘗試公開額度，權限與配額以 API 回應為準。

首次執行歷史下載：

```sh
.venv/bin/python script/download_us.py
```

下載範圍尚未另行指定時，預設為 20 個代碼：`^DJI ^GSPC ^IXIC ^SOX ^VIX SPY QQQ SMH AAPL MSFT NVDA AMD AVGO TSM ASML AMAT LRCX MU INTC QCOM`。可直接指定清單，或使用 `--all` 讀取 `USStockInfo` 的所有代碼：

```sh
.venv/bin/python script/download_us.py --symbols '^IXIC' '^GSPC' NVDA AMD TSM
.venv/bin/python script/download_us.py --all
```

清單與起始日期保存於 `us_data/config.json`。後續更新腳本與 manager 都會沿用這份設定。全市場代碼很多，初次下載與每日更新會需要大量請求，請依自身額度選擇範圍。

增量更新使用：

```sh
.venv/bin/python script/update_us.py
```

- 歷史資料依年份下載，每個成功區間立即保存。中斷後重新執行會依進度續傳。
- 歷史下載完成後，每次更新會重抓最近 7 個日曆日，依日期合併、去重，API 新列覆蓋相同日期。可用 `--overlap-days` 調整範圍。需要回補更早的調整收盤價修訂時，使用 `--full` 重抓全部年份。
- 每次請求預設至少相隔 13 秒。429 會保存等待期限，依 `Retry-After` 等待，缺少該欄位時等待一小時。再次執行會繼續進度。
- `--max-requests 25` 可限制單次請求數。達到上限會保存游標，之後重跑接續；`--output-dir` 可指定其他輸出目錄。Manager 固定使用 repo 的 `us_data/`。
- CSV 以原子替換保存，API 或欄位錯誤時保留之前成功下載的內容。`metadata.json` 記錄各代碼的列數、最新資料日期、下載進度與錯誤，輸出的 `pending` 代表仍有未完成的工作。

檔案結構如下：

```text
us_data/
  config.json
  metadata.json
  prices/
    AAPL.csv
    NVDA.csv
    ^IXIC.csv
```

CSV 欄位為 `date,stock_id,Open,High,Low,Close,Adj_Close,Volume`，每個代碼一份檔案。原始價格與還原收盤價按 API 提供的值保存。模型可從 `MANAGER_US_DATA_DIR` 取得此目錄，再自行讀取需要的代碼。

先完成一次歷史下載，再啟動或重啟 manager。當 `us_data/config.json` 存在時，manager 會在啟動同步、05:10、22:10，以及手動「同步資料與預測」時更新美股資料。另外每天 **08:10（台北時間）** 有一個美股更新排程，配合文件標示的 08:00 更新時間與本專案 08:30 提交排程。Manager 每次最多 25 個 API 請求，未完成或失敗時會再嘗試分批續傳。完整市場的大量資料可能無法在提交前更新完畢。

美股 CSV 的內容雜湊會與比賽資料版本一起保存。美股有更新時，也會為所有模型重新產生預測；內容沒有改變時會略過成功且日期相符的模型。美股連線失敗會記錄於「執行動態」，比賽資料同步會繼續進行。

可執行離線測試：

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_finmind_us.py' -v
```

## 同步、排程與計分

- 台北時間每日 **05:10、22:10** 抓取資料；啟動時立即同步，網頁也可手動觸發。
- 每 5 分鐘更新比賽資訊。下載失敗每 10 分鐘重試；睡眠或停機期間不會執行，重啟時立即抓最新資料。
- 比較 ZIP 中的檔名與內容雜湊，忽略 ZIP 封裝時間；內容改變才為所有模型重新預測。新模型、上次失敗的模型也會在同步時執行。
- 訓練只在按鈕觸發時執行，資料更新只預測。資料同步與模型工作依序執行，防止資料替換與模型讀檔同時發生；提交使用獨立佇列及已封存的 CSV，不會被其他模型的長時間訓練擋住。某模型失敗後仍繼續其他模型。
- 依 `final.pdf`，每個股票日的貢獻是 `(2 * prediction - 1) * (etl:adj_close - etl:adj_open) / etl:adj_open`。只有某日全部 50 支股票的價格都有效時才公布當日平均，缺漏日期保持待計分，後續同步會重算已保存預測。
- 每次預測、訓練 Log、資料版本雜湊與分數永久保存在 `manager_data/`；不要把它提交到 Git。
- 當日 09:00 前完成的預測列為「即時」，其他列為「歷史回算」。模型卡片的即時成績採每個日期最後一份截止前預測再取日均，不會重複計入同一天。歷史回算不代表樣本外績效，manager 無法驗證模型內部是否有資料洩漏。

## 提交

Phase 2 若當日有開放場次，截止時間為**該交易日早上 09:00（Asia/Taipei）**，不是晚上。Manager 在 09:00 起停止接受該場次提交；即使 API 缺少截止時間或回報較晚時間，也會套用此限制，官方若提早關閉仍以官方狀態為準。

05:10、22:10 同步資料與產生預測。自動提交在每天 **08:30（Asia/Taipei）** 開始，依模型名稱排序上傳 `model/` 下的所有模型，**選定模型移到最後一個上傳**。啟用方式是先選定模型，再勾選「每天 08:30 提交全部模型」。原本的 `autoSubmit` 開關設定會沿用。手動「提交選定模型」按鈕仍會立即提交目前選定的單一模型。

每天的批次首次執行時保存模型清單、順序與選定模型，重啟或限流重試會依同一批次續傳。批次開始後調整選擇，會影響後續日期的新批次。送出其他模型前，會先檢查最後選定模型的預測有效性。模型預測若落後資料版本或場次日期，會排入重新預測並等待下一次重試。最新執行已失敗的其他模型會略過並記錄原因，選定模型失敗則等待修復後重試。

**每個模型各占一次額度**，即使兩個模型輸出完全相同也各自上傳。額度不足時，保留最後一次給選定模型，略過的模型與原因會顯示在執行動態。每次提交前會再次檢查官方場次、截止時間及額度。08:30 是批次開始時間，09:00 為截止時間，實際收到時間取決於模型是否就緒及網路回應。

服務需要持續執行。若 08:30～09:00 間啟動或才開啟自動提交，會補跑當日批次。尚未完成的批次每分鐘重試，09:00 起停止提交。已送出的模型會保存進度，部分完成時重新啟動會從剩餘模型繼續。上傳結果不明時，先讀取官方紀錄，用該次唯一檔名確認是否已收件，確認後再接續批次。未確認的上傳會保持等待並顯示錯誤原因。

查詢狀態與上傳共用登入 session，遇到 429 時依官方 `Retry-After` 等待，缺少該欄位時預設等待 60 秒。登入失敗或官方回傳 429 後，排程可在截止前重試。批次執行期間，手動上傳也使用同一把提交鎖，維持 manager 的提交順序。

### 官方分數

每次 manager 上傳都附帶獨立檔名，以及包含模型名稱和 run ID 的備註。Manager 每 5 分鐘讀取官方 `mySubmissions` 與 `submissionHistory`，將分數對回各模型的提交紀錄。管理員公布成績後，網頁「官方提交與分數」會更新所有模型的分數，包含已被最後選定模型取代的歷史提交。尚未計分顯示 Pending，分數 0 會正常顯示。模型卡片同時顯示官方日均分數。

官方分數與本機依股價計算的分數分別保存。本機分數仍在「預測紀錄與分數」顯示。已保存的舊版提交若只有通用 `prediction.csv` 檔名，無法可靠對回官方模型分數，會保留原紀錄。唯一檔名的對應從本次更新後的提交開始建立。

隊友若在 manager 批次完成後另行上傳，官方仍依最後一筆有效提交採計，請協調當日的提交操作。

Phase 1 保留既有 CLI：

```sh
.venv/bin/python script/submit.py path/to/prediction.csv --phase 1 --submit
```

## 本機使用與測試

預設只監聽 `127.0.0.1`；本服務會執行本機指令且未提供帳號登入，請在自己的電腦使用。畫面只提供 manager 的狀態、預測與 Log，不會提供 `.env` 或任意 repo 檔案。網頁每 5 秒更新，不必重新整理。

```sh
.venv/bin/python -m unittest discover -s tests -v
```

測試使用暫存目錄與 mock 提交，涵蓋官方計分公式、延後計分、封裝差異、路徑檢查、模型失敗、預測封存、提交日期與重複提交、HTTP 操作。不會送出真實比賽提交。

## 測試 08:30 自動提交

1. 更新程式並重啟服務（可保留自訂 host 與 port）：

   ```sh
   .venv/bin/python -m manager --host 0.0.0.0 --port 8123
   ```

2. **不連線、不上傳的排程測試**（任何時間都可執行，不必修改工作站時鐘）：

   ```sh
   .venv/bin/python -m unittest discover -s tests -p 'test_scheduled_submit.py' -v
   ```

   使用暫存目錄、模擬時鐘與 mock 官方 API，檢查 08:29 不送、08:30 觸發、台北時區、09:00 不送、重啟防重複、失敗重試、訓練不中斷提交及關閉開關的行為。

3. **檢查工作站的真實預測，但不上傳**：在網頁選定模型，完成訓練與當日預測，再按「檢查提交（不上傳）」。也可以：

   ```sh
   curl -sS -X POST http://127.0.0.1:8123/api/submit-preview \
     -H 'Content-Type: application/json' -d '{}'
   ```

   此操作會登入取得最新比賽狀態並檢查預測檔，但不呼叫提交 API、不消耗額度、不標記排程完成。通過時回傳 `status: "dry-run"`、模型、日期與股票數；如果場次未開或已過 09:00，會回報無法提交，這是預期行為。若只想做不實際上傳的測試，請先關閉自動提交開關。

4. **實際排程驗證**：選好模型、確認當日預測已成功、勾選自動提交，讓服務在 08:30 前持續執行。08:30 後查看「執行動態」及官方 Submission Status，確認有當日提交；不要用手動提交按鈕取代排程測試。

批次上傳與官方分數的離線測試使用暫存資料與模擬 API，可在任何時間執行：

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_batch_submit.py' -v
```

涵蓋全部模型排序、選定模型最後提交、相同預測分別提交、額度預留、限流續傳、重啟續傳、收件結果確認，以及當前／歷史提交的官方分數回填。
