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
| `MANAGER_DATA_DIR` | 解壓後的 `data/`，含 `market/*.csv`、`global/global_data.csv` |
| `MANAGER_TEMPLATE` | 本次參考模板，包含正確股票順序與日期 |

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

05:10、22:10 僅同步資料與產生預測，**不會立即自動上傳**。自動提交改為每天 **08:30（Asia/Taipei）**，使用選定模型最新一次執行的已封存預測；該次必須成功，且符合當日日期與目前資料版本；08:30 不會重新訓練或等待所有模型完成。排程每 15 秒檢查一次，網路延遲會影響實際收到時間。

在模型卡片按「選為提交模型」，並勾選「每天 08:30 自動提交（台北時間）」。預設關閉自動提交；設定會保存，舊版已開啟的自動提交設定會直接沿用為 08:30 排程。手動「提交選定模型」仍會立即嘗試上傳。

服務需要持續執行。若 08:30～09:00 間啟動或才開啟自動提交，會補交；場次未開、沒有有效預測或網路讀取失敗時，每分鐘重試，09:00 起停止。當日排程一旦已送出或結果不明，就不再自動重送，即使換模型或重啟服務亦同。若 09:00 後才啟動，不會補交已截止場次。執行動態會顯示排入佇列、失敗原因與排程完成狀態。

目前提交按鈕與自動提交針對 **Phase 2**：每次提交前重抓 state，確認場次仍開放、額度已知且大於零、預測僅有當日一欄、使用目前資料版本，且模型最新執行成功。相同日期與 CSV 內容只提交一次。若提交網路結果不明，會保存 `uncertain` 並停止重送相同檔案，請到官方 Submission Status 確認是否收到，再用既有 CLI 處理必要的人工重送。

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
