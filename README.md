# Taipei Food Bot 台北餐廳推薦聊天機器人

用聊天的方式找台北的餐廳：輸入「我們三個人想在信義吃麻辣鍋 一人一千」，系統會解析出地區、料理類型、人數和預算，並推薦最合適的 5 家餐廳。

## 系統架構

```
使用者在 Open WebUI 輸入需求
        │
openwebui_pipe.py      解析需求（規則解析，或 LLM 解析）
        │  POST /recommend
api.py                 推薦 API（預設 port 5000）
        │
src/recommender.py     篩選 → 計算推薦指數 → 取前 5 名
        │
data/taipei_all_restaurants.csv   台北 12 區 × 14 類，約 2,900 家餐廳
        ▲
collector_taipei.py    從 Google Places API (New) 採集資料
```

## 檔案說明

| 檔案 | 用途 |
|---|---|
| `openwebui_pipe.py` | Open WebUI 的 Pipe 函式，負責解析需求、呼叫 API、組合回答 |
| `api.py` | 推薦 API。有 Flask 用 Flask，沒有則用 Python 內建伺服器 |
| `src/recommender.py` | 推薦核心：篩選、加權推薦指數、Google Maps 連結 |
| `collector_taipei.py` | 資料採集（需要 Google API 金鑰，平常不用執行） |
| `data/taipei_all_restaurants.csv` | 餐廳資料 |
| `eval/` | 成效評估：測試問題與評估腳本 |

## 快速開始

### 1. 安裝套件

```bash
pip install -r requirements.txt
```

Windows 若找不到 `pip`，改用 `python -m pip install -r requirements.txt`。

### 2. 啟動推薦 API

**在自己電腦：**

```bash
python api.py
```

看到 `Running on http://127.0.0.1:5000` 即成功，視窗保持開著。

**在實驗室平台（lab-30-docker）：**

```bash
source /opt/venv_torch/bin/activate
cd ~/projects/taipei-food-bot
nohup env HOST=0.0.0.0 python api.py > api.log 2>&1 &
```

`HOST=0.0.0.0` 讓 Open WebUI 容器連得到；`nohup` 讓 API 在背景持續執行。

測試 API 是否正常：

```bash
curl -s -X POST localhost:5000/recommend -H "Content-Type: application/json" \
  -d '{"district":"大安區","category":"燒肉"}'
```

### 3. 在 Open WebUI 加入 Pipe

1. 左下角頭像 → **管理員控制台** → **函式** → **建立**
2. 貼上 `openwebui_pipe.py` 的完整內容，填寫名稱與 ID 後儲存，並打開開關
3. 點函式右邊的**齒輪**設定：

| 設定 | 自己電腦 | 實驗室平台 |
|---|---|---|
| Api Url | 預設即可（`http://127.0.0.1:5000/recommend`） | `http://testbed:5000/recommend` |
| Llm Api Key | 選填 | 實驗室 `.env` 裡的 `LLM_API_KEY` |
| Llm Model | 選填 | `Qwen3.5-35B-A3B` |

4. 新增對話，模型選這個 Pipe，輸入「大安燒肉 兩個人 預算800」測試

> 注意：Pipe 是貼在 Open WebUI 裡的，`git pull` 不會自動更新。`openwebui_pipe.py` 有改動時，要重新貼一次。

### 4. 更新到最新版

```bash
git pull
```

API 要重新啟動才會套用新程式。實驗室平台：

```bash
pkill -f api.py; nohup env HOST=0.0.0.0 python api.py > api.log 2>&1 &
```

## 推薦邏輯

### 需求解析

- **規則解析**（預設，不需要任何金鑰）：支援行政區省略「區」、料理同義詞（麻辣鍋→火鍋、韓國菜→韓式料理）、中文數字（三個人、一千）、「1k」「一人500」「500以內」等說法
- **LLM 解析**（在齒輪填入 LLM 金鑰後啟用）：理解更自由的句子，支援多輪對話（「再便宜一點」），並由 LLM 撰寫推薦理由。LLM 失敗時自動退回規則解析

回答第一行會顯示解析出的條件，以及使用哪種解析方式。

### 資訊不足時反問

地區和類別都沒提到，或總共只知道 1 項條件時（例如「我想吃火鍋」），機器人會先反問缺少的資訊，使用者回答後再合併條件推薦。每次需求只反問一次；使用者回「都可以」「隨便」就直接推薦。可在齒輪的 `Ask Followup` 關閉。

### 推薦根據

每次推薦的最後會列出「📊 推薦根據」：篩選條件與符合家數、推薦指數的計算方式、第一名的理由。每家餐廳的推薦指數也會拆成「評分 + 人氣 + 預算」三項分數。可在齒輪的 `Show Basis` 關閉。

### 推薦指數（0～100）

| 項目 | 權重 | 說明 |
|---|---|---|
| 評分可信度 | 60% | 貝氏平均：評論數少的店，評分會被拉向整體平均，避免「5 則評論 4.9 分」排在「8,000 則評論 4.8 分」前面 |
| 人氣 | 25% | 評論數取 log 後正規化 |
| 預算吻合度 | 15% | 價格區間落在預算內的比例 |

權重可在 `src/recommender.py` 開頭調整。

## 成效評估

```bash
python eval/evaluate.py
```

設定環境變數 `LLM_API_KEY` 後，會一併評估 LLM 解析。

| 指標 | 原始版本 | 目前版本 |
|---|---|---|
| 條件解析整句全對（30 句） | 47% | 100% |
| 前 5 名中評論少於 200 則的比例 | 25% | 1% |
| 前 5 名評論數中位數 | 約 1,150 | 約 2,480 |

> 規則解析是參考這 30 句測試題調整的，100% 偏樂觀。正式報告前，建議另外收集一批新句子再測一次。

## 重新採集資料（選用）

平常不需要執行。要更新餐廳資料時，需要自己的 Google API 金鑰（Google Cloud 啟用 **Places API (New)**，並將金鑰限制為只能使用此 API）。

```powershell
# Windows PowerShell
$env:GOOGLE_MAPS_API_KEY="你的金鑰"
python collector_taipei.py
```

```bash
# macOS / Linux
GOOGLE_MAPS_API_KEY=你的金鑰 python collector_taipei.py
```

抓不到資料時不會覆蓋原本的 CSV。

## 安全注意事項

- **不要把任何 API 金鑰寫進程式碼或 commit 到 GitHub**（本 repo 是公開的）。金鑰一律用環境變數或 Open WebUI 的齒輪設定
- 舊版程式碼曾包含一把 Google 金鑰，該金鑰已失效，請勿再使用
- `.gitignore` 已排除 `.env`、快取檔與壓縮檔
