# 微調實驗：用小模型做「解析條件」

目前機器人用實驗室伺服器上的 Qwen3.6-35B-A3B 解析使用者的話（對話 → 條件 JSON），再用規則把關。
這個實驗想知道：**在虛擬機上微調一個小模型（Qwen3.5-4B），能不能做到跟 35B 差不多、甚至少犯一些程式要修正的錯？**

這個資料夾是獨立的實驗，不會改到機器人原本的程式（只會 import 來用）。

## 流程

| 步驟 | 在哪裡跑 | 指令 |
|---|---|---|
| 1. 收集訓練資料 | 本機（需要 `LLM_API_KEY`） | `python finetune/collect.py train --n 300` |
| 2. 收集評估資料 | 本機 | `python finetune/collect.py eval` |
| 2b. 第二份評估資料 | 本機 | `python finetune/collect.py cases` |
| 2c. 老師（35B）的分數 | 本機或虛擬機（不用 GPU） | `python finetune/evaluate.py --teacher` |
| 3. 檢查資料長度 | 虛擬機（不用 GPU） | `python finetune/train.py --dry-run` |
| 4. 評估沒微調的小模型（基準） | 虛擬機（GPU） | `python finetune/evaluate.py --model unsloth/Qwen3.5-4B` |
| 5. 微調 | 虛擬機（GPU） | `python finetune/train.py --steps 300` |
| 6. 評估微調後 | 虛擬機（GPU） | `python finetune/evaluate.py --model finetune/outputs/parser-lora` |

虛擬機上用**原本的 python**（`/opt/venv_torch`，有 torch、unsloth），不是跑 API 用的 `~/venvs/foodbot`。

## 資料怎麼來

- **訓練**：`dialogues.py` 用元件隨機組出對話（地區／捷運站、類型的各種說法、人數、預算、其他條件，加上換一批、
  不要第幾間、換類型、加減預算等追問），讓真的機器人跑，記下每次解析時交給 LLM 的內容和 35B（老師）回的 JSON。
- **評估**：`tests/scenarios.py` 的 99 個情境，說法跟訓練資料不同，避免只是背答案。
- **標準答案用「程式修正後」的結果**（`labels.py`），不是直接用老師的答案：老師讀錯、程式靠規則修正的地方
  （例如「2人 500 貴一點沒差」被當成總預算），答案寫的是修正後的值。這樣小模型才有機會「本來就做對」，
  之後對應的規則才有理由拿掉。每一筆的 `changed` 列出修正了哪些欄位，人工抽查就看這些。
- 第二份評估資料 `eval_cases.json`：附預期結果的 153 個情境（`python finetune/collect.py cases`）。
- 老師自己的分數不用 GPU 就能算：`python finetune/evaluate.py --teacher`。
- 收集時反問、閒聊不呼叫 LLM（機器人改用固定句子），每句大約只花 1 次呼叫。

## 注意

- **GPU 是共用的**（RTX 5080 16GB）。`train.py`、`evaluate.py` 開始前會檢查 GPU 剩多少空間，不夠就不跑。
- 訓練好的 LoRA 存在 `finetune/outputs/`，不進 git。
- 這是實驗，結果好壞都可以寫進報告；機器人正式用的還是 35B＋規則。
