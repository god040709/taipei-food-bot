"""
收集微調資料（在本機跑，需要環境變數 LLM_API_KEY）：讓真的機器人跑對話，記下每一次「解析條件」時
交給 LLM 的內容（messages），以及現在的 35B 模型（老師）回的 JSON。

標準答案（label）用「程式修正後」的結果（finetune/labels.py），老師原本的答案另外存在 teacher，
changed 列出程式修正了哪些欄位（人工抽查就看這些）。

用法：
    python finetune/collect.py train --n 300          # 產生 300 段訓練對話（finetune/dialogues.py）
    python finetune/collect.py eval                   # tests/scenarios.py 的情境，當評估資料
    python finetune/collect.py cases                  # eval_cases.json 的 153 個情境（第二份評估資料）
    python finetune/collect.py focus --n 150          # 補強資料：針對小模型弱點的對話（dialogues.py 的 FocusGen）
    python finetune/collect.py train --n 5 --max-calls 20   # 小量試跑
中斷後再跑一次會接著做（已經收集過的對話會跳過）。
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, HERE)

from bot import Bot  # noqa: E402
from labels import changed_fields, gold_label  # noqa: E402
from llm import DirectLLM, LLMError, is_parse_request  # noqa: E402

DATA_DIR = os.path.join(HERE, "data")


class Recorder:
    """包住真的 LLM：記下解析條件的那一次呼叫。other=False 時，其他呼叫（反問的措辭、閒聊）不打 LLM，
    讓機器人改用固定的句子，省呼叫次數（解析時只會看到「助理（反問）：…」，影響不大）。"""

    def __init__(self, llm, other: bool):
        self.llm = llm
        self.other = other
        self.last = None

    @property
    def enabled(self):
        return self.llm.enabled

    def chat(self, messages, max_tokens=400, schema=None, temperature=0):
        parse = is_parse_request(schema)
        if not parse and not self.other:
            raise LLMError("收集資料時不用")
        out = self.llm.chat(messages, max_tokens, schema, temperature)
        if parse:
            self.last = {"messages": messages, "output": out}
        return out


def eval_dialogues() -> list:
    from scenarios import SCENARIOS
    return [(f"eval-{i:03d}", turns) for i, (_, turns) in enumerate(SCENARIOS)]


def case_dialogues() -> list:
    """第二份評估資料：附預期結果的 153 個情境（eval_cases.json，說法跟 tests/scenarios.py 又不一樣）。"""
    with open(os.path.join(HERE, "eval_cases.json"), encoding="utf-8") as f:
        return [(f"case-{i:03d}", c["turns"]) for i, c in enumerate(json.load(f))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["train", "focus", "eval", "cases"])
    ap.add_argument("--n", type=int, default=300, help="訓練對話的段數")
    ap.add_argument("--max-calls", type=int, default=1500, help="這次最多呼叫 LLM 幾次")
    ap.add_argument("--other-llm", action="store_true", help="反問、閒聊也用 LLM（比較像正式環境，呼叫次數約多一倍）")
    args = ap.parse_args()

    llm = DirectLLM(max_calls=args.max_calls)
    if not llm.enabled:
        sys.exit("沒有設定環境變數 LLM_API_KEY")
    rec = Recorder(llm, args.other_llm)
    bot = Bot(rec)
    bot.valves.USE_LLM_REASON = False  # 推薦理由跟解析無關，不寫

    if args.split == "train":
        from dialogues import training_dialogues
        dialogues = training_dialogues(args.n)
    elif args.split == "focus":
        from dialogues import focus_dialogues
        dialogues = focus_dialogues(args.n)
    elif args.split == "eval":
        dialogues = eval_dialogues()
    else:
        dialogues = case_dialogues()

    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{args.split}.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {json.loads(line)["dialogue_id"] for line in f if line.strip()}

    n_rows = n_changed = 0
    with open(path, "a", encoding="utf-8") as out:
        for did, turns in dialogues:
            if did in done:
                continue
            rows, messages = [], []
            try:
                for t, text in enumerate(turns):
                    messages.append({"role": "user", "content": text})
                    rec.last = None
                    reply = bot.pipe({"messages": messages})
                    messages.append({"role": "assistant", "content": reply})
                    trace = bot.trace
                    if rec.last is None or trace.get("llm_raw") is None:
                        continue  # 這一句沒有走到 LLM 解析（例如「重新開始」）或 LLM 失敗
                    teacher = trace["llm_raw"]
                    gold = gold_label(teacher, trace, reply, text)
                    rows.append({
                        "dialogue_id": did, "turn": t, "user": text,
                        "messages": rec.last["messages"],
                        "label": json.dumps(gold, ensure_ascii=False, separators=(",", ":")),
                        "teacher": json.dumps(teacher, ensure_ascii=False, separators=(",", ":")),
                        "changed": changed_fields(teacher, gold),  # 程式修正了老師的哪些欄位
                        "notes": trace.get("notes", []),
                    })
            except LLMError as e:
                print(f"{did} 停止：{e.note}")
                break  # 呼叫上限或連不到：這段對話不寫，下次接著跑
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
                n_changed += bool(row["changed"])
            out.flush()
            print(f"{did}：{len(rows)} 筆（累計呼叫 {llm.calls} 次）")
    print(f"新增 {n_rows} 筆，其中程式有修正老師的 {n_changed} 筆 → {path}")


if __name__ == "__main__":
    main()
