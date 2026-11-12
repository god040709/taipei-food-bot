"""
評估小模型解析條件的能力（在虛擬機上跑，用 /opt/venv_torch 的 python）。

拿 finetune/data/eval.jsonl（tests/scenarios.py 的對話，說法跟訓練資料不同）逐筆讓小模型回答，跟答案比：
  - 程式沒修正的那幾筆：答案＝老師（35B）的 JSON
  - 程式有修正的那幾筆：四個基本欄位跟「程式修正後」的值比，看小模型會不會犯老師一樣的錯

用法：
    python finetune/evaluate.py --model unsloth/Qwen3.5-4B              # 沒微調的小模型（基準）
    python finetune/evaluate.py --model finetune/outputs/parser-lora     # 微調後
    python finetune/evaluate.py --model finetune/outputs/parser-lora --bot   # 另外讓整個機器人用小模型跑一遍情境
結果寫在 finetune/outputs/eval_<名稱>.txt（--bot 的對話在 bot_<名稱>.txt）。
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, HERE)

from collect import eval_dialogues, is_parse_request  # noqa: E402
from llm import LLMError  # noqa: E402

CORE = ["district", "category", "party_size", "budget"]
SCALARS = ["intent"] + CORE + ["total_budget", "choose_category", "near", "no_chain", "hidden_gem", "budget_min",
                                "sort", "outside_area", "unknown_food"]
SETS = ["waived", "reject", "keep", "relax", "avoid_categories", "categories"]


class LocalModel:
    """用 unsloth 載入小模型（或微調後的 LoRA），介面跟機器人用的 LLM 一樣：chat(messages, …) → 文字。
    只回答解析條件；其他呼叫（反問措辭、閒聊、推薦理由）丟 LLMError，機器人會改用固定的句子。"""
    enabled = True

    def __init__(self, name: str, max_len: int = 4096):
        from unsloth import FastLanguageModel
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=name, max_seq_length=max_len, load_in_4bit=True)
        FastLanguageModel.for_inference(self.model)
        self.seconds = []

    def generate(self, messages: list, max_tokens: int = 800) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False)
        inputs = self.tokenizer(text=prompt, return_tensors="pt").to("cuda")
        start = time.time()
        out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, use_cache=True)
        self.seconds.append(time.time() - start)
        return self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def chat(self, messages, max_tokens=400, schema=None, temperature=0):
        if not is_parse_request(schema):
            raise LLMError("小模型只負責解析條件")
        return self.generate(messages, max_tokens)


def extract_json(text: str):
    from bot import Bot
    try:
        return Bot._extract_json(text)
    except ValueError:
        return None


def recent_names(messages: list) -> set:
    system = messages[0]["content"]
    if "最近推薦過的店：\n" not in system:
        return set()
    lines = system.split("最近推薦過的店：\n", 1)[1].splitlines()
    return {line.split(". ", 1)[1] for line in lines if ". " in line}


def norm(field, value, names=None):
    if field in SETS:
        items = set(value) if isinstance(value, list) else set()
        return items & names if field in ("reject", "keep") else items
    return value if value not in ("", []) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="基準模型名稱，或 finetune/outputs/ 底下的 LoRA 資料夾")
    ap.add_argument("--data", default=os.path.join(HERE, "data", "eval.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="只評估前幾筆（試跑用）")
    ap.add_argument("--bot", action="store_true", help="另外讓整個機器人用小模型跑 tests/scenarios.py")
    ap.add_argument("--min-free-gb", type=float, default=6)
    args = ap.parse_args()

    import torch
    free = torch.cuda.mem_get_info()[0] / 1024 ** 3
    if free < args.min_free_gb:
        raise SystemExit(f"GPU 只剩 {free:.1f} GB（需要 {args.min_free_gb} GB），可能有人在用，晚點再跑")

    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    llm = LocalModel(args.model)
    tag = os.path.basename(os.path.normpath(args.model)).replace("/", "_")
    os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)

    hit = {f: 0 for f in SCALARS + SETS}
    clean = [r for r in rows if not r["corrected"]]
    corrected = [r for r in rows if r["corrected"]]
    json_ok = all_core = fixed_like_program = same_as_teacher = 0
    lines = []
    for r in rows:
        raw = llm.generate(r["messages"])
        pred = extract_json(raw)
        json_ok += pred is not None
        pred = pred or {}
        names = recent_names(r["messages"])
        label = json.loads(r["label"])
        if not r["corrected"]:
            wrong = []
            for f in SCALARS + SETS:
                ok = norm(f, pred.get(f), names) == norm(f, label.get(f), names)
                hit[f] += ok
                if not ok:
                    wrong.append(f"{f}：答案 {label.get(f)}／小模型 {pred.get(f)}")
            all_core += all(norm(f, pred.get(f)) == norm(f, label.get(f)) for f in ["intent"] + CORE)
            if wrong:
                lines.append(f"✗ {r['dialogue_id']} 第{r['turn']}句「{r['user']}」\n    " + "\n    ".join(wrong))
        else:
            # 程式有修正：差異的欄位，小模型是跟程式修正後一樣（好），還是跟老師一樣（同樣讀錯）
            for f in r["differs"]:
                if pred.get(f) == r["final"][f]:
                    fixed_like_program += 1
                elif pred.get(f) == label.get(f):
                    same_as_teacher += 1
            lines.append(f"△ {r['dialogue_id']} 第{r['turn']}句「{r['user']}」程式修正 {r['differs']} {r['notes']}："
                         f"老師 {[label.get(f) for f in r['differs']]}／程式 {[r['final'][f] for f in r['differs']]}"
                         f"／小模型 {[pred.get(f) for f in r['differs']]}")

    n_diff = sum(len(r["differs"]) for r in corrected)
    seconds = sorted(llm.seconds)
    summary = [
        f"模型：{args.model}",
        f"評估 {len(rows)} 筆（老師答案可信的 {len(clean)} 筆、程式有修正的 {len(corrected)} 筆）",
        f"JSON 格式正確：{json_ok}/{len(rows)}",
        f"意圖＋四個基本欄位全對：{all_core}/{len(clean)}",
        "各欄位正確率：" + "、".join(f"{f} {hit[f] / max(1, len(clean)):.0%}" for f in SCALARS + SETS),
        f"程式有修正的欄位共 {n_diff} 個：小模型跟程式修正後一樣 {fixed_like_program}、跟老師一樣讀錯 {same_as_teacher}",
        f"每句花費：中位數 {seconds[len(seconds) // 2]:.1f} 秒、最慢 {seconds[-1]:.1f} 秒" if seconds else "",
    ]
    path = os.path.join(HERE, "outputs", f"eval_{tag}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n\n" + "\n".join(lines) + "\n")
    print("\n".join(summary))
    print(f"細節：{path}")

    if args.bot:
        from bot import Bot
        bot = Bot(llm)
        bot.valves.USE_LLM_REASON = False
        out = []
        for did, turns in eval_dialogues():
            messages = []
            out.append(f"{'=' * 60}\n【{did}】")
            for text in turns:
                messages.append({"role": "user", "content": text})
                reply = bot.pipe({"messages": messages})
                messages.append({"role": "assistant", "content": reply})
                out.append(f"\n👤 {text}\n\n{reply}\n")
        path = os.path.join(HERE, "outputs", f"bot_{tag}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(out))
        print(f"整個機器人的對話：{path}")


if __name__ == "__main__":
    main()
