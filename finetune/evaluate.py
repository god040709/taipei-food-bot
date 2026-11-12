"""
評估小模型解析條件的能力（在虛擬機上跑，用 /opt/venv_torch 的 python）。

拿評估資料（說法跟訓練資料不同）逐筆讓模型回答，跟標準答案（程式修正後的 JSON，見 labels.py）比。
另外統計「老師（35B）讀錯、程式修正過」的那些欄位，這個模型有沒有讀對。

用法：
    python finetune/evaluate.py --teacher                               # 老師（35B）的分數，不用 GPU
    python finetune/evaluate.py --model unsloth/Qwen3.5-4B              # 沒微調的小模型（基準）
    python finetune/evaluate.py --model finetune/outputs/parser-lora     # 微調後
    python finetune/evaluate.py --model finetune/outputs/parser-lora --bot   # 另外讓整個機器人用小模型跑情境、對答案
    python finetune/evaluate.py --model finetune/outputs/parser-lora --bot --off total_words,category_name
                                                                         # 關掉某些保險規則再跑，看還需不需要
    （加 --data finetune/data/cases.jsonl 換第二份評估資料）
結果寫在 finetune/outputs/eval_<名稱>_<資料>.txt（--bot 的對話在 bot_<名稱>.txt）。
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


def score(rows: list, preds: list) -> tuple:
    """preds：每一筆的 JSON（dict，格式錯誤是 None）。回傳 (摘要的幾行, 錯誤細節的幾行)。
    答案是程式修正後的 JSON（labels.py）；changed 不是空的那幾筆＝老師（35B）讀錯、程式修正過的。"""
    fields = SCALARS + SETS
    hit = {f: 0 for f in fields}
    json_ok = all_core = hard_ok = hard_total = 0
    lines = []
    for r, pred in zip(rows, preds):
        json_ok += pred is not None
        pred = pred or {}
        names = recent_names(r["messages"])
        label = json.loads(r["label"])
        wrong = []
        for f in fields:
            ok = norm(f, pred.get(f), names) == norm(f, label.get(f), names)
            hit[f] += ok
            if not ok:
                wrong.append(f"{f}：答案 {label.get(f)}／模型 {pred.get(f)}")
        all_core += all(norm(f, pred.get(f)) == norm(f, label.get(f)) for f in ["intent"] + CORE)
        for f in r.get("changed", []):  # 老師讀錯的欄位，這個模型有沒有讀對
            hard_total += 1
            hard_ok += norm(f, pred.get(f), names) == norm(f, label.get(f), names)
        if wrong:
            mark = "△" if r.get("changed") else "✗"
            lines.append(f"{mark} {r['dialogue_id']} 第{r['turn']}句「{r['user']}」"
                         + "".join("\n    " + w for w in wrong))
    n = max(1, len(rows))
    summary = [
        f"評估 {len(rows)} 筆（其中老師讀錯、程式修正過的 {sum(bool(r.get('changed')) for r in rows)} 筆）",
        f"JSON 格式正確：{json_ok}/{len(rows)}",
        f"意圖＋四個基本欄位全對：{all_core}/{len(rows)}（{all_core / n:.0%}）",
        "各欄位正確率：" + "、".join(f"{f} {hit[f] / n:.0%}" for f in fields),
        f"老師讀錯的欄位：這個模型讀對 {hard_ok}/{hard_total}",
    ]
    return summary, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="基準模型名稱，或 finetune/outputs/ 底下的 LoRA 資料夾")
    ap.add_argument("--teacher", action="store_true", help="評估老師（35B）收集時的答案，不用 GPU")
    ap.add_argument("--data", default=os.path.join(HERE, "data", "eval.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="只評估前幾筆（試跑用）")
    ap.add_argument("--bot", action="store_true", help="另外讓整個機器人用這個模型跑 eval_cases.json，逐句對答案")
    ap.add_argument("--off", default="", help="--bot 時關掉的保險規則（逗號分隔，名稱見 src/understand.py 的 GUARDS）")
    ap.add_argument("--min-free-gb", type=float, default=6)
    args = ap.parse_args()
    if not (args.model or args.teacher):
        ap.error("要指定 --model 或 --teacher")

    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    data_tag = os.path.splitext(os.path.basename(args.data))[0]
    os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)

    llm = None
    if args.teacher:
        tag = "teacher"
        preds = [json.loads(r["teacher"]) for r in rows]
        extra = []
    else:
        import torch
        free = torch.cuda.mem_get_info()[0] / 1024 ** 3
        if free < args.min_free_gb:
            raise SystemExit(f"GPU 只剩 {free:.1f} GB（需要 {args.min_free_gb} GB），可能有人在用，晚點再跑")
        llm = LocalModel(args.model)
        tag = os.path.basename(os.path.normpath(args.model)).replace("/", "_")
        preds = [extract_json(llm.generate(r["messages"])) for r in rows]
        seconds = sorted(llm.seconds)
        extra = [f"每句花費：中位數 {seconds[len(seconds) // 2]:.1f} 秒、最慢 {seconds[-1]:.1f} 秒"] if seconds else []

    summary, lines = score(rows, preds)
    summary = [f"模型：{'老師（35B，收集時的答案）' if args.teacher else args.model}｜資料：{data_tag}"] + summary + extra
    path = os.path.join(HERE, "outputs", f"eval_{tag}_{data_tag}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n\n" + "\n".join(lines) + "\n")
    print("\n".join(summary))
    print(f"細節：{path}")

    if args.bot:
        if llm is None:  # 老師：真的呼叫 35B（需要 LLM_API_KEY），反問措辭、閒聊不呼叫，跟小模型的條件一樣
            from collect import Recorder
            from llm import DirectLLM
            llm = Recorder(DirectLLM(max_calls=600), other=False)
        run_bot(llm, args.off, f"{tag}{'_off-' + args.off.replace(',', '+') if args.off else ''}")


def run_bot(llm, off: str, tag: str):
    """整個機器人用這個模型解析，跑 eval_cases.json 的情境，每一句對答案（checks.py）。"""
    from bot import Bot
    from checks import check
    bot = Bot(llm, {"OFF_GUARDS": off, "USE_LLM_REASON": False})
    with open(os.path.join(HERE, "eval_cases.json"), encoding="utf-8") as f:
        cases = json.load(f)
    out, failed, n_turns = [], [], 0
    for case in cases:
        messages = []
        out.append(f"{'=' * 60}\n【{case['title']}】")
        for text, expect in zip(case["turns"], case["expect"]):
            messages.append({"role": "user", "content": text})
            reply = bot.pipe({"messages": messages})
            messages.append({"role": "assistant", "content": reply})
            n_turns += 1
            problems = check(reply, expect)
            out.append(f"\n👤 {text}\n\n{reply}\n" + (f"\n✗ {'；'.join(problems)}\n" if problems else ""))
            if problems:
                failed.append(f"【{case['title']}】{text} → {'；'.join(problems)}")
    path = os.path.join(HERE, "outputs", f"bot_{tag}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"關掉的規則：{off or '無'}｜{n_turns} 句、有問題 {len(failed)} 句\n\n" + "\n".join(failed)
                + "\n\n" + "\n".join(out))
    print(f"整個機器人（關掉的規則：{off or '無'}）：{n_turns} 句、有問題 {len(failed)} 句 → {path}")


if __name__ == "__main__":
    main()
