"""
跑 tests/scenarios.py 裡的對話情境，輸出整段對話。

用法：
    python tests/run_scenarios.py                       # 只用規則（不呼叫 LLM），直接跑機器人，不用開 API
    python tests/run_scenarios.py --check               # 跟 tests/golden/rules.txt 比對，有差異就列出來
    python tests/run_scenarios.py --update              # 把這次的結果存成新的 tests/golden/rules.txt
    python tests/run_scenarios.py --pipe URL            # 透過 openwebui_pipe.py → API（URL 例如 http://127.0.0.1:5000/chat）
    python tests/run_scenarios.py --llm                 # 用真的 LLM（需要環境變數 LLM_API_KEY；會先顯示呼叫上限）
    python tests/run_scenarios.py --only 子類型           # 只跑名稱含「子類型」的情境
"""
import argparse
import difflib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from scenarios import SCENARIOS  # noqa: E402

GOLDEN = os.path.join(ROOT, "tests", "golden", "rules.txt")


def make_client(args):
    """回傳 (函式：messages → 回覆, 說明)。"""
    if args.pipe:
        from openwebui_pipe import Pipe
        pipe = Pipe()
        pipe.valves.API_URL = args.pipe
        if args.llm:
            pipe.valves.LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
            pipe.valves.LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen3.6-35B-A3B")
        return (lambda messages: pipe.pipe({"messages": messages})), None
    from bot import Bot
    from llm import DirectLLM, NoLLM
    llm = DirectLLM(max_calls=args.max_calls) if args.llm else NoLLM()
    bot = Bot(llm)
    return (lambda messages: bot.pipe({"messages": messages})), llm


def run(args) -> str:
    scenarios = SCENARIOS
    if args.only:
        scenarios = [s for s in scenarios if args.only in s[0]]
    chat, llm = make_client(args)
    lines = []
    for title, turns in scenarios:
        lines.append(f"{'=' * 60}\n【{title}】")
        messages = []
        for text in turns:
            messages.append({"role": "user", "content": text})
            reply = chat(messages)
            messages.append({"role": "assistant", "content": reply})
            lines.append(f"\n👤 {text}\n\n{reply}\n")
    if llm is not None and args.llm:
        lines.append(f"\n（呼叫 LLM {llm.calls} 次）")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--pipe")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--max-calls", type=int, default=120)
    parser.add_argument("--only")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.llm:
        print(f"⚠️ 會呼叫真的 LLM，最多 {args.max_calls} 次", file=sys.stderr)

    text = run(args)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    if args.update:
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print(f"已更新 {GOLDEN}")
    elif args.check:
        with open(GOLDEN, encoding="utf-8") as f:
            golden = f.read()
        diff = list(difflib.unified_diff(golden.splitlines(), text.splitlines(), "golden", "now", lineterm=""))
        print("\n".join(diff) if diff else "✅ 跟 golden 完全一樣")
        sys.exit(1 if diff else 0)
    elif not args.out:
        print(text)


if __name__ == "__main__":
    main()
