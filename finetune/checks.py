"""
對答案：eval_cases.json 每一句都有預期結果（類型、預算、要不要反問…），另外每一句回覆都做共同檢查
（簡體字、照抄指示、理由崩壞、超過 5 家、同一批重複品牌、沒指定類型卻推酒吧甜點）。
evaluate.py --bot 用它來看整個機器人換成小模型、關掉某些保險規則之後，表現有沒有變差。
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from recommender import brand_key  # noqa: E402

SIMPLIFIED = set("这们说时会还对没发让样问题选欢吗为么东车门书长开关买卖读写实际应该与过见现红马鱼鸡汤热饭间区择个错适价种经质环气评论优烧锅预荐厅务体验订够总")
PARROT = ["要讓使用者知道", "開場", "請自然", "條件裡", "這批推薦有些事", "intro", "facts", "JSON"]


def cond_parts(reply):
    m = re.search(r"🔎 搜尋條件：([^（\n]*)", reply)
    return m.group(1) if m else None


def kind(reply):
    if reply.startswith("🤔"):
        return "ask"
    if reply.startswith("💬"):
        return "chat"
    if cond_parts(reply) is not None:
        return "rec"
    return "other"


def check(reply, exp):
    problems = []
    cond = cond_parts(reply) or ""
    parts = cond.split("｜") if cond else []
    if exp.get("k") and kind(reply) != exp["k"]:
        problems.append(f"應該是 {exp['k']}，實際是 {kind(reply)}")
    if exp.get("d") and (len(parts) < 1 or exp["d"] not in parts[0]):
        problems.append(f"地區應該是 {exp['d']}")
    if exp.get("c") and (len(parts) < 2 or not parts[1].startswith(exp["c"])):
        problems.append(f"類型應該是 {exp['c']}")
    if exp.get("p") and exp["p"] not in parts[2:3]:
        problems.append(f"人數應該是 {exp['p']}")
    if exp.get("b") and exp["b"] not in parts[3:4]:
        problems.append(f"預算應該是 {exp['b']}")
    for s in exp.get("has", []):
        if s not in cond:
            problems.append(f"條件行應該有「{s}」")
    for s in exp.get("lacks", []):
        if s in cond:
            problems.append(f"條件行不該有「{s}」")
    for s in exp.get("rhas", []):
        if s not in reply:
            problems.append(f"回覆應該有「{s}」")
    # 共同檢查
    if "❌" in reply or "格式不對" in reply or "連不到 LLM" in reply or "LLM 回應錯誤" in reply:
        problems.append("錯誤訊息／LLM 沒用上")
    llm_lines = [l for l in reply.splitlines() if l.startswith(("💬", "🤔", "✨"))]
    text = "\n".join(llm_lines)
    bad = sorted({ch for ch in text if ch in SIMPLIFIED})
    if bad:
        problems.append(f"簡體字：{''.join(bad)}")
    for w in PARROT:
        if w in text:
            problems.append(f"照抄指示：{w}")
    for l in reply.splitlines():
        if l.startswith("✨") and (len(re.findall(r"[\u4e00-\u9fff]", l)) < 8 or re.search(r"[{}\[\]_]", l)):
            problems.append(f"理由崩壞：{l[:30]}")
    if re.search(r"(?<!\d)1\s*(?:至|到|-|～|~)\s*\d+\s*元", text):
        problems.append("價格寫成 1 至 X 元")
    if parts[1:2] == ["不限類別"] or "🎲" in reply:
        shown_cats = re.findall(r"^📍 [^・]+・([^｜]+)", reply, flags=re.M)
        if any(c in ("酒吧", "甜點") for c in shown_cats):
            problems.append("沒指定類型卻推酒吧／甜點")
    names = re.findall(r"^\*\*\d+\. (.+?)\*\*$", reply, flags=re.M)
    if len(names) > 5:
        problems.append("超過 5 家")
    brands = [brand_key(n) for n in names if brand_key(n)]
    if len(brands) != len(set(brands)):
        problems.append("同一批有重複品牌")
    return problems
