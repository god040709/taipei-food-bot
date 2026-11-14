"""
標準答案：把「程式修正後」的結果寫回 LLM 的 JSON 格式。

35B（老師）的 JSON 有時讀錯，程式靠規則修正。如果直接拿老師的答案訓練，小模型會連錯誤一起學走，
規則就永遠拿不掉；所以答案要用程式最後採用的值，讓小模型「本來就該這樣回答」。

只改程式真的有修正的欄位，其他照老師的答案。changed_fields() 列出跟老師不一樣的欄位，方便人工抽查。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lexicon import BUDGET_MIN_WORDS, CHOOSE_MARK, FIELDS  # noqa: E402
from recommender import BUDGET_STEP  # noqa: E402


def gold_label(raw: dict, trace: dict, reply: str, last_text: str) -> dict:
    """raw：老師回的 JSON；trace：機器人這一句的除錯紀錄；reply：機器人的回覆；last_text：使用者這一句。"""
    d = trace.get("llm") or {}
    final = list(trace.get("final") or [None] * 4)
    prefs = trace.get("prefs") or {}
    notes = trace.get("notes") or []
    g = dict(raw)

    # 不喜歡／保留的店：只能是推薦過的店名（還沒推薦時 LLM 會填「第5間」）
    g["reject"], g["keep"] = d.get("reject", []), d.get("keep", [])
    if raw.get("intent") == "chat":
        return chat_label(raw, g)

    # 地區、類型、人數：程式最後採用的值（包含老師漏掉、由前文沿用的）
    for f, v in zip(FIELDS[:3], final[:3]):
        g[f] = v
    # 「你幫我挑」：類型留空，由程式隨機挑
    if "🎲" in reply or CHOOSE_MARK in reply:
        g["choose_category"], g["category"] = True, None

    # 總預算：這次沒說總共、合計，程式不採用
    if any(n.startswith("LLM 判斷總預算") for n in notes):
        g["total_budget"] = None
    # 預算：程式自己會加 200、會把總預算除以人數，所以答案填加之前的、總預算時填 null
    if g.get("total_budget"):
        g["budget"] = None
    elif final[3] is not None and any(n.startswith("加預算") for n in notes):
        g["budget"] = final[3] - BUDGET_STEP
    else:
        g["budget"] = final[3]
    if any(n.startswith("LLM 判斷要加預算") for n in notes):
        g["relax"] = [x for x in g.get("relax") or [] if x != "budget"]

    # 這句話沒提到、程式不採用的預算下限（LLM 會受前一句「1000以上」影響）。
    # （「不要連鎖／隱藏版」老師常把沒提到的填 false，程式當成沒說，效果跟 null 一樣，不改）
    if g.get("budget_min") and not any(w in last_text for w in BUDGET_MIN_WORDS):
        g["budget_min"] = None

    # 不要的類別：程式沒採用的拿掉（「換成拉麵」不代表不要原本的；這次說想吃的不會同時不要）
    avoid = prefs.get("avoid", [])
    g["avoid_categories"] = [c for c in g.get("avoid_categories") or [] if c in avoid]
    # 多類別：老師給了兩種以上、程式卻沒採用 → 用程式的
    if len(g.get("categories") or []) >= 2 and g["categories"] != prefs.get("cats"):
        g["categories"] = prefs.get("cats", [])
    # 捷運站：規則讀到的優先（LLM 會把「西門町」換成行政區）
    llm_near = (d.get("prefs") or {}).get("near")
    if prefs.get("near") and llm_near != prefs["near"]:
        g["near"] = prefs.get("near_label") or prefs["near"]
    # 「不限」：只算本來就不知道的欄位
    g["waived"] = [f for f in g.get("waived") or [] if final[FIELDS.index(f)] is None] \
        if all(f in FIELDS for f in g.get("waived") or []) else []
    return g


def chat_label(raw: dict, gold: dict) -> dict:
    """閒聊（「謝謝」「股票會漲嗎」）：照老師的答案，不把前面的條件填進來（閒聊時程式不看條件，
    填了反而教小模型閒聊也要抄條件）；只保留程式整理過的店名。"""
    return {**raw, "reject": gold.get("reject", []), "keep": gold.get("keep", [])}


def changed_fields(raw: dict, gold: dict) -> list:
    def same(a, b):
        if isinstance(a, list) or isinstance(b, list):
            return set(a or []) == set(b or [])
        return a == b or (a in (None, "", []) and b in (None, "", []))
    return [k for k in gold if not same(raw.get(k), gold.get(k))]
