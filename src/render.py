"""
排版：組出回覆畫面（條件行、開場白與推薦理由、餐廳卡片、說明、除錯資訊）。
是 Bot（bot.py）的一部分：函式裡的 self 就是 Bot，可以互相呼叫。
"""
import json
from typing import Optional

from lexicon import (  # 詞庫、提示詞、文字處理
    ASK_MARK, CHAT_MARK, DEBUG_MARK, FIELDS, LABELS, NEW_REQUEST_MARK, REASON_PROMPT, RELAX_LABELS,
    SORT_LABELS, fix_homophones,
)


class RenderMixin:
    def write_reasons(self, texts: list, conditions, restaurants: list, subtype=None, unsupported=None,
                      notes=None, budget_min=None):
        """回傳 (開場白, 每家的推薦理由 list)。LLM 不能用時，用資料中的事實組成理由。
        unsupported：使用者提到、但資料沒有的條件（例如包廂、停車），不可以說店家符合。
        notes：要讓開場提到的事（例如這次依什麼排序、為什麼沒辦法依距離排）。"""
        fallback = [("；".join(r.get("facts") or []) + "。") if r.get("facts") else ""
                    for r in restaurants]
        if not (self.valves.USE_LLM_REASON and self._llm_enabled() and restaurants):
            return None, fallback
        district, category, party_size, budget = conditions
        need = "；".join(fix_homophones(t)[0] for t in texts[-4:])
        cond = (f"地區={district or '不限'}，類型={category or '不限'}，"
                f"人數={party_size or '不限'}，每人預算上限={budget or '不限'}"
                + (f"，每人預算下限={budget_min}（每人至少花這麼多）" if budget_min else ""))
        if subtype:
            cond += (f"，想吃的子類型={subtype}（subtype_match=true 代表店名看得出是{subtype}；"
                     f"false 的店不可以說它有{subtype}）")
            if not any(r.get("subtype_match") for r in restaurants):
                cond += f"。這批店都看不出有{subtype}，開場和理由都不可以說這些店是{subtype}"
        if unsupported:
            cond += (f"。使用者提到的「{'、'.join(unsupported)}」資料裡沒有，"
                     "不知道店家有沒有，開場和理由都不可以說店家符合這些條件")
        if notes:
            cond += "。" + "。".join(notes)
        keys = ["name", "category", "min_price", "max_price", "subtype_match", "distance",
                "facts", "summary", "highlights", "reviews"]
        data = [{k: r.get(k) for k in keys if r.get(k) not in (None, "", [])} for r in restaurants]
        n = len(restaurants)
        schema = {"type": "object", "additionalProperties": False, "required": ["intro", "reasons"],
                  "properties": {"intro": {"type": "string"},
                                 "reasons": {"type": "array", "items": {"type": "string"},
                                             "minItems": n, "maxItems": n}}}
        try:
            raw = self._chat(
                [
                    {"role": "system", "content": REASON_PROMPT},
                    {"role": "user", "content": f"使用者需求：{need}\n條件：{cond}\n"
                                                "\n"
                                                f"餐廳資料：{json.dumps(data, ensure_ascii=False)}"},
                ],
                max_tokens=900, schema=schema,
            )
            result = self._extract_json(raw)
            reasons = [str(x).strip() for x in result.get("reasons", [])]
            if len(reasons) != len(restaurants):
                raise ValueError("推薦理由數量不符")
            return (result.get("intro") or None), reasons
        except Exception:
            return None, fallback

    @staticmethod
    def near_text(prefs: dict) -> str:
        """「捷運忠孝復興站附近」「西門町・捷運西門站附近」"""
        station, label = prefs["near"], prefs.get("near_label") or prefs["near"]
        if label in station.split("/"):
            return f"捷運{station}站附近"
        return f"{label}・捷運{station}站附近"

    @classmethod
    def describe_conditions(cls, district, category, party_size, budget, source, subtype=None, note="",
                            prefs=None) -> str:
        prefs = prefs or {}
        if prefs.get("near"):
            where = cls.near_text(prefs)
        else:
            where = district or "不限地區"
        parts = [
            where,
            (f"{category}・{subtype}" if subtype else category) if category else "不限類別",
            f"{party_size} 人" if party_size else "人數不限",
            f"每人 {budget} 元內" if budget else "預算不限",
        ] + ([cls.prefs_text(prefs)] if cls.prefs_text(prefs) else [])
        return f"🔎 搜尋條件：{'｜'.join(parts)}（{source}解析{'・' + note if note else ''}）"

    @staticmethod
    def prefs_text(prefs: dict) -> str:
        """其他條件「不要連鎖｜隱藏版｜…」（接在 🔎 搜尋條件那一行後面，下一句會讀回來沿用）。"""
        parts = []
        if prefs.get("cats"):
            parts.append("類型：" + "、".join(prefs["cats"]))
        if prefs.get("no_chain"):
            parts.append("不要連鎖")
        if prefs.get("hidden_gem"):
            parts.append("隱藏版")
        if prefs.get("budget_min"):
            parts.append(f"每人 {prefs['budget_min']} 元以上")
        if prefs.get("avoid"):
            parts.append("不要：" + "、".join(prefs["avoid"]))
        if prefs.get("sort"):
            parts.append(SORT_LABELS[prefs["sort"]])
        parts += [RELAX_LABELS[r] for r in prefs.get("relax", [])]
        return "｜".join(parts)

    @staticmethod
    def price_text(r: dict) -> str:
        # 價格區間是「多數人用餐的花費」，不是最低／最高消費
        low, high = int(r["min_price"]), int(r["max_price"])
        text = f"多數人花費 ${high} 以下 / 人" if low <= 1 else f"多數人花費 ${low} - ${high} / 人"
        if r.get("budget_fit") == "partial":
            text += "（⚠️ 可能超出預算）"
        elif r.get("budget_fit") == "over":
            text += "（⚠️ 超出預算）"
        return text

    def format_card(self, i: int, r: dict, reason: str) -> str:
        lines = [
            f"**{i}. {r['name']}**",
            f"📍 {r['district']}・{r['category']}｜⭐ {r['rating']}"
            f"（{int(r['user_ratings_total'] or 0):,} 則評論）｜💰 {self.price_text(r)}",
        ]
        if reason and self.valves.SHOW_REASONS:
            lines.append(f"✨ {reason}")
        lines.append(f"🏠 {r['address']}｜🗺️ [在 Google Maps 開啟]({r['maps_url']})")
        return "\n".join(lines)

    def explain_facts(self, prefs, data, info, conditions, total, subtype, restaurants, party_relaxed,
                      no_location, exclude, where) -> list:
        """這批推薦要讓使用者知道的事（簡短，不寫放寬到多少）。有 LLM 時交給開場白帶過，沒有時直接顯示。"""
        district, category, party_size, budget = conditions
        found = len(restaurants)
        facts = []
        if no_location:
            facts.append("我讀不到你的位置，沒辦法依距離排，先照一般的推薦順序")
        if prefs.get("near") and not data.get("near"):
            facts.append(f"還沒有捷運{prefs['near']}站的位置資料，先不限地區找")
        elif data.get("radius_relaxed"):
            facts.append(f"捷運{prefs['near']}站附近符合的店不多，範圍放寬了一點")
        relaxed = data.get("relaxed_budget")
        if budget and relaxed:
            # 放寬到原本預算的兩倍以內才說「稍微」（例如 50 元放寬到 450 元就不算稍微）
            over = "稍微超出預算" if relaxed <= budget * 2 else "超出預算"
            facts.append(f"沒有完全符合預算的店，以下是{over}的選擇" if data.get("matched", found) == 0
                         else f"符合預算的店不多，另外補上{over}的店")
        if party_relaxed:
            facts.append(f"資料裡的店標示的人數大多在 10 人以內，{party_size} 人先不限人數找，建議先打電話訂位")
        if total and party_size:
            facts.append(f"總預算 {total} 元、{party_size} 個人，大約每人 {budget} 元")
        if prefs.get("hidden_gem") and 0 < found < 5:
            facts.append(f"符合隱藏版的店只有 {found} 家")
        matched_subtype = data.get("matched_subtype")
        if subtype and matched_subtype is not None and restaurants:
            general = self.shop_label(category)
            shown_sub = sum(1 for r in restaurants if r.get("subtype_match"))
            overview = data.get("subtype_overview") or {}
            if shown_sub == 0 and exclude and matched_subtype == 0:
                facts.append(f"店名看得出是{subtype}的店都推薦過了，這批是一般的{general}，不一定有{subtype}")
            elif shown_sub == 0:
                elsewhere = "，想看其他區的可以說「不限地區」" if overview.get("total") else ""
                facts.append(f"{where or '台北'}找不到店名看得出是{subtype}的店，先推薦一般的{general}，"
                             f"不一定有{subtype}{elsewhere}")
            elif shown_sub < found:
                facts.append(f"前 {shown_sub} 家店名看得出是{subtype}，後面是一般的{general}，不一定有{subtype}")
        if prefs.get("unsupported"):
            facts.append(f"資料沒有{'、'.join(prefs['unsupported'])}的資訊，出發前請在 Google Maps 確認")
        return facts

    @staticmethod
    def new_request_line(info: dict) -> str:
        """「🆕 這是新的需求，…」（也是狀態：之後讀回不喜歡的店時，從這裡重新算）。"""
        notes = []
        if info.get("dropped_prefs"):
            notes.append(f"之前的「{info['dropped_prefs']}」已經不再套用")
        if info.get("cleared_rejects"):
            notes.append("之前不想去的" + "".join(f"「{n}」" for n in info["cleared_rejects"])
                         + "也不再排除，還是不想去可以再跟我說")
        return NEW_REQUEST_MARK + "，".join(notes) if notes else ""

    @staticmethod
    def shop_label(category: Optional[str]) -> str:
        """「火鍋」→「火鍋店」，「美式餐廳」「酒吧」本身就是店，不用再加「店」。"""
        if not category:
            return "餐廳"
        return category if category.endswith(("餐廳", "店", "吧")) else category + "店"

    @staticmethod
    def describe_alternatives(conditions, found: int, alternatives: list, where_label=None) -> str:
        """結果不到 5 家時，告訴使用者可以怎麼放寬條件。"""
        if not alternatives:
            return ""
        district = conditions[0]
        where = where_label or district or "目前條件下"
        if found:
            head = f"💡 {where}符合的只有 {found} 家（資料庫裡的店家數有限）。想看更多選擇，可以直接回覆："
        else:
            head = f"💡 {where}沒有符合的店。可以試試直接回覆："
        lines = ["---", head]
        for alt in alternatives:
            lines.append(f"- 「{alt['reply']}」→ {alt['text']}")
        return "\n".join(lines)

    def note(self, text: str):
        """除錯模式：記下程式整理時做了什麼調整。"""
        self.trace.setdefault("notes", []).append(text)

    @staticmethod
    def pref_texts(prefs: dict) -> list:
        """除錯模式：其他條件寫成中文（「強調＝評價優先」「不要連鎖」）。"""
        names = {"near": "捷運站", "budget_min": "預算下限", "avoid": "不要的類別", "cats": "多類別",
                 "unsupported": "資料沒有的條件"}
        out = []
        for k, v in prefs.items():
            if k == "sort":
                out.append(f"強調＝{SORT_LABELS[v]}")
            elif k == "relax":
                out.append("放寬＝" + "、".join({"distance": "距離", "budget": "加預算"}[r] for r in v))
            elif k in ("no_chain", "hidden_gem") and isinstance(v, bool):
                out.append({"no_chain": "不要連鎖", "hidden_gem": "隱藏版"}[k] + ("" if v else "（取消）"))
            elif k in names:
                out.append(f"{names[k]}＝{'、'.join(map(str, v)) if isinstance(v, list) else v}")
        return out

    def debug_block(self, reply: str) -> str:
        """回覆最後的「🔧 除錯資訊」（Open WebUI 會顯示成可以收合的區塊）。"""
        t = self.trace
        fmt = lambda v: "不限" if v in (None, "", []) else ("、".join(map(str, v)) if isinstance(v, list) else str(v))
        names = ("地區", "類型", "人數", "預算")
        lines = []
        d = t.get("llm")
        if d:
            lines.append("**① LLM 理解**：意圖＝" + d["intent"] + "｜"
                         + "｜".join(f"{n}＝{fmt(d[f])}" for n, f in zip(names, FIELDS))
                         + "".join(f"｜{x}" for x in self.pref_texts(d["prefs"])))
        else:
            lines.append(f"**① LLM 理解**：沒有使用 LLM（{t.get('llm_note') or '未開啟'}），全部用規則解析")
        if "final" in t:
            parts = []
            llm_values = [d[f] for f in FIELDS] if d else [None] * 4
            for n, lv, rv, fv in zip(names, llm_values, t.get("rules") or [None] * 4, t["final"]):
                if not d:
                    parts.append(f"{n}＝{fmt(fv)}")
                elif lv == fv:
                    parts.append(f"{n}＝{fmt(fv)}（LLM）")
                elif rv == fv and fv is not None:
                    parts.append(f"{n}＝{fmt(fv)}（規則）")
                else:
                    parts.append(f"{n}＝{fmt(fv)}（LLM {fmt(lv)}、規則 {fmt(rv)}，程式調整）")
            lines.append("**② 程式整理**：" + "｜".join(parts + self.pref_texts(t.get("prefs") or {})))
            lines += [f"　・{n}" for n in t.get("notes", [])]
        if reply.startswith(CHAT_MARK):
            lines.append("**決定**：閒聊，由 LLM 回覆")
        elif reply.startswith(ASK_MARK):
            missing = [LABELS[f] for f in t.get("missing") or []]
            lines.append("**決定**：反問" + (f"（還缺：{'、'.join(missing)}）" if missing else ""))
        if "query" in t:
            payload, data = t["query"]
            sent = {k: v for k, v in payload.items() if v not in (None, False, [], "") and k != "exclude"}
            lines.append(f"**③ 查詢**：{json.dumps(sent, ensure_ascii=False)}"
                         f"｜排除 {len(payload.get('exclude') or [])} 家（推薦過、不喜歡）"
                         f"｜符合 {data.get('matched', 0)} 家，顯示 {len(data.get('restaurants') or [])} 家")
        if t.get("llm_raw") is not None:
            lines.append("**LLM 原始回傳**：\n```json\n" + json.dumps(t["llm_raw"], ensure_ascii=False) + "\n```")
        return f"\n\n{DEBUG_MARK}\n\n" + "\n\n".join(lines) + "\n\n</details>"
