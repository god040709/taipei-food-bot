"""
台北美食推薦機器人的對話流程：理解需求 → 反問／排除／換一批 → 查餐廳 → 組合回覆。
api.py 的 /chat 會呼叫這裡；Open WebUI 的 pipe 只負責把對話轉過來。
"""
import hashlib
import json
import random
import re
from typing import Optional

import service
from lexicon import (  # 詞庫、提示詞、文字處理
    ASK_HINT_WORDS, ASK_MARK, BUDGET_FILLER, CATEGORIES, CATEGORY_QUESTION_RE, CATEGORY_SYNONYMS,
    CHAT_MARK, CHAT_PROMPT, CHOOSE_MARK, CN_INDEX, DISTRICTS, FIELDS, FIELD_SKIP_WORDS, LABELS,
    MORE_WORDS, MRT_STATIONS, NEAR_ASK_MARK, NEG_AFTER_RE, NEW_REQUEST_MARK, NEG_BEFORE_RE, NUM, PARSE_PROMPT, PERSON_PREFS,
    PREFS_MARK, PREF_OFF_WORDS, QUESTIONS, REASON_PROMPT, REJECTED_MARK, REJECT_SOFT, REJECT_STRONG,
    RELAX_BUDGET, RELAX_LABELS, RESET_MARK, RESET_WORDS, SAY_PROMPT, SKIP_WORDS, SORTS, SORT_LABELS, STYLE_WORDS, SUBTYPES,
    TOTAL_RE, UNSURE_WORDS, find_near, fix_homophones, to_number,
)
from llm import LLMError

def strip_marks(text: str) -> str:
    """去掉 LLM 模仿對話紀錄而加在開頭的符號（💬 🤔），避免越疊越多。"""
    return re.sub(r"^(?:\s*(?:💬|🤔)\s*)+", "", text or "").strip()


def as_list(value) -> list:
    """LLM 該回清單的欄位，有時會回一個字串（「停車」）：當成只有一項，不要拆成一個一個字。"""
    if isinstance(value, list):
        return value
    return [value] if isinstance(value, str) and value.strip() else []


class Options:
    """機器人的設定（從 Open WebUI 函式的 Valves 傳過來，沒給就用預設值）。"""
    USE_LLM_PARSER = True   # 用 LLM 理解使用者需求
    USE_LLM_REASON = True   # 用 LLM 寫推薦理由
    SHOW_CONDITIONS = True  # 在回答開頭顯示解析出的條件
    ASK_FOLLOWUP = True     # 資訊不足時持續反問，直到地區、類型、人數、預算都清楚
    MAX_ASK = 4             # 同一個需求最多反問幾次（避免一直問不完）
    SHOW_REASONS = True     # 在每家餐廳附上推薦理由

    def __init__(self, values: Optional[dict] = None):
        for key, value in (values or {}).items():
            if hasattr(Options, key) and isinstance(value, type(getattr(Options, key))):
                setattr(self, key, value)


class Bot:
    """對話機器人：收到整段對話，回傳下一則回覆。
    llm：有 enabled 屬性和 chat(messages, max_tokens, schema) 方法的物件（見 llm.py）。"""

    def __init__(self, llm, options: Optional[dict] = None):
        self.llm = llm
        self.valves = Options(options)

    # ------------------------------------------------------------------
    # LLM 呼叫
    # ------------------------------------------------------------------
    def _llm_enabled(self) -> bool:
        return bool(getattr(self.llm, "enabled", False))

    def _chat(self, messages: list, max_tokens: int = 400, schema: Optional[dict] = None,
              temperature: float = 0) -> str:
        return self.llm.chat(messages, max_tokens, schema, temperature)

    @staticmethod
    def _extract_json(raw: str) -> dict:
        """從 LLM 回應裡找出 JSON 物件（可能包在 ```json 裡、前後有說明文字）。"""
        decoder = json.JSONDecoder()
        found = None
        i = 0
        while i < len(raw):
            if raw[i] != "{":
                i += 1
                continue
            try:
                obj, end = decoder.raw_decode(raw, i)
            except ValueError:
                i += 1
                continue
            if isinstance(obj, dict):
                found = obj  # 取最後一個完整的物件（前面可能是思考或範例）
            i = end  # 跳過整個物件，不要把裡面的 {"near": …} 當成另一個物件
        if found is None:
            raise ValueError(f"LLM 沒有回傳 JSON：{raw[:60]}")
        return found

    # ------------------------------------------------------------------
    # 條件解析：LLM 優先，失敗時退回規則
    # ------------------------------------------------------------------
    def parse_user_query(self, text: str):
        """規則解析（不需要 LLM），支援常見口語說法。"""
        district = None
        category = None
        budget = None
        party_size = None
        text = fix_homophones(text)[0]  # 同音錯字：「大按」→「大安」

        # 行政區：「大安區」或省略「區」的「大安」
        for d in DISTRICTS:
            if d in text or d.replace("區", "") in text:
                district = d
                break

        # 類別：先找標準名稱，再找同義詞（麻辣鍋→火鍋、韓國菜→韓式料理…）；
        # 「不要火鍋」「剛吃過拉麵」這種說不要的不算
        wanted = [c for _, c, neg in self.category_mentions(text) if not neg]
        category = next((c for c in CATEGORIES if c in wanted and c in text), None) \
            or (wanted[0] if wanted else None)

        # 人數：「3人」「3位」「三個人」「我們6個人」「一個人」
        party_match = re.search(NUM + r"\s*個?\s*(?:人|位)", text)
        if party_match:
            party_size = to_number(party_match.group(1))

        # 預算：「預算1000」「預算一千」「1000元」「500以內」「一人500」「每人一千」「1k」
        filler = r"\s*(?:大概|大約|約|差不多|抓|在|是|最多|頂多|上限|不超過|：|:)*\s*"
        budget_patterns = [
            r"預算" + filler + NUM,
            r"(?:一人|每人|每位|一位)" + filler + NUM,
            NUM + r"\s*(?:元|塊|以內|以下)",
            NUM + r"\s*(?:元|塊)?\s*(?:左右|上下)",
            r"(\d+(?:\.\d+)?\s*[kK])",
        ]
        for pattern in budget_patterns:
            m = re.search(pattern, text)
            if m:
                budget = to_number(m.group(1))
                break
        if budget is None:
            # 單獨的數字（「大安 火鍋 2人 800」的 800）：前後是空白或標點才算，「台北101」的 101 不算
            m = re.search(r"(?:^|[\s,，、。:：])(\d{3,5})(?=$|[\s,，、。!！?？~～])", text)
            if m and int(m.group(1)) >= 100:
                budget = int(m.group(1))

        return district, category, party_size, budget

    def parse_with_llm(self, messages: list):
        """用 LLM 解析整段對話，回傳 (district, category, party_size, budget)。"""
        d = self.parse_with_llm_full(messages, self.last_recommended(messages))
        return d["district"], d["category"], d["party_size"], d["budget"]

    @staticmethod
    def parse_schema(recent_names: list) -> dict:
        """LLM 回傳的 JSON 格式（伺服器支援時會保證照這個格式）。"""
        def nullable(kind, enum=None):
            schema = {"type": [kind, "null"]}
            if enum is not None:
                schema["enum"] = list(enum) + [None]
            return schema

        def array(enum=None):
            return {"type": "array", "items": {"type": "string", **({"enum": list(enum)} if enum else {})}}

        props = {
            "intent": {"type": "string",
                       "enum": ["search", "new", "more", "reject", "chat", "categories", "reset"]},
            "district": nullable("string", DISTRICTS),
            "category": nullable("string", CATEGORIES),
            "party_size": nullable("integer"),
            "budget": nullable("integer"),
            "waived": array(FIELDS),
            "choose_category": nullable("boolean"),
            "reject": array(recent_names or None),
            "keep": array(recent_names or None),
            "outside_area": nullable("string"),
            "unknown_food": nullable("string"),
            "near": nullable("string"),
            "no_chain": nullable("boolean"),
            "hidden_gem": nullable("boolean"),
            "budget_min": nullable("integer"),
            "sort": nullable("string", SORTS),
            "relax": array(RELAX_LABELS),
            "avoid_categories": array(CATEGORIES),
            "categories": array(CATEGORIES),
            "unsupported": array(),
        }
        return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}

    def parse_with_llm_full(self, messages: list, recent_names: list) -> dict:
        """用 LLM 解析條件、意圖、不喜歡的店、其他條件。"""
        # 把對話整理成「一段文字」交給 LLM，而不是原樣當成多輪對話：
        # 否則模型會模仿助理先前的回答（「🔎 搜尋條件：…」），不回 JSON。
        lines = []
        for m in messages[-12:]:
            content = m.get("content", "")
            role = m.get("role")
            if not isinstance(content, str) or role not in ("user", "assistant"):
                continue
            if role == "user":
                lines.append(f"使用者：{fix_homophones(content[:300])[0]}")
            elif content.startswith(ASK_MARK):
                lines.append(f"助理（反問）：{content[:200]}")
            elif content.startswith(CHAT_MARK):
                lines.append(f"助理（閒聊）：{content[len(CHAT_MARK):][:150]}")
            else:
                # 推薦結果只留先前的條件，其餘省略
                cond = re.search(r"🔎 搜尋條件：([^（\n]+)", content)
                extra = next((l[len(PREFS_MARK):] for l in content.splitlines() if l.startswith(PREFS_MARK)), "")
                picked = "；料理類型是你幫使用者挑的" if "🎲 幫你挑了" in content else ""
                lines.append(f"助理：（推薦了餐廳；當時的條件是 {cond.group(1).strip() if cond else '未知'}"
                             f"{'；其他條件：' + extra if extra else ''}{picked}）")

        system = PARSE_PROMPT
        if recent_names:
            system += "\n最近推薦過的店：\n" + "\n".join(f"{i}. {n}" for i, n in enumerate(recent_names, 1))
        user = "以下是對話紀錄，最後一行是使用者這次說的話：\n\n" + "\n".join(lines) + "\n\n請輸出 JSON。"
        raw = self._chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                         max_tokens=800, schema=self.parse_schema(recent_names))
        data = self._extract_json(raw)

        def to_int(v):
            try:
                v = int(float(v))
                return v if v > 0 else None
            except (TypeError, ValueError):
                return None

        district = data.get("district") if data.get("district") in DISTRICTS else None
        category = data.get("category") if data.get("category") in CATEGORIES else None
        intents = ("search", "new", "more", "reject", "chat", "categories", "reset")
        intent = data.get("intent") if data.get("intent") in intents else "search"
        return {"district": district, "category": category,
                "party_size": to_int(data.get("party_size")), "budget": to_int(data.get("budget")),
                "intent": intent, "reject": [n for n in as_list(data.get("reject")) if n in recent_names],
                "waived": {f for f in as_list(data.get("waived")) if f in FIELDS},
                "keep": [n for n in as_list(data.get("keep")) if n in recent_names],
                "outside_area": str(data["outside_area"]).strip()[:10]
                if isinstance(data.get("outside_area"), str) and data["outside_area"].strip() else None,
                "unknown_food": str(data["unknown_food"]).strip()[:10]
                if isinstance(data.get("unknown_food"), str) and data["unknown_food"].strip() else None,
                "choose_category": data.get("choose_category") is True,
                "prefs": self.clean_llm_prefs(data, to_int)}

    @staticmethod
    def clean_llm_prefs(data: dict, to_int) -> dict:
        """檢查 LLM 回傳的其他條件，不合規定的丟掉（例如不存在的車站、類別）。"""
        p = {}
        near = data.get("near")
        if isinstance(near, str) and near.strip():
            found = find_near(near.strip()) or find_near(near.strip() + "站")
            if found:
                p["near"], p["near_label"] = found
        for k in ("no_chain", "hidden_gem"):
            if isinstance(data.get(k), bool):
                p[k] = data[k]
        if to_int(data.get("budget_min")):
            p["budget_min"] = to_int(data.get("budget_min"))
        if data.get("sort") in SORTS:
            p["sort"] = data["sort"]
        relax = [r for r in RELAX_LABELS if r in as_list(data.get("relax"))]
        if relax:
            p["relax"] = relax
        avoid = [c for c in as_list(data.get("avoid_categories")) if c in CATEGORIES]
        if avoid:
            p["avoid"] = list(dict.fromkeys(avoid))
        cats = list(dict.fromkeys(c for c in as_list(data.get("categories")) if c in CATEGORIES))
        if len(cats) >= 2:
            p["cats"] = cats
        unsupported = [str(u).strip()[:10] for u in as_list(data.get("unsupported")) if str(u).strip()]
        if unsupported:
            p["unsupported"] = unsupported[:4]
        return p

    @staticmethod
    def current_round(messages: list):
        """這次需求的對話：從上一次「推薦」之後開始算（中間的反問不算結束）。
        回傳 (使用者訊息 list, 已經反問幾次)。"""
        texts, asks = [], 0
        for m in reversed(messages):
            content = m.get("content", "")
            if not isinstance(content, str):
                continue
            if m.get("role") == "user":
                texts.append(content)
            elif m.get("role") == "assistant":
                if content.startswith(CHAT_MARK):
                    continue  # 閒聊回覆：跳過
                if not content.startswith(ASK_MARK):
                    break
                asks += 1
        texts.reverse()
        return texts, asks

    @staticmethod
    def previous_conditions(messages: list):
        """上一次推薦用的條件（從回答第一行「🔎 搜尋條件：…」讀回來），沒有就回傳 None。
        讓使用者推薦完之後可以說「不限地區」「預算 500」「換成燒肉」來修改搜尋。"""
        for m in reversed(messages[:-1]):
            content = m.get("content", "")
            if m.get("role") != "assistant" or not isinstance(content, str):
                continue
            if content.startswith((ASK_MARK, CHAT_MARK)):
                continue
            match = re.search(r"🔎 搜尋條件：(.+?)（", content)
            if not match:
                return None
            parts = match.group(1).split("｜")
            district = next((p for p in parts if p in DISTRICTS), None)
            category = next((p.split("・")[0] for p in parts if p.split("・")[0] in CATEGORIES), None)
            party = re.search(r"(\d+) 人", match.group(1))
            budget = re.search(r"每人 (\d+) 元內", match.group(1))
            return (district, category,
                    int(party.group(1)) if party else None,
                    int(budget.group(1)) if budget else None)
        return None

    @staticmethod
    def previous_waived(messages: list) -> set:
        """上一次推薦時「不限」的欄位（條件行寫「不限地區」「不限類別」「人數不限」「預算不限」）。
        使用者已經回答過了，推薦完之後的追問、修改都不用再問。"""
        for m in reversed(messages[:-1]):
            content = m.get("content", "")
            if m.get("role") != "assistant" or not isinstance(content, str) \
                    or content.startswith((ASK_MARK, CHAT_MARK)):
                continue
            cond = re.search(r"🔎 搜尋條件：(.+?)（", content)
            if not cond:
                return set()
            parts = cond.group(1).split("｜")
            labels = {"district": "不限地區", "category": "不限類別", "party_size": "人數不限", "budget": "預算不限"}
            return {f for f, label in labels.items() if label in parts}
        return set()

    def parse_rounds(self, messages: list):
        """規則解析多輪：機器人反問後，把使用者這幾次的回答合併起來。"""
        texts, _ = self.current_round(messages)
        merged = list(self.previous_conditions(messages) or [None, None, None, None])
        for i, text in enumerate(texts):
            parsed = list(self.parse_user_query(text))
            # 回答反問時，單獨一個數字（例如「800」）視為每人預算
            if i > 0 and parsed[3] is None:
                for m in re.finditer(NUM + r"(?!\s*個?\s*(?:人|位))", text):
                    n = to_number(m.group(1))
                    if n and n >= 50:
                        parsed[3] = n
                        break
            merged = [new if new is not None else old for old, new in zip(merged, parsed)]
        return tuple(merged)

    @staticmethod
    def previous_subtype(messages: list) -> Optional[str]:
        """上一次推薦的子類型（「🔎 搜尋條件：…｜火鍋・壽喜燒｜…」），沒有就回傳 None。"""
        for m in reversed(messages[:-1]):
            content = m.get("content", "")
            if m.get("role") != "assistant" or not isinstance(content, str) \
                    or content.startswith((ASK_MARK, CHAT_MARK)):
                continue
            match = re.search(r"🔎 搜尋條件：[^（]*?・([^｜（]+)", content)
            return match.group(1) if match and match.group(1) in SUBTYPES else None
        return None

    @staticmethod
    def detect_subtype(text: str, category: Optional[str]) -> Optional[str]:
        """從一句話找出子類型。同時符合多個時，優先選屬於 category 的，再選說法比較長的。"""
        lower = text.lower()
        found = []
        for name, (parent, triggers, _, _) in SUBTYPES.items():
            hit = max((len(t) for t in triggers if t.lower() in lower), default=0)
            if hit:
                found.append((parent == category, hit, name))
        return max(found)[2] if found else None

    @staticmethod
    def total_budget(texts: list) -> Optional[int]:
        """這次需求裡提到的「總預算」（不是每人），沒有就回傳 None。"""
        for text in reversed(texts):
            m = re.search(TOTAL_RE + BUDGET_FILLER + NUM, text)
            if m:
                return to_number(m.group(1))
        return None

    # ------------------------------------------------------------------
    # 其他條件：捷運站附近、不要連鎖、隱藏版、預算下限、不要的類別、多類別、類似某間
    # ------------------------------------------------------------------
    @staticmethod
    def category_mentions(text: str) -> list:
        """這句話提到的類別：[(位置, 類別, 是不是「不要」)]，依出現順序。
        子類型的說法（「韓式烤肉」）算它的大類別，而且優先於比較短的詞（不會變成「韓式」＋「烤肉」）。"""
        lower = text.lower()
        words = [(c, c, False) for c in CATEGORIES]
        words += [(w, c, False) for c, ws in CATEGORY_SYNONYMS.items() for w in ws]
        words += [(t, parent, True) for parent, triggers, _, _ in SUBTYPES.values() for t in triggers]
        taken = [False] * len(text)
        found = []
        for word, cat, is_sub in sorted(words, key=lambda w: -len(w[0])):
            w = word.lower()
            start = lower.find(w)
            while start != -1:
                end = start + len(w)
                rest = lower[end:].lstrip("的 ")
                modifier = word in STYLE_WORDS and any(
                    rest.startswith(x.lower()) for x in CATEGORIES + [s for ws in CATEGORY_SYNONYMS.values() for s in ws])
                if modifier:
                    pass  # 「日式燒肉」的「日式」只是形容「燒肉」
                elif not any(taken[start:end]):
                    for k in range(start, end):
                        taken[k] = True
                    neg = (re.search(NEG_BEFORE_RE, text[max(0, start - 6):start]) is not None
                           or re.match(NEG_AFTER_RE, text[end:end + 5]) is not None)
                    found.append((start, cat, neg and not is_sub))  # 「不要麻辣鍋」不代表整個火鍋類都不要
                start = lower.find(w, end)
        return sorted(found)

    def parse_prefs(self, text: str) -> dict:
        """規則抓得到的其他條件：車站／地標、預算區間，以及用類別詞庫就判斷得出的「不要的類別」
        「多類別」（LLM 掛掉時的備案）。不要連鎖、隱藏版、資料沒有的條件交給 LLM 判斷。"""
        t = fix_homophones(text)[0]
        p = {}
        near = find_near(t)
        if near:
            p["near"], p["near_label"] = near
        # 預算區間「500 到 1000」→ 下限 500、上限 1000；「1000 以上」→ 只有下限
        m = re.search(NUM + r"\s*(?:元|塊)?\s*(?:到|至|~|～|-|－)\s*" + NUM + r"\s*(?:元|塊)?(?!\s*個?\s*(?:人|位))", t)
        lo = hi = None
        if m:
            lo, hi = to_number(m.group(1)), to_number(m.group(2))
        if lo and hi and 50 <= lo < hi:
            p["budget_min"], p["budget_max"] = lo, hi
        else:
            m = re.search(NUM + r"\s*(?:元|塊)?\s*(?:以上|起跳)", t)
            if m and (to_number(m.group(1)) or 0) >= 50:
                p["budget_min"] = to_number(m.group(1))
        mentions = self.category_mentions(t)
        avoid = list(dict.fromkeys(c for _, c, neg in mentions if neg))
        want = list(dict.fromkeys(c for _, c, neg in mentions if not neg))
        if avoid:
            p["avoid"] = avoid  # 「剛吃過火鍋」「火鍋以外」
        if len(want) >= 2:
            p["cats"] = want  # 「日式或韓式都可以」
        return p

    @staticmethod
    def previous_prefs(messages: list) -> dict:
        """上一次推薦用的其他條件（從條件行和「🎛️ 其他條件：…」讀回來）。"""
        for m in reversed(messages[:-1]):
            content = m.get("content", "")
            if m.get("role") != "assistant" or not isinstance(content, str) \
                    or content.startswith((ASK_MARK, CHAT_MARK)):
                continue
            cond = re.search(r"🔎 搜尋條件：(.+?)（", content)
            if not cond:
                return {}
            p = {}
            near = re.search(r"捷運([^｜]+?)站附近", cond.group(1))
            if near and near.group(1) in MRT_STATIONS:
                label = re.search(r"([^｜]+?)・捷運", cond.group(1))
                p["near"], p["near_label"] = near.group(1), (label.group(1) if label else near.group(1))
            line = next((l for l in content.splitlines() if l.startswith(PREFS_MARK)), "")
            for part in line[len(PREFS_MARK):].split("｜") if line else []:
                if part == "不要連鎖":
                    p["no_chain"] = True
                elif part.startswith("隱藏版"):
                    p["hidden_gem"] = True
                elif re.match(r"每人 (\d+) 元以上", part):
                    p["budget_min"] = int(re.match(r"每人 (\d+) 元以上", part).group(1))
                elif part.startswith("不要："):
                    p["avoid"] = [c for c in part[3:].split("、") if c in CATEGORIES]
                elif part.startswith("類型："):
                    p["cats"] = [c for c in part[3:].split("、") if c in CATEGORIES]
                elif part in SORT_LABELS.values():
                    p["sort"] = next(k for k, v in SORT_LABELS.items() if v == part)
                elif part in RELAX_LABELS.values():
                    p.setdefault("relax", []).append(next(k for k, v in RELAX_LABELS.items() if v == part))
            return p
        return {}

    def merge_prefs(self, messages: list, texts: list, llm_prefs: Optional[dict] = None,
                    new_request: bool = False) -> dict:
        """上一次的其他條件 + 這次需求的修改。
        new_request：使用者重新提出全新的需求 → 只保留「跟著人走」的偏好（不要連鎖）。"""
        prefs = self.previous_prefs(messages)
        if new_request:
            prefs = {k: v for k, v in prefs.items() if k in PERSON_PREFS}
        llm_prefs = llm_prefs or {}
        rule_near = False
        for text in texts:
            p = self.parse_prefs(text)
            if "near" in p:
                prefs["near"], prefs["near_label"] = p["near"], p["near_label"]
                rule_near = True
            elif self.parse_user_query(text)[0]:  # 改說行政區 → 不再是「某站附近」
                prefs.pop("near", None)
                prefs.pop("near_label", None)
            if "budget_min" in p:
                prefs["budget_min"] = p["budget_min"]
            if p.get("cats"):
                prefs["cats"] = p["cats"]
            if p.get("avoid"):
                prefs["avoid"] = list(dict.fromkeys(prefs.get("avoid", []) + p["avoid"]))
        # LLM 看最後一句話判斷的條件
        if llm_prefs.get("near") and not rule_near:
            prefs["near"], prefs["near_label"] = llm_prefs["near"], llm_prefs["near_label"]
        last_text = texts[-1] if texts else ""
        for k in ("no_chain", "hidden_gem"):
            if llm_prefs.get(k) is True:
                prefs[k] = True
            elif llm_prefs.get(k) is False and any(w in last_text for w in PREF_OFF_WORDS[k]):
                prefs[k] = False  # 「連鎖也可以」「有名的也可以」（LLM 常把沒提到的填成 false）
        if llm_prefs.get("budget_min") and not prefs.get("budget_min"):
            prefs["budget_min"] = llm_prefs["budget_min"]
        if llm_prefs.get("sort"):
            prefs["sort"] = llm_prefs["sort"]  # 強調的重點只有一個，以最新的說法為準
        if llm_prefs.get("relax"):
            prefs["relax"] = [r for r in RELAX_LABELS if r in prefs.get("relax", []) + llm_prefs["relax"]]
        if llm_prefs.get("sort") == "price_low":
            prefs["relax"] = [r for r in prefs.get("relax", []) if r != "budget"]  # 改說越便宜越好
        if not prefs.get("near"):
            prefs["relax"] = [r for r in prefs.get("relax", []) if r != "distance"]  # 沒有車站，談不上距離
        if llm_prefs.get("cats"):
            prefs["cats"] = llm_prefs["cats"]
        elif any(self.parse_user_query(t)[1] for t in texts) and \
                not any(self.parse_prefs(t).get("cats") for t in texts):
            prefs.pop("cats", None)  # 改說單一類別 → 不再是「日式或韓式」
        if llm_prefs.get("avoid"):
            prefs["avoid"] = list(dict.fromkeys(prefs.get("avoid", []) + llm_prefs["avoid"]))
        waived = self.waived_fields(texts)
        if "district" in waived:
            prefs.pop("near", None)
            prefs.pop("near_label", None)
        if "budget" in waived:
            prefs.pop("budget_min", None)
        if "category" in waived:
            prefs.pop("cats", None)
        # 只對這一句有效的：預算上限（區間）、資料沒有的條件
        last = self.parse_prefs(texts[-1]) if texts else {}
        if last.get("budget_max"):
            prefs["budget_max"] = last["budget_max"]
        if llm_prefs.get("unsupported"):
            prefs["unsupported"] = llm_prefs["unsupported"]
        return {k: v for k, v in prefs.items() if v not in (None, False, [], "")}

    def is_new_request(self, messages: list, texts: list, intent: Optional[str]) -> bool:
        """推薦完之後，使用者重新提出一個全新的需求（不是修改）。
        LLM 判斷 intent 是 new，或這句話一次講了 3 項以上的基本條件。"""
        if len(texts) != 1 or self.previous_conditions(messages) is None:
            return False  # 還在反問中，或還沒推薦過
        if intent == "new":
            return True
        district, category, party_size, budget = self.parse_user_query(texts[0])
        where = district or self.parse_prefs(texts[0]).get("near")
        what = category or self.detect_subtype(fix_homophones(texts[0])[0], None)
        return sum(bool(x) for x in (where, what, party_size, budget)) >= 3

    @staticmethod
    def prefs_line(prefs: dict) -> str:
        """「🎛️ 其他條件：不要連鎖｜隱藏版｜…」（下一句會讀回來沿用）。"""
        parts = []
        if prefs.get("cats"):
            parts.append("類型：" + "、".join(prefs["cats"]))
        if prefs.get("no_chain"):
            parts.append("不要連鎖")
        if prefs.get("hidden_gem"):
            parts.append("隱藏版（評分高、評論數較少）")
        if prefs.get("budget_min"):
            parts.append(f"每人 {prefs['budget_min']} 元以上")
        if prefs.get("avoid"):
            parts.append("不要：" + "、".join(prefs["avoid"]))
        if prefs.get("sort"):
            parts.append(SORT_LABELS[prefs["sort"]])
        parts += [RELAX_LABELS[r] for r in prefs.get("relax", [])]
        return PREFS_MARK + "｜".join(parts) if parts else ""

    def get_conditions(self, messages: list):
        """回傳 (條件 tuple, 解析方式, info)。
        info：total（總預算）、subtype（子類型）、intent / reject（LLM 判斷的意圖與不喜歡的店）、
              prefs（其他條件）、fixes（同音錯字修正）、llm_note（LLM 沒用上的原因，顯示在條件行）。
        有 LLM 時以 LLM 為主；LLM 沒填的欄位用規則補上。總預算會除以人數，換算成每人預算。"""
        source = "規則"
        info = {"intent": None, "reject": [], "llm_note": "", "llm_prefs": {}, "llm_waived": set(),
                "choose_category": False, "keep": [], "outside_area": None, "unknown_food": None}
        texts, _ = self.current_round(messages)
        rules = list(self.parse_rounds(messages))
        conditions = rules
        waived = self.waived_fields(texts)
        if self.valves.USE_LLM_PARSER and not self._llm_enabled():
            info["llm_note"] = "未設定 LLM 金鑰"
        elif self.valves.USE_LLM_PARSER:
            try:
                d = self.parse_with_llm_full(messages, self.last_recommended(messages))
                llm = [d["district"], d["category"], d["party_size"], d["budget"]]
                # LLM 沒填的欄位用規則補（例如 LLM 漏掉沿用的條件）。
                # 「不限」只看使用者有沒有說「地區都可以」「預算不限」這類話（LLM 常把沒說不限的欄位也標成不限）
                conditions = [lv if lv is not None else rv for lv, rv in zip(llm, rules)]
                # 使用者強調重點或說可以放寬時，LLM 常順手把預算 ×1.3 或 ×0.7 → 預算用規則讀到的
                if (d["prefs"].get("sort") or d["prefs"].get("relax")) and rules[3]:
                    conditions[3] = rules[3]
                # LLM 說「不限」的欄位，只有在那一項本來就不知道時才採用（避免把已知的條件清掉）
                unsure = any(w in (texts[-1] if texts else "") for w in UNSURE_WORDS)
                info["llm_waived"] = set() if unsure else                     {f for f, v in zip(FIELDS, conditions) if f in d["waived"] and v is None}
                info["choose_category"] = d["choose_category"]
                info.update(keep=d["keep"], outside_area=d["outside_area"], unknown_food=d["unknown_food"])
                source = "LLM"
                info.update(intent=d["intent"], reject=d["reject"], llm_prefs=d["prefs"])
            except LLMError as e:
                info["llm_note"] = e.note
            except Exception as e:
                # 附上一小段原因（LLM 實際回了什麼），方便除錯
                detail = re.sub(r"[\s（）()｜]+", " ", str(e)).strip()[:50]
                info["llm_note"] = f"LLM 回傳格式不對：{detail}" if detail else "LLM 回傳格式不對"
                print(f"[taipei_food_bot] LLM parse failed: {e!r}")
        conditions = [None if f in waived else v for f, v in zip(FIELDS, conditions)]

        # 全新的需求：這句話沒提到、只是從上一次沿用下來的條件不要用，缺的再問
        new_request = self.is_new_request(messages, texts, info["intent"])
        previous = self.previous_conditions(messages)
        if new_request and previous:
            said = self.parse_user_query(texts[-1])
            conditions = [None if (now is None and v == old) else v
                          for v, now, old in zip(conditions, said, previous)]
        info["new_request"] = new_request

        # 「你幫我挑一種料理」：這一句說的，或這次需求的反問裡已經記下來的
        asked_choose = False
        for m in reversed(messages[:-1]):  # 往回看到上一次推薦為止（中間的反問、閒聊）
            content = str(m.get("content", ""))
            if m.get("role") != "assistant" or content.startswith(CHAT_MARK):
                continue
            if not content.startswith(ASK_MARK):
                break
            if CHOOSE_MARK in content:
                asked_choose = True
                break
        said_category = any(self.parse_user_query(t)[1] for t in texts)
        if (info["choose_category"] or asked_choose) and not said_category:
            info["choose_category"] = True
            conditions[1] = None  # 不用 LLM 自己挑的類別，等條件齊了由程式隨機挑
        else:
            info["choose_category"] = False

        fixed_texts = [fix_homophones(t)[0] for t in texts]
        info["fixes"] = fix_homophones(texts[-1])[1] if texts else []

        # 總預算：有人數就換算成每人；還不知道人數就先不填，等問完人數再算
        total = self.total_budget(fixed_texts)
        district, category, party_size, budget = conditions
        if total:
            budget = total // party_size if party_size else None

        # 子類型（例如「壽喜燒」）：這次需求有提到就用；沒提到就沿用上一次的（類別沒換的話）
        subtype = None
        said_category = False  # 較新的一句已經明確說了類別、沒說子類型（例如回答「一般火鍋」）
        for text in reversed(fixed_texts):
            subtype = self.detect_subtype(text, category)
            if subtype:
                break
            if any(not neg for _, _, neg in self.category_mentions(text)):
                said_category = True
                break
        if subtype is None and not said_category and not new_request:
            prev = self.previous_subtype(messages)
            if prev and SUBTYPES[prev][0] == category:
                subtype = prev
        if subtype:
            parent = SUBTYPES[subtype][0]
            if category in (None, parent):
                category = parent  # 「想吃壽喜燒」→ 類別是火鍋
            elif any(category in t for t in fixed_texts[-1:]):
                subtype = None  # 這次明確說了別的類別（例如「換成拉麵」）
            else:
                category = parent

        # 其他條件（捷運站附近、不要連鎖、隱藏版、預算下限、不要的類別、多類別）
        prefs = self.merge_prefs(messages, texts, info["llm_prefs"], new_request)
        if new_request:
            old = {k: v for k, v in self.previous_prefs(messages).items() if k not in PERSON_PREFS}
            info["dropped_prefs"] = self.prefs_line(old)[len(PREFS_MARK):].replace("｜", "、")
        if prefs.get("budget_max"):
            budget = prefs["budget_max"]  # 「500 到 1000」→ 上限 1000
        if prefs.get("budget_min") and budget and budget <= prefs["budget_min"]:
            budget = None  # 「1000 以上」：1000 是下限，不是上限
        if prefs.get("cats"):
            category = prefs["cats"][0]
        if category in prefs.get("avoid", []):
            # 這次明確說想吃這個類別（例如「吃羊肉爐好了」）→ 不排除它；
            # 否則是「不要」目前的類別（例如推薦完火鍋後說「剛吃過火鍋」）→ 類別清空
            prev_category = previous[1] if previous else None
            asked = {c for t in fixed_texts for _, c, neg in self.category_mentions(t) if not neg}
            if source == "LLM" and conditions[1] != prev_category:
                asked.add(conditions[1])
            if category in asked:
                prefs["avoid"] = [c for c in prefs["avoid"] if c != category]
                if not prefs["avoid"]:
                    prefs.pop("avoid")
            else:
                category, subtype = None, None
        conditions = (district, category, party_size, budget)
        info.update(total=total, subtype=subtype, prefs=prefs)
        return conditions, source, info

    # ------------------------------------------------------------------
    # 資訊不足時持續反問
    # ------------------------------------------------------------------
    @staticmethod
    def waived_fields(texts: list) -> set:
        """使用者說過「不限」的欄位，例如「地區都可以」→ {"district"}。"""
        waived = set()
        for text in texts:
            for field, words in FIELD_SKIP_WORDS.items():
                if any(w in text for w in words):
                    waived.add(field)
        return waived

    @staticmethod
    def skip_all(text: str) -> bool:
        """使用者這次說「都可以」「隨便」→ 剩下的條件都不限。"""
        for words in FIELD_SKIP_WORDS.values():
            for w in words:
                text = text.replace(w, "")  # 先拿掉「地區都可以」這類只針對單一條件的說法
        return any(w in text for w in SKIP_WORDS)

    @staticmethod
    def answered_by_prefs(prefs: Optional[dict]) -> set:
        """其他條件已經回答了的欄位：「忠孝復興附近」→ 地區；「不要火鍋」→ 類型；
        「1000 以上」「最便宜的」「最貴的」→ 預算（依價格排序時，預算不是重點）。"""
        prefs = prefs or {}
        done = set()
        if prefs.get("near"):
            done.add("district")
        if prefs.get("avoid") or prefs.get("cats"):
            done.add("category")
        if prefs.get("budget_min") or prefs.get("sort") in ("price_high", "price_low"):
            done.add("budget")
        return done

    def should_ask(self, messages: list, conditions, total=None, prefs=None, more_waived=None) -> list:
        """回傳還要問的欄位；空 list 代表資訊足夠，可以推薦。"""
        if not self.valves.ASK_FOLLOWUP:
            return []
        texts, asks = self.current_round(messages)
        if not texts or self.skip_all(texts[-1]):
            return []
        if asks >= self.valves.MAX_ASK:
            return []  # 已經問很多次了，用目前的條件推薦
        waived = self.waived_fields(texts) | self.answered_by_prefs(prefs) | set(more_waived or ())
        if total:
            waived.add("budget")  # 已經有總預算，問到人數就能換算
        return [f for f, v in zip(FIELDS, conditions) if v is None and f not in waived]

    def ask_question(self, messages: list, conditions, missing: list, total=None, subtype=None,
                     prefs=None, choose=False, note="") -> str:
        """缺條件時反問。choose：使用者要機器人挑料理類型（不問類型，並記在最後一行）。
        note：問之前要先說明的事（例如讀不到使用者的位置）。"""
        text = self._ask_question(messages, conditions, missing, total, subtype, prefs, choose, note)
        return text + ("\n\n" + CHOOSE_MARK if choose else "")

    @staticmethod
    def asked_in_round(messages: list, mark: str) -> bool:
        """這次需求的反問裡（上一次推薦之後），有沒有出現過 mark。"""
        for m in reversed(messages[:-1]):
            content = str(m.get("content", ""))
            if m.get("role") != "assistant" or content.startswith(CHAT_MARK):
                continue
            if not content.startswith(ASK_MARK):
                return False
            if mark in content:
                return True
        return False

    def _ask_question(self, messages: list, conditions, missing: list, total=None, subtype=None,
                      prefs=None, choose=False, note="") -> str:
        texts, asks = self.current_round(messages)
        waived = self.waived_fields(texts)
        prefs = prefs or {}
        known = []
        for field, value in zip(FIELDS, conditions):
            if field == "district" and prefs.get("near"):
                known.append(self.near_text(prefs))
            elif field == "budget" and total:
                known.append(f"總預算 {total} 元")
            elif field == "category" and value and subtype:
                known.append(f"{value}・{subtype}")
            elif value is not None:
                known.append({"party_size": f"{value} 人", "budget": f"每人 {value} 元內"}.get(field, value))
            elif field in waived:
                known.append(f"{LABELS[field]}不限")

        extra = self.prefs_line(prefs)[len(PREFS_MARK):]
        if self._llm_enabled():
            asked = [{"district": "地區（台北 12 區都可以，也可以說捷運站附近）", "category": "想吃的料理類型",
                      "party_size": "人數", "budget": "每人預算"}[f] for f in missing]
            unsure = bool(texts) and any(w in texts[-1] for w in UNSURE_WORDS)
            text = self.say(
                "還缺一些資訊才能推薦餐廳。請自然地問使用者「還需要問」的每一項，全部都要問到（可以合在一兩句裡），"
                "已經知道的不要再問。可以順便讓使用者知道，不在意的項目說不限就好。"
                + ("使用者剛剛說不知道，所以這次一定要針對還需要問的每一項，直接給 2~3 個具體的選項讓他挑"
                   "（例如地區給幾個熱門的區、人數給幾個常見的人數、預算給幾個常見的金額），不要只說可以給選項。"
                   if unsure else ""),
                [f"使用者剛剛說：「{texts[-1]}」" if texts else "",
                 note,
                 f"目前已知：{'、'.join(known + ([extra] if extra else [])) or '還沒有'}",
                 "料理類型使用者要你幫他挑，等條件齊了你會挑，這次不用問類型" if choose else "",
                 f"還需要問：{'；'.join(asked)}",
                 f"這是第 {asks + 1} 次追問，不要再打招呼" if asks else "",
                 f"台北的行政區：{'、'.join(DISTRICTS)}；料理類型：{'、'.join(CATEGORIES)}"])
            if text:
                # 確認每一項都有問到，LLM 漏掉的補一句
                lacking = [QUESTIONS[f] for f in missing if not any(w in text for w in ASK_HINT_WORDS[f])]
                return ASK_MARK + text + ("\n" + "\n".join(lacking) if lacking else "")

        if asks == 0:
            opening = "想幫你找到更適合的餐廳，可以再多告訴我一些嗎？"
        else:
            opening = "了解！還差一點點資訊，我就能幫你挑出最合適的餐廳："
        lines = [f"{ASK_MARK}{opening}"]
        if note:
            lines[0] = f"{ASK_MARK}我讀不到你的位置，沒辦法依距離排，先換個方式幫你找。{opening}"
        if extra:
            known.append(extra)
        if known:
            lines.append(f"目前知道：{'｜'.join(known)}")
        lines.append("")
        for i, field in enumerate(missing, 1):
            lines.append(f"{i}. {QUESTIONS[field]}")
        lines.append("")
        lines.append("直接回覆就可以，例如「信義區 3個人 每人800」；"
                     "某一項不在意可以說「地區都可以」，全部不在意就回「都可以」。")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 使用者問「有什麼類型可以選」→ 列出料理類型
    # ------------------------------------------------------------------
    @staticmethod
    def is_category_question(text: str) -> bool:
        if any(c in text for c in CATEGORIES):
            return False  # 已經說了想吃的類型，例如「有什麼火鍋推薦」
        if any(t.lower() in text.lower() for _, triggers, _, _ in SUBTYPES.values() for t in triggers):
            return False  # 說了子類型，例如「有什麼壽喜燒」
        if any(w in text for words in FIELD_SKIP_WORDS.values() for w in words):
            return False  # 「什麼都可以吃」是「不限」，不是在問
        return re.search(CATEGORY_QUESTION_RE, text) is not None

    def list_categories(self, conditions) -> str:
        district, _, party_size, budget = conditions
        counts = {}
        try:
            r = service.handle_categories({"district": district, "party_size": party_size, "budget": budget})
            counts = {c["category"]: c["count"] for c in r.get("categories", [])}
        except Exception:
            pass  # API 舊版或連不到：只列類型，不列家數

        where = district or "全台北"
        if budget:
            where += f"、每人 {budget} 元內"
        order = sorted(CATEGORIES, key=lambda c: -counts.get(c, 0)) if counts else CATEGORIES
        lines = [f"{ASK_MARK}這些料理類型都可以選（{where}）：", ""]
        if counts:
            lines += ["| 類型 | 店家數 | 也可以這樣說 |", "|---|---|---|"]
        else:
            lines += ["| 類型 | 也可以這樣說 |", "|---|---|"]
        for c in order:
            if counts and not counts.get(c):
                continue  # 這個條件下沒有店的類型就不列
            examples = "、".join(CATEGORY_SYNONYMS.get(c, [])[:3]) or "—"
            lines.append(f"| {c} | {counts[c]} 家 | {examples} |" if counts else f"| {c} | {examples} |")
        lines += ["", "想吃哪一種？直接回覆類型就可以，例如「燒肉」；"
                      "還沒決定的話，回「類型都可以」我就從各種料理裡挑。"]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 「推薦別的」→ 排除已經推薦過的店
    # ------------------------------------------------------------------
    @staticmethod
    def wants_more(text: str) -> bool:
        return any(w in text for w in MORE_WORDS)

    @staticmethod
    def shown_restaurants(messages: list, condition_line: str, extra_line: str = "") -> list:
        """同一組搜尋條件（含其他條件）下，之前已經推薦過的店名。"""
        names = []
        for m in messages[:-1]:
            content = m.get("content", "")
            if m.get("role") != "assistant" or not isinstance(content, str):
                continue
            if condition_line.split("（")[0] not in content:
                continue
            old_extra = next((l for l in content.splitlines() if l.startswith(PREFS_MARK)), "")
            if old_extra != extra_line:
                continue
            names += re.findall(r"^\*\*\d+\. (.+?)\*\*$", content, flags=re.MULTILINE)
        return list(dict.fromkeys(names))  # 同一家可能出現在好幾次回覆裡，只算一次

    # ------------------------------------------------------------------
    # 「我不喜歡第 2 間」「輕鬆煮吃過了」→ 之後都不再推薦
    # ------------------------------------------------------------------
    @staticmethod
    def last_recommended(messages: list) -> list:
        """最近一次推薦的店名（依顯示順序）。"""
        for m in reversed(messages[:-1]):
            content = m.get("content", "")
            if m.get("role") != "assistant" or not isinstance(content, str) \
                    or content.startswith((ASK_MARK, CHAT_MARK)):
                continue
            return re.findall(r"^\*\*\d+\. (.+?)\*\*$", content, flags=re.MULTILINE)
        return []

    @staticmethod
    def rejected_history(messages: list) -> list:
        """這段對話裡使用者說過不喜歡／吃過的店（從「🚫 不再推薦：「A」「B」」讀回來）。
        提出全新的需求之後（回覆裡有「🆕 這是新的需求」），之前的就不算了。"""
        names = []
        for m in messages[:-1]:
            content = m.get("content", "")
            if m.get("role") != "assistant" or not isinstance(content, str):
                continue
            if NEW_REQUEST_MARK in content:
                names = []
            for line in content.splitlines():
                if line.startswith(REJECTED_MARK):
                    names += re.findall(r"「(.+?)」", line)
        return names

    @staticmethod
    def find_rejects(text: str, names: list, force: bool = False):
        """回傳 (使用者指定不要的店名, 是否在表達「不喜歡／吃過」)。
        force=True：剛問過「是哪一間呢？」，這次的回覆就算沒說「吃過」也是在指定店家。"""
        strong = any(w in text for w in REJECT_STRONG) or force
        soft = any(w in text.lower() for w in REJECT_SOFT)
        if not (strong or soft) or not names:
            return [], strong
        result = Bot.match_names(text, names)
        if not result and strong and re.search(r"都|全部|這些|這幾|以上", text):
            result = list(names)
        return result, (strong or bool(result))

    @staticmethod
    def match_names(text: str, names: list) -> list:
        """句子裡指到的店：「第 2 間」「3 號」「前兩間」「最後一間」「倒數第 2 間」、店名或店名開頭。"""
        picked = []
        if re.search(r"最後一?\s*(?:間|家|個)", text):
            picked.append(len(names))
        for m in re.finditer(r"倒數第\s*([一二兩三四五1-9])\s*(?:間|家|個|名)?", text):
            picked.append(len(names) - (CN_INDEX.get(m.group(1)) or int(m.group(1))) + 1)
        text = re.sub(r"倒數第\s*[一二兩三四五1-9]", "", text)  # 不要再被當成「第 2 間」
        for m in re.finditer(r"第\s*([一二兩三四五1-9])\s*(?:間|家|個|名)?", text):
            picked.append(CN_INDEX.get(m.group(1)) or int(m.group(1)))
        for m in re.finditer(r"(?<!\d)([1-9])\s*號", text):
            picked.append(int(m.group(1)))
        m = re.search(r"前\s*([兩二三四2-4])\s*(?:間|家|個|名)", text)
        if m:
            picked += list(range(1, (CN_INDEX.get(m.group(1)) or int(m.group(1))) + 1))
        result = [names[i - 1] for i in picked if 1 <= i <= len(names)]
        for name in names:  # 直接說店名，或店名的開頭（例如「肉老大頂級肉品…」說成「肉老大」）
            key = re.split(r"[\s｜|（(\-—－/／:：,，。]", name)[0]
            if name in text or (len(key) >= 2 and key in text):
                result.append(name)
                continue
            # 店名開頭至少 3 個字出現在句子裡（太短的像「燒肉」容易誤判，不算）
            if any(key[:k] in text for k in range(len(key), 2, -1)):
                result.append(name)
        return list(dict.fromkeys(result))  # 去重、保留順序

    def ask_which(self, names: list, user_text: str = "") -> str:
        """使用者說吃過／不喜歡，但沒說是哪一間（或說了不存在的第 6 間）→ 問是哪一間（店名清單由程式列出）。"""
        numbers = [CN_INDEX.get(n) or (int(n) if n.isdigit() else 0)
                   for n in re.findall(r"第\s*([一二兩三四五1-9]\d*)\s*(?:間|家|個|名)", user_text)]
        too_big = [n for n in numbers if n > len(names)]
        fact = f"剛剛只推薦了 {len(names)} 間，沒有第 {too_big[0]} 間" if too_big else ""
        lead = self.say("使用者說吃過或不喜歡推薦過的店，但沒說清楚是哪一間。請自然地問是哪一間，"
                        "告訴他可以說第幾間或店名，之後就不會再推薦。店名清單會另外列在下面。",
                        [f"使用者剛剛說：「{user_text}」", fact]) \
            or (f"{fact}，" if fact else "") + "是哪一間呢？可以直接說「第 2 間」或店名，我就不會再推薦它："
        return "\n".join([ASK_MARK + lead] + [f"{i}. {n}" for i, n in enumerate(names, 1)])

    def say(self, task: str, facts: list) -> Optional[str]:
        """請 LLM 用自然的話說出 task（facts 是它能用的資訊）。沒有 LLM 或失敗時回傳 None。"""
        if not self._llm_enabled():
            return None
        content = "要做的事：" + task + "\n資訊：\n" + "\n".join(f"- {f}" for f in facts if f)
        try:
            text = self._chat([{"role": "system", "content": SAY_PROMPT}, {"role": "user", "content": content}],
                              max_tokens=200, temperature=0.7)
        except Exception:
            return None
        return strip_marks(text) or None

    def chat_reply(self, messages: list) -> Optional[str]:
        """不是要新推薦的訊息（道謝、閒聊、問細節）→ 用 LLM 自然回覆。"""
        recent = ""
        for m in reversed(messages[:-1]):
            content = m.get("content", "")
            if m.get("role") == "assistant" and isinstance(content, str) \
                    and not content.startswith((ASK_MARK, CHAT_MARK)):
                recent = content[:2500]
                break
        history = [{"role": m["role"], "content": strip_marks(str(m.get("content", "")))[:500]}
                   for m in messages[-6:] if m.get("role") in ("user", "assistant")]
        system = (CHAT_PROMPT + "\n可以推薦的料理類型：" + "、".join(CATEGORIES)
                  + "\n最近的推薦內容：\n" + (recent or "（還沒有推薦過餐廳）"))
        try:
            return strip_marks(self._chat([{"role": "system", "content": system}] + history,
                                          max_tokens=300, temperature=0.7)) or None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 推薦理由（每家一段，寫餐廳特色）
    # ------------------------------------------------------------------
    def write_reasons(self, texts: list, conditions, restaurants: list, subtype=None, unsupported=None,
                      notes=None):
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
                f"人數={party_size or '不限'}，每人預算上限={budget or '不限'}")
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

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
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
        ]
        return f"🔎 搜尋條件：{'｜'.join(parts)}（{source}解析{'・' + note if note else ''}）"

    @staticmethod
    def price_text(r: dict) -> str:
        # 價格區間是「多數人用餐的花費」，不是最低／最高消費
        low, high = int(r["min_price"]), int(r["max_price"])
        text = f"多數人花費 ${high} 以下 / 人" if low <= 1 else f"多數人花費 ${low} - ${high} / 人"
        if r.get("budget_fit") == "partial":
            text += "（⚠️ 可能超出預算）"
        return text

    def format_card(self, i: int, r: dict, reason: str, anchor: str = "") -> str:
        lines = [
            f"**{i}. {r['name']}**",
            f"📍 {r['district']}・{r['category']}｜⭐ {r['rating']}"
            f"（{int(r['user_ratings_total'] or 0):,} 則評論）｜💰 {self.price_text(r)}",
        ]
        if anchor and r.get("distance") is not None:
            lines[1] += f"｜🚶 離{anchor}約 {int(r['distance']):,} 公尺"
        if reason and self.valves.SHOW_REASONS:
            lines.append(f"✨ {reason}")
        lines.append(f"🏠 {r['address']}｜🗺️ [在 Google Maps 開啟]({r['maps_url']})")
        return "\n".join(lines)

    def pipe(self, body: dict) -> str:
        messages = body.get("messages", [])
        if not messages:
            return "無有效的對話訊息。"

        user_text = messages[-1].get("content", "")
        user_text = user_text if isinstance(user_text, str) else str(user_text)

        # 「重新開始」：之前的條件、排除的店全部清掉（只看「🔁 已重新開始」之後的對話）
        messages = self.after_reset(messages)
        if len(user_text) <= 12 and any(w in user_text for w in RESET_WORDS):
            return self.reset_reply(user_text)

        # 1. 解析用戶條件
        conditions, source, info = self.get_conditions(messages)
        district, category, party_size, budget = conditions
        total, subtype = info["total"], info["subtype"]

        prefs = info["prefs"]
        said = self.parse_prefs(user_text)  # 這一句提到的其他條件
        if info["intent"] == "reset":
            return self.reset_reply(user_text)

        # 問「有什麼類型可以選」→ 列出料理類型
        if source == "LLM":
            category_question = info["intent"] == "categories"  # 「不管是什麼類型」不會被誤判
        else:
            category_question = self.is_category_question(user_text) and not said.get("near")
        if category_question:
            return self.list_categories(conditions)

        last_names = self.last_recommended(messages)
        new_search = bool(any(self.parse_user_query(user_text))
                          or self.detect_subtype(fix_homophones(user_text)[0], None)
                          or said or any(info["llm_prefs"].values()))  # LLM 常把沒提到的填 false，不算

        # 「不喜歡第 2 間」「吃過了」→ 找出是哪幾間
        prev_msg = str(messages[-2].get("content", "")) if len(messages) >= 2 else ""
        # 上一則是「是哪一間呢？」（有列出剛剛推薦的店名清單）
        asked_which = bool(prev_msg.startswith(ASK_MARK) and last_names and f"\n1. {last_names[0]}" in prev_msg)
        rejects, reject_said = self.find_rejects(user_text, last_names, force=asked_which)
        rejects = [n for n in rejects if n not in info["keep"]]
        if not rejects and info["intent"] == "reject" and (
                asked_which or self.match_names(user_text, last_names)
                or re.search(r"都|全部|這些|這幾|以上", user_text)):
            rejects = info["reject"]  # 原話有指出哪一間（或全部），才採用 LLM 判斷的店
        if reject_said and not rejects and last_names and not new_search:
            if self.wants_more(user_text):
                rejects = list(last_names)  # 「吃過了，換別的」→ 這批都不要
            else:
                return self.ask_which(last_names, user_text)

        # 不是要新推薦（問你會做什麼、道謝、閒聊、問細節）→ LLM 自然回覆。
        # 這句話有提到任何條件就當成找餐廳，避免把需求誤判成閒聊
        if info["intent"] == "chat" and not rejects and not self.wants_more(user_text) and not new_search \
                and not info["choose_category"]:  # 「你幫我挑一種料理」是要找餐廳，不是閒聊
            reply = self.chat_reply(messages)
            if reply:
                return CHAT_MARK + reply

        # 台北市以外的地點、14 類以外的料理 → 說明並建議（LLM 判斷）
        if info["outside_area"] and not district and not prefs.get("near"):
            return self.ask_outside(info["outside_area"], user_text)
        if info["unknown_food"] and not category and not info["choose_category"] and not prefs.get("cats"):
            return self.ask_unknown_food(info["unknown_food"], user_text)

        # 「越近越好」但不知道在哪個車站附近 → 先問；給不出車站 → 說明沒辦法依距離排，照一般方式推薦
        # （還不知道地區的話，下面會照常問地區）。同一次需求只問一次車站
        asked_now = prev_msg.startswith(ASK_MARK) and NEAR_ASK_MARK in prev_msg
        if asked_now and not prefs.get("sort"):
            prefs["sort"] = "near_first"  # 反問那一句的「越近越好」
        no_location = False
        if prefs.get("sort") == "near_first" and not prefs.get("near"):
            if not self.asked_in_round(messages, NEAR_ASK_MARK):
                return self.ask_station(user_text)
            prefs.pop("sort")
            no_location = asked_now  # 剛剛問了車站、使用者給不出來 → 這次要說明

        # 資訊不足 → 繼續反問
        more_waived = info["llm_waived"] | ({"category"} if info["choose_category"] else set())
        if not info["new_request"]:
            more_waived |= self.previous_waived(messages)  # 推薦過時說過不限的，不再問
        missing = self.should_ask(messages, conditions, total, prefs, more_waived)
        if missing:
            note = ("使用者給不出所在的捷運站，你讀不到他的位置，沒辦法依距離排。"
                    "請先委婉說明這件事，再問還需要的資訊。地區只是用來找店，不能說會幫他找近的店"
                    "（也可以讓他知道，之後想到附近的捷運站再告訴你就好）") if no_location else ""
            return self.ask_question(messages, conditions, missing, total, subtype, prefs,
                                     info["choose_category"], note)

        # 「你幫我挑一種料理」：條件齊了，從目前條件下有店的類別隨機挑一種
        picked = None
        if info["choose_category"] and category is None:
            picked = category = self.pick_category(messages, conditions, prefs)
            conditions = (district, category, party_size, budget)

        # 上一則問過「資料庫沒有這種店，要改成哪一種？」，這次沒有選別的子類型（例如回「都可以」）
        # → 推薦一般的店，不再問一次
        if subtype and prev_msg.startswith(ASK_MARK) \
                and f"「一般{self.shop_label(SUBTYPES[subtype][0])}」" in prev_msg \
                and not self.detect_subtype(fix_homophones(user_text)[0], None):
            subtype = None

        # 「推薦別的」→ 排除同樣條件下已經推薦過的店
        condition_line = self.describe_conditions(district, category, party_size, budget, source, subtype,
                                                  info["llm_note"], prefs)
        extra_line = self.prefs_line(prefs)
        more = self.wants_more(user_text) or info["intent"] == "more"
        before = self.rejected_history(messages)
        if info["new_request"]:
            # 全新的需求：之前不想去的店不再排除（畫面上會說明，使用者還是不想去可以再說）
            info["cleared_rejects"], before = [n for n in before if n not in rejects], []
        rejected = list(dict.fromkeys(before + rejects))
        history = self.shown_restaurants(messages, condition_line, extra_line)
        if more:
            # 換一批：同條件下推薦過的都排除（使用者說要保留的除外）
            skipped = [n for n in history if n not in info["keep"]]
        elif rejects:
            # 只換掉不喜歡的那幾間：保留目前這批，更早幾批的照樣不要再出現
            skipped = [n for n in history if n not in last_names]
        else:
            skipped = []
        shown = [n for n in skipped if n not in rejected]  # 不喜歡的另外用 🚫 列，不重複計算
        exclude = list(dict.fromkeys(rejected + skipped))

        # 2. 呼叫推薦 API（「貴一點沒差」→ 預算多 200 元來找）
        search_budget = budget + RELAX_BUDGET if budget and "budget" in prefs.get("relax", []) else budget
        payload = {
            "district": district,
            "category": category,
            "party_size": party_size,
            "budget": search_budget,
            "exclude": exclude,
        }
        if subtype:
            parent, _, keywords, cross = SUBTYPES[subtype]
            siblings = {name: [kw, cr] for name, (par, _, kw, cr) in SUBTYPES.items()
                        if par == parent and name != subtype}
            payload.update({"subtype": subtype, "subtype_keywords": keywords, "subtype_cross": cross,
                            "subtype_siblings": siblings})
        payload.update({
            "near": prefs.get("near"),
            "categories": prefs.get("cats"),
            "avoid_categories": prefs.get("avoid"),
            "no_chain": bool(prefs.get("no_chain")),
            "hidden_gem": bool(prefs.get("hidden_gem")),
            "budget_min": prefs.get("budget_min"),
            "sort": prefs.get("sort"),
            "relax": prefs.get("relax"),
            "pin": info["keep"] if more else None,
        })
        try:
            # 跟以前透過 HTTP 呼叫一樣，轉成 JSON 再轉回來（數字型別、NaN 處理都一致）
            data = json.loads(json.dumps(service.handle_recommend(payload), ensure_ascii=False, default=str))
        except Exception as e:
            return f"❌ 推薦服務發生錯誤：{e}"

        party_relaxed = False
        if not data.get("restaurants") and party_size:
            # 資料裡的店標示的人數大多在 10 人以內：人數太多找不到時，先不限人數再找一次
            retry = json.loads(json.dumps(service.handle_recommend({**payload, "party_size": None}),
                                          ensure_ascii=False, default=str))
            if retry.get("restaurants"):
                data, party_relaxed = retry, True
        restaurants = data.get("restaurants", [])
        where = self.near_text(prefs) if prefs.get("near") else district
        overview = data.get("subtype_overview") or {}
        if subtype and overview.get("total") == 0:
            return self.ask_no_subtype(subtype, category, where, overview)
        tips = self.describe_alternatives(conditions, len(restaurants), data.get("alternatives") or [], where)
        banners = self.pref_banners(prefs, data, info, category, len(restaurants))
        if not restaurants or "maps_url" not in restaurants[0]:
            # 沒有結果，或 api.py 還是舊版：直接顯示 API 的文字
            sections = []
            if self.valves.SHOW_CONDITIONS:
                sections.append(condition_line)
                if extra_line:
                    sections.append(extra_line)
            sections += banners
            if rejects:
                sections.append(REJECTED_MARK + "".join(f"「{n}」" for n in rejects))
            if more and shown:
                sections.append(f"🔄 這個條件下的店都推薦過了（已推薦 {len(shown)} 家）。")
            else:
                sections.append(data.get("result", "沒有取得有效的推薦結果。"))
            if tips:
                sections.append(tips)
            return "\n\n".join(sections)

        # 3. 組合回答：條件 → 開場 → 每家餐廳（含推薦理由）
        sections = []
        if self.valves.SHOW_CONDITIONS:
            sections.append(condition_line)
            if extra_line:
                sections.append(extra_line)
        sections += banners
        if party_relaxed:
            sections.append(f"👥 資料裡的店標示的人數大多在 10 人以內，{party_size} 人找不到符合的，以下先不限人數推薦；"
                            "人數比較多，建議先打電話確認座位或訂位")
        if picked:
            sections.append(f"🎲 幫你挑了「{picked}」，想換一種可以再跟我說")
        if total and party_size:
            sections.append(f"🧮 總預算 {total} 元 ÷ {party_size} 人 ≈ 每人 {budget} 元")
        if rejects:
            sections.append(REJECTED_MARK + "".join(f"「{n}」" for n in rejects))
        if more and shown:
            sections.append(f"🔄 換一批：已排除先前推薦過的 {len(shown)} 家")
        matched = data.get("matched", len(restaurants))
        relaxed = data.get("relaxed_budget")
        if budget and relaxed:
            if matched == 0:
                sections.append(f"⚠️ 沒有符合每人 {search_budget} 元預算的店，"
                                f"以下把預算稍微放寬到每人約 {relaxed} 元來找。")
            else:
                sections.append(f"⚠️ 符合每人 {search_budget} 元預算的店只有 {matched} 家，"
                                f"另外補上比預算稍微高一點（放寬到每人約 {relaxed} 元）的選擇。")
        # 子類型：誠實說明哪些店看得出是這種、哪些只是一般同類別的店
        matched_subtype = data.get("matched_subtype")
        general = self.shop_label(category)  # 「一般的火鍋店」「一般的美式餐廳」
        if subtype and matched_subtype is not None:
            where = self.near_text(prefs) if prefs.get("near") else (district or "台北")
            shown_sub = sum(1 for r in restaurants if r.get("subtype_match"))
            if shown_sub == 0 and exclude and matched_subtype == 0:
                sections.append(f"⚠️ 店名看得出是「{subtype}」的店都推薦過了，"
                                f"以下是一般的{general}，不一定有{subtype}。")
            elif shown_sub == 0:
                elsewhere = (f"（全台北有 {overview['total']} 家，想看可以說「不限地區」）"
                             if overview.get("total") else "")
                sections.append(f"⚠️ 資料庫裡找不到{where}店名看得出是「{subtype}」的店{elsewhere}，"
                                f"以下先推薦一般的{general}，不一定有{subtype}。")
            elif shown_sub < len(restaurants):
                over_sub = sum(1 for r in restaurants if r.get("subtype_match") and r.get("budget_fit") == "over")
                if over_sub == shown_sub:
                    note = "（比預算稍微高一點）"
                elif over_sub:
                    note = f"（其中 {over_sub} 家比預算稍微高一點）"
                else:
                    note = ""
                sections.append(f"🏷️ 店名看得出是「{subtype}」的有 {shown_sub} 家，排在最前面{note}；"
                                f"後面是一般的{general}，不一定有{subtype}。")
        # LLM 開場白和推薦理由（上面的說明畫面上已經顯示，開場白不再重複）
        texts, _ = self.current_round(messages)
        # 人數太多、不限人數找的：不讓 LLM 說「適合 30 人」這種資料沒辦法確認的事
        reason_conditions = (district, category, None if party_relaxed else party_size, search_budget)
        notes = []
        if no_location:
            notes.append("使用者想要越近越好，但沒有提供所在的捷運站，你讀不到使用者的位置，"
                         "所以這批是照一般的推薦順序。開場請先委婉說明這件事")
        elif prefs.get("sort"):
            notes.append(f"這批店依「{SORT_LABELS[prefs['sort']]}」排序，開場可以順帶提一下")
        if search_budget != budget:
            notes.append(f"使用者說預算可以超過一點，所以找到每人 {search_budget} 元")
        intro, reasons = self.write_reasons(texts, reason_conditions, restaurants, subtype,
                                            prefs.get("unsupported"), notes)
        if intro:
            sections.append(f"💬 {intro}")
        elif no_location:
            sections.append("📍 我讀不到你的位置，沒辦法依距離排，以下先照一般的推薦順序。")
        blocks = []
        for i, (r, reason) in enumerate(zip(restaurants, reasons), 1):
            prev = restaurants[i - 2] if i > 1 else None
            is_over = r.get("budget_fit") == "over"
            # 子類型的店之後，接著是一般同類別的店 → 加分隔標題
            if subtype and prev and prev.get("subtype_match") and not r.get("subtype_match"):
                extra = "，比預算稍微高一點" if is_over else ""
                blocks.append(f"🍽️ **以下是一般的{general}（店名看不出是不是{subtype}{extra}）**")
            # 一般的店裡，預算內的店之後接著是比預算稍微高一點的店
            elif (is_over and prev and prev.get("budget_fit") != "over"
                  and not (subtype and r.get("subtype_match"))):
                blocks.append(f"💸 **以下是比預算稍微高一點的選擇（放寬到每人約 {relaxed or '更高'} 元）**")
            anchor = f"捷運{data['near']}站" if data.get("near") else ""
            blocks.append(self.format_card(i, r, reason, anchor))
        sections.append("\n\n---\n\n".join(blocks))
        if tips:
            sections.append(tips)
        return "\n\n".join(sections)

    @staticmethod
    def after_reset(messages: list) -> list:
        """只留最後一次「重新開始」之後的對話。"""
        for i in range(len(messages) - 2, -1, -1):
            if messages[i].get("role") == "assistant" and RESET_MARK in str(messages[i].get("content", "")):
                return messages[i + 1:]
        return messages

    def reset_reply(self, user_text: str) -> str:
        text = self.say("使用者想重新開始，之前的條件、排除的店都已經清掉了。請自然地回應，並邀請他說想吃什麼。",
                        [f"使用者剛剛說：「{user_text}」"]) or "好，我們重新開始！想吃什麼、在哪一區呢？"
        return f"{CHAT_MARK}{text}\n\n{RESET_MARK}"

    def ask_station(self, user_text: str) -> str:
        """「越近越好」但沒說在哪 → 說明讀不到位置，請使用者提供最近的捷運站。"""
        text = self.say("使用者希望越近越好，但你讀不到使用者的位置。請自然地說明這件事，"
                        "並請他告訴你離他最近的捷運站或地標。",
                        [f"使用者剛剛說：「{user_text}」"])
        return ASK_MARK + (text or "我讀不到你現在的位置，可以告訴我離你最近的捷運站嗎？") + "\n\n" + NEAR_ASK_MARK

    def ask_outside(self, area: str, user_text: str) -> str:
        """台北市以外的地點 → 說明只有台北市的資料，並建議附近的台北市區域。"""
        text = self.say("使用者說的地點不在台北市。請說明目前只有台北市的資料，並建議一兩個離那裡比較近的"
                        "台北市行政區或捷運站，請他選一個（或說地區不限）。",
                        [f"使用者剛剛說：「{user_text}」", f"不在台北市的地點：{area}",
                         f"台北市的行政區：{'、'.join(DISTRICTS)}"])
        return ASK_MARK + (text or f"目前只有台北市的資料，「{area}」不在範圍內。想改在台北哪一區呢？"
                                    "（例如：" + "、".join(DISTRICTS[:4]) + "…，或說地區不限）")

    def ask_unknown_food(self, food: str, user_text: str) -> str:
        """14 類以外的料理 → 說明沒有這一類，並建議相近的類別。"""
        text = self.say("使用者想吃的料理不在資料的類別裡。請說明資料裡沒有這一類，並建議一到三個最相近的類別讓他選，"
                        "也可以說類型不限。",
                        [f"使用者剛剛說：「{user_text}」", f"想吃的料理：{food}",
                         f"資料裡的料理類型：{'、'.join(CATEGORIES)}"])
        return ASK_MARK + (text or f"資料裡沒有「{food}」這一類。可以選的類型有：" + "、".join(CATEGORIES)
                                    + "，想改成哪一種？")

    def pick_category(self, messages: list, conditions, prefs: dict) -> Optional[str]:
        """隨機挑一種料理：只從目前地區、預算下有店的類別挑，避開不要的類別和上一次的類別。
        用對話內容當亂數種子：同一則訊息重跑時（API 每次都從頭跑）會挑到同一種。"""
        district, _, party_size, budget = conditions
        counts = {c["category"]: c["count"] for c in service.handle_categories(
            {"district": None if prefs.get("near") else district, "party_size": party_size,
             "budget": budget}).get("categories", [])}
        previous = (self.previous_conditions(messages) or [None, None])[1]
        avoid = set(prefs.get("avoid", [])) | {previous}
        candidates = sorted(c for c in CATEGORIES if counts.get(c, 0) >= 5 and c not in avoid) \
            or sorted(c for c in CATEGORIES if counts.get(c, 0) and c not in avoid) \
            or [c for c in CATEGORIES if c not in avoid]
        seed = hashlib.md5(json.dumps([m.get("content") for m in messages], ensure_ascii=False,
                                      default=str).encode("utf-8")).hexdigest()
        return random.Random(seed).choice(candidates)

    def ask_no_subtype(self, subtype: str, category, where, overview: dict) -> str:
        """整個資料庫都沒有這種子類型的店 → 先問使用者要改成什麼，不直接列一般的店。"""
        general = self.shop_label(category)
        place = where or "台北"
        options = []
        if overview.get("general"):
            options.append(f"- 「一般{general}」→ {place}有 {overview['general']} 家符合")
        for name, n in (overview.get("siblings") or [])[:4]:
            options.append(f"- 「{name}」→ {place}有 {n} 家")
        lead = self.say(f"使用者想吃{subtype}，但整個台北的資料裡都沒有店名看得出是{subtype}的店，沒辦法確定哪家有。"
                        "請自然地說明，並請他從下面另外列出的選項挑一個，或直接說想換的類型。",
                        [f"使用者想吃：{subtype}（屬於{category}）",
                         f"整個台北（所有行政區）都沒有店名看得出是{subtype}的店，不是只有{place}沒有"])
        if lead:
            return "\n".join([ASK_MARK + lead, ""] + options)
        lines = [f"{ASK_MARK}資料庫裡沒有店名看得出是「{subtype}」的店（整個台北都找不到），"
                 f"沒辦法確定哪家有{subtype}。要改成下面哪一種？", ""] + options
        lines += ["", "直接回覆上面「」裡的說法就可以；想換別的類型也可以直接說。"]
        return "\n".join(lines)

    @staticmethod
    def shop_label(category: Optional[str]) -> str:
        """「火鍋」→「火鍋店」，「美式餐廳」「酒吧」本身就是店，不用再加「店」。"""
        if not category:
            return "餐廳"
        return category if category.endswith(("餐廳", "店", "吧")) else category + "店"

    def pref_banners(self, prefs: dict, data: dict, info: dict, category, found: int) -> list:
        """其他條件的說明：錯字修正、新需求、附近範圍、排除連鎖、隱藏版、資料沒有的條件。"""
        lines = []
        if info.get("fixes"):
            lines.append("✏️ " + "、".join(f"「{a}」看成「{b}」" for a, b in info["fixes"]))
        notes = []
        if info.get("dropped_prefs"):
            notes.append(f"之前的「{info['dropped_prefs']}」已經不再套用")
        if info.get("cleared_rejects"):
            notes.append("之前不想去的" + "".join(f"「{n}」" for n in info["cleared_rejects"])
                         + "也不再排除，還是不想去可以再跟我說")
        if notes:
            lines.append(NEW_REQUEST_MARK + "，".join(notes))
        if prefs.get("near") and data.get("near"):
            if data.get("radius_relaxed"):
                lines.append(f"📏 捷運{prefs['near']}站 {data.get('radius_first', 800)} 公尺內符合的店不到 5 家，"
                             f"範圍放寬到 {data['radius'] / 1000:g} 公里。")
        elif prefs.get("near"):
            lines.append(f"⚠️ 推薦服務還沒有「捷運{prefs['near']}站」的座標（api.py 可能是舊版），先不限地區找。")
        if prefs.get("no_chain") and data.get("chain_removed"):
            lines.append(f"🏪 已排除 {data['chain_removed']} 家連鎖店（同一個品牌在資料裡有 2 家以上的店）")
        if prefs.get("hidden_gem") and 0 < found < 5:
            # 定義已經寫在「🎛️ 其他條件：隱藏版（評分高、評論數較少）」，這裡只在家數不夠時說明
            lines.append(f"🔍 符合隱藏版的店只有 {found} 家")
        if prefs.get("unsupported"):
            items = "".join(f"「{u}」" for u in prefs["unsupported"])
            lines.append(f"⚠️ 目前的資料沒有{items}的資訊，沒辦法幫你篩選，出發前請點 Google Maps 確認。")
        return lines

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

