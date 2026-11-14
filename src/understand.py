"""
理解：把使用者的話變成條件（LLM 解析為主、規則補漏與保險）、意圖判斷。
是 Bot（bot.py）的一部分：函式裡的 self 就是 Bot，可以互相呼叫。
"""
import re
from typing import Optional

from recommender import BUDGET_STEP  # 預算的一個級距（200 元）
from lexicon import (  # 詞庫、提示詞、文字處理
    ASK_MARK, BUDGET_MIN_WORDS, BUDGET_UP_WORDS, CATEGORIES, CATEGORY_SYNONYMS, CHAT_MARK, CHOOSE_MARK, CN_INDEX, DISTRICTS,
    FIELDS, FIELD_SKIP_WORDS, MORE_WORDS, NEG_AFTER_RE, NEG_BEFORE_RE, NEG_JOIN_RE, NUM, PARSE_PROMPT, PERSON_PREFS,
    PREF_OFF_WORDS, REJECT_SOFT, REJECT_STRONG, RELAXES, RELAX_LABELS, SKIP_WORDS, SORTS, STYLE_WORDS,
    SUBTYPES, TOTAL_WORDS, UNSURE_WORDS, as_list, find_near, fix_homophones, to_number,
)
from llm import LLMError

# 防 LLM 讀錯的保險規則（都是 LLM 實測讀錯過才加的）。設定 OFF_GUARDS 可以個別關掉，
# 用來測試換成微調的模型之後，哪些規則已經不需要（finetune/）
GUARDS = {
    "homophone": "交給 LLM 前先修同音錯字（大按→大安）",
    "fill_from_rules": "LLM 沒填的基本條件用規則讀到的補（沿用前文）",
    "near_rule": "規則讀到的捷運站／地標優先（LLM 會把西門町換成行政區）",
    "avoid_rule": "規則讀到的不要的類別、多類別（剛吃過火鍋、日式或韓式）",
    "pref_off": "取消不要連鎖／隱藏版，這句要真的有提到",
    "budget_min_words": "預算下限，這句要有以上、到、起跳",
    "switching": "換成別的類型時，不採用 LLM 判斷的不要原本類型",
    "budget_from_rules": "說了強調或放寬時，預算用規則讀到的（LLM 會順手改預算）",
    "budget_up_words": "加預算，這句要有貴、高、多、超過、加",
    "category_name": "句子寫了類型的正式名稱，以使用者寫的為準",
    "unsure": "說不知道時，LLM 判斷的不限不算",
    "total_words": "總預算，這次要有總、共、加起來、合計",
    "choose_on_avoid": "排除目前的類型又沒說換什麼，一律幫你挑",
}


class UnderstandMixin:
    def guard(self, name: str) -> bool:
        """這條保險規則有沒有開（預設都開）。"""
        assert name in GUARDS, name
        return name not in {g.strip() for g in self.valves.OFF_GUARDS.split(",")}

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
            r"(?<!總)預算" + filler + NUM,  # 「總預算」是總金額，交給 LLM 換算，規則不當成每人預算
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
            "total_budget": nullable("integer"),
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
            "relax": array(RELAXES),
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
                text = content[:300]
                lines.append(f"使用者：{fix_homophones(text)[0] if self.guard('homophone') else text}")  # 先修好錯字
            elif content.startswith(ASK_MARK):
                lines.append(f"助理（反問）：{content[:200]}")
            elif content.startswith(CHAT_MARK):
                lines.append(f"助理（閒聊）：{content[len(CHAT_MARK):][:150]}")
            else:
                # 推薦結果只留先前的條件，其餘省略
                cond = re.search(r"🔎 搜尋條件：([^（\n]+)", content)
                picked = "；料理類型是你幫使用者挑的" if "🎲 幫你挑了" in content else ""
                lines.append(f"助理：（推薦了餐廳；當時的條件是 {cond.group(1).strip() if cond else '未知'}"
                             f"{picked}）")

        system = PARSE_PROMPT
        if recent_names:
            system += "\n最近推薦過的店：\n" + "\n".join(f"{i}. {n}" for i, n in enumerate(recent_names, 1))
        user = "以下是對話紀錄，最後一行是使用者這次說的話：\n\n" + "\n".join(lines) + "\n\n請輸出 JSON。"
        raw = self._chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                         max_tokens=800, schema=self.parse_schema(recent_names))
        data = self._extract_json(raw)
        self.trace["llm_raw"] = data

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
                "total_budget": to_int(data.get("total_budget")),
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
        relax = [r for r in RELAXES if r in as_list(data.get("relax"))]
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
    def category_mentions(text: str) -> list:
        """這句話提到的類別：[(位置, 類別, 是不是「不要」)]，依出現順序。
        子類型的說法（「韓式烤肉」）算它的大類別，而且優先於比較短的詞（不會變成「韓式」＋「烤肉」）。"""
        lower = text.lower()
        words = [(c, c, False) for c in CATEGORIES]
        words += [(w, c, False) for c, ws in CATEGORY_SYNONYMS.items() for w in ws]
        words += [(t, parent, True) for parent, triggers, _, _ in SUBTYPES.values() for t in triggers]
        taken = [False] * len(text)
        found, ends, subs = [], {}, {}
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
                    ends[start], subs[start] = end, is_sub
                start = lower.find(w, end)
        found.sort()
        # 「義式和中式都不要」「不要火鍋跟燒肉」：用「和、跟」連在一起的類別，其中一個是「不要」，另一個也是
        # （子類型的「不要」本來就不算，這裡也不傳）
        for _ in range(2):  # 往前、往後各傳一次
            for i in range(len(found) - 1):
                (s1, c1, n1), (s2, c2, n2) = found[i], found[i + 1]
                between = text[ends[s1]:s2]
                if n1 != n2 and re.match(NEG_JOIN_RE, between) and not (subs[s1] or subs[s2]):
                    found[i], found[i + 1] = (s1, c1, True), (s2, c2, True)
        return found

    def rule_prefs(self, text: str) -> dict:
        """規則判斷的其他條件，都是 LLM 實測讀錯過、留著當保險的（沒有 LLM 時也用得到）：
        捷運站／地標（LLM 會把「西門町」換成行政區）、不要的類別（「剛吃過火鍋」）、多類別（「日式或韓式」）。
        其他條件（預算區間、不要連鎖、隱藏版…）都交給 LLM。"""
        t = fix_homophones(text)[0]
        p = {}
        near = find_near(t)
        if near:
            p["near"], p["near_label"] = near
        mentions = self.category_mentions(t)
        avoid = list(dict.fromkeys(c for _, c, neg in mentions if neg))
        want = list(dict.fromkeys(c for _, c, neg in mentions if not neg))
        if avoid:
            p["avoid"] = avoid  # 「剛吃過火鍋」「火鍋以外」
        if len(want) >= 2:
            p["cats"] = want  # 「日式或韓式都可以」
        return p

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
            p = self.rule_prefs(text)
            if not self.guard("near_rule"):
                p.pop("near", None)
            if not self.guard("avoid_rule"):
                p.pop("avoid", None)
                p.pop("cats", None)
            if "near" in p:
                prefs["near"], prefs["near_label"] = p["near"], p["near_label"]
                rule_near = True
            elif self.parse_user_query(text)[0]:  # 改說行政區 → 不再是「某站附近」
                prefs.pop("near", None)
                prefs.pop("near_label", None)
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
            elif llm_prefs.get(k) is False and (not self.guard("pref_off")
                                                 or any(w in last_text for w in PREF_OFF_WORDS[k])):
                prefs[k] = False  # 「連鎖也可以」「有名的也可以」（LLM 常把沒提到的填成 false）
        if llm_prefs.get("budget_min") and (not self.guard("budget_min_words")
                                            or any(w in last_text for w in BUDGET_MIN_WORDS)):
            prefs["budget_min"] = llm_prefs["budget_min"]
        if llm_prefs.get("sort"):
            prefs["sort"] = llm_prefs["sort"]  # 強調的重點只有一個，以最新的說法為準
        if llm_prefs.get("relax"):
            prefs["relax"] = [r for r in RELAX_LABELS if r in prefs.get("relax", []) + llm_prefs["relax"]]
        if not prefs.get("near"):
            prefs["relax"] = [r for r in prefs.get("relax", []) if r != "distance"]  # 沒有車站，談不上距離
        if llm_prefs.get("cats"):
            prefs["cats"] = llm_prefs["cats"]
        elif any(self.parse_user_query(t)[1] for t in texts) and \
                not any(self.rule_prefs(t).get("cats") for t in texts):
            prefs.pop("cats", None)  # 改說單一類別 → 不再是「日式或韓式」
        # 「換成拉麵」：這句說了要改吃別的類型、又沒有排斥的說法 → 不採用 LLM 判斷的「不要的類別」
        # （LLM 會把原本的類型也當成不要，而這個條件會一直沿用）。「別再推火鍋了」提到的是火鍋本身，照常採用
        switching = any(not neg and c not in llm_prefs.get("avoid", [])
                        for _, c, neg in self.category_mentions(fix_homophones(last_text)[0])) \
            and not self.rule_prefs(last_text).get("avoid") and self.guard("switching")
        if llm_prefs.get("avoid") and not switching:
            prefs["avoid"] = list(dict.fromkeys(prefs.get("avoid", []) + llm_prefs["avoid"]))
        waived = self.waived_fields(texts)
        if "district" in waived:
            prefs.pop("near", None)
            prefs.pop("near_label", None)
        if "budget" in waived:
            prefs.pop("budget_min", None)
        if "category" in waived:
            prefs.pop("cats", None)
        # 只對這一句有效的：資料沒有的條件
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
        where = district or self.rule_prefs(texts[0]).get("near")
        what = category or self.detect_subtype(fix_homophones(texts[0])[0], None)
        return sum(bool(x) for x in (where, what, party_size, budget)) >= 3

    def get_conditions(self, messages: list):
        """回傳 (條件 tuple, 解析方式, info)。
        info：total（總預算）、subtype（子類型）、intent / reject（LLM 判斷的意圖與不喜歡的店）、
              prefs（其他條件）、llm_note（LLM 沒用上的原因，顯示在條件行）。
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
                self.trace.update(llm=d, rules=rules)
                # LLM 沒填的欄位用規則補（例如 LLM 漏掉沿用的條件）。
                # 「不限」只看使用者有沒有說「地區都可以」「預算不限」這類話（LLM 常把沒說不限的欄位也標成不限）
                conditions = [lv if lv is not None or not self.guard("fill_from_rules") else rv
                              for lv, rv in zip(llm, rules)]
                # 「預算再高一點」「貴一點沒差」→ 預算 +200（說幾次加幾次，條件行看得到）。
                # LLM 常把前面說過的放寬再填一次，這句話真的有提到才算
                last_text = texts[-1] if texts else ""
                raise_budget = "budget" in d["prefs"].get("relax", []) \
                    and (any(w in last_text for w in BUDGET_UP_WORDS) or not self.guard("budget_up_words"))
                # 使用者強調重點或說要加預算時，LLM 常順手把預算 ×1.3 或 ×0.7 → 預算用規則讀到的
                if (d["prefs"].get("sort") or "distance" in d["prefs"].get("relax", []) or raise_budget) \
                        and rules[3] and self.guard("budget_from_rules"):
                    if conditions[3] != rules[3]:
                        self.note(f"預算用規則讀到的 {rules[3]}（LLM 填了 {conditions[3]}；"
                                  "說了強調或加預算時，LLM 常順手改預算）")
                    conditions[3] = rules[3]
                if "budget" in d["prefs"].get("relax", []) and not raise_budget:
                    self.note("LLM 判斷要加預算，但這句話沒有提到，不採用（多半是抄前面說過的）")
                if raise_budget and conditions[3]:
                    self.note(f"加預算：{conditions[3]} + {BUDGET_STEP}")
                    conditions[3] += BUDGET_STEP
                # 這句直接寫了某個類別的正式名稱（「換成拉麵」）→ 以使用者寫的為準（LLM 偶爾把拉麵歸成日式料理）
                fixed_last = fix_homophones(last_text)[0]
                named = {c for _, c, neg in self.category_mentions(fixed_last)
                         if not neg and c in fixed_last and c not in d["prefs"].get("avoid", [])}
                if len(named) == 1 and conditions[1] not in named and self.guard("category_name"):
                    self.note(f"類型用這句寫的「{min(named)}」（LLM 填了 {conditions[1]}）")
                    conditions[1] = min(named)
                # LLM 說「不限」的欄位，只有在那一項本來就不知道時才採用（避免把已知的條件清掉）
                unsure = any(w in (texts[-1] if texts else "") for w in UNSURE_WORDS) and self.guard("unsure")
                info["llm_waived"] = set() if unsure else                     {f for f, v in zip(FIELDS, conditions) if f in d["waived"] and v is None}
                info["choose_category"] = d["choose_category"]
                info.update(keep=d["keep"], outside_area=d["outside_area"], unknown_food=d["unknown_food"])
                source = "LLM"
                info.update(intent=d["intent"], reject=d["reject"], llm_prefs=d["prefs"], total=d["total_budget"])
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

        # 總預算（LLM 讀出來的）：有人數就換算成每人；還不知道人數就先不填，等問完人數再算
        total = info.pop("total", None)
        if total and not any(w in t for t in texts for w in TOTAL_WORDS) and self.guard("total_words"):
            self.note(f"LLM 判斷總預算 {total}，但這次沒說總共、合計，不採用（多半是把每人預算當成總額）")
            total = None
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
            info["dropped_prefs"] = self.prefs_text(old).replace("｜", "、")
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
            # 「別再推火鍋了」：LLM 判斷在排斥目前的類別 → 不算想吃（規則讀不出這種否定，會以為提到火鍋＝想吃）。
            # 這次才提出的類別不會同時不要（「大安 中式 要素食」LLM 會把中式也填進不要的類別）
            if prev_category in info["llm_prefs"].get("avoid", []) and not new_request:
                asked.discard(prev_category)
            if category in asked:
                prefs["avoid"] = [c for c in prefs["avoid"] if c != category]
                if not prefs["avoid"]:
                    prefs.pop("avoid")
            else:
                category, subtype = None, None
                # 不要目前的類別、又沒說要換成什麼 → 一律幫使用者挑一種（不看 LLM 有沒有判斷成「幫我挑」）
                if not prefs.get("cats") and self.guard("choose_on_avoid"):
                    info["choose_category"] = True
        conditions = (district, category, party_size, budget)
        info.update(total=total, subtype=subtype, prefs=prefs)
        self.trace.update(source=source, final=conditions, prefs=prefs, llm_note=info["llm_note"])
        return conditions, source, info

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
    def wants_more(text: str) -> bool:
        return any(w in text for w in MORE_WORDS)

    @staticmethod
    def find_rejects(text: str, names: list, force: bool = False):
        """回傳 (使用者指定不要的店名, 是否在表達「不喜歡／吃過」)。
        force=True：剛問過「是哪一間呢？」，這次的回覆就算沒說「吃過」也是在指定店家。"""
        strong = any(w in text for w in REJECT_STRONG) or force
        soft = any(w in text.lower() for w in REJECT_SOFT)
        if not (strong or soft) or not names:
            return [], strong
        result = UnderstandMixin.match_names(text, names)
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
