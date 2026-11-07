"""
title: Taipei Restaurant Recommendation Pipe
author: Taipei Food Bot
version: 3.3.0
description: 全台北市美食餐廳推薦系統 Pipe（LLM 理解需求 + 加權推薦 + 推薦理由）
"""

import json
import re
from typing import Generator, Iterator, Optional, Union

import requests
from pydantic import BaseModel, Field


DISTRICTS = [
    "大安區", "信義區", "中山區", "中正區", "萬華區", "松山區",
    "士林區", "北投區", "內湖區", "南港區", "文山區", "大同區",
]

CATEGORIES = [
    "火鍋", "燒肉", "日式料理", "拉麵", "義式料理", "平價小吃",
    "中式餐廳", "小籠包", "美式餐廳", "早午餐", "韓式料理", "泰式料理", "酒吧", "甜點",
]

# 規則解析用：口語同義詞 → 標準類別
CATEGORY_SYNONYMS = {
    "火鍋": ["麻辣鍋", "涮涮鍋", "鍋物", "石頭鍋", "薑母鴨", "羊肉爐"],
    "燒肉": ["燒烤", "烤肉"],
    "日式料理": ["日本料理", "日料", "壽司", "生魚片", "丼飯", "居酒屋"],
    "拉麵": ["ramen"],
    "義式料理": ["義大利麵", "義大利菜", "披薩", "pizza", "燉飯"],
    "平價小吃": ["小吃", "夜市", "便當", "滷肉飯"],
    "中式餐廳": ["中菜", "合菜", "台菜", "熱炒", "港式", "粵菜"],
    "美式餐廳": ["漢堡", "牛排", "美式"],
    "早午餐": ["早餐", "brunch"],
    "韓式料理": ["韓國菜", "韓國料理", "韓式", "韓料"],
    "泰式料理": ["泰國菜", "泰國料理", "泰式"],
    "酒吧": ["喝酒", "調酒", "小酌", "bar"],
    "甜點": ["蛋糕", "下午茶", "甜食", "冰品"],
}

CN_DIGITS = {"零": 0, "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_UNITS = {"十": 10, "百": 100, "千": 1000}
NUM = r"(\d+(?:\.\d+)?\s*[kK]|\d+|[零一二兩三四五六七八九十百千]+)"


def to_number(token: str) -> Optional[int]:
    """把「1000」「1k」「一千」「一千五」「兩百」轉成整數。"""
    token = token.strip()
    if not token:
        return None
    if token[-1] in "kK":
        return int(float(token[:-1].strip()) * 1000)
    if token.isdigit():
        return int(token)
    total, num, last_unit = 0, 0, 1
    for ch in token:
        if ch in CN_DIGITS:
            num = CN_DIGITS[ch]
        elif ch in CN_UNITS:
            last_unit = CN_UNITS[ch]
            total += (num or 1) * last_unit
            num = 0
    if num:
        # 「一千五」= 1500、「十二」= 12
        total += num * (last_unit // 10 if last_unit >= 100 else 1)
    return total or None


PARSE_PROMPT = f"""你是台北餐廳推薦系統的「搜尋條件抽取器」。
請根據對話，找出使用者「目前」想要的搜尋條件，只輸出一個 JSON 物件，不要任何其他文字。

JSON 格式：
{{"district": 字串或 null, "category": 字串或 null, "party_size": 整數或 null, "budget": 整數或 null}}

規則：
1. district 只能是以下其中之一，或 null：{"、".join(DISTRICTS)}
   - 「大安」「信義」這類省略「區」的說法要補成完整名稱。
   - 提到地標時，只有在非常確定所屬行政區時才填（例如「士林夜市」→士林區），不確定就填 null。
2. category 只能是以下其中之一，或 null：{"、".join(CATEGORIES)}
   - 同義詞要對應：麻辣鍋/涮涮鍋→火鍋、日本料理/壽司→日式料理、韓國菜→韓式料理、
     泰國菜→泰式料理、義大利麵/披薩→義式料理、小吃/夜市→平價小吃、早餐→早午餐、
     蛋糕/下午茶→甜點、喝酒/調酒→酒吧、中菜/合菜→中式餐廳、漢堡/牛排→美式餐廳。
3. party_size 是用餐人數（整數）。「我們兩個」=2、「一個人」=1、「三個人」=3。沒提到就 null。
4. budget 是「每人」預算上限（整數，台幣）。「一千」=1000、「1k」=1000、「500以內」=500。
   如果使用者給的是總預算且有人數，換算成每人（總預算 ÷ 人數，取整數）。沒提到就 null。
5. 多輪對話：沿用先前的條件，只更新使用者這次改動的部分。
   - 「再便宜一點」：把先前的 budget 乘以 0.7 取整數；先前沒有 budget 就設為 300。
   - 「換成○○」「改○○」：只替換對應欄位。
   - 「不限地區」「哪裡都可以」：district 設為 null。
6. 如果助理上一則是在反問（以 🤔 開頭），使用者這次的回覆就是在回答那些問題，
   要和更早的需求合併。例如助理問「每人預算大概多少？」，使用者回「800」→ budget=800。
"""

REASON_PROMPT = """你是台北餐廳推薦助理。根據使用者的需求和每家餐廳的資料，為每一家寫推薦理由。
規則：
- 每家寫 1~2 句，說明這家店的特色，以及為什麼適合這位使用者（人數、預算、想吃的類型）。
- min_price ~ max_price 是「多數人用餐的每人花費區間」，不是最低消費或最高消費，提到價格時請這樣理解。
- 只能根據資料裡的內容：名稱、類別、價格、facts（客觀事實）、summary（店家簡介）、
  highlights（服務特色）、reviews（顧客評論摘錄）。資料沒有的菜色、裝潢、營業時間不可以編造。
- reviews 裡顧客常提到的招牌菜、湯頭、服務、氣氛，可以用「評論常提到…」的方式引用。
- 不要寫推薦指數、分數或計算方式。不要每家都用一樣的句型開頭。
- 預算超出的部分（budget_fit 是 "partial" 或 "over"）系統會另外顯示原因，推薦理由裡不用再提。
- 只輸出 JSON，不要其他文字，格式：
  {"intro": "一句話的開場，例如這批推薦的整體方向", "reasons": ["第 1 家的理由", "第 2 家的理由", ...]}
  reasons 的數量和順序要和餐廳資料完全一樣。
"""

# 反問訊息的開頭符號（用來判斷上一則是不是反問）
ASK_MARK = "🤔 "
# 使用者表示「全部都不在意」的說法 → 不再反問，直接推薦
SKIP_WORDS = ["都可以", "隨便", "都行", "沒差", "無所謂", "皆可", "直接推薦", "不用問"]
# 針對單一條件表示「不限」的說法（先比對這些，再比對上面的 SKIP_WORDS）
FIELD_SKIP_WORDS = {
    "district": ["不限地區", "地區不限", "哪裡都", "哪一區都", "哪區都", "地區都可以", "地點都可以",
                 "區域都可以", "地區沒差", "地點不限"],
    "category": ["不限類別", "類別不限", "類型不限", "類型都可以", "種類都可以", "什麼都可以",
                 "什麼都好", "什麼都吃", "吃什麼都", "都可以吃", "料理都可以"],
    "party_size": ["人數不限", "人數都可以", "人數不確定", "不確定幾個人", "人數還不知道"],
    "budget": ["不限預算", "預算不限", "預算都可以", "沒有預算", "預算沒差", "不在意預算",
               "多少都可以", "預算無上限", "沒預算"],
}
FIELDS = ["district", "category", "party_size", "budget"]
QUESTIONS = {
    "district": "想在哪一區吃？（例如：大安、信義、中山、士林…）",
    "category": "想吃什麼類型？（例如：火鍋、燒肉、日式料理、拉麵、早午餐…）",
    "party_size": "幾個人用餐？",
    "budget": "每人預算上限大概多少？（例如：300、500、1000）",
}
LABELS = {"district": "地區", "category": "類型", "party_size": "人數", "budget": "預算"}


class Pipe:
    class Valves(BaseModel):
        API_URL: str = Field(
            default="http://127.0.0.1:5000/recommend",
            description="推薦 API 網址（api.py）",
        )
        LLM_BASE_URL: str = Field(
            default="http://163.14.73.40:8088/v1",
            description="LLM 服務網址（OpenAI 相容格式，結尾是 /v1）",
        )
        LLM_API_KEY: str = Field(
            default="",
            description="LLM API Key（實驗室 .env 裡的 LLM_API_KEY）。留空則只用規則解析",
        )
        LLM_MODEL: str = Field(
            default="",
            description="模型名稱。留空會自動使用服務提供的第一個模型",
        )
        USE_LLM_PARSER: bool = Field(default=True, description="用 LLM 理解使用者需求")
        USE_LLM_REASON: bool = Field(default=True, description="用 LLM 寫推薦理由")
        SHOW_CONDITIONS: bool = Field(default=True, description="在回答開頭顯示解析出的條件")
        ASK_FOLLOWUP: bool = Field(default=True, description="資訊不足時持續反問，直到地區、類型、人數、預算都清楚")
        MAX_ASK: int = Field(default=4, description="同一個需求最多反問幾次（避免一直問不完）")
        SHOW_REASONS: bool = Field(default=True, description="在每家餐廳附上推薦理由")

    def __init__(self):
        self.valves = self.Valves()
        self._model_cache = None

    # ------------------------------------------------------------------
    # LLM 呼叫
    # ------------------------------------------------------------------
    def _llm_enabled(self) -> bool:
        return bool(self.valves.LLM_API_KEY and self.valves.LLM_BASE_URL)

    def _base_url(self) -> str:
        return self.valves.LLM_BASE_URL.rstrip("/")

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.valves.LLM_API_KEY}"}

    def _model(self) -> str:
        if self.valves.LLM_MODEL:
            return self.valves.LLM_MODEL
        if self._model_cache is None:
            r = requests.get(f"{self._base_url()}/models", headers=self._headers(), timeout=10)
            r.raise_for_status()
            self._model_cache = r.json()["data"][0]["id"]
        return self._model_cache

    def _chat(self, messages: list, max_tokens: int = 400) -> str:
        payload = {
            "model": self._model(),
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            # Qwen3 系列：關閉思考模式，回應更快（不支援的伺服器會忽略）
            "chat_template_kwargs": {"enable_thinking": False},
        }
        r = requests.post(
            f"{self._base_url()}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"] or ""
        # 移除可能出現的思考內容
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # ------------------------------------------------------------------
    # 條件解析：LLM 優先，失敗時退回規則
    # ------------------------------------------------------------------
    def parse_user_query(self, text: str):
        """規則解析（不需要 LLM），支援常見口語說法。"""
        district = None
        category = None
        budget = None
        party_size = None
        lower = text.lower()

        # 行政區：「大安區」或省略「區」的「大安」
        for d in DISTRICTS:
            if d in text or d.replace("區", "") in text:
                district = d
                break

        # 類別：先找標準名稱，再找同義詞（麻辣鍋→火鍋、韓國菜→韓式料理…）
        for c in CATEGORIES:
            if c in text:
                category = c
                break
        if category is None:
            for c, words in CATEGORY_SYNONYMS.items():
                if any(w.lower() in lower for w in words):
                    category = c
                    break

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

        return district, category, party_size, budget

    def parse_with_llm(self, messages: list):
        """用 LLM 解析整段對話，回傳 (district, category, party_size, budget)。"""
        history = []
        for m in messages[-12:]:
            content = m.get("content", "")
            if isinstance(content, str) and m.get("role") in ("user", "assistant"):
                # 助理的長篇回答只保留開頭（含先前解析的條件），節省 token
                history.append({"role": m["role"], "content": content[:300]})

        raw = self._chat([{"role": "system", "content": PARSE_PROMPT}] + history, max_tokens=200)
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise ValueError(f"LLM 沒有回傳 JSON：{raw[:100]}")
        data = json.loads(match.group(0))

        district = data.get("district")
        district = district if district in DISTRICTS else None
        category = data.get("category")
        category = category if category in CATEGORIES else None

        def to_int(v):
            try:
                v = int(float(v))
                return v if v > 0 else None
            except (TypeError, ValueError):
                return None

        return district, category, to_int(data.get("party_size")), to_int(data.get("budget"))

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
            if content.startswith(ASK_MARK):
                continue
            match = re.search(r"🔎 搜尋條件：(.+?)（", content)
            if not match:
                return None
            parts = match.group(1).split("｜")
            district = next((p for p in parts if p in DISTRICTS), None)
            category = next((p for p in parts if p in CATEGORIES), None)
            party = re.search(r"(\d+) 人", match.group(1))
            budget = re.search(r"每人 (\d+) 元內", match.group(1))
            return (district, category,
                    int(party.group(1)) if party else None,
                    int(budget.group(1)) if budget else None)
        return None

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

    def get_conditions(self, messages: list):
        """回傳 (條件 tuple, 解析方式)。"""
        conditions, source = None, "規則"
        if self.valves.USE_LLM_PARSER and self._llm_enabled():
            try:
                conditions, source = self.parse_with_llm(messages), "LLM"
            except Exception:
                pass  # LLM 失敗時退回規則解析
        if conditions is None:
            conditions = self.parse_rounds(messages)
        # 使用者說「不限地區」「預算不限」→ 那一項清空
        texts, _ = self.current_round(messages)
        waived = self.waived_fields(texts)
        conditions = tuple(None if f in waived else v for f, v in zip(FIELDS, conditions))
        return conditions, source

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

    def should_ask(self, messages: list, conditions) -> list:
        """回傳還要問的欄位；空 list 代表資訊足夠，可以推薦。"""
        if not self.valves.ASK_FOLLOWUP:
            return []
        texts, asks = self.current_round(messages)
        if not texts or self.skip_all(texts[-1]):
            return []
        if asks >= self.valves.MAX_ASK:
            return []  # 已經問很多次了，用目前的條件推薦
        waived = self.waived_fields(texts)
        return [f for f, v in zip(FIELDS, conditions) if v is None and f not in waived]

    def ask_question(self, messages: list, conditions, missing: list) -> str:
        texts, asks = self.current_round(messages)
        waived = self.waived_fields(texts)
        known = []
        for field, value in zip(FIELDS, conditions):
            if value is not None:
                known.append({"party_size": f"{value} 人", "budget": f"每人 {value} 元內"}.get(field, value))
            elif field in waived:
                known.append(f"{LABELS[field]}不限")

        if asks == 0:
            opening = "想幫你找到更適合的餐廳，可以再多告訴我一些嗎？"
        else:
            opening = "了解！還差一點點資訊，我就能幫你挑出最合適的餐廳："
        lines = [f"{ASK_MARK}{opening}"]
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
    # 推薦理由（每家一段，寫餐廳特色）
    # ------------------------------------------------------------------
    def write_reasons(self, texts: list, conditions, restaurants: list):
        """回傳 (開場白, 每家的推薦理由 list)。LLM 不能用時，用資料中的事實組成理由。"""
        fallback = [("；".join(r.get("facts") or []) + "。") if r.get("facts") else ""
                    for r in restaurants]
        if not (self.valves.USE_LLM_REASON and self._llm_enabled() and restaurants):
            return None, fallback
        district, category, party_size, budget = conditions
        need = "；".join(texts[-4:])
        cond = (f"地區={district or '不限'}，類型={category or '不限'}，"
                f"人數={party_size or '不限'}，每人預算上限={budget or '不限'}")
        keys = ["name", "category", "min_price", "max_price",
                "facts", "summary", "highlights", "reviews"]
        data = [{k: r.get(k) for k in keys if r.get(k) not in (None, "", [])} for r in restaurants]
        try:
            raw = self._chat(
                [
                    {"role": "system", "content": REASON_PROMPT},
                    {"role": "user", "content": f"使用者需求：{need}\n條件：{cond}\n\n"
                                                f"餐廳資料：{json.dumps(data, ensure_ascii=False)}"},
                ],
                max_tokens=900,
            )
            result = json.loads(re.search(r"\{.*\}", raw, flags=re.DOTALL).group(0))
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
    def describe_conditions(district, category, party_size, budget, source) -> str:
        parts = [
            district or "不限地區",
            category or "不限類別",
            f"{party_size} 人" if party_size else "人數不限",
            f"每人 {budget} 元內" if budget else "預算不限",
        ]
        return f"🔎 搜尋條件：{'｜'.join(parts)}（{source}解析）"

    @staticmethod
    def price_text(r: dict) -> str:
        # 價格區間是「多數人用餐的花費」，不是最低／最高消費
        low, high = int(r["min_price"]), int(r["max_price"])
        text = f"多數人花費 ${high} 以下 / 人" if low <= 1 else f"多數人花費 ${low} - ${high} / 人"
        if r.get("budget_fit") == "partial":
            text += "（⚠️ 可能超出預算）"
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

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        messages = body.get("messages", [])
        if not messages:
            return "無有效的對話訊息。"

        # 1. 解析用戶條件
        conditions, source = self.get_conditions(messages)
        district, category, party_size, budget = conditions

        # 資訊不足 → 繼續反問
        missing = self.should_ask(messages, conditions)
        if missing:
            return self.ask_question(messages, conditions, missing)

        # 2. 呼叫推薦 API
        payload = {
            "district": district,
            "category": category,
            "party_size": party_size,
            "budget": budget,
        }
        try:
            response = requests.post(self.valves.API_URL, json=payload, timeout=10)
            if response.status_code != 200:
                return f"❌ API 回應異常 (HTTP {response.status_code})"
            data = response.json()
        except Exception as e:
            return f"❌ 無法連線至本機推薦 API 服務，請確認 `python api.py` 是否已啟動。\n錯誤訊息: {e}"

        restaurants = data.get("restaurants", [])
        tips = self.describe_alternatives(conditions, len(restaurants), data.get("alternatives") or [])
        if not restaurants or "maps_url" not in restaurants[0]:
            # 沒有結果，或 api.py 還是舊版：直接顯示 API 的文字
            sections = []
            if self.valves.SHOW_CONDITIONS:
                sections.append(self.describe_conditions(district, category, party_size, budget, source))
            sections.append(data.get("result", "沒有取得有效的推薦結果。"))
            if tips:
                sections.append(tips)
            return "\n\n".join(sections)

        # 3. 組合回答：條件 → 開場 → 每家餐廳（含推薦理由）
        texts, _ = self.current_round(messages)
        intro, reasons = self.write_reasons(texts, conditions, restaurants)

        sections = []
        if self.valves.SHOW_CONDITIONS:
            sections.append(self.describe_conditions(district, category, party_size, budget, source))
        matched = data.get("matched", len(restaurants))
        relaxed = data.get("relaxed_budget")
        if budget and relaxed:
            if matched == 0:
                sections.append(f"⚠️ 沒有符合每人 {budget} 元預算的店，"
                                f"以下把預算稍微放寬到每人約 {relaxed} 元來找。")
            else:
                sections.append(f"⚠️ 符合每人 {budget} 元預算的店只有 {matched} 家，"
                                f"後面補上比預算稍微高一點（放寬到每人約 {relaxed} 元）的選擇。")
        if intro:
            sections.append(f"💬 {intro}")
        blocks = []
        for i, (r, reason) in enumerate(zip(restaurants, reasons), 1):
            # 前面有符合預算的店時，才需要分隔標題（第一家就超過預算時，開頭的 ⚠️ 已經說明）
            if r.get("budget_fit") == "over" and i > 1 and restaurants[i - 2].get("budget_fit") != "over":
                blocks.append(f"💸 **以下是比預算稍微高一點的選擇（放寬到每人約 {relaxed or '更高'} 元）**")
            blocks.append(self.format_card(i, r, reason))
        sections.append("\n\n---\n\n".join(blocks))
        if tips:
            sections.append(tips)
        return "\n\n".join(sections)

    @staticmethod
    def describe_alternatives(conditions, found: int, alternatives: list) -> str:
        """結果不到 5 家時，告訴使用者可以怎麼放寬條件。"""
        if not alternatives:
            return ""
        district = conditions[0]
        where = district or "目前條件下"
        if found:
            head = f"💡 {where}符合的只有 {found} 家（資料庫裡的店家數有限）。想看更多選擇，可以直接回覆："
        else:
            head = f"💡 {where}沒有符合的店。可以試試直接回覆："
        lines = ["---", head]
        for alt in alternatives:
            lines.append(f"- 「{alt['reply']}」→ {alt['text']}")
        return "\n".join(lines)
