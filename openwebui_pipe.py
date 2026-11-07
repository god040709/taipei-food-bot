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
{{"intent": "search" 或 "more" 或 "reject" 或 "chat", "district": 字串或 null, "category": 字串或 null,
  "party_size": 整數或 null, "budget": 整數或 null, "reject": [店名, ...]}}

規則：
1. district 只能是以下其中之一，或 null：{"、".join(DISTRICTS)}
   - 「大安」「信義」這類省略「區」的說法要補成完整名稱。
   - 提到地標時，只有在非常確定所屬行政區時才填（例如「士林夜市」→士林區），不確定就填 null。
2. category 只能是以下其中之一，或 null：{"、".join(CATEGORIES)}
   - 同義詞要對應：麻辣鍋/涮涮鍋/壽喜燒/羊肉爐→火鍋、日本料理/壽司→日式料理、韓國菜→韓式料理、
     泰國菜→泰式料理、義大利麵/披薩→義式料理、小吃/夜市→平價小吃、早餐→早午餐、
     蛋糕/下午茶→甜點、喝酒/調酒→酒吧、中菜/合菜→中式餐廳、漢堡/牛排→美式餐廳。
3. party_size 是用餐人數（整數）。「我們兩個」=2、「一個人」=1、「三個人」=3。沒提到就 null。
4. budget 是「每人」預算上限（整數，台幣）。「一千」=1000、「1k」=1000、「500以內」=500。
   如果使用者給的是總預算（「總預算」「總共」「一共」「全部加起來」）且有人數，換算成每人（總預算 ÷ 人數，取整數）；
   總預算但還不知道人數時填 null。沒提到就 null。
5. 多輪對話：沿用先前的條件，只更新使用者這次改動的部分。
   - 「再便宜一點」：把先前的 budget 乘以 0.7 取整數；先前沒有 budget 就設為 300。
   - 「換成○○」「改○○」：只替換對應欄位。
   - 「不限地區」「哪裡都可以」：district 設為 null。
6. 如果助理上一則是在反問（以 🤔 開頭），使用者這次的回覆就是在回答那些問題，
   要和更早的需求合併。例如助理問「每人預算大概多少？」，使用者回「800」→ budget=800。
7. intent 是使用者這句話的意圖，只能是以下其中之一：
   - "search"：提出或修改搜尋條件、回答助理的反問。
   - "more"：要求換一批、推薦別的、還有其他的。
   - "reject"：表示不喜歡、吃過、去過、不想去「推薦過的某幾間店」。
     reject 要填那些店的完整名稱，只能從「最近推薦過的店」清單裡選；「第2間」就是清單第 2 家。
   - "chat"：道謝、閒聊、或詢問已經推薦的店的細節，不是要新的推薦。
   不管 intent 是什麼，district / category / party_size / budget 都要填目前沿用的條件。
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
- intro 是一兩句自然的開場，要回應使用者最新的那句話（例如「了解，幫你避開吃過的那家」）；
  如果有「系統說明」（排除了店家、預算放寬、找不到某種子類型），用自然的口吻帶到。
- 只輸出 JSON，不要其他文字，格式：
  {"intro": "開場", "reasons": ["第 1 家的理由", "第 2 家的理由", ...]}
  reasons 的數量和順序要和餐廳資料完全一樣。
"""

CHAT_PROMPT = """你是「台北美食推薦機器人」的對話助理，用繁體中文、自然友善地回覆。
- 使用者這句話不是要新的推薦（例如道謝、閒聊、問推薦過的店的細節）。
- 關於餐廳的問題，只能根據下面「最近的推薦內容」回答；資料沒有的（營業時間、菜單、停車、訂位方式…）
  就說資料裡沒有，建議點 Google Maps 連結查看。不要編造餐廳或資料。
- 回覆 1~3 句。最後可以自然地提一下能做的事：換一批、改條件、說出想吃的類型。
"""

# 反問訊息的開頭符號（用來判斷上一則是不是反問）
ASK_MARK = "🤔 "
# 閒聊回覆的開頭符號（不影響條件的沿用）
CHAT_MARK = "💬 "
# 「不喜歡／吃過」：強烈的說法單獨出現就算；「不要」這類要搭配指定的店（第幾間、店名）才算
REJECT_STRONG = ["不喜歡", "吃過", "去過", "踩雷", "難吃", "沒興趣", "不想去", "不想吃", "不考慮", "排除"]
REJECT_SOFT = ["不要", "換掉", "跳過", "pass"]
REJECTED_MARK = "🚫 不再推薦："
CN_INDEX = {"一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5}
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

# 子類型：大類別裡更細的種類。資料集只有 14 個大類，子類型是「從店名判斷」的：
#   子類型: (所屬類別, 使用者會說的詞, 店名關鍵字, 是否跨類別比對)
# 「是否跨類別比對」= True：別的類別裡店名有關鍵字的店也算（例如被分在「燒肉」的壽喜燒店）；
#   關鍵字太廣（例如「韓」「麵」）的設 False，只在原本類別裡比對。
SUBTYPES = {
    # 火鍋
    "麻辣鍋": ("火鍋", ["麻辣鍋", "麻辣火鍋", "麻辣"], ["麻辣"], True),
    "壽喜燒": ("火鍋", ["壽喜燒", "壽喜鍋", "sukiyaki"], ["壽喜燒", "壽喜鍋", "すき焼", "sukiyaki"], True),
    "涮涮鍋": ("火鍋", ["涮涮鍋", "shabu"], ["涮涮", "shabu", "しゃぶ"], True),
    "石頭火鍋": ("火鍋", ["石頭鍋", "石頭火鍋"], ["石頭"], True),
    "酸菜白肉鍋": ("火鍋", ["酸菜白肉"], ["酸菜白肉", "酸白菜"], True),
    "羊肉爐": ("火鍋", ["羊肉爐"], ["羊肉爐"], True),
    "薑母鴨": ("火鍋", ["薑母鴨"], ["薑母鴨"], True),
    "鴛鴦鍋": ("火鍋", ["鴛鴦鍋"], ["鴛鴦"], False),
    "海鮮鍋": ("火鍋", ["海鮮鍋"], ["海鮮"], False),
    "個人小火鍋": ("火鍋", ["小火鍋", "個人鍋"], ["小火鍋", "個人鍋", "個人"], False),
    # 燒肉
    "韓式烤肉": ("燒肉", ["韓式烤肉", "韓國烤肉", "韓式燒肉"], ["韓式", "韓國", "韓"], False),
    # 日式料理
    "壽司": ("日式料理", ["壽司", "sushi"], ["壽司", "鮨", "sushi", "すし"], True),
    "丼飯": ("日式料理", ["丼飯", "丼"], ["丼"], True),
    "居酒屋": ("日式料理", ["居酒屋"], ["居酒屋"], True),
    "無菜單料理": ("日式料理", ["無菜單", "omakase"], ["無菜單", "omakase", "おまかせ"], True),
    "鰻魚飯": ("日式料理", ["鰻魚"], ["鰻"], True),
    "炸豬排": ("日式料理", ["炸豬排", "豬排"], ["豬排", "とんかつ"], True),
    "燒鳥": ("日式料理", ["燒鳥", "串燒"], ["燒鳥", "串燒", "串焼"], True),
    "鐵板燒": ("日式料理", ["鐵板燒"], ["鐵板燒"], True),
    "定食": ("日式料理", ["定食"], ["定食"], False),
    # 拉麵
    "沾麵": ("拉麵", ["沾麵"], ["沾麵", "つけ麺"], True),
    "豚骨拉麵": ("拉麵", ["豚骨"], ["豚骨"], False),
    "雞白湯拉麵": ("拉麵", ["雞白湯"], ["雞白湯", "鶏白湯"], False),
    # 義式料理
    "披薩": ("義式料理", ["披薩", "pizza"], ["披薩", "pizza"], True),
    "義大利麵": ("義式料理", ["義大利麵", "pasta"], ["義大利麵", "pasta", "麵"], False),
    # 美式餐廳
    "漢堡": ("美式餐廳", ["漢堡", "burger"], ["漢堡", "burger"], True),
    "牛排": ("美式餐廳", ["牛排", "steak"], ["牛排", "steak"], True),
    "炸雞": ("美式餐廳", ["炸雞"], ["炸雞", "chicken"], False),
    # 韓式料理
    "韓式炸雞": ("韓式料理", ["韓式炸雞", "韓國炸雞"], ["炸雞", "chicken"], False),
    "部隊鍋": ("韓式料理", ["部隊鍋"], ["部隊鍋"], True),
    # 中式餐廳
    "港式": ("中式餐廳", ["港式", "飲茶", "茶餐廳", "港點"], ["港式", "茶餐廳", "飲茶", "點心", "粵"], True),
    "台菜": ("中式餐廳", ["台菜", "台式料理"], ["台菜", "台式"], False),
    "熱炒": ("中式餐廳", ["熱炒", "快炒"], ["熱炒", "快炒"], True),
    "川菜": ("中式餐廳", ["川菜", "四川菜", "川味"], ["川菜", "川味", "四川", "成都", "重慶"], False),
    "江浙菜": ("中式餐廳", ["江浙菜", "上海菜"], ["江浙", "上海", "杭州"], False),
    "烤鴨": ("中式餐廳", ["烤鴨"], ["烤鴨"], True),
    # 平價小吃
    "牛肉麵": ("平價小吃", ["牛肉麵"], ["牛肉麵"], True),
    "滷肉飯": ("平價小吃", ["滷肉飯", "魯肉飯"], ["滷肉飯", "魯肉飯"], True),
    "水餃": ("平價小吃", ["水餃", "餃子", "鍋貼"], ["水餃", "餃子", "鍋貼"], True),
    "臭豆腐": ("平價小吃", ["臭豆腐"], ["臭豆腐"], True),
    # 甜點
    "蛋糕": ("甜點", ["蛋糕"], ["蛋糕", "cake"], True),
    "冰品": ("甜點", ["冰品", "剉冰", "雪花冰", "刨冰"], ["冰"], False),
    "豆花": ("甜點", ["豆花"], ["豆花"], True),
    "鬆餅": ("甜點", ["鬆餅"], ["鬆餅", "pancake", "waffle"], True),
    # 酒吧
    "精釀啤酒": ("酒吧", ["精釀", "啤酒"], ["精釀", "啤酒", "beer", "brew"], False),
}

# 總預算的說法：「總預算1500」「總共1500」「全部加起來 2000」→ 會除以人數換算成每人預算
TOTAL_RE = r"(?:總預算|預算總共|預算一共|預算合計|全部加起來|加起來|總共|總計|一共|合計)"
BUDGET_FILLER = r"\s*(?:大概|大約|約|差不多|抓|在|是|最多|頂多|上限|不超過|：|:)*\s*"
# 「推薦別的」：沿用上一次的條件，排除已經推薦過的店
MORE_WORDS = ["別的", "別家", "別間", "換一批", "換一些", "換幾間", "換幾家", "其他選擇", "其他的",
              "其他間", "其他家", "還有嗎", "還有沒有", "還有其他", "再推薦", "再給我", "下一批",
              "更多選擇", "不要這些", "不要這幾"]
# 「有什麼類型可以選」：列出料理類型
CATEGORY_QUESTION_RE = (r"(?:什麼|甚麼|哪些|哪幾種|哪種|啥|有什麼|有甚麼).{0,4}(?:類型|種類|類別|料理|選擇|可以吃|能吃|好吃的)"
                        r"|(?:類型|種類|類別|料理).{0,3}(?:有哪些|有什麼|有甚麼|可以選)")


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
        # 貼上金鑰時常會多帶空白、換行或「Bearer 」，先清掉
        key = self.valves.LLM_API_KEY.strip()
        key = re.sub(r"^Bearer\s+", "", key, flags=re.IGNORECASE)
        return {"Authorization": f"Bearer {key}"}

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
        d = self.parse_with_llm_full(messages, self.last_recommended(messages))
        return d["district"], d["category"], d["party_size"], d["budget"]

    def parse_with_llm_full(self, messages: list, recent_names: list) -> dict:
        """用 LLM 解析條件、意圖（search/more/reject/chat）與不喜歡的店。"""
        history = []
        for m in messages[-12:]:
            content = m.get("content", "")
            if isinstance(content, str) and m.get("role") in ("user", "assistant"):
                # 助理的長篇回答只保留開頭（含先前解析的條件），節省 token
                history.append({"role": m["role"], "content": content[:300]})

        system = PARSE_PROMPT
        if recent_names:
            system += "\n最近推薦過的店：\n" + "\n".join(f"{i}. {n}" for i, n in enumerate(recent_names, 1))
        raw = self._chat([{"role": "system", "content": system}] + history, max_tokens=300)
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

        intent = data.get("intent") if data.get("intent") in ("search", "more", "reject", "chat") else "search"
        reject = [n for n in (data.get("reject") or []) if n in recent_names]
        return {"district": district, "category": category,
                "party_size": to_int(data.get("party_size")), "budget": to_int(data.get("budget")),
                "intent": intent, "reject": reject}

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

    def get_conditions(self, messages: list):
        """回傳 (條件 tuple, 解析方式, info)。
        info：total（總預算）、subtype（子類型）、intent / reject（LLM 判斷的意圖與不喜歡的店）、
              llm_note（LLM 沒用上的原因，顯示在條件行，方便除錯）。
        總預算會除以人數，換算成每人預算。"""
        conditions, source = None, "規則"
        info = {"intent": None, "reject": [], "llm_note": ""}
        if self.valves.USE_LLM_PARSER and not self._llm_enabled():
            info["llm_note"] = "未設定 LLM 金鑰"
        elif self.valves.USE_LLM_PARSER:
            try:
                d = self.parse_with_llm_full(messages, self.last_recommended(messages))
                conditions, source = (d["district"], d["category"], d["party_size"], d["budget"]), "LLM"
                info.update(intent=d["intent"], reject=d["reject"])
            except requests.exceptions.HTTPError as e:
                info["llm_note"] = f"LLM 回應錯誤 HTTP {e.response.status_code}"
            except requests.exceptions.RequestException:
                info["llm_note"] = "連不到 LLM 服務"
            except Exception:
                info["llm_note"] = "LLM 回傳格式不對"
        if conditions is None:
            conditions = self.parse_rounds(messages)
        # 使用者說「不限地區」「預算不限」→ 那一項清空
        texts, _ = self.current_round(messages)
        waived = self.waived_fields(texts)
        conditions = tuple(None if f in waived else v for f, v in zip(FIELDS, conditions))

        # 總預算：有人數就換算成每人；還不知道人數就先不填，等問完人數再算
        total = self.total_budget(texts)
        if total:
            district, category, party_size, _ = conditions
            budget = total // party_size if party_size else None
            conditions = (district, category, party_size, budget)

        # 子類型（例如「壽喜燒」）：這次需求有提到就用；沒提到就沿用上一次的（類別沒換的話）
        district, category, party_size, budget = conditions
        subtype = None
        for text in reversed(texts):
            subtype = self.detect_subtype(text, category)
            if subtype:
                break
        if subtype is None:
            prev = self.previous_subtype(messages)
            if prev and SUBTYPES[prev][0] == category:
                subtype = prev
        if subtype:
            parent = SUBTYPES[subtype][0]
            if category in (None, parent):
                category = parent  # 「想吃壽喜燒」→ 類別是火鍋
            elif any(category in t for t in texts[-1:]):
                subtype = None  # 這次明確說了別的類別（例如「換成拉麵」）
            else:
                category = parent
        conditions = (district, category, party_size, budget)
        info.update(total=total, subtype=subtype)
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

    def should_ask(self, messages: list, conditions, total=None) -> list:
        """回傳還要問的欄位；空 list 代表資訊足夠，可以推薦。"""
        if not self.valves.ASK_FOLLOWUP:
            return []
        texts, asks = self.current_round(messages)
        if not texts or self.skip_all(texts[-1]):
            return []
        if asks >= self.valves.MAX_ASK:
            return []  # 已經問很多次了，用目前的條件推薦
        waived = self.waived_fields(texts)
        if total:
            waived.add("budget")  # 已經有總預算，問到人數就能換算
        return [f for f, v in zip(FIELDS, conditions) if v is None and f not in waived]

    def ask_question(self, messages: list, conditions, missing: list, total=None, subtype=None) -> str:
        texts, asks = self.current_round(messages)
        waived = self.waived_fields(texts)
        known = []
        for field, value in zip(FIELDS, conditions):
            if field == "budget" and total:
                known.append(f"總預算 {total} 元")
            elif field == "category" and value and subtype:
                known.append(f"{value}・{subtype}")
            elif value is not None:
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
            url = self.valves.API_URL.rsplit("/", 1)[0] + "/categories"
            r = requests.post(url, json={"district": district, "party_size": party_size,
                                         "budget": budget}, timeout=10)
            counts = {c["category"]: c["count"] for c in r.json().get("categories", [])}
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
    def shown_restaurants(messages: list, condition_line: str) -> list:
        """同一組搜尋條件下，之前已經推薦過的店名。"""
        names = []
        for m in messages[:-1]:
            content = m.get("content", "")
            if m.get("role") != "assistant" or not isinstance(content, str):
                continue
            if condition_line.split("（")[0] not in content:
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
        """這段對話裡使用者說過不喜歡／吃過的店（從「🚫 不再推薦：「A」「B」」讀回來）。"""
        names = []
        for m in messages[:-1]:
            content = m.get("content", "")
            if m.get("role") != "assistant" or not isinstance(content, str):
                continue
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
        picked = []
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
        if not result and strong and re.search(r"都|全部|這些|這幾|以上", text):
            result = list(names)
        result = list(dict.fromkeys(result))  # 去重、保留順序
        return result, (strong or bool(result))

    def ask_which(self, names: list) -> str:
        lines = [f"{ASK_MARK}是哪一間呢？可以直接說「第 2 間」或店名，我就不會再推薦它："]
        lines += [f"{i}. {n}" for i, n in enumerate(names, 1)]
        return "\n".join(lines)

    def chat_reply(self, messages: list) -> Optional[str]:
        """不是要新推薦的訊息（道謝、閒聊、問細節）→ 用 LLM 自然回覆。"""
        recent = ""
        for m in reversed(messages[:-1]):
            content = m.get("content", "")
            if m.get("role") == "assistant" and isinstance(content, str) \
                    and not content.startswith((ASK_MARK, CHAT_MARK)):
                recent = content[:2500]
                break
        history = [{"role": m["role"], "content": str(m.get("content", ""))[:500]}
                   for m in messages[-6:] if m.get("role") in ("user", "assistant")]
        try:
            return self._chat([{"role": "system", "content": CHAT_PROMPT + "\n最近的推薦內容：\n" + recent}]
                              + history, max_tokens=300)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 推薦理由（每家一段，寫餐廳特色）
    # ------------------------------------------------------------------
    def write_reasons(self, texts: list, conditions, restaurants: list, subtype=None, notes=None):
        """回傳 (開場白, 每家的推薦理由 list)。LLM 不能用時，用資料中的事實組成理由。"""
        fallback = [("；".join(r.get("facts") or []) + "。") if r.get("facts") else ""
                    for r in restaurants]
        if not (self.valves.USE_LLM_REASON and self._llm_enabled() and restaurants):
            return None, fallback
        district, category, party_size, budget = conditions
        need = "；".join(texts[-4:])
        cond = (f"地區={district or '不限'}，類型={category or '不限'}，"
                f"人數={party_size or '不限'}，每人預算上限={budget or '不限'}")
        if subtype:
            cond += (f"，想吃的子類型={subtype}（subtype_match=true 代表店名看得出是{subtype}；"
                     f"false 的店不可以說它有{subtype}）")
        keys = ["name", "category", "min_price", "max_price", "subtype_match",
                "facts", "summary", "highlights", "reviews"]
        data = [{k: r.get(k) for k in keys if r.get(k) not in (None, "", [])} for r in restaurants]
        try:
            raw = self._chat(
                [
                    {"role": "system", "content": REASON_PROMPT},
                    {"role": "user", "content": f"使用者需求：{need}\n條件：{cond}\n"
                                                f"系統說明：{'；'.join(notes or []) or '無'}\n\n"
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
    def describe_conditions(district, category, party_size, budget, source, subtype=None, note="") -> str:
        parts = [
            district or "不限地區",
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

        user_text = messages[-1].get("content", "")
        user_text = user_text if isinstance(user_text, str) else str(user_text)

        # 1. 解析用戶條件
        conditions, source, info = self.get_conditions(messages)
        district, category, party_size, budget = conditions
        total, subtype = info["total"], info["subtype"]

        # 問「有什麼類型可以選」→ 列出料理類型
        if self.is_category_question(user_text):
            return self.list_categories(conditions)

        # 「不喜歡第 2 間」「吃過了」→ 找出是哪幾間
        last_names = self.last_recommended(messages)
        prev_msg = str(messages[-2].get("content", "")) if len(messages) >= 2 else ""
        asked_which = prev_msg.startswith(f"{ASK_MARK}是哪一間呢")
        rejects, reject_said = self.find_rejects(user_text, last_names, force=asked_which)
        if not rejects and info["intent"] == "reject":
            rejects = info["reject"]
        new_search = any(self.parse_user_query(user_text)) or self.detect_subtype(user_text, None)
        if reject_said and not rejects and last_names and not new_search:
            if self.wants_more(user_text):
                rejects = list(last_names)  # 「吃過了，換別的」→ 這批都不要
            else:
                return self.ask_which(last_names)

        # 不是要新推薦（道謝、閒聊、問細節）→ LLM 自然回覆
        if info["intent"] == "chat" and not rejects and not self.wants_more(user_text) and last_names:
            reply = self.chat_reply(messages)
            if reply:
                return CHAT_MARK + reply

        # 資訊不足 → 繼續反問
        missing = self.should_ask(messages, conditions, total)
        if missing:
            return self.ask_question(messages, conditions, missing, total, subtype)

        # 「推薦別的」→ 排除同樣條件下已經推薦過的店
        condition_line = self.describe_conditions(district, category, party_size, budget, source, subtype,
                                                  info["llm_note"])
        more = self.wants_more(user_text) or info["intent"] == "more"
        rejected = list(dict.fromkeys(self.rejected_history(messages) + rejects))
        history = self.shown_restaurants(messages, condition_line)
        if more:
            # 換一批：同條件下推薦過的都排除
            skipped = history
        elif rejects:
            # 只換掉不喜歡的那幾間：保留目前這批，更早幾批的照樣不要再出現
            skipped = [n for n in history if n not in last_names]
        else:
            skipped = []
        shown = [n for n in skipped if n not in rejected]  # 不喜歡的另外用 🚫 列，不重複計算
        exclude = list(dict.fromkeys(rejected + skipped))

        # 2. 呼叫推薦 API
        payload = {
            "district": district,
            "category": category,
            "party_size": party_size,
            "budget": budget,
            "exclude": exclude,
        }
        if subtype:
            _, _, keywords, cross = SUBTYPES[subtype]
            payload.update({"subtype": subtype, "subtype_keywords": keywords, "subtype_cross": cross})
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
                sections.append(condition_line)
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
                sections.append(f"⚠️ 沒有符合每人 {budget} 元預算的店，"
                                f"以下把預算稍微放寬到每人約 {relaxed} 元來找。")
            else:
                sections.append(f"⚠️ 符合每人 {budget} 元預算的店只有 {matched} 家，"
                                f"另外補上比預算稍微高一點（放寬到每人約 {relaxed} 元）的選擇。")
        # 子類型：誠實說明哪些店看得出是這種、哪些只是一般同類別的店
        matched_subtype = data.get("matched_subtype")
        general = self.shop_label(category)  # 「一般的火鍋店」「一般的美式餐廳」
        if subtype and matched_subtype is not None:
            where = district or "台北"
            shown_sub = sum(1 for r in restaurants if r.get("subtype_match"))
            if shown_sub == 0 and exclude and matched_subtype == 0:
                sections.append(f"⚠️ 店名看得出是「{subtype}」的店都推薦過了，"
                                f"以下是一般的{general}，不一定有{subtype}。")
            elif shown_sub == 0:
                sections.append(f"⚠️ 資料庫裡找不到{where}店名看得出是「{subtype}」的店，"
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
        # LLM 開場白：把上面的系統說明一起告訴 LLM，讓它自然地帶到
        texts, _ = self.current_round(messages)
        notes = [line for line in sections if line.startswith(("🚫", "🔄", "⚠️", "🏷️", "🧮"))]
        intro, reasons = self.write_reasons(texts, conditions, restaurants, subtype, notes)
        if intro:
            sections.append(f"💬 {intro}")
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
            blocks.append(self.format_card(i, r, reason))
        sections.append("\n\n---\n\n".join(blocks))
        if tips:
            sections.append(tips)
        return "\n\n".join(sections)

    @staticmethod
    def shop_label(category: Optional[str]) -> str:
        """「火鍋」→「火鍋店」，「美式餐廳」「酒吧」本身就是店，不用再加「店」。"""
        if not category:
            return "餐廳"
        return category if category.endswith(("餐廳", "店", "吧")) else category + "店"

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
