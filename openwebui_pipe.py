"""
title: Taipei Restaurant Recommendation Pipe
author: Taipei Food Bot
version: 3.4.0
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
    "日式料理": ["日本料理", "日料", "日式", "日本菜", "壽司", "生魚片", "丼飯", "居酒屋"],
    "拉麵": ["ramen"],
    "義式料理": ["義大利麵", "義大利菜", "義式", "意式", "披薩", "pizza", "燉飯"],
    "平價小吃": ["小吃", "夜市", "便當", "滷肉飯"],
    "中式餐廳": ["中菜", "合菜", "中式", "台菜", "熱炒", "港式", "粵菜"],
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
  "party_size": 整數或 null, "budget": 整數或 null, "reject": [店名, ...], "extra": {{...}}}}

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
8. 使用者可能打錯同音字或近音字，請依讀音判斷原本的意思，例如「大按」=大安、「信意」=信義、
   「伙鍋」=火鍋、「少肉」=燒肉、「拉免」=拉麵、「中孝復興」=忠孝復興、「預酸」=預算。
9. 預算給的是區間（「500 到 1000」）時，budget 填上限 1000；「1000 以上」時 budget 填 null。
10. extra 只看「使用者最後一句話」有沒有提到以下條件，沒提到的欄位就不要放：
   - "near"：捷運站或地標（例如「忠孝復興」「台北101」「西門町」），使用者想在那附近吃。
   - "no_chain"：true = 不要連鎖店；false = 說了連鎖也可以。
   - "hidden_gem"：true = 想要比較少人知道、隱藏版、私房的店；false = 說了有名的也可以。
   - "budget_min"：每人預算下限（「500 到 1000」的 500、「1000 以上」的 1000）。
   - "avoid_categories"：不想吃的類別（「剛吃過拉麵」「火鍋以外都可以」），只能用上面 category 的名稱。
   - "categories"：同時可以接受好幾種類別時（「日式或韓式都可以」），列出全部，只能用 category 的名稱。
   - "pick_one"：true = 使用者要你直接決定、只要一間。
   - "unsupported"：使用者在意、但只靠名稱和價格判斷不了的條件，例如營業時間、停車、包廂、
     寵物、素食、安靜、適合約會，用簡短的詞列出。
   格式例如 "extra": {{"near": "忠孝復興", "no_chain": true}}；都沒提到就填 {{}}。
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
SKIP_WORDS = ["都可以", "隨便", "都行", "沒差", "無所謂", "皆可", "直接推薦", "不用問",
              "你決定", "你來決定", "你幫我決定", "你幫我選", "你幫我挑"]
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

# 「日式」「韓式」後面直接接別的類別時（「日式燒肉」「韓式炸雞」），只是形容，不算另一個類別
STYLE_WORDS = {"日式", "韓式", "美式", "義式", "意式", "泰式", "中式", "台式", "港式", "日本", "韓國"}

# ---- 其他條件（推薦結果下方「🎛️ 其他條件：…」那一行會記住，下一句沿用）----
PREFS_MARK = "🎛️ 其他條件："
# 地標 → 最近的捷運站（MRT_STATIONS 在檔案最下面，由 tools/build_lookups.py 產生）
LANDMARKS = {
    "台北101": "台北101/世貿", "臺北101": "台北101/世貿", "101大樓": "台北101/世貿", "世貿": "台北101/世貿",
    "信義商圈": "市政府", "新光三越": "市政府", "微風南山": "台北101/世貿", "誠品信義": "市政府",
    "西門町": "西門", "紅樓": "西門", "士林夜市": "劍潭", "饒河夜市": "松山", "饒河街": "松山",
    "寧夏夜市": "雙連", "師大夜市": "台電大樓", "師大": "古亭", "通化夜市": "信義安和", "臨江街": "信義安和",
    "公館夜市": "公館", "台大": "公館", "臺大": "公館", "東區": "忠孝敦化", "永康街": "東門",
    "華山文創": "忠孝新生", "華山": "忠孝新生", "松菸": "國父紀念館", "松山文創": "國父紀念館",
    "大巨蛋": "國父紀念館", "小巨蛋": "台北小巨蛋", "南港展覽館": "南港展覽館", "南港展館": "南港展覽館",
    "南軟": "南港軟體園區", "內科": "西湖", "內湖科學園區": "西湖", "美麗華": "劍南路", "北投溫泉": "新北投",
    "迪化街": "北門", "大稻埕": "北門", "龍山寺": "龍山寺", "華西街": "龍山寺", "貓空": "動物園",
    "赤峰街": "中山", "南西": "中山", "中山商圈": "中山", "行天宮": "行天宮", "花博": "圓山",
    "北車": "台北車站", "台北車站": "台北車站", "臺北車站": "台北車站", "台北站": "台北車站",
    "京站": "台北車站", "象山": "象山", "大安森林公園": "大安森林公園", "中正紀念堂": "中正紀念堂",
    "國父紀念館": "國父紀念館", "松山機場": "松山機場", "市政府": "市政府", "市府": "市政府",
}
NEAR_WORDS = r"(?:站|附近|周邊|周圍|旁邊|一帶|那邊|這邊|商圈)"
# 跟行政區同名的站（大安、中山、士林…）一定要說「站」或「捷運」才算車站，否則當成行政區
COMMON_WORD_STATIONS = {"永春", "明德", "忠義", "文德", "幸福"}

# 連鎖店：「不要連鎖」→ 排除；「連鎖也可以」→ 取消
CHAIN_OFF_RE = r"(?:不要|不想|不吃|避開|別|排除|非)\s*(?:去)?\s*(?:吃)?\s*(?:連鎖|分店)|在地小店|獨立小店|個人小店"
CHAIN_OK_RE = r"連鎖(?:店|的)?\s*(?:也可以|也行|也沒關係|沒關係|無所謂|都可以|也ok|ok)"
# 隱藏版：「比較沒人知道的」「私房」→ 評分高但評論數少的店
HIDDEN_ON_RE = (r"沒(?:什麼)?人知道|很少人知道|比較少人|少人知道|冷門|隱藏版|隱藏的|私房|口袋名單|小眾|"
                r"內行人|在地人(?:才|會)?(?:知道|去)|不要(?:太)?(?:熱門|有名|觀光客)|人不要太多")
HIDDEN_OFF_RE = r"有名的(?:也可以|也行|就好|沒關係)|熱門的(?:也可以|也行|就好)|不用隱藏版|知名的(?:也可以|就好)"
# 不要某個類別：「不要火鍋」「剛吃過拉麵」「火鍋以外」「吃膩燒肉了」
NEG_BEFORE_RE = r"(?:不要|不想要|不想吃|不想|不吃|別吃|別|剛吃過|剛剛吃過|昨天吃過|吃過|吃膩了?|除了|不考慮|排除|避開)\s*(?:再)?\s*(?:吃)?\s*$"
NEG_AFTER_RE = r"^\s*(?:以外|之外|除外|吃膩|吃到膩|吃過了|剛吃過|不要|就不要|不吃|免了|先不要)"
# 「你決定」「給我一間就好」→ 只推薦 1 家
PICK_ONE_RE = r"你(?:來)?決定|你幫我(?:選|挑|決定)|幫我(?:選|挑)一(?:間|家)|(?:給我|推薦|只要|選)一(?:間|家)就好|一(?:間|家)就好|直接告訴我(?:去)?哪(?:一)?(?:間|家)"
# 推薦完之後問「你推薦哪一間？」→ 從剛剛那批裡挑一家
PICK_LAST_RE = r"(?:你|妳)?(?:最)?(?:推薦|推)哪(?:一)?(?:間|家)|哪(?:一)?(?:間|家)(?:最|比較)(?:好|推|值得|適合)|你會選哪|(?:從|在)(?:這|剛剛)(?:幾|些)?(?:間|家)?(?:裡|中|裡面)?(?:幫我)?(?:選|挑)"
# 「跟第 2 間差不多的」「類似肉老大的」→ 同類型、附近、價位相近
SIMILAR_RE = r"差不多|類似|像是|很像|一樣的|同類型|同一種|相似"
# 目前資料沒有的條件 → 誠實說明沒辦法篩選
UNSUPPORTED = {
    "營業時間": ["營業時間", "有開", "還開", "開到", "營業到", "幾點", "宵夜", "消夜", "深夜", "半夜", "凌晨",
             "現在開", "星期一", "週一", "禮拜一", "公休"],
    "停車": ["停車"],
    "包廂": ["包廂"],
    "訂位": ["訂位", "預約", "訂得到"],
    "寵物友善": ["寵物", "帶狗", "帶貓", "毛小孩"],
    "親子設施": ["兒童椅", "嬰兒椅", "親子", "帶小孩", "小朋友", "推車"],
    "素食": ["素食", "吃素", "蔬食", "全素", "蛋奶素"],
    "辣度": ["不吃辣", "不能吃辣", "不辣", "怕辣"],
    "氣氛": ["安靜", "氣氛", "浪漫", "約會", "景觀", "夜景", "聊天", "拍照", "網美"],
    "外帶外送": ["外帶", "外送", "外賣"],
    "插座／Wi-Fi": ["插座", "wifi", "wi-fi"],
    "無障礙": ["無障礙", "輪椅"],
    "服務費": ["服務費"],
}
# 天氣、心情 → 反問類型時給建議
MOOD_HINTS = [
    (r"天氣冷|好冷|很冷|變冷|寒流|下雨|暖身|想吃熱的|熱呼呼|熱湯", "天氣冷的話，火鍋、拉麵、薑母鴨、羊肉爐都很適合"),
    (r"好熱|很熱|太熱|熱死|清爽|消暑|冰的", "天氣熱的話，甜點（冰品、豆花）或日式料理比較清爽"),
    (r"趕時間|很急|快一點|半小時|沒什麼時間|速戰速決", "趕時間的話，平價小吃、拉麵、早午餐出餐比較快"),
    (r"慶生|生日|紀念日|請客|約會|求婚|宴客", "慶祝或請客的話，可以考慮燒肉、日式料理、義式料理，記得先訂位"),
    (r"喝一杯|喝酒|小酌|續攤", "想喝一杯的話可以選酒吧，或居酒屋（日式料理）"),
]
# 錯字修正時不能動的常用字（避免「今日是」被改成「今日式」）
FUNCTION_CHARS = set("是的在有一不了我他她你妳這那就也都和要會上下個人們來去說吧嗎呢啊喔")
# 讀音相近、但本身就是常用詞的，不改（例如「美食」不是「美式」、「大致上」不是「大直」）
PROTECTED_WORDS = ["美食", "大致", "大志", "大智", "大治", "大指", "大隻", "中彩", "種菜", "石林",
                   "角子", "轎子", "竹排", "舔食", "南戲", "治山", "大概", "時事", "實施", "事實"]


def _is_cjk(text: str) -> bool:
    return bool(text) and all("一" <= ch <= "鿿" for ch in text)


def homophone_terms(stations=None) -> set:
    """同音錯字要比對的詞：地區、類別、同義詞、子類型、捷運站、地標。"""
    terms = {d.replace("區", "") for d in DISTRICTS} | set(DISTRICTS) | set(CATEGORIES)
    for words in CATEGORY_SYNONYMS.values():
        terms |= {w for w in words if _is_cjk(w)}
    for _, triggers, _, _ in SUBTYPES.values():
        terms |= {t for t in triggers if _is_cjk(t)}
    for s in (stations if stations is not None else MRT_STATIONS):
        terms |= {p for p in s.split("/") if _is_cjk(p)}
    terms |= {k for k in LANDMARKS if _is_cjk(k)}
    terms |= {"預算", "連鎖", "隱藏版", "附近", "捷運"}
    return {t for t in terms if len(t) >= 2}


_HOMOPHONE_CACHE = {}


def _homophone_tables():
    """（字 → 讀音集合, [(詞, 讀音 tuple)]），第一次用到時才建立。"""
    if not _HOMOPHONE_CACHE:
        char_readings = {}
        for reading, chars in PINYIN_CHARS.items():
            for ch in chars:
                char_readings.setdefault(ch, set()).add(reading)
        terms = []
        for term in homophone_terms():
            reading = TERM_READINGS.get(term)
            if reading:
                terms.append((term, tuple(reading.split())))
            elif all(ch in char_readings for ch in term):
                terms.append((term, tuple(sorted(char_readings[ch])[0] for ch in term)))
        terms.sort(key=lambda t: -len(t[0]))
        _HOMOPHONE_CACHE.update(chars=char_readings, terms=terms)
    return _HOMOPHONE_CACHE["chars"], _HOMOPHONE_CACHE["terms"]


def fix_homophones(text: str):
    """把同音錯字改回正確的詞，例如「大按」→「大安」、「伙鍋」→「火鍋」、「中孝復興」→「忠孝復興」。
    回傳 (修正後的文字, [(原本的字, 改成的詞), ...])。
    比對讀音時不管聲調（「大按」「拉免」也算）。為了避免誤改：至少要有一個字完全一樣、
    不同的字不超過一半、不改「是、的、有」這類常用字、不改 PROTECTED_WORDS 裡的常用詞。"""
    if not text or not PINYIN_CHARS:
        return text, []
    char_readings, terms = _homophone_tables()
    chars = list(text)
    covered = [False] * len(chars)
    for term in [t for t, _ in terms] + PROTECTED_WORDS:  # 已經寫對的詞、常用詞先保護起來
        start = text.find(term)
        while start != -1:
            for k in range(start, start + len(term)):
                covered[k] = True
            start = text.find(term, start + 1)
    fixes = []
    for term, reading in terms:
        size = len(term)
        for i in range(len(chars) - size + 1):
            window = chars[i:i + size]
            diff = [j for j in range(size) if window[j] != term[j]]
            if not diff or len(diff) > max(1, size // 2):
                continue
            if any(covered[i + j] for j in diff):
                continue  # 要改的字屬於一個已經寫對的詞（「麻辣國」的「麻辣」沒關係，「國」可以改）
            ok = True
            for j in diff:
                got = char_readings.get(window[j], set())
                ok = window[j] not in FUNCTION_CHARS and                     reading[j].rstrip("012345") in {r.rstrip("012345") for r in got}
                if not ok:
                    break
            if ok:
                fixes.append(("".join(window), term))
                chars[i:i + size] = list(term)
                for k in range(i, i + size):
                    covered[k] = True
    return "".join(chars), fixes


def find_near(text: str):
    """找出「捷運站／地標附近」。回傳 (車站名, 使用者說的地標或站名) 或 None。"""
    lower = text.lower()
    for landmark in sorted(LANDMARKS, key=len, reverse=True):
        if landmark.lower() in lower:
            return LANDMARKS[landmark], landmark
    for station in sorted(MRT_STATIONS, key=len, reverse=True):
        for key in [p for p in station.split("/") if len(p) >= 2]:
            start = text.find(key)
            if start == -1:
                continue
            after = text[start + len(key):start + len(key) + 3]
            before = text[max(0, start - 3):start]
            explicit = after.startswith("站") or "捷運" in before
            near_word = re.match(NEAR_WORDS, after) is not None
            if key + "區" in DISTRICTS or key in COMMON_WORD_STATIONS:
                if explicit:  # 「大安站」「捷運中山」才算車站，「大安」是行政區
                    return station, key
            elif explicit or near_word or len(key) >= 3 or after[:1] not in ("路", "街", "國", "里"):
                return station, key  # 「公館吃火鍋」：2 個字的站名單獨出現也算（「公館路」不算）
    return None


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
        choice = r.json()["choices"][0]
        text = choice.get("message", {}).get("content") or ""
        # 移除思考內容：有些模板只輸出結尾的 </think>，所以取最後一個 </think> 之後的文字
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[1]
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()  # 思考到一半被截斷
        if not text and choice.get("finish_reason") == "length":
            raise ValueError("LLM 回應被截斷（可能在思考）")
        return text

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

    def parse_with_llm_full(self, messages: list, recent_names: list) -> dict:
        """用 LLM 解析條件、意圖（search/more/reject/chat）與不喜歡的店。"""
        # 把對話整理成「一段文字」交給 LLM，而不是原樣當成多輪對話：
        # 否則模型會模仿助理先前的回答（「🔎 搜尋條件：…」），不回 JSON。
        lines = []
        for m in messages[-12:]:
            content = m.get("content", "")
            role = m.get("role")
            if not isinstance(content, str) or role not in ("user", "assistant"):
                continue
            if role == "user":
                lines.append(f"使用者：{content[:300]}")
            elif content.startswith(ASK_MARK):
                lines.append(f"助理（反問）：{content[:200]}")
            elif content.startswith(CHAT_MARK):
                lines.append(f"助理（閒聊）：{content[len(CHAT_MARK):][:150]}")
            else:
                # 推薦結果只留先前的條件，其餘省略
                cond = re.search(r"🔎 搜尋條件：([^（\n]+)", content)
                extra = next((l[len(PREFS_MARK):] for l in content.splitlines() if l.startswith(PREFS_MARK)), "")
                lines.append(f"助理：（推薦了餐廳；當時的條件是 {cond.group(1).strip() if cond else '未知'}"
                             f"{'；其他條件：' + extra if extra else ''}）")

        system = PARSE_PROMPT
        if recent_names:
            system += "\n最近推薦過的店：\n" + "\n".join(f"{i}. {n}" for i, n in enumerate(recent_names, 1))
        user = ("以下是對話紀錄，最後一行是使用者這次說的話：\n\n" + "\n".join(lines)
                + "\n\n請依規則只輸出 JSON 物件，第一個字元必須是 {。")
        raw = self._chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                         max_tokens=800)
        data = self._extract_json(raw)

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
                "intent": intent, "reject": reject, "prefs": self.clean_llm_prefs(data.get("extra"), to_int)}

    @staticmethod
    def clean_llm_prefs(extra, to_int) -> dict:
        """檢查 LLM 回傳的其他條件，不合規定的丟掉（例如不存在的車站、類別）。"""
        if not isinstance(extra, dict):
            return {}
        p = {}
        near = extra.get("near")
        if isinstance(near, str) and near.strip():
            found = find_near(near.strip()) or find_near(near.strip() + "站")
            if found:
                p["near"], p["near_label"] = found
        for k in ("no_chain", "hidden_gem", "pick_one"):
            if isinstance(extra.get(k), bool):
                p[k] = extra[k]
        if to_int(extra.get("budget_min")):
            p["budget_min"] = to_int(extra.get("budget_min"))
        for src, dst in (("avoid_categories", "avoid"), ("categories", "cats")):
            cats = [c for c in (extra.get(src) or []) if c in CATEGORIES]
            if cats and (dst == "avoid" or len(cats) >= 2):
                p[dst] = list(dict.fromkeys(cats))
        unsupported = [str(u)[:10] for u in (extra.get("unsupported") or []) if str(u).strip()]
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
        """這句話「有提到」的其他條件（沒提到的不會出現在結果裡）。"""
        t = fix_homophones(text)[0]
        lower = t.lower()
        p = {}
        near = find_near(t)
        if near:
            p["near"], p["near_label"] = near
        if re.search(CHAIN_OK_RE, lower):
            p["no_chain"] = False
        elif re.search(CHAIN_OFF_RE, t):
            p["no_chain"] = True
        if re.search(HIDDEN_OFF_RE, t):
            p["hidden_gem"] = False
        elif re.search(HIDDEN_ON_RE, t):
            p["hidden_gem"] = True
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
            p["avoid"] = avoid
        if want:
            p["want"] = want
        if len(want) >= 2:
            p["cats"] = want  # 「日式或韓式都可以」
        if re.search(PICK_ONE_RE, t):
            p["pick_one"] = True
        unsupported = [label for label, ws in UNSUPPORTED.items() if any(w in lower for w in ws)]
        if unsupported:
            p["unsupported"] = unsupported
        return p

    def find_similar(self, text: str, names: list) -> Optional[str]:
        """「跟第 2 間差不多的」「類似肉老大的」→ 那家店的店名。"""
        if not names or not re.search(SIMILAR_RE, text):
            return None
        picked = self.match_names(text, names)
        return picked[0] if picked else None

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
            similar = re.search(r"和「(.+?)」類似", cond.group(1))
            if similar:
                p["similar"] = similar.group(1)
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
            return p
        return {}

    def merge_prefs(self, messages: list, texts: list) -> dict:
        """上一次的其他條件 + 這次需求裡每一句話的修改。"""
        prefs = self.previous_prefs(messages)
        names = self.last_recommended(messages)
        for text in texts:
            p = self.parse_prefs(text)
            district = self.parse_user_query(text)[0]
            similar = self.find_similar(text, names)
            if similar:
                prefs["similar"] = similar
                prefs.pop("near", None)
            elif "near" in p:
                prefs["near"], prefs["near_label"] = p["near"], p["near_label"]
                prefs.pop("similar", None)
            elif district:  # 改說行政區 → 不再是「某站附近」
                for k in ("near", "near_label", "similar"):
                    prefs.pop(k, None)
            for k in ("no_chain", "hidden_gem", "budget_min"):
                if k in p:
                    prefs[k] = p[k]
            if p.get("want"):
                prefs["avoid"] = [c for c in prefs.get("avoid", []) if c not in p["want"]]
                prefs["cats"] = p.get("cats")
                if not similar:
                    prefs.pop("similar", None)
            if p.get("avoid"):
                prefs["avoid"] = list(dict.fromkeys(prefs.get("avoid", []) + p["avoid"]))
        waived = self.waived_fields(texts)
        if "district" in waived:
            for k in ("near", "near_label", "similar"):
                prefs.pop(k, None)
        if "budget" in waived:
            prefs.pop("budget_min", None)
        if "category" in waived:
            prefs.pop("cats", None)
        # 只對這一句有效的：只要一間、資料沒有的條件
        last = self.parse_prefs(texts[-1]) if texts else {}
        for k in ("pick_one", "budget_max"):
            if last.get(k):
                prefs[k] = last[k]
        unsupported = [u for t in texts for u in self.parse_prefs(t).get("unsupported", [])]
        if unsupported:
            prefs["unsupported"] = list(dict.fromkeys(unsupported))
        return {k: v for k, v in prefs.items() if v not in (None, False, [], "")}

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
        return PREFS_MARK + "｜".join(parts) if parts else ""

    def pick_from_last(self, messages: list, user_text: str) -> str:
        """推薦完之後問「你推薦哪一間？」→ 從剛剛那批挑一家。"""
        recent = ""
        for m in reversed(messages[:-1]):
            content = m.get("content", "")
            if m.get("role") == "assistant" and isinstance(content, str) \
                    and not content.startswith((ASK_MARK, CHAT_MARK)):
                recent = content
                break
        card = re.search(r"\*\*1\. (.+?)\*\*\n(.*?)(?:\n\n|\Z)", recent, flags=re.DOTALL)
        if not card:
            return CHAT_MARK + "還沒有推薦過餐廳喔，先告訴我想在哪裡吃、吃什麼吧！"
        if self._llm_enabled():
            try:
                reply = self._chat([
                    {"role": "system", "content": CHAT_PROMPT
                     + "\n使用者請你從最近推薦的店裡挑一間。只能從清單裡挑，根據資料說明理由，1~3 句。"
                     + "\n最近的推薦內容：\n" + recent[:2500]},
                    {"role": "user", "content": user_text},
                ], max_tokens=300)
                if reply:
                    return CHAT_MARK + reply
            except Exception:
                pass
        reason = re.search(r"✨ (.+)", card.group(2))
        return (f"{CHAT_MARK}如果只選一間，我會選第 1 間「{card.group(1)}」"
                f"{'：' + reason.group(1) if reason else '。'}\n"
                f"它是這批裡綜合評分、評論數和預算最符合的。想看其他選擇可以說「推薦別的」。")

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
                info.update(intent=d["intent"], reject=d["reject"], llm_prefs=d.get("prefs", {}))
            except requests.exceptions.HTTPError as e:
                info["llm_note"] = f"LLM 回應錯誤 HTTP {e.response.status_code}"
            except requests.exceptions.RequestException:
                info["llm_note"] = "連不到 LLM 服務"
            except Exception as e:
                # 附上一小段原因（LLM 實際回了什麼），方便除錯
                detail = re.sub(r"[\s（）()｜]+", " ", str(e)).strip()[:50]
                info["llm_note"] = f"LLM 回傳格式不對：{detail}" if detail else "LLM 回傳格式不對"
                print(f"[taipei_food_bot] LLM parse failed: {e!r}")
        if conditions is None:
            conditions = self.parse_rounds(messages)
        # 使用者說「不限地區」「預算不限」→ 那一項清空
        texts, _ = self.current_round(messages)
        waived = self.waived_fields(texts)
        conditions = tuple(None if f in waived else v for f, v in zip(FIELDS, conditions))
        fixed_texts = [fix_homophones(t)[0] for t in texts]
        info["fixes"] = fix_homophones(texts[-1])[1] if texts else []

        # 總預算：有人數就換算成每人；還不知道人數就先不填，等問完人數再算
        total = self.total_budget(fixed_texts)
        if total:
            district, category, party_size, _ = conditions
            budget = total // party_size if party_size else None
            conditions = (district, category, party_size, budget)

        # 子類型（例如「壽喜燒」）：這次需求有提到就用；沒提到就沿用上一次的（類別沒換的話）
        district, category, party_size, budget = conditions
        subtype = None
        for text in reversed(fixed_texts):
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
            elif any(category in t for t in fixed_texts[-1:]):
                subtype = None  # 這次明確說了別的類別（例如「換成拉麵」）
            else:
                category = parent

        # 其他條件（捷運站附近、不要連鎖、隱藏版、預算下限、不要的類別、多類別、類似某間）
        prefs = self.merge_prefs(messages, texts)
        for k, v in (info.get("llm_prefs") or {}).items():
            prefs.setdefault(k, v)  # 規則沒抓到、LLM 有抓到的才補上
        if prefs.get("budget_max"):
            budget = prefs["budget_max"]  # 「500 到 1000」→ 上限 1000
        if prefs.get("budget_min") and budget and budget <= prefs["budget_min"]:
            budget = None  # 「1000 以上」：1000 是下限，不是上限
        if prefs.get("cats"):
            category = prefs["cats"][0]
        if category in prefs.get("avoid", []):
            category, subtype = None, None  # 「不要火鍋」→ 不能再找火鍋
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
        """其他條件已經回答了的欄位：「忠孝復興附近」→ 地區；「不要火鍋」→ 類型；「1000 以上」→ 預算。"""
        prefs = prefs or {}
        done = set()
        if prefs.get("near") or prefs.get("similar"):
            done.add("district")
        if prefs.get("similar") or prefs.get("avoid") or prefs.get("cats"):
            done.add("category")
        if prefs.get("similar") or prefs.get("budget_min"):
            done.add("budget")
        return done

    def should_ask(self, messages: list, conditions, total=None, prefs=None) -> list:
        """回傳還要問的欄位；空 list 代表資訊足夠，可以推薦。"""
        if not self.valves.ASK_FOLLOWUP:
            return []
        texts, asks = self.current_round(messages)
        if not texts or self.skip_all(texts[-1]):
            return []
        if asks >= self.valves.MAX_ASK:
            return []  # 已經問很多次了，用目前的條件推薦
        waived = self.waived_fields(texts) | self.answered_by_prefs(prefs)
        if total:
            waived.add("budget")  # 已經有總預算，問到人數就能換算
        return [f for f, v in zip(FIELDS, conditions) if v is None and f not in waived]

    def ask_question(self, messages: list, conditions, missing: list, total=None, subtype=None,
                     prefs=None) -> str:
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

        if asks == 0:
            opening = "想幫你找到更適合的餐廳，可以再多告訴我一些嗎？"
        else:
            opening = "了解！還差一點點資訊，我就能幫你挑出最合適的餐廳："
        lines = [f"{ASK_MARK}{opening}"]
        extra = self.prefs_line(prefs)[len(PREFS_MARK):]
        if extra:
            known.append(extra)
        if known:
            lines.append(f"目前知道：{'｜'.join(known)}")
        lines.append("")
        for i, field in enumerate(missing, 1):
            lines.append(f"{i}. {QUESTIONS[field]}")
            if field == "category":
                # 天氣、心情、場合 → 給點建議
                hints = [hint for pattern, hint in MOOD_HINTS if any(re.search(pattern, t) for t in texts)]
                if hints:
                    lines.append(f"   💡 {hints[0]}")
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
        result = Pipe.match_names(text, names)
        if not result and strong and re.search(r"都|全部|這些|這幾|以上", text):
            result = list(names)
        return result, (strong or bool(result))

    @staticmethod
    def match_names(text: str, names: list) -> list:
        """句子裡指到的店：「第 2 間」「3 號」「前兩間」、店名或店名開頭。"""
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
        return list(dict.fromkeys(result))  # 去重、保留順序

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
        keys = ["name", "category", "min_price", "max_price", "subtype_match", "distance",
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
        if prefs.get("similar"):
            where = f"和「{prefs['similar']}」類似"
        elif prefs.get("near"):
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

        prefs = info["prefs"]
        said = self.parse_prefs(user_text)  # 這一句提到的其他條件

        # 問「有什麼類型可以選」→ 列出料理類型
        if self.is_category_question(user_text) and not (said.get("near") or said.get("hidden_gem")):
            return self.list_categories(conditions)

        last_names = self.last_recommended(messages)
        new_search = (any(self.parse_user_query(user_text))
                      or self.detect_subtype(fix_homophones(user_text)[0], None)
                      or any(k in said for k in ("near", "avoid", "want", "no_chain", "hidden_gem", "budget_min"))
                      or self.find_similar(user_text, last_names))

        # 推薦完之後問「你推薦哪一間？」→ 從剛剛那批挑一家
        if last_names and not new_search and re.search(PICK_LAST_RE, user_text):
            return self.pick_from_last(messages, user_text)

        # 「不喜歡第 2 間」「吃過了」→ 找出是哪幾間
        prev_msg = str(messages[-2].get("content", "")) if len(messages) >= 2 else ""
        asked_which = prev_msg.startswith(f"{ASK_MARK}是哪一間呢")
        rejects, reject_said = self.find_rejects(user_text, last_names, force=asked_which)
        if prefs.get("similar") and prefs["similar"] in rejects:
            rejects = [n for n in rejects if n != prefs["similar"]]  # 「跟第 2 間差不多」不是不要第 2 間
            reject_said = False
        if not rejects and info["intent"] == "reject":
            rejects = info["reject"]
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
        missing = self.should_ask(messages, conditions, total, prefs)
        if missing:
            return self.ask_question(messages, conditions, missing, total, subtype, prefs)

        # 「推薦別的」→ 排除同樣條件下已經推薦過的店
        condition_line = self.describe_conditions(district, category, party_size, budget, source, subtype,
                                                  info["llm_note"], prefs)
        extra_line = self.prefs_line(prefs)
        more = self.wants_more(user_text) or info["intent"] == "more"
        rejected = list(dict.fromkeys(self.rejected_history(messages) + rejects))
        history = self.shown_restaurants(messages, condition_line, extra_line)
        if self.find_similar(user_text, last_names):
            # 「跟第 2 間差不多的」：剛剛那批都看過了，找別家
            skipped = list(last_names)
        elif more:
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
        payload.update({
            "near": prefs.get("near"),
            "similar_to": prefs.get("similar"),
            "categories": prefs.get("cats"),
            "avoid_categories": prefs.get("avoid"),
            "no_chain": bool(prefs.get("no_chain")),
            "hidden_gem": bool(prefs.get("hidden_gem")),
            "budget_min": prefs.get("budget_min"),
            "top_n": 1 if prefs.get("pick_one") else 5,
        })
        try:
            response = requests.post(self.valves.API_URL, json=payload, timeout=10)
            if response.status_code != 200:
                return f"❌ API 回應異常 (HTTP {response.status_code})"
            data = response.json()
        except Exception as e:
            return f"❌ 無法連線至本機推薦 API 服務，請確認 `python api.py` 是否已啟動。\n錯誤訊息: {e}"

        restaurants = data.get("restaurants", [])
        where = self.near_text(prefs) if prefs.get("near") else district
        tips = self.describe_alternatives(conditions, len(restaurants), data.get("alternatives") or [], where)
        if data.get("similar"):
            # 「跟某間差不多」：顯示那家店的類別（下一句「推薦別的」會沿用）
            category = data["similar"]["category"]
            budget = budget or data["similar"].get("budget")
            conditions = (district, category, party_size, budget)
            condition_line = self.describe_conditions(district, category, party_size, budget, source, subtype,
                                                      info["llm_note"], prefs)
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
        notes = [line for line in sections
                 if line.startswith(("🚫", "🔄", "⚠️", "🏷️", "🧮", "✏️", "📏", "🏪", "🔍", "🔁", "👉", PREFS_MARK))]
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
            if data.get("similar"):
                short = re.split(r"[\s｜|（(\-—－/／:：,，。]", data["similar"]["name"])[0][:10]
                anchor = f"「{short}」"
            else:
                anchor = f"捷運{data['near']}站" if data.get("near") else ""
            blocks.append(self.format_card(i, r, reason, anchor))
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

    def pref_banners(self, prefs: dict, data: dict, info: dict, category, found: int) -> list:
        """其他條件的說明：錯字修正、附近範圍、排除連鎖、隱藏版、類似、只選一家、資料沒有的條件。"""
        lines = []
        if info.get("fixes"):
            lines.append("✏️ " + "、".join(f"「{a}」看成「{b}」" for a, b in info["fixes"]))
        if data.get("similar"):
            s = data["similar"]
            km = (data.get("radius") or 0) / 1000
            lines.append(f"🔁 找和「{s['name']}」類似的店：同樣是{s['category']}、"
                         f"距離它 {km:g} 公里內、每人 {s.get('budget')} 元內")
        elif prefs.get("near") and data.get("near"):
            if data.get("radius_relaxed"):
                lines.append(f"📏 捷運{prefs['near']}站 {data.get('radius_first', 800)} 公尺內符合的店不到 5 家，"
                             f"範圍放寬到 {data['radius'] / 1000:g} 公里。")
        elif prefs.get("near"):
            lines.append(f"⚠️ 推薦服務還沒有「捷運{prefs['near']}站」的座標（api.py 可能是舊版），先不限地區找。")
        if prefs.get("no_chain") and data.get("chain_removed"):
            lines.append(f"🏪 已排除 {data['chain_removed']} 家連鎖店（同一個品牌在資料裡有 2 家以上的店）")
        if prefs.get("hidden_gem"):
            note = f"符合的只有 {found} 家" if 0 < found < 5 else ""
            lines.append("🔍 隱藏版：評分 4.3 以上、評論數比同區同類型一半的店少（但至少 50 則，評分才可靠）"
                         + (f"，{note}" if note else ""))
        if prefs.get("pick_one") and found:
            lines.append("👉 只挑一家：綜合評分、評論數和預算，最推薦這間。")
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


# ==== 以下由 tools/build_lookups.py 自動產生，請勿手動修改 ====
MRT_STATIONS = ["中山", "中山國中", "中山國小", "中正紀念堂", "信義安和", "內湖", "公館", "六張犁", "劍南路", "劍潭", "動物園", "北投", "北門", "南京三民", "南京復興", "南港", "南港展覽館", "南港軟體園區", "古亭", "台北101/世貿", "台北小巨蛋", "台北車站", "台大醫院", "台電大樓", "唭哩岸", "善導寺", "國父紀念館", "圓山", "士林", "大安", "大安森林公園", "大橋頭", "大湖公園", "大直", "奇岩", "小南門", "市政府", "後山埤", "復興崗", "忠孝復興", "忠孝敦化", "忠孝新生", "忠義", "文德", "新北投", "昆陽", "明德", "景美", "木柵", "東湖", "東門", "松山", "松山機場", "松江南京", "民權西路", "永春", "港墘", "石牌", "科技大樓", "芝山", "萬芳社區", "萬芳醫院", "萬隆", "葫洲", "行天宮", "西湖", "西門", "象山", "辛亥", "關渡", "雙連", "麟光", "龍山寺"]
TERM_READINGS = {"上海菜": "shang4 hai3 cai4", "下午茶": "xia4 wu3 cha2", "世貿": "shi4 mao4", "中山": "zhong1 shan1", "中山區": "zhong1 shan1 qu1", "中山商圈": "zhong1 shan1 shang1 quan1", "中山國中": "zhong1 shan1 guo2 zhong1", "中山國小": "zhong1 shan1 guo2 xiao3", "中式": "zhong1 shi4", "中式餐廳": "zhong1 shi4 can1 ting1", "中正": "zhong1 zheng4", "中正區": "zhong1 zheng4 qu1", "中正紀念堂": "zhong1 zheng4 ji4 nian4 tang2", "中菜": "zhong1 cai4", "串燒": "chuan4 shao1", "丼飯": "jing3 fan4", "京站": "jing1 zhan4", "便當": "bian4 dang1", "信義": "xin4 yi4", "信義區": "xin4 yi4 qu1", "信義商圈": "xin4 yi4 shang1 quan1", "信義安和": "xin4 yi4 an1 he2", "個人鍋": "ge4 ren2 guo1", "內湖": "nei4 hu2", "內湖區": "nei4 hu2 qu1", "內湖科學園區": "nei4 hu2 ke1 xue2 yuan2 qu1", "內科": "nei4 ke1", "公館": "gong1 guan3", "公館夜市": "gong1 guan3 ye4 shi4", "六張犁": "liu4 zhang1 li2", "冰品": "bing1 pin3", "刨冰": "bao4 bing1", "剉冰": "cuo4 bing1", "劍南路": "jian4 nan2 lu4", "劍潭": "jian4 tan2", "動物園": "dong4 wu4 yuan2", "北投": "bei3 tou2", "北投區": "bei3 tou2 qu1", "北投溫泉": "bei3 tou2 wen1 quan2", "北車": "bei3 che1", "北門": "bei3 men2", "南京三民": "nan2 jing1 san1 min2", "南京復興": "nan2 jing1 fu4 xing4", "南港": "nan2 gang3", "南港區": "nan2 gang3 qu1", "南港展覽館": "nan2 gang3 zhan3 lan3 guan3", "南港展館": "nan2 gang3 zhan3 guan3", "南港軟體園區": "nan2 gang3 ruan3 ti3 yuan2 qu1", "南西": "nan2 xi1", "南軟": "nan2 ruan3", "古亭": "gu3 ting2", "台北小巨蛋": "tai2 bei3 xiao3 ju4 dan4", "台北站": "tai2 bei3 zhan4", "台北車站": "tai2 bei3 che1 zhan4", "台大": "tai2 da4", "台大醫院": "tai2 da4 yi1 yuan4", "台式料理": "tai2 shi4 liao4 li3", "台菜": "tai2 cai4", "台電大樓": "tai2 dian4 da4 lou2", "合菜": "he2 cai4", "唭哩岸": "qi4 li1 an4", "啤酒": "pi2 jiu3", "善導寺": "shan4 dao3 si4", "喝酒": "he1 jiu3", "四川菜": "si4 chuan1 cai4", "國父紀念館": "guo2 fu4 ji4 nian4 guan3", "圓山": "yuan2 shan1", "士林": "shi4 lin2", "士林區": "shi4 lin2 qu1", "士林夜市": "shi4 lin2 ye4 shi4", "壽司": "shou4 si1", "壽喜燒": "shou4 xi3 shao1", "壽喜鍋": "shou4 xi3 guo1", "夜市": "ye4 shi4", "大同": "da4 tong2", "大同區": "da4 tong2 qu1", "大安": "da4 an1", "大安區": "da4 an1 qu1", "大安森林公園": "da4 an1 sen1 lin2 gong1 yuan2", "大巨蛋": "da4 ju4 dan4", "大橋頭": "da4 qiao2 tou2", "大湖公園": "da4 hu2 gong1 yuan2", "大直": "da4 zhi2", "大稻埕": "da4 dao4 cheng2", "奇岩": "qi2 yan2", "定食": "ding4 shi2", "寧夏夜市": "ning2 xia4 ye4 shi4", "小南門": "xiao3 nan2 men2", "小吃": "xiao3 chi1", "小巨蛋": "xiao3 ju4 dan4", "小火鍋": "xiao3 huo3 guo1", "小籠包": "xiao3 long2 bao1", "小酌": "xiao3 zhuo2", "居酒屋": "ju1 jiu3 wu1", "川味": "chuan1 wei4", "川菜": "chuan1 cai4", "市府": "shi4 fu3", "市政府": "shi4 zheng4 fu3", "師大": "shi1 da4", "師大夜市": "shi1 da4 ye4 shi4", "平價小吃": "ping2 jia4 xiao3 chi1", "後山埤": "hou4 shan1 pi2", "復興崗": "fu4 xing4 gang3", "微風南山": "wei1 feng1 nan2 shan1", "忠孝復興": "zhong1 xiao4 fu4 xing4", "忠孝敦化": "zhong1 xiao4 dun1 hua4", "忠孝新生": "zhong1 xiao4 xin1 sheng1", "忠義": "zhong1 yi4", "快炒": "kuai4 chao3", "意式": "yi4 shi4", "披薩": "pi1 sa4", "拉麵": "la1 mian4", "捷運": "jie2 yun4", "文山": "wen2 shan1", "文山區": "wen2 shan1 qu1", "文德": "wen2 de2", "新光三越": "xin1 guang1 san1 yue4", "新北投": "xin1 bei3 tou2", "日式": "ri4 shi4", "日式料理": "ri4 shi4 liao4 li3", "日料": "ri4 liao4", "日本料理": "ri4 ben3 liao4 li3", "日本菜": "ri4 ben3 cai4", "早午餐": "zao3 wu3 can1", "早餐": "zao3 can1", "昆陽": "kun1 yang2", "明德": "ming2 de2", "景美": "jing3 mei3", "木柵": "mu4 shan1", "東區": "dong1 qu1", "東湖": "dong1 hu2", "東門": "dong1 men2", "松山": "song1 shan1", "松山區": "song1 shan1 qu1", "松山文創": "song1 shan1 wen2 chuang4", "松山機場": "song1 shan1 ji1 chang3", "松江南京": "song1 jiang1 nan2 jing1", "松菸": "song1 yan1", "民權西路": "min2 quan2 xi1 lu4", "水餃": "shui3 jiao3", "永康街": "yong3 kang1 jie1", "永春": "yong3 chun1", "江浙菜": "jiang1 zhe4 cai4", "沾麵": "zhan1 mian4", "泰國料理": "tai4 guo2 liao4 li3", "泰國菜": "tai4 guo2 cai4", "泰式": "tai4 shi4", "泰式料理": "tai4 shi4 liao4 li3", "海鮮鍋": "hai3 xian1 guo1", "涮涮鍋": "shuan4 shuan4 guo1", "港墘": "gang3 qian2", "港式": "gang3 shi4", "港點": "gang3 dian3", "滷肉飯": "lu3 rou4 fan4", "漢堡": "han4 bao3", "火鍋": "huo3 guo1", "炸豬排": "zha4 zhu1 pai2", "炸雞": "zha4 ji1", "烤肉": "kao3 rou4", "烤鴨": "kao3 ya1", "無菜單": "wu2 cai4 dan1", "熱炒": "re4 chao3", "燉飯": "dun4 fan4", "燒烤": "shao1 kao3", "燒肉": "shao1 rou4", "燒鳥": "shao1 niao3", "牛排": "niu2 pai2", "牛肉麵": "niu2 rou4 mian4", "甜食": "tian2 shi2", "甜點": "tian2 dian3", "生魚片": "sheng1 yu2 pian4", "石牌": "shi2 pai2", "石頭火鍋": "shi2 tou2 huo3 guo1", "石頭鍋": "shi2 tou2 guo1", "科技大樓": "ke1 ji4 da4 lou2", "粵菜": "yue4 cai4", "精釀": "jing1 niang4", "紅樓": "hong2 lou2", "羊肉爐": "yang2 rou4 lu2", "美式": "mei3 shi4", "美式餐廳": "mei3 shi4 can1 ting1", "美麗華": "mei3 li4 hua2", "義大利菜": "yi4 da4 li4 cai4", "義大利麵": "yi4 da4 li4 mian4", "義式": "yi4 shi4", "義式料理": "yi4 shi4 liao4 li3", "臨江街": "lin2 jiang1 jie1", "臭豆腐": "chou4 dou4 fu3", "臺北車站": "tai2 bei3 che1 zhan4", "臺大": "tai2 da4", "芝山": "zhi1 shan1", "花博": "hua1 bo2", "茶餐廳": "cha2 can1 ting1", "華山": "hua2 shan1", "華山文創": "hua2 shan1 wen2 chuang4", "華西街": "hua2 xi1 jie1", "萬芳社區": "wan4 fang1 she4 qu1", "萬芳醫院": "wan4 fang1 yi1 yuan4", "萬華": "wan4 hua2", "萬華區": "wan4 hua2 qu1", "萬隆": "wan4 long2", "葫洲": "hu2 zhou1", "薑母鴨": "jiang1 mu3 ya1", "蛋糕": "dan4 gao1", "行天宮": "xing2 tian1 gong1", "西湖": "xi1 hu2", "西門": "xi1 men2", "西門町": "xi1 men2 ting1", "誠品信義": "cheng2 pin3 xin4 yi4", "調酒": "diao4 jiu3", "豆花": "dou4 hua1", "豚骨": "tun2 gu3", "象山": "xiang4 shan1", "豬排": "zhu1 pai2", "貓空": "mao1 kong1", "赤峰街": "chi4 feng1 jie1", "辛亥": "xin1 hai4", "迪化街": "di2 hua4 jie1", "通化夜市": "tong1 hua4 ye4 shi4", "連鎖": "lian2 suo3", "部隊鍋": "bu4 dui4 guo1", "酒吧": "jiu3 ba1", "酸菜白肉": "suan1 cai4 bai2 rou4", "鍋物": "guo1 wu4", "鍋貼": "guo1 tie1", "鐵板燒": "tie3 ban3 shao1", "關渡": "guan1 du4", "附近": "fu4 jin4", "隱藏版": "yin3 cang2 ban3", "雙連": "shuang1 lian2", "雞白湯": "ji1 bai2 tang1", "雪花冰": "xue3 hua1 bing1", "韓國料理": "han2 guo2 liao4 li3", "韓國炸雞": "han2 guo2 zha4 ji1", "韓國烤肉": "han2 guo2 kao3 rou4", "韓國菜": "han2 guo2 cai4", "韓式": "han2 shi4", "韓式料理": "han2 shi4 liao4 li3", "韓式炸雞": "han2 shi4 zha4 ji1", "韓式烤肉": "han2 shi4 kao3 rou4", "韓式燒肉": "han2 shi4 shao1 rou4", "韓料": "han2 liao4", "預算": "yu4 suan4", "飲茶": "yin3 cha2", "餃子": "jiao3 zi5", "饒河夜市": "rao2 he2 ye4 shi4", "饒河街": "rao2 he2 jie1", "鬆餅": "song1 bing3", "魯肉飯": "lu3 rou4 fan4", "鰻魚": "man2 yu2", "鴛鴦鍋": "yuan1 yang1 guo1", "麟光": "lin2 guang1", "麻辣": "ma2 la4", "麻辣火鍋": "ma2 la4 huo3 guo1", "麻辣鍋": "ma2 la4 guo1", "龍山寺": "long2 shan1 si4"}
PINYIN_CHARS = {"an1": "侒厂啽垵媕安峖庵桉氨痷盒盦腤菴萻裺諳鉗闇陰鞍韽馣鴳鵪黯", "an2": "儑啽犴玵雸", "an3": "侒俺匼唵垵晻碪銨闇", "an4": "儑婩岸干按晻暗案桉洝犴胺荌豻錌闇頇頞黯", "ba1": "仈八叭吧哱峇巴扒捌捭朳柭疤笆粑芭蚆豝鈀", "ba2": "八抪拔湃犮癹胈茇茷菝萆詙跋軷魃鼥", "ba3": "把鈀靶", "ba4": "伯壩弝把杷灞爸矲罷耙跁霸靶", "ba5": "叭吧罷", "bai1": "挀掰擘", "bai2": "拜白", "bai3": "伯佰捭擺柏百粨絔罷襬", "bai4": "唄啡庍扒拜排敗派猈稗粺薜鞁", "ban1": "卑彬扳搫搬攽斑斒班瘢籓般褩豳賁頒鳻", "ban3": "昄板版粄舨般蝂覂鈑闆阪魬", "ban4": "並伴半坢姅怑扮拌柈湴瓣絆跘辦辨辯鉡靽", "ban5": "螁", "bao1": "包孢枹炮煲笣胞苞苴葆蕔裒褒鮑齙", "bao2": "砲窇薄鉋雹", "bao3": "保呆堡媬寶怉葆褓飹飽駂鳵鴇", "bao4": "儤刨嘐嚗報忁怉抱暴曝瀑爆菢虣袌袍豹趵鉋鑤髱鮑", "bei1": "俾卑埤悲揹杯柸桮棓椑波盃碑箄背萆藣襬諀錍陂鞞鵯", "bei3": "北悖", "bei4": "倍偝備北哱孛怫悖憊拔梖棑棓焙牬犕狽珼琲碚糒背茀菩萯葡蓓蜚被褙誖貝跋輩邶郥鄁鋇鞁骳", "bei5": "唄臂", "ben1": "奔本栟泍渀蟦賁錛鐼", "ben3": "本畚翉苯", "ben4": "体坋坌夯奔泍渀炃笨軬", "bian1": "便扁揙煸猵甂砭稹箯籩編萹蝙邊鞭", "bian3": "匾封惼扁疺碥稨窆編緶萹褊豍貶辨辯釆鴘鶣", "bian4": "便卞弁忭抃拚揙汳汴編緶艑變辨辮辯遍邲釆閞頨", "bian5": "邊", "bing1": "兵冰屏并掤栟檳絣", "bing3": "丙偋寎屏庰怲摒昺枋柄炳秉稟窉綆苪蛃跰邴鉼陃鞞餅", "bing4": "丙並併偋寎屏平并庰拼摒燹病痭誁", "bo1": "僠剝哱啵嶓帗拔撥播波潑潘玻番發皪砵碆磻缽脖般菠蕃袚趵蹳餑鱍", "bo2": "亳伯佛僰勃募博哱嚗壆孛帛彴怕懪拍挬搏暴服柏柭桲榑欂殕泊浡渤溥瀑爆犦猼瓝瓟白百礡穛箔簙簿胉脖膊舶艴茀茷菩葧蒲蔔薄薜袹襏襮詙豰跑踣郣鈸鉑鋍鎛鑮餺馞駁駮髆魄鵓", "bo3": "播簸蚾跛", "bo4": "擗擘柏檗簸糪薄薜蘗譒", "bo5": "卜啵蔔", "bu1": "哺峬晡誧逋鈽餔鯆鵏", "bu2": "不僕輹轐醭鳪", "bu3": "卜哺堡拊捕撲補鵏鸔", "bu4": "不佈吥咘埔埠婄尃布怖抪捗步溥瓿篰簿蔀薄踄部附餔", "cai1": "偲啋揌猜", "cai2": "才扐材纔裁財", "cai3": "啋埰婇寀彩採睬綵跴踩采", "cai4": "埰寀棌縩菜蔡采", "can1": "參囋嬠嵾餐驂", "can2": "慚戔摲殘淺蚕蠶蹔", "can3": "噆慘憯朁穇篸縿黲", "can4": "參孱摻澯燦璨粲鏒", "cang1": "倉傖凔嵢滄獊瑲艙蒼鶬", "cang2": "臧藏鑶", "cang3": "蒼", "cha1": "偛叉喳嗏嚓差扠扱捷接插杈疀碴臿艖荖釵鍤鎈", "cha2": "叉垞察捈搽斜查梌楂槎碴秅苴茬茶荼鉏", "cha3": "叉土衩蹅", "cha4": "侘叉奼岔差杈汊紁衩詫", "chang1": "倀倡娼昌淐猖琩菖裮錩閶闛鯧鼚", "chang2": "仩倘償嘗嚐場嫦尚常徜棖脹腸萇裳鋿長鱨", "chang3": "儻厂場廠惝敞昶氅淌鋹", "chang4": "倡唱悵昌暢淌瑒韔鬯", "chao1": "剿劋勦吵嘮弨怊抄摷樔焯紹綽訬超鈔", "chao2": "勦嘲巢晁朝樔漅潮濤牊窲罺轈鄛", "chao3": "吵抄槱炒焣眧訬謅超趫鈔", "chao4": "吵巢抄晁縐耖超趠", "che1": "硨莗車", "che3": "奲尺扯撦", "che4": "呫喢坼宅屮徹拆掣揊摰撤斥池澈烢硩詀謵迠", "cheng1": "偁噌搶撐朾棦棱槍樘橕橖檉浾湞牚琤瑲睖瞠秤稱竀虰蟶赬蹚鎗鏿鐺饓", "cheng2": "丞乘倀傖呈嗆城埕埩塍宬峸嵊徵懲成承掁搶撜晟朾棖橙氶浧淨溗澂澄珵盛盯碀程脀脭荿裎觕誠趟郕郢酲醒鋮騬", "cheng3": "呈庱悜裎踜逞騁", "cheng4": "敞牚秤稱竀黨", "chi1": "吃哧喫嗤噭媸抬摛摴攡樆瓻痴癡眵瞝离笞絺胝胵蚩蚳螭誺郗離魑鴟黐齝", "chi2": "匙坻墀她弛徲持提柢歭池沱沶治泜漦狋祇箈箎篪耛茌茬荎莉蚳蛇蝭謘謻貾趍跢踟踶軧迡遲鍉馳驪鶗", "chi3": "侈呎哆奓尺恥扡拖拸搋欼移紕胣蚇袲褫誃豉鉹齒齣", "chi4": "佁傺剟叱呬哆啻喜嘯彳懘抶拆捇敕斥柅湁滯熾痸瘈瘛眙翅翨誃赤趐跅跮踅迣遫鉓飭飾饎騺鳷鶒鷘", "chi5": "麶", "chou1": "妯婤抽掫揄搊牰犨瘳眣篘紬謅跾", "chou2": "仇儔圳嬦幬怞惆愁懤擣栦椆檮燽畤疇盩稠籌紬絒綢菗薵裯詶譸讎躊酬雔鯈", "chou3": "丑偢吜媿扭杻杽瞅醜鈕", "chou4": "殠溴簉臭", "chuan1": "川氚猭瑏穿釧", "chuan2": "傳圌團掾暷椯椽歂舡船輲遄", "chuan3": "喘惴歂膞舛荈踳踹", "chuan4": "串丳汌猭玔穿賗釧鶨", "chuang1": "創戧摐瘡窗舂", "chuang2": "幢床朣橦", "chuang3": "傸愴漺磢闖", "chuang4": "倉凔刱創愴", "chun1": "媋春暙杶椿楯槆橁芚踳輇輴鰆鶞", "chun2": "唇沌淳漘犉純肫脣膞蓴醇錞鯙鶉", "chun3": "偆惷春朐萶蠢賰踳", "cuo1": "差搓撮最營玼瑳磋蹉遳醝髊", "cuo2": "嵯澨痤瘥矬蒫蔖躦酇鹺齹", "cuo3": "搓脞醝", "cuo4": "剉剒厝夎挫措摧斮昔棤縒莝蓌銼錯", "da1": "咑嗒噠塌塔搭撘毼瘩矺笚答耷荅褡鎝", "da2": "僤匒呾噠妲怛憚打搨沓炟畣瘩笪答羍荅薘躂迖迭達鎉鐽靼韃龘", "da3": "打觰", "da4": "大汏", "da5": "嗒塔疸瘩繨", "dan1": "丹儋冉勯匰單娊愖擔檐殫湛澹甔癉眈砃簞耽聃聸襌襜覘蹛鄲酖", "dan3": "丼亶伔刐單嘽抌撢撣澸燀玬疸癉皽紞膽衴黕黮黵", "dan4": "亶但倓僤儃儋呾唌啖啗啿單嘾噉嚪壇帎彈忱怛惔憚憺憾撢撣擔旦柦欿氮沊泹淡潭澶澹狚甔癉癚石禫窞繵膻萏蛋蜑蜒蟺觛訑詹誕譂贉贍霮餤髧鴠", "dang1": "儅噹檔澢璫當簹艡蟷襠鐺", "dang3": "党擋攩欓讜黨", "dang4": "儅場婸宕崵嵣愓擋檔潒燙瑒璗瓽當瘍盪碭礑簜菪蕩讜逿闣", "dang5": "當", "dao1": "刀叨啁忉惆氘舠虭裯魛", "dao2": "叨捯", "dao3": "倒嘄壔導島忑捯搗擣禂禱菿蹈道鳥", "dao4": "佻倒儔到受導幬悼敦檤檮洮濤燾瓙盜稻箌絩纛翢翿菿薵道陶", "de2": "得德淂登陟", "de5": "地底得的", "di1": "低嘀堤彽提樀氐渧滴牴的磾羝袛鍉鏑隄鞮", "di2": "儥嘀嚁嫡扚敵杓梑樀櫂浟滌狄的笛篴糴翟苖荻蓧蔋藋藡覿踧蹢迪逐適鍉鏑靮馰鬄魡鸐", "di3": "厎呧坻堤底弤抵提柢楴氐牴砥聜菧觝詆軧邸阺", "di4": "勺啻地墆墑墬娣嵽帝弔弟揥旳杕柢梊棣渧焍玓珶甋疐的眱睇碲禘第約締胝芍菂蒂蔕蚳蝃螮諟諦踶蹄逮遞遰釱題", "dian1": "佔傎厧巔拈掂敁槙滇玷甸瘨癲詀蹎顛齻", "dian3": "典婰痶碘蒧蕇跕踮點", "dian4": "佃唸坫墊壂奠婝店惦扂橝殿沾涎淀澱玷甸癜磹簟腍蜓鈿阽電靛驔", "diao1": "凋刀刁叼彫敦椆殦淍琱瞗碉虭蛁貂軺錭雕鯛鳭鵰", "diao3": "佻屌扚鳥鵃", "diao4": "伄倜吊啁嬥弔挑掉稠窵糶絩綢蓧藋蜩誂調趙跳踔釣銚雿魡", "ding1": "丁仃叮帄玎町疔盯耵虰酊釘靪", "ding3": "仃嵿濎艼葶薡酊頂顁鼎", "ding4": "啶奠定掟椗汀碇碠萣訂釘鋌錠顁飣", "dong1": "冬咚埬崠東氡涷苳菄蝀鶇鼕", "dong3": "墥嬞懂揰湩董蕫", "dong4": "侗凍動垌姛峒崠恫戙挏桐棟洞湩烔狪甬硐筒筩絧胴衕詷迵酮霘駧騆", "dou1": "兜唗氀瞗窬都", "dou3": "抖斗枓蚪鈄陡", "dou4": "吋投斗梪浢瀆痘竇脰荳讀豆逗逾郖酘鋀餖鬥", "du1": "剢嘟堵督都醏闍", "du2": "儥噣櫝殰毒瀆牘犢獨皾碡竇纛罜蝳裻襡讀讟贕錖韇韣韥頓髑黷", "du3": "堵帾杜睹竺篤肚賭鍺陼", "du4": "剫土塗妒宅度斁杜橐渡睪秺肚芏蠹詫鍍", "dui1": "埻堆搥敦槌痽碓追鎚鐓頧鴭", "dui3": "啍頧", "dui4": "兌奪對憝懟敦杸濧瀢瀩碓祋薱譈鈗錞鐓隊", "dun1": "俊吨噸墩惇撉敦橔燉礅蜳豚蹲鐓驐", "dun3": "盹躉", "dun4": "伅囤坉庉忳敦楯沌潡炖燉盾腞腯豚遁遯鈍頓", "fan1": "勫墦嬏帆幡拚旛潘犿番籓繙翻蕃藩轓颿鱕", "fan2": "伋凡墦帆杋柉楓樊橎氾渢瀪瀿煩燔璠番礬笲籵繁繙羳膰蕃薠藩蘩蟠蠜袢蹯釩鐇鷭", "fan3": "反返釩", "fan4": "反奿帆梵樊氾汎泛滼犯畈笵範范販軓軬釩飯", "fang1": "匚坊妨放方枋汸淓祊芳邡鈁雱", "fang2": "坊妨彷房方肪防魴", "fang3": "仿倣彷放方昉枋瓬紡舫訪髣鶭", "fang4": "匚放紡邡", "feng1": "丰偑妦封峰崶方桻楓檒灃烽犎猦瘋葑蘴蚌蜂豊豐酆鋒風飌麷", "feng2": "堸夆捀摓渢縫舽艂逄逢馮", "feng3": "唪泛覂諷風", "feng4": "俸奉捧桻焨甮縫葑諷賵風鳳鵬", "fu1": "不伕傅垺夫姇孵尃怤拊捊敷枎柎沕溥烰琈璷甫痡砆稃糐紨綒胕膚芣荂荴蓲衭趺跗邞鄜鈇附鳺麩", "fu2": "伏佛俘冹刜包匐咈咐垘垺夫姇孚宓岪巿市帗幅弗彳彿怫扶拂服枎枹柫柭桴棴榑氟汱沸泭洑浮涪澓烰玸琈砩祓福符笰箙紱紼綍纀罘罦翇胕艀艴芙芣芾苻茀茯莩菔萉葍虙蚨蜉蝠袚袱複襆諨費踾輻郛韍鳧鴔鵩鶝黻", "fu3": "俛俯哺嘸尃府弣拊撫斧柎滏父甫簠胕脯腐腑莆蚥蜅軵輔郙釜鈇頫黼", "fu4": "付伏偩偪傅副咐報复婦嬔富峊復怀掊服柎榑洑父祔福紨緮縛腹萯蕧蚥蚹蛗蝜蝮複褔覆訃負賦賻赴跗軵輹鍑阜附馥駙鮒鰒", "gang1": "亢伉剛堈堽岡崗扛抗杠棡犅疘矼碙綱缸罡肛肮釭鋼阬頏", "gang3": "崗港溝犺", "gang4": "戇杠槓鋼", "gao1": "咎槁槔櫜滜皋睪睾篙糕羔膏韟高鷎鼛", "gao3": "搞暠杲槁檺浩稿筶縞蒿鎬", "gao4": "勂告祰筶膏誥郜鋯高", "ge1": "仡割可咯哥圪屹戈扢擖擱格歌滒犵疙紇肐胳菏袼鉻鉿閘閤鴐鴿", "ge2": "佫佮假嗝噶塥愅挌搿擱杚格浩滆猲獦砝胳膈茖葛蛤觡輵轕鉀鎑鎘閣閤隔革鞈鞷韐頜騔骼髂鬲鮥鮯鵅", "ge3": "個各合哿舸葛蓋魺", "ge4": "介個各吤硌箇虼鉻", "gong1": "供公共功匑塨宮工幊弓恭愩攻杠疘篢紅肱蚣觥躬釭魟龔", "gong3": "共嗊廾拱拲栱汞珙磺礦蛩輁鞏", "gong4": "供共嗊愩渱羾虹貢贛", "gu1": "估呱呴咕姑嫴孤家枯柧橭沽泒瓠皋盬箍箛罛胍菇菰蓇蛄觚軱轂辜酤鈲骨鮕鴣", "gu2": "骨鶻", "gu3": "古唃啒嘏夃尳愲扢榖榾汩沽淈滑濲瀔牯盬瞽穀縎罟羖股臌苦蓇薣蛄蛌蠱角詁谷賈轂鈷離骨骰鵠鼓", "gu4": "估僱凅古告固堌崮故梏棝牿痼盬稒鈷錮雇顧", "gu5": "咕", "guan1": "倌冠官棺淪琯瘝矜綸莞菅觀閞關鰥鱞", "guan3": "幹斡琯痯筦管莞錧館", "guan4": "丱串冠婠悹悺慣懽摜果棺櫬權毌涫淉灌爟琯瓘盥矔祼罐觀貫鑵雚鰥鱹鸛", "guang1": "光垙廣恍桄洸潢炚烡珖胱茪趪銧", "guang3": "廣獷臩", "guang4": "俇光廣撗擴桄臦迋逛", "guo1": "咶咼嘓堝墎崞彉活渦濄瘑矌聒蜮蟈過郭鍋鐹", "guo2": "唬囗國埻幗慖摑漍簂聝膕蔮虢郭馘", "guo3": "划惈果槨淉猓粿蜾蝸蠃裹褁輠錁鍋鐹", "guo4": "划過", "guo5": "過", "hai1": "咍咳嗨", "hai2": "咳孩還頦骸", "hai3": "浬海烸胲醢", "hai4": "亥侅嗐妎害拸氦猲絯閡餀駭駴", "han1": "哻嫨憨歛犴甘蚶谽酣頇魽鼾", "han2": "函含咁唅寒嵅幹梒椷榦汗涵淊淦焓琀甝笒肣虷蚶蜬邗邯鈐鋡韓頜馯魽", "han3": "厂喊嚂嵌罕蔊豃軒闞鬫", "han4": "含悍感憾扞捍撖撼旰旱暵桿汗泔涆涵淊漢澉澣瀚灘焊熯睅矸罕翰肣菡蔊蜭螒譀邯酣釬銲鋎閈雗靬頷顄顩馯駻鳱鶾", "he1": "呵哈喝欱苛荷訶", "he2": "何劾合和咼哈嗑噈姀害峆挌揭敆曷核楁毼河洽涸渮渴澕熆犵盉盍盒礉禾篕籺紇繳翮荷菏藿蚵蝎螛覈貉輅轄郃鉌鑉閡闔鞨頜餲魺鶡麧齕龢", "he4": "何佫呼和哧喝嗃嚇壑愒曷格煂熇猲硅翯荷藃褐謞賀赫霍鬩鵠鶡鶴齃", "hong1": "吽哄哅嚝巆揈渹灴烘焢薨訇謍谾輷轟鍧", "hong2": "仜共吰妅宏弘彋揈汪汯泓洚洪浤渱玒瓨竑竤篊紅紘翃耾舼苰葒蕻虹谹谼鈜鋐閎霐霟鞃魟鴻黌", "hong3": "哄嗊愩", "hong4": "哄港澋澒蕻虹訌闀鬨鴻", "hou1": "呴齁", "hou2": "侯喉猴瘊篌翭翵腄葔鄇銗鍭餱鯸", "hou3": "吼吽呴", "hou4": "侯候厚后垕堠後洉詬逅郈鱟", "hu1": "乎匢呼垀寣幠忽惚惡戲昒曶欻歑淲淴滹糊膴芔芴虍虖謼雽", "hu2": "和嘝囫壺媩弧抇搰斛核楜槲汩沍湖瀫焀狐猢瑚瓠瓡礐糊縎縠胡葫虖蝴螜衚觳醐鈷鋘餬鬍魱鰗鴩鵠鶘鶦鶻", "hu3": "唬汻淲滸滹琥箎虎許", "hu4": "互冱嗀嗃姱婟嫭嫮岵怙戶戽扈擭昈枑楛槴沍洿淈滬濩熩瓠祜穫笏糊綔羽胍芐苦蔰虎虖觷護豰鄠鍙雇雽韄頀魱", "hua1": "化吪嘩砉花華蒍蘤錵", "hua2": "侉划劃哇嘩姡學找搳滑澮猾獪華螖譁豁釪釫鋘鏵驊鷨", "hua4": "划劃化叱咶嫿嬅崋摦杹槬樺檴澅畫稞粿繣罫華觟話諙輠魤鮭", "huo1": "吙呵嚄搉攉火豁騞", "huo2": "佸和姡活秮萿越趏", "huo3": "伙夥扮漷火鈥", "huo4": "化和嚄壑惑或捇掝擭攉曤檴沎湱漷濊濩瀖獲瓠眓矐砉禍穫篧膕臛藿蠖諕謋豁豰貨過鑊隻霍靃", "huo5": "伙和", "ji1": "丌乩倚其几刉剞勣呰咭唧嘰圾基墼奇妓姬居屐嵇幾庴憿懠撽擊攲敧期机枅棋機櫅毄汲激犄璣畸畿癪磯禨稘稽積笄箕簊緝績羇羈肌芨苙萁蘄虀虮覘觭諅譏谿跂跡踑踦蹟躋鄿銈錤鐖鑇隔隮雞鞿飢饑鸄齊齎齏", "ji2": "亟伋佶偮即厝及吇吉圾堲塉姞嫉尐岋岌嶯庴彶忣急戢揖揤极棘楖楫極槉橶殛汲洁湒潗濈狤疾瘠瞿秸笈簎籍級緝耤胔脊膌莋蒩蒺蓻蕀蕺藉蜡蝍螏蠀襋覿觙諔谻趌踖蹐躤輯轚郅銡鏶集霵革鞊鶺鷑", "ji3": "丮几妀己幾戟掎撠擠極泲濟穖紀給脊虮蟣蟻踦颳魕麂", "ji4": "伎偈其冀劑嚌塈妓姼季寂寄幾徛忌悸惎懠懻技擊旡既暨暩檕櫅汥洎漃漈濟瀱猗璾畟疵痵瘵癠睽瞉祭禨稷穄穊穖穧粢系紀紒結繫繼罽肌臮芰茍茤蔇薊薺蘮蘻蟿裚覬計訐記誋諅跽踑際霽騎驥髻鬾鮆鯚鯽鰶鰿鱀鱭鵋齊齌齍", "jia1": "伽佳傢加咖嘉夾家抸挾擖枷梜椵毠泇浹犌猳珈痂笳筴耞茄葭袈袷裌豭貑跏迦鉿鎵駕麚", "jia2": "唊夾岬恝戛扴拮挈揩揳猰絜莢蛺袷裌跲郟鋏頡頰骱鴶鵊", "jia3": "假呷嘏夏婽岬徦押斝暇柙椵榎檟玾甲瘕胛蝦賈鉀", "jia4": "价假價嫁家幏架枷稼賈駕駱", "jia5": "家", "jian1": "傔兼咸喊堅奸姦孱尖帴幵惤戔揃搛攕椷櫼殲沮淺湔漸濺瀸煎熞熸牋犍猏瑊監礛箋籈籛緘縑纖肩艱菅菺葌蒹蔪蕑虃豜趼軒鋑鋻鑯閒間靬鞬韉餰鰔鰜鰹鳽鵳鶼麉黚", "jian3": "儉前剪劗咸囝堿帴戩挸揀揃揵撿暕柬楗橏檢涀減湕煎犍瑐瞼筧箴簡籛繭翦茧藆襉襺謇譾趼蹇錢閒間險騫鬋鰎鹹鹼齊", "jian4": "件俴健僣僭劍塹寋建揵朁栫楗榗檻毽洊淺湔漸澗濫濺瀳煎牮犴珔監瞷箭糋腱艦茛荐葥蔪薦螹袸見諓諫譖賤跈踐轞鋻鍊鍵鐱鑑鑒鑳閒間鞬餞", "jiang1": "僵姜將彊槳橿殭江漿疆礓糨翞茳蔣薑螿豇韁", "jiang3": "傋將槳獎耩膙蔣講顜", "jiang4": "匠將嵹弶強彊洚漿疆糨紅絳虹袶謽醬降", "jiao1": "交佼僥僬儌咬嘄嘐姣嫶嬌嵺嶕徼憍憢挍撟教椒橋湫澆激焦燋矯礁穚簥膠膲艽茭菽萩蕉蕎蛟蟭跤轇郊鄗鐎驕骹鮫鵁鷦鷮", "jiao2": "嚼矯", "jiao3": "佼僑僥僬儌剿劋勦卻咬喬妖姣徼恔憿捁摎摷撟攪敥敿曒校樔橋湫潐灚煍狡皎皛皦矯笅筊糾絞繳腳膠蟜角譑蹻鉸餃鱎", "jiao4": "僬叫嘂嘄噭嚼嬓學嶠徼悎挍教斠校橋滘潐激灂爝獥珓皭窌窖筊絞繳覺譥趫趭較轎酵醮釂餃", "jie1": "亥偕喈嗟嚌差接揭椄楷湝痎癤皆祖秸節結罝苴菨蝔街謯鍇階鞂鶛", "jie2": "倢偈傑劫劼唧啑圾契婕孑尐岊崨嵑嵥巀幯截扢拮捷擳昅杰桀桔椄楬楶概榤洁洯渴滐潔狤疌睫砝碣竭節紇紒結絜羯脥蛣蝍蠽衱袷袺訐詰趌跲踕迼鉣鍻鞊頡髻鮚", "jie3": "她姐媎檞獬袓解鍇", "jie4": "介价借吤唶嗟妎屆岕嶰悈戒拾暨楷犗玠界疥砎籍紒耤艐芥藉蚧蛶褯解誡雃骱鶛", "jie5": "价假價家", "jin1": "今埐堻嶜巾惍斤津濜珒璡矜禁竻筋紟肋菳衿襟觔金釿", "jin3": "僅儘堇婜巹廑慬斳槿瑾盡紾緊菫蓳謹錦饉", "jin4": "伒僅僸儘勁吟噤嚍堇墐妗嬐嬧寖慬搢晉枃榗殣浸湛溍濜煡燼瑾璡璶盡祲禁笒紟縉臸藎覲賮贐近進金鋟靳", "jing1": "京仱兢婛巠徑惊旌旍晟晶殑氏涇睛箐粳精經荊莖菁蜻醒鋞青驚鯨鵛鶁鶄麠鼱", "jing3": "丼井儆剄幜憬憼擏景暻汫烴獷璟璥睛蟼警阱頸", "jing4": "井俓倞凈勁境婧弳徑憼擏敬桱檠殑淨獍痙竟竫競精經脛葝逕鏡陘靖靘靚靜", "jiu1": "九勼啾愁揂揪揫摎救朻樛湫牞穋究糾繆萛蝤赳鬮鳩", "jiu3": "久九氿灸玖酒韭", "jiu4": "僦咎就廄慦捄救柩殧湫疚究臼舅舊赳鯦鷲麔", "jiu5": "蹴", "ju1": "且仇佝俱匊告娵娶居岨崌拘挶捄据掬揈揟梮椐沮泃涺狙琚疽痀砠租罝腒臄苴菹蛆蜛裾趄踘車軥鋦鋸陱雎鞠鞫駒驕鬻鮈鶋", "ju2": "侷僪局屈拘拱桔椈橘淗湨焗犑狊菊蓻蘜趉趜跼蹻輂郹鋦閰雛駶驧鵙鵴鶪鼰鼳", "ju3": "咀岨拒拘枸柜椇櫸沮疽矩筥籧舉莒萭蒟踽郰鉏鋤齟", "ju4": "俱倨具劇句埧壉姐姖寠屨岠巨怐怚懅懼拒据據沮洰渠澽炬焣犋珇瞿秬窶簍粔耟聚苣菹蒩蔞蘧處虡蚷螶袓裾詎豦貗趡足跙距踞躆遽鄒鄹醵鉅鋸鐻雛颶駏駒", "kang1": "嫝康慷杭漮糠荒邟鏮", "kang2": "扛摃", "kang3": "伉骯", "kang4": "亢伉匟囥坑康抗杭沆炕犺邟鈧閌阬", "kao1": "尻訄", "kao3": "拷撟攷栲洘烤熇考薧", "kao4": "搞槁洘熇犒銬靠", "ke1": "匼呵嗑坷嵙搕柯棵榼樖牁犐珂痾瞌砢磕科稞窠簻苛薖蝌趷軻鈳鉿錒頦顆髁", "ke2": "咳殼翗頦", "ke3": "可坷岢嵑嶱敤棵毼渴盍碣磕顆", "ke4": "克刻剋勀可喀嗑堁客峇恪悈愘歁氪溘濭硞科緙艐蚵課錁騍龕", "ke5": "喀", "kong1": "倥崆悾控椌涳硿穹空箜錓", "kong3": "倥孔恐悾空", "kong4": "控矼硿空腔鞚", "kuai3": "蒯", "kuai4": "傀儈噲塊廥快旝會檜澮狤獪璯筷膾蒯蕢鄶駃鬠魁鱠", "kun1": "卵坤崑昆晜混焜猑琨菎褌豤錕頑餛騉髡鯤鰥鶤", "kun3": "壼悃捆梱硱稛綑裍閫齦", "kun4": "困昆涃睏", "la1": "啦喇垃拉搚摺柆砬翋菈邋", "la2": "剌喇拉旯砬", "la3": "喇拉藞", "la4": "儠剌拉揧擸楋瘌癩腊臘落蜡蠟辣鑞鬎", "la5": "啦喇垃拉藍鞡", "lan2": "婪嵐惏攔斕暕欄欗漣濫瀾灆灡籃籣藍蘭襤襴譋讕躝郴鑭闌", "lan3": "壈婪嬾廩懶攬欖浨濫灠爦纜覽醂顲", "lan4": "僋嚂壏湅濫灠煉爁爛糷覽諫連鑭闌", "li1": "哩", "li2": "儷剺劙厘喱嫠孋孷廲斄杝柂梨梩樆氂漓灑灕犁犛狸琍璃离筣篱籬糎縭纚罹莉菞蔾藜蘺蜊蠡蠫褵謧貍酈醨釃釐鑗離霾驪鯬鱺鵹鸝麗黎黐黧", "li3": "俚兣哩娌峛悝李欐欚浬澧理禮粴纚蠡裡豊邐醴里鋰離鯉鱧鱺麗", "li4": "仂位例俐俚儮儷列利力勵厤厲叻吏哩唎唳嚦塛壢娳孋屴岦悷慄戾扐扚捩搮擽攦攭曆曞朸栗櫟櫪欐歷沴泣浰淚溧濼濿瀝犡猁珞瑮瓅瓥痢癘皪盭砅砬硌磿礪礫禲秝立笠篥粒糲綟翮苙荔莉蒚蒞藶蚸蜧蝕蝷蠡蠣觻詈讈赲躒轢轣酈鎘鑠隸離靂靋颯鬁鬲鱱鱳鴗鷅麗麜", "li5": "哩里", "lian2": "令僆嗹奩帘廉怜憐搛斂槏槤溓漣濂璉瞵磏簾縺羸翴聯苓蓮薕蠊褳覝謰譧蹥連鎌鏈鐮零鬑鰱", "lian3": "嬚摙斂槤歛溓璉羷膦臉蓮薟蘞襝輦連鄻", "lian4": "僆堜孌戀摙撿攣楝欄殮浰湅澰瀲煉練萰連鍊鏈鱄", "liao1": "撩", "liao2": "佬僇僚勞嘹嫽寥寮屪嵺嶚廖憀憭摎撩敹料暸樂樛橑漻潦燎獠療窌簝繆繚聊膋膫蟟蟧豂轑遼鏐鐐镽飂飉鷯", "liao3": "了僚嫽憭撩潦燎瞭繆膫蓼轑鄝釕镽", "liao4": "了嘹嫽尥廖撂撩料炓燎療瞭窷繆蟉轑釕", "lin1": "拎", "lin2": "壣嶙惏暽林淋溓滲潾燐琳璘疄痳瞵碄磷箖粼繗翷臨蹸轔遴鄰鏻霖驎鱗麟", "lin3": "亃伈僯凜嶙廩懍檁磷稟箖粼菻顲", "lin4": "亃任僯吝橉淋焛甐疄痳瞵磷膦臨藺賃蹸躪轔遴鄰閵", "liu1": "溜瀏蹓", "liu2": "劉媹嵧懰摎斿旒榴泵流游溜漻瀏琉留瘤硐硫聊蓅藰蟉裗遛鎏鎦鏐飀餾騮鰡鶹麍", "liu3": "僂嬼懰柳泖珋留綹罶蓼蔞藰鉚", "liu4": "六塯廇溜留碌磟窌翏蹓遛鎦鏐陸雡霤飂餾鷚", "long1": "隆", "long2": "儱嚨寵巃曨朧櫳瀧瓏癃矓礱窿籠聾蘢蝕蠪蠬襱谾豅躘鏧鑨隆靇鸗龍龐龒", "long3": "儱壟徿攏朧篢籠蘢躘隴", "long4": "儱哢弄徿梇蘢衖襱豅", "lou1": "摟瞜", "lou2": "僂嘍婁寠廔慺樓漊漏熡牢瞜窶耬艛蔞螻謱鞻髏", "lou3": "嘍塿嶁摟漊簍耬", "lou4": "漏瘺鏤陋露", "lou5": "嘍", "lu1": "嚕擼謢", "lu2": "嚧壚廬攎櫨瀘爐玈璷盧籚纑罏膚臚艫蘆蠦轤鑪顱鱸鸕鹵", "lu3": "嚕擄擼櫓氌滷磠艣蓾蘆虜鏀鐪魯鹵", "lu4": "僇六坴塶廘彔戮摝攄淕淥漉潞熝琭璐甪瘳盝睩硉碌磟祿稑穋簏簬籙繆翏菉蓼蔍蕗螰角觻谷賁賂路踛蹗輅轆逯鄜酪醁錄錴鏕陸露騄鯥鱳鵱鷺鹿麓", "ma1": "媽嬤孖抹摩螞麻", "ma2": "嗎嘛犘痲痳蟆貊靡麻", "ma3": "傌嗎榪溤瑪碼螞鎷馬鷌", "ma4": "傌嗎嘜榪礣禡罵螞貉驀鬕", "ma5": "么嗎嘛嚜", "man1": "嫚顢", "man2": "僈埋姏悗慢慲槾樠瞞矕絻蔓蠻謾蹣鞔顢饅鬗鬘鰻", "man3": "屘滿睌矕", "man4": "僈墁嫚幔幕慢曼槾漫澫熳獌縵蔓謾鄤鏝", "mao1": "貓", "mao2": "侔兞務堥旄枆毛氂渵犛矛罞耗茅茆蛑蝥蟊貓軞酕鉾錨髦髳鶜", "mao3": "冇卯昴泖茆蓩鉚", "mao4": "冒勖務媢帽懋描旄楙毛毣毷牟瑁眊瞀秏耄耗芼茂萺蝐袤貌貿鄚鄮霿", "mei2": "呅坶堳塺媒嵋徾攗枚某梅楣沒湄溦煤玫瑂眉禖穈糜脢腜膴苺莓蘪郿鋂霉黴", "mei3": "媄媺挴楣每浼渼燘美鎂", "mei4": "味嚜墨妹媒媚寐昧沬煝痗眊眛蝞袂韎魅黴", "men1": "悶", "men2": "亹們怋捫汶璊瞞穈菛虋鍆門", "men4": "們呇悗悶惛懣暪殙滿燜瞞鞔", "men5": "們", "mian2": "冥婂媔棉櫋眠瞑矊矏綿緡蝒蠠", "mian3": "丏俛偭免冕勉喕娩媔愐沔湎澠眄眠絻緬葂靦鮸黽", "mian4": "冥泯湣眄面麵", "min2": "岷崏忞怋旻旼暋民汶玟琝痻盷緡罠苠鈱錉閔", "min3": "僶刡呡愍憫抿敃敏敯暋泯湣潣皿眠笢簢緡繩蠠鈱閔閩黽", "ming2": "冥名嫇明暝榠洺溟盟眳瞑茗萌蓂螟覭鄍銘鳴", "ming3": "姳嫇慏溟皿酩", "ming4": "名命詺", "mu2": "模母氁", "mu3": "坶姆姥峔拇母牡牳畝砪茻踇鉧", "mu4": "募嘿坶墓婺幕幙慔慕暮木楘毣沐炑牟牧目睦穆縸繆艒苜莫莯萺蚞鉬霂鞪鶩", "nan1": "囝囡", "nan2": "冉南喃弇抩暔柟楠男莮萳諵難", "nan3": "喃嫨戁揇湳腩萳蝻赧", "nan4": "妠婻攤灘諵難", "nei3": "哪娞婑浽腇那餒餧", "nei4": "內氝那", "nian1": "拈捻蔫", "nian2": "哖姩年溓粘鯰黏", "nian3": "拈捵捻撚攆涊淰碾簐趁跈蹍躎輦輾", "nian4": "卄唸姩廿念榐", "niang2": "娘孃釀", "niang4": "釀", "niao3": "嫋嬝嬲尥蔦裊褭鳥", "niao4": "尿溺", "ning2": "儜冰凝嚀嬣宁寍寧擰攘檸濘獰甯疑聹苧薴鑏鬤鸋", "ning3": "擰鑏", "ning4": "佞宁寧年擰泞泥濘甯", "niu1": "妞", "niu2": "怓牛", "niu3": "忸扭杻炄狃紐莥鈕", "niu4": "拗蚴", "pai1": "拍", "pai2": "俳啡徘排棑椑牌犤箄簰脾", "pai3": "排迫", "pai4": "派湃蒎鎃", "pi1": "丕伓伾劈坏坯培帔憵批披枇狉痦砒礔秠紕翍苤被辟邳釽鈚鈹銔錍鎞闢霹駓髬魾", "pi2": "仳卑啤埤奊媲岯崥庇庳批枇椑比毗毘毞猈琵番疲皮笓篦紕罷羆脾膍芘蕃蚌蚍蚽蚾蜱螷蠯裨貔郫鄱鈹錍阰陂陴鞞頗魾鵧鼙", "pi3": "仳劈匹否嚭圮庀庇擗比疋痞癖諀銢鴄", "pi4": "俾僻副埤媲屁扑拂揊淠潎澼濞甓睥稫罷脾萆薜被譬辟釽鈲闢鷿", "pian1": "偏媥扁片篇翩鶣", "pian2": "便平扁楄楩璸緶胼萹蝙褊諞跰蹁辯駢骿", "pian3": "諞貵", "pian4": "片猵辨騙", "pin1": "姘拚拼砏礗驞", "pin2": "嚬姘嬪玭矉蘋蠙貧頻顰", "pin3": "品", "pin4": "匕娉泵牝聘", "ping1": "乒俜堋娉涄甹砯砰竮艵頩", "ping2": "倗呯坪屏帡平憑枰泙洴淜玶瓶秤缾苹荓萍蓱蘋蛢評軿郱鉼馮", "ping3": "頩", "ping4": "聘", "qi1": "七倛傶僛凄吃嘁妻娸宿恓悽慼戚攲期柒栖桼棲榿槭欹欺沏淒溪漆碕磎緀緝萋蛣螇諆諿踦蹊迉郪鏚霋魌鶈鸂", "qi2": "七丌亓伎俟倛其勤圻埼奇岐岓崎幾忮忯懠扺技抵掑支旂旗枝棋歧淇濝濟玂琦琪畦畸疧碕示祁祇祈祺禨稘綦耆肵臍艩萁薺藄蘄蚑蚔蚚蜞蟣蠐衹觭跂踑軝逗鄿釮錡鐖隑頎饑騎騏騹鬐鬾鬿鮨鯕鰭鵸鶀麒麡齊", "qi3": "乞企呇啟婍屺崎己敧杞梩棨气碕稽綮綺芑觭豈起跂邔齮", "qi4": "乞亟伎倛偈切刺吱咠唭器契妻忔恝悽愒愾憩扢扱挈揭栔棄槭气氣汔汽泣湆湇滊漬甈甭盵砌磧磩絜綦緝罊肐芞葺薺蟿訖趿跂踖躩迄鼜", "qian1": "仟僉兛千嗛奷婜孅寨岍幵忏愆慳扦掔揃搴攐攓杴檶櫏汧牽瓩磏簽籤粁縴羥芊褰謙遷鈆鉛鋟鐱鑯開阡雃韆顅騫鳽鶼", "qian2": "乾仱前墘媊岒忴扲拑掮揵朁柑榩涔湔漸潛灊灒煔燂燖犍筋箝羬腱葥葴蕁虔蚙赶軡鈐鉆鉗錢鍼靬馯騚騝鰬黔黚", "qian3": "凵唊嗛忏慊撖槏歁淺繾脥蜸譴遣顩齦", "qian4": "俔倩傔嗛塹嵌廞慊棈槧欠欿歉淒牽篟綪縴芊芡茜蒨輤遣錎鰜", "qiao1": "劁喿塙墝墽幓幧悄搞摮敲校橇橋橾毃毳燆碻磝磽礉窯繑繰蕉趬跤踍蹺蹻郻鄗鄡銚鍬鏒鐰雀頝骹", "qiao2": "僑劁喬嘺嫶嶠憔招樵橋潐焦燋犞睄瞧礄翹荍菬蕉蕎蟜譙趫醮驕", "qiao3": "巧悄愀愁磽雀髜", "qiao4": "俏偢削嘺噭墝墽峭帩悄捎撬撽殼硝磽礉竅箾翹茭誚譑譙踃蹺躈鞘顤", "qu1": "佉區去取屈岨嶇怚抾敺曲毆浀焌祛紶組胠脥蛆蛐袪詘趍趜趣趨跙軀镼阹鞠鞫騶驅髷魼鱋鶌麴", "qu2": "佢劬句巨斪朐枸欋氍淭渠灈璩瞿磲籧絇翑翵胊臞菃葋蕖蘧蚼螶蟝蠷蠼衢躣軥遽鉤鐻鮈鴝鼩", "qu3": "取娶曲竘紶苣蝺齲", "qu4": "刞去呿蜡覷誇趣趨跼闃麮鼁", "qu5": "戌", "quan1": "卷圈弮恮悛棬腃酄鐉駩", "quan2": "佺全卷圈埢姾婘巏惓拳拴捲栓棬槫權泉灥牷犈狋獾瑔痊矔筌純絟縓腃荃蜷蠸觠詮謜譔跧踡輇醛銓顴鬈鸛齤", "quan3": "圳汱烇犬甽畎綣虇", "quan4": "串券勸灥牶韏", "rao2": "嬈橈犪蕘襓饒", "rao3": "嬈扰擾繚繞隢", "rao4": "撓穘繞蟯遶", "re2": "捼若", "re3": "喏惹若", "re4": "偌渃熱若", "ren2": "人仁任儿壬妊芢銋鵀", "ren3": "忍栠棯涊稔荏荵銋", "ren4": "仞任刃妊屻忍恁牣紉紝肕腍葚衽訒認軔韌飪", "ri4": "日鈤馹", "rou2": "厹媃揉柔煣禸粈糅葇蝚蹂輮鍒鑐鞣騥髳鰇鶔", "rou3": "楺煣葇蹂輮韖", "rou4": "楺肉莥", "ruan2": "堧壖撋擩", "ruan3": "朊檽濡燸瓀礝緛耎蝡軟阮需", "ruan4": "緛", "sa1": "仨挲撒", "sa3": "撒攃洒灑纚躠靸鞈", "sa4": "卅摋殺泧蔡薩趿鈒鎝鏾隡霅颯馺", "san1": "三參散毿糝蔘鬖", "san3": "傘參散糝繖蔘鏒鏾霰", "san4": "帴散閐鬖", "sen1": "摻森槮滲襂", "sen3": "洒", "shan1": "刪剼埏姍山幓扇挻搧摻杉柵櫼潸烻煔煽狦猭珊痁穇笘縿纔羶膻舢芟苫葠蔪衫襂襳跚釤顫髟", "shan3": "摻晱煔睒覢閃陝顃", "shan4": "僤儃儋剡善單墠墡壇嬗扇掞撣擅擔攙樿檀汕潬澹灗疝禪繕膳苫蟬蟺訕謆贍赸鄯釤騸鱔鱣", "shang1": "傷商場愓殤湯漡熵禓蔏螪觴謪踼鬺", "shang3": "上晌曏賞", "shang4": "上尚恦蠰鞝", "shang5": "裳", "shao1": "弰捎旓梢溲燒稍筲綃艄莦萷蕱蛸輎鞘髾鮹", "shao2": "勺招杓玿芍苕韶", "shao3": "少捎搜髾", "shao4": "削劭卲召哨少捎梢潲燒燿睄稍笤紹艄袑邵髾", "she1": "奓奢檨猞畬賒輋", "she2": "佘折抴揲碟磼舌蛇蛞邪鉈闍鞨", "she3": "捨舍", "she4": "厙射慴懾拾挕摵攝歙涉涻灄睫社聶舍葉蔎蠂設赦韘騇麝", "sheng1": "升呏姓昇泩湦焺牲狌生甥笙聲胜鉎陞鵿鼪", "sheng2": "憴澠繩譝鱦", "sheng3": "偗冼渻省眚箵", "sheng4": "丞乘剩勝圣媵嵊晟甸盛繩聖胜貹賸", "shi1": "厔失尸屍師施湤溮溼濕灑獅箷篩絁葹蓍虱蝨褷襹訑詩邿釃鉈鰤鳲鶳", "shi2": "什十堤塒姼寔實拾提時榯汁湜溡石示祏蒔蝕識鉐食鰣鼫鼭", "shi3": "使史始屎彖斯施狶痑矢豕駛", "shi4": "世事仕似侍勢咶唑嗜噬埶士奭媞室寺屍峙崼市式弒徥忕忯恀恃惿戺拭挈揓是柿栻楴檡殖殺氏沶液澤澨烒畤示筮簭繹耆肢舍舐蒔螫褆襫視觢試誓諟諡謚識豉貰赫跩踶軾适逝遞適遰遾郝醳釋鈰鉽銴鍉鎩飭飾齛", "shi5": "匙殖", "shou1": "收敊", "shou2": "熟", "shou3": "守手掱艏首", "shou4": "受售壽守授濤狩獸瘦綬", "shuan1": "拴栓閂", "shuan4": "專栓槫汕涮踹", "shuang1": "孀孇瀧爽礵艭雙霜驦鷞", "shuang3": "傱塽慡樉漺爽縔鷞", "shuang4": "淙灀", "shui2": "脽誰", "shui3": "水", "shui4": "娷帨涗睡稅裞", "si1": "俬偲傂凘司嘶廝徙思撕斯析楒榹泀澌磃禗禠私糸絲緦罳菥蕬虒蜤螄螔蟴謕鉰鍶颸鷥麗鼶", "si3": "偲死", "si4": "以伺似佁俟儩兕台司嗣四姒寺已巳柶梩汜泗洍涘祀祠笥耜肂肄肆蕼覗貄鈶雉食飴飼駟騃", "si5": "廁", "song1": "倯娀崧嵩忪松棇淞濍硹菘蘴蜙鍶鬆", "song2": "鬆", "song3": "傱嵷悚愯慫漎竦聳蓯駷", "song4": "宋訟誦送頌鬆", "suan1": "狻痠酸", "suan3": "匴篹", "suan4": "撰筭算蒜選", "suo1": "些傞唆嗍嗦娑戲抄挲摍桫梭沙犧獻簑縮羧莎莏蓑衰趖逡魦", "suo3": "嗩娑惢所溹瑣璅索縒葰鎈鎍鎖霍靃", "suo4": "些娑歲溹逤", "suo5": "嗦", "tai1": "台咍漦珆胎苔釐駘", "tai2": "儓台嬯抬斄旲檯炱箈籉能臺苔薹詒跆邰鈶颱駘鮐", "tai3": "呔騃", "tai4": "儓大太忕態汏汰泰溙燤鈦", "tan1": "嘽坍怹撣攤漢潬灘痑癱舑舔貪", "tan2": "倓單嘾埮壇婒彈惔憛曇榃橝檀沈淡湛潭澹炎燂痰癉磹禪繵罈蕁蕈藫覃談譚譠貚郯醰錟顃餤鷤", "tan3": "但僋儃嗿坦壇忐憳暺毯緂菼袒裧襢醓鉭", "tan4": "倓傝僋嘆埮探撢歎湠炭碳舕賧黮", "tang1": "湯簜羰膛蕩薚蝪趟蹚鏜鐋鐺閶闛闣鼞", "tang2": "唐堂塘嵣愓搪棠榶樘橖溏漟煻瑭磄糖膛蓎螗螳赯踼逿醣鎕鏜闛餳鶶", "tang3": "倘儻帑惝戃擴攩曭欓淌爣矘踼躺鎲钂黨", "tang4": "儻摥湯漡燙蕩趟蹚躺鐋", "ti1": "体剔擿梯睇踢銻體鵜鷈", "ti2": "偍厗啼堤姼媞崹徲惿折提是桋梯渧珶睇磃稊綈緹荑蕛虒蝭蟬褆諦謕趧踶蹄醍銻鍗隄題騠鯷鵜鶗鶙鷤", "ti3": "体徥衹醍體", "ti4": "俶倜剃剔啑嚏屜弟悌悐惕揥替棣楴殢洟涕狄瓋籊綈肆薙裼褅詆趯躍逖適錫鬄", "tian1": "吞天婖沾添苫酟", "tian2": "佃嗔填寘屇恬搷撣沺湉滇甜田甸畇畋盷瞋磌窴緂菾鈿鎮闐餂鷏", "tian3": "倎典填娗忝悿捵晪栝殄淟琠町痶紾腆舔蚕賟銛錪靦餂", "tian4": "掭添琠瑱睼瞋菾蚺跈", "tie1": "占帖怗怙惵聑蝶詀貼跕鉆", "tie3": "僣帖蛈銕鋨鐵驖鴩", "tie4": "呫帖餮", "ting1": "听庍廳朾桯汀渟烴町綎聽艼", "ting2": "亭停奠婷嵉庭廷挺楟榳渟筳聤莛葶蜓蝏閮霆鼮", "ting3": "侹圢娗挺梃烶珵珽町脡艇莛鋌閮頲", "ting4": "梃汀濎", "tong1": "侗恫恿桐炵熥痌絧蓪通", "tong2": "仝佟侗偅僮同哃垌峒彤曈朣桐橦氃洞浵湩潼烔爞犝狪獞痌瞳硐穜童筒筩粡絧膧艟茼蜼蟲衕詷赨酮重鉖銅餇鮦鼕", "tong3": "侗垌捅桶筒筩統", "tong4": "同慟痛衕通", "tou1": "偷媮愉鋀", "tou2": "投緰褕逗頭骰", "tou3": "妵敨斢紏蘣諭諳黈", "tou4": "埱透", "tou5": "頭", "tun1": "吞啍敦旽暾汭涒焞蜳", "tun2": "吨囤坉屯庉忳敦沌炖燉窀純肫臀芚豚軘逐飩魨", "tun3": "吨畽錪", "tun4": "褪", "wan1": "剜夗婠彎捥潫灣睕蜿豌貫鋺關", "wan2": "丸刓完岏忨抏捖汍烷玩紈翫芄蚖頑骫", "wan3": "倇免夗娩婉宛惋惌挽捥晚晥晼朊涴琬畹皖睕碗綩綰脘莞菀蜿踠輓鋺關", "wan4": "万卍妧忨捥掔槾澫睕瞣絻綄翫脕腕莧萬蔓薍蟃輐鄤鋄", "wei1": "倭偎危委威崴巍微愄揋椳渨溦烓煨燰猗畏癐萎葨葳薇蜲蝛覣詴踒逶隇隈魏", "wei2": "于偽唯喡囗圍圩堤寪峗峞嵬帷幃惟撝桅欈洈湋溈潿濰為犩癓磑維蓶覹觿違鄬醀鍏闈阢隹霺韋魏鮠", "wei3": "亹偉偽儰唯壝委娓寪尾峗峞崣崴嵬巋廆徫惟捼撱斖暐有椲沇洧浘濻瀢煒猥瑋瓗畏痏痿睢硊碨磈緯腲艉芛芟荽萎葦蒍薳蜲蜼諉趡踓鍡隗韙韡頠骫鮪", "wei4": "位味喂喡噲委媦尉慰未机渭濊為煟熨犚猥瓗畏眭瞶碨磑立罻胃菋萎蔚藯蘶蜼蝟衛褽謂讆躗遺鏏霨餧餵饖魏鮇", "wen1": "塭殟溫瘟褞豱轀韞", "wen2": "文汶炆玟紋聞芠蚊閺閿闅雯駇魰鳼鼤", "wen3": "刎吻呅呡忞抆昧桽歾眼穩笏紊限", "wen4": "免問妏娩搵汶璺紊紋絻聞脕顐鼤", "wu1": "兀剭喔嗚圬屋巫惡扜於杅杇歍汙污洿渥烏盓窏螐誣趶鄔釫鎢鴮", "wu2": "亡侉俉務吳吾唔峿幠廡憮捂梧橆毋母洖浯無牾珸瞀莁蕪蜈蝥譕郚鋘鋙鯃鷡鼯齬", "wu3": "五仵伍侮俉倵務午啎嘸埡堥塢嫵廡忤憮捂摀橆武母潕牾玝甒瞴碔膴舞蕪躌迕逜鵡", "wu4": "仡兀務勿卼唔嗚噁塢婺寤屼岉嵨忤悟惡戊扤旄晤杌柮梧沕烏焐物痦矹笏粅芴蘁蝥誤迕逜鋈阢雺霧霿靰騖鶩齀", "xi1": "俙傒僖兮凞吸咥咦咭唏喜嘻夕奚娭媐嬉屎屖嶲巂巇希徆徯恓息悉悕惁惜戲扱撕昔晞晰曦析栖桸棲榽樨橀欷歙氥汐浠淅溪潝烯焟熄熙熹燨爔犀犧狶獻琋瘜皙睎矽硒碏磎礂禧稀穸窸粞緆羛義羲翕肸脅腊膝茜莃菥蒠虒蜤蜥螅螇蟋蠵裼西觿訢詑誒譆谿豨豯貕蹊郗鄎酅醯釐釸錫錯鑴雟鵗鸂鼷", "xi2": "媳嵇嶍席徯擊槢檄濕習莔蓆薂蜴蝷褶襲覡謑謵郋鎴隰霫飁騱騽驨鰓鰼鼳", "xi3": "喜囍奊嬉屣徙枲歖洒洗漇灑璽瓕禧縰纚葸蓰蟢諰謑蹝釐銑", "xi4": "係傒卌卻呬呰咥塈屭忥怬恄愾慀戲摡既欯氣滊潟熂盻矽禊系細綌繫舄蔇蕮虩衋覤謚赩郤釳鈒闟隙餼鬩黖", "xia1": "呀呷岈毳煆瘕瞎蝦颬魻", "xia2": "俠假匣呷嗑夾峽徦押搳斜暇柙欱歃浹烚狎狹珨瑕硤碬磍祫笚給縖舝舺葭蕸螛赮轄遐郃鍜陜霞騢魻鶷黠", "xia3": "閜", "xia4": "下假呼哧唬嗄嚇夏廈暇煆罅芐諕謑鏬", "xian1": "仙仚佡僊先嘕奾妗姍姺嬐孅屳廯忺憸掀揱搟攕暹杴枮氙珗祆秈綅縿纖薟蘞褼襳跣躚釤銑銛錟韱馦鮮鶱", "xian2": "伭咁咸唌啣嗛堿妶娹嫌嫻弦慊慳憪探撏涎溓澖玹癇盷瞷矣礥絃羬肩胘舷蚿蛝誸諴賢銜閑閒鰔鷳鷴鹹麙", "xian3": "仙冼尟嶮幰憲搟攇毨洒洗濂灑灦烍燹獫獮玁省禒禰筅蘚蜆譣赻跣軒銑鍌險韅顯鮮", "xian4": "俔僩埳壏姭娊寰峴憪憲捍搚撊晛槏涀溓瀗獻現睍硍筧粯糮綖線縣羡羨脅腺莧蜆見譀豏賢軒鋧錎鏾闞限陷霰顈顯餡鮮黹鼸", "xiang1": "儴勷啍廂忀欀湘瓖相箱緗纕膷舡葙薌蘘襄鄉鑲香驤", "xiang2": "庠洋瓨祥翔詳迒降", "xiang3": "亨享嚮想攘晑曏薌蠁鄉響餉饗饟鯗", "xiang4": "傢像向嚮姠巷曏樣橡潒相缿萫蟓衖襐象鄉銗鐌降項鬨鱌", "xiao1": "俏削叟呺呼咻哨哮唬啋嗃嘐嘮嘵囂宵庨憢捎揱搜撓枵梟梢橚櫹歊毊消潚瀟灱烋熇燆猇獟獢痚痟睄硝穘窙箾簫綃縿翛肖脩膮莦萷蕭藃虈虓蛸蟂蠨謞謼譊踃逍銷霄騷驍驕骹髇髐魈鴞鵁鷍", "xiao2": "佼姣崤殽洨淆筊絞郩", "xiao3": "小曉皛筱篠簫謏轇", "xiao4": "俏傚哨哮嗃嘯奡孝恔效校殽涍漻澩熽爻狡笑肖芍茭薂詨較霄鷕", "xin1": "噷妡廞心忻新昕杺欣款歆炘盺芯莘薪訢辛鈊鋅鑫馨馫", "xin2": "尋撢枔橝礥襑鐔鬵", "xin3": "伈", "xin4": "信囟庍愖憖焮興舋芯釁阠", "xing1": "垶嫈惺星煋狌猩瑆箵胜腥興蛵觲騂", "xing2": "侀刑型娙巠形洐滎熒研硎行邢鈃鉶鋞陘餳", "xing3": "擤省箵醒", "xing4": "倖姓婞幸性悻杏涬興荇莕行鋞", "xue1": "削哮疶蒆薛靴韡", "xue2": "噱壆學岤嶨澩燢穴袕觷踅鷽", "xue3": "雪鱈", "xue4": "吷怴決泧泬滈炔狘瞲血謔趐蹻", "ya1": "丫亞厭呀啞埡壓婭孲押枒椏烏錏閘雅鴉鴨鵪", "ya2": "伢吾堐岈崖庌枒涯牙玡琊睚芽蚜衙邪雅齖", "ya3": "厊啞婭庌掗疋蕥雅鴉", "ya4": "亞呾圔圠埡堊壓姶婭御掗揠札枒椻歇氬浥潝牙犽猰玡砑碣磍穵窫聐襾訝軋輅輵迓錏齖齾", "ya5": "呀", "yan1": "偣傿厭咽奄嫣嬮崦弇懨橪殗殷洇淊淹湮漹焉煙燕珚胭腌臙菸蔫郾鄢醃閹閼黫黰", "yan2": "厂唌嚴埏壛壧妍娮孍岩嵒巖巡延挻揅楌檐氤汧沿炎狠狿癌研碞礹筵簷綖羡羬芫莚薟虤蜒衍言贗趼郔鉛鋋閆閻阽險顏鹽麙", "yan3": "偃儼兗剡匽厭厴喭噞嚴奄媕嬐孍嵃嶮巘庵弇愝戭扊抁掞掩揜晻曮棪椼沇沿淊淹渰演琰瓛甗眼礹縯罨菴萒蝘衍裺褗覃躽郾酀酓錟閻阭隒靨顩魘鰋鶠黤黭黶鼴齗齞齴龑", "yan4": "但俺傿匽厭咽唁喭嚥堰姲嬮嬿宴嵃彥掞掩敥晏曣棪椻殗沿洝涎淡淫溎演灩炎烻焰焱燄燕爓牪猒研硯艷蔫蜒裺觾言諺讌讞豔贗趼躽這酀酓醼釅閻雁顑饜騴驗驠鬳鳱鴈鴳鷃鷰鹽", "yang1": "佒央姎怏抰歍殃泱秧胦英鉠雵霙鞅鴦", "yang2": "佯垟婸將崵崸徉愓揚昜暘楊洋湯烊煬珜瑒痒瘍眻禓羊蛘詳鍚鐊陽颺鸉", "yang3": "仰佒卬坱岟懩攁映柍氧洋潒瀁炴痒癢紻蛘蝆軮鞅養", "yang4": "怏恙昂柍樣漾瀁烊煬羕鞅養", "ye1": "噎掖暍枒椰洇窫耶蠮饐", "ye2": "揶擨斜涂爺耶荼邪餘", "ye3": "也冶埜墅漜蠱野", "ye4": "偞咽喝堨夜射嶪抴拽捙掖揲擖擛擫曄曳楪業歋殕殗液澲焆煠熀燁瓛痷瞱聶腋葉謁鄴鍱鎑鐷靨頁饁黦", "yi1": "一伊依倚億咿噫壹嫛崎意懿戲揖椅檹欹毉洢漪犄猗瑿禕稦繄蛜衣褘詑郼醫銥鷖黝黟黳", "yi2": "一也佗侇儀儗匜台听咦圯坨夷姨姬媐它宜宧寲峓嶬嶷异彝怠怡恞戺拸搋暆杝柂栘桋椸沂沶焉熙熪狋珆瓵疑痍眙礙移箷簃維羠羡義耛胰荑萓蛇蛦螔衪袘觺訑詑詒誃謻譺貤貽跠迆迤迻遺酏鉈銕鏔阤頤顊飴鸃", "yi3": "乙以佁依倚偯奇尾崺嶬已巳扆扡掎敼旖椅檥歖汽猗矣硪礒綺臆艤苡蛾螘蟻譺踦轙迆迤酏醫醷釔鉯錡钀陭雉靉顗鳦齮", "yi4": "一乂乙亄亦仡伿佚佾俋億儗刈劓劮勩印厭唈喦嗌噎噫囈圛圪垼埶埸墿夕失奕嫕寱射屹嶧帟帠廙异弈弋役忔悒意憶懌懿抑挹掜搤擇斁施易昳晹曀杙枍枻栺槷槸樴檍欥欭殔殪毅泄泆洩洫浂浥渫溢潩澤澺瀷焱焲熠熤熼燚燡燱玴異疙疫瘞瘱益睪移紲絏縊繄繶繹羛義羿翊翌翳翼耴肄肊膉臆艗艾芅蓺薏藙藝蘙蜴衣衵袂袘裔裛襗襼詍詣誒誼謚譯議譺豷貤賹贀跇踦軼輗迭逸邑醳醷釋釴鎰鐿阣陭隿霅霬靾食饐驛鮨鶂鷁鷊鷖鷾黓齸", "yin1": "凐喑噾因垔堙壹姻愔慇欭歅殷氤洇湮瘖禋窨絪茵蒑蔭裀諲銦闇闉陰霒霠音韾駰", "yin2": "伒冘吟嚚圁圻垠夤婬寅崟憖檭欽殥沂淫湛潯烎狺硍芩苂荶蔩螾蟫言訢誾鄞釿銀霪鷣齗齦", "yin3": "吟听喑尹引朄檃殷湛潭濦癮硍縯蘟蚓螾讔趛輑酓釿鈏闇隱靷飲齗", "yin4": "傿印喑垽堷廕憖梀湚猌瘖窨胤蔭酳陰隱飲", "yin5": "粌", "yong1": "佣傭嗈噰墉壅嫞庸廱慵擁槦滽澭灉癰臃蕹邕郺鄘鏞雍雝饔鷛", "yong2": "喁嵱庸揘遇顒鰫", "yong3": "俑傛勇噰埇塎容嵱恿悀永泳涌湧澭甬禜筩臾蛹詠踊踴邕", "yong4": "佣傭用醟", "yu1": "于吁唹扜於汙淤渝瘀盓穻箊紆菸迂陓", "yu2": "与予于余俞吳吾喁喻圩堣堬妤娛媮崳嵎庾悇愉愚揄於旟杅楰榆歈歟歶毹氀汙渝湡漁澞牏狳玗瑜璵畬盂睮硢禺窬窳竽籅緰羭腧腴臾舁與艅芋萸蕍藇蘛虞蜍蝓螸衙衧褕覦諛謣貗踰輿逾邘郚釪鋙隃隅雓雩顒餘魚鮽鰅鱮鸆齵", "yu3": "与予俁偊傴噢噳圄圉嫗宇寙峿嶼庾愉敔斔斞栩梧楀瑀瘐祤禹穻窳羽臾與芋萭藇蘌蝺語貐鄅鋙雨頨麌齬", "yu4": "与余僪儥吁唷喻噊噢域堉嫗宛寓尉峪崛彧御悆惌愈慾懊或戫拗捥敔昱栯棜棫欥欲毓汩浴淢淯潏澦澳灪煜煨熨燏燠爩獄獝玉王痏瘉瘐癒矞砡礜禦禺稢篽籲粥緎繘罭聿育與舒芋苑菀菸蒮蓹蔚蕷薁藇蜮蟈衙袬裕語諭譽谷豫貍軉輍輿轝遇遹郁醧鈺鋊錥鐍閼閾隩雨雩預飫饇馭驈鬱鬻魊鱊鳿鴥鵒鷸鸒", "yuan1": "冤咽夗嫚嬽宛悁惌棩涴淵灁焆眢穿苑葾蒬蜎蜵蝝裷鋺駌鳶鴛鵷鼘", "yuan2": "允元原員園圓圜垣妧媛媴嫄捐援杬楥榞榬櫞沅湲源溒爰猿獂笎緣羱芫蒝蚖蝝蝯螈袁褑謜轅邍邧鎱阮隕騵鶢黿", "yuan3": "畹薳輐遠鋺願", "yuan4": "噮垸夗妴媛怨愿掾杬涓瑗禐緣肙苑裫褑遠院願", "yue1": "妁扚曰矱箹約", "yue3": "噦", "yue4": "刖囝岳嶽悅戉抈擽月樂樾櫟汋瀹焆爍爚玥礿禴籥粵臒蘥蚎蠖越趯躍躒軏鈅鉞鑠鑰閱髺鸑鸙黦龠", "yun1": "奫暈氳涒瘟蜵蝹贇頵馧", "yun2": "云伝勻員妘尹昀沄溳澐熉畇眃筍筠篔紜縜耘芸荺蕓鄖鋆雲", "yun3": "允宛抎殞溳狁盾荺褞賱鈗阭隕霣馻齫", "yun4": "員均孕尉怨惲慍暈枟煇熅熨緷縕芸苑菀薀蘊運鄆醞韗韞韻餫", "zao1": "傮槽糟蹧遭", "zao2": "鑿", "zao3": "早棗澡璅璪繅繰藻蚤", "zao4": "喿噪慥灶煰燥皁皂窖簉草譟趮躁造鑿", "zha1": "劄吒咋哳喳囃奓扎抯挓柤查楂樝渣溠潳皻紮蔖藸觰謯譇蹅齇齟", "zha2": "偞劄喋扎擖札渫炸煠箑蚻譗軋鍘閘霅鞢鰈", "zha3": "厏扎扠插眨砟苲苴觰謯鮓", "zha4": "乍吒咋哆喥奓怍搾柞柵榨溠炸痄砟笮苲蚱蜡詐醡霅鮓齰", "zha5": "喳", "zhan1": "亶占怗惉旃栴氈沾澶瞻站粘薝蛅襢詀詹譫邅霑餰饘驙魙鱣鳽鸇點", "zhan3": "儃嫸展嶄搌撣斬棧榐橏湔琖皽盞蹍輾醆颭黵", "zhan4": "佔偡占單孱嵁戰斬棧欃湛站綻蘸虥袒襢詀謙譧跕躔輚轏邅醮顫驏驙", "zhang1": "傽嫜張彰慞暲樟漲漳獐璋章粻蟑遧鄣障騿鱆", "zhang3": "仉掌漲長鞝", "zhang4": "丈仗墇嶂帳幛張扙杖漲瘴瞕章粀脹賬鄣長障", "zhe1": "嫬庶折蜇螫謶遮", "zhe2": "乇哲嘀囁悊慴慹折摺攝晢樀矺砓磔粍耷聑聶虴蜇蟄謫謺讋軼輒轍適陬馲鮿鷙", "zhe3": "堵者褚褶赭鍺", "zhe4": "斥柘浙淛潪蔗蟅這鷓", "zhe5": "著", "zheng1": "丁丞偵埥埩姃媜崝崢嶒征徵怔掙政朾正炡烝爭猙症癥眐睜禎箏篜糽綪聇脀蒸諍貞趟鉦錚鯖", "zheng3": "丞承拯掟撜整晸氶糽", "zheng4": "倀塣奠幀怔承政敞正氶浧爭町症瞠脀証諍證趟鄭", "zhi1": "之卮只吱媞戠指搘擲支智枝枳栺梔椥榰氏氐汁汥泜知祇祗禔秖秪稙織肢胑胝脂臸芝蘵蜘衹衼跂躓隻鳷", "zhi2": "侄值埴執姪嬂慹戠拓指摭植樀樴殖淔犆瓡直縶職膱臷蟙贄跖蹠蹢躑酈馽", "zhi3": "厎只咫址坁徵恉扺抵指旨枳栺止汦沚淽疻砥祁祇祉紙絺耆脂芷藢蚔衹觝趾軹酯阯黹", "zhi4": "之乿伎偫傂制剬厔咥坁埃墆寘實峙帙幟庢庤廌彘志忮恃懥懫扻挃捗摨摯擲擳擿支昵晊晢智杝柣桎楖樲櫍櫛歭治泜洷滍滯潪瀄炙狾猘畤疐痔痣瘈知礩示祑秩秷稙稚窒紩緻織置翐胵膣至致薙虒蛭螲袟製覟觶誌識豸質贄跱踶蹛躓輊迣遲郅銍銴鋕鑕阤陟雉駤騭騺鳩鴙鶨鷙", "zhong1": "中伀妐彸忠忪柊橦潼炂盅童籦終舂舯蔠蚣蝩螤螽衷鈆鍾鐘鼨", "zhong3": "冢塚尰煄种種穜腫董踵", "zhong4": "中仲偅堹狆眾祌种種穜緟蟲衶衷踵重", "zhou1": "侏侜倜僽周啁喌婤州徟掫洀洲淍珘皺盩粥縐翢舟調諏謅譸賙輈輖週銂騆鬻鵃", "zhou2": "妯柚碡軸", "zhou3": "帚扭睭肘胕舳鯞", "zhou4": "僽冑咒咮啄喙噣宙扭晝椆注甃皺祝籀紂紬縐繇育胄薵詶軸逐酎鈾駎騶驟", "zhu1": "之侏咮搊朝朱株楮櫧櫫洙瀦珠硃絑茱藷藸蛛蝫蠩袾誅諸豬跦邾銖騶鮢鴸", "zhu2": "噣孎斀斸朮欘泏灟燭瘃磩窋竹竺笁筑築篴舳茁茿蓫蚰蠋蠾躅軸逐钃鱁鸀", "zhu3": "主囑屬拄斗斸枓柱渚煮矚砫罜褚陼麈", "zhu4": "主予佇住助咮壴宁尌嵀庶杼柱柷樦櫡殶泞注澍炷疰眝砫祝祩筑箸築篫紵紸羜翥芧苧莇著蛀註貯跓跙軴逗鉏鉒鑄阻除馵駐麆", "zhuo1": "倬卓拙捉掇桌棳棹涿淖焯穛穱肫蝃蠿踔", "zhuo2": "剢勺卓啄啅啜噣圴彴捔擢斀斫斮斲杓梲椓櫡汋浞涿準濁濯灂灼焯燋琢矠禚箸篧籗繳罬茁著蓔蕞藋蠗諑謶趠趵蹠躅酌鐲鵫鷟", "zi1": "仔吱呲咨姿孖孜孳崰嵫栥椔次淄滋澬玆璾甾疵秶粢紎純緇芓茲菑薋觜訾諮貲資趑載輜鄑鈭錙鎡髭鯔鶅鼒齊齍齜", "zi2": "蓻", "zi3": "仔吇呰姊子杍梓沝泚滓矷秭笫籽紫耔胏芓茈訾訿跐釨鋅", "zi4": "事倳剚姿字恣柴漬牸眥穧胔胾自芓菑資輜", "zi5": "子"}
# ==== 自動產生結束 ====
