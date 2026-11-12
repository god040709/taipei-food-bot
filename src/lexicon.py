"""
詞庫、提示詞與文字處理（地區、類別、子類型、地標、同音錯字、捷運站）。
捷運站名從 data/mrt_stations.csv 讀；同音錯字的拼音表在 data/lexicon_generated.json。
兩個檔案都由 tools/build_lookups.py 產生。
"""
import csv
import json
import os
import re
from typing import Optional

_STATIONS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mrt_stations.csv")
try:
    with open(_STATIONS, encoding="utf-8") as _f:
        MRT_STATIONS = [row["name"] for row in csv.DictReader(_f)]
except FileNotFoundError:
    MRT_STATIONS = []
try:
    with open(os.path.join(os.path.dirname(_STATIONS), "lexicon_generated.json"), encoding="utf-8") as _f:
        _gen = json.load(_f)
except FileNotFoundError:
    _gen = {}
TERM_READINGS = _gen.get("TERM_READINGS", {})
PINYIN_CHARS = _gen.get("PINYIN_CHARS", {})

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


PARSE_PROMPT = f"""你是台北餐廳推薦系統的「搜尋條件抽取器」。讀對話紀錄，輸出使用者「現在」要的搜尋條件。

欄位：
- intent：使用者最後一句話的意圖。
  search＝提出或修改條件、回答助理的反問；想吃東西、肚子餓、想找吃的（就算還沒說任何條件）也是 search；
  new＝提出一個跟之前推薦無關的全新需求（重新講了地區、類型等條件）；
  more＝要換一批、推薦別的；
  reject＝不喜歡、吃過、不想去推薦過的某幾間店；
  chat＝問你會做什麼、打招呼、道謝、閒聊、問推薦過的店的細節、請你從推薦過的店裡挑一間，
       或是跟找餐廳無關的話；
  reset＝要重新開始、清除所有條件；
  categories＝問有哪些料理類型可以選（例如「有什麼類型」「有哪些料理」）。
       說「不管什麼類型」「什麼類型都可以」是類型不限，不是 categories。
- district、category、party_size（人數）、budget（每人預算上限，台幣）：目前要用的條件。
  沒有改動的就沿用之前的；從來沒提過、或使用者說不限，就填 null。
- waived：使用者「明確說」不限、不在意、都可以的欄位名稱（district / category / party_size / budget）。
  只是沒提到的欄位不要放進來；使用者說「不知道」「不確定」「沒想法」不算不限，不要放進來。
- choose_category：true＝使用者請你幫他挑一種料理（例如問你推薦吃什麼、要你決定吃什麼）；沒有就填 null。
  這時 intent 是 search（不是 chat），category 填 null，不要自己挑。
  使用者說換一種、換別的類型，但沒說要換成哪一種時（不管上一次的類型是誰決定的），choose_category 也是 true。
- 以下欄位只看使用者「最後一句話」，沒提到就填 null 或空陣列（沒提到不可以填 false）：
  reject：不喜歡的店，只能從「最近推薦過的店」清單選，「第 2 間」就是清單第 2 家。
  keep：換一批時要保留的店（例如「第 1 間留著」），只能從「最近推薦過的店」清單選。
  outside_area：使用者說的地點不在台北市（例如新北市的板橋、新莊）時，照原話填地名。
  unknown_food：使用者想吃的料理不屬於下面的 category（例如越南菜、印度料理）時，照原話填，這時 category 填 null。
  near：想在某個捷運站或地標附近吃，照使用者的原話填站名或地標。
  no_chain：true＝不要連鎖店；false＝說了連鎖也可以。
  hidden_gem：true＝想要比較少人知道、隱藏版、私房的店；false＝說了有名的也可以。
  budget_min：每人預算下限。
  sort：使用者「明確強調」最在意的一項才填，只是提到不算（例如只說「公館附近」不是 near_first）：
    price_high＝想要最貴、最高級的；price_low＝想要最便宜的、便宜最重要；
    near_first＝越近越好、不想走太遠、近最重要；rating＝評價最重要、要好吃、不想踩雷；
    popularity＝要有名的、人氣高、大家都去的。沒有強調就填 null。
  relax：使用者「明確說可以放寬」的條件：distance＝走遠一點沒關係、遠一點也可以；
    budget＝想多花一點（預算再高一點、預算提高一點、貴一點也可以、超過一點沒關係）。沒說就填空陣列。
  填 sort 或 relax 時，budget 維持原本的金額，不要自己調高或調低（要加多少由程式計算）。
  avoid_categories：明確說不想吃、剛吃過、要排除的類別。使用者說想吃的類別絕對不能放這裡。
    只是說換成別的類型（例如「換成拉麵」），不代表不要原本的類型；不能從口味自己推論類別（例如「不喜歡辣」不等於不要某一類）。
  categories：同時接受好幾種類別時，全部列出。
  unsupported：使用者在意、但只靠店名和價格判斷不了的條件（營業時間、停車、包廂、寵物、素食、氣氛等），用簡短的詞。

換算規則：
- district 只能是：{"、".join(DISTRICTS)}。可以省略「區」。地標只有在非常確定所屬行政區時才填。
- category 只能是：{"、".join(CATEGORIES)}。
  同義詞要對應，例如麻辣鍋、壽喜燒、羊肉爐→火鍋，壽司→日式料理，韓國菜→韓式料理，小吃、夜市→平價小吃，
  早餐→早午餐，蛋糕、下午茶→甜點，喝酒→酒吧，中菜、合菜→中式餐廳，漢堡、牛排→美式餐廳。
- 中文數字要轉成數字（一千＝1000、1k＝1000）。
- 總預算（總共、一共、加起來）：total_budget 填總金額（不要除以人數），budget 填 null，程式會自己換算。
- 預算是區間時，budget 填上限、budget_min 填下限；只說「X 以上」時 budget 填 null、budget_min 填 X。
- 「再便宜一點」（要調低預算，不是「越便宜越好」）：budget 變成原本的預算減 200（最少 100）。
- 使用者可能打錯同音字或近音字，請依讀音判斷原本的意思。
- 助理上一則是反問（以 🤔 開頭）時，使用者這句話是在回答反問，要跟之前的需求合併。
"""

REASON_PROMPT = """你是台北餐廳推薦助理。根據使用者的需求和每家餐廳的資料，為每一家寫推薦理由，並寫一段開場。
規則：
- 每家寫 1~2 句，說明這家店的特色，以及為什麼適合這位使用者（人數、預算、想吃的類型）。
- min_price ~ max_price 是「多數人用餐的每人花費區間」，不是最低消費或最高消費。
- 只能根據資料裡的內容：名稱、類別、價格、distance（距離，公尺）、facts（客觀事實）、
  summary（店家簡介）、highlights（服務特色）、reviews（顧客評論摘錄）。資料沒有的菜色、裝潢、營業時間不可以編造。
- 只有 reviews 有內容時才可以提到顧客評論。
- 只有資料裡有 distance 時才可以提到距離或「附近」。
- 不要寫推薦指數、分數或計算方式。不要每家都用一樣的句型開頭。
- 預算都是「每人」的金額。不要自己把預算乘上人數算出總共多少錢（條件裡有寫總預算時才可以提）。
- 不要在推薦理由裡提預算超出：價格超出預算的店，畫面上已經另外標示了。
- 不要提到「系統」「系統說明」「資料」這些字眼，直接用自然的方式說。
- intro 是簡短的開場（一到三句），自然地回應使用者最新那句話。
  條件裡如果有「要讓使用者知道的事」（例如放寬了範圍或預算、總預算換算、資料沒有的資訊），要用自己的話、
  一兩句自然地帶過（不要照抄原句，也不要寫「要讓使用者知道」這類字），不用寫出放寬到多少；
  沒有的話就只寫一句。搜尋條件畫面上已經顯示了，不用重複列出。
  使用者沒說過、也沒發生的事不可以提。
- reasons 的數量和順序要和餐廳資料完全一樣。
"""

CHAT_PROMPT = """你是「台北美食推薦機器人」，用繁體中文（不要出現簡體字）、自然友善地回覆，像朋友聊天一樣，不要用制式的開場或結尾。
使用者這句話不是要找新的餐廳（例如問你會做什麼、打招呼、道謝、問推薦過的店的細節、請你從推薦過的店裡挑一間）。

你能做的事（只能說這些，不可以說你會做清單以外的事，例如訂位、查營業時間、查菜單）：
- 依台北 12 個行政區、料理類型、人數、每人預算推薦餐廳；資訊不夠時會反問
- 找某個捷運站或地標附近的店
- 細分的種類，例如麻辣鍋、壽喜燒、壽司（從店名判斷）
- 不要連鎖店、找評論少但評分高的隱藏版、預算區間、同時接受幾種料理、排除不想吃的料理
- 換一批、排除不喜歡或吃過的店
- 每家附上推薦理由、價格、地址和 Google Maps 連結

回覆規則：
- 1~3 句。只挑跟使用者問題有關的部分說，不要把清單整份念出來。
- 跟吃飯無關的問題，簡短說明你專門推薦台北的餐廳，不要回答那個問題本身。
- 不可以說出「最近的推薦內容」以外的任何店名；還沒有推薦過時，不要提任何店名，請使用者說出想吃什麼。
- 關於推薦過的店，只能根據下面「最近的推薦內容」回答；資料沒有的（營業時間、菜單、停車、訂位方式…）
  就說資料裡沒有，可以點 Google Maps 連結查看。不要編造餐廳或資料。
- 請你挑一間時，只能從最近的推薦裡挑，並用推薦內容裡的資訊說明理由。
- 最後用自己的話，自然地邀請使用者說出想吃什麼或想怎麼調整；對話裡已經說過的條件不要再問，只問還不知道的。
"""

SAY_PROMPT = """你是「台北美食推薦機器人」，正在跟使用者對話，現在要用繁體中文對使用者說一小段話。
- 像朋友聊天一樣自然口語，1~2 句。不要用制式的開場，不要編號或條列。全部用繁體字，不要出現簡體字。
- 只能根據下面給的資訊，不可以編造。不要提到「系統」這類內部用語。
- 如果提到「下面會另外列出」的清單，不要在你的話裡重複清單內容，自然地帶到下面的選項就好。
"""

# 檢查 LLM 寫的反問有沒有問到每一項（沒問到的，程式會補一句）
ASK_HINT_WORDS = {
    "district": ["區", "哪裡", "哪邊", "地方", "地點", "附近", "一帶"],
    "category": ["吃什麼", "類型", "料理", "種類", "想吃", "口味"],
    "party_size": ["幾個人", "幾位", "幾人", "人數", "多少人"],
    "budget": ["預算", "元", "價位", "花費", "多少錢", "價錢", "價格"],
}

# 使用者強調的重點（「最便宜的」「越近越好」「評價最重要」）→ 推薦的排序方式
SORTS = ["price_high", "price_low", "near_first", "rating", "popularity"]
SORT_LABELS = {"price_high": "依價格由高到低", "price_low": "依價格由低到高", "near_first": "越近越好",
               "rating": "評價優先", "popularity": "人氣優先"}
# 使用者說可以放寬的條件（「走遠一點沒關係」「貴一點沒差」）
RELAXES = ["distance", "budget"]
RELAX_LABELS = {"distance": "距離可放寬到 1.5 公里"}  # 會記在 🔎 搜尋條件那一行、下一句沿用的
# 這句話有這些字，才採用 LLM 判斷的「加預算」（LLM 常把前面說過的再填一次）
BUDGET_UP_WORDS = ["貴", "高", "多", "超過", "加"]
# 這句話有這些字，才採用 LLM 判斷的「預算下限」（LLM 會受前一句「1000以上」影響，把下一句的「800」也當成下限）
BUDGET_MIN_WORDS = ["以上", "起跳", "到", "至", "~", "～"]
# 這次需求有這些字，才採用 LLM 判斷的「總預算」（LLM 偶爾把「2人 500」的 500 當成兩人合計）
TOTAL_WORDS = ["總", "共", "加起來", "合計"]
# 不算正餐的類型：使用者沒指定類型時（類型不限、幫你挑），不推這兩類
NON_MEAL = ["酒吧", "甜點"]
# 「越近越好」但不知道使用者在哪：反問時加上這一行（下一句讀回來，知道剛剛問過車站）
NEAR_ASK_MARK = "📍 依距離排序需要知道你在哪個捷運站附近"

# 使用者請機器人挑料理類型：反問時會加上這一行（下一句讀回來，就不會再問類型）
CHOOSE_MARK = "🎲 料理類型交給我挑"

# 使用者說「不知道」時不算「不限」（要給具體選項讓他挑）；LLM 偶爾會把它當成不限，用這個保險
UNSURE_WORDS = ["不知道", "不確定", "沒想法", "不清楚", "還沒想"]

# 「重新開始」：回覆最後加上這一行，之後只看這一行以後的對話
RESET_MARK = "🔁 已重新開始"
RESET_WORDS = ["重新開始", "重來", "全部重來", "清除條件", "清空條件"]

# 反問訊息的開頭符號（用來判斷上一則是不是反問）
ASK_MARK = "🤔 "
# 閒聊回覆的開頭符號（不影響條件的沿用）
CHAT_MARK = "💬 "
# 「不喜歡／吃過」：強烈的說法單獨出現就算；「不要」這類要搭配指定的店（第幾間、店名）才算
REJECT_STRONG = ["不喜歡", "吃過", "去過", "踩雷", "難吃", "沒興趣", "不想去", "不想吃", "不考慮", "排除"]
REJECT_SOFT = ["不要", "換掉", "跳過", "pass"]
REJECTED_MARK = "🚫 不再推薦："
# 全新的需求：之前只對那次搜尋有效的條件、不想去的店都不再套用（之後讀回來時從這裡重新算）
NEW_REQUEST_MARK = "🆕 這是新的需求，"
# 除錯模式的區塊開頭（讀回對話時，這行以後都拿掉）
DEBUG_MARK = "<details>\n<summary>🔧 除錯資訊</summary>"
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

# 「推薦別的」：沿用上一次的條件，排除已經推薦過的店
MORE_WORDS = ["別的", "別家", "別間", "換一批", "換一些", "換幾間", "換幾家", "其他選擇", "其他的",
              "其他間", "其他家", "還有嗎", "還有沒有", "還有其他", "再推薦", "再給我", "下一批",
              "更多選擇", "不要這些", "不要這幾"]
# 「日式」「韓式」後面直接接別的類別時（「日式燒肉」「韓式炸雞」），只是形容，不算另一個類別
STYLE_WORDS = {"日式", "韓式", "美式", "義式", "意式", "泰式", "中式", "台式", "港式", "日本", "韓國"}

# ---- 其他條件（記在「🔎 搜尋條件」那一行後面，下一句沿用）----
# 跟著「人」走的偏好：使用者提出全新的需求時也保留（其他條件只屬於那一次搜尋）
PERSON_PREFS = {"no_chain"}
# LLM 判斷「取消」某個偏好時，使用者原話裡要有這些字才採用
PREF_OFF_WORDS = {"no_chain": ["連鎖"], "hidden_gem": ["有名", "知名", "熱門", "隱藏", "人氣"]}
# 地標 → 最近的捷運站（站名清單 MRT_STATIONS 從 data/mrt_stations.csv 讀）
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

# 規則解析時，「不要火鍋」「剛吃過拉麵」「火鍋以外」的類別不能當成想吃的類別
# （要排除哪些類別由 LLM 判斷；這裡只是避免規則把它誤認成想吃的）
NEG_BEFORE_RE = r"(?:不要|不想要|不想吃|不想|不吃|別吃|別|剛吃過|剛剛吃過|昨天吃過|吃過|吃膩了?|除了|不考慮|排除|避開)\s*(?:再)?\s*(?:吃)?\s*$"
NEG_AFTER_RE = (r"^\s*(?:以外|之外|除外|吃膩|吃到膩|吃過了|剛吃過"
                r"|(?:(?:就|先)?不要|不吃|免了)\s*(?:了|啦|喔)?\s*(?:$|[\s，,。.!！?？~～]))")
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
        terms.sort(key=lambda t: (-len(t[0]), t[0]))  # 長的詞先比對；同長度依字排，結果固定
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


def strip_marks(text: str) -> str:
    """去掉 LLM 模仿對話紀錄而加在開頭的符號（💬 🤔），避免越疊越多。"""
    return re.sub(r"^(?:\s*(?:💬|🤔)\s*)+", "", text or "").strip()


# LLM 偶爾會寫出簡體字：常見的換成繁體（店名裡本來就有的「点、满、处、来、边」不換，免得對不上店名）
SIMPLIFIED = str.maketrans(dict(zip(
    "间区选择个错这说们为时会对适价欢还种样吗么应让经过东门实质环气评论优饭烧锅汤鸡鱼虾预荐厅务体验热闹订够总条许别话认识读确从发现觉没带单将无关难远钟楼岛线专营业",
    "間區選擇個錯這說們為時會對適價歡還種樣嗎麼應讓經過東門實質環氣評論優飯燒鍋湯雞魚蝦預薦廳務體驗熱鬧訂夠總條許別話認識讀確從發現覺沒帶單將無關難遠鐘樓島線專營業")))


def to_traditional(text: str) -> str:
    return (text or "").translate(SIMPLIFIED)


def as_list(value) -> list:
    """LLM 該回清單的欄位，有時會回一個字串（「停車」）：當成只有一項，不要拆成一個一個字。"""
    if isinstance(value, list):
        return value
    return [value] if isinstance(value, str) and value.strip() else []
