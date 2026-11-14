"""
產生訓練用的對話（只有使用者說的話；機器人的回覆在收集資料時由真的機器人產生）。

說法是用元件隨機組合的：地區／捷運站、料理類型（正式名稱、同義詞、子類型）、人數、預算、其他條件，
再加上推薦之後常見的追問（換一批、不要第幾間、換類型、加減預算…）。
評估用的對話另外取自 tests/scenarios.py，兩邊的說法不一樣，避免模型只是背答案。
"""
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lexicon import CATEGORIES, CATEGORY_SYNONYMS, DISTRICTS, MRT_STATIONS, SUBTYPES  # noqa: E402

CN = {1: "一", 2: "兩", 3: "三", 4: "四", 5: "五", 6: "六", 8: "八", 10: "十"}
CN_MONEY = {300: "三百", 500: "五百", 600: "六百", 800: "八百", 1000: "一千", 1200: "一千二", 1500: "一千五"}
BUDGETS = [150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000]
EXTRAS = ["不要連鎖", "不想吃連鎖店", "越近越好", "離我近一點", "評價高一點的", "要評分最好的", "人氣高的",
          "便宜就好", "越便宜越好", "想吃高級一點的", "要有停車位", "要有包廂", "可以帶寵物",
          "想找私房一點的", "不要太有名的", "走遠一點沒關係", "貴一點沒差", "要安靜一點"]
OPENERS = ["{}", "{}", "{}", "我想{}", "幫我找{}", "{}有推薦嗎", "{}，有什麼推薦", "請問{}", "想問{}"]


class Gen:
    def __init__(self, seed: int):
        self.r = random.Random(seed)

    def pick(self, seq):
        return self.r.choice(list(seq))

    # ---- 元件 ----
    def where(self):
        """回傳 (說法, 是不是捷運站)。"""
        if self.r.random() < 0.25:
            s = self.pick(MRT_STATIONS)
            return self.pick([f"{s}站附近", f"捷運{s}站", f"{s}站", f"{s}站周邊"]), True
        d = self.pick(DISTRICTS)
        return self.pick([d, d[:-1], d[:-1], f"在{d[:-1]}", f"{d}這邊", f"{d[:-1]}一帶"]), False

    def food(self):
        """回傳 (說法, 類別)。"""
        roll = self.r.random()
        if roll < 0.45:
            c = self.pick(CATEGORIES)
            return self.pick([c, c, f"想吃{c}", f"吃{c}"]), c
        if roll < 0.75:
            c = self.pick([c for c in CATEGORIES if CATEGORY_SYNONYMS.get(c)])
            w = self.pick(CATEGORY_SYNONYMS[c])
            if w.startswith(("喝", "吃", "小酌")):  # 本身就是動作的說法（喝酒）不再加「想吃」
                return w, c
            return self.pick([w, f"想吃{w}", f"來點{w}"]), c
        name = self.pick(SUBTYPES)
        return self.pick([name, f"想吃{name}"]), SUBTYPES[name][0]

    def people(self):
        n = self.pick([1, 2, 2, 2, 3, 4, 4, 5, 6, 8, 10])
        forms = [f"{n}人", f"{n}個人", f"{n}位"]
        if n in CN:
            forms += [f"{CN[n]}個人", f"{CN[n]}位"] + ([f"我們{CN[n]}個"] if n > 1 else [])
        if n == 1:
            forms += ["一個人", "我自己", "自己一個人"]
        if n == 2:
            forms += ["我跟朋友兩個", "兩個人", "我和男友"]
        if n == 4:
            forms += ["一家四口"]
        return self.pick(forms), n

    def money(self, n):
        b = self.pick(BUDGETS)
        forms = [f"每人{b}", f"預算{b}", f"{b}以內", f"{b}左右", f"一人{b}", f"{b}元", f"{b}", f"每個人{b}上下"]
        if b in CN_MONEY:
            forms += [f"每人{CN_MONEY[b]}", f"預算{CN_MONEY[b]}"]
        if self.r.random() < 0.12:
            return f"{b}到{b + self.pick([200, 300, 500])}", b
        if self.r.random() < 0.08:
            return f"{b}以上", b
        if self.r.random() < 0.10 and n:
            return self.pick([f"總共{b * n}", f"總預算{b * n}", f"加起來{b * n}"]), b
        return self.pick(forms), b

    def join(self, parts):
        self.r.shuffle(parts)
        text = self.pick([" ", " ", "，", ""]).join(parts)
        if text.startswith(("想", "我", "吃", "來點")):
            return text  # 已經是完整的說法，不再加「我想」「幫我找」
        return self.pick(OPENERS).format(text)

    # ---- 一段對話 ----
    def request(self, k: int):
        """說出 k 項條件（地區、類型、人數、預算裡挑），回傳 (那句話, 用到的類別)。"""
        fields = self.r.sample(["where", "food", "people", "money"], k)
        parts, cat, n = [], None, None
        if "people" in fields:
            text, n = self.people()
            parts.append(text)
        for f in fields:
            if f == "where":
                parts.append(self.where()[0])
            elif f == "food":
                text, cat = self.food()
                parts.append(text)
            elif f == "money":
                parts.append(self.money(n)[0])
        if self.r.random() < 0.25:
            parts.append(self.pick(EXTRAS))
        return self.join(parts), cat

    def answer_missing(self):
        """機器人反問之後的回答。"""
        roll = self.r.random()
        if roll < 0.15:
            return self.pick(["都可以", "隨便", "都行", "沒差", "你決定"])
        if roll < 0.25:
            return self.pick(["不知道", "還沒想好", "沒想法"])
        if roll < 0.35:
            return self.pick(["地區不限", "類型都可以", "預算不限", "你幫我挑一種吃的", "人數還不確定"])
        return self.request(self.pick([1, 2, 2, 3]))[0]

    def followup(self, cat):
        k = self.pick([1, 2, 3, 4, 5])
        other = self.pick([c for c in CATEGORIES if c != cat])
        options = [
            "換一批", "還有別家嗎", "再推薦幾間", "其他的呢",
            f"第{k}間不要", f"第{'一二三四五'[k - 1]}間吃過了", f"{k}號不喜歡", "最後一間不要", "前兩間都去過",
            f"第{k}間留著 其他換掉", "這些都不要",
            f"換成{other}", f"改吃{other}好了", f"那{other}呢",
            "再便宜一點", "便宜一點的", "可以貴一點", "預算再高一點", f"預算改成{self.pick(BUDGETS)}",
            f"改成{self.pick([2, 3, 4, 6])}個人", f"換到{self.pick(DISTRICTS)}", "地區不限",
            "不要連鎖", "連鎖也可以", "越近越好", "評價高一點的",
            "謝謝", "你是誰", f"第{k}間在哪", "有哪些類型可以選", "你幫我挑一種", "換一種", "重新開始",
            self.request(self.pick([3, 4]))[0],  # 全新的需求
        ]
        if cat:
            options += [f"{cat}吃膩了", f"不想吃{cat}了", f"今天不要{cat}", f"剛吃過{cat}，換別的",
                        f"{cat}不要了 改吃{other}"]
        return self.pick(options)

    def dialogue(self):
        k = self.pick([1, 2, 3, 3, 4, 4, 4])
        first, cat = self.request(k)
        turns = [first]
        if k < 4 and self.r.random() < 0.7:
            turns.append(self.answer_missing())
        for _ in range(self.pick([0, 1, 1, 2])):
            turns.append(self.followup(cat))
        return turns


# ---- 補強資料：針對第一次微調後小模型的弱點（閒聊、無關問題、排除的說法、菜名、範圍外、排序、沒想法） ----
# 評估情境（tests/scenarios.py、eval_cases.json）裡出現過的句子會自動排除，避免模型背答案
DISHES = {
    "泰式料理": ["綠咖哩", "月亮蝦餅", "冬蔭功", "泰式炒河粉", "涼拌青木瓜", "椒麻雞"],
    "日式料理": ["握壽司", "天婦羅", "烏龍麵", "蕎麥麵", "親子丼", "章魚燒"],
    "韓式料理": ["石鍋拌飯", "辣炒年糕", "人蔘雞湯", "海鮮煎餅", "韓式燉雞"],
    "義式料理": ["燉飯", "千層麵", "青醬麵", "焗烤"],
    "美式餐廳": ["烤肋排", "熱狗堡", "水牛城雞翅"],
    "中式餐廳": ["東坡肉", "宮保雞丁", "麻婆豆腐", "糖醋排骨", "北京烤鴨"],
    "平價小吃": ["蚵仔煎", "肉圓", "大腸麵線", "鹽酥雞", "胡椒餅", "滷味"],
    "早午餐": ["班尼迪克蛋", "法式吐司", "貝果"],
    "火鍋": ["鍋物", "牛奶鍋", "番茄鍋"],
    "拉麵": ["味噌拉麵", "醬油拉麵", "家系拉麵"],
    "小籠包": ["蟹黃小籠", "鮮肉小籠"],
    "甜點": ["提拉米蘇", "舒芙蕾", "可麗餅", "芒果冰"],
}
OUTSIDE = ["板橋", "新莊", "中和", "新店", "土城", "蘆洲", "林口", "樹林", "桃園", "基隆", "新竹", "台中", "高雄"]
SORTS = ["想吃高檔一點", "要吃好一點的", "預算不是問題 要好吃", "想奢侈一下", "找最便宜的", "省錢為主",
         "要評價最好的", "不想踩雷", "要排隊名店", "人氣最旺的", "最近的那幾間"]
CHAT_AFTER = ["好喔", "嗯嗯", "OK", "了解", "收到", "讚", "太好了", "酷耶", "哈哈", "辛苦了", "謝啦", "3Q",
              "你人真好", "第{k}間要訂位嗎", "第{k}間有停車場嗎", "第{k}間怎麼去", "第{k}間貴不貴",
              "你最推哪間", "哪一間比較適合約會", "這些你都吃過嗎", "你是機器人嗎", "你叫什麼名字",
              "你是誰做的", "你怎麼知道這些店的", "幫我訂位", "可以外帶嗎"]
OFF_TOPIC = ["幫我寫數學作業", "3加5等於多少", "明天天氣如何", "推薦一部電影", "幫我翻譯hello", "你會寫程式嗎",
             "教我煮咖哩", "台北101有多高", "股票會漲嗎", "講個笑話", "我好無聊", "今天好累", "我失戀了",
             "幫我寫情書", "asdfgh", "？？？", "😂", "hello", "how are you", "你好", "嗨", "在嗎", "早安"]
AVOID_NOW = ["{c}以外的都可以", "除了{c}其他都好", "{c}先不要", "這次跳過{c}", "{c}最近吃太多了", "{c}有點吃膩",
             "對{c}沒什麼胃口", "上禮拜才吃{c}", "{c}吃到不想再吃", "不想再看到{c}了", "{c}就算了吧",
             "{c}今天不考慮"]
AVOID_FIRST = ["除了{c}什麼都可以", "{c}以外都可以", "不吃{c}", "不要{c}", "{c}跟{c2}以外都行", "{c}和{c2}都不要"]
UNSURE = ["還沒想好欸", "不曉得", "想不到", "沒什麼想法", "你問倒我了"]
CHOOSE = ["你幫我決定吃什麼", "想不到要吃什麼 你挑", "類型隨你", "吃什麼都好 你選"]
MOODS = ["考完試想放鬆", "下班好累", "今天發薪水", "要幫朋友慶生", "第一次約會", "寫完報告了", "下雨天"]


class FocusGen(Gen):
    def meal_cat(self):
        return self.pick([c for c in CATEGORIES if c not in ("酒吧", "甜點")])

    def full_request(self, food=None, where=None, extra=None):
        """四項條件都說的需求；food / where 可以換成指定的說法。"""
        n_text, n = self.people()
        parts = [where or self.where()[0], food if food is not None else self.food()[0], n_text, self.money(n)[0]]
        parts = [p for p in parts if p]
        if extra:
            parts.append(extra)
        return self.join(parts)

    def no_food_request(self, extra):
        n_text, n = self.people()
        return self.join([self.where()[0], n_text, self.money(n)[0], extra])

    def dialogue(self):
        roll = self.r.random()
        k = self.pick([1, 2, 3, 4, 5])
        if roll < 0.22:  # 推薦完之後閒聊、問無關的事
            turns = [self.full_request(), self.pick(CHAT_AFTER + OFF_TOPIC).format(k=k)]
            if self.r.random() < 0.3:
                turns.append(self.followup(None))
            return turns
        if roll < 0.32:  # 一開始就閒聊、問無關的事，之後才說需求
            turns = [self.pick(OFF_TOPIC + ["你是誰", "你能幹嘛", "你會推薦什麼"])]
            if self.r.random() < 0.6:
                turns.append(self.full_request())
            return turns
        if roll < 0.52:  # 推薦完之後排除目前的類型（沒說換什麼）
            c = self.meal_cat()
            return [self.full_request(food=c), self.pick(AVOID_NOW).format(c=c)]
        if roll < 0.62:  # 一開始就說不要哪些類型
            c, c2 = self.r.sample([c for c in CATEGORIES if c not in ("酒吧", "甜點")], 2)
            return [self.no_food_request(self.pick(AVOID_FIRST).format(c=c, c2=c2))]
        if roll < 0.72:  # 菜名 → 類型
            c = self.pick(DISHES)
            dish = self.pick(DISHES[c])
            return [self.full_request(food=self.pick([dish, f"想吃{dish}", f"來點{dish}"]))]
        if roll < 0.79:  # 台北市以外的地點
            place = self.pick(OUTSIDE)
            return [self.full_request(where=self.pick([place, f"在{place}", f"{place}那邊"]))]
        if roll < 0.86:  # 強調排序方式
            return [self.full_request(extra=self.pick(SORTS))]
        if roll < 0.93:  # 反問後說沒想法、要我挑
            return [self.request(self.pick([1, 2]))[0], self.pick(UNSURE + CHOOSE)]
        # 閒聊裡帶著需求（要找店，不是閒聊）
        return [self.pick(MOODS) + "，" + self.full_request()]


def eval_sentences() -> set:
    """評估情境裡使用者說過的每一句（產生訓練資料時要避開）。"""
    import json
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from scenarios import SCENARIOS
    out = {t for _, turns in SCENARIOS for t in turns}
    for name in ("eval_cases.json", "eval_cases2.json"):
        path = os.path.join(ROOT, "finetune", name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                out |= {t for c in json.load(f) for t in c["turns"]}
    return out


def focus_dialogues(n: int, seed: int = 7) -> list:
    """補強資料：[(編號, [使用者的話…])]，跟評估情境重複的對話跳過。"""
    gen, blocked, out = FocusGen(seed), eval_sentences(), []
    while len(out) < n:
        turns = gen.dialogue()
        if not any(t in blocked for t in turns):
            out.append((f"focus-{len(out):04d}", turns))
    return out


def training_dialogues(n: int, seed: int = 2026) -> list:
    """[(編號, [使用者的話…])]。同一個 seed 每次產生一樣的內容（收集可以中斷後接著跑）。"""
    gen = Gen(seed)
    return [(f"train-{i:04d}", gen.dialogue()) for i in range(n)]


if __name__ == "__main__":
    for did, turns in training_dialogues(int(sys.argv[1]) if len(sys.argv) > 1 else 10):
        print(did, " ／ ".join(turns))
