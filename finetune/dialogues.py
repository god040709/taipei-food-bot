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


def training_dialogues(n: int, seed: int = 2026) -> list:
    """[(編號, [使用者的話…])]。同一個 seed 每次產生一樣的內容（收集可以中斷後接著跑）。"""
    gen = Gen(seed)
    return [(f"train-{i:04d}", gen.dialogue()) for i in range(n)]


if __name__ == "__main__":
    for did, turns in training_dialogues(int(sys.argv[1]) if len(sys.argv) > 1 else 10):
        print(did, " ／ ".join(turns))
