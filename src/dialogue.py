"""
對話：反問、閒聊、各種「問使用者」的回覆，以及幫使用者挑料理類型。
是 Bot（bot.py）的一部分：函式裡的 self 就是 Bot，可以互相呼叫。
"""
import hashlib
import json
import random
import re
from typing import Optional

import service
from lexicon import (  # 詞庫、提示詞、文字處理
    ASK_HINT_WORDS, ASK_MARK, CATEGORIES, CATEGORY_SYNONYMS, CHAT_MARK, CHAT_PROMPT, CHOOSE_MARK, CN_INDEX,
    DISTRICTS, FIELDS, LABELS, NEAR_ASK_MARK, QUESTIONS, RESET_MARK, SAY_PROMPT, UNSURE_WORDS, strip_marks,
)


class DialogueMixin:
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

        extra = self.prefs_text(prefs)
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

    def list_categories(self, conditions) -> str:
        district, _, party_size, budget = conditions
        r = service.handle_categories({"district": district, "party_size": party_size, "budget": budget})
        counts = {c["category"]: c["count"] for c in r.get("categories", [])}

        where = district or "全台北"
        if budget:
            where += f"、每人 {budget} 元內"
        order = sorted(CATEGORIES, key=lambda c: -counts.get(c, 0))
        lines = [f"{ASK_MARK}這些料理類型都可以選（{where}）：", "",
                 "| 類型 | 店家數 | 也可以這樣說 |", "|---|---|---|"]
        for c in order:
            if not counts.get(c):
                continue  # 這個條件下沒有店的類型就不列
            examples = "、".join(CATEGORY_SYNONYMS.get(c, [])[:3]) or "—"
            lines.append(f"| {c} | {counts[c]} 家 | {examples} |")
        lines += ["", "想吃哪一種？直接回覆類型就可以，例如「燒肉」；"
                      "還沒決定的話，回「類型都可以」我就從各種料理裡挑。"]
        return "\n".join(lines)

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
