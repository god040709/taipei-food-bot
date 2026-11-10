"""
記憶：從之前的回覆讀回狀態（上次的條件、其他條件、推薦過／不喜歡的店、反問過什麼）。
是 Bot（bot.py）的一部分：函式裡的 self 就是 Bot，可以互相呼叫。
"""
import re
from typing import Optional

from lexicon import (  # 詞庫、提示詞、文字處理
    ASK_MARK, CATEGORIES, CHAT_MARK, DISTRICTS, MRT_STATIONS, NEW_REQUEST_MARK, REJECTED_MARK, RELAX_LABELS,
    RESET_MARK, SORT_LABELS, SUBTYPES,
)


class MemoryMixin:
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
    def previous_prefs(messages: list) -> dict:
        """上一次推薦用的其他條件（從「🔎 搜尋條件」那一行後面的「不要連鎖｜隱藏版｜…」讀回來）。"""
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
            for part in cond.group(1).split("｜"):
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

    @staticmethod
    def shown_restaurants(messages: list, condition_line: str) -> list:
        """同一組搜尋條件（含其他條件）下，之前已經推薦過的店名。"""
        names = []
        current = condition_line.split("（")[0]
        for m in messages[:-1]:
            content = m.get("content", "")
            if m.get("role") != "assistant" or not isinstance(content, str):
                continue
            old = re.search(r"🔎 搜尋條件：[^（\n]*", content)
            if not old or old.group(0) != current:
                continue
            names += re.findall(r"^\*\*\d+\. (.+?)\*\*$", content, flags=re.MULTILINE)
        return list(dict.fromkeys(names))  # 同一家可能出現在好幾次回覆裡，只算一次

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

    @staticmethod
    def after_reset(messages: list) -> list:
        """只留最後一次「重新開始」之後的對話。"""
        for i in range(len(messages) - 2, -1, -1):
            if messages[i].get("role") == "assistant" and RESET_MARK in str(messages[i].get("content", "")):
                return messages[i + 1:]
        return messages
