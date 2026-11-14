"""
台北美食推薦機器人的對話流程：理解需求 → 反問／排除／換一批 → 查餐廳 → 組合回覆。
api.py 的 /chat 會呼叫這裡；Open WebUI 的 pipe 只負責把對話轉過來。
"""
import json
import re
from typing import Optional

import service
from lexicon import (  # 詞庫、提示詞、文字處理
    ASK_MARK, CHAT_MARK, DEBUG_MARK, NEAR_ASK_MARK, NON_MEAL, REJECTED_MARK, RESET_WORDS, SUBTYPES,
    fix_homophones, to_traditional,
)
from dialogue import DialogueMixin
from memory import MemoryMixin
from render import RenderMixin
from understand import UnderstandMixin


class Options:
    """機器人的設定（從 Open WebUI 函式的 Valves 傳過來，沒給就用預設值）。"""
    USE_LLM_PARSER = True   # 用 LLM 理解使用者需求
    USE_LLM_REASON = True   # 用 LLM 寫推薦理由
    SHOW_CONDITIONS = True  # 在回答開頭顯示解析出的條件
    ASK_FOLLOWUP = True     # 資訊不足時持續反問，直到地區、類型、人數、預算都清楚
    MAX_ASK = 4             # 同一個需求最多反問幾次（避免一直問不完）
    SHOW_REASONS = True     # 在每家餐廳附上推薦理由
    DEBUG = False           # 除錯模式：回覆最後附上「🔧 除錯資訊」
    OFF_GUARDS = ""         # 關掉的保險規則（逗號分隔，名稱見 understand.py 的 GUARDS），實驗用

    def __init__(self, values: Optional[dict] = None):
        for key, value in (values or {}).items():
            if hasattr(Options, key) and isinstance(value, type(getattr(Options, key))):
                setattr(self, key, value)


class Bot(UnderstandMixin, MemoryMixin, DialogueMixin, RenderMixin):
    """對話機器人：收到整段對話，回傳下一則回覆。
    功能分在幾個檔案：understand.py（理解）、memory.py（記憶）、dialogue.py（對話）、render.py（排版）；
    這裡只放設定、LLM 呼叫和主流程。
    llm：有 enabled 屬性和 chat(messages, max_tokens, schema) 方法的物件（見 llm.py）。"""

    def __init__(self, llm, options: Optional[dict] = None):
        self.llm = llm
        self.valves = Options(options)
        self.trace = {}  # 除錯模式用：這次回覆每一步的結果

    def _llm_enabled(self) -> bool:
        return bool(getattr(self.llm, "enabled", False))

    def _chat(self, messages: list, max_tokens: int = 400, schema: Optional[dict] = None,
              temperature: float = 0) -> str:
        return to_traditional(self.llm.chat(messages, max_tokens, schema, temperature))  # LLM 偶爾寫出簡體字

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

    def pipe(self, body: dict) -> str:
        # 之前回覆裡的除錯區塊不算對話內容（不然會被讀成條件、或被 LLM 模仿）
        messages = [{**m, "content": m["content"].split(DEBUG_MARK)[0].rstrip()}
                    if isinstance(m.get("content"), str) and DEBUG_MARK in m["content"] else m
                    for m in body.get("messages", [])]
        self.trace = {}
        reply = self._pipe({**body, "messages": messages})
        return reply + self.debug_block(reply) if self.valves.DEBUG else reply

    def _pipe(self, body: dict) -> str:
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
        said = self.rule_prefs(user_text)  # 這一句規則讀得到的其他條件（捷運站、不要的類別、多類別）
        if info["intent"] == "reset":
            return self.reset_reply(user_text)

        # 問「有什麼類型可以選」→ 列出料理類型（LLM 判斷）
        if info["intent"] == "categories":
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
        # 「第1間留著 其他換掉」「2號留下來 其他幫我換」：LLM 判斷有要保留的店（提示詞裡 keep 就是
        # 「換一批時要保留的」）→ 換一批、保留那幾間，不用問是哪一間
        keep_rest = bool(info["keep"] and not rejects)
        if reject_said and not rejects and last_names and not new_search and not keep_rest:
            if self.wants_more(user_text):
                rejects = list(last_names)  # 「吃過了，換別的」→ 這批都不要
            else:
                return self.ask_which(last_names, user_text)

        # 不是要新推薦（問你會做什麼、道謝、閒聊、問細節）→ LLM 自然回覆。
        # 這句話有提到任何條件就當成找餐廳，避免把需求誤判成閒聊
        # 剛推薦完說「嗯」「好」「幫我寫作業」：LLM 判斷成找店，但這句話沒有任何新的東西（條件跟上次推薦一樣、
        # 沒有其他偏好、沒指定店）→ 重列一樣的店沒有意義，也當成閒聊。上一則是反問（在回答問題）或閒聊時不算
        after_rec = bool(last_names) and "🔎 搜尋條件：" in prev_msg and not prev_msg.startswith((ASK_MARK, CHAT_MARK))
        nothing_new = (info["intent"] == "search" and after_rec and not reject_said and not keep_rest
                       and not info["llm_waived"] and not info["outside_area"] and not info["unknown_food"]
                       and tuple(conditions) == self.previous_conditions(messages))
        if (info["intent"] == "chat" or nothing_new) and not rejects and not self.wants_more(user_text) \
                and not new_search and not info["choose_category"]:  # 「你幫我挑一種料理」是要找餐廳，不是閒聊
            if nothing_new:
                self.note("LLM 判斷是找店，但這句話沒有新的條件，當成閒聊")
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
        self.trace["missing"] = missing
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
        more = self.wants_more(user_text) or info["intent"] == "more" or keep_rest
        before = self.rejected_history(messages)
        if info["new_request"]:
            # 全新的需求：之前不想去的店不再排除（畫面上會說明，使用者還是不想去可以再說）
            info["cleared_rejects"], before = [n for n in before if n not in rejects], []
        rejected = list(dict.fromkeys(before + rejects))
        history = self.shown_restaurants(messages, condition_line)
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

        # 2. 呼叫推薦 API
        payload = {
            "district": district,
            "category": category,
            "party_size": party_size,
            "budget": budget,
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
            # 沒指定類型（類型不限）：不推酒吧、甜點這種不算正餐的（不寫進條件行，只影響這次查詢）
            "avoid_categories": (prefs.get("avoid") or []) + ([] if category or prefs.get("cats") else NON_MEAL),
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

        self.trace["query"] = (payload, data)
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
        # 要讓使用者知道的事（放寬了什麼、總預算換算、資料沒有的條件…）：交給開場白用一兩句帶過
        facts = self.explain_facts(prefs, data, info, conditions, total, subtype, restaurants,
                                   party_relaxed, no_location, exclude, where)
        sections = [condition_line] if self.valves.SHOW_CONDITIONS else []
        # 狀態用的行（下一句會讀回來）：新的需求、幫你挑的類型、不再推薦的店
        if self.new_request_line(info):
            sections.append(self.new_request_line(info))
        if picked:
            sections.append(f"🎲 幫你挑了「{picked}」，想換一種可以再跟我說")
        if rejects:
            sections.append(REJECTED_MARK + "".join(f"「{n}」" for n in rejects))
        fallback = f"💬 {'；'.join(facts)}。" if facts else ""
        if not restaurants:
            # 沒有結果：顯示原因（例如找不到符合條件的店）
            if fallback:
                sections.append(fallback)
            if more and shown:
                sections.append(f"🔄 這個條件下的店都推薦過了（已推薦 {len(shown)} 家）。")
            else:
                sections.append(data.get("result", "沒有取得有效的推薦結果。"))
            if tips:
                sections.append(tips)
            return "\n\n".join(sections)

        # 開場白（LLM）＋每家餐廳的推薦理由
        texts, _ = self.current_round(messages)
        # 人數太多、不限人數找的：不讓 LLM 說「適合 30 人」這種資料沒辦法確認的事
        reason_conditions = (district, category, None if party_relaxed else party_size, budget)
        notes = [f"要讓使用者知道的事：{'；'.join(facts)}"] if facts else []
        intro, reasons = self.write_reasons(texts, reason_conditions, restaurants, subtype,
                                            prefs.get("unsupported"), notes, prefs.get("budget_min"))
        if intro:
            sections.append(f"💬 {intro}")
        elif fallback:
            sections.append(fallback)
        sections.append("\n\n---\n\n".join(self.format_card(i, r, reason)
                                              for i, (r, reason) in enumerate(zip(restaurants, reasons), 1)))
        if tips:
            sections.append(tips)
        return "\n\n".join(sections)
