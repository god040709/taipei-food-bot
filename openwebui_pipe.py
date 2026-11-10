"""
title: Taipei Restaurant Recommendation Pipe
author: Taipei Food Bot
version: 4.0.0
description: 台北美食推薦機器人（轉送版）：對話交給 api.py 處理；需要 LLM 時，用這裡設定的金鑰代問。
"""

# 這個檔案只負責「轉送」，機器人的邏輯都在 api.py（src/bot.py）。
# 之後更新機器人只要在虛擬機上 git pull、重開 api.py，不用再重貼這個檔案。
#
# 流程：
#   1. 把整段對話送到 API 的 /chat
#   2. API 回 {"type": "llm", "request": ...} → 用下面設定的金鑰問 LLM，把答案送回 API，回到 2
#   3. API 回 {"type": "reply", "content": ...} → 顯示給使用者
# 金鑰只存在 Open WebUI，API 不會拿到。

import re
from typing import Generator, Iterator, Union

import requests
from pydantic import BaseModel, Field

MAX_ROUNDS = 6  # 一則訊息最多問 LLM 幾次（避免無窮迴圈）


class Pipe:
    class Valves(BaseModel):
        API_URL: str = Field(
            default="http://127.0.0.1:5000/chat",
            description="機器人 API 網址（api.py 的 /chat）。實驗室平台通常是 http://testbed:5000/chat",
        )
        LLM_BASE_URL: str = Field(
            default="http://163.14.73.40:8088/v1",
            description="LLM 服務網址（OpenAI 相容格式，結尾是 /v1）",
        )
        LLM_API_KEY: str = Field(
            default="",
            description="自己的 LLM API Key（實驗室 .env 裡的 LLM_API_KEY）。留空則只用規則解析",
        )
        LLM_MODEL: str = Field(
            default="",
            description="模型名稱（例如 Qwen3.6-35B-A3B）。留空會自動使用服務提供的第一個模型",
        )
        USE_LLM_PARSER: bool = Field(default=True, description="用 LLM 理解使用者需求")
        USE_LLM_REASON: bool = Field(default=True, description="用 LLM 寫推薦理由")
        SHOW_CONDITIONS: bool = Field(default=True, description="在回答開頭顯示解析出的條件")
        ASK_FOLLOWUP: bool = Field(default=True, description="資訊不足時持續反問")
        MAX_ASK: int = Field(default=4, description="同一個需求最多反問幾次")
        SHOW_REASONS: bool = Field(default=True, description="在每家餐廳附上推薦理由")

    def __init__(self):
        self.valves = self.Valves()
        self._model_cache = None

    # ---------------- 呼叫 API ----------------
    def _api_url(self) -> str:
        url = self.valves.API_URL.strip().rstrip("/")
        # 舊版的設定是 …/recommend，自動改成 …/chat
        return re.sub(r"/(recommend|categories)$", "", url) + "/chat" if not url.endswith("/chat") else url

    def _options(self) -> dict:
        return {k: getattr(self.valves, k) for k in ("USE_LLM_PARSER", "USE_LLM_REASON", "SHOW_CONDITIONS",
                                                     "ASK_FOLLOWUP", "MAX_ASK", "SHOW_REASONS")}

    # ---------------- 呼叫 LLM（只在 API 要求時） ----------------
    def _key(self) -> str:
        # 貼上金鑰時常會多帶空白、換行或「Bearer 」，先清掉
        return re.sub(r"^Bearer\s+", "", self.valves.LLM_API_KEY.strip(), flags=re.IGNORECASE)

    def _llm_url(self) -> str:
        return self.valves.LLM_BASE_URL.strip().rstrip("/")

    def _model(self) -> str:
        if self.valves.LLM_MODEL.strip():
            return self.valves.LLM_MODEL.strip()
        if self._model_cache is None:
            r = requests.get(f"{self._llm_url()}/models", timeout=10,
                             headers={"Authorization": f"Bearer {self._key()}"})
            r.raise_for_status()
            self._model_cache = r.json()["data"][0]["id"]
        return self._model_cache

    def _ask_llm(self, request: dict) -> dict:
        """照 API 給的內容問 LLM，回傳 {"ok": True, "content": ...} 或 {"ok": False, "note": 原因}。"""
        try:
            headers = {"Authorization": f"Bearer {self._key()}"}
            url = f"{self._llm_url()}/chat/completions"
            payload = dict(request, model=self._model())
            r = requests.post(url, json=payload, timeout=90, headers=headers)
            if r.status_code == 400 and "response_format" in payload:
                # 伺服器不支援指定 JSON 格式：拿掉再試一次
                payload.pop("response_format")
                r = requests.post(url, json=payload, timeout=90, headers=headers)
            if r.status_code != 200:
                return {"ok": False, "note": f"LLM 回應錯誤 HTTP {r.status_code}"}
            choice = r.json()["choices"][0]
            return {"ok": True, "content": choice.get("message", {}).get("content") or "",
                    "finish_reason": choice.get("finish_reason")}
        except requests.exceptions.HTTPError as e:
            return {"ok": False, "note": f"LLM 回應錯誤 HTTP {e.response.status_code}"}
        except requests.exceptions.RequestException:
            return {"ok": False, "note": "連不到 LLM 服務"}
        except Exception:
            return {"ok": False, "note": "LLM 回傳格式不對"}

    # ---------------- 主流程 ----------------
    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        messages = [{"role": m.get("role"), "content": m.get("content")}
                    for m in body.get("messages", []) if m.get("role") in ("user", "assistant")]
        if not messages:
            return "無有效的對話訊息。"
        payload = {"messages": messages, "llm_enabled": bool(self._key() and self._llm_url()),
                   "llm_results": [], "options": self._options()}
        for _ in range(MAX_ROUNDS + 1):
            try:
                r = requests.post(self._api_url(), json=payload, timeout=60)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                return ("❌ 連不到機器人 API，請確認虛擬機上的 `python api.py` 有在執行，"
                        f"以及齒輪裡的 Api Url 是否正確（目前：{self._api_url()}）。\n錯誤訊息：{e}")
            if data.get("type") == "reply":
                return data.get("content", "")
            if data.get("type") != "llm":
                return "❌ 機器人 API 回應的格式不對，可能是 api.py 還是舊版，請在虛擬機上 git pull 後重開 api.py。"
            payload["llm_results"] = list(data.get("llm_results") or []) + [self._ask_llm(data["request"])]
        return "❌ 這則訊息問了 LLM 太多次，已停止。"
