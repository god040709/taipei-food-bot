"""
LLM 呼叫。機器人的程式只認得一個介面：llm.chat(messages, max_tokens, schema) → 文字。

- ReplayLLM：給 API 用。API 自己「不呼叫」LLM，需要時丟出 NeedLLM，請 Open WebUI 的 pipe
  用使用者自己的金鑰去問，pipe 把答案送回來後，API 從頭重跑一次，已經拿到的答案直接沿用。
  這樣金鑰只存在 Open WebUI，別人連到 API 也用不到任何人的 LLM 額度。
- DirectLLM：給本機測試、評估腳本用，直接呼叫 LLM（金鑰從環境變數 LLM_API_KEY 讀取）。
"""
import os
import re
import time
from typing import Optional


class NeedLLM(BaseException):
    """需要問 LLM。繼承 BaseException：機器人程式裡的 `except Exception` 不會不小心把它吃掉。"""

    def __init__(self, request: dict):
        super().__init__("need llm")
        self.request = request


class LLMError(Exception):
    """LLM 沒有可用的回答。note 會顯示在條件行，例如「連不到 LLM 服務」「LLM 回應錯誤 HTTP 401」。"""

    def __init__(self, note: str):
        super().__init__(note)
        self.note = note


def build_request(messages: list, max_tokens: int, schema: Optional[dict] = None) -> dict:
    """OpenAI 相容的 /chat/completions 內容（不含 model，由呼叫的一方填）。"""
    request = {
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        # Qwen3 系列：關閉思考模式，回應更快（不支援的伺服器會忽略）
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if schema:
        # 伺服器支援時，保證回傳符合格式的 JSON
        request["response_format"] = {"type": "json_schema",
                                      "json_schema": {"name": "result", "schema": schema, "strict": True}}
    return request


def clean_text(text: str, finish_reason: Optional[str] = None) -> str:
    """移除思考內容：有些模板只輸出結尾的 </think>，所以取最後一個 </think> 之後的文字。"""
    text = text or ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()  # 思考到一半被截斷
    if not text and finish_reason == "length":
        raise LLMError("LLM 回應被截斷（可能在思考）")
    return text


class ReplayLLM:
    """API 用：results 是 pipe 依序送回來的答案 [{"ok": true, "content": "...", "finish_reason": "..."}
    或 {"ok": false, "note": "連不到 LLM 服務"}]。用完了還要再問，就丟出 NeedLLM。"""

    def __init__(self, enabled: bool, results: Optional[list] = None):
        self.enabled = enabled
        self.results = list(results or [])
        self.used = 0

    def chat(self, messages: list, max_tokens: int = 400, schema: Optional[dict] = None) -> str:
        if not self.enabled:
            raise LLMError("未設定 LLM 金鑰")
        if self.used < len(self.results):
            result = self.results[self.used]
            self.used += 1
            if not result.get("ok"):
                raise LLMError(result.get("note") or "LLM 沒有回應")
            return clean_text(result.get("content"), result.get("finish_reason"))
        raise NeedLLM(build_request(messages, max_tokens, schema))


class NoLLM:
    """不用 LLM（只用規則）。本機測試用。"""
    enabled = False

    def chat(self, messages: list, max_tokens: int = 400, schema: Optional[dict] = None) -> str:
        raise LLMError("未設定 LLM 金鑰")


class DirectLLM:
    """本機測試、評估用：直接呼叫 LLM。金鑰只從環境變數讀，不會印出來。"""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None, max_calls: int = 200,
                 pause: float = 1.0):
        self.key = os.environ.get("LLM_API_KEY", "").strip()
        self.enabled = bool(self.key)
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or "http://163.14.73.40:8088/v1").rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL") or "Qwen3.6-35B-A3B"
        self.max_calls = max_calls
        self.pause = pause
        self.calls = 0

    def chat(self, messages: list, max_tokens: int = 400, schema: Optional[dict] = None) -> str:
        import requests

        if not self.enabled:
            raise LLMError("未設定 LLM 金鑰")
        if self.calls >= self.max_calls:
            raise LLMError("超過本次測試的呼叫上限")
        self.calls += 1
        time.sleep(self.pause)
        request = build_request(messages, max_tokens, schema)
        request["model"] = self.model
        try:
            r = requests.post(f"{self.base_url}/chat/completions", json=request, timeout=90,
                              headers={"Authorization": f"Bearer {self.key}"})
        except requests.exceptions.RequestException:
            raise LLMError("連不到 LLM 服務")
        if r.status_code != 200:
            raise LLMError(f"LLM 回應錯誤 HTTP {r.status_code}")
        choice = r.json()["choices"][0]
        return clean_text(choice.get("message", {}).get("content"), choice.get("finish_reason"))
