"""
推薦 API（機器人的大腦都在這裡，Open WebUI 的 pipe 只負責轉送）
- 本機執行：python api.py                （http://127.0.0.1:5000）
- 實驗室平台：HOST=0.0.0.0 python api.py   （讓 Open WebUI 容器連得到）
沒有安裝 Flask 時，會自動改用 Python 內建的 http.server，不需要額外套件。

POST /chat        ：收到整段對話，回傳下一則回覆；需要 LLM 時請 pipe 代問（見 handle_chat）
POST /recommend   ：推薦餐廳（給除錯、測試用）
POST /categories  ：各料理類型有幾家店

這個 API 不保存、也不使用任何 LLM 金鑰：需要問 LLM 時，會把要問的內容回傳給 pipe，
由 pipe 用 Open WebUI 裡設定的金鑰去問。所以別人就算連到這個 API，也用不到任何人的 LLM 額度。
"""
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from bot import Bot  # noqa: E402
from llm import NeedLLM, ReplayLLM  # noqa: E402
from service import handle_categories, handle_recommend  # noqa: E402

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "5000"))
PROTOCOL = 1  # pipe 和 API 之間的格式版本


def handle_chat(data: dict) -> dict:
    """
    請求：{"messages": [...], "llm_enabled": true, "llm_results": [...], "options": {...}}
      llm_results：之前請 pipe 問 LLM 得到的答案，依序排列
        [{"ok": true, "content": "...", "finish_reason": "stop"} 或 {"ok": false, "note": "連不到 LLM 服務"}]
    回應：
      {"type": "reply", "content": "給使用者看的回覆"}
      {"type": "llm", "request": {...}, "llm_results": [...]}
        → pipe 用 request 去問 LLM，把答案加到 llm_results 最後面，再呼叫一次 /chat
    每次都從頭重跑，已經有的答案直接沿用（同樣的對話 + 同樣的答案 → 同樣的結果）。
    """
    llm = ReplayLLM(bool(data.get("llm_enabled")), data.get("llm_results") or [])
    bot = Bot(llm, data.get("options") or {})
    try:
        content = bot.pipe({"messages": data.get("messages") or []})
    except NeedLLM as need:
        return {"type": "llm", "request": need.request, "llm_results": llm.results, "protocol": PROTOCOL}
    except Exception as e:
        traceback.print_exc()
        return {"type": "reply", "content": f"❌ 機器人發生錯誤：{type(e).__name__}: {e}", "protocol": PROTOCOL}
    return {"type": "reply", "content": content, "protocol": PROTOCOL}


ROUTES = {"/chat": handle_chat, "/recommend": handle_recommend, "/categories": handle_categories}

try:
    from flask import Flask, Response, request

    app = Flask(__name__)

    def _json(payload):
        # 不用 jsonify：它會把 key 照字母排序，也會把中文轉成 \\uXXXX
        return Response(json.dumps(payload, ensure_ascii=False), mimetype="application/json")

    @app.route("/chat", methods=["POST"])
    def chat():
        return _json(handle_chat(request.json or {}))

    @app.route("/recommend", methods=["POST"])
    def recommend():
        return _json(handle_recommend(request.json or {}))

    @app.route("/categories", methods=["POST"])
    def categories():
        return _json(handle_categories(request.json or {}))

    def run():
        app.run(host=HOST, port=PORT)

except ImportError:  # 沒有 Flask：改用內建伺服器
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    app = None

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            handler = ROUTES.get(self.path)
            if handler is None:
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                data = {}
            body = json.dumps(handler(data), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def run():
        print(f" * Running on http://{HOST}:{PORT}（內建伺服器，未安裝 Flask）")
        ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    run()
