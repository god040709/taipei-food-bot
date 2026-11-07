"""
推薦 API
- 本機執行：python api.py                （http://127.0.0.1:5000）
- 實驗室平台：HOST=0.0.0.0 python api.py   （讓 Open WebUI 容器連得到）
沒有安裝 Flask 時，會自動改用 Python 內建的 http.server，不需要額外套件。

POST /recommend   ：推薦餐廳
POST /categories  ：各料理類型有幾家店（使用者問「有什麼類型可以選」時用）
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from src.recommender import (  # noqa: E402
    category_counts, format_results, relaxations_for, search_restaurants, to_records,
)

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "5000"))


def handle_recommend(data: dict) -> dict:
    conditions = dict(
        district=data.get("district"),
        category=data.get("category"),
        party_size=data.get("party_size"),
        budget=data.get("budget"),
    )
    exclude = data.get("exclude") or []  # 已經推薦過、這次要排除的店名
    results, error = search_restaurants(**conditions, exclude=exclude)
    if error:
        return {"result": error, "restaurants": [], "matched": 0,
                "alternatives": relaxations_for(**conditions, exclude=exclude)}
    # result：顯示用文字；restaurants：結構化資料，給 LLM 寫推薦理由
    # matched：符合條件的餐廳總數；matched_within：價格完全在預算內的家數
    # alternatives：結果太少時，放寬條件的建議
    return {"result": format_results(results), "restaurants": to_records(results),
            "matched": results.attrs.get("matched", len(results)),
            "matched_within": results.attrs.get("matched_within", len(results)),
            "alternatives": results.attrs.get("alternatives", []),
            # relaxed_budget：符合預算的店不夠時，放寬到的預算（後面補上的店標示為 over）
            "relaxed_budget": results.attrs.get("relaxed_budget")}


def handle_categories(data: dict) -> dict:
    return {"categories": category_counts(
        district=data.get("district"),
        party_size=data.get("party_size"),
        budget=data.get("budget"),
    )}


ROUTES = {"/recommend": handle_recommend, "/categories": handle_categories}

try:
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    @app.route("/recommend", methods=["POST"])
    def recommend():
        return jsonify(handle_recommend(request.json or {}))

    @app.route("/categories", methods=["POST"])
    def categories():
        return jsonify(handle_categories(request.json or {}))

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
