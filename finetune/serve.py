"""
在虛擬機上提供微調後的小模型，介面跟 OpenAI 相容（POST /v1/chat/completions），機器人的 API 可以直接呼叫。
只用 Python 內建的 http.server（/opt/venv_torch 沒有 flask），一次處理一個請求就夠了。

用法（/opt/venv_torch 的 python，需要 GPU）：
    python finetune/serve.py --model finetune/outputs/parser-lora --port 8100
    curl -s localhost:8100/health
只綁 127.0.0.1：同一台虛擬機上的 API 用得到，外面連不進來。
"""
import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def make_handler(llm, model_name: str):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict):
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                self._send(200, {"ok": True, "model": model_name})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(length) or b"{}")
                start = time.time()
                text = llm.generate(req["messages"], int(req.get("max_tokens") or 800))
                self._send(200, {
                    "model": model_name,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": text}}],
                    "seconds": round(time.time() - start, 2),
                })
            except Exception as e:  # 回錯誤給呼叫的一方（機器人會退回用 35B）
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            sys.stderr.write("[serve] " + (fmt % args) + "\n")

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="finetune/outputs/ 底下的 LoRA 資料夾，或基準模型名稱")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--min-free-gb", type=float, default=6)
    args = ap.parse_args()

    import torch
    free = torch.cuda.mem_get_info()[0] / 1024 ** 3
    if free < args.min_free_gb:
        raise SystemExit(f"GPU 只剩 {free:.1f} GB（需要 {args.min_free_gb} GB），可能有人在用，晚點再跑")

    from evaluate import LocalModel
    llm = LocalModel(args.model)
    server = HTTPServer(("127.0.0.1", args.port), make_handler(llm, args.model))
    print(f"小模型服務：http://127.0.0.1:{args.port}/v1/chat/completions（{args.model}）")
    server.serve_forever()


if __name__ == "__main__":
    main()
