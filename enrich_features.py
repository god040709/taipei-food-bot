"""
補充餐廳特色（Google Places API (New) 的 Place Details）
產生 data/restaurant_features.csv，讓推薦理由能寫出「餐廳特色」，而不只是分數。

抓的資料：
- summary     ：Google 的店家簡介（有些店沒有）
- highlights  ：服務特色，例如「可訂位、適合團體、提供素食、可帶寵物」
- reviews     ：最多 3 則顧客評論摘錄（讓 LLM 知道招牌菜、氣氛）

使用方式：
    Windows PowerShell： $env:GOOGLE_MAPS_API_KEY="你的key"; python enrich_features.py
    macOS / Linux：      GOOGLE_MAPS_API_KEY=你的key python enrich_features.py

費用注意：
- 每家餐廳呼叫 1 次 Place Details。評論、簡介、服務特色屬於費用較高的欄位等級，
  執行前請先到 Google Cloud 查看 Places API (New) 的價格與每月免費額度。
- 預設只抓「每個行政區 × 類別」推薦指數前 TOP_PER_GROUP 名（預設 8），
  設 TOP_PER_GROUP=0 則抓全部（約 2,900 家）。
- 可以中斷後重跑：已經抓過的店會自動跳過。
"""
import os
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from recommender import DF_PATH, FEATURES_PATH, compute_scores  # noqa: E402

API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
BASE_URL = os.environ.get("PLACES_BASE_URL", "https://places.googleapis.com/v1")
TOP_PER_GROUP = int(os.environ.get("TOP_PER_GROUP", "8"))

# 服務特色欄位 → 顯示文字（值為 True 才列出）
ATTRIBUTES = {
    "reservable": "可訂位",
    "goodForGroups": "適合團體聚餐",
    "goodForChildren": "適合帶小孩",
    "menuForChildren": "有兒童餐",
    "servesVegetarianFood": "提供素食",
    "servesBeer": "有啤酒",
    "servesWine": "有葡萄酒",
    "servesCocktails": "有調酒",
    "servesDessert": "有甜點",
    "servesCoffee": "有咖啡",
    "servesBreakfast": "供應早餐",
    "servesBrunch": "供應早午餐",
    "outdoorSeating": "有戶外座位",
    "allowsDogs": "可帶寵物",
    "liveMusic": "有現場音樂",
    "takeout": "可外帶",
    "delivery": "可外送",
}
FIELD_MASK = ",".join(["id", "editorialSummary", "generativeSummary", "reviews"] + list(ATTRIBUTES))


def fetch_details(place_id):
    r = requests.get(
        f"{BASE_URL}/places/{place_id}",
        params={"languageCode": "zh-TW", "regionCode": "TW"},
        headers={"X-Goog-Api-Key": API_KEY, "X-Goog-FieldMask": FIELD_MASK},
        timeout=15,
    )
    if r.status_code != 200:
        try:
            message = r.json()["error"]["message"]
        except Exception:
            message = r.text[:200]
        raise RuntimeError(f"HTTP {r.status_code}: {message}")
    return r.json()


def to_features(place_id, d):
    summary = ((d.get("generativeSummary") or {}).get("overview") or {}).get("text") \
        or (d.get("editorialSummary") or {}).get("text") or ""
    highlights = [label for key, label in ATTRIBUTES.items() if d.get(key) is True]
    reviews = []
    for rv in (d.get("reviews") or [])[:3]:
        text = ((rv.get("text") or rv.get("originalText") or {}).get("text") or "").strip()
        text = " ".join(text.split())
        if text:
            reviews.append(text[:120] + ("…" if len(text) > 120 else ""))
    return {
        "place_id": place_id,
        "summary": summary,
        "highlights": "、".join(highlights),
        "reviews": "｜".join(reviews),
    }


def pick_places(df):
    """選出要補資料的餐廳：每個行政區 × 類別的前 TOP_PER_GROUP 名。"""
    if TOP_PER_GROUP <= 0:
        return df
    df = df.copy()
    df["score"] = pd.concat(compute_scores(g) for _, g in df.groupby(["district", "category"]))
    df = df.sort_values("score", ascending=False)
    return df.groupby(["district", "category"]).head(TOP_PER_GROUP)


def main():
    if not API_KEY:
        raise SystemExit("❌ 請先設定環境變數 GOOGLE_MAPS_API_KEY")

    df = pd.read_csv(DF_PATH)
    targets = pick_places(df)

    done = pd.read_csv(FEATURES_PATH) if os.path.exists(FEATURES_PATH) else \
        pd.DataFrame(columns=["place_id", "summary", "highlights", "reviews"])
    done_ids = set(done["place_id"])
    todo = [pid for pid in targets["place_id"] if pid not in done_ids]

    print(f"📋 這次要抓 {len(todo)} 家（已抓過 {len(done_ids)} 家，會跳過）")
    if not todo:
        return
    if input("確定要呼叫 Google API 嗎？(y/N) ").strip().lower() != "y":
        raise SystemExit("已取消")

    rows = done.to_dict(orient="records")
    for i, pid in enumerate(todo, 1):
        try:
            rows.append(to_features(pid, fetch_details(pid)))
        except Exception as e:
            print(f"❌ {pid} 失敗：{e}")
            if "HTTP 4" in str(e) and "404" not in str(e):
                print("   可能是金鑰或權限問題，先停止。已抓到的資料會保存。")
                break
        if i % 50 == 0 or i == len(todo):
            pd.DataFrame(rows).to_csv(FEATURES_PATH, index=False, encoding="utf-8-sig")
            print(f"💾 已保存 {i}/{len(todo)}")
        time.sleep(0.1)

    pd.DataFrame(rows).to_csv(FEATURES_PATH, index=False, encoding="utf-8-sig")
    print(f"🎉 完成！共 {len(rows)} 家的特色資料存到 {FEATURES_PATH}")


if __name__ == "__main__":
    main()
