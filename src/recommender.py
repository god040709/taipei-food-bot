import os
import math
from urllib.parse import quote

import pandas as pd

# 資料庫路徑（以專案根目錄為準，從任何資料夾執行都找得到）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DF_PATH = os.path.join(BASE_DIR, "data", "taipei_all_restaurants.csv")

# 加權評分的權重（加總為 1，可依需求調整）
W_RATING = 0.6      # 評分可信度
W_POPULARITY = 0.25  # 人氣（評論數）
W_BUDGET = 0.15     # 預算吻合度

# 貝氏平均的門檻：評論數少於這個數字時，評分會被拉向全體平均
BAYES_M = 100


def score_components(df, budget=None):
    """回傳三項分數（各自已乘上權重，加總 = 推薦指數 0~100）。"""
    reviews = df["user_ratings_total"].fillna(0)
    rating = df["rating"].fillna(df["rating"].mean())

    # 1. 評分可信度：貝氏平均，避免「5 則評論 4.9 分」贏過「8000 則評論 4.8 分」
    c = rating.mean()
    bayes = (reviews / (reviews + BAYES_M)) * rating + (BAYES_M / (reviews + BAYES_M)) * c
    rating_score = bayes / 5

    # 2. 人氣：評論數取 log 後正規化到 0~1
    log_reviews = reviews.apply(lambda x: math.log10(x + 1))
    max_log = log_reviews.max() or 1
    popularity_score = log_reviews / max_log

    # 3. 預算吻合度：整個價格區間都在預算內 = 1，部分在內則依比例
    if budget:
        span = (df["max_price"] - df["min_price"]).clip(lower=1)
        budget_score = ((budget - df["min_price"]) / span).clip(0, 1)
    else:
        budget_score = pd.Series(1.0, index=df.index)

    return pd.DataFrame({
        "s_rating": (W_RATING * rating_score * 100).round(1),
        "s_popularity": (W_POPULARITY * popularity_score * 100).round(1),
        "s_budget": (W_BUDGET * budget_score * 100).round(1),
    }, index=df.index)


def compute_scores(df, budget=None):
    """計算每家餐廳的推薦指數 (0~100)。"""
    parts = score_components(df, budget)
    return parts.sum(axis=1).round(1)


def maps_link(row):
    """產生 Google Maps 連結，點了直接開啟該餐廳。"""
    url = f"https://www.google.com/maps/search/?api=1&query={quote(str(row['name']))}"
    if isinstance(row.get("place_id"), str) and row["place_id"]:
        url += f"&query_place_id={row['place_id']}"
    return url


def search_restaurants(district=None, category=None, party_size=None, budget=None, top_n=5):
    """篩選並排序餐廳。回傳 (DataFrame, None)；失敗或沒結果時回傳 (None, 錯誤訊息)。"""
    if not os.path.exists(DF_PATH):
        return None, f"❌ 找不到資料庫檔案：{DF_PATH}，請確認是否已執行採集腳本。"

    try:
        df = pd.read_csv(DF_PATH)
    except Exception as e:
        return None, f"❌ 資料庫讀取失敗: {e}"

    # 1. 行政區篩選
    if district and district != "全區":
        df = df[df['district'].str.contains(district, regex=False, na=False)]

    # 2. 料理類別篩選
    if category and category != "不限":
        df = df[df['category'].str.contains(category, regex=False, na=False)]

    # 3. 人數篩選
    if party_size:
        df = df[(df['min_party'] <= party_size) & (df['max_party'] >= party_size)]

    # 4. 預算篩選
    if budget:
        df = df[df['min_price'] <= budget]

    if df.empty:
        return None, "⚠️ 沒有找到完全符合條件的餐廳，建議放寬預算或地區條件！"

    # 5. 依加權推薦指數排序
    df = df.copy()
    parts = score_components(df, budget)
    df = df.join(parts)
    df["score"] = parts.sum(axis=1).round(1)
    df = df.sort_values(by="score", ascending=False)
    results = df.head(top_n).copy()
    results["maps_url"] = results.apply(maps_link, axis=1)
    results.attrs["matched"] = len(df)  # 符合條件的餐廳總數
    return results, None


def price_text(row):
    """Google 的「$1-200」代表 200 元以下，顯示成比較自然的文字。"""
    low, high = int(row["min_price"]), int(row["max_price"])
    if low <= 1:
        return f"${high} 以下 / 人"
    return f"${low} - ${high} / 人"


def format_results(results):
    """把餐廳結果轉成聊天介面顯示的文字。"""
    output = []
    for _, row in results.iterrows():
        output.append(
            f"📍 **{row['name']}** ({row['district']} - {row['category']})\n"
            f"🏆 推薦指數: {row['score']} / 100"
            f"（評分 {row['s_rating']} + 人氣 {row['s_popularity']} + 預算 {row['s_budget']}）\n"
            f"⭐ 評分: {row['rating']} ({int(row['user_ratings_total'])} 則評論)\n"
            f"💰 約略預算: {price_text(row)}\n"
            f"🏠 地址: {row['address']}\n"
            f"🗺️ [在 Google Maps 開啟]({row['maps_url']})\n"
        )
    return "\n---\n".join(output)


def to_records(results):
    """給 LLM 寫推薦理由用的精簡資料。"""
    cols = ["name", "district", "category", "rating", "user_ratings_total",
            "min_price", "max_price", "score", "s_rating", "s_popularity", "s_budget"]
    return [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
            for r in results[cols].to_dict(orient="records")]


def filter_restaurants(district=None, category=None, party_size=None, budget=None, top_n=5):
    """相容舊版：直接回傳顯示用文字。"""
    results, error = search_restaurants(district, category, party_size, budget, top_n)
    return error if error else format_results(results)


if __name__ == "__main__":
    print("🧪 測試篩選結果（大安區、燒肉、2人、預算 1000 元）：\n")
    print(filter_restaurants(district="大安區", category="燒肉", party_size=2, budget=1000))
