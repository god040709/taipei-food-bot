import os
import math
from urllib.parse import quote

import pandas as pd

# 資料庫路徑（以專案根目錄為準，從任何資料夾執行都找得到）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DF_PATH = os.path.join(BASE_DIR, "data", "taipei_all_restaurants.csv")
# 餐廳特色（enrich_features.py 產生，沒有這個檔案也能運作）
FEATURES_PATH = os.path.join(BASE_DIR, "data", "restaurant_features.csv")

# 加權評分的權重（加總為 1，可依需求調整）
W_RATING = 0.6      # 評分可信度
W_POPULARITY = 0.25  # 人氣（評論數）
W_BUDGET = 0.15     # 預算吻合度

# 貝氏平均的門檻：評論數少於這個數字時，評分會被拉向全體平均
BAYES_M = 100

# 從店名看得出來的特色
NAME_KEYWORDS = {
    "吃到飽": "吃到飽", "和牛": "主打和牛", "寵物友善": "寵物友善", "石頭": "石頭鍋",
    "麻辣": "麻辣鍋", "酸菜白肉": "酸菜白肉鍋", "壽喜燒": "壽喜燒", "海鮮": "海鮮",
    "蔬食": "蔬食", "素食": "素食", "24H": "24 小時營業", "24小時": "24 小時營業",
    "旗艦": "旗艦店", "本店": "本店", "總店": "總店", "老店": "老店", "餐酒": "可小酌",
    "沾麵": "沾麵", "豚骨": "豚骨拉麵", "牛肉麵": "牛肉麵", "鐵板燒": "鐵板燒",
    "無菜單": "無菜單料理", "omakase": "Omakase", "居酒屋": "居酒屋", "丼": "丼飯",
}


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

    # 3. 預算吻合度：價格區間落在預算內的比例（整個區間都在預算內 = 1）
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


def load_data():
    df = pd.read_csv(DF_PATH)
    if os.path.exists(FEATURES_PATH):
        features = pd.read_csv(FEATURES_PATH).drop_duplicates("place_id")
        df = df.merge(features, on="place_id", how="left")
    for col in ("summary", "highlights", "reviews"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    # 同區同類型的比較基準（用來寫「評論數第幾多」「價格親民」這類事實）
    group = df.groupby(["district", "category"])
    df["group_size"] = group["place_id"].transform("count")
    df["review_rank"] = group["user_ratings_total"].rank(ascending=False, method="min").astype(int)
    mid = (df["min_price"] + df["max_price"]) / 2
    df["group_mid_price"] = mid.groupby([df["district"], df["category"]]).transform("median")
    return df


def search_restaurants(district=None, category=None, party_size=None, budget=None, top_n=5):
    """篩選並排序餐廳。回傳 (DataFrame, None)；失敗或沒結果時回傳 (None, 錯誤訊息)。"""
    if not os.path.exists(DF_PATH):
        return None, f"❌ 找不到資料庫檔案：{DF_PATH}，請確認是否已執行採集腳本。"

    try:
        all_df = load_data()
    except Exception as e:
        return None, f"❌ 資料庫讀取失敗: {e}"

    df = apply_filters(all_df, district, category, party_size, budget)
    if budget:
        df["budget_fit"] = (df["max_price"] <= budget).map({True: "within", False: "partial"})
    else:
        df["budget_fit"] = ""

    # 結果不到 top_n 家：算出放寬哪個條件可以多找到幾家
    alternatives = suggest_relaxations(all_df, district, category, party_size, budget) \
        if len(df) < top_n else []

    if df.empty:
        return None, "⚠️ 沒有找到符合條件的餐廳，建議放寬預算或地區條件！"

    # 5. 依加權推薦指數排序（完全在預算內的店一定排在前面）
    parts = score_components(df, budget)
    df = df.join(parts)
    df["score"] = parts.sum(axis=1).round(1)
    df["_fit_order"] = (df["budget_fit"] == "partial").astype(int)
    df = df.sort_values(by=["_fit_order", "score"], ascending=[True, False])

    results = df.head(top_n).copy()
    results["maps_url"] = results.apply(maps_link, axis=1)
    results["facts"] = results.apply(lambda r: describe_facts(r, budget), axis=1)
    results.attrs["matched"] = len(df)  # 符合條件的餐廳總數
    results.attrs["matched_within"] = int((df["budget_fit"] == "within").sum()) if budget else len(df)
    results.attrs["alternatives"] = alternatives
    return results, None


def apply_filters(df, district=None, category=None, party_size=None, budget=None):
    """依條件篩選。預算是「每人上限」：
    - within  = 整個價格區間都在預算內（最高價 ≤ 預算）→ 優先推薦
    - partial = 最低價低於預算、最高價超過預算 → 排在後面，並標示
    - 最低價 ≥ 預算的店（例如預算 800、店家 800~1000）不推薦"""
    if district and district != "全區":
        df = df[df['district'].str.contains(district, regex=False, na=False)]
    if category and category != "不限":
        df = df[df['category'].str.contains(category, regex=False, na=False)]
    if party_size:
        df = df[(df['min_party'] <= party_size) & (df['max_party'] >= party_size)]
    if budget:
        df = df[df['min_price'] < budget]
    return df.copy()


def relaxations_for(district=None, category=None, party_size=None, budget=None):
    """給 API 在「完全沒有結果」時使用。"""
    try:
        return suggest_relaxations(load_data(), district, category, party_size, budget)
    except Exception:
        return []


def suggest_relaxations(df, district, category, party_size, budget):
    """結果太少時，建議放寬的方式，例如「不限地區 → 64 家」「預算 500 → 5 家」。"""
    current = len(apply_filters(df, district, category, party_size, budget))
    options = []
    if district:
        n = len(apply_filters(df, None, category, party_size, budget))
        if n > current:
            options.append({"reply": "不限地區", "text": f"全台北有 {n} 家符合", "count": n})
    if budget:
        for factor in (1.5, 2, 3):
            new_budget = int(math.ceil(budget * factor / 100) * 100)
            n = len(apply_filters(df, district, category, party_size, new_budget))
            if n > current:
                where = district or "全台北"
                options.append({"reply": f"預算 {new_budget}", "text": f"{where}有 {n} 家符合", "count": n})
                break
    if category:
        n = len(apply_filters(df, district, None, party_size, budget))
        if n > current:
            where = district or "全台北"
            options.append({"reply": "類型都可以", "text": f"{where}各種料理共 {n} 家符合", "count": n})
    return options


def describe_facts(row, budget=None):
    """從資料推得出來的客觀事實，給 LLM 寫推薦理由（也是沒有 LLM 時的推薦理由）。"""
    facts = []
    name = str(row["name"])
    tags = []
    for word, tag in NAME_KEYWORDS.items():
        if word.lower() in name.lower() and tag not in tags:
            tags.append(tag)
    if tags:
        facts.append("店名主打" + "、".join(tags))

    if row["group_size"] >= 3 and row["review_rank"] <= 3:
        facts.append(f"{row['district']}{row['category']}中評論數第 {row['review_rank']} 多"
                     f"（共 {row['group_size']} 家）")
    reviews = int(row["user_ratings_total"] or 0)
    if row["rating"] >= 4.7 and reviews >= 1000:
        facts.append(f"評分 {row['rating']}、超過 {reviews // 1000 * 1000:,} 則評論，口碑穩定")
    elif row["rating"] >= 4.7:
        facts.append(f"評分高達 {row['rating']}")

    mid = (row["min_price"] + row["max_price"]) / 2
    if row["group_mid_price"] and mid <= row["group_mid_price"] * 0.75:
        facts.append("價格比同區同類型的多數店家親民")
    elif row["group_mid_price"] and mid >= row["group_mid_price"] * 1.3:
        facts.append("屬於同區同類型中較高價位的選擇")

    if budget:
        if row["budget_fit"] == "within":
            facts.append(f"價格完全在每人 {budget} 元預算內")
        else:
            facts.append(f"最低消費在預算內，但部分餐點超過每人 {budget} 元")
    return facts


def price_text(row):
    """Google 的「$1-200」代表 200 元以下，顯示成比較自然的文字。"""
    low, high = int(row["min_price"]), int(row["max_price"])
    if low <= 1:
        return f"${high} 以下 / 人"
    return f"${low} - ${high} / 人"


def fallback_reason(facts):
    """沒有 LLM 時，用事實組成推薦理由。"""
    return "；".join(facts) + "。" if facts else ""


def format_card(i, row, reason=""):
    """一家餐廳的顯示文字。row 可以是 dict 或 pandas Series。"""
    price = price_text(row)
    if row.get("budget_fit") == "partial":
        price += "（⚠️ 部分餐點可能超出預算）"
    lines = [
        f"**{i}. {row['name']}**",
        f"📍 {row['district']}・{row['category']}｜⭐ {row['rating']}"
        f"（{int(row['user_ratings_total']):,} 則評論）｜💰 {price}",
    ]
    if reason:
        lines.append(f"✨ {reason}")
    lines.append(f"🏠 {row['address']}｜🗺️ [在 Google Maps 開啟]({row['maps_url']})")
    return "\n".join(lines)


def format_results(results):
    """把餐廳結果轉成聊天介面顯示的文字。"""
    cards = [format_card(i, row, fallback_reason(row["facts"]))
             for i, (_, row) in enumerate(results.iterrows(), 1)]
    return "\n\n---\n\n".join(cards)


def to_records(results):
    """給 Pipe 顯示與 LLM 寫推薦理由用的資料。"""
    cols = ["name", "district", "category", "rating", "user_ratings_total",
            "min_price", "max_price", "score", "address", "maps_url", "budget_fit",
            "facts", "summary", "highlights", "reviews"]
    return [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
            for r in results[cols].to_dict(orient="records")]


def filter_restaurants(district=None, category=None, party_size=None, budget=None, top_n=5):
    """相容舊版：直接回傳顯示用文字。"""
    results, error = search_restaurants(district, category, party_size, budget, top_n)
    return error if error else format_results(results)


if __name__ == "__main__":
    print("🧪 測試篩選結果（大安區、燒肉、2人、預算 1000 元）：\n")
    print(filter_restaurants(district="大安區", category="燒肉", party_size=2, budget=1000))
