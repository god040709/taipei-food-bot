import os
import math
import re
from urllib.parse import quote

import pandas as pd

# 資料庫路徑（以專案根目錄為準，從任何資料夾執行都找得到）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DF_PATH = os.path.join(BASE_DIR, "data", "taipei_all_restaurants.csv")
# 餐廳特色（enrich_features.py 產生，沒有這個檔案也能運作）
FEATURES_PATH = os.path.join(BASE_DIR, "data", "restaurant_features.csv")
# 台北市捷運站座標（tools/build_lookups.py 產生）
STATIONS_PATH = os.path.join(BASE_DIR, "data", "mrt_stations.csv")

# 「捷運站附近」的範圍（公尺）：先找 800 公尺（走路約 10 分鐘），不夠再放寬
NEAR_STEPS = (800, 1500)
# 「隱藏版」：評分夠高、評論數至少這麼多（太少的評分不可靠），且少於同區同類型的中位數
HIDDEN_MIN_RATING = 4.3
HIDDEN_MIN_REVIEWS = 50

# 加權評分的權重（加總為 1，可依需求調整）
W_RATING = 0.6      # 評分可信度
W_POPULARITY = 0.25  # 人氣（評論數）
W_BUDGET = 0.15     # 預算吻合度

# 貝氏平均的門檻：評論數少於這個數字時，評分會被拉向全體平均
BAYES_M = 100

# 預算的一個級距（價格大多以 200 元為一級）。使用者說要多花一點就加一級；
# 符合預算的店不夠時，依序加 1、2、3 級來補（找到足夠的店就停）
BUDGET_STEP = 200
RELAX_STEPS = (1, 2, 3)

# 從店名看得出來的特色
NAME_KEYWORDS = {
    "吃到飽": "吃到飽", "和牛": "和牛", "寵物友善": "寵物友善", "石頭": "石頭鍋",
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

    # 連鎖店：資料裡沒有這個欄位，用「同一個品牌名出現在 2 家以上不同的店」來推測
    df["brand"] = df["name"].map(brand_key)
    counts = df[df["brand"] != ""].groupby("brand")["place_id"].nunique()
    df["brand_count"] = df["brand"].map(counts).fillna(1).astype(int)
    df["is_chain"] = (df["brand"] != "") & (df["brand_count"] >= 2)

    # 隱藏版：評分高、但評論數比同區同類型的一半以上店家少（又不能少到評分不可靠）
    df["group_median_reviews"] = group["user_ratings_total"].transform("median")
    df["is_hidden"] = ((df["rating"] >= HIDDEN_MIN_RATING)
                       & (df["user_ratings_total"] >= HIDDEN_MIN_REVIEWS)
                       & (df["user_ratings_total"] < df["group_median_reviews"]))
    return df


BRAND_SEP = r"[｜|（(\-—－/／:：,，。·・\[\]【】「」]"


def brand_key(name) -> str:
    """店名 → 品牌名（用來判斷連鎖）。例如「1+1鍋物—忠孝店」→「1+1鍋物」、
    「好食多涮涮鍋 大安店」→「好食多涮涮鍋」。太短、太通用的（例如「麵屋」「The」）回傳空字串。"""
    token = re.split(BRAND_SEP, str(name).strip())[0].strip().lower()
    if re.match(r"[a-z0-9]", token):
        # 英文店名：取前兩個字（「TGI Fridays 信義店」→「tgi fridays」）
        m = re.match(r"[a-z0-9&'.+ ]+", token)
        key = " ".join(m.group(0).split()[:2]) if m else ""
        return key if len(key.replace(" ", "")) >= 5 else ""
    token = token.split()[0] if token.split() else ""
    # 去掉結尾的分店名：「鼎泰豐信義店」→「鼎泰豐」
    stripped = re.sub(r"(?:台北|臺北)?[一-鿿]{1,4}(?:分店|總店|門市|店)$", "", token)
    token = stripped if len(stripped) >= 3 else token
    return token if len(token) >= 3 else ""


def load_stations() -> dict:
    """捷運站名 → (緯度, 經度)。沒有座標檔時回傳空 dict。"""
    if not os.path.exists(STATIONS_PATH):
        return {}
    st = pd.read_csv(STATIONS_PATH)
    return {r["name"]: (float(r["lat"]), float(r["lng"])) for _, r in st.iterrows()}


def distance_m(df, lat, lng):
    """每家店到某個點的直線距離（公尺）。"""
    rad = math.pi / 180
    dlat = (df["lat"] - lat) * rad
    dlng = (df["lng"] - lng) * rad
    a = (dlat / 2).map(math.sin) ** 2 + math.cos(lat * rad) * (df["lat"] * rad).map(math.cos) \
        * (dlng / 2).map(math.sin) ** 2
    return (2 * 6371000 * a.map(lambda x: math.asin(math.sqrt(min(1.0, x))))).round()


def search_restaurants(district=None, category=None, party_size=None, budget=None, top_n=5,
                       exclude=None, subtype=None, subtype_keywords=None, subtype_cross=False, opts=None):
    """篩選並排序餐廳。回傳 (DataFrame, None)；失敗或沒結果時回傳 (None, 錯誤訊息)。
    exclude：要排除的餐廳名稱（使用者說「推薦別的」時，排除已經推薦過的店）。
    subtype / subtype_keywords：子類型（例如「壽喜燒」）與店名關鍵字。
        店名含關鍵字的店排在前面，其他同類別的店排在後面。
    subtype_cross：True 時，別的類別裡店名含關鍵字的店也算進來（例如被分在「燒肉」的壽喜燒店）。
        關鍵字太廣的子類型（例如「麵」）設 False，只在原本類別裡比對。
    opts：其他條件（都可以不給）
        near：捷運站名，找附近的店（不限行政區）
        categories：多個類別（「日式或韓式都可以」）
        avoid_categories：不要的類別（「剛吃過拉麵」）
        no_chain：不要連鎖店；hidden_gem：隱藏版；budget_min：每人預算下限
        sort：price_high（最貴的在前）、price_low（最便宜的在前）、near_first（近的在前）、
              rating（評分可信度高的在前）、popularity（評論數多的在前），同樣時再看推薦指數
        relax：["distance"] → 附近搜尋直接用最大範圍
        pin：要保留、排在最前面的店名（「換一批，但第 1 間留著」）"""
    if not os.path.exists(DF_PATH):
        return None, f"❌ 找不到資料庫檔案：{DF_PATH}，請確認是否已執行採集腳本。"

    try:
        all_df = load_data()
    except Exception as e:
        return None, f"❌ 資料庫讀取失敗: {e}"
    opts = dict(opts or {})
    info = {}

    # 「捷運站附近」：以車站為中心，不限行政區
    center, steps = None, None
    if opts.get("near"):
        point = load_stations().get(opts["near"])
        if point:
            # 「走遠一點沒關係」→ 直接用最大範圍
            center, steps = point, NEAR_STEPS[-1:] if "distance" in (opts.get("relax") or []) else NEAR_STEPS
            district = None
            info["near"] = opts["near"]
    if exclude:
        # 推薦過、不喜歡的店：同品牌的其他分店也不要（要保留的店除外）
        brands = set(all_df.loc[all_df["name"].isin(exclude), "brand"]) - {""}
        drop = all_df["name"].isin(exclude) | all_df["brand"].isin(brands)
        all_df = all_df[~drop | all_df["name"].isin(opts.get("pin") or [])]
    if center:
        all_df = all_df.copy()
        all_df["distance"] = distance_m(all_df, *center)

    kw = subtype_keywords or []
    cross_kw = kw if subtype_cross else None  # 用來把別的類別的店也納入

    # 附近的店不夠 top_n 家：範圍放寬一級
    if center:
        for radius in steps:
            opts["radius"] = radius
            if len(apply_filters(all_df, district, category, party_size, budget, cross_kw, opts)) >= top_n:
                break
        info["radius"] = opts["radius"]
        info["radius_relaxed"] = opts["radius"] != steps[0]
        info["radius_first"] = steps[0]

    df = apply_filters(all_df, district, category, party_size, budget, cross_kw, opts)
    # 排除連鎖店時，告訴使用者排掉了幾家
    if opts.get("no_chain"):
        info["chain_removed"] = len(apply_filters(all_df, district, category, party_size, budget, cross_kw,
                                                  {**opts, "no_chain": False})) - len(df)
    if budget:
        df["budget_fit"] = (df["max_price"] <= budget).map({True: "within", False: "partial"})
    else:
        df["budget_fit"] = ""
    matched = len(df)  # 符合原本條件的家數

    # 符合的不到 top_n 家：預算一點一點往上加，補上「稍微超過預算」的店
    relaxed_budget = None
    if budget and matched < top_n:
        for step in RELAX_STEPS:
            cap = budget + BUDGET_STEP * step
            extra = apply_filters(all_df, district, category, party_size, cap, cross_kw, opts)
            extra = extra[~extra["place_id"].isin(df["place_id"])]
            relaxed_budget = cap
            if matched + len(extra) >= top_n or step == RELAX_STEPS[-1]:
                break
        if len(extra):
            extra = extra.copy()
            extra["budget_fit"] = "over"
            df = pd.concat([df, extra])
        else:
            relaxed_budget = None

    # 有指定子類型、但預算內沒有這種店：預算一點一點往上加，找看看有沒有稍微超過預算的
    if budget and kw and not name_matches(df, kw).any():
        for step in RELAX_STEPS:
            cap = budget + BUDGET_STEP * step
            extra = apply_filters(all_df, district, category, party_size, cap, cross_kw, opts)
            extra = extra[name_matches(extra, kw) & ~extra["place_id"].isin(df["place_id"])]
            if len(extra):
                extra = extra.copy()
                extra["budget_fit"] = "over"
                df = pd.concat([df, extra])
                relaxed_budget = max(relaxed_budget or 0, cap)
                break

    if df.empty:
        return None, "⚠️ 沒有找到符合條件的餐廳，建議放寬預算或地區條件！"

    # 依加權推薦指數排序：完全在預算內 → 部分超出 → 稍微超過預算（越接近預算越前面）
    parts = score_components(df, budget)
    df = df.join(parts)
    df["score"] = parts.sum(axis=1).round(1)
    if opts.get("hidden_gem"):
        # 隱藏版：不看人氣（評論數），只看評分和預算
        df["score"] = (parts["s_rating"] + parts["s_budget"]).round(1)
    df["_fit_order"] = df["budget_fit"].map({"within": 0, "partial": 1, "over": 2}).fillna(0)
    df["_over_order"] = df["min_price"].where(df["budget_fit"] == "over", 0)
    df["over_by"] = (df["min_price"] - budget).clip(lower=0).where(df["budget_fit"] == "over") \
        if budget else None
    # 子類型優先：店名看得出是這種的店一定排最前面（就算比預算稍微高一點），
    # 接著是一般同類別的店（預算內 → 比預算稍微高一點）
    df["subtype_match"] = name_matches(df, kw) if kw else False
    df["_over"] = (df["budget_fit"] == "over").astype(int)
    df["_sub_order"] = (~df["subtype_match"]).astype(int)
    # 附近搜尋放寬範圍時：原本範圍內（例如 800 公尺）的店排在前面
    df["_far"] = (df["distance"] > steps[0]).astype(int) if center else 0
    # 使用者說要保留的店排最前面
    df["_pin"] = (~df["name"].isin(opts.get("pin") or [])).astype(int)
    if opts.get("sort") in ("price_high", "price_low"):
        # 依價格排序：用多數人花費區間的中間值；預算放寬才補上的店照樣排在後面
        df["_price"] = (df["min_price"] + df["max_price"]) / 2
        df = df.sort_values(by=["_pin", "_sub_order", "_over", "_far", "_price", "score"],
                            ascending=[True, True, True, True, opts["sort"] == "price_low", False])
    else:
        # 使用者強調的重點：在同一種預算等級裡，依那一項排（越近越好要有車站才算得出距離）
        key = {"near_first": ("distance", True) if center else None, "rating": ("s_rating", False),
               "popularity": ("user_ratings_total", False)}.get(opts.get("sort"))
        by = ["_pin", "_sub_order", "_over", "_far", "_fit_order", "_over_order"] + ([key[0]] if key else [])
        df = df.sort_values(by=by + ["score"],
                            ascending=[True] * 6 + ([key[1]] if key else []) + [False])
    # 同品牌只留一家：有車站時留最近的，沒有時留評分可信度最高的
    # （要保留的店、店名符合子類型的、預算內的優先）
    branded = df[df["brand"] != ""]
    if branded["brand"].duplicated().any():
        pick = branded.sort_values(by=["_pin", "_sub_order", "_over", "distance" if center else "s_rating"],
                                   ascending=[True, True, True, bool(center)], kind="mergesort")
        keep = pick.drop_duplicates("brand")["place_id"]
        df = df[(df["brand"] == "") | df["place_id"].isin(keep)]
        matched = int((df["budget_fit"] != "over").sum())  # 符合原本條件的家數（不算重複的分店）
    cats = opts.get("categories") or []
    if len(cats) >= 2 and not opts.get("sort"):
        # 多個類別（「日式或韓式都可以」）：各類別輪流排，不會整批都是同一種
        df["_cat_rank"] = df.groupby("category").cumcount()
        df = df.sort_values(by=["_sub_order", "_over", "_cat_rank"], kind="mergesort")

    results = df.head(top_n).copy()
    has_within = bool((results["budget_fit"] == "within").any())
    results["maps_url"] = results.apply(maps_link, axis=1)
    results["facts"] = results.apply(lambda r: describe_facts(r, budget, has_within, opts), axis=1)
    if subtype:
        results["facts"] = [
            ([f"店名看得出是{subtype}"] + [f for f in facts if not f.startswith("店名特色")]
             + [f for f in facts if f.startswith("店名特色") and subtype not in f]) if match else facts
            for facts, match in zip(results["facts"], results["subtype_match"])
        ]
    results.attrs["matched_subtype"] = int(df["subtype_match"].sum()) if kw else None
    results.attrs["matched"] = matched
    results.attrs["matched_within"] = int((df["budget_fit"] == "within").sum()) if budget else matched
    results.attrs["relaxed_budget"] = relaxed_budget if (results["budget_fit"] == "over").any() else None
    # 補完還是不到 top_n 家：算出放寬哪個條件可以多找到幾家
    results.attrs["alternatives"] = suggest_relaxations(all_df, district, category, party_size, budget,
                                                         shown=len(results), opts=opts) \
        if len(results) < top_n else []
    results.attrs.update(info)
    return results, None


def name_matches(df, keywords):
    """店名是否含任一關鍵字（不分大小寫）。"""
    names = df["name"].astype(str).str.lower()
    mask = pd.Series(False, index=df.index)
    for k in keywords:
        mask |= names.str.contains(k.lower(), regex=False)
    return mask


def apply_filters(df, district=None, category=None, party_size=None, budget=None, name_keywords=None,
                  opts=None):
    """依條件篩選。預算是「每人上限」：
    價格區間（min_price ~ max_price）是 Google 的「多數人用餐花費」區間，不是最低／最高消費。
    - within  = 多數人的花費都在預算內（區間上限 ≤ 預算）→ 優先推薦
    - partial = 預算落在花費區間中間（區間下限 < 預算 < 上限）→ 排在後面，並標示
    - 區間下限 ≥ 預算的店（例如預算 800、多數人花 800~1000）不算符合預算，
      只在符合的店不夠時，當作「比預算稍微高一點」的選擇補在最後
    opts：categories（多個類別）、avoid_categories、radius（要先有 distance 欄位）、
          no_chain、hidden_gem、budget_min"""
    opts = opts or {}
    if district and district != "全區":
        df = df[df['district'].str.contains(district, regex=False, na=False)]
    cats = [c for c in (opts.get("categories") or []) if c] or ([category] if category and category != "不限" else [])
    if cats:
        in_category = pd.Series(False, index=df.index)
        for c in cats:
            in_category |= df['category'].str.contains(c, regex=False, na=False)
        if name_keywords:
            # 子類型：店名含關鍵字的店，就算原本被分在別的類別也算（例如壽喜燒店被分在燒肉）
            in_category |= name_matches(df, name_keywords)
        df = df[in_category]
    if opts.get("avoid_categories"):
        df = df[~df["category"].isin(opts["avoid_categories"])]
    if opts.get("radius") and "distance" in df.columns:
        df = df[df["distance"] <= opts["radius"]]
    if opts.get("no_chain") and "is_chain" in df.columns:
        df = df[~df["is_chain"]]
    if opts.get("hidden_gem") and "is_hidden" in df.columns:
        df = df[df["is_hidden"]]
    if party_size:
        df = df[(df['min_party'] <= party_size) & (df['max_party'] >= party_size)]
    if budget:
        df = df[df['min_price'] < budget]
    if opts.get("budget_min"):
        # 預算下限：多數人的花費至少要到這個金額（「不要太寒酸」「500 到 1000」）
        df = df[df['max_price'] >= opts["budget_min"]]
    return df.copy()


def relaxations_for(district=None, category=None, party_size=None, budget=None, exclude=None, opts=None):
    """給 API 在「完全沒有結果」時使用。"""
    try:
        df = load_data()
        if exclude:
            df = df[~df["name"].isin(exclude)]
        opts = dict(opts or {})
        if opts.get("near"):
            point = load_stations().get(opts["near"])
            if point:
                df = df.copy()
                df["distance"] = distance_m(df, *point)
                opts["radius"] = NEAR_STEPS[-1]
                district = None
        return suggest_relaxations(df, district, category, party_size, budget, opts=opts)
    except Exception:
        return []


def subtype_overview(category, keywords, cross=False, siblings=None, district=None, party_size=None,
                     budget=None, opts=None):
    """子類型的店家數，讓機器人誠實說明「有沒有這種店」：
    total：整個資料庫裡店名看得出是這種的家數（不限地區、預算）
    general：目前條件下，同類別的店有幾家
    siblings：目前條件下，同類別的其他子類型各有幾家（只列有店的，家數多的在前）：[[子類型, 家數], ...]
    siblings 的格式：{子類型: [店名關鍵字, 是否跨類別]}"""
    df = load_data()

    def subtype_mask(d, kw, cr):
        in_cat = d["category"].str.contains(category, regex=False, na=False) if category else True
        return name_matches(d, kw) & (True if cr else in_cat)

    total = int(subtype_mask(df, keywords, cross).sum()) if keywords else 0
    opts = dict(opts or {})
    if opts.get("near"):
        point = load_stations().get(opts["near"])
        if point:
            df = df.copy()
            df["distance"] = distance_m(df, *point)
            opts["radius"] = NEAR_STEPS[-1]
            district = None
    here = apply_filters(df, district, None, party_size, budget, opts={**opts, "categories": None})
    general = int(here["category"].str.contains(category, regex=False, na=False).sum()) if category else len(here)
    counts = {}
    for name, (kw, cr) in (siblings or {}).items():
        n = int(subtype_mask(here, kw, cr).sum())
        if n:
            counts[name] = n
    # 用 list 回傳：JSON 物件的順序可能被重排（Flask 會照字母排序 key）
    return {"total": total, "general": general,
            "siblings": sorted(([name, n] for name, n in counts.items()), key=lambda x: -x[1])}


def category_counts(district=None, party_size=None, budget=None):
    """各料理類型有幾家店（使用者問「有什麼類型可以選」時用）。"""
    df = apply_filters(load_data(), district, None, party_size, budget)
    counts = df["category"].value_counts()
    return [{"category": c, "count": int(n)} for c, n in counts.items()]


def suggest_relaxations(df, district, category, party_size, budget, shown=0, opts=None):
    """結果太少時，建議放寬的方式，例如「不限地區 → 64 家」「預算 500 → 5 家」。
    shown：目前已經顯示幾家；建議要能找到比這更多的店才列出來。"""
    opts = opts or {}
    current = max(len(apply_filters(df, district, category, party_size, budget, opts=opts)), shown)
    where = district or ("附近" if opts.get("radius") else "全台北")
    options = []

    def count(**changes):
        args = {"district": district, "category": category, "budget": budget, "opts": opts}
        args.update(changes)
        return len(apply_filters(df, args["district"], args["category"], party_size, args["budget"],
                                 opts=args["opts"]))

    if district or opts.get("radius"):
        n = count(district=None, opts={**opts, "radius": None})
        if n > current:
            options.append({"reply": "不限地區", "text": f"全台北有 {n} 家符合", "count": n})
    if budget:
        for step in range(1, 11):  # 一級一級往上加（最多 +2000），找到能多出店家的金額
            new_budget = budget + BUDGET_STEP * step
            n = count(budget=new_budget)
            if n > current:
                options.append({"reply": f"預算 {new_budget}", "text": f"{where}有 {n} 家符合", "count": n})
                break
    if category or opts.get("categories"):
        n = count(category=None, opts={**opts, "categories": None})
        if n > current:
            options.append({"reply": "類型都可以", "text": f"{where}各種料理共 {n} 家符合", "count": n})
    if opts.get("no_chain"):
        n = count(opts={**opts, "no_chain": False})
        if n > current:
            options.append({"reply": "連鎖店也可以", "text": f"{where}有 {n} 家符合", "count": n})
    if opts.get("hidden_gem"):
        n = count(opts={**opts, "hidden_gem": False})
        if n > current:
            options.append({"reply": "有名的也可以", "text": f"{where}有 {n} 家符合", "count": n})
    if opts.get("budget_min"):
        n = count(opts={**opts, "budget_min": None})
        if n > current:
            options.append({"reply": "便宜一點的也可以", "text": f"{where}有 {n} 家符合", "count": n})
    return options


def describe_facts(row, budget=None, has_within=True, opts=None):
    """從資料推得出來的客觀事實，給 LLM 寫推薦理由（也是沒有 LLM 時的推薦理由）。"""
    opts = opts or {}
    facts = []
    if opts.get("hidden_gem") and row.get("is_hidden"):
        facts.append(f"評論只有 {int(row['user_ratings_total'])} 則、評分卻有 {row['rating']}，"
                     f"是比較少人知道的店")
    name = str(row["name"])
    tags = []
    for word, tag in NAME_KEYWORDS.items():
        if word.lower() in name.lower() and tag not in tags:
            tags.append(tag)
    if tags:
        facts.append("店名特色：" + "、".join(tags))

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

    if budget and row["budget_fit"] == "within":
        facts.append(f"多數人的花費都在每人 {budget} 元預算內")
    return facts


def to_records(results):
    """給 Pipe 顯示與 LLM 寫推薦理由用的資料。"""
    cols = ["name", "district", "category", "rating", "user_ratings_total",
            "min_price", "max_price", "score", "address", "maps_url", "budget_fit",
            "over_by", "subtype_match", "facts", "summary", "highlights", "reviews",
            "distance", "is_chain", "is_hidden"]
    cols = [c for c in cols if c in results.columns]
    records = [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
               for r in results[cols].to_dict(orient="records")]
    for r in records:  # NaN 不能轉成 JSON
        for k, v in r.items():
            if isinstance(v, float) and math.isnan(v):
                r[k] = None
    return records
