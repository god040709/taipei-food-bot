"""
查餐廳的服務：/recommend、/categories 的內容。api.py 和機器人（bot.py）都用這裡的函式。
"""
from recommender import (
    category_counts, format_results, relaxations_for, search_restaurants, subtype_overview, to_records,
)


def handle_recommend(data: dict) -> dict:
    conditions = dict(
        district=data.get("district"),
        category=data.get("category"),
        party_size=data.get("party_size"),
        budget=data.get("budget"),
    )
    exclude = data.get("exclude") or []  # 已經推薦過、這次要排除的店名
    # 其他條件：捷運站附近、多類別、不要的類別、不要連鎖、隱藏版、預算下限
    opts = {k: data.get(k) for k in ("near", "categories", "avoid_categories",
                                     "no_chain", "hidden_gem", "budget_min", "sort") if data.get(k)}
    top_n = data.get("top_n") or 5
    # subtype：子類型（例如「壽喜燒」），subtype_keywords：用來比對店名的關鍵字
    results, error = search_restaurants(**conditions, exclude=exclude, top_n=top_n,
                                        subtype=data.get("subtype"),
                                        subtype_keywords=data.get("subtype_keywords") or [],
                                        subtype_cross=bool(data.get("subtype_cross")),
                                        opts=opts)
    # 子類型的店家數：整個資料庫有沒有這種店、同類別其他子類型各有幾家（讓機器人誠實說明）
    overview = {}
    if data.get("subtype"):
        overview = {"subtype_overview": subtype_overview(
            conditions["category"], data.get("subtype_keywords") or [], bool(data.get("subtype_cross")),
            data.get("subtype_siblings") or {}, conditions["district"], conditions["party_size"],
            conditions["budget"], opts)}
    if error:
        return {"result": error, "restaurants": [], "matched": 0,
                "alternatives": relaxations_for(**conditions, exclude=exclude, opts=opts), **overview}
    # result：顯示用文字；restaurants：結構化資料，給 LLM 寫推薦理由
    # matched：符合條件的餐廳總數；matched_within：價格完全在預算內的家數
    # alternatives：結果太少時，放寬條件的建議
    return {"result": format_results(results), "restaurants": to_records(results),
            "matched": results.attrs.get("matched", len(results)),
            "matched_within": results.attrs.get("matched_within", len(results)),
            "alternatives": results.attrs.get("alternatives", []),
            # relaxed_budget：符合預算的店不夠時，放寬到的預算（後面補上的店標示為 over）
            "relaxed_budget": results.attrs.get("relaxed_budget"),
            # matched_subtype：店名看得出是這個子類型的家數（沒有指定子類型時是 None）
            "matched_subtype": results.attrs.get("matched_subtype"),
            # near / radius / radius_relaxed：附近搜尋用的車站、範圍（公尺）、是否放寬過
            # chain_removed：排除了幾家連鎖店
            **{k: results.attrs[k] for k in ("near", "radius", "radius_relaxed", "radius_first",
                                             "chain_removed") if k in results.attrs},
            **overview}


def handle_categories(data: dict) -> dict:
    return {"categories": category_counts(
        district=data.get("district"),
        party_size=data.get("party_size"),
        budget=data.get("budget"),
    )}
