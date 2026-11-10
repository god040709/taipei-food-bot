"""
產生機器人用的查詢資料（只需要在本機跑，虛擬機和 Open WebUI 不用安裝任何套件）：

1. data/mrt_stations.csv：台北市捷運站座標（找「捷運站附近」的店）
2. data/lexicon_generated.json：
   - MRT_STATIONS：站名清單（判斷使用者說的是不是車站）
   - TERM_READINGS / PINYIN_CHARS：同音錯字比對用的拼音表

用法：
    pip install pypinyin
    python tools/build_lookups.py                      # 修改 src/lexicon.py 的詞庫後，重新產生拼音表
    python tools/build_lookups.py 車站資料.geojson      # 連捷運站座標一起更新

車站資料來源：https://github.com/repeat/northern-taiwan-metro-stations
（output/northern-taiwan.geojson）
"""
import csv
import json
import os
import re
import sys

from pypinyin import Style, lazy_pinyin, pinyin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIONS_CSV = os.path.join(ROOT, "data", "mrt_stations.csv")
GENERATED = os.path.join(ROOT, "data", "lexicon_generated.json")


def load_geojson(path):
    """只留台北市的站；轉乘站（同名多筆）取平均座標。"""
    features = json.load(open(path, encoding="utf-8"))["features"]
    points = {}
    for f in features:
        p = f["properties"]
        if not re.match(r"(台北市|臺北市)", p.get("地址", "")):
            continue
        points.setdefault(p["中文站名"], []).append((float(p["緯度"]), float(p["經度"])))
    return {name: (sum(a for a, _ in pts) / len(pts), sum(b for _, b in pts) / len(pts))
            for name, pts in sorted(points.items())}


def load_csv():
    with open(STATIONS_CSV, encoding="utf-8") as f:
        return {r["name"]: (float(r["lat"]), float(r["lng"])) for r in csv.DictReader(f)}


def readings(text):
    return lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)


def main(geojson=None):
    if geojson:
        stations = load_geojson(geojson)
        with open(STATIONS_CSV, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "lat", "lng"])
            for name, (lat, lng) in stations.items():
                w.writerow([name, f"{lat:.6f}", f"{lng:.6f}"])
        print(f"寫入 {STATIONS_CSV}（{len(stations)} 站）")
    else:
        stations = load_csv()

    sys.path.insert(0, os.path.join(ROOT, "src"))
    import lexicon  # noqa: E402

    terms = sorted(lexicon.homophone_terms(list(stations)))
    term_readings = {t: " ".join(readings(t)) for t in terms}
    toneless = {r.rstrip("012345") for t in terms for r in readings(t)}

    # 常用繁體字（Big5 編得出來的）裡，讀音跟這些詞有關的字
    table = {}
    for code in range(0x4E00, 0xA000):
        ch = chr(code)
        try:
            ch.encode("big5")
        except UnicodeEncodeError:
            continue
        for r in set(pinyin(ch, style=Style.TONE3, heteronym=True, neutral_tone_with_five=True)[0]):
            if r.rstrip("012345") in toneless:
                table.setdefault(r, []).append(ch)
    table = {r: "".join(chs) for r, chs in sorted(table.items())}

    with open(GENERATED, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"MRT_STATIONS": list(stations), "TERM_READINGS": term_readings, "PINYIN_CHARS": table},
                  f, ensure_ascii=False)
    size = sum(len(v) for v in table.values())
    print(f"寫入 {GENERATED}：{len(terms)} 個詞、拼音表 {len(table)} 個讀音／{size} 個字")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        sys.exit(__doc__)
    main(sys.argv[1] if len(sys.argv) == 2 else None)
