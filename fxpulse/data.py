"""Data loaders: FX prices (Yahoo chart API), macro series (FRED's public
fredgraph.csv endpoint) and news headlines (Google News RSS).

No API keys anywhere. Every loader caches to data/ so the pipeline,
dashboard and eval run offline after one online refresh.
"""
import os
import re
import time
from datetime import datetime

import pandas as pd
import requests

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
HEADERS = {"User-Agent": "Mozilla/5.0 (FXPulse)"}

DEFAULT_PAIR = "USDINR=X"
MACRO_SERIES = {
    "DGS10": "US 10Y treasury yield",
    "DTWEXBGS": "Broad USD index",
    "DCOILWTICO": "WTI crude oil",
}
NEWS_QUERY = "USD INR rupee dollar RBI"


def _cache_path(name):
    os.makedirs(DATA, exist_ok=True)
    return os.path.join(DATA, name)


def _fresh(path, max_age_hours):
    return (os.path.exists(path)
            and time.time() - os.path.getmtime(path) < max_age_hours * 3600)


def fetch_fx(pair=DEFAULT_PAIR, refresh=False):
    """Daily FX closes, ~5 years. Returns DataFrame(date, close)."""
    path = _cache_path("fx_" + re.sub(r"[^A-Za-z0-9]", "_", pair) + ".csv")
    if not refresh and _fresh(path, 24):
        return pd.read_csv(path, parse_dates=["date"])
    url = "https://query1.finance.yahoo.com/v8/finance/chart/" + pair
    r = requests.get(url, params={"interval": "1d", "range": "5y"},
                     headers=HEADERS, timeout=15)
    res = r.json()["chart"]["result"][0]
    closes = res["indicators"]["quote"][0]["close"]
    df = pd.DataFrame({
        "date": pd.to_datetime(res["timestamp"], unit="s").normalize(),
        "close": closes,
    }).dropna().drop_duplicates("date").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def fetch_macro(refresh=False):
    """FRED series via the public fredgraph.csv endpoint (no key).
    Returns DataFrame(date, <one column per series>), daily, NaN on
    non-publication days."""
    path = _cache_path("macro.csv")
    if not refresh and _fresh(path, 24):
        return pd.read_csv(path, parse_dates=["date"])
    frames = []
    for sid in MACRO_SERIES:
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                         params={"id": sid}, headers=HEADERS, timeout=20)
        from io import StringIO
        raw = pd.read_csv(StringIO(r.text))
        raw.columns = ["date", sid]
        raw["date"] = pd.to_datetime(raw["date"])
        raw[sid] = pd.to_numeric(raw[sid], errors="coerce")
        frames.append(raw.set_index("date"))
    df = pd.concat(frames, axis=1).reset_index()
    df.to_csv(path, index=False)
    return df


def fetch_headlines(query=NEWS_QUERY, refresh=False):
    """Google News RSS headlines. Returns DataFrame(date, title).
    RSS only reaches back a few weeks, so the cache is append-only:
    every refresh adds new headlines and keeps the old ones."""
    path = _cache_path("headlines.csv")
    old = (pd.read_csv(path, parse_dates=["date"])
           if os.path.exists(path) else pd.DataFrame(columns=["date", "title"]))
    if not refresh and _fresh(path, 12):
        return old
    try:
        r = requests.get("https://news.google.com/rss/search",
                         params={"q": query}, headers=HEADERS, timeout=15)
        items = re.findall(r"<item>(.*?)</item>", r.text, re.S)
    except requests.RequestException:
        return old
    rows = []
    for it in items:
        m_t = re.search(r"<title>(.*?)</title>", it, re.S)
        m_d = re.search(r"<pubDate>(.*?)</pubDate>", it, re.S)
        if not (m_t and m_d):
            continue
        title = re.sub(r"<!\[CDATA\[|\]\]>", "", m_t.group(1)).strip()
        try:
            d = datetime.strptime(m_d.group(1).strip()[:16], "%a, %d %b %Y")
        except ValueError:
            continue
        rows.append({"date": pd.Timestamp(d), "title": title})
    new = pd.DataFrame(rows, columns=["date", "title"])
    allrows = pd.concat([old, new]).drop_duplicates("title")
    allrows = allrows.sort_values("date").reset_index(drop=True)
    if len(allrows):
        allrows.to_csv(path, index=False)
    return allrows
