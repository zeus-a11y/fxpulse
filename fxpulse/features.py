"""Feature engineering with a hard no-lookahead rule: every feature for
day t is computed from information available at the close of day t, and
the target is the NEXT day's direction. Macro series and sentiment are
additionally lagged one day, since FRED publishes with a delay and news
sentiment for 'today' is only complete after the close.
"""
import numpy as np
import pandas as pd

FEATURE_COLS = None  # filled by build_dataset


def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # all-up and all-flat windows: RSI is 100 / 50 by convention, not NaN
    out = out.where(~(down.eq(0) & up.gt(0)), 100.0)
    out = out.where(~(down.eq(0) & up.eq(0)), 50.0)
    return out


def build_dataset(fx, macro=None, senti=None):
    """Returns (df, feature_cols). df has one row per trading day with
    features, 'ret' (that day's return), and 'target' = 1 if the NEXT
    day's close is higher, else 0. The last row has target NaN (it is
    tomorrow's unknown direction) and is kept for live prediction."""
    df = fx.sort_values("date").reset_index(drop=True).copy()
    df["ret"] = df["close"].pct_change()
    for k in (1, 2, 3, 4, 5):
        df["ret_lag%d" % k] = df["ret"].shift(k - 1)
    df["vol_5"] = df["ret"].rolling(5).std()
    df["vol_20"] = df["ret"].rolling(20).std()
    df["rsi_14"] = rsi(df["close"])
    df["mom_10"] = df["close"].pct_change(10)
    feats = ["ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4", "ret_lag5",
             "vol_5", "vol_20", "rsi_14", "mom_10"]

    if macro is not None and len(macro):
        m = macro.sort_values("date").set_index("date")
        m = m.reindex(pd.date_range(m.index.min(), df["date"].max()))
        m = m.ffill()
        m_lagged = m.shift(1)  # only yesterday's macro is knowable
        for col in m.columns:
            df["m_" + col] = m_lagged[col].reindex(df["date"]).values
            df["m_" + col + "_chg"] = (m_lagged[col].pct_change(5)
                                       .reindex(df["date"]).values)
            feats += ["m_" + col, "m_" + col + "_chg"]

    if senti is not None and len(senti):
        s = senti.sort_values("date").set_index("date")["sentiment"]
        s = s.reindex(pd.date_range(s.index.min(), df["date"].max()))
        s_lagged = s.shift(1).rolling(3, min_periods=1).mean()
        df["sentiment"] = s_lagged.reindex(df["date"]).fillna(0.0).values
        feats.append("sentiment")

    df["target"] = (df["close"].shift(-1) > df["close"]).astype(float)
    df.loc[df.index[-1], "target"] = np.nan  # tomorrow is unknown
    df = df.dropna(subset=feats + ["ret"]).reset_index(drop=True)
    return df, feats
