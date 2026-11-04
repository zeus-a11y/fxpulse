"""Eval harness for FXPulse.

    python eval.py      deterministic checks, no network, no API key
                        (needs one prior `python pipeline.py` for cache)

The checks that matter most here are the leakage checks: a daily-FX
model with impressive accuracy is usually a model that peeked.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from fxpulse import data as D
from fxpulse import features as F
from fxpulse import sentiment as S
from fxpulse.backtest import walk_forward
from fxpulse.models import NaiveUp, Momentum

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def _ctx():
    fx = D.fetch_fx()
    macro = D.fetch_macro()
    senti = S.daily_sentiment(D.fetch_headlines())
    df, feats = F.build_dataset(fx, macro, senti)
    with open(os.path.join(D.DATA, "results.json")) as fh:
        res = json.load(fh)
    return fx, macro, senti, df, feats, res


@check("no lookahead: truncating the future never changes a past feature row")
def _c1(fx, macro, senti, df, feats, res):
    cut = len(fx) - 30
    df_cut, _ = F.build_dataset(fx.iloc[:cut], macro, senti)
    full_row = df[df["date"] == df_cut["date"].iloc[-1]]
    assert len(full_row) == 1
    a = df_cut.iloc[-1][feats].astype(float).values
    b = full_row.iloc[0][feats].astype(float).values
    assert np.allclose(a, b, equal_nan=True), "features changed when the future was removed"


@check("target is strictly tomorrow's direction")
def _c2(fx, macro, senti, df, feats, res):
    scored = df.dropna(subset=["target"])
    sample = scored.sample(50, random_state=1)
    closes = df.set_index("date")["close"]
    dates = list(df["date"])
    for _, row in sample.iterrows():
        i = dates.index(row["date"])
        want = 1.0 if closes.iloc[i + 1] > closes.iloc[i] else 0.0
        assert row["target"] == want


@check("macro features only use information from before each day")
def _c3(fx, macro, senti, df, feats, res):
    m = macro.sort_values("date").set_index("date")["DGS10"].dropna()
    sample = df.dropna(subset=["m_DGS10"]).sample(25, random_state=2)
    for _, row in sample.iterrows():
        known = m[m.index < row["date"]]
        assert len(known)
        assert abs(row["m_DGS10"] - known.iloc[-1]) < 1e-9, row["date"]


@check("walk-forward folds never train on or after the fold start")
def _c4(fx, macro, senti, df, feats, res):
    for name in res:
        if isinstance(res[name], dict) and "folds" in res[name]:
            for train_end, test_start, test_end in res[name]["folds"]:
                assert train_end < test_start <= test_end


@check("reported accuracy recomputes from stored predictions and targets")
def _c5(fx, macro, senti, df, feats, res):
    y = np.array(res["targets"])
    for name in res:
        if isinstance(res[name], dict) and "accuracy" in res[name]:
            pred = np.array(res[name]["pred"])
            assert len(pred) == len(y)
            assert abs(float((pred == y).mean())
                       - res[name]["accuracy"]) < 1e-6, name


@check("strategy return recomputes from predictions and next-day returns")
def _c6(fx, macro, senti, df, feats, res):
    scored = df.dropna(subset=["target"]).reset_index(drop=True)
    start = len(scored) - res["n_test"]
    next_ret = scored["ret"].shift(-1).values[start:]
    name = "xgboost"
    pred = np.array(res[name]["pred"])
    strat = np.where(pred == 1, np.nan_to_num(next_ret), 0.0)
    want = round(100 * (np.prod(1 + strat) - 1), 2)
    assert abs(want - res[name]["strategy_return_pct"]) < 0.02


@check("naive baseline sits in the coin-flip band (40-70%)")
def _c7(fx, macro, senti, df, feats, res):
    assert 0.40 <= res["naive_majority"]["accuracy"] <= 0.70


@check("backtester scores a known toy sequence correctly")
def _c8(fx, macro, senti, df, feats, res):
    # strictly rising closes -> target always 1 -> NaiveUp scores 100%
    n = 600
    toy_fx = pd.DataFrame({
        "date": pd.date_range("2023-01-02", periods=n, freq="B"),
        "close": np.linspace(50, 120, n)})
    toy, toy_feats = F.build_dataset(toy_fx)
    out = walk_forward(toy, toy_feats, [NaiveUp(), Momentum()],
                       test_size=200, refit_every=50)
    assert out["naive_majority"]["accuracy"] == 1.0
    assert out["momentum_1d"]["accuracy"] == 1.0


@check("sentiment scorer: rallies positive, crashes negative")
def _c9(fx, macro, senti, df, feats, res):
    assert S.score_text("Rupee rallies as inflows surge, markets cheer") > 0
    assert S.score_text("Rupee crashes to record low amid panic selling") < 0


@check("daily sentiment aggregates to one row per day with counts")
def _c10(fx, macro, senti, df, feats, res):
    if len(senti) == 0:
        return
    assert senti["date"].is_unique
    assert (senti["n_headlines"] >= 1).all()
    assert senti["sentiment"].between(-1, 1).all()


def main():
    ctx = _ctx()
    failed = 0
    for name, fn in CHECKS:
        try:
            fn(*ctx)
            print("[PASS] " + name)
        except AssertionError as exc:
            failed += 1
            print("[FAIL] {} - {}".format(name, exc))
        except Exception as exc:
            failed += 1
            print("[ERROR] {} - {}: {}".format(name, type(exc).__name__,
                                               exc))
    print("deterministic: {}/{} checks passed".format(len(CHECKS) - failed,
                                                      len(CHECKS)))
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
