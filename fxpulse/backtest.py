"""Walk-forward backtest with an expanding window: fit on everything
before the fold, predict the fold, step forward. The model never sees a
row at or after the day it predicts - the eval harness asserts this from
the fold bookkeeping, so a leak is a test failure, not a footnote.
"""
import numpy as np


def walk_forward(df, feats, models, test_size=250, refit_every=25):
    """Returns {model_name: {"preds": [...], "proba": [...], "acc": float,
    "strategy_return": float, "folds": [(train_end, test_start, test_end)]}}
    plus the actual targets. Rows with NaN target (the live row) are
    excluded from scoring."""
    scored = df.dropna(subset=["target"]).reset_index(drop=True)
    n = len(scored)
    start = n - test_size
    if start < 200:
        raise ValueError("not enough history for a %d-day test" % test_size)
    results = {m.name: {"proba": [], "folds": []} for m in models}
    for t0 in range(start, n, refit_every):
        t1 = min(t0 + refit_every, n)
        train, test = scored.iloc[:t0], scored.iloc[t0:t1]
        for m in models:
            m.fit(train, feats)
            p = np.clip(np.asarray(m.predict_proba_up(test), dtype=float),
                        0.0, 1.0)
            results[m.name]["proba"].extend(p.tolist())
            results[m.name]["folds"].append((int(t0 - 1), int(t0),
                                             int(t1 - 1)))
    y = scored["target"].values[start:].astype(int)
    next_ret = scored["ret"].shift(-1).values[start:]
    out = {"n_test": int(n - start), "targets": y.tolist()}
    for name, r in results.items():
        proba = np.array(r["proba"])
        pred = (proba > 0.5).astype(int)
        acc = float((pred == y).mean())
        # long when predicting up, flat otherwise; next day's return
        strat = np.where(pred == 1, np.nan_to_num(next_ret), 0.0)
        out[name] = {
            "accuracy": round(acc, 4),
            "proba": [round(float(x), 4) for x in proba],
            "pred": pred.tolist(),
            "strategy_return_pct": round(100 * (np.prod(1 + strat) - 1), 2),
            "folds": r["folds"],
        }
    bh = np.nan_to_num(next_ret)
    out["buy_hold_return_pct"] = round(100 * (np.prod(1 + bh) - 1), 2)
    return out
