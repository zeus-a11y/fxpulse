"""Run the full FXPulse pipeline: fetch -> features -> walk-forward
backtest -> results JSON + insight report.

    python pipeline.py                 # cached data, all models incl. GRU
    python pipeline.py --refresh       # re-fetch FX, macro and headlines
    python pipeline.py --no-gru        # skip the neural model (faster)
    python pipeline.py --pair EURINR=X
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

from fxpulse import data as D
from fxpulse import features as F
from fxpulse import models as M
from fxpulse import sentiment as S
from fxpulse.backtest import walk_forward

HERE = os.path.dirname(os.path.abspath(__file__))


def run(pair=D.DEFAULT_PAIR, refresh=False, with_gru=True):
    fx = D.fetch_fx(pair, refresh=refresh)
    macro = D.fetch_macro(refresh=refresh)
    heads = D.fetch_headlines(refresh=refresh)
    senti = S.daily_sentiment(heads)
    df, feats = F.build_dataset(fx, macro, senti)
    macro_cols = [c for c in feats if c.startswith("m_")]
    models = M.model_set(feats, macro_cols, with_gru=with_gru)
    print("rows: {} | features: {} | models: {}".format(
        len(df), len(feats), ", ".join(m.name for m in models)))
    results = walk_forward(df, feats, models)
    results["pair"] = pair
    results["as_of"] = str(df["date"].iloc[-1].date())
    results["last_close"] = float(df["close"].iloc[-1])
    results["n_headlines"] = int(len(heads))
    out = os.path.join(D.DATA, "results.json")
    with open(out, "w") as fh:
        json.dump(results, fh)
    df.to_csv(os.path.join(D.DATA, "dataset.csv"), index=False)
    print("\n{:<16} {:>9} {:>14}".format("model", "dir.acc", "strategy ret"))
    for m in models:
        r = results[m.name]
        print("{:<16} {:>8.1f}% {:>13.2f}%".format(
            m.name, 100 * r["accuracy"], r["strategy_return_pct"]))
    print("{:<16} {:>9} {:>13.2f}%".format("buy & hold", "-",
                                           results["buy_hold_return_pct"]))
    print("\nwrote " + out)
    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    pair = D.DEFAULT_PAIR
    if "--pair" in args:
        pair = args[args.index("--pair") + 1]
    run(pair=pair, refresh="--refresh" in args,
        with_gru="--no-gru" not in args)
