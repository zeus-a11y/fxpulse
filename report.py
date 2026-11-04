"""Automated insight report: turns the latest data, backtest results and
news sentiment into a stakeholder-ready markdown note.

    python report.py          # deterministic note -> data/insight_report.md
    python report.py --llm    # + Claude-written summary of the SAME figures

Every figure comes from the pipeline's results.json; the optional LLM
pass only writes prose around numbers it is handed.
"""
import json
import os
import sys

import pandas as pd

from fxpulse import data as D
from fxpulse import sentiment as S

HERE = os.path.dirname(os.path.abspath(__file__))

HONESTY = ("Daily FX direction is close to a coin flip; the honest story "
           "is the margin over the naive baseline and the strategy "
           "return, not a headline accuracy.")


def build():
    with open(os.path.join(D.DATA, "results.json")) as fh:
        res = json.load(fh)
    df = pd.read_csv(os.path.join(D.DATA, "dataset.csv"),
                     parse_dates=["date"])
    heads = D.fetch_headlines()
    senti = S.daily_sentiment(heads)

    last, prev30 = df["close"].iloc[-1], df["close"].iloc[-22]
    recent_s = senti.tail(5)
    model_names = [k for k in res
                   if isinstance(res[k], dict) and "accuracy" in res[k]]
    best = max(model_names, key=lambda k: res[k]["strategy_return_pct"])
    live_proba = {k: res[k]["proba"][-1] for k in model_names}

    lines = []
    add = lines.append
    add("# FXPulse insight report - {} ({})".format(res["pair"],
                                                    res["as_of"]))
    add("")
    add("- Last close: **{:.4f}**; one-month move {:+.2f}%".format(
        last, 100 * (last / prev30 - 1)))
    if len(recent_s):
        add("- News sentiment (VADER over {} headlines, 5-day): "
            "**{:+.3f}**".format(int(recent_s["n_headlines"].sum()),
                                 recent_s["sentiment"].mean()))
    add("- Walk-forward test: last **{}** trading days, expanding window, "
        "refit every 25".format(res["n_test"]))
    add("")
    add("| Model | Directional accuracy | Strategy return | Buy & hold |")
    add("| --- | --- | --- | --- |")
    for k in model_names:
        add("| {} | {:.1f}% | {:+.2f}% | {:+.2f}% |".format(
            k, 100 * res[k]["accuracy"], res[k]["strategy_return_pct"],
            res["buy_hold_return_pct"]))
    add("")
    add("**Best strategy:** `{}` ({:+.2f}% vs {:+.2f}% buy & hold). "
        "{}".format(best, res[best]["strategy_return_pct"],
                    res["buy_hold_return_pct"], HONESTY))
    add("")
    add("**Tomorrow's signal (prob. of closing higher):** "
        + ", ".join("{} {:.0f}%".format(k, 100 * v)
                    for k, v in live_proba.items())
        + ". Signals are probabilities, not advice.")
    add("")
    if len(heads):
        add("**Latest headlines (scored):**")
        add("")
        for _, row in heads.tail(5).iloc[::-1].iterrows():
            add("- ({:+.2f}) {}".format(S.score_text(row["title"]),
                                        row["title"]))
    return "\n".join(lines), res


def llm_summary(note):
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=os.environ.get("FXPULSE_MODEL", "claude-sonnet-5-5"),
        max_tokens=400,
        system=("Write a 90-130 word executive summary of this FX report "
                "for a non-technical stakeholder. Use ONLY figures present "
                "in the report, keep the honest-baseline caveat, no "
                "advice."),
        messages=[{"role": "user", "content": note}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def main():
    note, _ = build()
    if "--llm" in sys.argv and os.environ.get("ANTHROPIC_API_KEY"):
        note = note.replace(
            "\n\n| Model |",
            "\n\n## Executive summary\n\n" + llm_summary(note)
            + "\n\n| Model |", 1)
    out = os.path.join(D.DATA, "insight_report.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(note)
    print("wrote " + out)


if __name__ == "__main__":
    main()
