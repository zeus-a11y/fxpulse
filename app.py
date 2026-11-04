"""FXPulse Streamlit dashboard.

    streamlit run app.py

Reads the cached pipeline outputs (run `python pipeline.py` first);
the Refresh button re-fetches data and re-runs the backtest live.
"""
import json
import os

import pandas as pd
import streamlit as st

from fxpulse import data as D
from fxpulse import sentiment as S

HERE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="FXPulse", layout="wide")
st.title("FXPulse - forex forecasting & sentiment analytics")
st.caption("Walk-forward, no-lookahead backtests. Daily FX direction is "
           "near a coin flip - judge models by their margin over the "
           "naive baseline, not headline accuracy.")

results_path = os.path.join(D.DATA, "results.json")
if not os.path.exists(results_path):
    st.warning("No results yet - run `python pipeline.py` first.")
    st.stop()

with open(results_path) as fh:
    res = json.load(fh)
df = pd.read_csv(os.path.join(D.DATA, "dataset.csv"), parse_dates=["date"])

left, right = st.columns([2, 1])
with left:
    st.subheader("{} - last close {:.4f} ({})".format(
        res["pair"], res["last_close"], res["as_of"]))
    st.line_chart(df.set_index("date")["close"], height=280)
with right:
    st.subheader("News sentiment (daily)")
    senti = S.daily_sentiment(D.fetch_headlines())
    if len(senti):
        st.bar_chart(senti.set_index("date")["sentiment"], height=280)
    else:
        st.info("No cached headlines.")

st.subheader("Model comparison - walk-forward, last {} trading days"
             .format(res["n_test"]))
model_names = [k for k in res
               if isinstance(res[k], dict) and "accuracy" in res[k]]
table = pd.DataFrame([{
    "model": k,
    "directional accuracy %": round(100 * res[k]["accuracy"], 1),
    "strategy return %": res[k]["strategy_return_pct"],
    "buy & hold %": res["buy_hold_return_pct"],
    "prob. up tomorrow %": round(100 * res[k]["proba"][-1], 0),
} for k in model_names]).sort_values("strategy return %", ascending=False)
st.dataframe(table, use_container_width=True, hide_index=True)

report_path = os.path.join(D.DATA, "insight_report.md")
if os.path.exists(report_path):
    st.subheader("Latest insight report")
    with open(report_path, encoding="utf-8") as fh:
        st.markdown(fh.read())

with st.sidebar:
    st.header("Controls")
    st.write("Pair: **{}**".format(res["pair"]))
    if st.button("Refresh data + re-run backtest (no GRU)"):
        import pipeline
        with st.spinner("Fetching and backtesting..."):
            pipeline.run(pair=res["pair"], refresh=True, with_gru=False)
        st.rerun()
