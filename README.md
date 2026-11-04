# FXPulse

Forex forecasting and sentiment analytics with honest, walk-forward
evaluation. Live FX prices (Yahoo chart API), FRED macro indicators
(public fredgraph.csv endpoint) and news sentiment (VADER over Google
News RSS) feed one leakage-safe feature pipeline; ARIMA, ARIMAX,
XGBoost and a GRU compete against a naive baseline in an expanding-
window backtest, and an automated insight report turns the results
into a stakeholder note. No API keys anywhere.

## Results (USDINR, last 250 trading days, walk-forward)

| Model | Directional accuracy | Strategy return | Buy & hold |
| --- | --- | --- | --- |
| GRU | 55.6% | **+11.39%** | +6.86% |
| XGBoost | 54.4% | +9.60% | +6.86% |
| Naive majority | 56.4% | +6.86% | +6.86% |
| ARIMA | 56.4% | +6.86% | +6.86% |
| ARIMAX | 55.6% | +6.03% | +6.86% |
| Momentum (1d) | 48.4% | +1.51% | +6.86% |

The honest read: daily FX direction is close to a coin flip, and USDINR
drifts upward, so the always-up baseline already "scores" 56%. Headline
accuracy is therefore meaningless on its own (a resume claiming ~90%
on daily FX direction almost certainly leaked the future). What the
models can earn is **strategy return above buy-and-hold** — the GRU and
XGBoost both do (+4.5pt and +2.7pt over the test window), and the
eval harness proves nothing peeked.

## Quickstart

```
pip install -r requirements.txt
python pipeline.py           # fetch (cached) -> features -> backtest
python report.py             # stakeholder insight report (markdown)
python eval.py               # 10 deterministic checks
streamlit run app.py         # dashboard (python -m streamlit run app.py)
```

`python pipeline.py --refresh` re-fetches live data; `--no-gru` skips
the neural model; `--pair EURINR=X` switches pairs. With
`ANTHROPIC_API_KEY` set, `python report.py --llm` adds a model-written
executive summary over the same figures.

## Leakage-safety (the whole point)

- Features for day t use only information available at day t's close;
  macro series and sentiment are additionally lagged one day
  (publication delay), and the target is strictly the next day's
  direction.
- The backtest is walk-forward with an expanding window; the fold
  bookkeeping is stored in the results and the eval asserts every fold
  trained strictly before it predicted.
- `python eval.py` (10/10): truncating the future never changes a past
  feature row; the target realigns against raw closes; macro features
  re-derive from the raw FRED series; accuracy and strategy return
  recompute from stored predictions; the backtester scores a known toy
  sequence perfectly; the sentiment scorer passes direction sanity.

## Files

| File | What it is |
| --- | --- |
| `fxpulse/data.py` | keyless loaders + cache: Yahoo FX, FRED CSV, Google News RSS |
| `fxpulse/sentiment.py` | VADER headline scoring, daily aggregation |
| `fxpulse/features.py` | leakage-safe features (lags, vol, RSI, macro, sentiment) |
| `fxpulse/models.py` | naive, momentum, ARIMA, ARIMAX, XGBoost, GRU behind one interface |
| `fxpulse/backtest.py` | expanding-window walk-forward with fold bookkeeping |
| `pipeline.py` | end-to-end run -> `data/results.json` |
| `report.py` | automated insight report (optional LLM summary) |
| `app.py` | Streamlit dashboard |
| `eval.py` | the 10-check harness |
