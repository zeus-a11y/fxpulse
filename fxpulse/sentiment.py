"""News sentiment: VADER over headlines, aggregated to a daily score."""
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def score_text(text):
    """Compound VADER score in [-1, 1]."""
    return _analyzer.polarity_scores(str(text))["compound"]


def daily_sentiment(headlines):
    """DataFrame(date, title) -> DataFrame(date, sentiment, n_headlines),
    one row per calendar day with at least one headline."""
    if headlines is None or len(headlines) == 0:
        return pd.DataFrame(columns=["date", "sentiment", "n_headlines"])
    df = headlines.copy()
    df["score"] = df["title"].map(score_text)
    g = df.groupby(df["date"].dt.normalize())["score"]
    out = g.agg(["mean", "count"]).reset_index()
    out.columns = ["date", "sentiment", "n_headlines"]
    return out
