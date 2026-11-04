"""The model zoo. Every model exposes fit(train_df, feats) and
predict_proba_up(rows) -> prob the NEXT day closes higher, so the
backtester treats statistical, tree and neural models identically.
"""
import warnings

import numpy as np

warnings.filterwarnings("ignore")


class NaiveUp:
    """Majority-class baseline: always predicts the training majority.
    Any model that cannot beat this has learned nothing."""
    name = "naive_majority"

    def fit(self, train, feats):
        self.p = float(train["target"].mean())
        return self

    def predict_proba_up(self, rows):
        return np.full(len(rows), self.p)


class Momentum:
    """Random-walk-with-drift heuristic: tomorrow repeats today's sign."""
    name = "momentum_1d"

    def fit(self, train, feats):
        return self

    def predict_proba_up(self, rows):
        return (rows["ret_lag1"].values > 0).astype(float) * 0.5 + 0.25


class Arima:
    """ARIMA(1,0,1) on returns; prob(up) from the forecast's sign via a
    normal approximation."""
    name = "arima"

    def __init__(self, exog_cols=None):
        self.exog_cols = exog_cols
        if exog_cols:
            self.name = "arimax"

    def fit(self, train, feats):
        from statsmodels.tsa.arima.model import ARIMA
        self.endog = train["ret"].values
        exog = train[self.exog_cols].values if self.exog_cols else None
        self.res = ARIMA(self.endog, exog=exog,
                         order=(1, 0, 1)).fit(method_kwargs={"maxiter": 50})
        self.sigma = float(np.std(self.res.resid)) or 1e-6
        return self

    def predict_proba_up(self, rows):
        from scipy.stats import norm
        exog = rows[self.exog_cols].values if self.exog_cols else None
        try:
            mu = self.res.forecast(steps=len(rows), exog=exog)
        except Exception:
            mu = np.zeros(len(rows))
        return norm.cdf(np.asarray(mu) / self.sigma)


class Xgb:
    name = "xgboost"

    def fit(self, train, feats):
        from xgboost import XGBClassifier
        self.feats = feats
        self.clf = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8, eval_metric="logloss",
            random_state=7)
        self.clf.fit(train[feats].values, train["target"].astype(int).values)
        return self

    def predict_proba_up(self, rows):
        return self.clf.predict_proba(rows[self.feats].values)[:, 1]


class Gru:
    """Small GRU over the last 20 daily returns."""
    name = "gru"
    LOOKBACK = 20

    def fit(self, train, feats):
        import tensorflow as tf
        tf.random.set_seed(7)
        r = train["ret"].values.astype("float32")
        X, y = [], []
        tgt = train["target"].values
        for i in range(self.LOOKBACK, len(r)):
            X.append(r[i - self.LOOKBACK:i])
            y.append(tgt[i - 1])  # target aligned to the row ending at i-1
        X = np.array(X)[..., None]
        y = np.array(y, dtype="float32")
        self.scale = float(np.std(r)) or 1e-6
        X /= self.scale
        self.model = tf.keras.Sequential([
            tf.keras.layers.GRU(16, input_shape=(self.LOOKBACK, 1)),
            tf.keras.layers.Dense(1, activation="sigmoid")])
        self.model.compile(optimizer="adam", loss="binary_crossentropy")
        self.model.fit(X, y, epochs=8, batch_size=64, verbose=0)
        self._train_rets = r
        return self

    def predict_proba_up(self, rows):
        # rows carry ret_lag1..5 only; rebuild each window from the
        # concatenated return history ending at that row.
        r = np.concatenate([self._train_rets, rows["ret"].values]) \
            .astype("float32")
        out = []
        n_train = len(self._train_rets)
        for j in range(len(rows)):
            end = n_train + j + 1
            window = r[end - self.LOOKBACK:end] / self.scale
            out.append(window)
        import numpy as _np
        preds = self.model.predict(_np.array(out)[..., None], verbose=0)
        return preds.ravel()


def model_set(feats, macro_cols, with_gru=True):
    models = [NaiveUp(), Momentum(), Arima(),
              Arima(exog_cols=macro_cols) if macro_cols else None, Xgb()]
    if with_gru:
        models.append(Gru())
    return [m for m in models if m is not None]
