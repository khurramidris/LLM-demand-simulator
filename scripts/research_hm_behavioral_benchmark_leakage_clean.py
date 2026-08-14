from __future__ import annotations

"""Leakage-clean confirmation wrapper for Experiment 005.

The first Stage A/B implementation included the full pre-cutoff article-rate estimate as an
I3 training feature. That is a legitimate fitted train statistic for future prediction, but
inside the training sample it behaves like target encoding unless cross-fitted. To make the
strong I3 result maximally conservative, this wrapper removes that feature entirely while
leaving the preregistered data, risk sets, dates, metrics, and decision logic unchanged.

I2 remains the explicitly defined article-rate model. I3 becomes pure lagged history/recency
plus calendar trend, using only quantities available strictly before each predicted day.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import scripts.research_hm_behavioral_benchmark as base


def leakage_clean_fit_incidence_models(panel, rule):
    train = panel[base._risk_mask(panel, rule) & (panel["date"] < base.CUTOFF)].copy()
    test = panel[base._risk_mask(panel, rule) & base._sample_mask(panel, "test")].copy()
    if len(train) == 0 or len(test) == 0:
        raise RuntimeError(f"Empty train/test under {rule}")

    global_p = float(np.clip(train["incidence"].mean(), base.EPS, 1 - base.EPS))
    predictions = test[["article_id", "date", "count", "incidence"]].copy()
    predictions["I0_global"] = global_p

    cal = Pipeline([
        ("features", ColumnTransformer([
            ("dow", OneHotEncoder(handle_unknown="ignore"), ["dow"]),
            ("num", StandardScaler(), ["time_index"]),
        ])),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")),
    ])
    cal.fit(train[["dow", "time_index"]], train["incidence"])
    predictions["I1_calendar"] = cal.predict_proba(test[["dow", "time_index"]])[:, 1]

    stats = train.groupby("article_id")["incidence"].agg(["sum", "count"])
    stats["p"] = (
        stats["sum"] + base.SMOOTHING_STRENGTH * global_p
    ) / (stats["count"] + base.SMOOTHING_STRENGTH)
    predictions["I2_article_rate"] = test["article_id"].map(stats["p"]).fillna(global_p).to_numpy()

    # No target-encoded article prior. Every feature is observable strictly before the day.
    feature_cols = [
        "lag_count_1",
        "lag_count_7",
        "lag_count_14",
        "lag_count_28",
        "days_since_sale",
        "time_index",
        "dow",
    ]
    hist = Pipeline([
        ("features", ColumnTransformer([
            ("dow", OneHotEncoder(handle_unknown="ignore"), ["dow"]),
            ("num", StandardScaler(), [c for c in feature_cols if c != "dow"]),
        ])),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")),
    ])
    hist.fit(train[feature_cols], train["incidence"])
    predictions["I3_history_recency"] = hist.predict_proba(test[feature_cols])[:, 1]

    models = {
        "global_p": global_p,
        "calendar": cal,
        "article_stats": stats,
        "history": hist,
        "train": train,
        "test": test,
    }
    return models, predictions


base._fit_incidence_models = leakage_clean_fit_incidence_models

if __name__ == "__main__":
    base.main()
