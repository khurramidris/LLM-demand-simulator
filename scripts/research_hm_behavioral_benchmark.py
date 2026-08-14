from __future__ import annotations

import argparse
import json
import math
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CUTOFF = pd.Timestamp("2019-09-20")
TEST_END = pd.Timestamp("2019-10-17")
VAL_START = pd.Timestamp("2019-08-23")
VAL_END = pd.Timestamp("2019-09-19")
RISK_WINDOWS = (7, 14, 28)
SMOOTHING_STRENGTH = 20.0
EPS = 1e-9


@dataclass
class MetricRow:
    risk_rule: str
    sample: str
    model: str
    n: int
    prevalence: float
    log_loss: float
    brier: float
    average_precision: float
    roc_auc: float
    count_nll: float
    count_mae: float
    count_rmse: float
    mean_pred_count: float


def _read_csv_from_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        members = [
            name for name in zf.namelist()
            if name.lower().endswith(".csv") and not name.startswith("__MACOSX/")
        ]
        if not members:
            raise RuntimeError(f"No CSV member found in {path}")
        preferred = [name for name in members if Path(name).name == "txns_trousers_online.csv"]
        member = preferred[0] if preferred else sorted(members)[0]
        with zf.open(member) as fh:
            return pd.read_csv(fh)


def _safe_float(x: float) -> float | None:
    return None if not np.isfinite(x) else float(x)


def _logseries_mean(p: float) -> float:
    return float(-p / ((1.0 - p) * math.log1p(-p)))


def _logseries_logpmf(y: np.ndarray, p: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return np.log(-1.0 / math.log1p(-p)) + y * math.log(p) - np.log(y)


def _fit_logseries(y_positive: np.ndarray) -> float:
    y = np.asarray(y_positive, dtype=float)
    if len(y) == 0 or np.any(y < 1):
        raise ValueError("log-series fit requires positive integer counts")

    def objective(p: float) -> float:
        return -float(np.sum(_logseries_logpmf(y, p)))

    result = minimize_scalar(objective, bounds=(1e-6, 1.0 - 1e-8), method="bounded")
    if not result.success:
        raise RuntimeError(f"log-series fit failed: {result.message}")
    return float(result.x)


def _hurdle_metrics(y: np.ndarray, q: np.ndarray, logseries_p: float) -> tuple[float, float, float, float]:
    y = np.asarray(y, dtype=int)
    q = np.clip(np.asarray(q, dtype=float), EPS, 1.0 - EPS)
    log_prob = np.empty(len(y), dtype=float)
    zero = y == 0
    log_prob[zero] = np.log1p(-q[zero])
    positive = ~zero
    if positive.any():
        log_prob[positive] = np.log(q[positive]) + _logseries_logpmf(y[positive], logseries_p)
    pred_mean = q * _logseries_mean(logseries_p)
    return (
        float(-np.mean(log_prob)),
        float(np.mean(np.abs(y - pred_mean))),
        float(np.sqrt(np.mean((y - pred_mean) ** 2))),
        float(np.mean(pred_mean)),
    )


def _future_purchase_flags(daily: pd.DataFrame, horizons=(1, 7, 14, 28)) -> pd.DataFrame:
    out = daily[["article_id", "date", "count"]].copy()
    for h in horizons:
        vals = np.zeros(len(out), dtype=int)
        for _, idx in out.groupby("article_id", sort=False).groups.items():
            idx = np.asarray(list(idx), dtype=int)
            arr = out.loc[idx, "count"].to_numpy(dtype=int)
            # future sum over t+1 ... t+h
            prefix = np.concatenate([[0], np.cumsum(arr)])
            future = np.zeros(len(arr), dtype=int)
            for j in range(len(arr)):
                lo = j + 1
                hi = min(len(arr), j + h + 1)
                future[j] = int(prefix[hi] - prefix[lo]) if lo < len(arr) else 0
            vals[idx] = future
        out[f"future_{h}d"] = vals
    return out


def _build_daily_grid(tx: pd.DataFrame, universe: np.ndarray) -> pd.DataFrame:
    subset = tx[tx["article_id"].isin(universe)].copy()
    counts = subset.groupby(["article_id", "date"], as_index=False).agg(
        count=("customer_id", "size"),
        n_paid_prices=("price", "nunique"),
        mean_paid_price=("price", "mean"),
    )
    start = subset["date"].min()
    end = max(TEST_END, subset["date"].max())
    idx = pd.MultiIndex.from_product(
        [np.sort(universe), pd.date_range(start, end, freq="D")],
        names=["article_id", "date"],
    )
    daily = counts.set_index(["article_id", "date"]).reindex(idx).reset_index()
    daily["count"] = daily["count"].fillna(0).astype(int)
    daily["n_paid_prices"] = daily["n_paid_prices"].fillna(0).astype(int)
    # Price remains NaN on zero-purchase days by design.
    daily["incidence"] = (daily["count"] > 0).astype(int)

    g = daily.groupby("article_id", group_keys=False, sort=False)
    for k in RISK_WINDOWS:
        daily[f"lag_count_{k}"] = g["count"].transform(lambda s: s.shift(1).rolling(k, min_periods=1).sum()).fillna(0)
        daily[f"at_risk_R{k}"] = daily[f"lag_count_{k}"] > 0
    daily["lag_count_1"] = g["count"].shift(1).fillna(0)

    def days_since_sale(group: pd.DataFrame) -> pd.Series:
        last = None
        values = []
        for date, count in zip(group["date"], group["count"]):
            values.append(999.0 if last is None else float((date - last).days))
            if count > 0:
                last = date
        return pd.Series(values, index=group.index, dtype=float)

    daily["days_since_sale"] = g.apply(days_since_sale, include_groups=False).sort_index()
    first_last = subset.groupby("article_id")["date"].agg(first_sale="min", last_sale="max")
    daily = daily.merge(first_last, on="article_id", how="left")
    daily["at_risk_Span"] = (daily["date"] >= daily["first_sale"]) & (daily["date"] <= daily["last_sale"])
    daily["dow"] = daily["date"].dt.dayofweek.astype(int)
    daily["time_index"] = (daily["date"] - daily["date"].min()).dt.days.astype(float)
    return daily


def _risk_mask(df: pd.DataFrame, rule: str) -> pd.Series:
    return df[f"at_risk_{rule}"].astype(bool)


def _sample_mask(df: pd.DataFrame, sample: str) -> pd.Series:
    if sample == "train":
        return df["date"] < CUTOFF
    if sample == "validation":
        return (df["date"] >= VAL_START) & (df["date"] <= VAL_END)
    if sample == "test":
        return (df["date"] >= CUTOFF) & (df["date"] <= TEST_END)
    raise ValueError(sample)


def _risk_summary(panel: pd.DataFrame, rule: str, sample: str, future_flags: pd.DataFrame) -> dict:
    mask = _risk_mask(panel, rule) & _sample_mask(panel, sample)
    df = panel.loc[mask].copy()
    merged = df.merge(future_flags, on=["article_id", "date", "count"], how="left")
    zeros = merged[merged["count"] == 0]
    out = {
        "risk_rule": rule,
        "sample": sample,
        "article_days": int(len(df)),
        "articles": int(df["article_id"].nunique()),
        "positive_days": int((df["count"] > 0).sum()),
        "zero_days": int((df["count"] == 0).sum()),
        "prevalence": float(df["incidence"].mean()) if len(df) else None,
        "mean_count": float(df["count"].mean()) if len(df) else None,
        "median_article_prevalence": float(df.groupby("article_id")["incidence"].mean().median()) if len(df) else None,
        "articles_with_ge7_days": int((df.groupby("article_id").size() >= 7).sum()) if len(df) else 0,
    }
    for h in (1, 7, 14, 28):
        out[f"zero_followed_by_purchase_{h}d"] = (
            float((zeros[f"future_{h}d"] > 0).mean()) if len(zeros) else None
        )
    return out


def _jaccard(panel: pd.DataFrame, a: str, b: str, sample: str) -> float:
    base = _sample_mask(panel, sample)
    aa = set(map(tuple, panel.loc[base & _risk_mask(panel, a), ["article_id", "date"]].to_numpy()))
    bb = set(map(tuple, panel.loc[base & _risk_mask(panel, b), ["article_id", "date"]].to_numpy()))
    union = aa | bb
    return float(len(aa & bb) / len(union)) if union else float("nan")


def _fit_incidence_models(panel: pd.DataFrame, rule: str) -> tuple[dict[str, object], pd.DataFrame]:
    train = panel[_risk_mask(panel, rule) & (panel["date"] < CUTOFF)].copy()
    test = panel[_risk_mask(panel, rule) & _sample_mask(panel, "test")].copy()
    if len(train) == 0 or len(test) == 0:
        raise RuntimeError(f"Empty train/test under {rule}")

    global_p = float(np.clip(train["incidence"].mean(), EPS, 1 - EPS))
    predictions = test[["article_id", "date", "count", "incidence"]].copy()
    predictions["I0_global"] = global_p

    # I1: calendar-only. One-hot day-of-week plus standardized historical time trend.
    cal = Pipeline([
        ("features", ColumnTransformer([
            ("dow", OneHotEncoder(handle_unknown="ignore"), ["dow"]),
            ("num", StandardScaler(), ["time_index"]),
        ])),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")),
    ])
    cal.fit(train[["dow", "time_index"]], train["incidence"])
    predictions["I1_calendar"] = cal.predict_proba(test[["dow", "time_index"]])[:, 1]

    # I2: pre-cutoff article historical rate with fixed beta shrinkage to global prevalence.
    stats = train.groupby("article_id")["incidence"].agg(["sum", "count"])
    stats["p"] = (stats["sum"] + SMOOTHING_STRENGTH * global_p) / (stats["count"] + SMOOTHING_STRENGTH)
    predictions["I2_article_rate"] = test["article_id"].map(stats["p"]).fillna(global_p).to_numpy()

    # I3: lagged history/recency + article historical rate + calendar features.
    feature_cols = ["lag_count_1", "lag_count_7", "lag_count_14", "lag_count_28", "days_since_sale", "time_index", "dow"]
    train = train.copy()
    test = test.copy()
    train["article_prior"] = train["article_id"].map(stats["p"]).fillna(global_p)
    test["article_prior"] = test["article_id"].map(stats["p"]).fillna(global_p)
    feature_cols2 = ["lag_count_1", "lag_count_7", "lag_count_14", "lag_count_28", "days_since_sale", "time_index", "article_prior", "dow"]
    hist = Pipeline([
        ("features", ColumnTransformer([
            ("dow", OneHotEncoder(handle_unknown="ignore"), ["dow"]),
            ("num", StandardScaler(), [c for c in feature_cols2 if c != "dow"]),
        ])),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")),
    ])
    hist.fit(train[feature_cols2], train["incidence"])
    predictions["I3_history_recency"] = hist.predict_proba(test[feature_cols2])[:, 1]

    models = {
        "global_p": global_p,
        "calendar": cal,
        "article_stats": stats,
        "history": hist,
        "train": train,
        "test": test,
    }
    return models, predictions


def _metric_row(rule: str, sample: str, model: str, y: np.ndarray, q: np.ndarray, count: np.ndarray, logseries_p: float) -> MetricRow:
    q = np.clip(np.asarray(q, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=int)
    count = np.asarray(count, dtype=int)
    ap = float(average_precision_score(y, q)) if len(np.unique(y)) > 1 else float("nan")
    auc = float(roc_auc_score(y, q)) if len(np.unique(y)) > 1 else float("nan")
    nll, mae, rmse, mean_pred = _hurdle_metrics(count, q, logseries_p)
    return MetricRow(
        risk_rule=rule,
        sample=sample,
        model=model,
        n=len(y),
        prevalence=float(np.mean(y)),
        log_loss=float(log_loss(y, q, labels=[0, 1])),
        brier=float(np.mean((y - q) ** 2)),
        average_precision=ap,
        roc_auc=auc,
        count_nll=nll,
        count_mae=mae,
        count_rmse=rmse,
        mean_pred_count=mean_pred,
    )


def _write_markdown(summary: dict, out: Path) -> None:
    lines = [
        "# Experiment 005 — H&M Behavioral Benchmark — Stage A/B Results",
        "",
        "Status: **AUDIT + NO-LLM BASELINES COMPLETE**",
        "",
        "## Source audit",
        "",
        f"- Purchase rows: **{summary['source']['rows']:,}**",
        f"- Unique customers: **{summary['source']['customers']:,}**",
        f"- Unique trouser articles: **{summary['source']['articles']:,}**",
        f"- Date span: **{summary['source']['min_date']} → {summary['source']['max_date']}**",
        f"- Train-only top-100 universe selected strictly before: **{summary['cutoff']}**",
        f"- Cold articles first appearing during primary 28-day test: **{summary['cold_audit']['cold_articles_in_test']:,}**",
        "",
        "## Availability conclusion",
        "",
        "The source remains a purchase-event log, not an inventory/exposure log. All zero rows below are **proxy at-risk zeros**, never observed availability-conditioned negatives.",
        "",
        "## Risk-set diagnostics",
        "",
        "| Rule | Sample | Article-days | Articles | Positive prevalence | ≥7-day articles | zero→purchase within 14d |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["risk_summaries"]:
        if row["sample"] not in {"validation", "test"}:
            continue
        z14 = row.get("zero_followed_by_purchase_14d")
        lines.append(
            f"| {row['risk_rule']} | {row['sample']} | {row['article_days']} | {row['articles']} | "
            f"{row['prevalence']:.4f} | {row['articles_with_ge7_days']} | {z14:.4f} |"
        )
    lines += [
        "",
        "## Future-test no-LLM baseline ladder",
        "",
        "| Risk | Model | Incidence log loss | Brier | AP | ROC-AUC | Hurdle count NLL | MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["metrics"]:
        lines.append(
            f"| {row['risk_rule']} | {row['model']} | {row['log_loss']:.6f} | {row['brier']:.6f} | "
            f"{row['average_precision']:.6f} | {row['roc_auc']:.6f} | {row['count_nll']:.6f} | {row['count_mae']:.6f} |"
        )
    gate = summary["decision"]
    lines += [
        "",
        "## Preregistered gate",
        "",
        f"- R14 test article-days ≥ 2,000: **{gate['r14_min_rows_pass']}**",
        f"- R14 ≥7-day article coverage ≥ 50: **{gate['r14_article_coverage_pass']}**",
        f"- Baseline ranking stable across R7/R14/R28: **{gate['ranking_stable']}**",
        f"- H&M killed as flagship behavioral benchmark: **{gate['kill_hm_flagship']}**",
        f"- Proceed to semantic/customer treatment stage: **{gate['proceed_to_stage_c']}**",
        "",
        "## Interpretation",
        "",
        summary["decision"]["interpretation"],
        "",
        "No LLM/API calls were made in Stage A/B.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transactions", type=Path, default=Path("outputs/products/txns_trousers_online.csv.zip"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tx = _read_csv_from_zip(args.transactions)
    required = {"t_dat", "customer_id", "article_id", "price", "sales_channel_id"}
    missing = required - set(tx.columns)
    if missing:
        raise RuntimeError(f"Missing expected columns: {sorted(missing)}")
    tx = tx[list(required)].copy()
    tx["date"] = pd.to_datetime(tx.pop("t_dat"))
    tx["article_id"] = pd.to_numeric(tx["article_id"], errors="raise").astype(np.int64)
    tx["price"] = pd.to_numeric(tx["price"], errors="coerce")
    if not (tx["sales_channel_id"] == 1).all():
        raise AssertionError("Committed source is expected to contain online channel only")

    source_summary = {
        "rows": int(len(tx)),
        "customers": int(tx["customer_id"].nunique()),
        "articles": int(tx["article_id"].nunique()),
        "min_date": tx["date"].min().date().isoformat(),
        "max_date": tx["date"].max().date().isoformat(),
        "columns": sorted(list(required)),
        "has_inventory_column": any("inventory" in c.lower() or "stock" in c.lower() or "avail" in c.lower() for c in tx.columns),
    }

    pre = tx[tx["date"] < CUTOFF].copy()
    distinct_prices = pre.groupby("article_id")["price"].nunique()
    eligible = distinct_prices[distinct_prices >= 5].index
    purchase_counts = pre[pre["article_id"].isin(eligible)].groupby("article_id").size().sort_values(ascending=False)
    universe = purchase_counts.head(100).index.to_numpy(dtype=np.int64)
    if len(universe) < 100:
        raise RuntimeError(f"Expected 100 eligible train-only articles, got {len(universe)}")

    first_sale_all = tx.groupby("article_id")["date"].min()
    cold = first_sale_all[(first_sale_all >= CUTOFF) & (first_sale_all <= TEST_END)]
    cold_ids = cold.index
    cold_test_tx = tx[(tx["article_id"].isin(cold_ids)) & (tx["date"] >= CUTOFF) & (tx["date"] <= TEST_END)]
    cold_audit = {
        "cold_articles_in_test": int(len(cold)),
        "cold_purchase_rows_in_test": int(len(cold_test_tx)),
        "cold_customers_in_test": int(cold_test_tx["customer_id"].nunique()),
    }

    panel = _build_daily_grid(tx, universe)
    # Restrict future-flag computation to relevant daily grid but preserve dates after test for diagnostic follow-up.
    future_flags = _future_purchase_flags(panel)

    risk_summaries = []
    rules = ["R7", "R14", "R28", "Span"]
    for rule in rules:
        for sample in ("validation", "test"):
            risk_summaries.append(_risk_summary(panel, rule, sample, future_flags))

    jaccards = []
    for sample in ("validation", "test"):
        for a, b in (("R7", "R14"), ("R14", "R28"), ("R7", "R28")):
            jaccards.append({"sample": sample, "a": a, "b": b, "jaccard": _jaccard(panel, a, b, sample)})

    # Price ambiguity diagnostics.
    positive_daily = panel[panel["count"] > 0]
    price_diag = {
        "positive_article_days": int(len(positive_daily)),
        "positive_days_multiple_paid_prices": int((positive_daily["n_paid_prices"] > 1).sum()),
        "fraction_positive_days_multiple_paid_prices": float((positive_daily["n_paid_prices"] > 1).mean()),
        "zero_days_with_observed_paid_price": int(panel.loc[panel["count"] == 0, "mean_paid_price"].notna().sum()),
    }

    metrics: list[MetricRow] = []
    prediction_files = []
    model_rankings = {}
    # Positive count law fit strictly from pre-cutoff purchase days of train-selected universe.
    positive_train_counts = panel[(panel["date"] < CUTOFF) & (panel["count"] > 0)]["count"].to_numpy(dtype=int)
    logseries_p = _fit_logseries(positive_train_counts)

    for rule in ("R7", "R14", "R28"):
        models, pred = _fit_incidence_models(panel, rule)
        y = pred["incidence"].to_numpy(dtype=int)
        count = pred["count"].to_numpy(dtype=int)
        model_cols = ["I0_global", "I1_calendar", "I2_article_rate", "I3_history_recency"]
        for model in model_cols:
            row = _metric_row(rule, "test", model, y, pred[model].to_numpy(), count, logseries_p)
            metrics.append(row)
        pred_path = args.output_dir / f"predictions_{rule}.csv"
        pred.to_csv(pred_path, index=False)
        prediction_files.append(str(pred_path.name))
        model_rankings[rule] = [r.model for r in sorted([x for x in metrics if x.risk_rule == rule], key=lambda z: z.log_loss)]

    metric_dicts = [asdict(row) for row in metrics]
    r14_summary = next(x for x in risk_summaries if x["risk_rule"] == "R14" and x["sample"] == "test")
    ranking_stable = model_rankings["R7"] == model_rankings["R14"] == model_rankings["R28"]
    size_pass = r14_summary["article_days"] >= 2000
    coverage_pass = r14_summary["articles_with_ge7_days"] >= 50

    # Proxy-dependence diagnostic: winner identity and prevalence spread across future-safe rules.
    winners = {rule: model_rankings[rule][0] for rule in ("R7", "R14", "R28")}
    prevalences = [
        next(x for x in risk_summaries if x["risk_rule"] == rule and x["sample"] == "test")["prevalence"]
        for rule in ("R7", "R14", "R28")
    ]
    prevalence_spread = float(max(prevalences) - min(prevalences))
    kill = (not size_pass) or (not coverage_pass) or (not ranking_stable)

    # Stage C needs stable proxy behavior plus meaningful headroom over global.
    r14_metrics = {x.model: x for x in metrics if x.risk_rule == "R14"}
    improvement = r14_metrics["I0_global"].log_loss - min(x.log_loss for x in r14_metrics.values())
    proceed = (not kill) and improvement > 0.005

    interpretation = (
        "The future-safe risk-set rules are sufficiently stable for a follow-on treatment experiment. "
        "This remains proxy-availability evidence, not exposure-ground-truth evidence."
        if proceed else
        "Do not spend LLM calls on this benchmark yet. Either the proxy panel fails a preregistered stability/size rule or the no-LLM ladder leaves insufficiently clear behavioral headroom. H&M should remain an engineering/reproduction dataset rather than Allegory's flagship scientific validation until a stronger exposure/choice dataset is used."
    )

    decision = {
        "r14_min_rows_pass": bool(size_pass),
        "r14_article_coverage_pass": bool(coverage_pass),
        "ranking_stable": bool(ranking_stable),
        "winners": winners,
        "test_prevalence_spread_R7_R14_R28": prevalence_spread,
        "best_R14_logloss_improvement_over_global": float(improvement),
        "kill_hm_flagship": bool(kill),
        "proceed_to_stage_c": bool(proceed),
        "interpretation": interpretation,
    }

    summary = {
        "experiment": "005_behavioral_benchmark",
        "cutoff": CUTOFF.date().isoformat(),
        "test_window": [CUTOFF.date().isoformat(), TEST_END.date().isoformat()],
        "validation_window": [VAL_START.date().isoformat(), VAL_END.date().isoformat()],
        "source": source_summary,
        "train_universe": {
            "n_articles": int(len(universe)),
            "min_distinct_paid_prices": 5,
            "selection_uses_post_cutoff": False,
            "article_ids": [int(x) for x in universe],
        },
        "cold_audit": cold_audit,
        "price_diagnostics": price_diag,
        "risk_summaries": risk_summaries,
        "risk_jaccards": jaccards,
        "positive_count_logseries_p": float(logseries_p),
        "positive_count_logseries_mean": _logseries_mean(logseries_p),
        "metrics": metric_dicts,
        "model_rankings": model_rankings,
        "decision": decision,
        "prediction_files": prediction_files,
        "new_llm_calls": 0,
    }

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(risk_summaries).to_csv(args.output_dir / "risk_summaries.csv", index=False)
    pd.DataFrame(jaccards).to_csv(args.output_dir / "risk_jaccards.csv", index=False)
    pd.DataFrame(metric_dicts).to_csv(args.output_dir / "baseline_metrics.csv", index=False)
    pd.DataFrame({"article_id": universe}).to_csv(args.output_dir / "train_only_top100.csv", index=False)
    _write_markdown(summary, args.output_dir / "results.md")
    print(json.dumps({
        "source": source_summary,
        "cold_audit": cold_audit,
        "price_diagnostics": price_diag,
        "decision": decision,
        "R14_metrics": [x for x in metric_dicts if x["risk_rule"] == "R14"],
    }, indent=2))


if __name__ == "__main__":
    main()
