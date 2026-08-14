from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


HISTORY_START = pd.Timestamp("2018-09-01")
HISTORY_END = pd.Timestamp("2019-09-19")
TARGET_START = pd.Timestamp("2019-09-20")
TARGET_END = pd.Timestamp("2019-12-31")
SEED = 2025


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 0: test whether richer behavioral customer state contains more "
            "out-of-sample H&M purchase signal than the paper's coarse persona state."
        )
    )
    parser.add_argument(
        "--customer-features",
        type=Path,
        default=Path("outputs/personas/customer_features.csv.zip"),
    )
    parser.add_argument(
        "--trouser-transactions",
        type=Path,
        default=Path("outputs/products/txns_trousers_online.csv.zip"),
    )
    parser.add_argument(
        "--product-info",
        type=Path,
        default=Path("outputs/products/product_info_top100_online.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/research/rich_persona_signal"))
    parser.add_argument("--history-start", default=str(HISTORY_START.date()))
    parser.add_argument("--history-end", default=str(HISTORY_END.date()))
    parser.add_argument("--target-start", default=str(TARGET_START.date()))
    parser.add_argument("--target-end", default=str(TARGET_END.date()))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--bootstrap-reps", type=int, default=100)
    parser.add_argument("--bootstrap-n", type=int, default=50_000)
    return parser.parse_args()


def _qcut(values: pd.Series, labels: list[str]) -> pd.Series:
    clean = values.copy()
    try:
        out = pd.qcut(clean, q=len(labels), labels=labels, duplicates="drop")
        if out.nunique(dropna=True) >= 2:
            return out.astype("object")
    except ValueError:
        pass
    ranked = clean.rank(method="average")
    n_bins = min(len(labels), max(int(ranked.nunique()), 1))
    if n_bins <= 1:
        return pd.Series([labels[0]] * len(values), index=values.index, dtype="object")
    return pd.qcut(ranked, q=n_bins, labels=labels[:n_bins], duplicates="drop").astype("object")


def build_paper_state(customer_features: pd.DataFrame, history_start: pd.Timestamp, history_end: pd.Timestamp) -> pd.DataFrame:
    df = customer_features.copy()
    df["customer_id"] = df["customer_id"].astype(str)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["txn_count"] = pd.to_numeric(df["txn_count"], errors="coerce").fillna(0.0)
    df["mean_price"] = pd.to_numeric(df["mean_price"], errors="coerce").fillna(0.0)
    df["top_product_type"] = df["top_product_type"].fillna("UNKNOWN").astype(str)

    span_months = max(((history_end - history_start).days + 1) / 30.4375, 1e-6)
    df["txn_per_month"] = df["txn_count"] / span_months
    df["age_bin"] = pd.cut(
        df["age"],
        bins=[16, 25, 35, 45, 55, 200],
        right=False,
        labels=["16-24", "25-34", "35-44", "45-54", "55+"],
    ).astype("object")

    active = df["txn_count"] > 0
    df["engagement_bin"] = "NO_PURCHASE"
    df["price_tier"] = "NO_PURCHASE"
    if active.any():
        df.loc[active, "engagement_bin"] = _qcut(
            df.loc[active, "txn_per_month"], ["low", "mid", "high"]
        ).to_numpy()
        df.loc[active, "price_tier"] = _qcut(
            df.loc[active, "mean_price"], ["low", "mid", "high"]
        ).to_numpy()

    df["age_bin"] = df["age_bin"].fillna("UNKNOWN")
    return df


def load_trouser_transactions(path: Path) -> pd.DataFrame:
    tx = pd.read_csv(path, compression="zip", usecols=["t_dat", "customer_id", "article_id", "price"])
    tx["t_dat"] = pd.to_datetime(tx["t_dat"], errors="coerce")
    tx["customer_id"] = tx["customer_id"].astype(str)
    tx["article_id"] = pd.to_numeric(tx["article_id"], errors="coerce")
    tx["price"] = pd.to_numeric(tx["price"], errors="coerce") * 590.0
    tx = tx.dropna(subset=["t_dat", "customer_id", "article_id", "price"])
    tx["article_id"] = tx["article_id"].astype(int)
    return tx


def build_rich_features(
    tx: pd.DataFrame,
    history_start: pd.Timestamp,
    history_end: pd.Timestamp,
) -> pd.DataFrame:
    hist = tx[(tx["t_dat"] >= history_start) & (tx["t_dat"] <= history_end)].copy()
    if hist.empty:
        raise RuntimeError("No trouser transactions found in the history window.")

    hist = hist.sort_values(["customer_id", "t_dat"])
    grouped = hist.groupby("customer_id", sort=False)
    rich = grouped.agg(
        tr_txn_count=("article_id", "size"),
        tr_unique_articles=("article_id", "nunique"),
        tr_mean_price=("price", "mean"),
        tr_price_std=("price", "std"),
        tr_price_min=("price", "min"),
        tr_price_max=("price", "max"),
        tr_first_date=("t_dat", "min"),
        tr_last_date=("t_dat", "max"),
        tr_active_days=("t_dat", "nunique"),
    ).reset_index()

    price_q = grouped["price"].quantile([0.25, 0.75]).unstack(fill_value=np.nan).reset_index()
    price_q = price_q.rename(columns={0.25: "tr_price_q25", 0.75: "tr_price_q75"})
    rich = rich.merge(price_q, on="customer_id", how="left")

    rich["tr_purchase_span_days"] = (rich["tr_last_date"] - rich["tr_first_date"]).dt.days.astype(float)
    rich["tr_days_since_last"] = (history_end - rich["tr_last_date"]).dt.days.astype(float)
    rich["tr_repeat_ratio"] = 1.0 - (
        rich["tr_unique_articles"].astype(float) / rich["tr_txn_count"].clip(lower=1).astype(float)
    )
    rich["tr_price_iqr"] = rich["tr_price_q75"] - rich["tr_price_q25"]

    for days in [30, 90, 180]:
        start = history_end - pd.Timedelta(days=days - 1)
        counts = (
            hist.loc[hist["t_dat"] >= start]
            .groupby("customer_id")
            .size()
            .rename(f"tr_txn_last_{days}d")
        )
        rich = rich.merge(counts, on="customer_id", how="left")

    months = ((history_end - history_start).days + 1) / 30.4375
    rich["tr_txn_per_month"] = rich["tr_txn_count"] / max(months, 1e-6)
    rich["tr_recent90_share"] = rich["tr_txn_last_90d"].fillna(0.0) / rich["tr_txn_count"].clip(lower=1)

    drop_cols = ["tr_first_date", "tr_last_date"]
    rich = rich.drop(columns=drop_cols)
    return rich


def build_label(
    tx: pd.DataFrame,
    top_product_ids: set[int],
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
) -> pd.Series:
    target = tx[
        (tx["t_dat"] >= target_start)
        & (tx["t_dat"] <= target_end)
        & (tx["article_id"].isin(top_product_ids))
    ]
    positive_ids = pd.Index(target["customer_id"].unique())
    return pd.Series(1, index=positive_ids, name="label", dtype=int)


def make_pipeline(categorical: list[str], numeric: list[str], seed: int) -> Pipeline:
    transformers = []
    if categorical:
        cat_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical))
    if numeric:
        num_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipe, numeric))

    prep = ColumnTransformer(transformers, sparse_threshold=0.3)
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        max_iter=1500,
        tol=1e-4,
        random_state=seed,
        average=True,
    )
    return Pipeline([("prep", prep), ("clf", clf)])


def evaluate(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1.0 - 1e-8)
    prevalence = float(np.mean(y_true))
    n_top = max(int(np.ceil(0.01 * len(p))), 1)
    top_idx = np.argpartition(p, -n_top)[-n_top:]
    precision_top1 = float(np.mean(y_true[top_idx]))
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "average_precision": float(average_precision_score(y_true, p)),
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, p)),
        "prevalence": prevalence,
        "precision_top_1pct": precision_top1,
        "lift_top_1pct": precision_top1 / max(prevalence, 1e-12),
    }


def paired_bootstrap(
    y: np.ndarray,
    p_control: np.ndarray,
    p_treatment: np.ndarray,
    reps: int,
    sample_n: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(y)
    sample_n = min(sample_n, n)
    diffs_ap: list[float] = []
    diffs_auc: list[float] = []
    attempts = 0
    while len(diffs_ap) < reps and attempts < reps * 3:
        attempts += 1
        idx = rng.integers(0, n, size=sample_n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        diffs_ap.append(
            float(average_precision_score(yy, p_treatment[idx]) - average_precision_score(yy, p_control[idx]))
        )
        diffs_auc.append(float(roc_auc_score(yy, p_treatment[idx]) - roc_auc_score(yy, p_control[idx])))

    def summarize(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=float)
        return {
            "mean_delta": float(arr.mean()),
            "ci95_low": float(np.quantile(arr, 0.025)),
            "ci95_high": float(np.quantile(arr, 0.975)),
            "p_delta_gt_0": float(np.mean(arr > 0.0)),
        }

    return {"average_precision": summarize(diffs_ap), "roc_auc": summarize(diffs_auc)}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    history_start = pd.Timestamp(args.history_start)
    history_end = pd.Timestamp(args.history_end)
    target_start = pd.Timestamp(args.target_start)
    target_end = pd.Timestamp(args.target_end)
    if target_start <= history_end:
        raise ValueError("Target window must begin strictly after the history window.")

    customer_features = pd.read_csv(args.customer_features, compression="zip")
    paper = build_paper_state(customer_features, history_start, history_end)
    tx = load_trouser_transactions(args.trouser_transactions)
    top_product_ids = set(pd.read_csv(args.product_info)["article_id"].astype(int).tolist())
    rich = build_rich_features(tx, history_start, history_end)
    positive_label = build_label(tx, top_product_ids, target_start, target_end)

    data = paper.merge(rich, on="customer_id", how="left")
    data["label"] = data["customer_id"].map(positive_label).fillna(0).astype(int)

    # The original paper only constructs personas for customers with observed history.
    # Restricting to those customers avoids turning NO_PURCHASE into a giant trivial group.
    data = data[data["txn_count"] > 0].reset_index(drop=True)
    if data["label"].nunique() < 2:
        raise RuntimeError("Target label has fewer than two classes.")

    idx_train, idx_test = train_test_split(
        np.arange(len(data)),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=data["label"].to_numpy(),
    )
    train = data.iloc[idx_train]
    test = data.iloc[idx_test]
    y_train = train["label"].to_numpy(int)
    y_test = test["label"].to_numpy(int)

    arms = {
        "paper_bins": {
            "categorical": ["age_bin", "engagement_bin", "price_tier", "top_product_type"],
            "numeric": [],
        },
        "paper_raw": {
            "categorical": ["top_product_type"],
            "numeric": ["age", "txn_count", "mean_price", "txn_per_month"],
        },
        "rich_behavior": {
            "categorical": ["top_product_type"],
            "numeric": [
                "age",
                "txn_count",
                "mean_price",
                "txn_per_month",
                "tr_txn_count",
                "tr_unique_articles",
                "tr_mean_price",
                "tr_price_std",
                "tr_price_min",
                "tr_price_max",
                "tr_active_days",
                "tr_price_q25",
                "tr_price_q75",
                "tr_purchase_span_days",
                "tr_days_since_last",
                "tr_repeat_ratio",
                "tr_price_iqr",
                "tr_txn_last_30d",
                "tr_txn_last_90d",
                "tr_txn_last_180d",
                "tr_txn_per_month",
                "tr_recent90_share",
            ],
        },
    }

    metrics: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, spec in arms.items():
        model = make_pipeline(spec["categorical"], spec["numeric"], args.seed)
        cols = spec["categorical"] + spec["numeric"]
        model.fit(train[cols], y_train)
        p = model.predict_proba(test[cols])[:, 1]
        predictions[name] = p
        metrics[name] = evaluate(y_test, p)

    bootstrap = paired_bootstrap(
        y_test,
        predictions["paper_raw"],
        predictions["rich_behavior"],
        reps=args.bootstrap_reps,
        sample_n=args.bootstrap_n,
        seed=args.seed + 1,
    )

    ap_control = metrics["paper_raw"]["average_precision"]
    ap_rich = metrics["rich_behavior"]["average_precision"]
    rel_ap_gain = (ap_rich - ap_control) / max(ap_control, 1e-12)
    auc_gain = metrics["rich_behavior"]["roc_auc"] - metrics["paper_raw"]["roc_auc"]

    # Pre-registered KEEP gate for this prerequisite test.
    # Passing this gate does NOT establish improved aggregate demand simulation.
    keep = bool(
        rel_ap_gain >= 0.05
        and bootstrap["average_precision"]["ci95_low"] > 0.0
        and auc_gain >= 0.01
    )

    result = {
        "experiment": "RICH-PERSONA-REPRESENTATION-SIGNAL-GATE-v1",
        "scientific_question": (
            "Does adding transaction-derived behavioral state improve prediction of future H&M top-100 trouser purchase "
            "for unseen customers, holding the classifier and split fixed?"
        ),
        "interpretation_scope": (
            "Prerequisite representation test only. It does not test LLM prompting, aggregate demand prediction, "
            "counterfactual pricing, or causal validity."
        ),
        "windows": {
            "history_start": str(history_start.date()),
            "history_end": str(history_end.date()),
            "target_start": str(target_start.date()),
            "target_end": str(target_end.date()),
        },
        "seed": args.seed,
        "n_customers": int(len(data)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "target_prevalence_all": float(data["label"].mean()),
        "n_positive_all": int(data["label"].sum()),
        "transaction_date_min": str(tx["t_dat"].min().date()),
        "transaction_date_max": str(tx["t_dat"].max().date()),
        "arms": arms,
        "metrics": metrics,
        "paired_bootstrap_rich_vs_paper_raw": bootstrap,
        "effect": {
            "average_precision_relative_gain": float(rel_ap_gain),
            "roc_auc_absolute_gain": float(auc_gain),
        },
        "keep_rule": {
            "relative_average_precision_gain_at_least": 0.05,
            "paired_bootstrap_ap_ci95_low_above_zero": True,
            "roc_auc_absolute_gain_at_least": 0.01,
        },
        "decision": "KEEP_FOR_LLM_SMOKE_TEST" if keep else "KILL_OR_REDESIGN_BEFORE_LLM_CALLS",
    }

    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2))

    rows = []
    for name, vals in metrics.items():
        rows.append({"arm": name, **vals})
    pd.DataFrame(rows).to_csv(args.output_dir / "metrics.csv", index=False)

    report = [
        "# Rich Persona Representation Signal Gate",
        "",
        f"**Decision:** `{result['decision']}`",
        "",
        "This experiment is deliberately narrower than the eventual Allegory test. It asks whether richer behavioral state "
        "contains additional real H&M future-purchase signal before spending money on new LLM simulations.",
        "",
        "## Metrics",
        "",
        "| arm | ROC AUC | Average precision | Log loss | Brier | Top-1% lift |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, vals in metrics.items():
        report.append(
            f"| {name} | {vals['roc_auc']:.6f} | {vals['average_precision']:.6f} | "
            f"{vals['log_loss']:.6f} | {vals['brier']:.6f} | {vals['lift_top_1pct']:.3f} |"
        )
    report += [
        "",
        "## Rich vs stronger paper_raw control",
        "",
        f"- Relative average-precision gain: **{rel_ap_gain:.2%}**",
        f"- Absolute ROC-AUC gain: **{auc_gain:.6f}**",
        f"- Paired-bootstrap AP delta 95% CI: "
        f"[{bootstrap['average_precision']['ci95_low']:.6f}, {bootstrap['average_precision']['ci95_high']:.6f}]",
        "",
        "## Scope",
        "",
        "A pass means the richer state deserves a controlled LLM smoke test. It does not yet mean that rich personas "
        "improve the H&M aggregate demand simulator. A failure means we should redesign the representation before paying for LLM calls.",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
