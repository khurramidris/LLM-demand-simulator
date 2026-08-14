from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from experiment_rich_persona_signal import (
    build_label,
    build_paper_state,
    build_rich_features,
    evaluate,
    load_trouser_transactions,
    make_pipeline,
    paired_bootstrap,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Confirmatory paired comparison of rich behavior vs original paper bins.")
    p.add_argument("--customer-features", type=Path, required=True)
    p.add_argument("--trouser-transactions", type=Path, required=True)
    p.add_argument("--product-info", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--bootstrap-reps", type=int, default=200)
    p.add_argument("--bootstrap-n", type=int, default=50_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    history_start = pd.Timestamp("2018-09-01")
    history_end = pd.Timestamp("2019-09-19")
    target_start = pd.Timestamp("2019-09-20")
    target_end = pd.Timestamp("2019-12-31")

    customer_features = pd.read_csv(args.customer_features, compression="zip")
    paper = build_paper_state(customer_features, history_start, history_end)
    tx = load_trouser_transactions(args.trouser_transactions)
    rich = build_rich_features(tx, history_start, history_end)
    top_product_ids = set(pd.read_csv(args.product_info)["article_id"].astype(int))
    positive_label = build_label(tx, top_product_ids, target_start, target_end)

    data = paper.merge(rich, on="customer_id", how="left")
    data["label"] = data["customer_id"].map(positive_label).fillna(0).astype(int)
    data = data[data["txn_count"] > 0].reset_index(drop=True)

    train_idx, test_idx = train_test_split(
        np.arange(len(data)),
        test_size=0.30,
        random_state=args.seed,
        stratify=data["label"].to_numpy(),
    )
    train = data.iloc[train_idx]
    test = data.iloc[test_idx]
    y_train = train["label"].to_numpy(int)
    y_test = test["label"].to_numpy(int)

    bins_cat = ["age_bin", "engagement_bin", "price_tier", "top_product_type"]
    rich_cat = ["top_product_type"]
    rich_num = [
        "age", "txn_count", "mean_price", "txn_per_month",
        "tr_txn_count", "tr_unique_articles", "tr_mean_price", "tr_price_std",
        "tr_price_min", "tr_price_max", "tr_active_days", "tr_price_q25", "tr_price_q75",
        "tr_purchase_span_days", "tr_days_since_last", "tr_repeat_ratio", "tr_price_iqr",
        "tr_txn_last_30d", "tr_txn_last_90d", "tr_txn_last_180d", "tr_txn_per_month", "tr_recent90_share",
    ]

    bins_model = make_pipeline(bins_cat, [], args.seed)
    rich_model = make_pipeline(rich_cat, rich_num, args.seed)
    bins_model.fit(train[bins_cat], y_train)
    rich_model.fit(train[rich_cat + rich_num], y_train)

    p_bins = bins_model.predict_proba(test[bins_cat])[:, 1]
    p_rich = rich_model.predict_proba(test[rich_cat + rich_num])[:, 1]
    bins_metrics = evaluate(y_test, p_bins)
    rich_metrics = evaluate(y_test, p_rich)
    bootstrap = paired_bootstrap(
        y_test,
        p_bins,
        p_rich,
        reps=args.bootstrap_reps,
        sample_n=args.bootstrap_n,
        seed=args.seed + 2,
    )

    result = {
        "analysis": "CONFIRMATORY-RICH-vs-PAPER-BINS",
        "note": "Added after the preregistered paper_raw gate because paper_bins empirically proved the stronger control. Does not alter the original gate.",
        "n_test": int(len(test)),
        "paper_bins": bins_metrics,
        "rich_behavior": rich_metrics,
        "effect": {
            "average_precision_absolute_delta": float(rich_metrics["average_precision"] - bins_metrics["average_precision"]),
            "average_precision_relative_delta": float(
                (rich_metrics["average_precision"] - bins_metrics["average_precision"])
                / bins_metrics["average_precision"]
            ),
            "roc_auc_absolute_delta": float(rich_metrics["roc_auc"] - bins_metrics["roc_auc"]),
            "log_loss_relative_delta": float(
                (rich_metrics["log_loss"] - bins_metrics["log_loss"]) / bins_metrics["log_loss"]
            ),
            "top_1pct_lift_relative_delta": float(
                (rich_metrics["lift_top_1pct"] - bins_metrics["lift_top_1pct"])
                / bins_metrics["lift_top_1pct"]
            ),
        },
        "paired_bootstrap_rich_minus_paper_bins": bootstrap,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
