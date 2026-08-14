from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
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

EPS = 1e-8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preregistered calibration check for rich behavioral state vs original paper bins."
    )
    p.add_argument("--customer-features", type=Path, required=True)
    p.add_argument("--trouser-transactions", type=Path, required=True)
    p.add_argument("--product-info", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--bootstrap-reps", type=int, default=200)
    p.add_argument("--bootstrap-n", type=int, default=50_000)
    return p.parse_args()


def monotone_logit_calibrator(y_val: np.ndarray, p_val: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(p_val, dtype=float), EPS, 1.0 - EPS)
    z = np.log(p / (1.0 - p))
    y = np.asarray(y_val, dtype=float)

    target = float(np.clip(y.mean(), EPS, 1.0 - EPS))
    raw = float(np.clip(p.mean(), EPS, 1.0 - EPS))
    x0 = np.array([
        np.log(target / (1.0 - target)) - np.log(raw / (1.0 - raw)),
        0.0,
    ])

    def objective(x: np.ndarray) -> float:
        intercept = float(x[0])
        slope = float(np.exp(x[1]))
        q = np.clip(expit(intercept + slope * z), EPS, 1.0 - EPS)
        return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))

    result = minimize(
        objective,
        x0=x0,
        method="L-BFGS-B",
        bounds=[(-30.0, 30.0), (-6.0, 6.0)],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Calibration optimization failed: {result.message}")
    return float(result.x[0]), float(np.exp(result.x[1]))


def apply_calibrator(p: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    z = np.log(p / (1.0 - p))
    return np.clip(expit(intercept + slope * z), EPS, 1.0 - EPS)


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
    y_all = data["label"].to_numpy(int)

    # Fixed 60/20/20 customer split. Test is never used to fit either the behavioral classifier or calibration.
    trainval_idx, test_idx = train_test_split(
        np.arange(len(data)), test_size=0.20, random_state=args.seed, stratify=y_all
    )
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=0.25,
        random_state=args.seed + 1,
        stratify=y_all[trainval_idx],
    )
    train, val, test = data.iloc[train_idx], data.iloc[val_idx], data.iloc[test_idx]
    y_train = train["label"].to_numpy(int)
    y_val = val["label"].to_numpy(int)
    y_test = test["label"].to_numpy(int)

    arms = {
        "paper_bins": {
            "categorical": ["age_bin", "engagement_bin", "price_tier", "top_product_type"],
            "numeric": [],
        },
        "rich_behavior": {
            "categorical": ["top_product_type"],
            "numeric": [
                "age", "txn_count", "mean_price", "txn_per_month",
                "tr_txn_count", "tr_unique_articles", "tr_mean_price", "tr_price_std",
                "tr_price_min", "tr_price_max", "tr_active_days", "tr_price_q25", "tr_price_q75",
                "tr_purchase_span_days", "tr_days_since_last", "tr_repeat_ratio", "tr_price_iqr",
                "tr_txn_last_30d", "tr_txn_last_90d", "tr_txn_last_180d", "tr_txn_per_month", "tr_recent90_share",
            ],
        },
    }

    raw_test: dict[str, np.ndarray] = {}
    cal_test: dict[str, np.ndarray] = {}
    results: dict[str, dict] = {}
    for name, spec in arms.items():
        cols = spec["categorical"] + spec["numeric"]
        model = make_pipeline(spec["categorical"], spec["numeric"], args.seed)
        model.fit(train[cols], y_train)
        p_val = model.predict_proba(val[cols])[:, 1]
        p_test = model.predict_proba(test[cols])[:, 1]
        intercept, slope = monotone_logit_calibrator(y_val, p_val)
        p_test_cal = apply_calibrator(p_test, intercept, slope)
        raw_test[name] = p_test
        cal_test[name] = p_test_cal
        results[name] = {
            "raw": evaluate(y_test, p_test),
            "calibrated": evaluate(y_test, p_test_cal),
            "calibration": {"intercept": intercept, "slope": slope},
        }

    bootstrap = paired_bootstrap(
        y_test,
        cal_test["paper_bins"],
        cal_test["rich_behavior"],
        reps=args.bootstrap_reps,
        sample_n=args.bootstrap_n,
        seed=args.seed + 3,
    )

    paper = results["paper_bins"]["calibrated"]
    richm = results["rich_behavior"]["calibrated"]
    ap_rel = (richm["average_precision"] - paper["average_precision"]) / paper["average_precision"]
    auc_delta = richm["roc_auc"] - paper["roc_auc"]
    logloss_rel = (richm["log_loss"] - paper["log_loss"]) / paper["log_loss"]
    brier_rel = (richm["brier"] - paper["brier"]) / paper["brier"]

    # Preregistered before this run. This is a gate to a small LLM experiment, not a claim that demand simulation improved.
    # We require a real rare-event ranking gain, statistically supported AP delta, and near-parity in calibrated probability quality.
    advance = bool(
        ap_rel >= 0.10
        and bootstrap["average_precision"]["ci95_low"] > 0.0
        and auc_delta >= -0.02
        and logloss_rel <= 0.05
        and brier_rel <= 0.02
    )

    output = {
        "analysis": "PREREGISTERED-CALIBRATED-RICH-vs-PAPER-BINS-v1",
        "hypothesis": (
            "The rich representation's poor raw log loss is substantially calibration error; after fitting the same "
            "monotone logit calibrator on a validation set, rich behavior retains its AP advantage while reaching "
            "near-parity with paper bins on proper probability scores."
        ),
        "split": {
            "seed": args.seed,
            "n_train": int(len(train)),
            "n_validation": int(len(val)),
            "n_test": int(len(test)),
            "target_prevalence_test": float(y_test.mean()),
        },
        "results": results,
        "paired_bootstrap_calibrated_rich_minus_paper_bins": bootstrap,
        "effect_calibrated": {
            "average_precision_relative_delta": float(ap_rel),
            "roc_auc_absolute_delta": float(auc_delta),
            "log_loss_relative_delta": float(logloss_rel),
            "brier_relative_delta": float(brier_rel),
        },
        "advance_rule": {
            "average_precision_relative_gain_at_least": 0.10,
            "paired_bootstrap_ap_ci95_low_above_zero": True,
            "roc_auc_delta_at_least": -0.02,
            "calibrated_log_loss_no_more_than_5pct_worse": True,
            "calibrated_brier_no_more_than_2pct_worse": True,
        },
        "decision": "ADVANCE_TO_LLM_SMOKE_TEST" if advance else "DO_NOT_ADVANCE_YET",
        "scope": (
            "Customer-level future-purchase signal gate only. It is not an aggregate-demand, counterfactual-price, "
            "or causal validation result. Rich features are trouser-history features, not a literal implementation of SimPersona/GenAgents."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
