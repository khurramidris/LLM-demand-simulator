from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, gammaln
from scipy.stats import binom

ROOT = Path("outputs/research/rich_persona_signal/temporal_holdout")
CUTOFF = pd.Timestamp("2019-05-31")
EXPOSURE_VALUES = [100, 150, 200, 250]
SEED = 2025


def logit(x: np.ndarray | float) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def calibrated(q: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    return np.clip(expit(intercept + slope * logit(q)), 1e-9, 1 - 1e-9)


def row_zt_nll(y: np.ndarray, n: int, p: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    out = np.full(len(y), np.inf, dtype=float)
    ok = (y > 0) & (y <= n)
    yy = y[ok]
    pp = p[ok]
    log_choose = gammaln(n + 1) - gammaln(yy + 1) - gammaln(n - yy + 1)
    log_pmf = log_choose + yy * np.log(pp) + (n - yy) * np.log1p(-pp)
    zero_mass = np.exp(n * np.log1p(-pp))
    out[ok] = -(log_pmf - np.log1p(-zero_mass))
    return out


def exact_zt_crps(y: np.ndarray, n: int, p: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    support = np.arange(1, n + 1)
    values = np.empty(len(y), dtype=float)
    for i, (obs, prob) in enumerate(zip(y, p)):
        zero = binom.pmf(0, n, prob)
        pmf = binom.pmf(support, n, prob) / max(1.0 - zero, 1e-12)
        cdf = np.cumsum(pmf)
        indicator = (support >= obs).astype(float)
        values[i] = float(np.sum((cdf - indicator) ** 2))
    return values


def fit_arm(train: pd.DataFrame) -> dict:
    q = train["q_agg"].to_numpy(float)
    y = train["demand"].to_numpy(int)
    best = None
    for n in EXPOSURE_VALUES:
        if n < int(y.max()):
            continue
        target = float(np.clip(y.mean() / n, 1e-6, 1 - 1e-6))
        qmean = float(np.clip(q.mean(), 1e-6, 1 - 1e-6))
        init_intercept = float(logit(target) - logit(qmean))

        def objective(params: np.ndarray) -> float:
            intercept = float(params[0])
            slope = float(np.exp(params[1]))
            p = calibrated(q, intercept, slope)
            vals = row_zt_nll(y, n, p)
            return float(np.mean(vals))

        res = minimize(
            objective,
            x0=np.array([init_intercept, 0.0]),
            method="L-BFGS-B",
            bounds=[(-20.0, 20.0), (-5.0, 5.0)],
            options={"maxiter": 300},
        )
        cand = {
            "exposure_n": int(n),
            "intercept": float(res.x[0]),
            "slope": float(np.exp(res.x[1])),
            "train_zt_nll": float(res.fun),
            "optimizer_success": bool(res.success),
            "optimizer_message": str(res.message),
        }
        if best is None or cand["train_zt_nll"] < best["train_zt_nll"]:
            best = cand
    if best is None:
        raise RuntimeError(f"No exposure N supports train max demand={int(y.max())}")
    return best


def score_arm(test: pd.DataFrame, fit: dict) -> pd.DataFrame:
    out = test.copy()
    n = int(fit["exposure_n"])
    p = calibrated(out["q_agg"].to_numpy(float), fit["intercept"], fit["slope"])
    zero = np.exp(n * np.log1p(-p))
    mean_pred = n * p / np.clip(1 - zero, 1e-12, None)
    y = out["demand"].to_numpy(int)
    out["purchase_prob"] = p
    out["mean_prediction"] = mean_pred
    out["nll"] = row_zt_nll(y, n, p)
    out["crps"] = exact_zt_crps(y, n, p)
    out["abs_error"] = np.abs(y - mean_pred)
    out["sq_error"] = (y - mean_pred) ** 2
    return out


def summarize(scored: pd.DataFrame) -> dict:
    return {
        "n_test_rows": int(len(scored)),
        "zt_avg_nll": float(scored["nll"].mean()),
        "zt_avg_crps": float(scored["crps"].mean()),
        "mae": float(scored["abs_error"].mean()),
        "rmse": float(np.sqrt(scored["sq_error"].mean())),
    }


def bootstrap(control: pd.DataFrame, rich: pd.DataFrame, n_boot: int = 5000) -> dict:
    keys = ["date", "article_id", "offer_price", "demand"]
    c = control.sort_values(keys).reset_index(drop=True)
    r = rich.sort_values(keys).reset_index(drop=True)
    if not c[keys].equals(r[keys]):
        raise RuntimeError("Control/rich test rows are not exactly paired")
    rng = np.random.default_rng(SEED)
    n = len(c)
    diffs = {"nll": [], "crps": [], "mae": [], "rmse": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs["nll"].append(float(r.loc[idx, "nll"].mean() - c.loc[idx, "nll"].mean()))
        diffs["crps"].append(float(r.loc[idx, "crps"].mean() - c.loc[idx, "crps"].mean()))
        diffs["mae"].append(float(r.loc[idx, "abs_error"].mean() - c.loc[idx, "abs_error"].mean()))
        diffs["rmse"].append(float(np.sqrt(r.loc[idx, "sq_error"].mean()) - np.sqrt(c.loc[idx, "sq_error"].mean())))
    result = {}
    for metric, vals in diffs.items():
        arr = np.asarray(vals)
        result[metric] = {
            "rich_minus_control_mean": float(arr.mean()),
            "ci95_low": float(np.quantile(arr, 0.025)),
            "ci95_high": float(np.quantile(arr, 0.975)),
            "p_rich_better": float(np.mean(arr < 0)),
        }
    return result


def load_responses() -> pd.DataFrame:
    records = []
    for path in sorted(ROOT.glob("responses_*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    if len(records) != 100:
        raise RuntimeError(f"Expected 100 response records, got {len(records)}")

    prompt_rows = [json.loads(line) for line in (ROOT / "prompt_pairs.jsonl").read_text().splitlines() if line.strip()]
    expected = {(x["pair_id"], arm) for x in prompt_rows for arm in ["control", "rich"]}
    observed = {(x["pair_id"], x["arm"]) for x in records}
    if observed != expected:
        raise RuntimeError(f"Response keys mismatch: missing={expected-observed}, extra={observed-expected}")

    expected_prices = {x["pair_id"]: [float(v) for v in json.loads(x["prices_json"])] for x in prompt_rows}
    long = []
    for rec in records:
        prices = [round(float(v), 2) for v in rec["response"]["prices"]]
        probs = [float(v) for v in rec["response"]["p_buy"]]
        exp_prices = [round(float(v), 2) for v in expected_prices[rec["pair_id"]]]
        if prices != exp_prices or len(prices) != len(probs):
            raise RuntimeError(f"Bad price grid for {rec['pair_id']} {rec['arm']}")
        if not np.all(np.isfinite(probs)) or np.any(np.asarray(probs) < 0) or np.any(np.asarray(probs) > 1):
            raise RuntimeError(f"Invalid probability for {rec['pair_id']} {rec['arm']}")
        for price, prob in zip(prices, probs):
            long.append({
                "arm": rec["arm"],
                "article_id": int(rec["article_id"]),
                "persona_id": rec["persona_id"],
                "offer_price": price,
                "p_buy": prob,
            })
    return pd.DataFrame(long)


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    personas = pd.read_csv(ROOT / "persona_prompts.csv")
    weights = personas[["persona_id", "n_customers"]].copy()
    weights["weight"] = weights["n_customers"] / weights["n_customers"].sum()

    long = load_responses().merge(weights[["persona_id", "weight"]], on="persona_id", how="left", validate="many_to_one")
    long["weighted_p"] = long["p_buy"] * long["weight"]
    q = long.groupby(["arm", "article_id", "offer_price"], as_index=False)["weighted_p"].sum().rename(columns={"weighted_p": "q_agg"})

    sales = pd.read_csv("outputs/products/sales_top100_online.csv")
    sales["date"] = pd.to_datetime(sales["date"])
    if "price" in sales.columns and "offer_price" not in sales.columns:
        sales = sales.rename(columns={"price": "offer_price"})
    sales["offer_price"] = sales["offer_price"].astype(float).round(2)
    sales["article_id"] = sales["article_id"].astype(int)
    sales["demand"] = sales["demand"].astype(int)
    products = sorted(q["article_id"].unique().tolist())
    product_sales = sales[sales["article_id"].isin(products) & (sales["demand"] > 0)].copy()

    coverage = []
    arm_frames = {}
    fits = {}
    scored = {}
    summaries = {}
    for arm in ["control", "rich"]:
        qa = q[q["arm"] == arm].drop(columns="arm")
        merged = product_sales.merge(qa, on=["article_id", "offer_price"], how="inner")
        train = merged[merged["date"] <= CUTOFF].copy()
        test = merged[merged["date"] > CUTOFF].copy()
        if train.empty or test.empty:
            raise RuntimeError(f"Arm {arm}: empty train/test after merge")
        fits[arm] = fit_arm(train)
        scored[arm] = score_arm(test, fits[arm])
        summaries[arm] = summarize(scored[arm])
        arm_frames[arm] = {"train": train, "test": test}

    # Same price grid means these should pair exactly.
    boot = bootstrap(scored["control"], scored["rich"])

    base_train = product_sales[product_sales["date"] <= CUTOFF]
    base_test = product_sales[product_sales["date"] > CUTOFF]
    matched_train = len(arm_frames["control"]["train"])
    matched_test = len(arm_frames["control"]["test"])
    coverage = {
        "all_selected_product_rows_train": int(len(base_train)),
        "matched_candidate_price_rows_train": int(matched_train),
        "train_row_coverage": float(matched_train / max(len(base_train), 1)),
        "all_selected_product_rows_test": int(len(base_test)),
        "matched_candidate_price_rows_test": int(matched_test),
        "test_row_coverage": float(matched_test / max(len(base_test), 1)),
        "test_start": str(arm_frames["control"]["test"]["date"].min().date()),
        "test_end": str(arm_frames["control"]["test"]["date"].max().date()),
    }

    c = summaries["control"]
    r = summaries["rich"]
    effects = {
        metric: {
            "absolute_delta_rich_minus_control": float(r[metric] - c[metric]),
            "relative_delta": float((r[metric] - c[metric]) / c[metric]),
        }
        for metric in ["zt_avg_nll", "zt_avg_crps", "mae", "rmse"]
    }
    advance = (
        r["zt_avg_nll"] < c["zt_avg_nll"]
        and r["zt_avg_crps"] < c["zt_avg_crps"]
        and r["mae"] <= 1.05 * c["mae"]
        and r["rmse"] <= 1.05 * c["rmse"]
    )
    decision = "ADVANCE" if advance else "DO_NOT_ADVANCE"

    result = {
        "protocol": manifest["protocol"],
        "decision": decision,
        "coverage": coverage,
        "fits": fits,
        "summary": summaries,
        "effects": effects,
        "paired_bootstrap": boot,
        "limitations": manifest["shared_limitations"],
        "claim_scope": "Small 5-product temporal real-demand pilot of incremental rich-history prompting; not the paper's product-holdout benchmark.",
    }
    (ROOT / "result.json").write_text(json.dumps(result, indent=2))

    paired = scored["control"][["date", "article_id", "offer_price", "demand", "q_agg", "purchase_prob", "mean_prediction", "nll", "crps", "abs_error", "sq_error"]].copy()
    paired = paired.rename(columns={c: f"control_{c}" for c in ["q_agg", "purchase_prob", "mean_prediction", "nll", "crps", "abs_error", "sq_error"]})
    rich_cols = scored["rich"][["date", "article_id", "offer_price", "demand", "q_agg", "purchase_prob", "mean_prediction", "nll", "crps", "abs_error", "sq_error"]].copy()
    rich_cols = rich_cols.rename(columns={c: f"rich_{c}" for c in ["q_agg", "purchase_prob", "mean_prediction", "nll", "crps", "abs_error", "sq_error"]})
    paired = paired.merge(rich_cols, on=["date", "article_id", "offer_price", "demand"], validate="one_to_one")
    paired.to_csv(ROOT / "test_predictions.csv", index=False)

    lines = [
        "# GPT-5.6 Sol Temporal H&M Demand Pilot",
        "",
        f"**Decision:** {decision}",
        "",
        "This is a real H&M temporal holdout pilot: behavior context and calibration stop at 2019-05-31; scored demand is after that date.",
        "",
        "## Test metrics (lower is better)",
        "",
        "| metric | control | rich | rich vs control |",
        "|---|---:|---:|---:|",
    ]
    for metric in ["zt_avg_nll", "zt_avg_crps", "mae", "rmse"]:
        lines.append(f"| {metric} | {c[metric]:.6f} | {r[metric]:.6f} | {effects[metric]['relative_delta']*100:+.2f}% |")
    lines += [
        "",
        "## Coverage",
        "",
        f"- Training matched rows: {coverage['matched_candidate_price_rows_train']} / {coverage['all_selected_product_rows_train']} ({coverage['train_row_coverage']*100:.1f}%).",
        f"- Held-out matched rows: {coverage['matched_candidate_price_rows_test']} / {coverage['all_selected_product_rows_test']} ({coverage['test_row_coverage']*100:.1f}%).",
        f"- Held-out dates: {coverage['test_start']} to {coverage['test_end']}.",
        "",
        "## Paired bootstrap",
        "",
    ]
    for metric, rec in boot.items():
        lines.append(f"- {metric}: rich-control mean {rec['rich_minus_control_mean']:.6f}, 95% CI [{rec['ci95_low']:.6f}, {rec['ci95_high']:.6f}], P(rich better)={rec['p_rich_better']:.3f}.")
    lines += [
        "",
        "## Scientific scope",
        "",
        "This tests whether cutoff-safe trouser-history enrichment improves future real H&M demand predictions for five fixed products under identical population weighting and calibration procedure.",
        "It does not establish product cold-start generalization, causal price elasticity, or a full improvement over the paper.",
        "The original persona-cell memberships and candidate price grids remain shared transductive artifacts from the repository.",
        "The GPT-5.6 outputs are conversation-batched rather than independent blinded API calls.",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
