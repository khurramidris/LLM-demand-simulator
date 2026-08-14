from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, gammaln
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand_sim.config import DEFAULT_OUTPUT_DIR, DEFAULT_PRODUCTS_DIR, DEFAULT_RESPONSES_DIR
from demand_sim.data import build_prompting_design_rows, load_probability_rows, load_sales


LOG_R_LOWER_BOUNDS = (-6.0, -8.0, -10.0, -12.0, -16.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe the r->0 limit of zero-truncated Negative Binomial fits.")
    p.add_argument("--sales", type=Path, default=DEFAULT_PRODUCTS_DIR / "sales_top100_online.csv")
    p.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES_DIR / "llm_responses_online_top100.csv")
    p.add_argument("--saved-eval-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "demand_prediction")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "research" / "nb_zero_limit")
    p.add_argument("--n-splits", type=int, default=10)
    return p.parse_args()


def pooled_llm_feature(rows: pd.DataFrame, persona_ids: list[str]) -> np.ndarray:
    q = np.clip(rows[persona_ids].to_numpy(float).mean(axis=1), 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def ztnb_nll(y: np.ndarray, x: np.ndarray, r: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.clip(np.asarray(x, dtype=float), 1e-12, 1 - 1e-12)
    r = max(float(r), 1e-15)
    log_pmf = (
        gammaln(y + r) - gammaln(r) - gammaln(y + 1)
        + r * np.log1p(-x) + y * np.log(x)
    )
    log_p0 = r * np.log1p(-x)
    log_positive_mass = np.log(-np.expm1(log_p0))
    return -log_pmf + log_positive_mass


def fit_global_nb(y: np.ndarray, lower_log_r: float) -> dict:
    y = np.asarray(y, dtype=int)
    bounds = [(-12.0, 12.0), (lower_log_r, 5.0)]

    def objective(v: np.ndarray) -> float:
        x = float(expit(v[0])); r = float(np.exp(v[1]))
        return float(ztnb_nll(y, np.full(len(y), x), r).sum())

    best = None
    starts = ([2.0, -4.0], [3.0, max(lower_log_r + 0.5, -5.5)], [1.0, -2.0])
    for raw in starts:
        start = np.array([raw[0], np.clip(raw[1], lower_log_r + 1e-6, 5.0)], dtype=float)
        res = minimize(objective, start, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 600, "ftol": 1e-12, "gtol": 1e-8})
        for params, value, success, message in (
            (start, objective(start), True, "initial"),
            (res.x, float(res.fun), bool(res.success), str(res.message)),
        ):
            if np.isfinite(value) and (best is None or value < best[1]):
                best = (np.asarray(params), value, success, message)
    if best is None:
        raise RuntimeError("No finite NB fit")
    params, value, success, message = best
    x = float(expit(params[0])); r = float(np.exp(params[1]))
    return {
        "x": x, "r": r, "mu": r * x / max(1 - x, 1e-15),
        "objective": value, "success": success, "message": message,
        "hit_lower_bound": abs(float(params[1]) - lower_log_r) < 1e-5,
    }


def logseries_nll(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -(y * np.log(p) - np.log(y) - np.log(-np.log1p(-p)))


def fit_logseries_global(y: np.ndarray) -> dict:
    def obj(v: np.ndarray) -> float:
        p = float(expit(v[0]))
        return float(logseries_nll(y, np.full(len(y), p)).sum())
    res = minimize(obj, np.array([2.0]), method="L-BFGS-B", bounds=[(-12.0, 12.0)],
                   options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-9})
    return {"p": float(expit(res.x[0])), "success": bool(res.success), "objective": float(res.fun)}


def fit_logseries_llm(y: np.ndarray, z: np.ndarray) -> dict:
    z = np.asarray(z, dtype=float)
    mean, std = float(z.mean()), float(z.std())
    std = std if std >= 1e-12 else 1.0
    zs = (z - mean) / std
    def obj(v: np.ndarray) -> float:
        p = expit(np.clip(v[0] + v[1] * zs, -12, 12))
        return float(logseries_nll(y, p).sum())
    res = minimize(obj, np.array([2.0, 0.0]), method="L-BFGS-B",
                   bounds=[(-12.0, 12.0), (0.0, 12.0)],
                   options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-9})
    return {"params": np.asarray(res.x), "z_mean": mean, "z_std": std,
            "success": bool(res.success), "objective": float(res.fun)}


def predict_logseries_llm(fit: dict, z: np.ndarray) -> np.ndarray:
    zs = (np.asarray(z, dtype=float) - fit["z_mean"]) / fit["z_std"]
    return expit(np.clip(fit["params"][0] + fit["params"][1] * zs, -12, 12))


def logseries_mean(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -p / ((1 - p) * np.log1p(-p))


def logseries_crps(y: np.ndarray, p: np.ndarray, support_max: int = 400) -> np.ndarray:
    """Deterministic CRPS from the closed-form logarithmic-series PMF.

    PMF(k) = -p^k / (k log(1-p)), k>=1.  Vectorizing this directly is
    substantially faster than repeated scipy.stats.logser.cdf calls.
    """
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    k = np.arange(1, support_max + 1, dtype=float)
    out = np.empty(len(y), dtype=float)
    for start in range(0, len(y), 1024):
        stop = min(start + 1024, len(y))
        pp = p[start:stop, None]
        pmf = -(pp ** k[None, :]) / (k[None, :] * np.log1p(-pp))
        cdf = np.cumsum(pmf, axis=1)
        indicator = (k[None, :] >= y[start:stop, None]).astype(float)
        out[start:stop] = np.sum((cdf - indicator) ** 2, axis=1)
    return out


def score_logseries(y: np.ndarray, p: np.ndarray, with_crps: bool) -> dict:
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    pred = logseries_mean(p)
    return {
        "avg_nll": float(logseries_nll(y, p).mean()),
        "avg_crps": float(logseries_crps(y, p).mean()) if with_crps else float("nan"),
        "mae": float(np.mean(np.abs(y - pred))),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "mean_prediction": float(pred.mean()),
        "mean_p": float(p.mean()),
    }


def paired_interval(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float); v = v[np.isfinite(v)]
    n = len(v); mean = float(v.mean()) if n else float("nan")
    if n < 2:
        return {"mean": mean, "ci95_low": float("nan"), "ci95_high": float("nan")}
    se = float(v.std(ddof=1) / math.sqrt(n)); crit = float(t.ppf(0.975, n - 1))
    return {"mean": mean, "se": se, "ci95_low": mean - crit * se,
            "ci95_high": mean + crit * se, "fraction_negative": float(np.mean(v < 0))}


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    sales = load_sales(args.sales)
    probabilities = load_probability_rows(args.responses)
    rows, persona_ids = build_prompting_design_rows(sales, probabilities)
    rows = rows[rows["demand"] > 0].copy().reset_index(drop=True)

    nb_records, log_records = [], []
    hist = rows["demand"].value_counts().sort_index()
    histogram = {str(int(k)): int(v) for k, v in hist.head(20).items()}

    for split_idx in range(args.n_splits):
        split_dir = args.saved_eval_dir / f"split_{split_idx:03d}"
        train_ids = pd.read_csv(split_dir / "train_products.csv")["article_id"].astype(int)
        test_ids = pd.read_csv(split_dir / "test_products.csv")["article_id"].astype(int)
        train = rows[rows["article_id"].isin(train_ids)].copy().reset_index(drop=True)
        test = rows[rows["article_id"].isin(test_ids)].copy().reset_index(drop=True)
        y_train = train["demand"].to_numpy(int); y_test = test["demand"].to_numpy(int)

        for lower_log_r in LOG_R_LOWER_BOUNDS:
            fit = fit_global_nb(y_train, lower_log_r)
            for sample, y in (("train", y_train), ("test", y_test)):
                nb_records.append({
                    "split": split_idx, "sample": sample, "lower_log_r": lower_log_r,
                    "lower_r": float(np.exp(lower_log_r)), "fitted_r": fit["r"],
                    "fitted_x": fit["x"], "implied_untruncated_mu": fit["mu"],
                    "avg_nll": float(ztnb_nll(y, np.full(len(y), fit["x"]), fit["r"]).mean()),
                    "optimizer_success": fit["success"], "hit_lower_bound": fit["hit_lower_bound"],
                })

        global_fit = fit_logseries_global(y_train)
        llm_fit = fit_logseries_llm(y_train, pooled_llm_feature(train, persona_ids))
        for sample, sample_rows, y in (("train", train, y_train), ("test", test, y_test)):
            global_p = np.full(len(y), global_fit["p"])
            llm_p = predict_logseries_llm(llm_fit, pooled_llm_feature(sample_rows, persona_ids))
            for name, p in (("logseries-global", global_p), ("logseries-llm-pooled", llm_p)):
                log_records.append({
                    "split": split_idx, "sample": sample, "model": name,
                    **score_logseries(y, p, with_crps=(sample == "test")),
                    "global_p": global_fit["p"], "llm_slope": float(llm_fit["params"][1]),
                })
        print(f"completed split {split_idx:03d}", flush=True)

    nb = pd.DataFrame(nb_records); logs = pd.DataFrame(log_records)
    nb.to_csv(args.output_dir / "nb_lower_bound_profile.csv", index=False)
    logs.to_csv(args.output_dir / "logseries_scores.csv", index=False)

    nb_summary = nb.groupby(["sample", "lower_log_r"], as_index=False).agg(
        mean_avg_nll=("avg_nll", "mean"), mean_fitted_r=("fitted_r", "mean"),
        median_fitted_r=("fitted_r", "median"), mean_implied_mu=("implied_untruncated_mu", "mean"),
        fraction_at_lower_bound=("hit_lower_bound", "mean"), optimizer_success_rate=("optimizer_success", "mean"))
    log_summary = logs.groupby(["sample", "model"], as_index=False).agg(
        avg_nll=("avg_nll", "mean"), avg_crps=("avg_crps", "mean"), mae=("mae", "mean"),
        rmse=("rmse", "mean"), mean_prediction=("mean_prediction", "mean"),
        mean_p=("mean_p", "mean"), mean_llm_slope=("llm_slope", "mean"))
    nb_summary.to_csv(args.output_dir / "nb_lower_bound_summary.csv", index=False)
    log_summary.to_csv(args.output_dir / "logseries_summary.csv", index=False)

    test = logs[logs["sample"] == "test"].pivot(index="split", columns="model", values=["avg_nll", "avg_crps", "mae"])
    llm_pairs = {}
    for metric in ("avg_nll", "avg_crps", "mae"):
        delta = test[(metric, "logseries-llm-pooled")].to_numpy() - test[(metric, "logseries-global")].to_numpy()
        llm_pairs[metric] = paired_interval(delta)

    wide = nb[(nb["sample"] == "test") & (nb["lower_log_r"] == -16.0)][["split", "avg_nll"]]
    base = nb[(nb["sample"] == "test") & (nb["lower_log_r"] == -6.0)][["split", "avg_nll"]]
    comp = wide.merge(base, on="split", suffixes=("_wide", "_base"))
    bound_delta = comp["avg_nll_wide"].to_numpy() - comp["avg_nll_base"].to_numpy()

    decision = {
        "experiment": "nb_zero_dispersion_limit", "n_splits": int(logs["split"].nunique()),
        "nb_test_nll_change_log_r_lower_-16_minus_-6": paired_interval(bound_delta),
        "fraction_nb_fits_at_lower_bound_by_bound": {
            str(r.lower_log_r): float(r.fraction_at_lower_bound)
            for r in nb_summary[nb_summary["sample"] == "train"].itertuples(index=False)},
        "logseries_test_means": {
            r.model: {"nll": float(r.avg_nll), "crps": float(r.avg_crps), "mae": float(r.mae),
                      "mean_prediction": float(r.mean_prediction)}
            for r in log_summary[log_summary["sample"] == "test"].itertuples(index=False)},
        "llm_minus_global_logseries_paired": llm_pairs,
        "positive_demand_histogram_first_20_counts": histogram,
        "positive_rows": int(len(rows)), "positive_demand_mean": float(rows["demand"].mean()),
        "positive_demand_median": float(rows["demand"].median()),
        "interpretation_rule": (
            "If lowering the NB dispersion bound keeps improving likelihood and converges to the logarithmic-series "
            "limit, finite NB dispersion is not identified. Positive-only rows primarily identify the conditional "
            "positive-count shape rather than demand incidence or exposure."),
        "new_llm_calls": 0, "api_cost_usd": 0,
    }
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2))
    print("\nNB lower-bound profile:\n", nb_summary.to_string(index=False))
    print("\nLogseries summary:\n", log_summary.to_string(index=False))
    print("\nDecision:\n", json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
