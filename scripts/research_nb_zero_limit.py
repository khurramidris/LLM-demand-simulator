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
from scipy.stats import logser, t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand_sim.config import DEFAULT_OUTPUT_DIR, DEFAULT_PRODUCTS_DIR, DEFAULT_RESPONSES_DIR
from demand_sim.data import build_prompting_design_rows, load_probability_rows, load_sales


LOG_R_LOWER_BOUNDS = (-6.0, -8.0, -10.0, -12.0, -16.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose the zero-dispersion limit of the zero-truncated Negative Binomial."
    )
    parser.add_argument("--sales", type=Path, default=DEFAULT_PRODUCTS_DIR / "sales_top100_online.csv")
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES_DIR / "llm_responses_online_top100.csv")
    parser.add_argument("--saved-eval-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "demand_prediction")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "research" / "nb_zero_limit",
    )
    parser.add_argument("--n-splits", type=int, default=10)
    return parser.parse_args()


def pooled_llm_feature(rows: pd.DataFrame, persona_ids: list[str]) -> np.ndarray:
    pooled = np.clip(rows[persona_ids].to_numpy(float).mean(axis=1), 1e-6, 1.0 - 1e-6)
    return np.log(pooled / (1.0 - pooled))


def stable_log_positive_mass(r: float, x: np.ndarray) -> np.ndarray:
    log_p0 = r * np.log1p(-x)
    return np.log(-np.expm1(log_p0))


def ztnb_direct_nll(
    y: np.ndarray,
    x: np.ndarray,
    r: float,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.clip(np.asarray(x, dtype=float), 1e-12, 1.0 - 1e-12)
    r = float(max(r, 1e-15))
    log_pmf = (
        gammaln(y + r)
        - gammaln(r)
        - gammaln(y + 1.0)
        + r * np.log1p(-x)
        + y * np.log(x)
    )
    return -log_pmf + stable_log_positive_mass(r, x)


def fit_global_nb(y: np.ndarray, lower_log_r: float) -> dict:
    y = np.asarray(y, dtype=int)
    starts = [
        np.array([2.0, -4.0]),
        np.array([3.0, max(lower_log_r + 0.5, -5.5)]),
        np.array([1.0, -2.0]),
    ]
    bounds = [(-12.0, 12.0), (lower_log_r, 5.0)]

    def objective(params: np.ndarray) -> float:
        x = float(expit(params[0]))
        r = float(np.exp(params[1]))
        return float(np.sum(ztnb_direct_nll(y, np.full(len(y), x), r)))

    best = None
    for start in starts:
        start = np.array(
            [start[0], float(np.clip(start[1], lower_log_r + 1e-6, 5.0))],
            dtype=float,
        )
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 600, "ftol": 1e-12, "gtol": 1e-8},
        )
        candidates = [
            (start, objective(start), True, "initial"),
            (result.x, float(result.fun), bool(result.success), str(result.message)),
        ]
        for params, value, success, message in candidates:
            if np.isfinite(value) and (best is None or value < best[1]):
                best = (np.asarray(params, dtype=float), value, success, message)
    if best is None:
        raise RuntimeError("No finite global NB fit")
    params, objective_value, success, message = best
    x = float(expit(params[0]))
    r = float(np.exp(params[1]))
    mu = r * x / max(1.0 - x, 1e-15)
    return {
        "x": x,
        "r": r,
        "mu": mu,
        "objective": objective_value,
        "success": success,
        "message": message,
        "hit_lower_bound": bool(abs(params[1] - lower_log_r) < 1e-5),
    }


def logseries_nll(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    normalizer = -np.log1p(-p)
    return -(y * np.log(p) - np.log(y) - np.log(normalizer))


def fit_logseries_global(y: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)

    def objective(theta: np.ndarray) -> float:
        p = float(expit(theta[0]))
        return float(np.sum(logseries_nll(y, np.full(len(y), p))))

    result = minimize(
        objective,
        np.array([2.0]),
        method="L-BFGS-B",
        bounds=[(-12.0, 12.0)],
        options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-9},
    )
    return {
        "params": np.asarray(result.x, dtype=float),
        "p": float(expit(result.x[0])),
        "objective": float(result.fun),
        "success": bool(result.success),
        "message": str(result.message),
    }


def fit_logseries_llm(y: np.ndarray, z: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    z = np.asarray(z, dtype=float)
    mean = float(np.mean(z))
    std = float(np.std(z))
    if std < 1e-12:
        std = 1.0
    zs = (z - mean) / std

    def objective(params: np.ndarray) -> float:
        p = expit(np.clip(params[0] + params[1] * zs, -12.0, 12.0))
        return float(np.sum(logseries_nll(y, p)))

    result = minimize(
        objective,
        np.array([2.0, 0.0]),
        method="L-BFGS-B",
        bounds=[(-12.0, 12.0), (0.0, 12.0)],
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-9},
    )
    return {
        "params": np.asarray(result.x, dtype=float),
        "z_mean": mean,
        "z_std": std,
        "objective": float(result.fun),
        "success": bool(result.success),
        "message": str(result.message),
    }


def predict_logseries_llm(fit: dict, z: np.ndarray) -> np.ndarray:
    zs = (np.asarray(z, dtype=float) - fit["z_mean"]) / fit["z_std"]
    return expit(np.clip(fit["params"][0] + fit["params"][1] * zs, -12.0, 12.0))


def logseries_mean(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    return -p / ((1.0 - p) * np.log1p(-p))


def logseries_crps(y: np.ndarray, p: np.ndarray, support_max: int = 400) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    support = np.arange(1, support_max + 1, dtype=int)
    out = np.empty(len(y), dtype=float)
    for start in range(0, len(y), 512):
        stop = min(start + 512, len(y))
        cdf = logser.cdf(support[None, :], p[start:stop, None])
        indicator = (support[None, :] >= y[start:stop, None]).astype(float)
        out[start:stop] = np.sum((cdf - indicator) ** 2, axis=1)
    return out


def score_logseries(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = logseries_mean(p)
    return {
        "avg_nll": float(np.mean(logseries_nll(y, p))),
        "avg_crps": float(np.mean(logseries_crps(y, p))),
        "mae": float(np.mean(np.abs(y - pred))),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "mean_prediction": float(np.mean(pred)),
        "mean_p": float(np.mean(p)),
    }


def paired_interval(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return {"mean": mean, "ci95_low": float("nan"), "ci95_high": float("nan")}
    se = float(np.std(values, ddof=1) / math.sqrt(n))
    crit = float(t.ppf(0.975, n - 1))
    return {
        "mean": mean,
        "se": se,
        "ci95_low": mean - crit * se,
        "ci95_high": mean + crit * se,
        "fraction_negative": float(np.mean(values < 0.0)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sales = load_sales(args.sales)
    probabilities = load_probability_rows(args.responses)
    rows, persona_ids = build_prompting_design_rows(sales, probabilities)
    rows = rows[rows["demand"] > 0].copy().reset_index(drop=True)

    nb_records: list[dict] = []
    log_records: list[dict] = []

    for split_idx in range(args.n_splits):
        split_dir = args.saved_eval_dir / f"split_{split_idx:03d}"
        train_ids = pd.read_csv(split_dir / "train_products.csv")["article_id"].astype(int)
        test_ids = pd.read_csv(split_dir / "test_products.csv")["article_id"].astype(int)
        train = rows[rows["article_id"].isin(train_ids)].copy().reset_index(drop=True)
        test = rows[rows["article_id"].isin(test_ids)].copy().reset_index(drop=True)
        y_train = train["demand"].to_numpy(int)
        y_test = test["demand"].to_numpy(int)

        for lower_log_r in LOG_R_LOWER_BOUNDS:
            fit = fit_global_nb(y_train, lower_log_r)
            for sample, y in (("train", y_train), ("test", y_test)):
                nll = ztnb_direct_nll(y, np.full(len(y), fit["x"]), fit["r"])
                nb_records.append(
                    {
                        "split": split_idx,
                        "sample": sample,
                        "lower_log_r": lower_log_r,
                        "lower_r": float(np.exp(lower_log_r)),
                        "fitted_r": fit["r"],
                        "fitted_x": fit["x"],
                        "implied_untruncated_mu": fit["mu"],
                        "avg_nll": float(np.mean(nll)),
                        "optimizer_success": bool(fit["success"]),
                        "hit_lower_bound": bool(fit["hit_lower_bound"]),
                    }
                )

        global_fit = fit_logseries_global(y_train)
        llm_fit = fit_logseries_llm(y_train, pooled_llm_feature(train, persona_ids))
        for sample, sample_rows, y in (("train", train, y_train), ("test", test, y_test)):
            global_p = np.full(len(y), global_fit["p"], dtype=float)
            llm_p = predict_logseries_llm(llm_fit, pooled_llm_feature(sample_rows, persona_ids))
            for model_name, p in (("logseries-global", global_p), ("logseries-llm-pooled", llm_p)):
                metrics = score_logseries(y, p)
                log_records.append(
                    {
                        "split": split_idx,
                        "sample": sample,
                        "model": model_name,
                        **metrics,
                        "global_p": global_fit["p"],
                        "llm_slope": float(llm_fit["params"][1]),
                    }
                )
        print(f"completed split {split_idx:03d}", flush=True)

    nb_profile = pd.DataFrame(nb_records)
    log_scores = pd.DataFrame(log_records)
    nb_profile.to_csv(args.output_dir / "nb_lower_bound_profile.csv", index=False)
    log_scores.to_csv(args.output_dir / "logseries_scores.csv", index=False)

    nb_summary = (
        nb_profile.groupby(["sample", "lower_log_r"], as_index=False)
        .agg(
            mean_avg_nll=("avg_nll", "mean"),
            mean_fitted_r=("fitted_r", "mean"),
            median_fitted_r=("fitted_r", "median"),
            mean_implied_mu=("implied_untruncated_mu", "mean"),
            fraction_at_lower_bound=("hit_lower_bound", "mean"),
            optimizer_success_rate=("optimizer_success", "mean"),
        )
    )
    log_summary = (
        log_scores.groupby(["sample", "model"], as_index=False)
        .agg(
            avg_nll=("avg_nll", "mean"),
            avg_crps=("avg_crps", "mean"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            mean_prediction=("mean_prediction", "mean"),
            mean_p=("mean_p", "mean"),
            mean_llm_slope=("llm_slope", "mean"),
        )
    )
    nb_summary.to_csv(args.output_dir / "nb_lower_bound_summary.csv", index=False)
    log_summary.to_csv(args.output_dir / "logseries_summary.csv", index=False)

    test_log = log_scores[log_scores["sample"] == "test"].pivot(index="split", columns="model", values=["avg_nll", "avg_crps", "mae"])
    paired = {}
    for metric in ("avg_nll", "avg_crps", "mae"):
        delta = (
            test_log[(metric, "logseries-llm-pooled")].to_numpy(float)
            - test_log[(metric, "logseries-global")].to_numpy(float)
        )
        paired[metric] = paired_interval(delta)

    widest = nb_profile[(nb_profile["sample"] == "test") & (nb_profile["lower_log_r"] == min(LOG_R_LOWER_BOUNDS))]
    narrow = nb_profile[(nb_profile["sample"] == "test") & (nb_profile["lower_log_r"] == max(LOG_R_LOWER_BOUNDS))]
    merged = widest[["split", "avg_nll"]].merge(
        narrow[["split", "avg_nll"]], on="split", suffixes=("_wide", "_original")
    )
    boundary_delta = merged["avg_nll_wide"].to_numpy(float) - merged["avg_nll_original"].to_numpy(float)

    decision = {
        "experiment": "nb_zero_dispersion_limit",
        "n_splits": int(log_scores["split"].nunique()),
        "nb_test_nll_change_log_r_lower_-16_minus_-6": paired_interval(boundary_delta),
        "fraction_nb_fits_at_lower_bound_by_bound": {
            str(row.lower_log_r): float(row.fraction_at_lower_bound)
            for row in nb_summary[nb_summary["sample"] == "train"].itertuples(index=False)
        },
        "logseries_test_means": {
            row.model: {
                "nll": float(row.avg_nll),
                "crps": float(row.avg_crps),
                "mae": float(row.mae),
                "mean_prediction": float(row.mean_prediction),
            }
            for row in log_summary[log_summary["sample"] == "test"].itertuples(index=False)
        },
        "llm_minus_global_logseries_paired": paired,
        "interpretation_rule": (
            "If widening the NB lower-dispersion bound keeps lowering likelihood and the fitted model approaches the "
            "logarithmic-series limit, then 'Negative Binomial overdispersion' is not an identified finite-dispersion "
            "result. The positive-only benchmark is primarily identifying the conditional positive-count shape."
        ),
        "new_llm_calls": 0,
        "api_cost_usd": 0,
    }
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2))

    print("\nNB lower-bound profile:\n")
    print(nb_summary.to_string(index=False))
    print("\nLogarithmic-series summary:\n")
    print(log_summary.to_string(index=False))
    print("\nDecision diagnostic:\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
