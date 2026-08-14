from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import kstest, t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand_sim.config import DEFAULT_OUTPUT_DIR, DEFAULT_PRODUCTS_DIR, DEFAULT_RESPONSES_DIR
from demand_sim.data import build_prompting_design_rows, load_probability_rows, load_sales

ARMS = ("G", "P", "C", "R")
EXPECTED_GLOBAL_NLL = 1.49120037731


def logseries_nll(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -(y * np.log(p) - np.log(y) - np.log(-np.log1p(-p)))


def logseries_mean(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -p / ((1 - p) * np.log1p(-p))


def logseries_crps(y: np.ndarray, p: np.ndarray, support_max: int = 400) -> np.ndarray:
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


def logseries_midpit(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Deterministic mid-P PIT diagnostic for a discrete distribution."""
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    out = np.empty(len(y), dtype=float)
    for i, (yy, pp) in enumerate(zip(y, p)):
        k = np.arange(1, yy + 1, dtype=float)
        pmf = -(pp ** k) / (k * np.log1p(-pp))
        fy_minus = float(pmf[:-1].sum()) if yy > 1 else 0.0
        out[i] = fy_minus + 0.5 * float(pmf[-1])
    return out


def score_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = logseries_mean(p)
    pit = logseries_midpit(y, p)
    return {
        "avg_nll": float(logseries_nll(y, p).mean()),
        "avg_crps": float(logseries_crps(y, p).mean()),
        "mae": float(np.mean(np.abs(y - pred))),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "pit_ks": float(kstest(pit, "uniform").statistic),
        "mean_prediction": float(pred.mean()),
        "mean_p": float(p.mean()),
    }


def fit_global(y: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    def obj(v: np.ndarray) -> float:
        pp = float(expit(v[0]))
        return float(logseries_nll(y, np.full(len(y), pp)).sum())
    res = minimize(obj, np.array([2.0]), method="L-BFGS-B", bounds=[(-12.0, 12.0)],
                   options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-9})
    return {"p": float(expit(res.x[0])), "success": bool(res.success), "objective": float(res.fun)}


def fit_feature(y: np.ndarray, z: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    z = np.asarray(z, dtype=float)
    mean = float(z.mean())
    std = float(z.std())
    if std < 1e-12:
        std = 1.0
    zs = (z - mean) / std

    def obj(v: np.ndarray) -> float:
        p = expit(np.clip(v[0] + v[1] * zs, -12.0, 12.0))
        return float(logseries_nll(y, p).sum())

    # Include the b=0 nested solution as a candidate to avoid optimizer noise.
    global_fit = fit_global(y)
    global_logit = float(np.log(global_fit["p"] / (1 - global_fit["p"])))
    candidates = [(np.array([global_logit, 0.0]), global_fit["objective"], True, "nested-global")]
    for start in (np.array([global_logit, 0.05]), np.array([2.0, 0.2]), np.array([global_logit, 1.0])):
        res = minimize(obj, start, method="L-BFGS-B", bounds=[(-12.0, 12.0), (0.0, 12.0)],
                       options={"maxiter": 800, "ftol": 1e-12, "gtol": 1e-9})
        if np.isfinite(res.fun):
            candidates.append((np.asarray(res.x), float(res.fun), bool(res.success), str(res.message)))
    params, value, success, message = min(candidates, key=lambda x: x[1])
    return {
        "params": params, "z_mean": mean, "z_std": std,
        "success": success, "message": message, "objective": value,
    }


def predict_feature(fit: dict, z: np.ndarray) -> np.ndarray:
    zs = (np.asarray(z, dtype=float) - fit["z_mean"]) / fit["z_std"]
    return expit(np.clip(fit["params"][0] + fit["params"][1] * zs, -12.0, 12.0))


def paired_interval(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    mean = float(v.mean()) if n else float("nan")
    if n < 2:
        return {"n": n, "mean": mean, "ci95_low": float("nan"), "ci95_high": float("nan")}
    se = float(v.std(ddof=1) / math.sqrt(n))
    crit = float(t.ppf(0.975, n - 1))
    return {
        "n": n, "mean": mean, "se": se,
        "ci95_low": mean - crit * se, "ci95_high": mean + crit * se,
        "fraction_negative": float(np.mean(v < 0)),
    }


def split_bootstrap(values: np.ndarray, seed: int = 20260814, reps: int = 20000) -> dict:
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=float)
    n = len(v)
    for i in range(reps):
        means[i] = rng.choice(v, size=n, replace=True).mean()
    return {
        "reps": reps, "mean": float(v.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "fraction_bootstrap_below_zero": float(np.mean(means < 0)),
    }


def derive_score(rows: pd.DataFrame, latent: pd.DataFrame, arm: str) -> np.ndarray:
    block = latent[latent["arm"] == arm][["article_id", "appeal_ref", "price_sensitivity", "reference_price"]]
    merged = rows[["article_id", "offer_price"]].merge(block, on="article_id", how="left", validate="many_to_one")
    if merged[["appeal_ref", "price_sensitivity", "reference_price"]].isna().any().any():
        raise ValueError(f"Missing frozen latent for arm {arm}")
    score = merged["appeal_ref"].to_numpy(float) - merged["price_sensitivity"].to_numpy(float) * np.log(
        merged["offer_price"].to_numpy(float) / merged["reference_price"].to_numpy(float)
    )
    return np.clip(score, 0.0, 100.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sales", type=Path, default=DEFAULT_PRODUCTS_DIR / "sales_top100_online.csv")
    p.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES_DIR / "llm_responses_online_top100.csv")
    p.add_argument("--saved-eval-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "demand_prediction")
    p.add_argument("--latents", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-splits", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    latent = pd.read_csv(args.latents)
    required = {"article_id", "arm", "appeal_ref", "price_sensitivity", "reference_price"}
    if required - set(latent.columns):
        raise ValueError("Frozen latent file has wrong schema")
    if len(latent) != 400 or latent.article_id.nunique() != 100 or set(latent.arm) != set(ARMS):
        raise ValueError("Frozen latent file does not contain 100 articles x 4 arms")

    # From here onward targets are intentionally unblinded.
    sales = load_sales(args.sales)
    probabilities = load_probability_rows(args.responses)
    rows, _ = build_prompting_design_rows(sales, probabilities)
    rows = rows[rows["demand"] > 0].copy().reset_index(drop=True)
    if len(rows) != 11691:
        raise AssertionError(f"Expected historical benchmark row universe of 11,691; found {len(rows)}")

    for arm in ARMS:
        rows[f"score_{arm}"] = derive_score(rows, latent, arm)

    records = []
    slope_records = []
    split_sizes = []
    for split_idx in range(args.n_splits):
        split_dir = args.saved_eval_dir / f"split_{split_idx:03d}"
        train_ids = pd.read_csv(split_dir / "train_products.csv")["article_id"].astype(int)
        test_ids = pd.read_csv(split_dir / "test_products.csv")["article_id"].astype(int)
        train = rows[rows["article_id"].isin(train_ids)].copy().reset_index(drop=True)
        test = rows[rows["article_id"].isin(test_ids)].copy().reset_index(drop=True)
        y_train = train["demand"].to_numpy(int)
        y_test = test["demand"].to_numpy(int)
        split_sizes.append({"split": split_idx, "train_rows": len(train), "test_rows": len(test),
                            "train_articles": train.article_id.nunique(), "test_articles": test.article_id.nunique()})

        global_fit = fit_global(y_train)
        for sample_name, yy in (("train", y_train), ("test", y_test)):
            pp = np.full(len(yy), global_fit["p"])
            records.append({"split": split_idx, "sample": sample_name, "model": "Global", **score_metrics(yy, pp)})

        for arm in ARMS:
            z_train = train[f"score_{arm}"].to_numpy(float)
            z_test = test[f"score_{arm}"].to_numpy(float)
            fit = fit_feature(y_train, z_train)
            slope_records.append({
                "split": split_idx, "arm": arm, "intercept": float(fit["params"][0]),
                "slope": float(fit["params"][1]), "score_train_mean": fit["z_mean"],
                "score_train_std": fit["z_std"], "optimizer_success": bool(fit["success"]),
            })
            for sample_name, yy, zz in (("train", y_train, z_train), ("test", y_test, z_test)):
                pp = predict_feature(fit, zz)
                records.append({"split": split_idx, "sample": sample_name, "model": arm, **score_metrics(yy, pp)})
        print(f"completed split {split_idx:03d}", flush=True)

    metrics = pd.DataFrame(records)
    slopes = pd.DataFrame(slope_records)
    pd.DataFrame(split_sizes).to_csv(args.output_dir / "split_sizes.csv", index=False)
    metrics.to_csv(args.output_dir / "per_split_metrics.csv", index=False)
    slopes.to_csv(args.output_dir / "fitted_slopes.csv", index=False)

    means = metrics.groupby(["sample", "model"], as_index=False).agg(
        avg_nll=("avg_nll", "mean"), avg_crps=("avg_crps", "mean"), mae=("mae", "mean"),
        rmse=("rmse", "mean"), pit_ks=("pit_ks", "mean"), mean_prediction=("mean_prediction", "mean"),
        mean_p=("mean_p", "mean"),
    )
    means.to_csv(args.output_dir / "metric_means.csv", index=False)

    global_test = float(means[(means["sample"] == "test") & (means["model"] == "Global")]["avg_nll"].iloc[0])
    if abs(global_test - EXPECTED_GLOBAL_NLL) > 2e-6:
        raise AssertionError(f"Global baseline reproduction failed: {global_test} vs {EXPECTED_GLOBAL_NLL}")

    test = metrics[metrics["sample"] == "test"].pivot(index="split", columns="model", values=["avg_nll", "avg_crps", "mae", "rmse", "pit_ks"])
    comparisons = {}
    pairs = [
        ("G_minus_Global", "G", "Global"), ("P_minus_Global", "P", "Global"),
        ("C_minus_Global", "C", "Global"), ("R_minus_Global", "R", "Global"),
        ("R_minus_C", "R", "C"), ("R_minus_P", "R", "P"),
        ("C_minus_G", "C", "G"), ("R_minus_G", "R", "G"),
    ]
    for label, left, right in pairs:
        comparisons[label] = {}
        for metric in ("avg_nll", "avg_crps", "mae", "rmse", "pit_ks"):
            delta = test[(metric, left)].to_numpy() - test[(metric, right)].to_numpy()
            comparisons[label][metric] = {
                "t_interval": paired_interval(delta),
                "split_bootstrap": split_bootstrap(delta),
                "per_split": [float(x) for x in delta],
            }

    score_diag_rows = []
    for arm in ARMS:
        z = rows[f"score_{arm}"].to_numpy(float)
        score_diag_rows.append({
            "arm": arm, "row_score_mean": float(z.mean()), "row_score_std": float(z.std()),
            "row_score_min": float(z.min()), "row_score_max": float(z.max()),
            "mean_fitted_slope": float(slopes[slopes.arm == arm].slope.mean()),
            "median_fitted_slope": float(slopes[slopes.arm == arm].slope.median()),
            "fraction_slope_effectively_zero": float(np.mean(slopes[slopes.arm == arm].slope.to_numpy() < 1e-6)),
        })
    score_diag = pd.DataFrame(score_diag_rows)
    score_diag.to_csv(args.output_dir / "score_diagnostics.csv", index=False)

    # Product-level latent correlations and mechanical monotonicity.
    appeal_wide = latent.pivot(index="article_id", columns="arm", values="appeal_ref")
    sens_wide = latent.pivot(index="article_id", columns="arm", values="price_sensitivity")
    corr = {
        "appeal": appeal_wide.corr().to_dict(),
        "price_sensitivity": sens_wide.corr().to_dict(),
    }
    monotone = {}
    for arm in ARMS:
        ok = []
        for article_id, group in rows.groupby("article_id"):
            g = group.sort_values("offer_price")
            diff = np.diff(g[f"score_{arm}"].to_numpy(float))
            ok.append(bool(np.all(diff <= 1e-12)))
        monotone[arm] = float(np.mean(ok))

    nll_r_global = comparisons["R_minus_Global"]["avg_nll"]["t_interval"]
    nll_r_c = comparisons["R_minus_C"]["avg_nll"]["t_interval"]
    nll_r_p = comparisons["R_minus_P"]["avg_nll"]["t_interval"]
    decisions = {
        "claim_A_rich_beats_global": bool(nll_r_global["ci95_high"] < 0),
        "claim_B_rich_beats_matched_coarse": bool(nll_r_c["ci95_high"] < 0),
        "claim_C_rich_beats_paper_personas": bool(nll_r_p["ci95_high"] < 0),
    }

    summary = {
        "experiment": "004_gpt56_persona_richness",
        "row_universe": int(len(rows)),
        "global_baseline_reproduced_nll": global_test,
        "reference_global_nll": EXPECTED_GLOBAL_NLL,
        "paper_llm_mix_cal_reference_nll": 1.83460248617,
        "test_metric_means": means[means["sample"] == "test"].set_index("model").to_dict(orient="index"),
        "comparisons": comparisons,
        "decisions": decisions,
        "score_correlations": corr,
        "price_monotonicity_fraction_by_arm": monotone,
        "known_limitation": "Positive-demand-only H&M benchmark; results do not establish incidence, exposure, causal pricing, or virtual-population validity.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decisions": decisions, "test_metric_means": summary["test_metric_means"]}, indent=2))


if __name__ == "__main__":
    main()
