from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand_sim.config import DEFAULT_OUTPUT_DIR, DEFAULT_PRODUCTS_DIR, DEFAULT_RESPONSES_DIR
from demand_sim.data import build_prompting_design_rows, load_probability_rows, load_sales
from demand_sim.io import load_pickle
from demand_sim.models.llm_mix import truncated_binomial_nll


DEFAULT_N_GRID = [250, 350, 500, 750, 1000, 2000, 10000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fast practical-identifiability diagnostic for exposure N. Starting from each "
            "saved llm-mix-cal fit, preserve lambda=N*q and evaluate the zero-truncated "
            "Binomial likelihood as N grows toward the zero-truncated Poisson limit."
        )
    )
    parser.add_argument("--sales", type=Path, default=DEFAULT_PRODUCTS_DIR / "sales_top100_online.csv")
    parser.add_argument(
        "--responses",
        type=Path,
        default=DEFAULT_RESPONSES_DIR / "llm_responses_online_top100.csv",
    )
    parser.add_argument(
        "--saved-eval-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "demand_prediction",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "research" / "exposure_identifiability",
    )
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--n-grid", type=int, nargs="+", default=DEFAULT_N_GRID)
    return parser.parse_args()


def zt_poisson_avg_nll(demand: np.ndarray, lam: np.ndarray) -> float:
    demand = np.asarray(demand, dtype=int)
    lam = np.clip(np.asarray(lam, dtype=float), 1e-12, None)
    log_pmf = demand * np.log(lam) - lam - gammaln(demand + 1)
    zero_mass = np.exp(-lam)
    log_positive_mass = np.log1p(-np.clip(zero_mass, 0.0, 1.0 - 1e-15))
    return float(-np.mean(log_pmf - log_positive_mass))


def zt_binomial_avg_nll(demand: np.ndarray, exposure_n: int, q: np.ndarray) -> float:
    return float(truncated_binomial_nll(demand, int(exposure_n), q) / len(demand))


def binomial_conditional_moments(exposure_n: int, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.clip(np.asarray(q, dtype=float), 1e-12, 1.0 - 1e-12)
    mean = exposure_n * q
    var = exposure_n * q * (1.0 - q)
    zero_mass = np.exp(exposure_n * np.log1p(-q))
    positive_mass = np.clip(1.0 - zero_mass, 1e-12, None)
    cond_mean = mean / positive_mass
    cond_second = (var + mean**2) / positive_mass
    cond_var = np.maximum(cond_second - cond_mean**2, 0.0)
    return cond_mean, cond_var


def poisson_conditional_moments(lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lam = np.clip(np.asarray(lam, dtype=float), 1e-12, None)
    zero_mass = np.exp(-lam)
    positive_mass = np.clip(1.0 - zero_mass, 1e-12, None)
    cond_mean = lam / positive_mass
    cond_second = (lam + lam**2) / positive_mass
    cond_var = np.maximum(cond_second - cond_mean**2, 0.0)
    return cond_mean, cond_var


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_grid = sorted(set(int(n) for n in args.n_grid if int(n) > 0))

    sales = load_sales(args.sales)
    probabilities = load_probability_rows(args.responses)
    rows, _persona_ids = build_prompting_design_rows(sales, probabilities)

    records: list[dict] = []
    for split_idx in range(args.n_splits):
        split_dir = args.saved_eval_dir / f"split_{split_idx:03d}"
        model_path = split_dir / "models" / "llm-mix-cal.pkl"
        if not model_path.exists():
            continue
        model = load_pickle(model_path)
        train_ids = pd.read_csv(split_dir / "train_products.csv")["article_id"].astype(int)
        test_ids = pd.read_csv(split_dir / "test_products.csv")["article_id"].astype(int)

        for sample, ids in [("train", train_ids), ("test", test_ids)]:
            sample_rows = rows[rows["article_id"].isin(ids) & (rows["demand"] > 0)].copy()
            demand = sample_rows["demand"].to_numpy(int)
            base_q = model.purchase_probability(sample_rows)
            base_n = int(model.exposure_n)
            lam = base_n * base_q
            poisson_nll = zt_poisson_avg_nll(demand, lam)
            poisson_mean, poisson_var = poisson_conditional_moments(lam)

            for exposure_n in n_grid:
                q_scaled = lam / float(exposure_n)
                if np.any(q_scaled >= 1.0) or np.any(demand > exposure_n):
                    continue
                nll = zt_binomial_avg_nll(demand, exposure_n, q_scaled)
                cond_mean, cond_var = binomial_conditional_moments(exposure_n, q_scaled)
                records.append(
                    {
                        "split": split_idx,
                        "sample": sample,
                        "saved_base_n": base_n,
                        "candidate_n": exposure_n,
                        "avg_zt_nll": nll,
                        "poisson_limit_avg_zt_nll": poisson_nll,
                        "nll_minus_poisson_limit": nll - poisson_nll,
                        "mean_abs_conditional_mean_delta_vs_poisson": float(np.mean(np.abs(cond_mean - poisson_mean))),
                        "mean_abs_conditional_variance_delta_vs_poisson": float(np.mean(np.abs(cond_var - poisson_var))),
                        "mean_lambda": float(np.mean(lam)),
                        "max_lambda": float(np.max(lam)),
                        "n_observations": int(len(demand)),
                    }
                )

    profile = pd.DataFrame(records)
    if profile.empty:
        raise RuntimeError("No saved llm-mix-cal split models were available for the diagnostic")
    profile.to_csv(args.output_dir / "poisson_limit_profile.csv", index=False)

    base_rows = profile[profile["candidate_n"] == profile["saved_base_n"]][
        ["split", "sample", "avg_zt_nll"]
    ].rename(columns={"avg_zt_nll": "base_avg_zt_nll"})
    merged = profile.merge(base_rows, on=["split", "sample"], how="left")
    merged["nll_delta_vs_saved_base"] = merged["avg_zt_nll"] - merged["base_avg_zt_nll"]
    merged.to_csv(args.output_dir / "poisson_limit_profile_with_base_delta.csv", index=False)

    summary = (
        merged.groupby(["sample", "candidate_n"], as_index=False)
        .agg(
            mean_avg_zt_nll=("avg_zt_nll", "mean"),
            mean_nll_delta_vs_saved_base=("nll_delta_vs_saved_base", "mean"),
            mean_nll_minus_poisson_limit=("nll_minus_poisson_limit", "mean"),
            max_abs_nll_minus_poisson_limit=("nll_minus_poisson_limit", lambda s: float(np.max(np.abs(s)))),
            mean_abs_conditional_mean_delta_vs_poisson=("mean_abs_conditional_mean_delta_vs_poisson", "mean"),
            mean_abs_conditional_variance_delta_vs_poisson=("mean_abs_conditional_variance_delta_vs_poisson", "mean"),
            n_splits=("split", "nunique"),
        )
    )
    summary.to_csv(args.output_dir / "poisson_limit_summary.csv", index=False)

    test = summary[summary["sample"] == "test"].sort_values("candidate_n")
    high_n = test.iloc[-1]
    base = test[test["candidate_n"] == 250]
    decision = {
        "diagnostic": "fixed-rate Poisson-limit path",
        "n_splits": int(profile["split"].nunique()),
        "warning": (
            "This is a practical-identifiability diagnostic, not a replacement for the full refit profile. "
            "It preserves lambda=N*q from each saved N=250 model while varying N."
        ),
        "test_poisson_limit_gap_at_largest_n": float(high_n["mean_nll_minus_poisson_limit"]),
        "test_nll_delta_largest_n_vs_saved_base": (
            float(high_n["mean_avg_zt_nll"] - base.iloc[0]["mean_avg_zt_nll"]) if not base.empty else None
        ),
        "interpretation_rule": (
            "If the zero-truncated likelihood and conditional moments change only negligibly along the "
            "constant-rate path toward the Poisson limit, the data have little distributional leverage "
            "to distinguish finite exposure N from a rate model at the fitted purchase probabilities."
        ),
    }
    (args.output_dir / "poisson_limit_decision.json").write_text(json.dumps(decision, indent=2))

    print(summary.to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
