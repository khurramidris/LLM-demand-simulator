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
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand_sim.config import DEFAULT_OUTPUT_DIR, DEFAULT_PERSONAS_DIR, DEFAULT_PRODUCTS_DIR, DEFAULT_RESPONSES_DIR
from demand_sim.data import build_prompting_design_rows, load_probability_rows, load_sales
from demand_sim.io import load_pickle
from demand_sim.models.llm_mix import truncated_binomial_nll


TREATMENTS = ["pooled_raw_mean", "uniform_fixed_population", "empirical_fixed_population"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether the fitted 50-persona mixture earns its complexity. Fixed-population "
            "treatments get their own monotone calibration and one real-population mass parameter, "
            "but no free persona mixture weights."
        )
    )
    parser.add_argument("--sales", type=Path, default=DEFAULT_PRODUCTS_DIR / "sales_top100_online.csv")
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES_DIR / "llm_responses_online_top100.csv")
    parser.add_argument("--personas", type=Path, default=DEFAULT_PERSONAS_DIR / "persona_cells.csv")
    parser.add_argument("--saved-eval-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "demand_prediction")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "research" / "persona_necessity",
    )
    parser.add_argument("--n-splits", type=int, default=10)
    return parser.parse_args()


def logit(value: float) -> float:
    value = float(np.clip(value, 1e-6, 1.0 - 1e-6))
    return math.log(value / (1.0 - value))


def q_logits(q_raw: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(q_raw, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(q / (1.0 - q))


def conditional_mean(exposure_n: int, q: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(q, dtype=float), 1e-12, 1.0 - 1e-12)
    zero_mass = np.exp(exposure_n * np.log1p(-q))
    return exposure_n * q / np.clip(1.0 - zero_mass, 1e-12, None)


def score(demand: np.ndarray, q: np.ndarray, exposure_n: int) -> dict[str, float]:
    demand = np.asarray(demand, dtype=int)
    q = np.clip(np.asarray(q, dtype=float), 1e-12, 0.5)
    nll = truncated_binomial_nll(demand, exposure_n, q) / len(demand)
    pred = conditional_mean(exposure_n, q)
    err = demand.astype(float) - pred
    return {
        "avg_zt_nll": float(nll),
        "mae": float(np.mean(np.abs(err))),
        # This is a direct row-level RMSE diagnostic. The paper repository's official
        # RMSE is computed at pair level and then aggregated, so the two must not be
        # compared numerically. NLL and MAE aggregate linearly and reproduce the
        # committed control metrics exactly.
        "row_rmse": float(np.sqrt(np.mean(err**2))),
        "mean_q": float(np.mean(q)),
    }


def treatment_q(
    raw_q: np.ndarray,
    params: np.ndarray,
    treatment: str,
    weights: np.ndarray,
) -> np.ndarray:
    intercept = float(params[0])
    slope = float(np.exp(params[1]))
    mass = float(expit(params[2]))

    if treatment == "pooled_raw_mean":
        pooled = np.clip(raw_q @ weights, 1e-6, 1.0 - 1e-6)
        pooled_logit = np.log(pooled / (1.0 - pooled))
        calibrated = expit(intercept + slope * pooled_logit)
        return np.clip(mass * calibrated, 1e-12, 0.5)

    calibrated = expit(intercept + slope * q_logits(raw_q))
    population_q = calibrated @ weights
    return np.clip(mass * population_q, 1e-12, 0.5)


def optimize_treatment(
    raw_q: np.ndarray,
    demand: np.ndarray,
    exposure_n: int,
    treatment: str,
    weights: np.ndarray,
    saved_model,
) -> tuple[np.ndarray, float, bool, str]:
    saved_mass = float(np.clip(np.asarray(saved_model.alpha[:-1], dtype=float).sum(), 1e-5, 1.0 - 1e-5))
    saved_slope = float(max(saved_model.slope, 1e-6))
    starts = [
        np.array([float(saved_model.intercept), math.log(saved_slope), logit(saved_mass)]),
        np.array([-4.5, math.log(0.15), logit(0.95)]),
        np.array([-3.0, math.log(0.5), logit(0.5)]),
        np.array([0.0, 0.0, logit(0.05)]),
    ]
    bounds = [(-20.0, 20.0), (-5.0, 5.0), (-10.0, 10.0)]

    def objective(params: np.ndarray) -> float:
        q = treatment_q(raw_q, params, treatment, weights)
        return float(truncated_binomial_nll(demand, exposure_n, q))

    best = None
    for start in starts:
        result = minimize(
            objective,
            x0=start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 250, "ftol": 1e-10},
        )
        candidates = [
            (start, objective(start), True, "initial"),
            (result.x, float(result.fun), bool(result.success), str(result.message)),
        ]
        for params, value, success, message in candidates:
            if np.isfinite(value) and (best is None or value < best[1]):
                best = (np.asarray(params, dtype=float), float(value), success, message)
    if best is None:
        raise RuntimeError(f"No finite fit for treatment={treatment}")
    return best


def empirical_weights(persona_path: Path, persona_ids: list[str]) -> np.ndarray:
    cells = pd.read_csv(persona_path)
    cells["persona_id"] = cells["persona_id"].astype(str)
    counts = cells.set_index("persona_id")["n_customers"]
    values = np.array([float(counts.loc[pid]) for pid in persona_ids], dtype=float)
    return values / values.sum()


def control_q(rows: pd.DataFrame, model) -> np.ndarray:
    return model.purchase_probability(rows)


def fitted_alpha_diagnostics(model) -> dict[str, float]:
    real = np.clip(np.asarray(model.alpha[:-1], dtype=float), 0.0, None)
    mass = float(real.sum())
    if mass <= 0:
        return {"control_real_mass": 0.0, "control_persona_ess": 0.0, "control_n_personas_ge_1pct": 0}
    w = real / mass
    return {
        "control_real_mass": mass,
        "control_persona_ess": float(1.0 / np.sum(w**2)),
        "control_n_personas_ge_1pct": int(np.sum(w >= 0.01)),
        "control_largest_persona_weight": float(w.max()),
    }


def paired_uncertainty(compared: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for (sample, model), group in compared.groupby(["sample", "model"], sort=True):
        if model == "control_free_50_persona":
            continue
        for metric, delta_col in [
            ("avg_zt_nll", "nll_delta_vs_control"),
            ("mae", "mae_delta_vs_control"),
            ("row_rmse", "row_rmse_delta_vs_control"),
        ]:
            values = group[delta_col].dropna().to_numpy(float)
            n = len(values)
            mean = float(np.mean(values)) if n else np.nan
            std = float(np.std(values, ddof=1)) if n > 1 else 0.0
            se = float(std / np.sqrt(n)) if n else np.nan
            critical = float(student_t.ppf(0.975, n - 1)) if n > 1 else 0.0
            halfwidth = critical * se if n > 1 else 0.0
            records.append(
                {
                    "sample": sample,
                    "model": model,
                    "metric": metric,
                    "n_splits": n,
                    "mean_delta_vs_control": mean,
                    "std_delta": std,
                    "se_delta": se,
                    "ci95_low": mean - halfwidth,
                    "ci95_high": mean + halfwidth,
                    "fraction_splits_better_than_control": float(np.mean(values < 0)) if n else np.nan,
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sales = load_sales(args.sales)
    probabilities = load_probability_rows(args.responses)
    rows, _ = build_prompting_design_rows(sales, probabilities)

    records: list[dict] = []
    fit_records: list[dict] = []

    for split_idx in range(args.n_splits):
        split_dir = args.saved_eval_dir / f"split_{split_idx:03d}"
        model_path = split_dir / "models" / "llm-mix-cal.pkl"
        if not model_path.exists():
            continue
        model = load_pickle(model_path)
        persona_ids = list(model.persona_ids)
        uniform = np.full(len(persona_ids), 1.0 / len(persona_ids), dtype=float)
        empirical = empirical_weights(args.personas, persona_ids)
        treatment_weights = {
            "pooled_raw_mean": uniform,
            "uniform_fixed_population": uniform,
            "empirical_fixed_population": empirical,
        }

        train_ids = pd.read_csv(split_dir / "train_products.csv")["article_id"].astype(int)
        test_ids = pd.read_csv(split_dir / "test_products.csv")["article_id"].astype(int)
        train_rows = rows[rows["article_id"].isin(train_ids) & (rows["demand"] > 0)].copy()
        test_rows = rows[rows["article_id"].isin(test_ids) & (rows["demand"] > 0)].copy()

        train_raw = train_rows[persona_ids].to_numpy(float)
        test_raw = test_rows[persona_ids].to_numpy(float)
        train_demand = train_rows["demand"].to_numpy(int)
        test_demand = test_rows["demand"].to_numpy(int)
        exposure_n = int(model.exposure_n)

        control_diag = fitted_alpha_diagnostics(model)
        for sample, sample_rows, sample_demand in [
            ("train", train_rows, train_demand),
            ("test", test_rows, test_demand),
        ]:
            q = control_q(sample_rows, model)
            metrics = score(sample_demand, q, exposure_n)
            records.append(
                {
                    "split": split_idx,
                    "sample": sample,
                    "model": "control_free_50_persona",
                    "exposure_n": exposure_n,
                    "n_fit_parameters_approx": 52,
                    **metrics,
                    **control_diag,
                }
            )

        for treatment in TREATMENTS:
            weights = treatment_weights[treatment]
            params, objective, success, message = optimize_treatment(
                train_raw,
                train_demand,
                exposure_n,
                treatment,
                weights,
                model,
            )
            fitted_mass = float(expit(params[2]))
            fit_records.append(
                {
                    "split": split_idx,
                    "model": treatment,
                    "exposure_n": exposure_n,
                    "train_objective": objective,
                    "intercept": float(params[0]),
                    "slope": float(np.exp(params[1])),
                    "real_population_mass": fitted_mass,
                    "optimizer_success": bool(success),
                    "optimizer_message": message,
                }
            )
            for sample, raw_q, demand in [
                ("train", train_raw, train_demand),
                ("test", test_raw, test_demand),
            ]:
                q = treatment_q(raw_q, params, treatment, weights)
                metrics = score(demand, q, exposure_n)
                records.append(
                    {
                        "split": split_idx,
                        "sample": sample,
                        "model": treatment,
                        "exposure_n": exposure_n,
                        "n_fit_parameters_approx": 3,
                        **metrics,
                    }
                )

    scores = pd.DataFrame(records)
    fits = pd.DataFrame(fit_records)
    if scores.empty:
        raise RuntimeError("No completed saved split models found")
    scores.to_csv(args.output_dir / "persona_necessity_scores.csv", index=False)
    fits.to_csv(args.output_dir / "persona_necessity_fits.csv", index=False)

    control = scores[scores["model"] == "control_free_50_persona"][
        ["split", "sample", "avg_zt_nll", "mae", "row_rmse"]
    ].rename(
        columns={
            "avg_zt_nll": "control_avg_zt_nll",
            "mae": "control_mae",
            "row_rmse": "control_row_rmse",
        }
    )
    compared = scores.merge(control, on=["split", "sample"], how="left")
    compared["nll_delta_vs_control"] = compared["avg_zt_nll"] - compared["control_avg_zt_nll"]
    compared["mae_delta_vs_control"] = compared["mae"] - compared["control_mae"]
    compared["row_rmse_delta_vs_control"] = compared["row_rmse"] - compared["control_row_rmse"]
    compared.to_csv(args.output_dir / "persona_necessity_scores_with_control_delta.csv", index=False)

    summary = (
        compared.groupby(["sample", "model"], as_index=False)
        .agg(
            mean_avg_zt_nll=("avg_zt_nll", "mean"),
            mean_nll_delta_vs_control=("nll_delta_vs_control", "mean"),
            mean_mae=("mae", "mean"),
            mean_mae_delta_vs_control=("mae_delta_vs_control", "mean"),
            mean_row_rmse=("row_rmse", "mean"),
            mean_row_rmse_delta_vs_control=("row_rmse_delta_vs_control", "mean"),
            n_splits=("split", "nunique"),
        )
    )
    summary.to_csv(args.output_dir / "persona_necessity_summary.csv", index=False)

    uncertainty = paired_uncertainty(compared)
    uncertainty.to_csv(args.output_dir / "persona_necessity_paired_uncertainty.csv", index=False)

    test = summary[summary["sample"] == "test"].copy()
    ctl = float(test.loc[test["model"] == "control_free_50_persona", "mean_avg_zt_nll"].iloc[0])
    decisions = {}
    for treatment in TREATMENTS:
        row = test[test["model"] == treatment].iloc[0]
        gap = float(row["mean_avg_zt_nll"] - ctl)
        rel_gap_pct = 100.0 * gap / ctl
        nll_ci = uncertainty[
            (uncertainty["sample"] == "test")
            & (uncertainty["model"] == treatment)
            & (uncertainty["metric"] == "avg_zt_nll")
        ].iloc[0]
        decisions[treatment] = {
            "test_mean_avg_zt_nll": float(row["mean_avg_zt_nll"]),
            "absolute_nll_gap_vs_control": gap,
            "relative_nll_gap_pct_vs_control": rel_gap_pct,
            "paired_nll_ci95_low": float(nll_ci["ci95_low"]),
            "paired_nll_ci95_high": float(nll_ci["ci95_high"]),
            "fraction_splits_better_than_control": float(nll_ci["fraction_splits_better_than_control"]),
        }

    control_test_rows = compared[(compared["sample"] == "test") & (compared["model"] == "control_free_50_persona")]
    decision = {
        "experiment": "persona_necessity_fast_diagnostic",
        "n_splits": int(scores["split"].nunique()),
        "new_llm_calls": 0,
        "control_test_mean_avg_zt_nll": ctl,
        "control_mean_effective_personas": float(control_test_rows["control_persona_ess"].mean()),
        "control_mean_personas_ge_1pct": float(control_test_rows["control_n_personas_ge_1pct"].mean()),
        "treatments": decisions,
        "interpretation_rule": (
            "A fixed or pooled 3-parameter population model within about 1% of the free 50-persona "
            "control would mean the current benchmark does not establish that freely fitted persona "
            "heterogeneity is necessary. A >2% stable held-out NLL gap would be evidence that the "
            "persona-response basis plus learned mixture weights adds meaningful predictive information."
        ),
        "warning": (
            "This experiment fixes N to the saved control value so population aggregation is the major changed variable. "
            "It tests fixed population aggregation with fresh calibration; it is not yet a full K-persona refit ladder. "
            "row_rmse is a row-level diagnostic and is not the repository's official pair-aggregated RMSE."
        ),
    }
    (args.output_dir / "persona_necessity_decision.json").write_text(json.dumps(decision, indent=2))

    print(summary.to_string(index=False))
    print("\nPaired uncertainty:\n")
    print(uncertainty.to_string(index=False))
    print("\nDecision:\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
