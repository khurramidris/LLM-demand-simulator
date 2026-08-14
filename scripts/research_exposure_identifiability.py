from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand_sim.config import DEFAULT_OUTPUT_DIR, DEFAULT_PRODUCTS_DIR, DEFAULT_RESPONSES_DIR
from demand_sim.data import build_prompting_design_rows, load_probability_rows, load_sales
from demand_sim.models.llm_mix import fit_llm_mix, truncated_binomial_nll
from demand_sim.models.llm_mix_cal import fit_llm_mix_cal


DEFAULT_EXPOSURE_GRID = [100, 150, 200, 250, 350, 500, 750, 1000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile the latent exposure parameter N for the LLM mixture models. "
            "This is a diagnostic experiment: every N is refit independently and "
            "evaluated on the same held-out product split."
        )
    )
    parser.add_argument("--sales", type=Path, default=DEFAULT_PRODUCTS_DIR / "sales_top100_online.csv")
    parser.add_argument(
        "--responses",
        type=Path,
        default=DEFAULT_RESPONSES_DIR / "llm_responses_online_top100.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "research" / "exposure_identifiability",
    )
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--exposure-grid", type=int, nargs="+", default=DEFAULT_EXPOSURE_GRID)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["llm-mix", "llm-mix-cal"],
        default=["llm-mix-cal"],
    )
    parser.add_argument("--calibration-iters", type=int, default=6)
    parser.add_argument("--solver", default=None)
    return parser.parse_args()


def split_products(product_ids: np.ndarray, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be between 0 and 1")
    rng = np.random.default_rng(seed)
    shuffled = np.array(product_ids, copy=True)
    rng.shuffle(shuffled)
    n_train = int(round(train_frac * len(shuffled)))
    n_train = min(max(n_train, 1), len(shuffled) - 1)
    return np.sort(shuffled[:n_train]), np.sort(shuffled[n_train:])


def conditional_mean(exposure_n: int, purchase_prob: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(purchase_prob, dtype=float), 1e-12, 1.0 - 1e-12)
    zero_mass = np.exp(exposure_n * np.log1p(-q))
    return exposure_n * q / np.clip(1.0 - zero_mass, 1e-12, None)


def score_rows(rows: pd.DataFrame, model) -> dict[str, float]:
    eval_rows = rows[rows["demand"] > 0].copy()
    demand = eval_rows["demand"].to_numpy(int)
    q = model.purchase_probability(eval_rows)
    nll = truncated_binomial_nll(demand, int(model.exposure_n), q)
    pred = conditional_mean(int(model.exposure_n), q)
    error = demand.astype(float) - pred
    return {
        "avg_zt_nll": float(nll / len(eval_rows)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mean_q": float(np.mean(q)),
        "mean_conditional_demand": float(np.mean(pred)),
    }


def alpha_diagnostics(model) -> dict[str, float]:
    alpha = np.asarray(model.alpha, dtype=float)
    real = np.clip(alpha[:-1], 0.0, None)
    real_mass = float(real.sum())
    if real_mass > 0:
        normalized = real / real_mass
        ess = float(1.0 / np.sum(normalized**2))
        entropy = float(-np.sum(normalized[normalized > 0] * np.log(normalized[normalized > 0])))
        max_weight = float(normalized.max())
        n_gt_1pct = int(np.sum(normalized >= 0.01))
    else:
        ess = 0.0
        entropy = 0.0
        max_weight = 0.0
        n_gt_1pct = 0
    return {
        "real_alpha_mass": real_mass,
        "dummy_alpha_mass": float(alpha[-1]),
        "persona_effective_n": ess,
        "persona_entropy": entropy,
        "largest_normalized_persona_weight": max_weight,
        "n_personas_ge_1pct": n_gt_1pct,
    }


def fit_one(model_name: str, rows: pd.DataFrame, persona_ids: list[str], exposure_n: int, args: argparse.Namespace):
    if model_name == "llm-mix":
        return fit_llm_mix(
            rows,
            persona_ids,
            [int(exposure_n)],
            fit_objective="truncated",
            solver=args.solver,
        )
    if model_name == "llm-mix-cal":
        return fit_llm_mix_cal(
            rows,
            persona_ids,
            [int(exposure_n)],
            fit_objective="truncated",
            calibration_iters=args.calibration_iters,
            solver=args.solver,
        )
    raise ValueError(model_name)


def version_metadata() -> dict:
    import scipy

    try:
        import cvxpy

        cvxpy_version = cvxpy.__version__
    except Exception:
        cvxpy_version = None

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "cvxpy": cvxpy_version,
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_ref": os.environ.get("GITHUB_REF"),
    }


def summarize_profile(profile: pd.DataFrame, exposure_grid: list[int]) -> pd.DataFrame:
    records: list[dict] = []
    max_n = max(exposure_grid)
    for (split, model), group in profile[profile["status"] == "ok"].groupby(["split", "model"], sort=True):
        group = group.sort_values("exposure_n")
        train_best = group.loc[group["train_avg_zt_nll"].idxmin()]
        test_best = group.loc[group["test_avg_zt_nll"].idxmin()]
        n250 = group[group["exposure_n"] == 250]
        nmax = group[group["exposure_n"] == max_n]
        record = {
            "split": int(split),
            "model": model,
            "train_best_n": int(train_best["exposure_n"]),
            "train_best_avg_zt_nll": float(train_best["train_avg_zt_nll"]),
            "test_best_n_diagnostic_only": int(test_best["exposure_n"]),
            "test_best_avg_zt_nll_diagnostic_only": float(test_best["test_avg_zt_nll"]),
            "train_best_hits_upper_boundary": bool(int(train_best["exposure_n"]) == max_n),
        }
        if not n250.empty and not nmax.empty:
            record["train_nll_delta_max_minus_250"] = float(
                nmax.iloc[0]["train_avg_zt_nll"] - n250.iloc[0]["train_avg_zt_nll"]
            )
            record["test_nll_delta_max_minus_250"] = float(
                nmax.iloc[0]["test_avg_zt_nll"] - n250.iloc[0]["test_avg_zt_nll"]
            )
        records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    if args.n_splits <= 0:
        raise SystemExit("--n-splits must be positive")
    exposure_grid = sorted(set(int(value) for value in args.exposure_grid if int(value) > 0))
    if not exposure_grid:
        raise SystemExit("--exposure-grid must contain positive integers")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = args.output_dir / "profile.csv"
    summary_path = args.output_dir / "split_summary.csv"
    metadata_path = args.output_dir / "metadata.json"

    sales = load_sales(args.sales)
    probabilities = load_probability_rows(args.responses)
    rows, persona_ids = build_prompting_design_rows(sales, probabilities)
    product_ids = np.array(sorted(rows["article_id"].unique()), dtype=int)

    metadata = {
        "experiment": "exposure_identifiability",
        "scientific_question": (
            "Is the latent Binomial exposure N identified, or does the likelihood continue "
            "to prefer the largest allowed value?"
        ),
        "sales": str(args.sales),
        "responses": str(args.responses),
        "n_rows": int(len(rows)),
        "n_products": int(len(product_ids)),
        "n_personas": int(len(persona_ids)),
        "n_splits": int(args.n_splits),
        "train_frac": float(args.train_frac),
        "seed": int(args.seed),
        "exposure_grid": exposure_grid,
        "models": args.models,
        "fit_objective": "truncated",
        "calibration_iters": int(args.calibration_iters),
        "software": version_metadata(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    completed: set[tuple[int, str, int]] = set()
    records: list[dict] = []
    if profile_path.exists():
        previous = pd.read_csv(profile_path)
        records.extend(previous.to_dict(orient="records"))
        completed = set(
            zip(
                previous["split"].astype(int),
                previous["model"].astype(str),
                previous["exposure_n"].astype(int),
            )
        )

    for split_idx in range(args.n_splits):
        train_ids, test_ids = split_products(product_ids, args.train_frac, args.seed + split_idx)
        train_rows = rows[rows["article_id"].isin(train_ids)].copy()
        test_rows = rows[rows["article_id"].isin(test_ids)].copy()
        max_observed = int(train_rows["demand"].max())

        for model_name in args.models:
            for exposure_n in exposure_grid:
                key = (split_idx, model_name, exposure_n)
                if key in completed:
                    continue
                base = {
                    "split": split_idx,
                    "model": model_name,
                    "exposure_n": exposure_n,
                    "n_train_products": int(len(train_ids)),
                    "n_test_products": int(len(test_ids)),
                    "n_train_rows": int(len(train_rows)),
                    "n_test_rows": int(len(test_rows)),
                    "max_train_demand": max_observed,
                }
                if exposure_n < max_observed:
                    records.append({**base, "status": "skipped_below_max_demand"})
                    pd.DataFrame(records).to_csv(profile_path, index=False)
                    continue

                try:
                    model = fit_one(model_name, train_rows, persona_ids, exposure_n, args)
                    train_scores = score_rows(train_rows, model)
                    test_scores = score_rows(test_rows, model)
                    result = {
                        **base,
                        "status": "ok",
                        "fit_objective_value": float(model.objective_value),
                        **{f"train_{k}": v for k, v in train_scores.items()},
                        **{f"test_{k}": v for k, v in test_scores.items()},
                        **alpha_diagnostics(model),
                    }
                    if hasattr(model, "intercept"):
                        result["logit_cal_intercept"] = float(model.intercept)
                        result["logit_cal_slope"] = float(model.slope)
                    records.append(result)
                except Exception as exc:
                    records.append({**base, "status": "error", "error": repr(exc)})

                pd.DataFrame(records).to_csv(profile_path, index=False)

    profile = pd.DataFrame(records)
    summary = summarize_profile(profile, exposure_grid)
    summary.to_csv(summary_path, index=False)

    if not summary.empty:
        boundary_rate = float(summary["train_best_hits_upper_boundary"].mean())
        decision = {
            "n_profiled_fits": int((profile["status"] == "ok").sum()),
            "upper_boundary_n": int(max(exposure_grid)),
            "fraction_of_split_models_with_train_optimum_at_upper_boundary": boundary_rate,
            "interpretation": (
                "Evidence against identifying N as exposure is strong if the profile optimum "
                "continues to sit at the upper boundary or is essentially flat over large N."
            ),
        }
        (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2))

    print(f"Wrote exposure profile to {profile_path}")
    print(f"Wrote split summary to {summary_path}")


if __name__ == "__main__":
    main()
