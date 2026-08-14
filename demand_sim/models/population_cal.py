from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .llm_mix import binomial_nll, truncated_binomial_nll


@dataclass
class PopulationCalModel:
    exposure_n: int
    weights: np.ndarray
    persona_ids: list[str]
    intercept: float
    slope: float
    fit_objective: str
    objective_value: float
    weight_mode: str

    @property
    def name(self) -> str:
        return self.weight_mode

    def purchase_probability(self, rows: pd.DataFrame) -> np.ndarray:
        q_raw = rows[self.persona_ids].to_numpy(float)
        q_cal = calibrate_probability_matrix(q_raw, self.intercept, self.slope)
        return np.clip(q_cal @ self.weights, 1e-9, 1.0 - 1e-9)

    def mean_demand(self, rows: pd.DataFrame) -> np.ndarray:
        return self.exposure_n * self.purchase_probability(rows)

    def alpha_table(self) -> pd.DataFrame:
        return pd.DataFrame({"persona_id": self.persona_ids, "alpha": self.weights})


def fit_population_cal(
    rows: pd.DataFrame,
    persona_ids: list[str],
    persona_table: pd.DataFrame,
    exposure_n_values: list[int],
    fit_objective: str = "truncated",
    weight_mode: str = "population-cal",
) -> PopulationCalModel:
    """
    Fit only a monotone logit calibration while holding population weights fixed.

    weight_mode="population-cal" uses observed persona cell counts.
    weight_mode="uniform-cal" gives every queried persona equal mass.
    """
    if weight_mode not in {"population-cal", "uniform-cal"}:
        raise ValueError("weight_mode must be population-cal or uniform-cal")

    fit_rows = rows[rows["demand"] > 0].copy() if fit_objective == "truncated" else rows.copy()
    if fit_rows.empty:
        raise ValueError("No rows available for fixed-population calibration.")

    weights = _population_weights(persona_ids, persona_table, weight_mode)
    q_raw = fit_rows[persona_ids].to_numpy(float)
    demand = fit_rows["demand"].to_numpy(int)

    best: PopulationCalModel | None = None
    for exposure_n in exposure_n_values:
        exposure_n = int(exposure_n)
        if exposure_n < int(demand.max()):
            continue
        intercept0, log_slope0 = _initial_calibration(q_raw, weights, demand, exposure_n)

        def objective(params: np.ndarray) -> float:
            intercept = float(params[0])
            slope = float(np.exp(params[1]))
            q_cal = calibrate_probability_matrix(q_raw, intercept, slope)
            q = np.clip(q_cal @ weights, 1e-9, 1.0 - 1e-9)
            if fit_objective == "truncated":
                return truncated_binomial_nll(demand, exposure_n, q)
            if fit_objective == "naive":
                return binomial_nll(demand, exposure_n, q)
            raise ValueError(f"Unknown fit objective: {fit_objective}")

        result = minimize(
            objective,
            x0=np.array([intercept0, log_slope0], dtype=float),
            method="L-BFGS-B",
            bounds=[(-20.0, 20.0), (-5.0, 5.0)],
            options={"maxiter": 300, "ftol": 1e-10},
        )
        candidate = PopulationCalModel(
            exposure_n=exposure_n,
            weights=weights.copy(),
            persona_ids=list(persona_ids),
            intercept=float(result.x[0]),
            slope=float(np.exp(result.x[1])),
            fit_objective=fit_objective,
            objective_value=float(result.fun),
            weight_mode=weight_mode,
        )
        if best is None or candidate.objective_value < best.objective_value:
            best = candidate

    if best is None:
        raise RuntimeError("No fixed-population calibrated model was fit.")
    return best


def calibrate_probability_matrix(q_matrix: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    clipped = np.clip(np.asarray(q_matrix, dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    return np.clip(expit(intercept + slope * logits), 1e-9, 1.0 - 1e-9)


def _population_weights(
    persona_ids: list[str],
    persona_table: pd.DataFrame,
    weight_mode: str,
) -> np.ndarray:
    if weight_mode == "uniform-cal":
        return np.full(len(persona_ids), 1.0 / len(persona_ids), dtype=float)

    required = {"persona_id", "n_customers"}
    missing = required.difference(persona_table.columns)
    if missing:
        raise ValueError(f"Persona table missing columns: {sorted(missing)}")

    counts = (
        persona_table.assign(persona_id=persona_table["persona_id"].astype(str))
        .set_index("persona_id")["n_customers"]
        .reindex(persona_ids)
    )
    if counts.isna().any():
        missing_ids = counts[counts.isna()].index.tolist()
        raise ValueError(f"Missing population counts for personas: {missing_ids[:10]}")
    weights = counts.to_numpy(float)
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Population counts must be non-negative with positive total.")
    return weights / weights.sum()


def _initial_calibration(
    q_raw: np.ndarray,
    weights: np.ndarray,
    demand: np.ndarray,
    exposure_n: int,
) -> tuple[float, float]:
    raw = np.clip(q_raw @ weights, 1e-6, 1.0 - 1e-6)
    raw_mean = float(raw.mean())
    target = float(np.clip(np.mean(demand) / exposure_n, 1e-6, 1.0 - 1e-6))
    raw_logit = math.log(raw_mean / (1.0 - raw_mean))
    target_logit = math.log(target / (1.0 - target))
    return target_logit - raw_logit, 0.0
