from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import digamma, gammaln
from scipy.stats import nbinom, poisson, t
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand_sim.config import DEFAULT_EMBEDDINGS_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_PRODUCTS_DIR, DEFAULT_RESPONSES_DIR
from demand_sim.data import build_prompting_design_rows, load_probability_rows, load_product_embeddings, load_sales
from demand_sim.metrics import summarize_pair_scores


L2_GRID = (0.1, 1.0, 10.0)
NON_LLM_MODELS = (
    "ztp-global",
    "ztnb-global",
    "ztp-price-spline",
    "ztnb-price-spline",
    "ztp-siglip-price",
    "ztnb-siglip-price",
)
LLM_MODELS = ("ztp-llm-pooled", "ztnb-llm-pooled")
ALL_NEW_MODELS = NON_LLM_MODELS + LLM_MODELS
METRICS = (
    "zt_avg_nll",
    "zt_avg_crps",
    "mae",
    "rmse",
    "zt_pit_ks",
    "zt_interval_score_0.9",
    "zt_interval_score_0.95",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 003: count-rate baseline ladder.")
    parser.add_argument("--sales", type=Path, default=DEFAULT_PRODUCTS_DIR / "sales_top100_online.csv")
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES_DIR / "llm_responses_online_top100.csv")
    parser.add_argument(
        "--product-embeddings",
        type=Path,
        default=DEFAULT_EMBEDDINGS_DIR / "siglip2_product_embeddings.csv",
    )
    parser.add_argument("--saved-eval-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "demand_prediction")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "research" / "count_rate_baselines",
    )
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--crps-tail-prob", type=float, default=1e-7)
    parser.add_argument("--max-crps-support", type=int, default=2000)
    return parser.parse_args()


def clip_mu(mu: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(mu, dtype=float), 1e-8, 1e5)


def ztp_objective(
    beta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    l2: float,
) -> tuple[float, np.ndarray]:
    eta = np.clip(X @ beta, -18.0, 18.0)
    mu = np.exp(eta)
    log_positive_mass = np.log(-np.expm1(-mu))
    nll_rows = mu - y * eta + gammaln(y + 1.0) + log_positive_mass
    objective = float(np.sum(nll_rows) + 0.5 * l2 * np.sum(beta[1:] ** 2))

    ratio = np.zeros_like(mu)
    stable = mu < 50.0
    ratio[stable] = mu[stable] / np.expm1(mu[stable])
    grad_eta = mu - y + ratio
    grad = X.T @ grad_eta
    grad[1:] += l2 * beta[1:]
    return objective, np.asarray(grad, dtype=float)


def ztnb_objective(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    l2: float,
) -> tuple[float, np.ndarray]:
    beta = params[:-1]
    log_r = float(params[-1])
    r = float(np.exp(np.clip(log_r, -6.0, 8.0)))
    eta = np.clip(X @ beta, -18.0, 18.0)
    mu = np.exp(eta)

    log_r_plus_mu = np.log(r + mu)
    log_pmf = (
        gammaln(y + r)
        - gammaln(r)
        - gammaln(y + 1.0)
        + r * (math.log(r) - log_r_plus_mu)
        + y * (eta - log_r_plus_mu)
    )
    log_p0 = r * (math.log(r) - log_r_plus_mu)
    p0 = np.exp(np.clip(log_p0, -745.0, -1e-14))
    log_positive_mass = np.log1p(-p0)
    nll_rows = -log_pmf + log_positive_mass
    objective = float(np.sum(nll_rows) + 0.5 * l2 * np.sum(beta[1:] ** 2))

    p0_ratio = 1.0 / np.expm1(np.clip(-log_p0, 1e-12, 700.0))
    grad_eta = mu * (r + y) / (r + mu) - y + mu * r * p0_ratio / (r + mu)
    grad_beta = X.T @ grad_eta
    grad_beta[1:] += l2 * beta[1:]

    dlogpmf_dr = (
        digamma(y + r)
        - digamma(r)
        + math.log(r)
        - log_r_plus_mu
        + 1.0
        - (r + y) / (r + mu)
    )
    dlogp0_dr = math.log(r) - log_r_plus_mu + 1.0 - r / (r + mu)
    dtruncated_loglik_dr = dlogpmf_dr + p0_ratio * dlogp0_dr
    grad_log_r = float(-r * np.sum(dtruncated_loglik_dr))
    return objective, np.concatenate([np.asarray(grad_beta, dtype=float), [grad_log_r]])


def fit_ztp(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 0.0,
    slope_nonnegative: bool = False,
) -> dict:
    init = np.zeros(X.shape[1], dtype=float)
    init[0] = math.log(max(float(np.mean(y)), 1e-3))
    bounds = [(-18.0, 18.0)] * len(init)
    if slope_nonnegative and len(init) >= 2:
        bounds[1] = (0.0, 18.0)

    result = minimize(
        lambda b: ztp_objective(b, X, y, l2),
        init,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-7},
    )
    return {
        "beta": np.asarray(result.x, dtype=float),
        "objective": float(result.fun),
        "success": bool(result.success),
        "message": str(result.message),
        "dispersion": np.inf,
    }


def fit_ztnb(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 0.0,
    slope_nonnegative: bool = False,
) -> dict:
    poisson_fit = fit_ztp(X, y, l2=l2, slope_nonnegative=slope_nonnegative)
    init = np.concatenate([poisson_fit["beta"], [math.log(5.0)]])
    bounds = [(-18.0, 18.0)] * X.shape[1] + [(-6.0, 8.0)]
    if slope_nonnegative and X.shape[1] >= 2:
        bounds[1] = (0.0, 18.0)

    result = minimize(
        lambda p: ztnb_objective(p, X, y, l2),
        init,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 700, "ftol": 1e-11, "gtol": 1e-7},
    )
    return {
        "beta": np.asarray(result.x[:-1], dtype=float),
        "objective": float(result.fun),
        "success": bool(result.success),
        "message": str(result.message),
        "dispersion": float(np.exp(result.x[-1])),
    }


def predict_mu(fit: dict, X: np.ndarray) -> np.ndarray:
    return clip_mu(np.exp(np.clip(X @ fit["beta"], -18.0, 18.0)))


def family_logpmf_and_p0(
    y: np.ndarray,
    mu: np.ndarray,
    family: str,
    dispersion: float,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=int)
    mu = clip_mu(mu)
    if family == "poisson":
        logpmf = y * np.log(mu) - mu - gammaln(y + 1.0)
        p0 = np.exp(-mu)
        return logpmf, p0
    r = float(dispersion)
    log_r_plus_mu = np.log(r + mu)
    logpmf = (
        gammaln(y + r)
        - gammaln(r)
        - gammaln(y + 1.0)
        + r * (math.log(r) - log_r_plus_mu)
        + y * (np.log(mu) - log_r_plus_mu)
    )
    log_p0 = r * (math.log(r) - log_r_plus_mu)
    return logpmf, np.exp(np.clip(log_p0, -745.0, -1e-14))


def raw_cdf(k: np.ndarray, mu: np.ndarray, family: str, dispersion: float) -> np.ndarray:
    if family == "poisson":
        return poisson.cdf(k, mu)
    r = float(dispersion)
    p = r / (r + mu)
    return nbinom.cdf(k, r, p)


def raw_ppf(tau: np.ndarray, mu: np.ndarray, family: str, dispersion: float) -> np.ndarray:
    if family == "poisson":
        return poisson.ppf(tau, mu)
    r = float(dispersion)
    p = r / (r + mu)
    return nbinom.ppf(tau, r, p)


def conditional_mean(mu: np.ndarray, p0: np.ndarray) -> np.ndarray:
    return np.asarray(mu, dtype=float) / np.clip(1.0 - p0, 1e-12, None)


def crps_support_max(
    mu: np.ndarray,
    y: np.ndarray,
    family: str,
    dispersion: float,
    tail_prob: float,
    cap: int,
) -> int:
    tau = 1.0 - float(tail_prob)
    if family == "poisson":
        quant = poisson.ppf(tau, float(np.max(mu)))
    else:
        r = float(dispersion)
        p = r / (r + np.asarray(mu, dtype=float))
        quant = float(np.nanmax(nbinom.ppf(tau, r, p)))
    if not np.isfinite(quant):
        quant = cap
    return int(min(cap, max(400, int(np.max(y)) + 20, int(math.ceil(float(quant))))))


def zero_truncated_crps(
    y: np.ndarray,
    mu: np.ndarray,
    family: str,
    dispersion: float,
    p0: np.ndarray,
    support_max: int,
    chunk_size: int = 512,
) -> tuple[np.ndarray, float]:
    y = np.asarray(y, dtype=int)
    mu = np.asarray(mu, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    support = np.arange(1, support_max + 1, dtype=int)
    values = np.empty(len(y), dtype=float)
    max_tail = 0.0

    for start in range(0, len(y), chunk_size):
        stop = min(start + chunk_size, len(y))
        m = mu[start:stop]
        z0 = p0[start:stop]
        raw = raw_cdf(support[None, :], m[:, None], family, dispersion)
        zcdf = np.clip((raw - z0[:, None]) / np.clip(1.0 - z0[:, None], 1e-12, None), 0.0, 1.0)
        obs = y[start:stop, None]
        indicator = (support[None, :] >= obs).astype(float)
        values[start:stop] = np.sum((zcdf - indicator) ** 2, axis=1)
        max_tail = max(max_tail, float(np.max(1.0 - zcdf[:, -1])))
    return values, max_tail


def pit_ks(values: np.ndarray) -> float:
    sorted_pit = np.sort(np.asarray(values, dtype=float))
    n = len(sorted_pit)
    if n == 0:
        return float("nan")
    upper = np.abs(np.arange(1, n + 1) / n - sorted_pit)
    lower = np.abs(sorted_pit - np.arange(0, n) / n)
    return float(np.max(np.maximum(upper, lower)))


def score_count_model(
    rows: pd.DataFrame,
    model_name: str,
    mu: np.ndarray,
    family: str,
    dispersion: float,
    split_label: str,
    seed: int,
    tail_prob: float,
    support_cap: int,
    full_distribution_metrics: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    working = rows[["article_id", "offer_price", "demand"]].copy().reset_index(drop=True)
    y = working["demand"].to_numpy(int)
    mu = clip_mu(mu)
    logpmf, p0 = family_logpmf_and_p0(y, mu, family, dispersion)
    row_nll = -logpmf + np.log1p(-p0)
    pred = conditional_mean(mu, p0)

    if full_distribution_metrics:
        cdf_y = raw_cdf(y, mu, family, dispersion)
        cdf_below = raw_cdf(y - 1, mu, family, dispersion)
        denom = np.clip(1.0 - p0, 1e-12, None)
        zcdf_y = np.clip((cdf_y - p0) / denom, 0.0, 1.0)
        zcdf_below = np.clip((cdf_below - p0) / denom, 0.0, 1.0)
        rng = np.random.default_rng(seed)
        pit = zcdf_below + rng.uniform(size=len(y)) * np.maximum(zcdf_y - zcdf_below, 0.0)

        support_max = crps_support_max(mu, y, family, dispersion, tail_prob, support_cap)
        row_crps, max_tail = zero_truncated_crps(
            y, mu, family, dispersion, p0, support_max=support_max
        )

        interval_scores: dict[str, np.ndarray] = {}
        for level in (0.9, 0.95):
            alpha = 1.0 - level
            low_target = p0 + (alpha / 2.0) * (1.0 - p0)
            high_target = p0 + (1.0 - alpha / 2.0) * (1.0 - p0)
            lower = np.maximum(1.0, raw_ppf(low_target, mu, family, dispersion))
            upper = np.maximum(1.0, raw_ppf(high_target, mu, family, dispersion))
            score = (
                upper
                - lower
                + (2.0 / alpha) * (lower - y) * (y < lower)
                + (2.0 / alpha) * (y - upper) * (y > upper)
            )
            interval_scores[f"zt_interval_score_{level:g}"] = np.asarray(score, dtype=float)
    else:
        pit = np.full(len(y), np.nan, dtype=float)
        row_crps = np.full(len(y), np.nan, dtype=float)
        support_max = 0
        max_tail = float("nan")
        interval_scores = {
            "zt_interval_score_0.9": np.full(len(y), np.nan, dtype=float),
            "zt_interval_score_0.95": np.full(len(y), np.nan, dtype=float),
        }

    working["row_nll"] = row_nll
    working["row_crps"] = row_crps
    working["pit"] = pit
    working["abs_err"] = np.abs(y - pred)
    working["sq_err"] = (y - pred) ** 2
    for col, values in interval_scores.items():
        working[col] = values

    records: list[dict] = []
    for (article_id, offer_price), group in working.groupby(["article_id", "offer_price"], sort=True):
        records.append(
            {
                "split": split_label,
                "model": model_name,
                "article_id": int(article_id),
                "offer_price": float(offer_price),
                "n_observations": int(len(group)),
                "mae": float(group["abs_err"].mean()),
                "rmse": float(np.sqrt(group["sq_err"].mean())),
                "zt_avg_nll": float(group["row_nll"].mean()),
                "zt_avg_crps": float(group["row_crps"].mean()),
                "zt_pit_ks": pit_ks(group["pit"].to_numpy(float)),
                "zt_interval_score_0.9": float(group["zt_interval_score_0.9"].mean()),
                "zt_interval_score_0.95": float(group["zt_interval_score_0.95"].mean()),
            }
        )
    pair_scores = pd.DataFrame(records)
    summary = summarize_pair_scores(pair_scores)
    diagnostics = {
        "support_max": int(support_max),
        "max_positive_tail_mass_at_support": float(max_tail),
        "mean_untruncated_rate": float(np.mean(mu)),
        "mean_positive_conditional_prediction": float(np.mean(pred)),
    }
    return pair_scores, summary, diagnostics


class DesignBuilder:
    def __init__(
        self,
        kind: str,
        embedding_df: pd.DataFrame,
        embedding_cols: list[str],
        persona_ids: list[str],
        pca_components: int,
    ):
        self.kind = kind
        self.embedding_df = embedding_df
        self.embedding_cols = embedding_cols
        self.persona_ids = persona_ids
        self.pca_components = pca_components
        self.spline: SplineTransformer | None = None
        self.pca: PCA | None = None
        self.scaler: StandardScaler | None = None

    def fit(self, rows: pd.DataFrame) -> "DesignBuilder":
        if self.kind == "global":
            return self
        if self.kind == "llm-pooled":
            z = self._llm_feature(rows)
            self.scaler = StandardScaler().fit(z)
            return self

        price = rows[["offer_price"]].to_numpy(float)
        self.spline = SplineTransformer(
            n_knots=5,
            degree=3,
            include_bias=False,
            knots="quantile",
        ).fit(price)
        blocks = [self.spline.transform(price)]

        if self.kind == "siglip-price":
            ids = rows["article_id"].astype(int).unique()
            unique_emb = self.embedding_df[self.embedding_df["article_id"].isin(ids)]
            unique_emb = unique_emb.drop_duplicates("article_id").sort_values("article_id")
            n_components = min(self.pca_components, len(unique_emb) - 1, len(self.embedding_cols))
            if n_components < 1:
                raise ValueError("Not enough training products for PCA.")
            self.pca = PCA(n_components=n_components, random_state=17)
            self.pca.fit(unique_emb[self.embedding_cols].to_numpy(float))
            blocks.insert(0, self._pca_rows(rows))

        raw = np.hstack(blocks)
        self.scaler = StandardScaler().fit(raw)
        return self

    def transform(self, rows: pd.DataFrame) -> np.ndarray:
        if self.kind == "global":
            return np.ones((len(rows), 1), dtype=float)
        if self.kind == "llm-pooled":
            assert self.scaler is not None
            raw = self.scaler.transform(self._llm_feature(rows))
            return np.column_stack([np.ones(len(rows), dtype=float), raw])

        assert self.spline is not None and self.scaler is not None
        price_block = self.spline.transform(rows[["offer_price"]].to_numpy(float))
        blocks = [price_block]
        if self.kind == "siglip-price":
            blocks.insert(0, self._pca_rows(rows))
        raw = self.scaler.transform(np.hstack(blocks))
        return np.column_stack([np.ones(len(rows), dtype=float), raw])

    def _llm_feature(self, rows: pd.DataFrame) -> np.ndarray:
        pooled = np.clip(rows[self.persona_ids].to_numpy(float).mean(axis=1), 1e-6, 1.0 - 1e-6)
        z = np.log(pooled / (1.0 - pooled))
        return z.reshape(-1, 1)

    def _pca_rows(self, rows: pd.DataFrame) -> np.ndarray:
        assert self.pca is not None
        merged = rows[["article_id"]].merge(
            self.embedding_df[["article_id"] + self.embedding_cols],
            on="article_id",
            how="left",
            validate="many_to_one",
        )
        if merged[self.embedding_cols].isna().any().any():
            raise ValueError("Missing product embedding for at least one design row.")
        return self.pca.transform(merged[self.embedding_cols].to_numpy(float))


def model_parts(model_name: str) -> tuple[str, str]:
    family = "nb" if model_name.startswith("ztnb-") else "poisson"
    suffix = model_name.split("-", 1)[1]
    if suffix == "global":
        kind = "global"
    elif suffix == "price-spline":
        kind = "price-spline"
    elif suffix == "siglip-price":
        kind = "siglip-price"
    elif suffix == "llm-pooled":
        kind = "llm-pooled"
    else:
        raise ValueError(model_name)
    return family, kind


def fit_family(
    family: str,
    X: np.ndarray,
    y: np.ndarray,
    l2: float,
    slope_nonnegative: bool,
) -> dict:
    if family == "poisson":
        return fit_ztp(X, y, l2=l2, slope_nonnegative=slope_nonnegative)
    return fit_ztnb(X, y, l2=l2, slope_nonnegative=slope_nonnegative)


def choose_l2(
    model_name: str,
    train_rows: pd.DataFrame,
    embedding_df: pd.DataFrame,
    embedding_cols: list[str],
    persona_ids: list[str],
    pca_components: int,
    inner_folds: int,
) -> tuple[float, pd.DataFrame]:
    family, kind = model_parts(model_name)
    if kind in {"global", "llm-pooled"}:
        return 0.0, pd.DataFrame()

    groups = train_rows["article_id"].to_numpy(int)
    unique_groups = np.unique(groups)
    n_folds = min(inner_folds, len(unique_groups))
    if n_folds < 2:
        return 1.0, pd.DataFrame()
    splitter = GroupKFold(n_splits=n_folds)
    records: list[dict] = []

    for l2 in L2_GRID:
        for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(train_rows, groups=groups)):
            inner_train = train_rows.iloc[fit_idx].copy()
            inner_val = train_rows.iloc[val_idx].copy()
            builder = DesignBuilder(kind, embedding_df, embedding_cols, persona_ids, pca_components).fit(inner_train)
            X_fit = builder.transform(inner_train)
            X_val = builder.transform(inner_val)
            y_fit = inner_train["demand"].to_numpy(int)
            y_val = inner_val["demand"].to_numpy(int)
            fit = fit_family(family, X_fit, y_fit, l2=l2, slope_nonnegative=False)
            mu_val = predict_mu(fit, X_val)
            logpmf, p0 = family_logpmf_and_p0(y_val, mu_val, "poisson" if family == "poisson" else "nb", fit["dispersion"])
            avg_nll = float(np.mean(-logpmf + np.log1p(-p0)))
            records.append(
                {
                    "model": model_name,
                    "l2": float(l2),
                    "fold": int(fold_idx),
                    "avg_val_zt_nll": avg_nll,
                    "optimizer_success": bool(fit["success"]),
                }
            )

    cv = pd.DataFrame(records)
    mean_cv = cv.groupby("l2", as_index=False)["avg_val_zt_nll"].mean().sort_values(["avg_val_zt_nll", "l2"])
    return float(mean_cv.iloc[0]["l2"]), cv


def load_saved_control_summary(split_dir: Path, sample: str) -> pd.DataFrame:
    path = split_dir / "evaluation" / sample / "llm-mix-cal_summary.csv"
    frame = pd.read_csv(path)
    frame["model"] = "paper-llm-mix-cal"
    return frame


def summary_to_record(summary: pd.DataFrame, split_idx: int, sample: str) -> dict:
    row = {"split_index": split_idx, "sample": sample}
    for record in summary.itertuples(index=False):
        row[str(record.metric)] = float(record.value)
    row["model"] = str(summary["model"].iloc[0])
    return row


def paired_interval(deltas: np.ndarray) -> dict:
    values = np.asarray(deltas, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    mean = float(np.mean(values)) if n else float("nan")
    if n < 2:
        return {"n_splits": n, "mean_delta": mean, "ci95_low": float("nan"), "ci95_high": float("nan")}
    se = float(np.std(values, ddof=1) / math.sqrt(n))
    critical = float(t.ppf(0.975, df=n - 1))
    return {
        "n_splits": n,
        "mean_delta": mean,
        "std_delta": float(np.std(values, ddof=1)),
        "se_delta": se,
        "ci95_low": mean - critical * se,
        "ci95_high": mean + critical * se,
        "fraction_splits_better": float(np.mean(values < 0.0)),
    }


def build_pairwise_uncertainty(split_metrics: pd.DataFrame) -> pd.DataFrame:
    test = split_metrics[split_metrics["sample"] == "test"].copy()
    records: list[dict] = []
    comparisons: list[tuple[str, str, str]] = []

    for model in ALL_NEW_MODELS:
        comparisons.append((model, "paper-llm-mix-cal", "new_minus_paper_control"))
    for llm in LLM_MODELS:
        for nonllm in NON_LLM_MODELS:
            comparisons.append((llm, nonllm, "llm_minus_nonllm"))

    for left, right, comparison_type in comparisons:
        left_df = test[test["model"] == left].set_index("split_index")
        right_df = test[test["model"] == right].set_index("split_index")
        common = left_df.index.intersection(right_df.index)
        for metric in METRICS:
            if metric not in left_df.columns or metric not in right_df.columns:
                continue
            deltas = left_df.loc[common, metric].to_numpy(float) - right_df.loc[common, metric].to_numpy(float)
            stats = paired_interval(deltas)
            records.append(
                {
                    "comparison_type": comparison_type,
                    "left_model": left,
                    "right_model": right,
                    "metric": metric,
                    **stats,
                }
            )
    return pd.DataFrame(records)


def build_family_uncertainty(split_metrics: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("ztnb-global", "ztp-global"),
        ("ztnb-price-spline", "ztp-price-spline"),
        ("ztnb-siglip-price", "ztp-siglip-price"),
        ("ztnb-llm-pooled", "ztp-llm-pooled"),
    ]
    test = split_metrics[split_metrics["sample"] == "test"].copy()
    records: list[dict] = []
    for nb_name, pois_name in pairs:
        nb = test[test["model"] == nb_name].set_index("split_index")
        pois = test[test["model"] == pois_name].set_index("split_index")
        common = nb.index.intersection(pois.index)
        for metric in METRICS:
            deltas = nb.loc[common, metric].to_numpy(float) - pois.loc[common, metric].to_numpy(float)
            records.append(
                {
                    "nb_model": nb_name,
                    "poisson_model": pois_name,
                    "metric": metric,
                    **paired_interval(deltas),
                }
            )
    return pd.DataFrame(records)


def mean_metric_table(split_metrics: pd.DataFrame, sample: str) -> pd.DataFrame:
    subset = split_metrics[split_metrics["sample"] == sample].copy()
    cols = [c for c in METRICS if c in subset.columns]
    return (
        subset.groupby("model", as_index=False)[cols]
        .mean()
        .sort_values(["zt_avg_nll", "zt_avg_crps"], ascending=True)
        .reset_index(drop=True)
    )


def decide(mean_test: pd.DataFrame, pairwise: pd.DataFrame, family_uncertainty: pd.DataFrame) -> dict:
    candidates = mean_test[mean_test["model"].isin(ALL_NEW_MODELS)].copy()
    llm = candidates[candidates["model"].isin(LLM_MODELS)].sort_values("zt_avg_nll").iloc[0]
    nonllm = candidates[candidates["model"].isin(NON_LLM_MODELS)].sort_values("zt_avg_nll").iloc[0]

    llm_nll = float(llm["zt_avg_nll"])
    non_nll = float(nonllm["zt_avg_nll"])
    llm_crps = float(llm["zt_avg_crps"])
    non_crps = float(nonllm["zt_avg_crps"])
    nll_adv_pct = 100.0 * (non_nll - llm_nll) / non_nll
    crps_adv_pct = 100.0 * (non_crps - llm_crps) / non_crps

    pair = pairwise[
        (pairwise["comparison_type"] == "llm_minus_nonllm")
        & (pairwise["left_model"] == llm["model"])
        & (pairwise["right_model"] == nonllm["model"])
    ]
    nll_pair = pair[pair["metric"] == "zt_avg_nll"].iloc[0].to_dict()
    crps_pair = pair[pair["metric"] == "zt_avg_crps"].iloc[0].to_dict()

    noninferior_both = non_nll <= 1.01 * llm_nll and non_crps <= 1.01 * llm_crps
    strong_llm = (
        (nll_adv_pct >= 2.0 and float(nll_pair["ci95_high"]) < 0.0)
        or (crps_adv_pct >= 2.0 and float(crps_pair["ci95_high"]) < 0.0)
    )
    if strong_llm:
        llm_decision = "KEEP_CORE"
    elif nll_adv_pct >= 1.0 or crps_adv_pct >= 1.0:
        llm_decision = "KEEP_USEFUL_NOT_CORE"
    elif noninferior_both:
        llm_decision = "DOWNGRADE_CORE_ROLE"
    else:
        llm_decision = "INCONCLUSIVE"

    return {
        "best_llm_rate_model": str(llm["model"]),
        "best_non_llm_rate_model": str(nonllm["model"]),
        "best_llm_test_zt_nll": llm_nll,
        "best_non_llm_test_zt_nll": non_nll,
        "llm_nll_advantage_pct_vs_best_nonllm": nll_adv_pct,
        "best_llm_test_zt_crps": llm_crps,
        "best_nonllm_test_zt_crps": non_crps,
        "llm_crps_advantage_pct_vs_best_nonllm": crps_adv_pct,
        "llm_role_decision": llm_decision,
        "llm_vs_best_nonllm_nll_paired_ci95": [
            float(nll_pair["ci95_low"]),
            float(nll_pair["ci95_high"]),
        ],
        "llm_vs_best_nonllm_crps_paired_ci95": [
            float(crps_pair["ci95_low"]),
            float(crps_pair["ci95_high"]),
        ],
        "observation_family_note": (
            "Inspect family_paired_uncertainty.csv; prefer Negative Binomial only where matched held-out "
            "NLL/CRPS gains are stable rather than merely improving training fit."
        ),
        "new_llm_calls": 0,
        "api_cost_usd": 0,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sales = load_sales(args.sales)
    probabilities = load_probability_rows(args.responses)
    rows, persona_ids = build_prompting_design_rows(sales, probabilities)
    rows = rows[rows["demand"] > 0].copy().reset_index(drop=True)
    embedding_df, embedding_cols = load_product_embeddings(args.product_embeddings)

    split_records: list[dict] = []
    fit_records: list[dict] = []
    cv_records: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []

    for split_idx in range(args.n_splits):
        split_dir = args.saved_eval_dir / f"split_{split_idx:03d}"
        if not split_dir.exists():
            continue
        train_ids = pd.read_csv(split_dir / "train_products.csv")["article_id"].astype(int)
        test_ids = pd.read_csv(split_dir / "test_products.csv")["article_id"].astype(int)
        train_rows = rows[rows["article_id"].isin(train_ids)].copy().reset_index(drop=True)
        test_rows = rows[rows["article_id"].isin(test_ids)].copy().reset_index(drop=True)
        if train_rows.empty or test_rows.empty:
            raise RuntimeError(f"Empty train/test rows in split {split_idx}")

        for sample in ("train", "test"):
            control_summary = load_saved_control_summary(split_dir, sample)
            split_records.append(summary_to_record(control_summary, split_idx, sample))

        for model_name in ALL_NEW_MODELS:
            family_short, kind = model_parts(model_name)
            family = "poisson" if family_short == "poisson" else "nb"
            l2, cv = choose_l2(
                model_name,
                train_rows,
                embedding_df,
                embedding_cols,
                persona_ids,
                args.pca_components,
                args.inner_folds,
            )
            if not cv.empty:
                cv = cv.copy()
                cv["split_index"] = split_idx
                cv_records.append(cv)

            builder = DesignBuilder(
                kind,
                embedding_df,
                embedding_cols,
                persona_ids,
                args.pca_components,
            ).fit(train_rows)
            X_train = builder.transform(train_rows)
            X_test = builder.transform(test_rows)
            y_train = train_rows["demand"].to_numpy(int)
            fit = fit_family(
                family_short,
                X_train,
                y_train,
                l2=l2,
                slope_nonnegative=(kind == "llm-pooled"),
            )
            fit_records.append(
                {
                    "split_index": split_idx,
                    "model": model_name,
                    "family": family,
                    "feature_kind": kind,
                    "l2": l2,
                    "n_parameters": int(len(fit["beta"]) + (1 if family == "nb" else 0)),
                    "dispersion": float(fit["dispersion"]) if np.isfinite(fit["dispersion"]) else np.inf,
                    "optimizer_success": bool(fit["success"]),
                    "optimizer_message": fit["message"],
                    "train_objective": float(fit["objective"]),
                }
            )

            for sample, sample_rows, X in (
                ("train", train_rows, X_train),
                ("test", test_rows, X_test),
            ):
                mu = predict_mu(fit, X)
                pair_scores, summary, diagnostics = score_count_model(
                    sample_rows,
                    model_name=model_name,
                    mu=mu,
                    family=family,
                    dispersion=float(fit["dispersion"]),
                    split_label=f"split_{split_idx:03d}_{sample}",
                    seed=7000 + split_idx * 100 + (0 if sample == "train" else 50),
                    tail_prob=args.crps_tail_prob,
                    support_cap=args.max_crps_support,
                    full_distribution_metrics=(sample == "test"),
                )
                pair_scores["split_index"] = split_idx
                pair_scores["sample"] = sample
                pair_frames.append(pair_scores)
                record = summary_to_record(summary, split_idx, sample)
                record.update(
                    {
                        "support_max": diagnostics["support_max"],
                        "max_positive_tail_mass_at_support": diagnostics["max_positive_tail_mass_at_support"],
                        "mean_untruncated_rate": diagnostics["mean_untruncated_rate"],
                        "mean_positive_conditional_prediction": diagnostics["mean_positive_conditional_prediction"],
                    }
                )
                split_records.append(record)

        print(f"completed split {split_idx:03d}", flush=True)

    split_metrics = pd.DataFrame(split_records)
    fit_summary = pd.DataFrame(fit_records)
    pair_scores_all = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    cv_scores = pd.concat(cv_records, ignore_index=True) if cv_records else pd.DataFrame()

    split_metrics.to_csv(args.output_dir / "split_metrics.csv", index=False)
    fit_summary.to_csv(args.output_dir / "fit_summary.csv", index=False)
    pair_scores_all.to_csv(args.output_dir / "pair_scores.csv", index=False)
    if not cv_scores.empty:
        cv_scores.to_csv(args.output_dir / "inner_cv_scores.csv", index=False)

    mean_test = mean_metric_table(split_metrics, "test")
    mean_train = mean_metric_table(split_metrics, "train")
    mean_test.to_csv(args.output_dir / "mean_metrics_test.csv", index=False)
    mean_train.to_csv(args.output_dir / "mean_metrics_train.csv", index=False)

    pairwise = build_pairwise_uncertainty(split_metrics)
    family_uncertainty = build_family_uncertainty(split_metrics)
    pairwise.to_csv(args.output_dir / "paired_uncertainty.csv", index=False)
    family_uncertainty.to_csv(args.output_dir / "family_paired_uncertainty.csv", index=False)

    decision = decide(mean_test, pairwise, family_uncertainty)
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2))

    print("\nHeld-out mean metrics:\n")
    print(mean_test.to_string(index=False))
    print("\nDecision:\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
