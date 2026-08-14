from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, gammaln
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

from demand_sim.data import load_probability_rows, load_sales
from demand_sim.metrics import pair_level_zero_truncated_binomial_scores, summarize_pair_scores

ROOT = Path("outputs/research/full_product_holdout_v1")
ROOT.mkdir(parents=True, exist_ok=True)
HISTORY_START = pd.Timestamp("2018-09-01")
HISTORY_END = pd.Timestamp("2019-09-19")
PRICE_SCALE = 590.0
K_BUYER_TYPES = 8
N_SPLITS = 7
EXPOSURE_VALUES = [100, 150, 200, 250]
SEED = 2025
EPS = 1e-6


def read_single_csv_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX/") and "/._" not in n]
        if len(names) != 1:
            raise RuntimeError(f"Expected one real CSV in {path}; found {names}")
        with zf.open(names[0]) as handle:
            return pd.read_csv(handle)


def robust_qcut(values: pd.Series, labels: list[str]) -> pd.Series:
    try:
        out = pd.qcut(values, q=len(labels), labels=labels, duplicates="drop")
        if out.nunique(dropna=True) >= 2:
            return out.astype(object)
    except ValueError:
        pass
    ranked = values.rank(method="average")
    n_bins = min(len(labels), max(int(ranked.nunique()), 1))
    if n_bins <= 1:
        return pd.Series([labels[0]] * len(values), index=values.index, dtype=object)
    return pd.qcut(ranked, q=n_bins, labels=labels[:n_bins], duplicates="drop").astype(object)


def assign_original_persona_membership(features: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the repository's original top-50 cell membership.

    NOTE: customer_features.csv is a repository artifact built from the full pre-2019-09-20
    period. This shared transductive limitation is recorded explicitly in the report.
    """
    d = features.copy()
    d["customer_id"] = d["customer_id"].astype(str)
    d["age"] = pd.to_numeric(d["age"], errors="coerce")
    d["txn_count"] = pd.to_numeric(d["txn_count"], errors="coerce").fillna(0.0)
    d["mean_price"] = pd.to_numeric(d["mean_price"], errors="coerce").fillna(0.0)
    d["top_product_type"] = d["top_product_type"].fillna("UNKNOWN").astype(str)
    d = d[d["txn_count"] > 0].copy()
    months = ((HISTORY_END - HISTORY_START).days + 1) / 30.4375
    d["txn_per_month"] = d["txn_count"] / months
    d["age_bin"] = pd.cut(
        d["age"], [16, 25, 35, 45, 55, 200], right=False,
        labels=["16-24", "25-34", "35-44", "45-54", "55+"]
    ).astype(object).fillna("UNKNOWN")
    d["engagement_bin"] = robust_qcut(d["txn_per_month"], ["low", "mid", "high"])
    d["price_tier"] = robust_qcut(d["mean_price"], ["low", "mid", "high"])
    top_types = d["top_product_type"].value_counts().head(100).index
    d["taste_bucket"] = np.where(d["top_product_type"].isin(top_types), d["top_product_type"], "OTHER")
    key = ["age_bin", "engagement_bin", "price_tier", "taste_bucket"]
    out = d.merge(cells[key + ["persona_id", "n_customers"]], on=key, how="inner")
    return out[["customer_id", "persona_id", "n_customers"]].drop_duplicates("customer_id")


def prepare_trouser_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    tx = raw.copy()
    tx["t_dat"] = pd.to_datetime(tx["t_dat"], errors="coerce")
    tx["customer_id"] = tx["customer_id"].astype(str)
    tx["article_id"] = pd.to_numeric(tx["article_id"], errors="coerce")
    tx["price"] = pd.to_numeric(tx["price"], errors="coerce") * PRICE_SCALE
    tx = tx.dropna(subset=["t_dat", "customer_id", "article_id", "price"])
    tx["article_id"] = tx["article_id"].astype(int)
    return tx[(tx["t_dat"] >= HISTORY_START) & (tx["t_dat"] <= HISTORY_END)].copy()


def behavior_features_for_allowed_products(tx: pd.DataFrame, allowed_products: set[int]) -> pd.DataFrame:
    d = tx[tx["article_id"].isin(allowed_products)].copy().sort_values(["customer_id", "t_dat"])
    g = d.groupby("customer_id", sort=False)
    f = g.agg(
        txn_count=("article_id", "size"),
        unique_articles=("article_id", "nunique"),
        mean_price=("price", "mean"),
        price_std=("price", "std"),
        active_days=("t_dat", "nunique"),
        first_date=("t_dat", "min"),
        last_date=("t_dat", "max"),
    ).reset_index()
    if f.empty:
        raise RuntimeError("No allowed-product transactions available for type construction")
    f["price_std"] = f["price_std"].fillna(0.0)
    f["recency_days"] = (HISTORY_END - f["last_date"]).dt.days.astype(float)
    f["span_days"] = (f["last_date"] - f["first_date"]).dt.days.astype(float)
    f["repeat_ratio"] = 1.0 - f["unique_articles"] / f["txn_count"].clip(lower=1)
    for days in [30, 90, 180]:
        start = HISTORY_END - pd.Timedelta(days=days - 1)
        counts = d[d["t_dat"] >= start].groupby("customer_id").size().rename(f"last_{days}d")
        f = f.merge(counts, on="customer_id", how="left")
    return f.drop(columns=["first_date", "last_date"]).fillna(0.0)


def build_learned_types(
    members: pd.DataFrame,
    tx: pd.DataFrame,
    train_products: set[int],
    persona_ids: list[str],
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return type weights, P(persona|type), and type audit table.

    All treatment-specific behavioral features exclude held-out product transactions.
    """
    f = behavior_features_for_allowed_products(tx, train_products)
    pop = members[["customer_id", "persona_id"]].merge(f, on="customer_id", how="left")
    feat = [
        "txn_count", "unique_articles", "mean_price", "price_std", "active_days",
        "recency_days", "span_days", "repeat_ratio", "last_30d", "last_90d", "last_180d",
    ]
    buyers = pop[pop["txn_count"].notna()].copy()
    if len(buyers) < K_BUYER_TYPES * 20:
        raise RuntimeError(f"Too few allowed-product buyers for clustering: {len(buyers)}")
    scaler = RobustScaler()
    x = scaler.fit_transform(buyers[feat])
    km = KMeans(n_clusters=K_BUYER_TYPES, random_state=split_seed, n_init=20)
    buyers["learned_type"] = [f"T{int(v)+1}" for v in km.fit_predict(x)]
    pop = pop.merge(buyers[["customer_id", "learned_type"]], on="customer_id", how="left")
    pop["learned_type"] = pop["learned_type"].fillna("T0_NONBUYER")

    type_ids = sorted(pop["learned_type"].unique(), key=lambda s: (s != "T0_NONBUYER", s))
    total = float(len(pop))
    weights = np.asarray([(pop["learned_type"] == t).sum() / total for t in type_ids], dtype=float)
    comp = np.zeros((len(type_ids), len(persona_ids)), dtype=float)
    rows = []
    for ti, t in enumerate(type_ids):
        g = pop[pop["learned_type"] == t]
        counts = g["persona_id"].value_counts()
        denom = float(counts.sum())
        for pi, pid in enumerate(persona_ids):
            comp[ti, pi] = float(counts.get(pid, 0.0)) / max(denom, 1.0)
        rec = {
            "type_id": t,
            "n_customers": int(len(g)),
            "weight": float(weights[ti]),
            "n_personas_represented": int((comp[ti] > 0).sum()),
        }
        if t != "T0_NONBUYER":
            for col in ["txn_count", "unique_articles", "mean_price", "recency_days", "last_90d"]:
                rec[f"median_{col}"] = float(g[col].median())
        rows.append(rec)
    if not np.allclose(comp.sum(axis=1), 1.0, atol=1e-8):
        raise RuntimeError("Type composition rows do not sum to one")
    if not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise RuntimeError("Type weights do not sum to one")
    return weights, comp, pd.DataFrame(rows)


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), EPS, 1.0 - EPS)
    return np.log(x / (1.0 - x))


def build_signal_tables(
    q_points: pd.DataFrame,
    persona_ids: list[str],
    train_products: set[int],
    empirical_persona_weights: np.ndarray,
    type_weights: np.ndarray,
    type_composition: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct two relative-utility signals.

    Empirical-persona control: aggregate raw persona probabilities first, then standardize
    logit utility using train products.

    Learned-type treatment: form a probability curve for each learned type from its
    persona composition, convert each type's curve to standardized relative logit utility
    using train products only, then population-weight the relative utilities.
    """
    qmat = q_points[persona_ids].to_numpy(float)
    train_mask = q_points["article_id"].isin(train_products).to_numpy()
    if not np.any(train_mask):
        raise RuntimeError("No train-product LLM points")

    q_emp = np.clip(qmat @ empirical_persona_weights, EPS, 1.0 - EPS)
    u_emp = logit(q_emp)
    emp_mu = float(u_emp[train_mask].mean())
    emp_sd = float(u_emp[train_mask].std())
    if emp_sd < 1e-8:
        emp_sd = 1.0
    emp_signal = (u_emp - emp_mu) / emp_sd

    q_type = np.clip(qmat @ type_composition.T, EPS, 1.0 - EPS)
    u_type = logit(q_type)
    mu = u_type[train_mask].mean(axis=0)
    sd = u_type[train_mask].std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    relative_u = np.clip((u_type - mu) / sd, -8.0, 8.0)
    learned_signal = relative_u @ type_weights

    keys = q_points[["article_id", "offer_price"]].copy()
    empirical = keys.copy()
    empirical["signal"] = emp_signal
    learned = keys.copy()
    learned["signal"] = learned_signal
    return empirical, learned


def row_zt_nll(y: np.ndarray, n: int, p: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    out = np.full(len(y), np.inf, dtype=float)
    ok = (y > 0) & (y <= n)
    yy = y[ok]
    pp = p[ok]
    log_choose = gammaln(n + 1) - gammaln(yy + 1) - gammaln(n - yy + 1)
    log_pmf = log_choose + yy * np.log(pp) + (n - yy) * np.log1p(-pp)
    zero_mass = np.exp(n * np.log1p(-pp))
    out[ok] = -(log_pmf - np.log1p(-zero_mass))
    return out


def fit_calibration(rows: pd.DataFrame) -> dict:
    x = rows["signal"].to_numpy(float)
    y = rows["demand"].to_numpy(int)
    best = None
    for n in EXPOSURE_VALUES:
        if n < int(y.max()):
            continue
        target = float(np.clip(y.mean() / n, EPS, 1.0 - EPS))
        init_intercept = math.log(target / (1.0 - target))

        def objective(par: np.ndarray) -> float:
            intercept = float(par[0])
            slope = float(np.exp(par[1]))
            p = expit(intercept + slope * x)
            return float(row_zt_nll(y, n, p).mean())

        res = minimize(
            objective,
            x0=np.asarray([init_intercept, 0.0]),
            method="L-BFGS-B",
            bounds=[(-20.0, 20.0), (-5.0, 5.0)],
            options={"maxiter": 300},
        )
        cand = {
            "exposure_n": int(n),
            "intercept": float(res.x[0]),
            "slope": float(np.exp(res.x[1])),
            "train_zt_nll": float(res.fun),
            "success": bool(res.success),
        }
        if best is None or cand["train_zt_nll"] < best["train_zt_nll"]:
            best = cand
    if best is None:
        raise RuntimeError("No calibration fit")
    return best


@dataclass
class SignalModel:
    model_name: str
    exposure_n: int
    intercept: float
    slope: float

    @property
    def name(self) -> str:
        return self.model_name

    def purchase_probability(self, rows: pd.DataFrame) -> np.ndarray:
        return np.clip(expit(self.intercept + self.slope * rows["signal"].to_numpy(float)), 1e-9, 1.0 - 1e-9)

    def mean_demand(self, rows: pd.DataFrame) -> np.ndarray:
        return self.exposure_n * self.purchase_probability(rows)


def score_model(rows: pd.DataFrame, fit: dict, model_name: str, split_idx: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = SignalModel(
        model_name=model_name,
        exposure_n=int(fit["exposure_n"]),
        intercept=float(fit["intercept"]),
        slope=float(fit["slope"]),
    )
    eval_rows = rows[rows["demand"] > 0].copy()
    p = model.purchase_probability(eval_rows)
    zero_mass = np.exp(model.exposure_n * np.log1p(-p))
    mean_prediction = model.mean_demand(eval_rows) / np.clip(1.0 - zero_mass, 1e-12, None)
    pair = pair_level_zero_truncated_binomial_scores(
        rows=eval_rows,
        model_name=model.name,
        exposure_n=model.exposure_n,
        purchase_prob=p,
        mean_prediction=mean_prediction,
        split_label=f"split_{split_idx:03d}_test",
        seed=SEED + 1000 * split_idx,
    )
    summary = summarize_pair_scores(pair)
    return pair, summary


def summary_to_dict(summary: pd.DataFrame) -> dict[str, float]:
    return {str(r.metric): float(r.value) for r in summary.itertuples(index=False)}


def load_baseline_summary(split_idx: int) -> dict[str, float]:
    path = Path(f"outputs/demand_prediction/split_{split_idx:03d}/evaluation/test/llm-mix-cal_summary.csv")
    return summary_to_dict(pd.read_csv(path))


def split_bootstrap(diffs: list[float], n_boot: int = 20000) -> dict:
    x = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(SEED)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        means.append(float(x[idx].mean()))
    b = np.asarray(means)
    return {
        "mean_difference": float(x.mean()),
        "ci95_low": float(np.quantile(b, 0.025)),
        "ci95_high": float(np.quantile(b, 0.975)),
        "n_splits_better": int(np.sum(x < 0)),
        "n_splits": int(len(x)),
        "bootstrap_p_mean_better": float(np.mean(b < 0)),
    }


def main() -> None:
    cells = pd.read_csv("outputs/personas/persona_cells.csv").head(50).copy()
    persona_ids = cells["persona_id"].astype(str).tolist()
    empirical_w = cells["n_customers"].to_numpy(float)
    empirical_w = empirical_w / empirical_w.sum()

    features = read_single_csv_zip(Path("outputs/personas/customer_features.csv.zip"))
    members = assign_original_persona_membership(features, cells)
    raw_tx = read_single_csv_zip(Path("outputs/products/txns_trousers_online.csv.zip"))
    tx = prepare_trouser_transactions(raw_tx)

    probs = load_probability_rows(Path("outputs/responses/llm_responses_online_top100.csv"))
    qwide = (
        probs.groupby(["article_id", "offer_price", "persona_id"], as_index=False)["p_buy"].mean()
        .pivot(index=["article_id", "offer_price"], columns="persona_id", values="p_buy")
        .reset_index()
    )
    missing_personas = [p for p in persona_ids if p not in qwide.columns]
    if missing_personas:
        raise RuntimeError(f"Missing persona columns in LLM responses: {missing_personas}")
    qwide = qwide.dropna(subset=persona_ids).copy()
    qwide["article_id"] = qwide["article_id"].astype(int)
    qwide["offer_price"] = qwide["offer_price"].astype(float).round(2)

    sales = load_sales(Path("outputs/products/sales_top100_online.csv"))
    sales = sales[sales["demand"] > 0].copy()

    split_records = []
    all_pair = []
    type_audits = []
    for split_idx in range(N_SPLITS):
        split_dir = Path(f"outputs/demand_prediction/split_{split_idx:03d}")
        train_ids = set(pd.read_csv(split_dir / "train_products.csv")["article_id"].astype(int))
        test_ids = set(pd.read_csv(split_dir / "test_products.csv")["article_id"].astype(int))
        if train_ids & test_ids:
            raise RuntimeError(f"Split {split_idx}: train/test overlap")

        type_w, composition, type_audit = build_learned_types(
            members=members,
            tx=tx,
            train_products=train_ids,
            persona_ids=persona_ids,
            split_seed=SEED + split_idx,
        )
        type_audit.insert(0, "split", split_idx)
        type_audits.append(type_audit)

        emp_signal, learned_signal = build_signal_tables(
            q_points=qwide,
            persona_ids=persona_ids,
            train_products=train_ids,
            empirical_persona_weights=empirical_w,
            type_weights=type_w,
            type_composition=composition,
        )

        baseline_summary = load_baseline_summary(split_idx)
        model_summaries = {"paper_llm_mix_cal": baseline_summary}
        fits = {}
        for model_name, signal in [
            ("empirical_persona_relative_utility", emp_signal),
            ("learned_type_relative_utility", learned_signal),
        ]:
            merged = sales.merge(signal, on=["article_id", "offer_price"], how="inner")
            train = merged[merged["article_id"].isin(train_ids)].copy()
            test = merged[merged["article_id"].isin(test_ids)].copy()
            if train.empty or test.empty:
                raise RuntimeError(f"Split {split_idx} {model_name}: empty train/test")
            fit = fit_calibration(train)
            fits[model_name] = fit
            pair, summary = score_model(test, fit, model_name, split_idx)
            model_summaries[model_name] = summary_to_dict(summary)
            pair.insert(0, "split_idx", split_idx)
            all_pair.append(pair)

            base_pair_path = split_dir / "evaluation/test/llm-mix-cal_pair_scores.csv"
            base_pair = pd.read_csv(base_pair_path)
            keys = ["article_id", "offer_price"]
            left = base_pair[keys + ["n_observations"]].sort_values(keys).reset_index(drop=True)
            right = pair[keys + ["n_observations"]].sort_values(keys).reset_index(drop=True)
            if not left.equals(right):
                raise RuntimeError(f"Split {split_idx} {model_name}: evaluation coverage differs from paper baseline")

        split_records.append({
            "split": split_idx,
            "n_train_products": len(train_ids),
            "n_test_products": len(test_ids),
            "n_population_customers": int(len(members)),
            "n_types": int(len(type_w)),
            "fits": fits,
            "summary": model_summaries,
        })
        print(f"Completed split {split_idx}: baseline NLL={baseline_summary['zt_avg_nll']:.6f}, learned NLL={model_summaries['learned_type_relative_utility']['zt_avg_nll']:.6f}")

    metrics = ["zt_avg_nll", "zt_avg_crps", "mae", "rmse"]
    models = ["paper_llm_mix_cal", "empirical_persona_relative_utility", "learned_type_relative_utility"]
    aggregate = {}
    for model in models:
        aggregate[model] = {}
        for metric in metrics:
            vals = np.asarray([r["summary"][model][metric] for r in split_records], dtype=float)
            aggregate[model][metric] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}

    paired = {}
    for challenger in ["empirical_persona_relative_utility", "learned_type_relative_utility"]:
        paired[challenger] = {}
        for metric in metrics:
            diffs = [r["summary"][challenger][metric] - r["summary"]["paper_llm_mix_cal"][metric] for r in split_records]
            paired[challenger][metric] = split_bootstrap(diffs)

    learned = aggregate["learned_type_relative_utility"]
    base = aggregate["paper_llm_mix_cal"]
    learned_nll_wins = paired["learned_type_relative_utility"]["zt_avg_nll"]["n_splits_better"]
    decision = (
        learned["zt_avg_nll"]["mean"] < base["zt_avg_nll"]["mean"]
        and learned["zt_avg_crps"]["mean"] < base["zt_avg_crps"]["mean"]
        and learned["mae"]["mean"] <= 1.01 * base["mae"]["mean"]
        and learned["rmse"]["mean"] <= 1.01 * base["rmse"]["mean"]
        and learned_nll_wins >= 4
    )

    result = {
        "protocol": "FULL-PRODUCT-HOLDOUT-LEARNED-UTILITY-v1",
        "decision": "ADVANCE_TO_FRESH_PAIRWISE_LLM" if decision else "DO_NOT_ADVANCE_TO_FRESH_PAIRWISE_LLM",
        "pre_registered_gate": {
            "requirements": [
                "mean NLL lower than paper llm-mix-cal",
                "mean CRPS lower than paper llm-mix-cal",
                "mean MAE and RMSE no worse than +1%",
                "NLL wins on at least 4 of 7 original paper product splits",
            ]
        },
        "aggregate": aggregate,
        "paired_split_bootstrap": paired,
        "splits": split_records,
        "limitations": [
            "Treatment-specific behavioral features remove every held-out product transaction before clustering.",
            "Original persona-cell membership comes from the repository's precomputed full-period customer_features artifact and cannot be rebuilt leakage-free without the raw all-category H&M transactions.",
            "Candidate price grids are historically observed and therefore transductive in both arms.",
            "This full-scale gate reuses the authors' frozen 5,000 LLM probability responses and converts them to relative utility; it is a surrogate for fresh pairwise elicitation, not fresh GPT-5.6 pairwise calls.",
        ],
        "interpretation_rule": "Only a passing result justifies spending fresh LLM judgments on all 100 products.",
    }
    (ROOT / "result.json").write_text(json.dumps(result, indent=2))
    pd.concat(type_audits, ignore_index=True).to_csv(ROOT / "type_audit_all_splits.csv", index=False)
    pd.concat(all_pair, ignore_index=True).to_csv(ROOT / "challenger_pair_scores.csv", index=False)

    lines = [
        "# Full Product-Holdout Learned-Type Relative-Utility Gate",
        "",
        f"**Decision:** {result['decision']}",
        "",
        "Seven original H&M product splits; 60 train / 40 held-out products per split. Lower is better.",
        "",
        "| model | NLL | CRPS | MAE | RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in models:
        a = aggregate[model]
        lines.append(
            f"| {model} | {a['zt_avg_nll']['mean']:.6f} | {a['zt_avg_crps']['mean']:.6f} | {a['mae']['mean']:.6f} | {a['rmse']['mean']:.6f} |"
        )
    lines += ["", "## Learned type vs paper baseline", ""]
    for metric in metrics:
        rec = paired["learned_type_relative_utility"][metric]
        rel = (aggregate["learned_type_relative_utility"][metric]["mean"] - aggregate["paper_llm_mix_cal"][metric]["mean"]) / aggregate["paper_llm_mix_cal"][metric]["mean"]
        lines.append(
            f"- {metric}: {rel*100:+.3f}% mean change; wins {rec['n_splits_better']}/7 splits; split-bootstrap 95% CI for absolute mean delta [{rec['ci95_low']:.6f}, {rec['ci95_high']:.6f}]."
        )
    lines += [
        "",
        "## Scientific scope",
        "",
        "This is the full 100-product, seven-split scaling gate using frozen author LLM outputs. Held-out-product transactions are excluded from learned behavioral-type construction. It does not yet constitute fresh pairwise GPT-5.6 elicitation.",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
