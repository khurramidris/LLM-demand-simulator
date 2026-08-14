from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from scipy.optimize import minimize
from scipy.special import expit, gammaln
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler

from demand_sim.data import load_probability_rows, load_sales
from demand_sim.metrics import pair_level_zero_truncated_binomial_scores, summarize_pair_scores

ROOT = Path("outputs/research/clean_all_category_types_v1")
ROOT.mkdir(parents=True, exist_ok=True)
SPLIT = Path("outputs/demand_prediction/split_000")
HF_REPO = "microsoft/hnm-search-data"
HF_REVISION = "refs/convert/parquet"
HISTORY_START = "2018-09-01"
HISTORY_END = "2019-09-19"
PRICE_SCALE = 590.0
SEED = 2025
K_COARSE = 8
K_RICH = 24
EXPOSURE_VALUES = [100, 150, 200, 250]
PRIOR_STRENGTH = 2000.0
PRICE_PENALTY_PER_10USD = 12.0

# Pre-registered before any clean all-category result exists.
PRIMARY_GATE = (
    "clean_all_category_types must have lower held-out NLL and CRPS than "
    "coarse_trouser_types, with MAE and RMSE no worse than +1%"
)

STYLE_COLS = [
    "denim_share", "woven_trouser_share", "jersey_share",
    "black_share", "blue_share", "grey_share", "khaki_share",
    "solid_share", "denim_app_share", "melange_share", "pattern_share",
]


def hf_parquet_files(prefix: str) -> list[str]:
    api = HfApi()
    files = api.list_repo_files(HF_REPO, repo_type="dataset", revision=HF_REVISION)
    matches = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
    if not matches:
        raise RuntimeError(f"No parquet files found for {prefix} at {HF_REPO}@{HF_REVISION}")
    cache = os.environ.get("HF_HOME", str(Path(tempfile.gettempdir()) / "hf-cache"))
    return [
        hf_hub_download(
            repo_id=HF_REPO,
            filename=f,
            repo_type="dataset",
            revision=HF_REVISION,
            cache_dir=cache,
        )
        for f in matches
    ]


def sql_list(paths: list[str]) -> str:
    return "[" + ",".join("'" + p.replace("'", "''") + "'" for p in paths) + "]"


def age_bin(age: pd.Series) -> pd.Series:
    x = pd.to_numeric(age, errors="coerce")
    out = pd.cut(x, [16, 25, 35, 45, 55, 200], right=False, labels=["16-24", "25-34", "35-44", "45-54", "55+"])
    return out.astype(object).fillna("UNKNOWN")


def row_zt_nll(y, n, p):
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    out = np.full(len(y), np.inf, float)
    ok = (y > 0) & (y <= n)
    yy = y[ok]
    pp = p[ok]
    lc = gammaln(n + 1) - gammaln(yy + 1) - gammaln(n - yy + 1)
    lp = lc + yy * np.log(pp) + (n - yy) * np.log1p(-pp)
    z = np.exp(n * np.log1p(-pp))
    out[ok] = -(lp - np.log1p(-z))
    return out


def fit_calibration(rows: pd.DataFrame) -> dict:
    x = rows.signal.to_numpy(float)
    y = rows.demand.to_numpy(int)
    best = None
    for n in EXPOSURE_VALUES:
        if n < int(y.max()):
            continue
        target = float(np.clip(y.mean() / n, 1e-6, 1 - 1e-6))
        init = math.log(target / (1 - target))

        def obj(par):
            intercept = float(par[0])
            slope = float(np.exp(par[1]))
            return float(row_zt_nll(y, n, expit(intercept + slope * x)).mean())

        res = minimize(
            obj, [init, 0.0], method="L-BFGS-B",
            bounds=[(-20, 20), (-5, 5)], options={"maxiter": 300},
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
        raise RuntimeError("Calibration failed")
    return best


@dataclass
class SignalModel:
    model_name: str
    exposure_n: int
    intercept: float
    slope: float

    @property
    def name(self):
        return self.model_name

    def purchase_probability(self, rows):
        return np.clip(expit(self.intercept + self.slope * rows.signal.to_numpy(float)), 1e-9, 1 - 1e-9)

    def mean_demand(self, rows):
        return self.exposure_n * self.purchase_probability(rows)


def score_model(name: str, test: pd.DataFrame, fit: dict):
    model = SignalModel(name, int(fit["exposure_n"]), float(fit["intercept"]), float(fit["slope"]))
    p = model.purchase_probability(test)
    zero = np.exp(model.exposure_n * np.log1p(-p))
    mean = model.mean_demand(test) / np.clip(1 - zero, 1e-12, None)
    pair = pair_level_zero_truncated_binomial_scores(
        test, model.name, model.exposure_n, p,
        mean_prediction=mean, split_label="split_000_test", seed=SEED,
    )
    summary = {str(r.metric): float(r.value) for r in summarize_pair_scores(pair).itertuples(index=False)}
    return pair, summary


def product_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 20000) -> dict:
    keys = ["article_id", "offer_price"]
    aa = a.sort_values(keys).reset_index(drop=True)
    bb = b.sort_values(keys).reset_index(drop=True)
    if not aa[keys + ["n_observations"]].equals(bb[keys + ["n_observations"]]):
        raise RuntimeError("Pair coverage mismatch")
    products = np.asarray(sorted(aa.article_id.unique()), int)
    aid_arr = aa.article_id.to_numpy()
    idx = {aid: np.flatnonzero(aid_arr == aid) for aid in products}
    rng = np.random.default_rng(SEED)
    out = {}
    for metric in ["zt_avg_nll", "zt_avg_crps", "mae"]:
        delta = bb[metric].to_numpy(float) - aa[metric].to_numpy(float)
        w = aa.n_observations.to_numpy(float)
        num = np.array([np.sum(delta[idx[a]] * w[idx[a]]) for a in products])
        den = np.array([np.sum(w[idx[a]]) for a in products])
        draws = rng.integers(0, len(products), size=(n_boot, len(products)))
        vals = num[draws].sum(axis=1) / den[draws].sum(axis=1)
        out[metric] = {
            "treatment_minus_control_mean": float(vals.mean()),
            "ci95_low": float(np.quantile(vals, 0.025)),
            "ci95_high": float(np.quantile(vals, 0.975)),
            "p_treatment_better": float(np.mean(vals < 0)),
        }
    return out


def build_customer_features(tx_paths: list[str], article_paths: list[str], customer_paths: list[str], test_ids: set[int]) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='6GB'")
    test_sql = ",".join(str(int(x)) for x in sorted(test_ids))
    tx_src = f"read_parquet({sql_list(tx_paths)}, union_by_name=true)"
    art_src = f"read_parquet({sql_list(article_paths)}, union_by_name=true)"
    cust_src = f"read_parquet({sql_list(customer_paths)}, union_by_name=true)"

    # Held-out products are removed here, before every behavioral feature below.
    con.execute(f"""
        CREATE TEMP TABLE hist AS
        SELECT
            CAST(t.t_dat AS DATE) AS t_dat,
            CAST(t.customer_id AS VARCHAR) AS customer_id,
            CAST(t.article_id AS BIGINT) AS article_id,
            CAST(t.price AS DOUBLE) * {PRICE_SCALE} AS price_usd,
            CAST(t.sales_channel_id AS INTEGER) AS sales_channel_id,
            COALESCE(CAST(a.product_type_name AS VARCHAR), 'UNKNOWN') AS product_type_name,
            COALESCE(CAST(a.garment_group_name AS VARCHAR), 'UNKNOWN') AS garment_group_name,
            COALESCE(CAST(a.colour_group_name AS VARCHAR), 'UNKNOWN') AS colour_group_name,
            COALESCE(CAST(a.graphical_appearance_name AS VARCHAR), 'UNKNOWN') AS graphical_appearance_name
        FROM {tx_src} t
        LEFT JOIN {art_src} a USING(article_id)
        WHERE CAST(t.t_dat AS DATE) BETWEEN DATE '{HISTORY_START}' AND DATE '{HISTORY_END}'
          AND CAST(t.article_id AS BIGINT) NOT IN ({test_sql})
    """)

    agg = con.execute(f"""
        SELECT
            customer_id,
            COUNT(*)::DOUBLE AS txn_count,
            COUNT(DISTINCT article_id)::DOUBLE AS unique_articles,
            AVG(price_usd) AS mean_price,
            COALESCE(STDDEV_SAMP(price_usd), 0) AS price_std,
            COUNT(DISTINCT t_dat)::DOUBLE AS active_days,
            DATE_DIFF('day', MAX(t_dat), DATE '{HISTORY_END}')::DOUBLE AS recency_days,
            DATE_DIFF('day', MIN(t_dat), MAX(t_dat))::DOUBLE AS span_days,
            SUM(CASE WHEN t_dat >= DATE '{HISTORY_END}' - INTERVAL 29 DAY THEN 1 ELSE 0 END)::DOUBLE AS last_30d,
            SUM(CASE WHEN t_dat >= DATE '{HISTORY_END}' - INTERVAL 89 DAY THEN 1 ELSE 0 END)::DOUBLE AS last_90d,
            SUM(CASE WHEN t_dat >= DATE '{HISTORY_END}' - INTERVAL 179 DAY THEN 1 ELSE 0 END)::DOUBLE AS last_180d,
            AVG(CASE WHEN sales_channel_id = 1 THEN 1.0 ELSE 0.0 END) AS online_share,
            AVG(CASE WHEN product_type_name = 'Trousers' THEN 1.0 ELSE 0.0 END) AS trouser_share,
            AVG(CASE WHEN product_type_name = 'Leggings/Tights' THEN 1.0 ELSE 0.0 END) AS leggings_share,
            AVG(CASE WHEN product_type_name = 'Dress' THEN 1.0 ELSE 0.0 END) AS dress_share,
            AVG(CASE WHEN product_type_name = 'Skirt' THEN 1.0 ELSE 0.0 END) AS skirt_share,
            AVG(CASE WHEN garment_group_name = 'Trousers Denim' THEN 1.0 ELSE 0.0 END) AS denim_share,
            AVG(CASE WHEN garment_group_name = 'Trousers' THEN 1.0 ELSE 0.0 END) AS woven_trouser_share,
            AVG(CASE WHEN garment_group_name LIKE 'Jersey%' THEN 1.0 ELSE 0.0 END) AS jersey_share,
            AVG(CASE WHEN colour_group_name = 'Black' THEN 1.0 ELSE 0.0 END) AS black_share,
            AVG(CASE WHEN colour_group_name LIKE '%Blue%' OR colour_group_name = 'Blue' THEN 1.0 ELSE 0.0 END) AS blue_share,
            AVG(CASE WHEN colour_group_name LIKE '%Grey%' OR colour_group_name = 'Grey' THEN 1.0 ELSE 0.0 END) AS grey_share,
            AVG(CASE WHEN colour_group_name LIKE '%Khaki%' THEN 1.0 ELSE 0.0 END) AS khaki_share,
            AVG(CASE WHEN graphical_appearance_name = 'Solid' THEN 1.0 ELSE 0.0 END) AS solid_share,
            AVG(CASE WHEN graphical_appearance_name = 'Denim' THEN 1.0 ELSE 0.0 END) AS denim_app_share,
            AVG(CASE WHEN graphical_appearance_name = 'Melange' THEN 1.0 ELSE 0.0 END) AS melange_share,
            AVG(CASE WHEN graphical_appearance_name IN ('All over pattern','Stripe','Check','Dot','Other pattern') THEN 1.0 ELSE 0.0 END) AS pattern_share
        FROM hist
        GROUP BY customer_id
    """).df()

    customers = con.execute(f"SELECT CAST(customer_id AS VARCHAR) AS customer_id, CAST(age AS DOUBLE) AS age FROM {cust_src}").df()
    con.close()
    df = customers.merge(agg, on="customer_id", how="left")
    numeric = [c for c in df.columns if c not in ["customer_id"]]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in [x for x in numeric if x != "age"]:
        df[c] = df[c].fillna(0.0)
    df["repeat_ratio"] = 1 - df.unique_articles / df.txn_count.clip(lower=1)
    df["age_bin"] = age_bin(df.age)
    return df


def coarse_trouser_features(tx_paths: list[str], article_paths: list[str], customers: pd.DataFrame, train_ids: set[int]) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    train_sql = ",".join(str(int(x)) for x in sorted(train_ids))
    tx_src = f"read_parquet({sql_list(tx_paths)}, union_by_name=true)"
    art_src = f"read_parquet({sql_list(article_paths)}, union_by_name=true)"
    d = con.execute(f"""
        SELECT
            CAST(t.customer_id AS VARCHAR) AS customer_id,
            COUNT(*)::DOUBLE AS ct_txn_count,
            COUNT(DISTINCT t.article_id)::DOUBLE AS ct_unique_articles,
            AVG(CAST(t.price AS DOUBLE) * {PRICE_SCALE}) AS ct_mean_price,
            COALESCE(STDDEV_SAMP(CAST(t.price AS DOUBLE) * {PRICE_SCALE}),0) AS ct_price_std,
            COUNT(DISTINCT CAST(t.t_dat AS DATE))::DOUBLE AS ct_active_days,
            DATE_DIFF('day', MAX(CAST(t.t_dat AS DATE)), DATE '{HISTORY_END}')::DOUBLE AS ct_recency_days,
            AVG(CASE WHEN a.garment_group_name='Trousers Denim' THEN 1.0 ELSE 0.0 END) AS denim_share,
            AVG(CASE WHEN a.garment_group_name='Trousers' THEN 1.0 ELSE 0.0 END) AS woven_trouser_share,
            AVG(CASE WHEN a.garment_group_name LIKE 'Jersey%' THEN 1.0 ELSE 0.0 END) AS jersey_share,
            AVG(CASE WHEN a.colour_group_name='Black' THEN 1.0 ELSE 0.0 END) AS black_share,
            AVG(CASE WHEN a.colour_group_name LIKE '%Blue%' OR a.colour_group_name='Blue' THEN 1.0 ELSE 0.0 END) AS blue_share,
            AVG(CASE WHEN a.colour_group_name LIKE '%Grey%' OR a.colour_group_name='Grey' THEN 1.0 ELSE 0.0 END) AS grey_share,
            AVG(CASE WHEN a.colour_group_name LIKE '%Khaki%' THEN 1.0 ELSE 0.0 END) AS khaki_share,
            AVG(CASE WHEN a.graphical_appearance_name='Solid' THEN 1.0 ELSE 0.0 END) AS solid_share,
            AVG(CASE WHEN a.graphical_appearance_name='Denim' THEN 1.0 ELSE 0.0 END) AS denim_app_share,
            AVG(CASE WHEN a.graphical_appearance_name='Melange' THEN 1.0 ELSE 0.0 END) AS melange_share,
            AVG(CASE WHEN a.graphical_appearance_name IN ('All over pattern','Stripe','Check','Dot','Other pattern') THEN 1.0 ELSE 0.0 END) AS pattern_share
        FROM {tx_src} t
        LEFT JOIN {art_src} a USING(article_id)
        WHERE CAST(t.t_dat AS DATE) BETWEEN DATE '{HISTORY_START}' AND DATE '{HISTORY_END}'
          AND CAST(t.sales_channel_id AS INTEGER)=1
          AND CAST(t.article_id AS BIGINT) IN ({train_sql})
        GROUP BY customer_id
    """).df()
    con.close()
    out = customers[["customer_id", "age", "age_bin"]].merge(d, on="customer_id", how="left")
    for c in out.columns:
        if c not in ["customer_id", "age_bin", "age"]:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def cluster_types(df: pd.DataFrame, kind: str, k: int, feature_cols: list[str], active_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    active = d[d[active_col] > 0].copy()
    inactive = d[d[active_col] <= 0].copy()
    x = active[feature_cols].copy()
    for c in feature_cols:
        if c in ["txn_count", "unique_articles", "active_days", "span_days", "last_30d", "last_90d", "last_180d", "ct_txn_count", "ct_unique_articles", "ct_active_days"]:
            x[c] = np.log1p(np.clip(x[c], 0, None))
    scaler = RobustScaler(quantile_range=(10, 90))
    xs = scaler.fit_transform(x)
    km = MiniBatchKMeans(n_clusters=k, random_state=SEED, n_init=10, batch_size=8192, max_iter=200)
    labels = km.fit_predict(xs)
    active["type_id"] = [f"{kind}B{int(v)+1:02d}" for v in labels]
    inactive["type_id"] = [f"{kind}N_{b}" for b in inactive.age_bin.astype(str)]
    members = pd.concat([active[["customer_id", "type_id"]], inactive[["customer_id", "type_id"]]], ignore_index=True)
    merged = d.merge(members, on="customer_id", how="left")
    total = len(merged)
    rows = []
    for tid, g in merged.groupby("type_id", sort=True):
        r = {"type_id": tid, "n_customers": int(len(g)), "weight": float(len(g) / total)}
        for c in ["age", active_col, "mean_price", "ct_mean_price", "recency_days", "ct_recency_days"] + STYLE_COLS + ["trouser_share", "leggings_share", "dress_share", "skirt_share", "online_share"]:
            if c in g.columns:
                r[c] = float(pd.to_numeric(g[c], errors="coerce").fillna(0).mean())
        rows.append(r)
    types = pd.DataFrame(rows).sort_values("type_id").reset_index(drop=True)
    return members, types


def smooth_profiles(types: pd.DataFrame, evidence_strength_col: str, global_prior: dict[str, float]) -> pd.DataFrame:
    t = types.copy()
    strength = pd.to_numeric(t.get(evidence_strength_col, 0), errors="coerce").fillna(0).to_numpy(float) * t.n_customers.to_numpy(float)
    # cap evidence so giant segments do not become numerically absolute
    strength = np.minimum(strength, 1_000_000.0)
    for c in STYLE_COLS:
        obs = pd.to_numeric(t.get(c, global_prior[c]), errors="coerce").fillna(global_prior[c]).to_numpy(float)
        t[c] = (strength * obs + PRIOR_STRENGTH * global_prior[c]) / (strength + PRIOR_STRENGTH)
    return t


def build_utility_signal(types: pd.DataFrame, catalog: pd.DataFrame, probs: pd.DataFrame, train_ids: set[int], global_prior: dict[str, float]) -> pd.DataFrame:
    t = types.copy()
    weights = t.weight.to_numpy(float)
    weights = weights / weights.sum()
    # Log enrichment relative to a common prior makes the same provider policy apply to every representation.
    eps = 1e-4
    enrichment = {c: np.log((np.clip(t[c].to_numpy(float), eps, 1) + eps) / (global_prior[c] + eps)) for c in STYLE_COLS}
    grids = probs.groupby("article_id").offer_price.apply(lambda s: sorted(set(round(float(v), 2) for v in s))).to_dict()
    cat = catalog.set_index("article_id")
    rows = []
    for aid in sorted(grids):
        r = cat.loc[int(aid)]
        garment = str(r.garment_group_name)
        color = str(r.colour_group_name)
        app = str(r.graphical_appearance_name)
        ref = float(r.price_mid)
        per_type = np.zeros(len(t), float)
        if garment == "Trousers Denim":
            per_type += 1.5 * enrichment["denim_share"]
        elif garment == "Trousers":
            per_type += 1.2 * enrichment["woven_trouser_share"]
        elif garment.startswith("Jersey"):
            per_type += 1.2 * enrichment["jersey_share"]
        if color == "Black":
            per_type += 0.8 * enrichment["black_share"]
        if "Blue" in color or color == "Blue":
            per_type += 0.8 * enrichment["blue_share"]
        if "Grey" in color or color == "Grey":
            per_type += 0.8 * enrichment["grey_share"]
        if "Khaki" in color:
            per_type += 0.8 * enrichment["khaki_share"]
        if app == "Solid":
            per_type += 0.6 * enrichment["solid_share"]
        elif app == "Denim":
            per_type += 0.6 * enrichment["denim_app_share"]
        elif app == "Melange":
            per_type += 0.6 * enrichment["melange_share"]
        elif app in {"All over pattern", "Stripe", "Check", "Dot", "Other pattern"}:
            per_type += 0.6 * enrichment["pattern_share"]
        for price in grids[int(aid)]:
            utility = per_type - PRICE_PENALTY_PER_10USD / 20.0 * ((float(price) - ref) / 10.0)
            rows.append((int(aid), round(float(price), 2), float(np.dot(weights, utility))))
    signal = pd.DataFrame(rows, columns=["article_id", "offer_price", "signal"])
    mask = signal.article_id.isin(train_ids)
    mu = float(signal.loc[mask, "signal"].mean())
    sd = float(signal.loc[mask, "signal"].std())
    if not np.isfinite(sd) or sd < 1e-8:
        sd = 1.0
    signal["signal"] = np.clip((signal.signal - mu) / sd, -8, 8)
    return signal


def main():
    train_ids = set(pd.read_csv(SPLIT / "train_products.csv").article_id.astype(int))
    test_ids = set(pd.read_csv(SPLIT / "test_products.csv").article_id.astype(int))
    if len(train_ids) != 60 or len(test_ids) != 40 or train_ids & test_ids:
        raise RuntimeError("Unexpected split_000")

    manifest = {
        "protocol": "CLEAN-ALL-CATEGORY-TYPES-v1",
        "history_start": HISTORY_START,
        "history_end": HISTORY_END,
        "heldout_rule": "40 split_000 test article_ids removed before every all-category behavioral feature",
        "source": HF_REPO,
        "source_revision": HF_REVISION,
        "source_use": "non-commercial research only; raw data downloaded at runtime and never committed",
        "primary_gate": PRIMARY_GATE,
        "provider": "fixed empirical semantic-utility policy; no held-out demand labels; no LLM calls in this gate",
        "k_coarse": K_COARSE,
        "k_rich": K_RICH,
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    tx_paths = hf_parquet_files("transactions/train/")
    article_paths = hf_parquet_files("articles/train/")
    customer_paths = hf_parquet_files("customers/train/")

    full = build_customer_features(tx_paths, article_paths, customer_paths, test_ids)
    coarse = coarse_trouser_features(tx_paths, article_paths, full, train_ids)

    rich_features = [
        "txn_count", "unique_articles", "mean_price", "price_std", "active_days", "recency_days", "span_days",
        "last_30d", "last_90d", "last_180d", "online_share", "trouser_share", "leggings_share", "dress_share", "skirt_share",
    ] + STYLE_COLS
    coarse_features = ["ct_txn_count", "ct_unique_articles", "ct_mean_price", "ct_price_std", "ct_active_days", "ct_recency_days"] + STYLE_COLS

    _, rich_types = cluster_types(full, "R", K_RICH, rich_features, "txn_count")
    _, coarse_types = cluster_types(coarse, "C", K_COARSE, coarse_features, "ct_txn_count")

    # Common global prior is estimated only after held-out products have been removed.
    active = full[full.txn_count > 0]
    global_prior = {c: float(active[c].mean()) for c in STYLE_COLS}
    rich_types = smooth_profiles(rich_types, "txn_count", global_prior)
    coarse_types = smooth_profiles(coarse_types, "ct_txn_count", global_prior)
    rich_types.to_csv(ROOT / "clean_rich_types.csv", index=False)
    coarse_types.to_csv(ROOT / "clean_coarse_types.csv", index=False)
    (ROOT / "global_style_prior.json").write_text(json.dumps(global_prior, indent=2))

    catalog = pd.read_csv("outputs/research/fresh_relative_utility_v1/catalog.csv")
    catalog.article_id = catalog.article_id.astype(int)
    probs = load_probability_rows(Path("outputs/responses/llm_responses_online_top100.csv"))
    sales = load_sales(Path("outputs/products/sales_top100_online.csv"))
    sales = sales[sales.demand > 0].copy()

    signals = {
        "coarse_trouser_types_empirical_utility": build_utility_signal(coarse_types, catalog, probs, train_ids, global_prior),
        "clean_all_category_types_empirical_utility": build_utility_signal(rich_types, catalog, probs, train_ids, global_prior),
    }
    fits, summaries, pairs = {}, {}, {}
    for name, signal in signals.items():
        merged = sales.merge(signal, on=["article_id", "offer_price"], how="inner")
        train = merged[merged.article_id.isin(train_ids)].copy()
        test = merged[merged.article_id.isin(test_ids)].copy()
        fit = fit_calibration(train)
        pair, summary = score_model(name, test, fit)
        fits[name] = fit
        summaries[name] = summary
        pairs[name] = pair
        pair.to_csv(ROOT / f"pair_scores_{name}.csv", index=False)

    paper = {str(r.metric): float(r.value) for r in pd.read_csv(SPLIT / "evaluation/test/llm-mix-cal_summary.csv").itertuples(index=False)}
    prior_best = json.loads(Path("outputs/research/fresh_relative_utility_v1/result.json").read_text())["summary"]["fresh_gpt56_learned_type_relative_utility"]
    summaries["paper_llm_mix_cal"] = paper
    summaries["previous_fresh_gpt56"] = prior_best

    c = summaries["coarse_trouser_types_empirical_utility"]
    r = summaries["clean_all_category_types_empirical_utility"]
    metrics = ["zt_avg_nll", "zt_avg_crps", "mae", "rmse"]
    effects = {m: {"absolute": r[m] - c[m], "relative": (r[m] - c[m]) / c[m]} for m in metrics}
    gate = r["zt_avg_nll"] < c["zt_avg_nll"] and r["zt_avg_crps"] < c["zt_avg_crps"] and r["mae"] <= 1.01 * c["mae"] and r["rmse"] <= 1.01 * c["rmse"]
    decision = "ADVANCE_CLEAN_RICH_TYPES_TO_GPT56" if gate else "DO_NOT_ADVANCE_CLEAN_RICH_TYPES_TO_GPT56"
    boot = product_bootstrap(pairs["coarse_trouser_types_empirical_utility"], pairs["clean_all_category_types_empirical_utility"])

    result = {
        "protocol": manifest["protocol"],
        "decision": decision,
        "primary_gate_passed": bool(gate),
        "summary": summaries,
        "fits": fits,
        "effects_rich_vs_coarse": effects,
        "product_bootstrap_rich_vs_coarse": boot,
        "population": {
            "n_customers": int(len(full)),
            "n_active_clean_history": int((full.txn_count > 0).sum()),
            "n_clean_rich_types": int(len(rich_types)),
            "n_clean_coarse_types": int(len(coarse_types)),
            "largest_rich_type_weight": float(rich_types.weight.max()),
            "largest_coarse_type_weight": float(coarse_types.weight.max()),
        },
        "manifest": manifest,
        "interpretation": "This gate tests whether leakage-safe all-category behavioral representation adds held-out demand signal under one fixed empirical relative-utility provider. It does not yet test GPT-5.6 elicitation.",
    }
    (ROOT / "result.json").write_text(json.dumps(result, indent=2))

    order = ["paper_llm_mix_cal", "previous_fresh_gpt56", "coarse_trouser_types_empirical_utility", "clean_all_category_types_empirical_utility"]
    lines = [
        "# Clean All-Category Population Representation",
        "",
        f"**Decision:** {decision}",
        "",
        "Held-out products are removed from the raw H&M transaction history before feature construction.",
        "",
        "| model | NLL | CRPS | MAE | RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in order:
        s = summaries[name]
        lines.append(f"| {name} | {s['zt_avg_nll']:.6f} | {s['zt_avg_crps']:.6f} | {s['mae']:.6f} | {s['rmse']:.6f} |")
    lines += ["", "## Rich vs coarse clean representation", ""]
    for m in metrics:
        lines.append(f"- {m}: {effects[m]['relative']*100:+.3f}%")
    lines += ["", "## Bootstrap", ""]
    for m, v in boot.items():
        lines.append(f"- {m}: 95% CI [{v['ci95_low']:.6f}, {v['ci95_high']:.6f}], P(rich better)={v['p_treatment_better']:.3f}")
    lines += [
        "", "## Population", "",
        f"- customers: {len(full):,}",
        f"- active in clean pre-cutoff history: {(full.txn_count > 0).sum():,}",
        f"- rich types: {len(rich_types)}; largest weight {rich_types.weight.max()*100:.2f}%",
        f"- coarse types: {len(coarse_types)}; largest weight {coarse_types.weight.max()*100:.2f}%",
        "", "This is a zero-LLM scientific gate. Only a passing result earns a fresh GPT-5.6 representation experiment.",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
