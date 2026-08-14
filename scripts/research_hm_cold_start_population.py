from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize


SEED = 20260814
N_PERSONAS = 50
OUTCOME_DAYS = 28
CUTOFF = pd.Timestamp("2019-09-20")
TEST_LAUNCH_START = pd.Timestamp("2019-09-20")
TEST_LAUNCH_END = pd.Timestamp("2019-10-17")
VAL_LAUNCH_START = pd.Timestamp("2019-07-26")
VAL_LAUNCH_END = pd.Timestamp("2019-08-22")
TRAIN_LAUNCH_END = pd.Timestamp("2019-06-27")
MIN_MAPPED_BUYERS = 10
TAU_GRID = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
ALPHA_GRID = (1e-5, 1e-4, 1e-3, 1e-2)
ARTICLE_DATASET = "microsoft/hnm-search-data"
ARTICLE_CONFIG = "articles"
ARTICLE_DATASET_REVISION = "ac35fedf926b4a7e7ca4a4303ee275db866abd8f"
EPS = 1e-12


def _read_csv_from_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX/")]
        if not members:
            raise RuntimeError(f"No CSV in {path}")
        preferred = [n for n in members if Path(n).stem in {path.stem.replace('.csv', ''), path.stem}]
        member = preferred[0] if preferred else sorted(members)[0]
        with zf.open(member) as fh:
            return pd.read_csv(fh)


def _qcut_with_rank_fallback(values: pd.Series, labels: list[str]) -> pd.Series:
    out = pd.qcut(values, q=len(labels), labels=labels, duplicates="drop")
    if out.nunique(dropna=True) >= 2:
        return out
    ranked = values.rank(method="average")
    n_bins = min(len(labels), int(ranked.nunique()))
    return pd.qcut(ranked, q=n_bins, labels=labels[:n_bins], duplicates="drop")


def _persona_assignment(customer_features: pd.DataFrame, cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cf = customer_features.copy()
    cf["customer_id"] = cf["customer_id"].astype(str)
    cf["txn_count"] = pd.to_numeric(cf["txn_count"], errors="coerce").fillna(0.0)
    cf["mean_price"] = pd.to_numeric(cf["mean_price"], errors="coerce").fillna(0.0)
    cf["age"] = pd.to_numeric(cf["age"], errors="coerce")
    active = cf[cf["txn_count"] > 0].copy()
    active["age_bin"] = pd.cut(
        active["age"],
        bins=[16, 25, 35, 45, 55, 200],
        right=False,
        labels=["16-24", "25-34", "35-44", "45-54", "55+"],
    )
    span_months = max(((pd.Timestamp("2019-09-19") - pd.Timestamp("2018-09-01")).days + 1) / 30.4375, 1e-6)
    active["txn_per_month"] = active["txn_count"] / span_months
    active["engagement_bin"] = _qcut_with_rank_fallback(active["txn_per_month"], ["low", "mid", "high"])
    active["price_tier"] = _qcut_with_rank_fallback(active["mean_price"], ["low", "mid", "high"])
    top_types = active["top_product_type"].value_counts().head(100).index
    active["taste_bucket"] = np.where(active["top_product_type"].isin(top_types), active["top_product_type"], "OTHER")

    frozen = cells.head(N_PERSONAS).copy().reset_index(drop=True)
    frozen["persona_idx"] = np.arange(len(frozen), dtype=int)
    key_cols = ["age_bin", "engagement_bin", "price_tier", "taste_bucket"]
    for col in key_cols:
        active[col] = active[col].astype("string")
        frozen[col] = frozen[col].astype("string")
    active = active.merge(frozen[key_cols + ["persona_id", "persona_idx"]], on=key_cols, how="left")
    mapped = active.dropna(subset=["persona_idx"]).copy()
    mapped["persona_idx"] = mapped["persona_idx"].astype(int)
    return mapped, frozen


def _load_articles(article_ids: set[int]) -> tuple[pd.DataFrame, str | None]:
    from datasets import load_dataset

    ds = load_dataset(
        ARTICLE_DATASET,
        ARTICLE_CONFIG,
        split="train",
        revision=ARTICLE_DATASET_REVISION,
    )
    fingerprint = getattr(ds, "_fingerprint", None)
    art = ds.to_pandas()
    art["article_id"] = pd.to_numeric(art["article_id"], errors="coerce").astype("Int64")
    art = art[art["article_id"].isin(article_ids)].copy()
    art["article_id"] = art["article_id"].astype(np.int64)
    art = art.drop_duplicates("article_id")
    return art, fingerprint


def _product_text(articles: pd.DataFrame) -> pd.Series:
    fields = [
        "prod_name",
        "product_type_name",
        "graphical_appearance_name",
        "colour_group_name",
        "perceived_colour_value_name",
        "perceived_colour_master_name",
        "department_name",
        "index_name",
        "index_group_name",
        "section_name",
        "garment_group_name",
        "detail_desc",
    ]
    out = []
    for row in articles[fields].fillna("").itertuples(index=False, name=None):
        out.append(" | ".join(str(x).strip() for x in row if str(x).strip()))
    return pd.Series(out, index=articles.index, dtype="string")


def _buyer_counts(tx: pd.DataFrame, customer_to_persona: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    first_sale = tx.groupby("article_id")["date"].min().sort_index()
    tmp = tx[["article_id", "customer_id", "date"]].copy()
    tmp["launch_date"] = tmp["article_id"].map(first_sale)
    tmp["days_from_launch"] = (tmp["date"] - tmp["launch_date"]).dt.days
    tmp = tmp[(tmp["days_from_launch"] >= 0) & (tmp["days_from_launch"] < OUTCOME_DAYS)]
    tmp = tmp.drop_duplicates(["article_id", "customer_id"])
    tmp["persona_idx"] = tmp["customer_id"].map(customer_to_persona)
    tmp = tmp.dropna(subset=["persona_idx"]).copy()
    tmp["persona_idx"] = tmp["persona_idx"].astype(int)
    counts = tmp.groupby(["article_id", "persona_idx"], as_index=False).size().rename(columns={"size": "buyer_count"})
    return first_sale, counts


def _qualified_ids(first_sale: pd.Series, counts: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp) -> list[int]:
    support = counts.groupby("article_id")["buyer_count"].sum()
    launch = first_sale.rename("launch_date").to_frame().join(support.rename("mapped_buyers"), how="left")
    launch["mapped_buyers"] = launch["mapped_buyers"].fillna(0).astype(int)
    mask = launch["launch_date"] <= end
    if start is not None:
        mask &= launch["launch_date"] >= start
    mask &= launch["mapped_buyers"] >= MIN_MAPPED_BUYERS
    return [int(x) for x in launch.index[mask].tolist()]


def _count_matrix(counts: pd.DataFrame, ids: list[int]) -> dict[int, np.ndarray]:
    wanted = counts[counts["article_id"].isin(ids)]
    out: dict[int, np.ndarray] = {}
    for aid, g in wanted.groupby("article_id"):
        v = np.zeros(N_PERSONAS, dtype=float)
        v[g["persona_idx"].to_numpy(dtype=int)] = g["buyer_count"].to_numpy(dtype=float)
        out[int(aid)] = v
    return out


def _repeat_prediction(ids: list[int], p: np.ndarray) -> dict[int, np.ndarray]:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, EPS, None)
    p = p / p.sum()
    return {int(a): p.copy() for a in ids}


def _nll_for_counts(counts_by_id: dict[int, np.ndarray], preds: dict[int, np.ndarray]) -> float:
    total = 0.0
    n = 0.0
    for aid, c in counts_by_id.items():
        p = np.clip(np.asarray(preds[aid], dtype=float), EPS, 1.0)
        p = p / p.sum()
        total += float(np.sum(c * -np.log(p)))
        n += float(c.sum())
    return total / n if n else float("nan")


def _js_divergence(a: np.ndarray, p: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    p = np.asarray(p, dtype=float)
    a = a / a.sum()
    p = np.clip(p, EPS, None)
    p = p / p.sum()
    m = 0.5 * (a + p)
    left = np.where(a > 0, a * np.log(a / m), 0.0)
    right = p * np.log(p / m)
    return float(0.5 * np.sum(left) + 0.5 * np.sum(right))


def _evaluate(model: str, ids: list[int], counts_by_id: dict[int, np.ndarray], preds: dict[int, np.ndarray]) -> tuple[dict, list[dict]]:
    total_loss = 0.0
    total_n = 0.0
    top1 = 0.0
    top5 = 0.0
    js_vals = []
    tv_vals = []
    product_rows = []
    for aid in ids:
        c = counts_by_id[aid]
        n = float(c.sum())
        p = np.clip(np.asarray(preds[aid], dtype=float), EPS, None)
        p = p / p.sum()
        loss_sum = float(np.sum(c * -np.log(p)))
        ranking = np.argsort(-p)
        top1_hits = float(c[ranking[0]])
        top5_hits = float(c[ranking[:5]].sum())
        actual = c / n
        js = _js_divergence(actual, p)
        tv = float(0.5 * np.abs(actual - p).sum())
        total_loss += loss_sum
        total_n += n
        top1 += top1_hits
        top5 += top5_hits
        js_vals.append(js)
        tv_vals.append(tv)
        product_rows.append({
            "article_id": int(aid),
            "model": model,
            "n_buyers": int(n),
            "nll_sum": loss_sum,
            "nll": loss_sum / n,
            "js_divergence": js,
            "tv_distance": tv,
        })
    metrics = {
        "model": model,
        "buyer_nll": total_loss / total_n,
        "buyer_top1": top1 / total_n,
        "buyer_top5": top5 / total_n,
        "macro_js": float(np.mean(js_vals)),
        "macro_tv": float(np.mean(tv_vals)),
        "buyers": int(total_n),
        "articles": int(len(ids)),
    }
    return metrics, product_rows


def _style_centroids(
    tx: pd.DataFrame,
    customer_to_persona: pd.Series,
    article_ids: list[int],
    article_to_row: dict[int, int],
    x_all: sparse.csr_matrix,
) -> sparse.csr_matrix:
    pre = tx[tx["date"] < CUTOFF][["article_id", "customer_id"]].drop_duplicates()
    pre["persona_idx"] = pre["customer_id"].map(customer_to_persona)
    pre = pre.dropna(subset=["persona_idx"])
    pre["persona_idx"] = pre["persona_idx"].astype(int)
    pre = pre[pre["article_id"].isin(article_to_row)]
    grouped = pre.groupby(["article_id", "persona_idx"], as_index=False).size().rename(columns={"size": "n"})
    rows = grouped["article_id"].map(article_to_row).to_numpy(dtype=int)
    cols = grouped["persona_idx"].to_numpy(dtype=int)
    data = grouped["n"].to_numpy(dtype=float)
    cmat = sparse.coo_matrix((data, (rows, cols)), shape=(len(article_ids), N_PERSONAS)).tocsr()
    centroids = cmat.T @ x_all
    centroids = normalize(centroids, norm="l2", axis=1, copy=False)
    return centroids.tocsr()


def _style_predictions(ids: list[int], article_to_row: dict[int, int], x_all: sparse.csr_matrix, centroids: sparse.csr_matrix, prior: np.ndarray, tau: float) -> dict[int, np.ndarray]:
    row_ids = [article_to_row[a] for a in ids]
    sims = (x_all[row_ids] @ centroids.T).toarray()
    base = np.log(np.clip(prior, EPS, None))[None, :]
    probs = softmax(base + tau * sims, axis=1)
    return {int(a): probs[i] for i, a in enumerate(ids)}


def _fit_content_classifier(
    ids: list[int],
    counts: pd.DataFrame,
    article_to_row: dict[int, int],
    x_all: sparse.csr_matrix,
    alpha: float,
) -> SGDClassifier:
    sub = counts[counts["article_id"].isin(ids)].copy()
    if sub.empty:
        raise RuntimeError("No supervised buyer-composition rows")
    rows = sub["article_id"].map(article_to_row).to_numpy(dtype=int)
    y = sub["persona_idx"].to_numpy(dtype=int)
    w = sub["buyer_count"].to_numpy(dtype=float)
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(alpha),
        max_iter=4000,
        tol=1e-6,
        random_state=SEED,
        average=True,
    )
    clf.fit(x_all[rows], y, sample_weight=w)
    return clf


def _classifier_predictions(clf: SGDClassifier, ids: list[int], article_to_row: dict[int, int], x_all: sparse.csr_matrix, fallback: np.ndarray) -> dict[int, np.ndarray]:
    rows = [article_to_row[a] for a in ids]
    raw = clf.predict_proba(x_all[rows])
    out = []
    for i in range(len(ids)):
        p = np.asarray(fallback, dtype=float).copy() * 1e-6
        p[np.asarray(clf.classes_, dtype=int)] += raw[i]
        p = np.clip(p, EPS, None)
        p = p / p.sum()
        out.append(p)
    return {int(a): out[i] for i, a in enumerate(ids)}


def _cluster_bootstrap_diff(product_rows: pd.DataFrame, model: str, baseline: str, rng: np.random.Generator, n_boot: int = 2000) -> tuple[float, float, float]:
    m = product_rows[product_rows["model"] == model].set_index("article_id")
    b = product_rows[product_rows["model"] == baseline].set_index("article_id")
    ids = sorted(set(m.index).intersection(b.index))
    diff_sum = (m.loc[ids, "nll_sum"] - b.loc[ids, "nll_sum"]).to_numpy(dtype=float)
    n = m.loc[ids, "n_buyers"].to_numpy(dtype=float)
    point = float(diff_sum.sum() / n.sum())
    boots = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        idx = rng.integers(0, len(ids), size=len(ids))
        boots[k] = float(diff_sum[idx].sum() / n[idx].sum())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi)


def _top_values(df: pd.DataFrame, col: str, k: int = 3) -> str:
    if col not in df or df.empty:
        return "unknown"
    vals = df[col].dropna().astype(str)
    vals = vals[vals.str.len() > 0]
    if vals.empty:
        return "unknown"
    return ", ".join(vals.value_counts().head(k).index.tolist())


def _write_query_plan(
    output_dir: Path,
    test_ids: list[int],
    counts_by_id: dict[int, np.ndarray],
    frozen_cells: pd.DataFrame,
    tx: pd.DataFrame,
    customer_to_persona: pd.Series,
    articles: pd.DataFrame,
) -> dict:
    article_lookup = articles.set_index("article_id")
    # Preregistered selection uses support only, never composition values in prompts.
    selected = sorted(test_ids, key=lambda a: (-int(counts_by_id[a].sum()), int(a)))[:50]

    pre = tx[tx["date"] < CUTOFF].copy()
    pre["persona_idx"] = pre["customer_id"].map(customer_to_persona)
    pre = pre.dropna(subset=["persona_idx"]).copy()
    pre["persona_idx"] = pre["persona_idx"].astype(int)
    pre = pre.merge(
        articles[["article_id", "prod_name", "colour_group_name", "graphical_appearance_name", "garment_group_name", "section_name"]],
        on="article_id",
        how="left",
    )

    rich = {}
    for pidx in range(N_PERSONAS):
        g = pre[pre["persona_idx"] == pidx]
        rich[pidx] = {
            "top_colours": _top_values(g, "colour_group_name", 4),
            "top_graphics": _top_values(g, "graphical_appearance_name", 3),
            "top_garments": _top_values(g, "garment_group_name", 3),
            "top_sections": _top_values(g, "section_name", 3),
            "representative_products": _top_values(g, "prod_name", 5),
            "trouser_unique_products": int(g["article_id"].nunique()),
            "trouser_purchase_rows": int(len(g)),
            "trouser_mean_paid_price_raw": float(pd.to_numeric(g["price"], errors="coerce").mean()) if len(g) else None,
        }

    rows = []
    for aid in selected:
        a = article_lookup.loc[aid]
        product = (
            f"Product name: {a.get('prod_name', '')}\n"
            f"Type: {a.get('product_type_name', '')}\n"
            f"Colour: {a.get('colour_group_name', '')}\n"
            f"Appearance: {a.get('graphical_appearance_name', '')}\n"
            f"Section: {a.get('section_name', '')}\n"
            f"Garment group: {a.get('garment_group_name', '')}\n"
            f"Description: {a.get('detail_desc', '')}"
        )
        for pidx, c in frozen_cells.iterrows():
            coarse = (
                f"You represent an H&M customer segment. Age: {c['age_bin']}. "
                f"Engagement: {c['engagement_bin']}. Typical paid-price tier: {c['price_tier']}. "
                f"Most common historical product type: {c['taste_bucket']}.\n\n{product}\n\n"
                "Estimate this segment's relative affinity for buying this newly appearing product. "
                "Return JSON exactly: {\"p_buy\": <number from 0 to 1>, \"reason\": \"<=25 words\"}."
            )
            r = rich[int(pidx)]
            rich_prompt = (
                f"You represent an H&M customer segment. Age: {c['age_bin']}. "
                f"Engagement: {c['engagement_bin']}. Typical paid-price tier: {c['price_tier']}. "
                f"Most common historical product type: {c['taste_bucket']}.\n"
                f"Pre-cutoff trouser style evidence: top colours {r['top_colours']}; "
                f"top appearances {r['top_graphics']}; top garment groups {r['top_garments']}; "
                f"top sections {r['top_sections']}; representative products {r['representative_products']}.\n\n"
                f"{product}\n\nEstimate this segment's relative affinity for buying this newly appearing product. "
                "Return JSON exactly: {\"p_buy\": <number from 0 to 1>, \"reason\": \"<=25 words\"}."
            )
            rows.append({"article_id": int(aid), "persona_id": c["persona_id"], "persona_idx": int(pidx), "arm": "LLM-C", "prompt": coarse})
            rows.append({"article_id": int(aid), "persona_id": c["persona_id"], "persona_idx": int(pidx), "arm": "LLM-R", "prompt": rich_prompt})

    plan = pd.DataFrame(rows).sort_values(["arm", "article_id", "persona_idx"]).reset_index(drop=True)
    payload = plan[["article_id", "persona_id", "persona_idx", "arm", "prompt"]].to_csv(index=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    plan.to_csv(output_dir / "llm_query_plan.csv", index=False)
    (output_dir / "llm_query_plan.sha256").write_text(digest + "\n", encoding="utf-8")
    return {
        "selected_products": len(selected),
        "queries_total": int(len(plan)),
        "queries_per_arm": int(len(plan) // 2),
        "sha256": digest,
    }


def _write_markdown(summary: dict, out: Path) -> None:
    lines = [
        "# Experiment 006 — Cold-Start Population Benchmark — Stage A Results",
        "",
        f"Status: **{summary['decision']['status']}**",
        "",
        "## Benchmark size",
        "",
        f"- Qualified cold test articles: **{summary['test']['qualified_articles']}**",
        f"- Mapped unique buyer events: **{summary['test']['mapped_buyers']}**",
        f"- Median mapped buyers/article: **{summary['test']['median_mapped_buyers']:.1f}**",
        f"- Test launch window: **{summary['test']['launch_window'][0]} → {summary['test']['launch_window'][1]}**",
        "",
        "## Stage-A metrics",
        "",
        "| Model | Buyer NLL ↓ | Top-1 ↑ | Top-5 ↑ | Macro JS ↓ | Macro TV ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in summary["metrics"]:
        lines.append(
            f"| {m['model']} | {m['buyer_nll']:.6f} | {m['buyer_top1']:.4f} | {m['buyer_top5']:.4f} | {m['macro_js']:.6f} | {m['macro_tv']:.6f} |"
        )
    lines += [
        "",
        "## Product-specific headroom",
        "",
        f"- Best non-product prior: **{summary['decision']['best_prior']}**",
        f"- Best product-specific model: **{summary['decision']['best_product_model']}**",
        f"- Relative NLL improvement: **{100 * summary['decision']['relative_nll_improvement']:.2f}%**",
        f"- Cluster-bootstrap ΔNLL 95% CI (product model − prior): **[{summary['decision']['bootstrap_ci'][0]:.6f}, {summary['decision']['bootstrap_ci'][1]:.6f}]**",
        "",
        "## Gate",
        "",
        f"- ≥30 qualified articles: **{summary['decision']['gate_articles']}**",
        f"- ≥500 mapped buyers: **{summary['decision']['gate_buyers']}**",
        f"- median ≥10 buyers/article: **{summary['decision']['gate_median']}**",
        f"- ≥1% product-specific NLL improvement: **{summary['decision']['gate_effect']}**",
        f"- bootstrap CI strictly below zero: **{summary['decision']['gate_ci']}**",
        f"- Proceed to LLM challenge: **{summary['decision']['proceed_to_llm']}**",
        "",
        "## Interpretation",
        "",
        summary["decision"]["interpretation"],
        "",
        "This benchmark predicts conditional buyer composition for first-sale-proxy cold products. It does not establish exposure-conditioned conversion, inventory-conditioned demand, causal price elasticity, or a verified listing date.",
        "",
        f"New LLM/API calls made: **{summary['new_llm_calls']}**.",
    ]
    if summary.get("query_plan"):
        lines += [
            "",
            "## Frozen Stage-B query plan",
            "",
            f"- Products: **{summary['query_plan']['selected_products']}**",
            f"- Queries per arm: **{summary['query_plan']['queries_per_arm']}**",
            f"- SHA-256: `{summary['query_plan']['sha256']}`",
        ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transactions", type=Path, default=Path("outputs/products/txns_trousers_online.csv.zip"))
    parser.add_argument("--customer-features", type=Path, default=Path("outputs/personas/customer_features.csv.zip"))
    parser.add_argument("--persona-cells", type=Path, default=Path("outputs/personas/persona_cells.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tx = _read_csv_from_zip(args.transactions)
    required = {"t_dat", "customer_id", "article_id", "price", "sales_channel_id"}
    missing = required - set(tx.columns)
    if missing:
        raise RuntimeError(f"Missing transaction columns: {sorted(missing)}")
    tx = tx[list(required)].copy()
    tx["date"] = pd.to_datetime(tx.pop("t_dat"))
    tx["customer_id"] = tx["customer_id"].astype(str)
    tx["article_id"] = pd.to_numeric(tx["article_id"], errors="raise").astype(np.int64)
    tx["price"] = pd.to_numeric(tx["price"], errors="coerce")
    if not (tx["sales_channel_id"] == 1).all():
        raise AssertionError("Expected committed trouser source to be online-only")

    customer_features = _read_csv_from_zip(args.customer_features)
    cells = pd.read_csv(args.persona_cells)
    mapped_customers, frozen_cells = _persona_assignment(customer_features, cells)
    customer_to_persona = mapped_customers.set_index("customer_id")["persona_idx"]

    first_sale, buyer_counts = _buyer_counts(tx, customer_to_persona)
    train_ids = _qualified_ids(first_sale, buyer_counts, None, TRAIN_LAUNCH_END)
    val_ids = _qualified_ids(first_sale, buyer_counts, VAL_LAUNCH_START, VAL_LAUNCH_END)
    test_ids = _qualified_ids(first_sale, buyer_counts, TEST_LAUNCH_START, TEST_LAUNCH_END)

    article_ids = set(int(x) for x in tx["article_id"].unique())
    articles, hf_fingerprint = _load_articles(article_ids)
    coverage = len(set(articles["article_id"]).intersection(article_ids)) / max(len(article_ids), 1)
    if coverage < 0.99:
        raise RuntimeError(f"Article metadata coverage too low: {coverage:.4f}")

    articles = articles[articles["article_id"].isin(first_sale.index)].copy().sort_values("article_id").reset_index(drop=True)
    articles["product_text"] = _product_text(articles)
    article_list = articles["article_id"].astype(int).tolist()
    article_to_row = {int(a): i for i, a in enumerate(article_list)}

    # Drop any launch ids without metadata before labels are evaluated.
    train_ids = [a for a in train_ids if a in article_to_row]
    val_ids = [a for a in val_ids if a in article_to_row]
    test_ids = [a for a in test_ids if a in article_to_row]

    pre_metadata_ids = [a for a in article_list if first_sale.loc[a] < CUTOFF]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=8000,
        sublinear_tf=True,
        norm="l2",
    )
    fit_text = articles.set_index("article_id").loc[pre_metadata_ids, "product_text"].astype(str).tolist()
    vectorizer.fit(fit_text)
    x_all = vectorizer.transform(articles["product_text"].astype(str).tolist()).tocsr()

    test_counts = _count_matrix(buyer_counts, test_ids)
    val_counts = _count_matrix(buyer_counts, val_ids)

    p0_counts = frozen_cells["n_customers"].to_numpy(dtype=float)
    p0 = p0_counts / p0_counts.sum()

    pre_pairs = tx[tx["date"] < CUTOFF][["article_id", "customer_id"]].drop_duplicates()
    pre_pairs["persona_idx"] = pre_pairs["customer_id"].map(customer_to_persona)
    pre_pairs = pre_pairs.dropna(subset=["persona_idx"])
    p1_counts = np.bincount(pre_pairs["persona_idx"].astype(int).to_numpy(), minlength=N_PERSONAS).astype(float)
    p1 = (p1_counts + 1.0) / (p1_counts.sum() + N_PERSONAS)

    preds_test: dict[str, dict[int, np.ndarray]] = {
        "P0_population_prior": _repeat_prediction(test_ids, p0),
        "P1_trouser_buyer_prior": _repeat_prediction(test_ids, p1),
    }

    centroids = _style_centroids(tx, customer_to_persona, article_list, article_to_row, x_all)
    tau_scores = []
    for tau in TAU_GRID:
        pred = _style_predictions(val_ids, article_to_row, x_all, centroids, p1, tau)
        tau_scores.append((float(tau), _nll_for_counts(val_counts, pred)))
    selected_tau, selected_tau_nll = min(tau_scores, key=lambda x: x[1])
    preds_test["P2_style_profile"] = _style_predictions(test_ids, article_to_row, x_all, centroids, p1, selected_tau)

    alpha_scores = []
    p3_available = bool(train_ids and val_ids)
    if p3_available:
        for alpha in ALPHA_GRID:
            clf = _fit_content_classifier(train_ids, buyer_counts, article_to_row, x_all, alpha)
            pred = _classifier_predictions(clf, val_ids, article_to_row, x_all, p1)
            alpha_scores.append((float(alpha), _nll_for_counts(val_counts, pred)))
        selected_alpha, selected_alpha_nll = min(alpha_scores, key=lambda x: x[1])
        refit_ids = sorted(set(train_ids).union(val_ids))
        clf = _fit_content_classifier(refit_ids, buyer_counts, article_to_row, x_all, selected_alpha)
        preds_test["P3_supervised_content"] = _classifier_predictions(clf, test_ids, article_to_row, x_all, p1)
    else:
        selected_alpha, selected_alpha_nll = None, None

    metrics = []
    product_rows = []
    for model, pred in preds_test.items():
        m, rows = _evaluate(model, test_ids, test_counts, pred)
        metrics.append(m)
        product_rows.extend(rows)
    metrics_df = pd.DataFrame(metrics).sort_values("buyer_nll").reset_index(drop=True)
    product_df = pd.DataFrame(product_rows)

    metric_map = {m["model"]: m for m in metrics}
    best_prior = min(("P0_population_prior", "P1_trouser_buyer_prior"), key=lambda k: metric_map[k]["buyer_nll"])
    product_models = [m for m in ("P2_style_profile", "P3_supervised_content") if m in metric_map]
    best_product = min(product_models, key=lambda k: metric_map[k]["buyer_nll"])
    prior_nll = metric_map[best_prior]["buyer_nll"]
    product_nll = metric_map[best_product]["buyer_nll"]
    relative_improvement = float((prior_nll - product_nll) / prior_nll)
    rng = np.random.default_rng(SEED)
    _, ci_lo, ci_hi = _cluster_bootstrap_diff(product_df, best_product, best_prior, rng, n_boot=2000)

    supports = np.array([test_counts[a].sum() for a in test_ids], dtype=float)
    test_buyers = int(supports.sum())
    median_support = float(np.median(supports)) if len(supports) else 0.0
    gate_articles = len(test_ids) >= 30
    gate_buyers = test_buyers >= 500
    gate_median = median_support >= 10
    gate_effect = relative_improvement >= 0.01
    gate_ci = ci_hi < 0.0
    proceed = bool(gate_articles and gate_buyers and gate_median and gate_effect and gate_ci)

    if proceed:
        status = "STAGE A PASSED — LLM POPULATION CHALLENGE EARNED"
        interpretation = (
            "Cold-product metadata contains statistically reliable signal about which frozen H&M persona groups become buyers, beyond generic population/trouser-buyer priors. "
            "H&M therefore supports a meaningful conditional cold-start population challenge. This does not yet show an LLM advantage; P3 is the hurdle an LLM population must beat."
        )
    else:
        status = "STAGE A FAILED — DO NOT SPEND LLM CALLS"
        interpretation = (
            "Under the preregistered construction, cold-product buyer composition does not clear every size/effect/uncertainty gate. "
            "Do not spend LLM calls trying to force an H&M persona win; keep H&M as an engineering/reproduction benchmark and seek exposure/choice-ground-truth data for flagship validation."
        )

    query_plan = None
    if proceed:
        query_plan = _write_query_plan(args.output_dir, test_ids, test_counts, frozen_cells, tx, customer_to_persona, articles)

    launch_table = first_sale.rename("launch_date").to_frame()
    support_table = buyer_counts.groupby("article_id")["buyer_count"].sum().rename("mapped_buyers")
    launch_table = launch_table.join(support_table, how="left").fillna({"mapped_buyers": 0})
    launch_table.loc[test_ids].reset_index().to_csv(args.output_dir / "qualified_test_articles.csv", index=False)
    metrics_df.to_csv(args.output_dir / "metrics.csv", index=False)
    product_df.to_csv(args.output_dir / "product_metrics.csv", index=False)
    pd.DataFrame(tau_scores, columns=["tau", "validation_buyer_nll"]).to_csv(args.output_dir / "p2_tau_validation.csv", index=False)
    pd.DataFrame(alpha_scores, columns=["alpha", "validation_buyer_nll"]).to_csv(args.output_dir / "p3_alpha_validation.csv", index=False)

    summary = {
        "experiment": "006_cold_start_population_benchmark",
        "source": {
            "transaction_rows": int(len(tx)),
            "transaction_articles": int(tx["article_id"].nunique()),
            "mapped_customers_top50": int(mapped_customers["customer_id"].nunique()),
            "article_metadata_dataset": ARTICLE_DATASET,
            "article_metadata_config": ARTICLE_CONFIG,
            "article_metadata_revision": ARTICLE_DATASET_REVISION,
            "article_metadata_fingerprint": hf_fingerprint,
            "article_metadata_coverage": float(coverage),
        },
        "time_design": {
            "cutoff": CUTOFF.date().isoformat(),
            "outcome_days": OUTCOME_DAYS,
            "train_launch_end": TRAIN_LAUNCH_END.date().isoformat(),
            "validation_launch_window": [VAL_LAUNCH_START.date().isoformat(), VAL_LAUNCH_END.date().isoformat()],
            "test_launch_window": [TEST_LAUNCH_START.date().isoformat(), TEST_LAUNCH_END.date().isoformat()],
        },
        "splits": {
            "qualified_train_articles": len(train_ids),
            "qualified_validation_articles": len(val_ids),
            "qualified_test_articles": len(test_ids),
        },
        "test": {
            "qualified_articles": len(test_ids),
            "mapped_buyers": test_buyers,
            "median_mapped_buyers": median_support,
            "launch_window": [TEST_LAUNCH_START.date().isoformat(), TEST_LAUNCH_END.date().isoformat()],
        },
        "tuning": {
            "P2_tau_grid": list(TAU_GRID),
            "P2_selected_tau": selected_tau,
            "P2_selected_validation_nll": selected_tau_nll,
            "P3_alpha_grid": list(ALPHA_GRID),
            "P3_selected_alpha": selected_alpha,
            "P3_selected_validation_nll": selected_alpha_nll,
        },
        "metrics": metrics,
        "decision": {
            "status": status,
            "best_prior": best_prior,
            "best_product_model": best_product,
            "relative_nll_improvement": relative_improvement,
            "bootstrap_ci": [ci_lo, ci_hi],
            "gate_articles": gate_articles,
            "gate_buyers": gate_buyers,
            "gate_median": gate_median,
            "gate_effect": gate_effect,
            "gate_ci": gate_ci,
            "proceed_to_llm": proceed,
            "interpretation": interpretation,
        },
        "query_plan": query_plan,
        "new_llm_calls": 0,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(summary, args.output_dir / "results.md")
    print(json.dumps(summary["decision"], indent=2))


if __name__ == "__main__":
    main()
