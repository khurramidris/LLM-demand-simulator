from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260814
TRAIN_START = pd.Timestamp("2018-09-01")
TRAIN_END = pd.Timestamp("2019-09-19")
SPAN_MONTHS = ((TRAIN_END - TRAIN_START).days + 1) / 30.4375


def qcut_with_rank_fallback(values: pd.Series, labels: list[str]) -> pd.Series:
    out = pd.qcut(values, q=len(labels), labels=labels, duplicates="drop")
    if out.nunique(dropna=True) >= 2:
        return out
    ranked = values.rank(method="average")
    n_bins = min(len(labels), int(ranked.nunique()))
    return pd.qcut(ranked, q=n_bins, labels=labels[:n_bins], duplicates="drop")


def allocate_largest_remainder(counts: pd.Series, n: int) -> pd.Series:
    raw = counts / counts.sum() * n
    base = np.floor(raw).astype(int)
    remaining = int(n - base.sum())
    if remaining > 0:
        order = (raw - base).sort_values(ascending=False, kind="mergesort").index[:remaining]
        base.loc[order] += 1
    excess = int((base - counts).clip(lower=0).sum())
    base = pd.concat([base, counts], axis=1).min(axis=1).astype(int)
    while excess > 0:
        room = counts - base
        eligible = room[room > 0]
        if eligible.empty:
            break
        desirability = (raw - base).loc[eligible.index]
        idx = desirability.sort_values(ascending=False, kind="mergesort").index[0]
        base.loc[idx] += 1
        excess -= 1
    assert int(base.sum()) == n
    return base


def hashed_id(customer_id: str) -> str:
    return hashlib.sha256(str(customer_id).encode("utf-8")).hexdigest()[:16]


def coarse_text(row: pd.Series) -> str:
    return (
        f"H&M customer, age {row.age_bin}; engagement {row.engagement_bin}; "
        f"price tier {row.price_tier}; most-purchased product type {row.top_product_type}."
    )


def rich_text(row: pd.Series) -> str:
    age = "unknown" if pd.isna(row.age) else f"{float(row.age):g}"
    return (
        f"H&M customer with observed age {age}; {int(row.txn_count)} purchases during "
        f"2018-09-01 through 2019-09-19 ({row.txn_per_month:.3f} purchases/month); "
        f"mean paid price ${row.mean_price:.2f}; most-purchased product type {row.top_product_type}."
    )


def read_customer_features(path: Path) -> pd.DataFrame:
    if path.suffix.lower() != ".zip":
        return pd.read_csv(path)
    with zipfile.ZipFile(path) as zf:
        members = [name for name in zf.namelist() if not name.startswith("__MACOSX/") and name.endswith("customer_features.csv")]
        if len(members) != 1:
            raise ValueError(f"Expected exactly one customer_features.csv member, found: {members}")
        with zf.open(members[0]) as handle:
            return pd.read_csv(handle)


def build_matched_states(customer_features_path: Path, n: int = 50) -> pd.DataFrame:
    df = read_customer_features(customer_features_path)
    required = {"customer_id", "age", "txn_count", "mean_price", "top_product_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"customer feature file missing columns: {sorted(missing)}")

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["txn_count"] = pd.to_numeric(df["txn_count"], errors="coerce")
    df["mean_price"] = pd.to_numeric(df["mean_price"], errors="coerce")
    df = df[(df.txn_count > 0) & df.age.notna() & (df.mean_price > 0)].copy()
    df["txn_per_month"] = df["txn_count"] / SPAN_MONTHS
    df["age_bin"] = pd.cut(
        df["age"], bins=[16, 25, 35, 45, 55, 200], right=False,
        labels=["16-24", "25-34", "35-44", "45-54", "55+"],
    )
    df = df[df.age_bin.notna()].copy()
    df["engagement_bin"] = qcut_with_rank_fallback(df["txn_per_month"], ["low", "mid", "high"])
    df["price_tier"] = qcut_with_rank_fallback(df["mean_price"], ["low", "mid", "high"])
    df = df[df.engagement_bin.notna() & df.price_tier.notna()].copy()

    strata_cols = ["age_bin", "engagement_bin", "price_tier"]
    df["stratum"] = df[strata_cols].astype(str).agg("|".join, axis=1)
    counts = df.groupby("stratum", observed=True).size().sort_index()
    alloc = allocate_largest_remainder(counts, n)

    rng = np.random.default_rng(SEED)
    chosen = []
    for stratum in sorted(alloc.index):
        k = int(alloc.loc[stratum])
        if k <= 0:
            continue
        block = df[df.stratum == stratum].copy().sort_values("customer_id", kind="mergesort")
        take = np.sort(rng.choice(len(block), size=k, replace=False))
        chosen.append(block.iloc[take])
    sample = pd.concat(chosen, ignore_index=True)
    assert len(sample) == n

    sample["customer_hash"] = sample.customer_id.astype(str).map(hashed_id)
    sample = sample.sort_values(["stratum", "customer_hash"], kind="mergesort").reset_index(drop=True)
    sample["state_id"] = [f"C{i+1:03d}" for i in range(len(sample))]
    sample["coarse_state"] = sample.apply(coarse_text, axis=1)
    sample["rich_state"] = sample.apply(rich_text, axis=1)

    cols = [
        "state_id", "customer_hash", "stratum", "age_bin", "engagement_bin", "price_tier",
        "top_product_type", "age", "txn_count", "txn_per_month", "mean_price",
        "coarse_state", "rich_state",
    ]
    return sample[cols]


def build_products(product_info_path: Path, query_plan_path: Path) -> pd.DataFrame:
    products = pd.read_csv(product_info_path)
    plan = pd.read_csv(query_plan_path, usecols=["article_id", "prices_json"])
    grids = plan.drop_duplicates("article_id").copy()
    if grids.article_id.duplicated().any():
        raise AssertionError("multiple price grids remain per article")
    cols = [
        "article_id", "prod_name", "product_type_name", "colour_group_name", "graphical_appearance_name",
        "department_name", "index_name", "section_name", "garment_group_name", "detail_desc",
    ]
    out = products[cols].drop_duplicates("article_id").merge(grids, on="article_id", how="inner", validate="one_to_one")
    out["prices"] = out.prices_json.map(json.loads)
    out["n_prices"] = out.prices.map(len)
    out = out.drop(columns="prices").sort_values("article_id").reset_index(drop=True)
    if len(out) != 100:
        raise AssertionError(f"expected 100 products, found {len(out)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--product-info", type=Path, default=Path("outputs/products/product_info_top100_online.csv"))
    ap.add_argument("--query-plan", type=Path, default=Path("outputs/query_plan/plan_top100_online.csv"))
    ap.add_argument("--persona-cells", type=Path, default=Path("outputs/personas/persona_cells.csv"))
    ap.add_argument("--customer-features", type=Path, default=Path("outputs/personas/customer_features.csv.zip"))
    ap.add_argument("--output-dir", type=Path, default=Path("research/experiments/004_blinded_inputs"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    products = build_products(args.product_info, args.query_plan)
    matched = build_matched_states(args.customer_features, n=50)
    paper = pd.read_csv(args.persona_cells).head(50).copy()
    paper_cols = [
        "persona_id", "n_customers", "age_bin", "engagement_bin", "price_tier", "taste_bucket",
        "txn_q25", "txn_q75", "price_q25", "price_q75", "persona_prompt",
    ]
    paper = paper[paper_cols]

    banned = {"demand", "sales", "target", "y", "total_demand", "popularity"}
    for name, frame in {"products": products, "matched": matched, "paper": paper}.items():
        overlap = banned.intersection(map(str.lower, frame.columns))
        if overlap:
            raise AssertionError(f"banned columns in blinded {name}: {sorted(overlap)}")

    products.to_csv(args.output_dir / "blinded_products.csv", index=False)
    matched.to_csv(args.output_dir / "matched_customer_states.csv", index=False)
    paper.to_csv(args.output_dir / "paper_personas_top50.csv", index=False)

    manifest = {
        "experiment": "004_gpt56_persona_richness",
        "seed": SEED,
        "n_products": int(len(products)),
        "n_matched_states": int(len(matched)),
        "n_paper_personas": int(len(paper)),
        "contains_demand_targets": False,
        "product_input_columns": products.columns.tolist(),
        "matched_state_columns": matched.columns.tolist(),
        "paper_persona_columns": paper.columns.tolist(),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
