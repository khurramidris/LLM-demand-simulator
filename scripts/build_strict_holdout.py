from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand_sim.config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_PRODUCTS_DIR
from demand_sim.preprocessing import PersonaPreprocessConfig, build_personas, build_query_plan
from demand_sim.research import filter_transactions_excluding_articles, make_product_holdout, write_holdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a leakage-safe global product holdout before persona construction. "
            "Held-out products are removed from customer history, personas are rebuilt, "
            "and a new LLM query plan is generated for the fixed product universe."
        )
    )
    parser.add_argument("--sales", type=Path, default=DEFAULT_PRODUCTS_DIR / "sales_top100_online.csv")
    parser.add_argument("--product-info", type=Path, default=DEFAULT_PRODUCTS_DIR / "product_info_top100_online.csv")
    parser.add_argument("--transactions", type=Path, default=DEFAULT_DATA_DIR / "transactions_train.csv")
    parser.add_argument("--customers", type=Path, default=DEFAULT_DATA_DIR / "customers.csv")
    parser.add_argument("--articles", type=Path, default=DEFAULT_DATA_DIR / "articles.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "strict_global_holdout")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--train-start", default="2018-09-01")
    parser.add_argument("--train-end", default="2019-09-19")
    parser.add_argument("--n-personas", type=int, default=100)
    parser.add_argument("--max-personas", type=int, default=50)
    parser.add_argument("--top-taste-k", type=int, default=100)
    parser.add_argument("--max-prices", type=int, default=10)
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sales = pd.read_csv(args.sales)
    product_ids = sorted(sales["article_id"].astype(int).unique().tolist())
    split = make_product_holdout(product_ids, train_frac=args.train_frac, seed=args.seed)

    split_dir = args.output_dir / "split"
    write_holdout(split, split_dir)

    filtered_transactions = args.output_dir / "data" / "transactions_persona_safe.csv"
    filter_stats = filter_transactions_excluding_articles(
        transactions_path=args.transactions,
        output_path=filtered_transactions,
        excluded_article_ids=set(split.test_ids),
        chunksize=args.chunksize,
    )

    personas_dir = args.output_dir / "personas"
    persona_cfg = PersonaPreprocessConfig(
        train_start=args.train_start,
        train_end=args.train_end,
        n_personas=args.n_personas,
        top_taste_k=args.top_taste_k,
        chunksize=args.chunksize,
    )
    customer_features, personas = build_personas(
        transactions_path=filtered_transactions,
        customers_path=args.customers,
        articles_path=args.articles,
        output_dir=personas_dir,
        cfg=persona_cfg,
    )

    query_dir = args.output_dir / "query_plan"
    query_path = query_dir / "plan_strict_global_holdout.pkl"
    plan = build_query_plan(
        sales_path=args.sales,
        product_info_path=args.product_info,
        persona_prompts_path=personas_dir / "persona_prompts.csv",
        output_path=query_path,
        max_prices=args.max_prices,
        max_personas=args.max_personas,
    )

    manifest = {
        "protocol": "STRICT-GLOBAL-HOLDOUT",
        "scientific_intent": (
            "Choose held-out products before customer-state construction and remove all "
            "held-out-product transactions from persona history. This prevents test "
            "products from influencing engagement, paid-price, or taste features."
        ),
        "seed": args.seed,
        "train_frac": args.train_frac,
        "n_products_total": len(product_ids),
        "n_train_products": len(split.train_ids),
        "n_test_products": len(split.test_ids),
        "train_products": list(split.train_ids),
        "test_products": list(split.test_ids),
        "persona_window": {"start": args.train_start, "end": args.train_end},
        "n_persona_cells": len(personas),
        "n_customer_feature_rows": len(customer_features),
        "max_personas_queried": args.max_personas,
        "query_rows": len(plan),
        "transaction_filter": filter_stats,
        "inputs": {
            "sales": str(args.sales),
            "product_info": str(args.product_info),
            "transactions": str(args.transactions),
            "customers": str(args.customers),
            "articles": str(args.articles),
        },
        "outputs": {
            "train_products": str(split_dir / "train_products.csv"),
            "test_products": str(split_dir / "test_products.csv"),
            "filtered_transactions": str(filtered_transactions),
            "persona_prompts": str(personas_dir / "persona_prompts.csv"),
            "query_plan": str(query_path),
        },
    }
    (args.output_dir / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        "Built STRICT-GLOBAL-HOLDOUT: "
        f"{len(split.train_ids)} train products, {len(split.test_ids)} test products, "
        f"{len(personas)} persona cells, {len(plan)} LLM query rows."
    )
    print(f"Removed {filter_stats['removed_rows']} held-out-product transactions from persona history.")
    print(f"Manifest: {args.output_dir / 'protocol_manifest.json'}")


if __name__ == "__main__":
    main()
