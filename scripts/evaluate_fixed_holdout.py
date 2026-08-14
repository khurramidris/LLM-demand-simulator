from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand_sim.config import DEFAULT_MAX_DEMAND_SUPPORT, DEFAULT_OUTPUT_DIR
from demand_sim.data import (
    build_prompting_design_rows,
    load_persona_embeddings,
    load_product_embeddings,
    load_sales,
)
from demand_sim.evaluation import evaluate_binomial_model, evaluate_rounded_gaussian_model
from demand_sim.io import save_pickle
from demand_sim.models.emb import fit_emb
from demand_sim.models.gaussian import fit_gaussian
from demand_sim.models.llm_mix import fit_llm_mix
from demand_sim.models.llm_mix_cal import fit_llm_mix_cal
from demand_sim.research import audit_alpha_table, load_probability_rows_strict, read_holdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all demand models on one fixed, auditable product holdout."
    )
    parser.add_argument("--sales", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--product-embeddings", type=Path, required=True)
    parser.add_argument("--persona-embeddings", type=Path, required=True)
    parser.add_argument("--train-products", type=Path, required=True)
    parser.add_argument("--test-products", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "fixed_holdout_evaluation")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llm-mix", "llm-mix-cal", "emb", "gaussian"],
        choices=["llm-mix", "llm-mix-cal", "emb", "gaussian"],
    )
    parser.add_argument("--exposure-n-values", type=int, nargs="+", default=[100, 150, 200, 250])
    parser.add_argument("--fit-objective", choices=["truncated", "naive"], default="truncated")
    parser.add_argument("--max-personas", type=int, default=50)
    parser.add_argument("--logit-calibration-iters", type=int, default=6)
    parser.add_argument("--embedding-outer-iters", type=int, default=8)
    parser.add_argument("--embedding-adam-steps", type=int, default=150)
    parser.add_argument("--embedding-l2", type=float, default=1.0)
    parser.add_argument("--gaussian-ridge", type=float, default=1e-3)
    parser.add_argument("--support-max", type=int, default=DEFAULT_MAX_DEMAND_SUPPORT)
    parser.add_argument("--solver", default=None)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = read_holdout(args.train_products, args.test_products)
    requested = list(dict.fromkeys(args.models))

    sales = load_sales(args.sales)
    probabilities, response_audit = load_probability_rows_strict(args.responses)
    prompting_rows, persona_ids = build_prompting_design_rows(sales, probabilities)

    product_embeddings, product_cols = load_product_embeddings(args.product_embeddings)
    persona_embeddings, embedding_persona_ids, persona_cols = load_persona_embeddings(
        args.persona_embeddings, args.max_personas
    )
    embedded_rows = prompting_rows.merge(product_embeddings, on="article_id", how="inner")

    observed_ids = set(prompting_rows["article_id"].astype(int).unique())
    missing_train = sorted(set(split.train_ids).difference(observed_ids))
    missing_test = sorted(set(split.test_ids).difference(observed_ids))
    if missing_train or missing_test:
        raise RuntimeError(
            f"Fixed split products missing from prompting rows. "
            f"missing_train={missing_train[:10]}, missing_test={missing_test[:10]}"
        )

    train_prompting = prompting_rows[prompting_rows["article_id"].isin(split.train_ids)].copy()
    test_prompting = prompting_rows[prompting_rows["article_id"].isin(split.test_ids)].copy()
    train_embedded = embedded_rows[embedded_rows["article_id"].isin(split.train_ids)].copy()
    test_embedded = embedded_rows[embedded_rows["article_id"].isin(split.test_ids)].copy()

    models: dict[str, object] = {}
    if "llm-mix" in requested:
        models["llm-mix"] = fit_llm_mix(
            train_prompting, persona_ids, args.exposure_n_values, args.fit_objective, args.solver
        )
    if "llm-mix-cal" in requested:
        models["llm-mix-cal"] = fit_llm_mix_cal(
            train_prompting,
            persona_ids,
            args.exposure_n_values,
            fit_objective=args.fit_objective,
            calibration_iters=args.logit_calibration_iters,
            solver=args.solver,
        )
    if "emb" in requested:
        models["emb"] = fit_emb(
            train_embedded,
            product_cols,
            persona_embeddings,
            embedding_persona_ids,
            persona_cols,
            args.exposure_n_values,
            fit_objective=args.fit_objective,
            outer_iters=args.embedding_outer_iters,
            adam_steps=args.embedding_adam_steps,
            l2=args.embedding_l2,
            solver=args.solver,
        )
    if "gaussian" in requested:
        models["gaussian"] = fit_gaussian(
            train_embedded,
            product_cols,
            weight_decay=args.gaussian_ridge,
        )

    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    fit_records = []
    for name, model in models.items():
        save_pickle(model, model_dir / f"{name}.pkl")
        record = {"model": name}
        for field in ["exposure_n", "fit_objective", "objective_value", "intercept", "slope", "sigma"]:
            if hasattr(model, field):
                value = getattr(model, field)
                if callable(value):
                    value = value()
                record[field] = float(value) if isinstance(value, (int, float)) else str(value)
        if hasattr(model, "alpha_table"):
            alpha = model.alpha_table()
            alpha.to_csv(model_dir / f"{name}_alpha.csv", index=False)
            record.update(audit_alpha_table(alpha))
        fit_records.append(record)
    pd.DataFrame(fit_records).to_csv(args.output_dir / "fit_summary.csv", index=False)

    summaries = []
    for sample_name, prompt_rows, emb_rows in [
        ("train", train_prompting, train_embedded),
        ("test", test_prompting, test_embedded),
    ]:
        sample_dir = args.output_dir / "evaluation" / sample_name
        for name, model in models.items():
            if name == "gaussian":
                _, summary = evaluate_rounded_gaussian_model(
                    emb_rows,
                    model,
                    sample_dir,
                    support_max=args.support_max,
                    split_label=sample_name,
                )
            else:
                rows = emb_rows if name == "emb" else prompt_rows
                _, summary = evaluate_binomial_model(
                    rows,
                    model,
                    sample_dir,
                    split_label=sample_name,
                    seed=args.seed,
                )
            summary.insert(0, "sample", sample_name)
            summaries.append(summary)

    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(args.output_dir / "summary_long.csv", index=False)
    pivot = all_summary.pivot_table(index=["sample", "model"], columns="metric", values="value", aggfunc="mean")
    pivot.reset_index().to_csv(args.output_dir / "summary_wide.csv", index=False)

    metadata = {
        "protocol": "FIXED-PRODUCT-HOLDOUT",
        "train_products": list(split.train_ids),
        "test_products": list(split.test_ids),
        "models": requested,
        "fit_objective": args.fit_objective,
        "response_audit": response_audit,
        "exposure_n_values": args.exposure_n_values,
        "n_train_rows": len(train_prompting),
        "n_test_rows": len(test_prompting),
        "inputs": {
            "sales": str(args.sales),
            "responses": str(args.responses),
            "product_embeddings": str(args.product_embeddings),
            "persona_embeddings": str(args.persona_embeddings),
            "train_products": str(args.train_products),
            "test_products": str(args.test_products),
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote fixed-holdout evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
