from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_hm_cold_start_population as exp
from research_hm_cold_start_population_runner import _load_articles_from_locked_hf_revision


EXP007_START = pd.Timestamp("2019-10-18")
EXP007_END = pd.Timestamp("2020-01-31")
FROZEN_TAU = 8.0
FROZEN_ALPHA = 0.0001
MIN_PRODUCTS = 60
MIN_BUYERS = 1500
MIN_MEDIAN_BUYERS = 10.0
MIN_RELATIVE_IMPROVEMENT = 0.01
BOOTSTRAPS = 2000
SEED = 20260814


def _noop_query_plan(*args, **kwargs):
    # Experiment 007 explicitly forbids LLM/API work. A later experiment may
    # create a query plan only if this confirmation passes.
    return None


def _nll_diff_for_ids(product_df: pd.DataFrame, ids: list[int], model: str, baseline: str) -> float:
    m = product_df[(product_df["model"] == model) & (product_df["article_id"].isin(ids))].set_index("article_id")
    b = product_df[(product_df["model"] == baseline) & (product_df["article_id"].isin(ids))].set_index("article_id")
    common = sorted(set(m.index).intersection(b.index))
    if not common:
        return float("nan")
    diff_sum = (m.loc[common, "nll_sum"] - b.loc[common, "nll_sum"]).to_numpy(dtype=float)
    n = m.loc[common, "n_buyers"].to_numpy(dtype=float)
    return float(diff_sum.sum() / n.sum())


def _bootstrap(product_df: pd.DataFrame, model: str, baseline: str) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    return exp._cluster_bootstrap_diff(product_df, model, baseline, rng, n_boot=BOOTSTRAPS)


def _write_results(summary: dict, out: Path) -> None:
    d = summary["exp007_decision"]
    lines = [
        "# Experiment 007 — Expanded H&M Cold-Start Population Confirmation — Results",
        "",
        f"Status: **{d['status']}**",
        "",
        "## Confirmatory cohort",
        "",
        f"- Launch window: **{EXP007_START.date().isoformat()} → {EXP007_END.date().isoformat()}**",
        f"- Qualified cold products: **{summary['test']['qualified_articles']}**",
        f"- Mapped unique buyers: **{summary['test']['mapped_buyers']}**",
        f"- Median buyers/product: **{summary['test']['median_mapped_buyers']:.1f}**",
        "",
        "## Frozen model metrics",
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
        "## Primary confirmatory comparison: P2 vs P1",
        "",
        f"- P1 NLL: **{d['p1_nll']:.6f}**",
        f"- P2 NLL: **{d['p2_nll']:.6f}**",
        f"- Relative improvement: **{100*d['p2_relative_improvement']:.3f}%**",
        f"- P2−P1 buyer-weighted ΔNLL: **{d['p2_minus_p1_delta_nll']:.6f}**",
        f"- Product-cluster-bootstrap 95% CI: **[{d['p2_minus_p1_ci'][0]:.6f}, {d['p2_minus_p1_ci'][1]:.6f}]**",
        f"- Early-half ΔNLL: **{d['early_half_delta_nll']:.6f}**",
        f"- Late-half ΔNLL: **{d['late_half_delta_nll']:.6f}**",
        "",
        "## Frozen gate",
        "",
        f"- ≥{MIN_PRODUCTS} qualified products: **{d['gate_products']}**",
        f"- ≥{MIN_BUYERS} mapped buyers: **{d['gate_buyers']}**",
        f"- median ≥{MIN_MEDIAN_BUYERS:.0f} buyers/product: **{d['gate_median']}**",
        f"- P2 relative NLL improvement ≥{100*MIN_RELATIVE_IMPROVEMENT:.1f}%: **{d['gate_effect']}**",
        f"- P2−P1 bootstrap CI strictly below zero: **{d['gate_ci']}**",
        f"- favorable direction in both chronological halves: **{d['gate_halves']}**",
        f"- P2 macro-JS degradation ≤5% vs P1: **{d['gate_js']}**",
        f"- Earn direct LLM-population challenge: **{d['earn_llm_challenge']}**",
        "",
        "## Interpretation",
        "",
        d["interpretation"],
        "",
        "No LLM/API calls were made. This remains buyer-composition prediction under a first-observed-sale launch proxy, not exposure-conditioned conversion or causal demand validation.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    # Freeze Experiment 007 cohort and hyperparameters before invoking the
    # Experiment 006 engine. These changes affect only the untouched future
    # test cohort; all pre-cutoff information construction remains identical.
    exp.TEST_LAUNCH_START = EXP007_START
    exp.TEST_LAUNCH_END = EXP007_END
    exp.TAU_GRID = (FROZEN_TAU,)
    exp.ALPHA_GRID = (FROZEN_ALPHA,)
    exp._load_articles = _load_articles_from_locked_hf_revision
    exp._write_query_plan = _noop_query_plan

    exp.main()

    # exp.main writes to --output-dir. Recover that path from the same CLI
    # convention without changing any scientific result.
    import argparse
    import sys

    argv = sys.argv[1:]
    output_dir = None
    for i, token in enumerate(argv):
        if token == "--output-dir" and i + 1 < len(argv):
            output_dir = Path(argv[i + 1])
            break
        if token.startswith("--output-dir="):
            output_dir = Path(token.split("=", 1)[1])
            break
    if output_dir is None:
        raise RuntimeError("--output-dir is required")

    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    product_df = pd.read_csv(output_dir / "product_metrics.csv")
    launches = pd.read_csv(output_dir / "qualified_test_articles.csv", parse_dates=["launch_date"])

    metric_map = {m["model"]: m for m in summary["metrics"]}
    p1 = metric_map["P1_trouser_buyer_prior"]
    p2 = metric_map["P2_style_profile"]
    p3 = metric_map.get("P3_supervised_content")

    delta, ci_lo, ci_hi = _bootstrap(product_df, "P2_style_profile", "P1_trouser_buyer_prior")
    rel = float((p1["buyer_nll"] - p2["buyer_nll"]) / p1["buyer_nll"])

    ordered = launches.sort_values(["launch_date", "article_id"])["article_id"].astype(int).tolist()
    split = len(ordered) // 2
    early_ids = ordered[:split]
    late_ids = ordered[split:]
    early_delta = _nll_diff_for_ids(product_df, early_ids, "P2_style_profile", "P1_trouser_buyer_prior")
    late_delta = _nll_diff_for_ids(product_df, late_ids, "P2_style_profile", "P1_trouser_buyer_prior")

    n_products = int(summary["test"]["qualified_articles"])
    n_buyers = int(summary["test"]["mapped_buyers"])
    median_buyers = float(summary["test"]["median_mapped_buyers"])
    js_ratio = float(p2["macro_js"] / p1["macro_js"]) if p1["macro_js"] > 0 else float("inf")

    gate_products = n_products >= MIN_PRODUCTS
    gate_buyers = n_buyers >= MIN_BUYERS
    gate_median = median_buyers >= MIN_MEDIAN_BUYERS
    gate_effect = rel >= MIN_RELATIVE_IMPROVEMENT
    gate_ci = ci_hi < 0.0
    gate_halves = early_delta < 0.0 and late_delta < 0.0
    gate_js = js_ratio <= 1.05
    earn = bool(gate_products and gate_buyers and gate_median and gate_effect and gate_ci and gate_halves and gate_js)

    if earn:
        status = "CONFIRMED — H&M EARNS DIRECT LLM-POPULATION CHALLENGE"
        interpretation = (
            "On a larger non-overlapping future launch cohort, the frozen pre-cutoff persona style profile predicts buyer composition better than the generic trouser-buyer prior with the preregistered effect size, product-level uncertainty, temporal consistency and divergence criteria. "
            "This confirms population-composition signal in H&M and earns a separate direct LLM-C/LLM-R versus P3 experiment. It does not itself establish an LLM advantage."
        )
    else:
        status = "NOT CONFIRMED — STOP H&M LLM-POPULATION VALIDATION"
        interpretation = (
            "The promising Experiment 006 population-composition result did not satisfy every preregistered confirmatory criterion on the independent expanded future cohort. "
            "Do not spend LLM calls trying to force an H&M population-validation claim."
        )

    decision = {
        "status": status,
        "p1_nll": float(p1["buyer_nll"]),
        "p2_nll": float(p2["buyer_nll"]),
        "p3_nll": None if p3 is None else float(p3["buyer_nll"]),
        "p2_relative_improvement": rel,
        "p2_minus_p1_delta_nll": float(delta),
        "p2_minus_p1_ci": [float(ci_lo), float(ci_hi)],
        "early_half_delta_nll": float(early_delta),
        "late_half_delta_nll": float(late_delta),
        "p2_vs_p1_macro_js_ratio": js_ratio,
        "gate_products": bool(gate_products),
        "gate_buyers": bool(gate_buyers),
        "gate_median": bool(gate_median),
        "gate_effect": bool(gate_effect),
        "gate_ci": bool(gate_ci),
        "gate_halves": bool(gate_halves),
        "gate_js": bool(gate_js),
        "earn_llm_challenge": earn,
        "interpretation": interpretation,
    }

    summary["experiment"] = "007_expanded_cold_start_confirmation"
    summary["time_design"]["test_launch_window"] = [EXP007_START.date().isoformat(), EXP007_END.date().isoformat()]
    summary["tuning"]["P2_tau_grid"] = [FROZEN_TAU]
    summary["tuning"]["P2_selected_tau"] = FROZEN_TAU
    summary["tuning"]["P3_alpha_grid"] = [FROZEN_ALPHA]
    summary["tuning"]["P3_selected_alpha"] = FROZEN_ALPHA
    summary["exp007_decision"] = decision
    summary["query_plan"] = None
    summary["new_llm_calls"] = 0
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_results(summary, output_dir / "results.md")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
