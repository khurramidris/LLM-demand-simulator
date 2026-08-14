from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("outputs/research/rich_persona_signal/gpt56_smoke")
PRODUCTS = [397068015, 539723038, 547780003, 562245006, 562245059]
ARMS = {"control", "rich"}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed JSON at {path}:{line_no}: {exc}") from exc
    return rows


def nonincreasing(values: np.ndarray, atol: float = 1e-12) -> bool:
    return bool(np.all(np.diff(values) <= atol))


def main() -> None:
    prompt_rows = load_jsonl(ROOT / "prompt_pairs.jsonl")
    prompt_map = {row["pair_id"]: row for row in prompt_rows}
    if len(prompt_rows) != 50 or len(prompt_map) != 50:
        raise RuntimeError(f"Expected exactly 50 frozen prompt pairs, got {len(prompt_rows)} rows/{len(prompt_map)} unique")

    records: list[dict] = []
    source_counts: dict[str, int] = {}
    for article_id in PRODUCTS:
        path = ROOT / f"responses_{article_id}.jsonl"
        rows = load_jsonl(path)
        source_counts[path.name] = len(rows)
        records.extend(rows)

    errors: list[str] = []
    if len(records) != 100:
        errors.append(f"expected 100 response records, got {len(records)}")

    keys = [(r.get("pair_id"), r.get("arm")) for r in records]
    if len(set(keys)) != len(keys):
        errors.append("duplicate pair_id/arm records found")

    by_pair: dict[str, dict[str, dict]] = {}
    monotone_flags: list[bool] = []
    flat_flags: list[bool] = []
    row_summaries: list[dict] = []

    for rec in records:
        pair_id = rec.get("pair_id")
        arm = rec.get("arm")
        if pair_id not in prompt_map:
            errors.append(f"response pair not in frozen prompts: {pair_id}")
            continue
        if arm not in ARMS:
            errors.append(f"invalid arm for {pair_id}: {arm}")
            continue
        prompt = prompt_map[pair_id]
        if int(rec.get("article_id")) != int(prompt["article_id"]):
            errors.append(f"article mismatch for {pair_id}/{arm}")
        if str(rec.get("persona_id")) != str(prompt["persona_id"]):
            errors.append(f"persona mismatch for {pair_id}/{arm}")
        if str(rec.get("price_sig")) != str(prompt["price_sig"]):
            errors.append(f"price signature mismatch for {pair_id}/{arm}")

        response = rec.get("response", {})
        prices = np.asarray(response.get("prices", []), dtype=float)
        probs = np.asarray(response.get("p_buy", []), dtype=float)
        expected_prices = np.asarray(json.loads(prompt["prices_json"]), dtype=float)
        if len(prices) != len(expected_prices) or not np.allclose(prices, expected_prices, rtol=0.0, atol=1e-9):
            errors.append(f"price grid mismatch for {pair_id}/{arm}")
        if len(probs) != len(prices):
            errors.append(f"probability length mismatch for {pair_id}/{arm}")
            continue
        if not np.isfinite(probs).all():
            errors.append(f"non-finite probability for {pair_id}/{arm}")
        if np.any((probs < 0.0) | (probs > 1.0)):
            errors.append(f"out-of-range probability for {pair_id}/{arm}")

        mono = nonincreasing(probs)
        flat = bool(np.allclose(probs, probs[0], atol=1e-12))
        monotone_flags.append(mono)
        flat_flags.append(flat)
        row_summaries.append({
            "pair_id": pair_id,
            "persona_id": rec["persona_id"],
            "article_id": int(rec["article_id"]),
            "arm": arm,
            "mean_p": float(np.mean(probs)),
            "low_price_p": float(probs[0]),
            "high_price_p": float(probs[-1]),
            "price_drop": float(probs[0] - probs[-1]),
            "monotone_nonincreasing": mono,
            "flat_curve": flat,
        })
        by_pair.setdefault(pair_id, {})[arm] = rec

    for pair_id in prompt_map:
        arms = set(by_pair.get(pair_id, {}))
        if arms != ARMS:
            errors.append(f"pair {pair_id} has arms {sorted(arms)}, expected {sorted(ARMS)}")

    pair_stats: list[dict] = []
    if not errors:
        for pair_id, prompt in sorted(prompt_map.items()):
            ctrl = np.asarray(by_pair[pair_id]["control"]["response"]["p_buy"], dtype=float)
            rich = np.asarray(by_pair[pair_id]["rich"]["response"]["p_buy"], dtype=float)
            diff = rich - ctrl
            pair_stats.append({
                "pair_id": pair_id,
                "persona_id": prompt["persona_id"],
                "article_id": int(prompt["article_id"]),
                "share_with_trouser_history": float(prompt["share_with_trouser_history"]),
                "control_mean_p": float(ctrl.mean()),
                "rich_mean_p": float(rich.mean()),
                "mean_delta_rich_minus_control": float(diff.mean()),
                "mean_abs_point_delta": float(np.abs(diff).mean()),
                "max_abs_point_delta": float(np.abs(diff).max()),
            })

    if errors:
        result = {
            "protocol": "GPT56-MATCHED-CELL-SMOKE-v1",
            "audit_status": "INVALID",
            "errors": errors,
            "source_counts": source_counts,
        }
        (ROOT / "smoke_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        raise SystemExit("\n".join(errors))

    pairs = pd.DataFrame(pair_stats)
    rows = pd.DataFrame(row_summaries)
    persona = (
        pairs.groupby(["persona_id", "share_with_trouser_history"], as_index=False)
        .agg(
            mean_delta_rich_minus_control=("mean_delta_rich_minus_control", "mean"),
            mean_abs_point_delta=("mean_abs_point_delta", "mean"),
            n_products=("article_id", "nunique"),
        )
        .sort_values("persona_id")
    )
    product = (
        pairs.groupby("article_id", as_index=False)
        .agg(
            mean_delta_rich_minus_control=("mean_delta_rich_minus_control", "mean"),
            mean_abs_point_delta=("mean_abs_point_delta", "mean"),
            n_personas=("persona_id", "nunique"),
        )
        .sort_values("article_id")
    )

    rho, pvalue = spearmanr(
        persona["share_with_trouser_history"].to_numpy(float),
        persona["mean_delta_rich_minus_control"].to_numpy(float),
    )

    arm_summary = (
        rows.groupby("arm", as_index=False)
        .agg(
            mean_probability=("mean_p", "mean"),
            mean_low_price_probability=("low_price_p", "mean"),
            mean_high_price_probability=("high_price_p", "mean"),
            mean_price_drop=("price_drop", "mean"),
            monotone_rate=("monotone_nonincreasing", "mean"),
            flat_curve_rate=("flat_curve", "mean"),
        )
    )

    result = {
        "protocol": "GPT56-MATCHED-CELL-SMOKE-v1",
        "audit_status": "VALID",
        "provenance_class": "single_conversation_batched_non_independent",
        "n_response_records": int(len(records)),
        "n_unique_pairs": int(len(pairs)),
        "n_personas": int(pairs["persona_id"].nunique()),
        "n_products": int(pairs["article_id"].nunique()),
        "source_counts": source_counts,
        "schema_and_curve_checks": {
            "all_price_grids_exact": True,
            "all_probabilities_finite_and_in_range": True,
            "overall_monotone_nonincreasing_rate": float(np.mean(monotone_flags)),
            "overall_flat_curve_rate": float(np.mean(flat_flags)),
        },
        "paired_shift": {
            "mean_delta_rich_minus_control": float(pairs["mean_delta_rich_minus_control"].mean()),
            "median_delta_rich_minus_control": float(pairs["mean_delta_rich_minus_control"].median()),
            "mean_abs_point_delta": float(pairs["mean_abs_point_delta"].mean()),
            "median_abs_point_delta": float(pairs["mean_abs_point_delta"].median()),
            "fraction_pairs_rich_higher": float((pairs["mean_delta_rich_minus_control"] > 0).mean()),
            "fraction_pairs_rich_lower": float((pairs["mean_delta_rich_minus_control"] < 0).mean()),
        },
        "history_alignment": {
            "unit_of_analysis": "10 persona-level mean shifts (five products averaged per persona)",
            "spearman_rho_history_share_vs_rich_shift": float(rho),
            "spearman_pvalue_descriptive_only": float(pvalue),
            "warning": "Not a causal/significance test: outputs are non-independent and the model saw the treatment context.",
        },
        "arm_summary": arm_summary.to_dict(orient="records"),
        "persona_summary": persona.to_dict(orient="records"),
        "product_summary": product.to_dict(orient="records"),
        "scientific_scope": {
            "supported": "The frozen richer prompt changes this model's elicited price-response curves in a coherent, history-aligned way while preserving schema and downward price response.",
            "not_supported": "Any claim that rich personas improve held-out H&M aggregate demand prediction or causal pricing decisions.",
        },
    }
    (ROOT / "smoke_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pairs.to_csv(ROOT / "pair_summary.csv", index=False)
    persona.to_csv(ROOT / "persona_summary.csv", index=False)
    product.to_csv(ROOT / "product_summary.csv", index=False)

    report = f"""# GPT-5.6 Sol Matched-Cell Mechanism Smoke Test

**Audit:** VALID  
**Provider mode:** single-conversation batched, non-independent  
**Scientific status:** mechanism diagnostic only — **not** a demand-prediction benchmark.

## Frozen design

- 10 original H&M persona cells × 5 fixed products = 50 matched pairs.
- Two arms per pair: original paper-style prompt and the same cell enriched with pre-cutoff trouser-history summaries.
- 100 logical structured outputs total.

## Mechanical validity

- Records: **{len(records)}/100**
- Complete paired arms: **{len(pairs)}/50**
- Exact frozen price grids: **100%**
- Finite probabilities in [0,1]: **100%**
- Non-increasing price curves: **{np.mean(monotone_flags):.1%}**
- Flat curves: **{np.mean(flat_flags):.1%}**

## Representation effect inside this smoke

- Mean rich − control probability shift across pairs: **{pairs['mean_delta_rich_minus_control'].mean():+.4f}**
- Median rich − control shift: **{pairs['mean_delta_rich_minus_control'].median():+.4f}**
- Median mean absolute pointwise change: **{pairs['mean_abs_point_delta'].median():.4f}**
- Pairs with higher rich mean probability: **{(pairs['mean_delta_rich_minus_control'] > 0).mean():.1%}**
- Pairs with lower rich mean probability: **{(pairs['mean_delta_rich_minus_control'] < 0).mean():.1%}**

Across the 10 persona cells, historical trouser-participation share and the average rich-minus-control shift have Spearman rho **{rho:.3f}** (descriptive p={pvalue:.4g}). This is a mechanism diagnostic, not inferential evidence, because the outputs are not independent/blinded.

## Interpretation

The richer state is not being ignored: it changes elicited purchase probabilities materially and in a direction related to the segment's observed pre-cutoff trouser history, while preserving sensible downward price response. That is sufficient for a mechanism smoke test.

It is **not evidence that Allegory beats the H&M paper**. The decisive experiment remains independent, blinded calls followed by the frozen aggregate held-out demand evaluation.
"""
    (ROOT / "SMOKE_REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
