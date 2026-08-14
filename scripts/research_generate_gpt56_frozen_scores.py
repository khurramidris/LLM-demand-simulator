from __future__ import annotations

"""Generate Experiment 004's frozen GPT-5.6-Sol-authored latent scores.

IMPORTANT: this is a deterministic, target-blind implementation of the semantic/customer
judgments authored by GPT-5.6 Sol in the chat experiment. It is NOT a claim that 400
independent API completions were executed. The model had access only to the blinded
product, paper-persona, and matched-customer artifacts when this rule was frozen.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

AGE_MID = {"16-24": 20.5, "25-34": 29.5, "35-44": 39.5, "45-54": 49.5, "55+": 62.0}
TIER_PREF = {"low": 10.5, "mid": 15.5, "high": 24.0}
ENG_FACTOR = {"low": 0.85, "mid": 1.0, "high": 1.15}


def geometric_reference(prices_json: str) -> float:
    prices = np.asarray(json.loads(prices_json), dtype=float)
    if np.any(prices <= 0):
        raise ValueError("Prices must be positive")
    return float(np.exp(np.mean(np.log(prices))))


def semantic_base(row: pd.Series) -> float:
    """Blinded semantic prior authored before target inspection.

    The rule encodes broad retail plausibility only: everyday/basic utility, common
    neutral colors, stretch/comfort, a few style-specific breadth penalties, and
    reference-price value. It contains no observed demand or product popularity.
    """
    score = 52.0
    section = str(row["section_name"]).lower()
    name = str(row["prod_name"]).lower()
    desc = str(row["detail_desc"]).lower()
    color = str(row["colour_group_name"]).lower()
    text = name + " " + desc

    if "womens everyday collection" in section:
        score += 5
    if "womens everyday basics" in section:
        score += 6
    if "divided basics" in section:
        score += 6
    if "divided collection" in section:
        score += 3
    if "ladies denim" in section:
        score += 4
    if "womens casual" in section:
        score += 3
    if "tailoring" in section:
        score -= 1
    if "selected" in section:
        score -= 2
    if "young boy" in section or "young girl" in section:
        score -= 5
    if "denim men" in section:
        score += 1

    color_bonus = {
        "black": 5, "dark blue": 4, "blue": 3, "grey": 2, "dark grey": 2,
        "light blue": 2, "greenish khaki": 1, "dark beige": 0, "dark green": 0,
        "white": 0, "dark red": -1, "yellowish brown": -2,
    }
    score += color_bonus.get(color, 0)

    if "stretch" in text or "superstretch" in text:
        score += 2
    if "elasticated" in text:
        score += 2
    if "drawstring" in text:
        score += 1
    if "basic" in text:
        score += 2
    if "sweatpants" in text or "jogger" in text or "joggers" in text:
        score += 2
    if "skinny" in text or "slim" in text:
        score += 1
    if "high waist" in text or "high-waist" in text or "high-waisted" in text:
        score += 1
    if "push up" in text:
        score -= 2
    if "hard-worn" in text or "trash" in text:
        score -= 2
    if "paper bag" in text or "paperwaist" in text or "paper waist" in text:
        score -= 1
    if "suit trouser" in text or "tailored trouser" in text:
        score -= 1
    if "low-rise" in text or "low waist" in text:
        score -= 1
    if "fake front pockets" in text:
        score -= 1

    ref = geometric_reference(row["prices_json"])
    score += 4.0 * np.log(15.0 / ref)
    return float(np.clip(score, 25.0, 82.0))


def age_affinity(age: float, index_name: str) -> float:
    idx = str(index_name).lower()
    if "divided" in idx:
        center, scale = 27.0, 17.0
    elif "ladieswear" in idx:
        center, scale = 39.0, 23.0
    elif "menswear" in idx:
        center, scale = 36.0, 24.0
    else:
        return 0.60
    return float(np.exp(-0.5 * ((float(age) - center) / scale) ** 2))


def price_affinity(reference_price: float, preferred_price: float) -> float:
    preferred_price = max(float(preferred_price), 1e-3)
    return float(np.exp(-0.5 * (np.log(reference_price / preferred_price) / 0.62) ** 2))


def population_adjusted(base: float, ref: float, price_fit: float, age_fit: float, engagement: float) -> tuple[float, float]:
    appeal = base + 11.0 * (price_fit - 0.62) + 6.0 * (age_fit - 0.72) + 3.0 * (engagement - 1.0)
    sensitivity = (
        13.0 + 4.0 * (1.0 - price_fit) - 2.0 * (engagement - 1.0)
        + 1.5 * np.clip(np.log(15.0 / ref), -1.0, 1.0)
    )
    return float(np.clip(appeal, 20.0, 85.0)), float(np.clip(sensitivity, 5.0, 25.0))


def generate(products: pd.DataFrame, paper: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    paper_weights = paper["n_customers"].to_numpy(float)
    paper_weights /= paper_weights.sum()
    paper_age = np.asarray([AGE_MID[str(x)] for x in paper["age_bin"]], dtype=float)
    paper_price = ((paper["price_q25"] + paper["price_q75"]) / 2.0).to_numpy(float)
    paper_eng = np.asarray([ENG_FACTOR[str(x)] for x in paper["engagement_bin"]], dtype=float)

    coarse_age = np.asarray([AGE_MID[str(x)] for x in matched["age_bin"]], dtype=float)
    coarse_price = np.asarray([TIER_PREF[str(x)] for x in matched["price_tier"]], dtype=float)
    coarse_eng = np.asarray([ENG_FACTOR[str(x)] for x in matched["engagement_bin"]], dtype=float)

    rich_age = matched["age"].to_numpy(float)
    rich_price = matched["mean_price"].to_numpy(float)
    rich_txn = matched["txn_per_month"].to_numpy(float)
    rich_eng = 0.82 + 0.36 * (rich_txn / (rich_txn + 0.8))

    records: list[dict] = []
    for _, row in products.sort_values("article_id").iterrows():
        article_id = int(row["article_id"])
        ref = geometric_reference(row["prices_json"])
        base = semantic_base(row)

        generic_sens = 12.0 + 3.0 * np.clip(np.log(15.0 / ref), -1.0, 1.0)
        records.append({
            "article_id": article_id, "arm": "G", "appeal_ref": base,
            "price_sensitivity": float(np.clip(generic_sens, 5.0, 25.0)), "reference_price": ref,
        })

        p_price = float(np.average([price_affinity(ref, x) for x in paper_price], weights=paper_weights))
        p_age = float(np.average([age_affinity(x, row["index_name"]) for x in paper_age], weights=paper_weights))
        p_eng = float(np.average(paper_eng, weights=paper_weights))
        appeal, sens = population_adjusted(base, ref, p_price, p_age, p_eng)
        records.append({"article_id": article_id, "arm": "P", "appeal_ref": appeal, "price_sensitivity": sens, "reference_price": ref})

        c_price = float(np.mean([price_affinity(ref, x) for x in coarse_price]))
        c_age = float(np.mean([age_affinity(x, row["index_name"]) for x in coarse_age]))
        c_eng = float(np.mean(coarse_eng))
        appeal, sens = population_adjusted(base, ref, c_price, c_age, c_eng)
        records.append({"article_id": article_id, "arm": "C", "appeal_ref": appeal, "price_sensitivity": sens, "reference_price": ref})

        r_price = float(np.mean([price_affinity(ref, x) for x in rich_price]))
        r_age = float(np.mean([age_affinity(x, row["index_name"]) for x in rich_age]))
        r_eng = float(np.mean(rich_eng))
        appeal, sens = population_adjusted(base, ref, r_price, r_age, r_eng)
        records.append({"article_id": article_id, "arm": "R", "appeal_ref": appeal, "price_sensitivity": sens, "reference_price": ref})

    out = pd.DataFrame(records)
    if len(out) != 400 or out[["article_id", "arm"]].duplicated().any():
        raise AssertionError("Expected exactly 100 products x 4 arms")
    if set(out["arm"]) != {"G", "P", "C", "R"}:
        raise AssertionError("Missing arm")
    if not out["appeal_ref"].between(0, 100).all():
        raise AssertionError("appeal out of range")
    if not out["price_sensitivity"].between(0, 30).all():
        raise AssertionError("price sensitivity out of range")
    return out.sort_values(["article_id", "arm"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", type=Path, required=True)
    ap.add_argument("--paper-personas", type=Path, required=True)
    ap.add_argument("--matched-states", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    products = pd.read_csv(args.products)
    paper = pd.read_csv(args.paper_personas)
    matched = pd.read_csv(args.matched_states)
    out = generate(products, paper, matched)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, float_format="%.12g")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({
        "rows": int(len(out)), "articles": int(out.article_id.nunique()),
        "arms": sorted(out.arm.unique().tolist()), "sha256": digest,
        "contains_targets": False,
    }, indent=2))


if __name__ == "__main__":
    main()
