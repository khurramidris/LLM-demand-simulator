from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_rich_cell_prompts import (
    PRICE_SCALE,
    HISTORY_START,
    assign_original_cells,
    fmt_range,
    read_single_csv_from_zip,
)

CUTOFF = pd.Timestamp("2019-05-31")
ROOT = Path("outputs/research/rich_persona_signal/temporal_holdout")
SMOKE_ROOT = Path("outputs/research/rich_persona_signal/gpt56_smoke")


def build_trouser_history_early(path: Path) -> pd.DataFrame:
    tx = read_single_csv_from_zip(path)
    tx["t_dat"] = pd.to_datetime(tx["t_dat"], errors="coerce")
    tx = tx[(tx["t_dat"] >= HISTORY_START) & (tx["t_dat"] <= CUTOFF)].copy()
    tx["customer_id"] = tx["customer_id"].astype(str)
    tx["article_id"] = pd.to_numeric(tx["article_id"], errors="coerce")
    tx["price"] = pd.to_numeric(tx["price"], errors="coerce") * PRICE_SCALE
    tx = tx.dropna(subset=["customer_id", "article_id", "price", "t_dat"])
    tx = tx.sort_values(["customer_id", "t_dat"])

    g = tx.groupby("customer_id", sort=False)
    out = g.agg(
        tr_txn_count=("article_id", "size"),
        tr_unique_articles=("article_id", "nunique"),
        tr_mean_price=("price", "mean"),
        tr_last_date=("t_dat", "max"),
    ).reset_index()
    out["tr_days_since_last"] = (CUTOFF - out["tr_last_date"]).dt.days.astype(float)
    out["tr_repeat_ratio"] = 1.0 - out["tr_unique_articles"] / out["tr_txn_count"].clip(lower=1)
    for days in [30, 90, 180]:
        start = CUTOFF - pd.Timedelta(days=days - 1)
        counts = tx.loc[tx["t_dat"] >= start].groupby("customer_id").size().rename(f"tr_txn_last_{days}d")
        out = out.merge(counts, on="customer_id", how="left")
        out[f"tr_txn_last_{days}d"] = out[f"tr_txn_last_{days}d"].fillna(0.0)
    return out.drop(columns=["tr_last_date"])


def build_early_personas() -> pd.DataFrame:
    features = read_single_csv_from_zip(Path("outputs/personas/customer_features.csv.zip"))
    cells = pd.read_csv("outputs/personas/persona_cells.csv").head(50).copy()
    members = assign_original_cells(features)
    tr = build_trouser_history_early(Path("outputs/products/txns_trousers_online.csv.zip"))
    members = members.merge(tr, on="customer_id", how="left")

    key = ["age_bin", "engagement_bin", "price_tier", "taste_bucket"]
    matched = members.merge(cells[key + ["persona_id", "n_customers", "persona_prompt"]], on=key, how="inner")
    chosen = json.loads((SMOKE_ROOT / "manifest.json").read_text())["chosen_personas"]
    matched = matched[matched["persona_id"].isin(chosen)].copy()

    numeric = [
        "tr_txn_count", "tr_unique_articles", "tr_mean_price", "tr_days_since_last",
        "tr_repeat_ratio", "tr_txn_last_30d", "tr_txn_last_90d", "tr_txn_last_180d",
    ]
    rows = []
    for persona_id in chosen:
        group = matched[matched["persona_id"] == persona_id].copy()
        if group.empty:
            raise RuntimeError(f"No matched members for {persona_id}")
        base_prompt = group["persona_prompt"].iloc[0]
        base = base_prompt.split("\n\nTask:")[0].strip()
        has_tr = group["tr_txn_count"].notna()
        share = float(has_tr.mean())
        buyers = group.loc[has_tr].copy()
        if len(buyers):
            stats = {c: (float(buyers[c].quantile(.25)), float(buyers[c].quantile(.75))) for c in numeric}
            extra = (
                f" Historical behavioral context for this same customer segment, using only trouser purchases through {CUTOFF.date()}: "
                f"about {share*100:.1f}% bought trousers at least once. Among those trouser buyers, the middle 50% made "
                f"{fmt_range(*stats['tr_txn_count'], 0)} trouser purchases across {fmt_range(*stats['tr_unique_articles'], 0)} distinct trouser articles, "
                f"paid about ${fmt_range(*stats['tr_mean_price'], 2)}, and last bought trousers {fmt_range(*stats['tr_days_since_last'], 0)} days before the cutoff. "
                f"Their repeat-item tendency was {fmt_range(*stats['tr_repeat_ratio'], 2)}; middle-50% trouser counts in the last 30/90/180 days were "
                f"{fmt_range(*stats['tr_txn_last_30d'], 0)}, {fmt_range(*stats['tr_txn_last_90d'], 0)}, and {fmt_range(*stats['tr_txn_last_180d'], 0)}."
            )
        else:
            extra = (
                f" Historical behavioral context for this same customer segment, using only trouser purchases through {CUTOFF.date()}: "
                "no members in the available history bought trousers."
            )
        instruction = (
            '\n\nTask: Given a product and a list of prices, return the probability you would buy at each price. '
            'Output JSON exactly: {"prices": [...], "p_buy": [...], "reason": "<=30 words"}.'
        )
        rows.append({
            "persona_id": persona_id,
            "n_customers": int(group["n_customers"].iloc[0]),
            "base_persona_prompt": base_prompt,
            "rich_persona_prompt": base + extra + instruction,
            "n_matched_customers": int(len(group)),
            "share_with_trouser_history": share,
        })
    return pd.DataFrame(rows)


def build_pairs(personas: pd.DataFrame) -> list[dict]:
    by_id = personas.set_index("persona_id")
    pairs = []
    for line in (SMOKE_ROOT / "prompt_pairs.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pid = row["persona_id"]
        marker = "\n\nProduct name:"
        suffix = row["control_prompt"].split(marker, 1)[1]
        row["rich_prompt"] = by_id.loc[pid, "rich_persona_prompt"] + marker + suffix
        row["share_with_trouser_history"] = float(by_id.loc[pid, "share_with_trouser_history"])
        row["n_customers"] = int(by_id.loc[pid, "n_customers"])
        pairs.append(row)
    if len(pairs) != 50:
        raise RuntimeError(f"Expected 50 frozen pairs, got {len(pairs)}")
    return pairs


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    personas = build_early_personas()
    personas.to_csv(ROOT / "persona_prompts.csv", index=False)
    pairs = build_pairs(personas)
    with (ROOT / "prompt_pairs.jsonl").open("w") as f:
        for row in pairs:
            f.write(json.dumps(row) + "\n")

    manifest = {
        "protocol": "GPT56-TEMPORAL-HOLDOUT-v1",
        "history_start": str(HISTORY_START.date()),
        "behavior_cutoff": str(CUTOFF.date()),
        "calibration_period": f"<= {CUTOFF.date()}",
        "heldout_period": f"> {CUTOFF.date()} through 2019-09-19",
        "n_personas": 10,
        "n_products": 5,
        "logical_calls_per_arm": 50,
        "primary_metrics": ["zero_truncated_nll", "zero_truncated_crps"],
        "secondary_metrics": ["mae", "rmse"],
        "advance_rule": "rich improves both primary metrics and neither MAE nor RMSE worsens by more than 5%",
        "shared_limitations": [
            "Original persona-cell memberships come from the repository's precomputed full-period customer features and therefore are not temporally rebuilt.",
            "Candidate price grids are the repository's original historically observed grids and are transductive.",
            "Both limitations are identical across control and treatment; only added trouser-history context is cutoff-safe.",
            "ChatGPT-in-conversation responses are not independent blinded API calls.",
        ],
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(personas[["persona_id", "n_customers", "share_with_trouser_history"]].to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
