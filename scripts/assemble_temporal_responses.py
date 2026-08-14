from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("outputs/research/rich_persona_signal/temporal_holdout")
SMOKE = Path("outputs/research/rich_persona_signal/gpt56_smoke")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    prompts = read_jsonl(ROOT / "prompt_pairs.jsonl")
    by_product: dict[int, list[dict]] = {}
    for row in prompts:
        by_product.setdefault(int(row["article_id"]), []).append(row)

    total = 0
    for article_id, product_prompts in sorted(by_product.items()):
        expected_pairs = {row["pair_id"] for row in product_prompts}
        old = read_jsonl(SMOKE / f"responses_{article_id}.jsonl")
        controls = [r for r in old if r.get("arm") == "control" and r.get("pair_id") in expected_pairs]
        rich = read_jsonl(ROOT / f"rich_responses_{article_id}.jsonl")
        if len(controls) != 10 or len(rich) != 10:
            raise RuntimeError(f"{article_id}: expected 10 controls + 10 rich, got {len(controls)} + {len(rich)}")
        keys = {(r["pair_id"], r["arm"]) for r in controls + rich}
        expected = {(pair, arm) for pair in expected_pairs for arm in ["control", "rich"]}
        if keys != expected:
            raise RuntimeError(f"{article_id}: response key mismatch")

        # Preserve model outputs exactly; only add provenance tags outside the response payload.
        for r in controls:
            r["provenance"] = "reused unchanged from GPT56-MATCHED-CELL-SMOKE-v1; identical control prompt"
        for r in rich:
            r["provenance"] = "generated from GPT56-TEMPORAL-HOLDOUT-v1 rich prompt using history through 2019-05-31"

        order = {row["pair_id"]: i for i, row in enumerate(product_prompts)}
        combined = sorted(controls + rich, key=lambda r: (order[r["pair_id"]], 0 if r["arm"] == "control" else 1))
        out = ROOT / f"responses_{article_id}.jsonl"
        out.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in combined))
        total += len(combined)
    if total != 100:
        raise RuntimeError(f"Expected 100 assembled records, got {total}")
    print(f"Assembled {total} frozen records across {len(by_product)} products")


if __name__ == "__main__":
    main()
