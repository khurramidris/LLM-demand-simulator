from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProductHoldout:
    """Deterministic product-level split used by leakage-safe experiments."""

    train_ids: tuple[int, ...]
    test_ids: tuple[int, ...]
    seed: int
    train_frac: float

    def validate(self) -> None:
        train = set(self.train_ids)
        test = set(self.test_ids)
        if not train or not test:
            raise ValueError("Both train and test product sets must be non-empty.")
        overlap = train.intersection(test)
        if overlap:
            raise ValueError(f"Train/test product sets overlap: {sorted(overlap)[:10]}")

    @property
    def all_ids(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.train_ids).union(self.test_ids)))

    def to_dict(self) -> dict:
        self.validate()
        payload = asdict(self)
        payload["train_ids"] = list(self.train_ids)
        payload["test_ids"] = list(self.test_ids)
        payload["n_train_products"] = len(self.train_ids)
        payload["n_test_products"] = len(self.test_ids)
        return payload


def make_product_holdout(
    product_ids: list[int] | np.ndarray | pd.Series,
    train_frac: float = 0.6,
    seed: int = 2025,
) -> ProductHoldout:
    """Create a deterministic product-level split without touching outcomes."""
    ids = np.array(sorted({int(x) for x in product_ids}), dtype=int)
    if len(ids) < 2:
        raise ValueError("Need at least two distinct products to create a holdout.")
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must lie strictly between 0 and 1.")

    rng = np.random.default_rng(seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    n_train = int(round(train_frac * len(shuffled)))
    n_train = min(max(n_train, 1), len(shuffled) - 1)

    split = ProductHoldout(
        train_ids=tuple(int(x) for x in np.sort(shuffled[:n_train])),
        test_ids=tuple(int(x) for x in np.sort(shuffled[n_train:])),
        seed=int(seed),
        train_frac=float(train_frac),
    )
    split.validate()
    return split


def write_holdout(split: ProductHoldout, output_dir: Path) -> None:
    """Persist a split in human-readable and machine-readable forms."""
    split.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(split.train_ids, name="article_id").to_csv(output_dir / "train_products.csv", index=False)
    pd.Series(split.test_ids, name="article_id").to_csv(output_dir / "test_products.csv", index=False)
    (output_dir / "split_manifest.json").write_text(json.dumps(split.to_dict(), indent=2))


def read_holdout(train_path: Path, test_path: Path, seed: int = -1, train_frac: float = -1.0) -> ProductHoldout:
    train_ids = tuple(pd.read_csv(train_path)["article_id"].astype(int).tolist())
    test_ids = tuple(pd.read_csv(test_path)["article_id"].astype(int).tolist())
    split = ProductHoldout(train_ids=train_ids, test_ids=test_ids, seed=seed, train_frac=train_frac)
    split.validate()
    return split


def filter_transactions_excluding_articles(
    transactions_path: Path,
    output_path: Path,
    excluded_article_ids: set[int] | list[int] | tuple[int, ...],
    chunksize: int = 2_000_000,
) -> dict:
    """
    Stream the raw H&M transaction file and remove every transaction whose
    article_id belongs to the held-out product set.

    This is deliberately performed before persona construction so held-out
    products cannot influence engagement, typical paid price, or taste.
    """
    excluded = {int(x) for x in excluded_article_ids}
    if not excluded:
        raise ValueError("excluded_article_ids must be non-empty.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    removed_rows = 0
    affected_customers: set[str] = set()
    wrote_header = False

    reader = pd.read_csv(transactions_path, chunksize=chunksize)
    for chunk in reader:
        total_rows += len(chunk)
        article_ids = pd.to_numeric(chunk["article_id"], errors="coerce")
        mask = article_ids.isin(excluded)
        removed_rows += int(mask.sum())
        if "customer_id" in chunk.columns and mask.any():
            affected_customers.update(chunk.loc[mask, "customer_id"].astype(str).tolist())
        kept = chunk.loc[~mask]
        kept.to_csv(output_path, mode="a" if wrote_header else "w", header=not wrote_header, index=False)
        wrote_header = True

    if not wrote_header:
        raise ValueError(f"No rows found in {transactions_path}")

    return {
        "input_rows": int(total_rows),
        "removed_rows": int(removed_rows),
        "kept_rows": int(total_rows - removed_rows),
        "excluded_product_count": len(excluded),
        "affected_customer_count": len(affected_customers),
        "input_sha256": sha256_file(transactions_path),
        "output_sha256": sha256_file(output_path),
    }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def probability_monotonicity_audit(probabilities: pd.DataFrame, tolerance: float = 1e-9) -> dict:
    """
    Measure violations of the economically natural condition that purchase
    probability should not increase when price increases for the same
    persona-product-draw.
    """
    required = {"persona_id", "article_id", "offer_price", "p_buy"}
    missing = required.difference(probabilities.columns)
    if missing:
        raise ValueError(f"Missing columns for monotonicity audit: {sorted(missing)}")

    group_cols = ["persona_id", "article_id"]
    if "draw_id" in probabilities.columns:
        group_cols.append("draw_id")

    n_groups = 0
    groups_with_violation = 0
    adjacent_pairs = 0
    violating_pairs = 0
    max_upward_jump = 0.0

    for _, group in probabilities.groupby(group_cols, sort=False):
        ordered = group.sort_values("offer_price")
        if len(ordered) < 2:
            continue
        n_groups += 1
        probs = ordered["p_buy"].to_numpy(float)
        diffs = np.diff(probs)
        violations = diffs > tolerance
        adjacent_pairs += len(diffs)
        violating_pairs += int(violations.sum())
        if violations.any():
            groups_with_violation += 1
            max_upward_jump = max(max_upward_jump, float(diffs[violations].max()))

    return {
        "n_groups": n_groups,
        "groups_with_violation": groups_with_violation,
        "group_violation_rate": groups_with_violation / max(n_groups, 1),
        "adjacent_price_pairs": adjacent_pairs,
        "violating_price_pairs": violating_pairs,
        "pair_violation_rate": violating_pairs / max(adjacent_pairs, 1),
        "max_upward_probability_jump": max_upward_jump,
    }


def effective_population_size(weights: np.ndarray | list[float]) -> float:
    """Inverse-Herfindahl effective number of non-zero mixture components."""
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 0.0, None)
    total = float(w.sum())
    if total <= 0:
        return 0.0
    w = w / total
    return float(1.0 / np.sum(w**2))


def audit_alpha_table(alpha_table: pd.DataFrame) -> dict:
    if not {"persona_id", "alpha"}.issubset(alpha_table.columns):
        raise ValueError("alpha table must contain persona_id and alpha.")
    working = alpha_table.copy()
    working["alpha"] = pd.to_numeric(working["alpha"], errors="coerce").fillna(0.0)
    dummy = float(working.loc[working["persona_id"] == "DUMMY_NO_BUY", "alpha"].sum())
    real = float(working.loc[working["persona_id"] != "DUMMY_NO_BUY", "alpha"].sum())
    real_weights = working.loc[working["persona_id"] != "DUMMY_NO_BUY", "alpha"].to_numpy(float)
    return {
        "real_alpha_mass": real,
        "dummy_alpha_mass": dummy,
        "effective_real_personas": effective_population_size(real_weights),
        "max_real_persona_weight": float(real_weights.max()) if len(real_weights) else 0.0,
    }


def load_probability_rows_strict(
    response_csv: Path,
    price_tolerance: float = 1e-6,
) -> tuple[pd.DataFrame, dict]:
    """
    Parse LLM outputs without silently clipping malformed probabilities.

    In addition to JSON validity, this checks that the model returned exactly
    the price grid it was asked about (when prices_json is present), that
    probabilities are finite and lie in [0, 1], and that prices are unique.
    """
    from .data import parse_llm_response

    raw = pd.read_csv(response_csv)
    rows: list[dict] = []
    counters = {
        "raw_rows": int(len(raw)),
        "unparsable": 0,
        "price_grid_mismatch": 0,
        "invalid_probability": 0,
        "duplicate_price": 0,
        "valid_rows": 0,
    }

    for record in raw.itertuples(index=False):
        parsed = parse_llm_response(getattr(record, "response", None))
        if parsed is None:
            counters["unparsable"] += 1
            continue
        prices, probs, reason = parsed

        if len(set(float(x) for x in prices)) != len(prices):
            counters["duplicate_price"] += 1
            continue

        arr = np.asarray(probs, dtype=float)
        if not np.all(np.isfinite(arr)) or np.any(arr < 0.0) or np.any(arr > 1.0):
            counters["invalid_probability"] += 1
            continue

        expected_json = getattr(record, "prices_json", None)
        if isinstance(expected_json, str) and expected_json.strip():
            try:
                expected = [float(x) for x in json.loads(expected_json)]
            except (json.JSONDecodeError, TypeError, ValueError):
                expected = []
            if len(expected) != len(prices) or not np.allclose(
                np.asarray(expected, dtype=float),
                np.asarray(prices, dtype=float),
                rtol=0.0,
                atol=price_tolerance,
            ):
                counters["price_grid_mismatch"] += 1
                continue

        counters["valid_rows"] += 1
        persona_id = str(getattr(record, "persona_id"))
        article_id = int(getattr(record, "article_id"))
        draw_id = int(getattr(record, "draw_id", 0))
        for price, prob in zip(prices, probs):
            rows.append(
                {
                    "persona_id": persona_id,
                    "article_id": article_id,
                    "offer_price": round(float(price), 2),
                    "draw_id": draw_id,
                    "p_buy": float(prob),
                    "reason": reason,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No strictly valid LLM probabilities parsed from {response_csv}")

    counters["valid_response_rate"] = counters["valid_rows"] / max(counters["raw_rows"], 1)
    counters["probability_rows"] = int(len(out))
    return out, counters
