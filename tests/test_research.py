from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from demand_sim.data import parse_llm_response
from demand_sim.models.llm_mix_cal import calibrate_probability_matrix
from demand_sim.research import (
    filter_transactions_excluding_articles,
    make_product_holdout,
    probability_monotonicity_audit,
)


def test_product_holdout_is_deterministic_and_disjoint() -> None:
    ids = list(range(100, 120))
    a = make_product_holdout(ids, train_frac=0.6, seed=17)
    b = make_product_holdout(ids, train_frac=0.6, seed=17)
    assert a == b
    assert set(a.train_ids).isdisjoint(a.test_ids)
    assert set(a.all_ids) == set(ids)
    assert len(a.train_ids) == 12
    assert len(a.test_ids) == 8


def test_transaction_filter_removes_heldout_articles(tmp_path: Path) -> None:
    source = tmp_path / "transactions.csv"
    target = tmp_path / "filtered.csv"
    pd.DataFrame(
        {
            "t_dat": ["2019-01-01"] * 4,
            "customer_id": ["a", "b", "a", "c"],
            "article_id": [1, 2, 3, 2],
            "price": [0.1, 0.2, 0.3, 0.2],
        }
    ).to_csv(source, index=False)

    stats = filter_transactions_excluding_articles(source, target, {2}, chunksize=2)
    out = pd.read_csv(target)
    assert out["article_id"].tolist() == [1, 3]
    assert stats["input_rows"] == 4
    assert stats["removed_rows"] == 2
    assert stats["kept_rows"] == 2
    assert stats["affected_customer_count"] == 2


def test_probability_monotonicity_audit_detects_upward_price_response() -> None:
    probabilities = pd.DataFrame(
        {
            "persona_id": ["P1", "P1", "P1", "P2", "P2"],
            "article_id": [7, 7, 7, 7, 7],
            "offer_price": [10, 20, 30, 10, 20],
            "p_buy": [0.5, 0.4, 0.45, 0.2, 0.1],
        }
    )
    report = probability_monotonicity_audit(probabilities)
    assert report["n_groups"] == 2
    assert report["groups_with_violation"] == 1
    assert report["violating_price_pairs"] == 1
    assert np.isclose(report["max_upward_probability_jump"], 0.05)


def test_parse_llm_response_accepts_json_fence() -> None:
    parsed = parse_llm_response(
        '```json\n{"prices":[10,20],"p_buy":[0.4,0.2],"reason":"test"}\n```'
    )
    assert parsed is not None
    prices, probs, reason = parsed
    assert prices == [10.0, 20.0]
    assert probs == [0.4, 0.2]
    assert reason == "test"


def test_logit_calibration_preserves_probability_order() -> None:
    q = np.array([[0.1, 0.2, 0.8]])
    calibrated = calibrate_probability_matrix(q, intercept=-4.0, slope=0.2)
    assert np.all(np.diff(calibrated[0]) > 0)
    assert np.all((calibrated > 0) & (calibrated < 1))
