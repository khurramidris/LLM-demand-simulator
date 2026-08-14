from __future__ import annotations

import numpy as np
import pandas as pd

from demand_sim.models.population_cal import fit_population_cal


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": [1, 1, 2, 2],
            "offer_price": [10.0, 20.0, 10.0, 20.0],
            "demand": [8, 4, 6, 3],
            "P001": [0.50, 0.30, 0.45, 0.25],
            "P002": [0.20, 0.10, 0.18, 0.08],
        }
    )


def test_population_cal_uses_observed_cell_counts() -> None:
    personas = pd.DataFrame(
        {"persona_id": ["P001", "P002"], "n_customers": [900, 100]}
    )
    model = fit_population_cal(
        _rows(),
        ["P001", "P002"],
        personas,
        exposure_n_values=[100],
        weight_mode="population-cal",
    )
    assert np.allclose(model.weights, [0.9, 0.1])
    assert model.slope > 0


def test_uniform_cal_ignores_cell_counts() -> None:
    personas = pd.DataFrame(
        {"persona_id": ["P001", "P002"], "n_customers": [999, 1]}
    )
    model = fit_population_cal(
        _rows(),
        ["P001", "P002"],
        personas,
        exposure_n_values=[100],
        weight_mode="uniform-cal",
    )
    assert np.allclose(model.weights, [0.5, 0.5])
