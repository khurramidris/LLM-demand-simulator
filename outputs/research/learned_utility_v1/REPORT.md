# Learned Buyer Types + Pairwise Utility H&M Pilot

**Decision:** ADVANCE

## Future-demand metrics (lower is better)

| arm | NLL | CRPS | MAE | RMSE |
|---|---:|---:|---:|---:|
| paper_probability_baseline | 1.385502 | 0.728917 | 1.321746 | 1.418163 |
| simple_pairwise_utility | 1.388238 | 0.731189 | 1.324626 | 1.421643 |
| learned_pairwise_utility | 1.380064 | 0.723548 | 1.315199 | 1.408331 |

## Interpretation

Simple utility vs baseline NLL: +0.20%.
Learned utility vs baseline NLL: -0.39%.
Learned vs simple utility NLL: -0.59%.

This is a non-blinded five-product temporal pilot, not a publication-grade product-holdout result.
