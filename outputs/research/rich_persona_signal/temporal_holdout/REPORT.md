# GPT-5.6 Sol Temporal H&M Demand Pilot

**Decision:** DO_NOT_ADVANCE

This is a real H&M temporal holdout pilot: behavior context and calibration stop at 2019-05-31; scored demand is after that date.

## Test metrics (lower is better)

| metric | control | rich | rich vs control |
|---|---:|---:|---:|
| zt_avg_nll | 1.385505 | 1.388033 | +0.18% |
| zt_avg_crps | 0.728920 | 0.731165 | +0.31% |
| mae | 1.321750 | 1.326010 | +0.32% |
| rmse | 1.418167 | 1.421545 | +0.24% |

## Coverage

- Training matched rows: 468 / 491 (95.3%).
- Held-out matched rows: 147 / 164 (89.6%).
- Held-out dates: 2019-06-01 to 2019-09-18.

## Paired bootstrap

- nll: rich-control mean 0.002558, 95% CI [-0.000187, 0.005232], P(rich better)=0.033.
- crps: rich-control mean 0.002272, 95% CI [-0.000052, 0.004553], P(rich better)=0.028.
- mae: rich-control mean 0.004300, 95% CI [0.000767, 0.007705], P(rich better)=0.009.
- rmse: rich-control mean 0.003431, 95% CI [-0.000067, 0.006947], P(rich better)=0.028.

## Scientific scope

This tests whether cutoff-safe trouser-history enrichment improves future real H&M demand predictions for five fixed products under identical population weighting and calibration procedure.
It does not establish product cold-start generalization, causal price elasticity, or a full improvement over the paper.
The original persona-cell memberships and candidate price grids remain shared transductive artifacts from the repository.
The GPT-5.6 outputs are conversation-batched rather than independent blinded API calls.
