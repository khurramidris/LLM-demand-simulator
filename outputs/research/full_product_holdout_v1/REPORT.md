# Full Product-Holdout Learned-Type Relative-Utility Gate

**Decision:** ADVANCE_TO_FRESH_PAIRWISE_LLM

Seven original H&M product splits; 60 train / 40 held-out products per split. Lower is better.

| model | NLL | CRPS | MAE | RMSE |
|---|---:|---:|---:|---:|
| paper_llm_mix_cal | 1.826956 | 0.941218 | 1.382173 | 1.784407 |
| empirical_persona_relative_utility | 1.820485 | 0.931847 | 1.367770 | 1.770735 |
| learned_type_relative_utility | 1.820491 | 0.931857 | 1.367795 | 1.770751 |

## Learned type vs paper baseline

- zt_avg_nll: -0.354% mean change; wins 5/7 splits; split-bootstrap 95% CI for absolute mean delta [-0.014638, 0.000393].
- zt_avg_crps: -0.995% mean change; wins 7/7 splits; split-bootstrap 95% CI for absolute mean delta [-0.014565, -0.005305].
- mae: -1.040% mean change; wins 7/7 splits; split-bootstrap 95% CI for absolute mean delta [-0.020517, -0.009141].
- rmse: -0.765% mean change; wins 7/7 splits; split-bootstrap 95% CI for absolute mean delta [-0.020752, -0.007981].

## Scientific scope

This is the full 100-product, seven-split scaling gate using frozen author LLM outputs. Held-out-product transactions are excluded from learned behavioral-type construction. It does not yet constitute fresh pairwise GPT-5.6 elicitation.
