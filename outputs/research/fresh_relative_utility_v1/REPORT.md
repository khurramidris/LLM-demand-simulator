# Fresh GPT-5.6 Learned-Type Relative-Utility Product Holdout

**Decision:** ADVANCE_FRESH_LEARNED_UTILITY

Split 0: 60 training products, 40 held-out products. Lower is better.

| model | NLL | CRPS | MAE | RMSE |
|---|---:|---:|---:|---:|
| paper_llm_mix_cal | 1.779956 | 0.916821 | 1.352946 | 1.728579 |
| empirical_persona_relative_utility_surrogate | 1.771393 | 0.905507 | 1.333254 | 1.711200 |
| fresh_gpt56_learned_type_relative_utility | 1.772439 | 0.902825 | 1.328201 | 1.707111 |

## Effects

- vs paper zt_avg_nll: -0.422%
- vs paper zt_avg_crps: -1.527%
- vs paper mae: -1.829%
- vs paper rmse: -1.242%

- vs empirical-utility surrogate zt_avg_nll: +0.059%
- vs empirical-utility surrogate zt_avg_crps: -0.296%
- vs empirical-utility surrogate mae: -0.379%
- vs empirical-utility surrogate rmse: -0.239%

## Product bootstrap vs paper

- zt_avg_nll: 95% CI [-0.027610, 0.015631], P(fresh better)=0.755
- zt_avg_crps: 95% CI [-0.027016, -0.001707], P(fresh better)=0.988
- mae: 95% CI [-0.047904, -0.002686], P(fresh better)=0.987

## Scope

Provider scores were frozen before the split labels and demand were reopened for evaluation. This remains a single-conversation GPT-5.6 provider experiment, not independent API calls.
