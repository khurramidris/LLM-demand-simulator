# Clean All-Category Population Representation

**Decision:** DO_NOT_ADVANCE_CLEAN_RICH_TYPES_TO_GPT56

Held-out products are removed from the raw H&M transaction history before feature construction.

| model | NLL | CRPS | MAE | RMSE |
|---|---:|---:|---:|---:|
| paper_llm_mix_cal | 1.779956 | 0.916821 | 1.352946 | 1.728579 |
| previous_fresh_gpt56 | 1.772439 | 0.902825 | 1.328201 | 1.707111 |
| coarse_trouser_types_empirical_utility | 1.773539 | 0.903234 | 1.328667 | 1.707473 |
| clean_all_category_types_empirical_utility | 1.771946 | 0.903819 | 1.330623 | 1.709444 |

## Rich vs coarse clean representation

- zt_avg_nll: -0.090%
- zt_avg_crps: +0.065%
- mae: +0.147%
- rmse: +0.115%

## Bootstrap

- zt_avg_nll: 95% CI [-0.011460, 0.006122], P(rich better)=0.622
- zt_avg_crps: 95% CI [-0.004280, 0.005310], P(rich better)=0.412
- mae: 95% CI [-0.008570, 0.012102], P(rich better)=0.360

## Population

- customers: 1,371,980
- active in clean pre-cutoff history: 1,004,835
- rich types: 30; largest weight 10.10%
- coarse types: 14; largest weight 27.55%

This is a zero-LLM scientific gate. Only a passing result earns a fresh GPT-5.6 representation experiment.
