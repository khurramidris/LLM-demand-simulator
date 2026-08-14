# Full-Population Representation Experiment

**Decision:** PROMISING_BUT_NOT_PROVEN

Pre-registered strict-vs-previous gate: **True**  
Same-provider representation gate: **{representation_gate}**  
Rich diagnostic signal (non-claimable): **True**

| model | NLL | CRPS | MAE | RMSE |
|---|---:|---:|---:|---:|
| paper_llm_mix_cal | 1.779956 | 0.916821 | 1.352946 | 1.728579 |
| previous_fresh_gpt56_9type | 1.772439 | 0.902825 | 1.328201 | 1.707111 |
| control9_common_provider | 1.772428 | 0.901831 | 1.323499 | 1.705338 |
| strict_full_population_common_provider | 1.771844 | 0.902294 | 1.325331 | 1.706277 |
| rich_diagnostic_common_provider | 1.772215 | 0.901772 | 1.323391 | 1.705410 |

## Strict full-population effects

### strict_vs_control9
- zt_avg_nll: -0.033%
- zt_avg_crps: +0.051%
- mae: +0.138%
- rmse: +0.055%
### strict_vs_previous_fresh
- zt_avg_nll: -0.034%
- zt_avg_crps: -0.059%
- mae: -0.216%
- rmse: -0.049%
### strict_vs_paper
- zt_avg_nll: -0.456%
- zt_avg_crps: -1.584%
- mae: -2.041%
- rmse: -1.290%

## Scientific scope

Strict non-buyer segmentation uses age only; observed trouser-buyer clustering uses only the 60 allowed training products. Rich diagnostic all-category features are explicitly not claimable because the raw full transaction history is absent from the repository.
