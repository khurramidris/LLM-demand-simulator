# Experiment 005 — H&M Behavioral Benchmark Reconstruction — Results

Status: **COMPLETE — TEMPORAL SIGNAL RESTORED; H&M STILL KILLED AS FLAGSHIP SCIENTIFIC VALIDATION DATASET**

Date: 2026-08-14  
Branch: `research/05-behavioral-benchmark`  
Authoritative leakage-clean workflow run: **31793351528**  
Result artifact: `exp005-behavioral-benchmark-results` (artifact ID **9216367476**)  
Artifact SHA-256: `bda87d01143eb1c6d569fb6d87b172ac5175be4b2c47050078a2e0b3f1f64fcb`

## Executive result

Experiment 005 successfully removed the fatal positive-only pathology of the paper benchmark: once the task includes proxy zero-purchase article-days and a genuinely future temporal test, simple lagged sales history contains large heldout predictive signal.

Under the primary R14 risk set, the leakage-clean history/recency model reduces future incidence log loss from **0.728078** for a global prevalence model to **0.318902**, a **56.20% relative reduction**. It reaches **0.9323 ROC-AUC** and **0.9659 average precision**. The hurdle count NLL falls from **2.635243** to **2.226067**.

The same history/recency model is the heldout incidence-log-loss winner under **R7, R14 and R28**, with relative log-loss reductions versus global of approximately **51.42%, 56.20%, and 60.19%**, respectively.

This is important: the old positive-count benchmark's conclusion that almost nothing beyond the count law matters was an artifact of the task construction. A future event task is genuinely learnable from prior article behavior.

However, the new signal does **not** validate virtual populations or customer personas. The public/committed H&M source is still a purchase-event log without article exposure, page views, stock-on-hand, inventory, listing state, or explicit availability. Therefore its inferred zeros are proxy negatives, not observed available-but-not-bought events. The strongest baseline may partly capture article lifecycle, replenishment, stock, promotion, or merchandising persistence in addition to consumer demand.

Per the preregistered conservative gate, **do not spend LLM calls on Stage C yet** and **do not promote H&M as Allegory's flagship scientific validation dataset**.

## Source audit

The committed online-trouser transaction artifact contains:

- **1,011,876** purchase rows;
- **365,459** unique customers;
- **7,937** unique trouser articles;
- dates from **2018-09-20 through 2020-09-22**;
- no inventory / stock / availability field.

The seen-product universe was rebuilt using only records strictly before **2019-09-20**: articles needed at least five distinct historical paid prices, were ranked by pre-cutoff purchase count, and the top 100 were selected. No post-cutoff outcome was used for this selection.

The primary future test is **2019-09-20 through 2019-10-17**, 28 days.

### Cold-product opportunity

There are **211** trouser articles whose first observed online purchase occurs during the 28-day future test window, accounting for **1,648** purchase rows from **1,551** customers.

This is potentially useful for a later true cold-start benchmark, but article metadata alone cannot establish that each candidate was available/exposed on each zero day. Cold-start work therefore needs a defensible candidate/availability construction before it can support scientific claims.

## Price audit

Price cannot be used as though it were an observed offer on all article-days:

- zero-purchase article-days have **zero observed paid prices** by construction;
- among **38,896** positive article-days, **25,381 (65.25%)** contain more than one paid price.

Thus the public H&M transaction log does not identify a unique article-day offer price, and Experiment 005 makes no causal price-elasticity claim.

## Proxy risk sets

The preregistered future-safe risk sets were:

- **R7:** article sold at least once in the prior 7 days;
- **R14:** article sold at least once in the prior 14 days;
- **R28:** article sold at least once in the prior 28 days.

The current day's outcome and future outcomes are never used to decide membership in these three risk sets.

`Span` (first through last observed purchase) is retrospective sensitivity only and is not used as an operational primary benchmark.

### Future-test diagnostics

| Rule | Article-days | Articles | Positive prevalence | Articles with >=7 risk days | Zero followed by purchase within 14d | Zero followed by purchase within 28d |
|---|---:|---:|---:|---:|---:|---:|
| R7 | 2,241 | 97 | 0.6997 | 93 | 0.8336 | 0.9331 |
| R14 | 2,490 | 98 | 0.6390 | 97 | 0.7809 | 0.8966 |
| R28 | 2,716 | 99 | 0.5902 | 99 | 0.7394 | 0.8500 |
| Span | 2,716 | 98 | 0.5920 | 98 | 0.7536 | 0.8673 |

The high future-repurchase rates make many inferred zero days plausibly transient no-purchase periods rather than simple permanent discontinuations. They still do **not** prove the article was in stock or exposed.

Risk-set overlap is substantial but not perfect. Test Jaccard overlap is:

- R7 vs R14: **0.9000**;
- R14 vs R28: **0.9168**;
- R7 vs R28: **0.8251**.

Test positive prevalence varies by about **10.95 percentage points** from R7 to R28. That sensitivity is one reason H&M remains a proxy benchmark.

## Leakage-clean no-LLM baseline ladder

The first Stage A/B run used a full pre-cutoff article-rate statistic as an I3 training feature. Although valid as a future fitted statistic, this acts like target encoding within the training sample unless cross-fitted. It was therefore removed entirely before accepting the result.

The authoritative I3 model uses only quantities available strictly before each predicted day:

- lagged 1-day purchase count;
- lagged 7/14/28-day purchase counts;
- days since last observed sale;
- calendar time trend;
- day of week.

No product identity target encoding, LLM output, embedding, future price, same-day price, persona, or customer state is used.

### R14 primary future test

| Model | Incidence log loss | Brier | Average precision | ROC-AUC | Hurdle count NLL | Count MAE |
|---|---:|---:|---:|---:|---:|---:|
| I0 global prevalence | 0.728078 | 0.258030 | 0.638956 | 0.5000 | 2.635243 | 5.313288 |
| I1 calendar | 0.658781 | 0.232583 | 0.671525 | 0.5384 | 2.565946 | 4.946526 |
| I2 shrunk article rate | 0.671886 | 0.236255 | 0.839525 | 0.7324 | 2.579051 | 5.174505 |
| **I3 history + recency** | **0.318902** | **0.103674** | **0.965928** | **0.9323** | **2.226067** | **3.658013** |

### Mandatory risk-set sensitivity

| Risk set | Global log loss | History/recency log loss | Relative log-loss reduction | Global count NLL | History count NLL |
|---|---:|---:|---:|---:|---:|
| R7 | 0.654953 | **0.318168** | **51.42%** | 2.760410 | **2.423625** |
| R14 | 0.728078 | **0.318902** | **56.20%** | 2.635243 | **2.226067** |
| R28 | 0.784146 | **0.312202** | **60.19%** | 2.538532 | **2.066587** |

The identity of the best model is stable: **I3 wins R7, R14, and R28**.

The exact full ranking is not identical. R7 orders I3 > I2 > I1 > I0, while R14/R28 order I3 > I1 > I2 > I0. The preregistered implementation conservatively requires the entire ordering to match, so `ranking_stable = false` and the automated flagship kill remains in force. We do not weaken that criterion after seeing the result.

## What changed relative to Experiments 003/004

Experiments 003/004 evaluated rows already known to have positive demand. A one-parameter positive-count distribution could therefore score extremely well without understanding whether a product would sell at all.

Experiment 005 asks a genuinely future event question on a proxy at-risk panel. The 0.728 -> 0.319 incidence-log-loss improvement demonstrates that prior article trajectory matters strongly once zeros/incidence enter the target.

This vindicates the decision to rebuild the benchmark rather than continue tuning personas against the old positive-only table.

It does **not** vindicate the virtual-population thesis. I3 contains no customer representation at all.

## What the strong I3 result probably contains

The predictive signal should be interpreted as **article trajectory / lifecycle signal**, not pure customer-behavior signal.

Because H&M does not reveal stock, listing or exposure, recent purchase history can proxy for several latent processes simultaneously:

- genuine consumer demand persistence;
- article lifecycle and fashion seasonality;
- inventory/replenishment state;
- stockouts or discontinuation;
- promotions or merchandising;
- traffic/exposure changes.

Moreover, each R7/R14/R28 risk set is itself defined from recent observed sales. Using finer lag/recency information within that risk set is operationally leakage-safe but naturally powerful. This is a valid forecasting baseline and an important benchmark hurdle, but it is not evidence that an AI has simulated consumers.

## Preregistered decision

- R14 test article-days >= 2,000: **PASS** (2,490).
- R14 articles with >=7 risk days >= 50: **PASS** (97).
- Same best baseline across R7/R14/R28: **YES** (I3 in all three).
- Exact full baseline ranking identical across R7/R14/R28: **NO**.
- Preregistered automated flagship kill: **TRIGGERED**.
- Proceed to semantic/customer/LLM Stage C: **NO**.

### Scientific decision

**KEEP H&M as an engineering, reproduction, and temporal-forecasting research dataset.**

**KILL H&M as Allegory's flagship scientific validation dataset for population simulation.**

Do not spend new LLM calls trying to make personas win this proxy benchmark. The appropriate next flagship dataset should expose at least one of: explicit impressions/exposures, known choice sets, inventory/availability, randomized interventions, real campaign treatment/control, or prospective outcomes.

A separate H&M cold-start engineering experiment may still be valuable because 211 new trouser articles appear in the fixed future window. It must be labeled as cold-start forecasting under uncertain availability, not population-simulation validation.

`main` remains untouched. No LLM/API calls were made in Experiment 005.
