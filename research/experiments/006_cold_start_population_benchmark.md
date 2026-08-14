# Experiment 006 — H&M Cold-Start Population Benchmark

Status: **PREREGISTERED — outcomes not inspected when this design was frozen**

Date: 2026-08-14
Branch: `research/06-cold-start-population-benchmark`
Base: `main` at `b56a7c0acad7406bff81b7cdf179314894b2fa97`

## Why this experiment exists

Experiments 003–005 showed that the paper's original positive-only demand benchmark is too weak to establish that an LLM virtual population predicts real consumers. Experiment 005 rebuilt a future article-day task and found strong signal, but the winning model was pure article history/recency. That benchmark therefore tests lifecycle forecasting, not whether customer representations explain *who* buys a product.

Experiment 006 gives customer/population modeling a task it can actually earn: **predict the composition of real buyers for products with no pre-launch sales history.**

The benchmark is deliberately conditional. It does **not** claim to identify absolute demand, inventory-conditioned conversion, price elasticity, or exposure-conditioned purchase probability. H&M has no inventory/impression/listing log. Instead it asks a narrower but defensible question:

> Given a newly appearing trouser article and only information available before the test cutoff, which pre-defined H&M customer/persona groups will disproportionately appear among its subsequent buyers?

If product-aware customer/population models cannot beat strong prior and historical-style baselines here, there is no scientific reason to spend money on an LLM-population treatment on H&M.

## Frozen source data

Primary local sources already committed on `main`:

- `outputs/products/txns_trousers_online.csv.zip`
  - online-channel trouser purchase events;
  - contains `t_dat`, `customer_id`, `article_id`, `price`, `sales_channel_id`;
  - no inventory, stock, listing, impression or availability column.
- `outputs/personas/customer_features.csv.zip`
  - constructed only from H&M transactions between 2018-09-01 and 2019-09-19;
  - contains age, total historical transaction count, historical mean paid price and top product type.
- `outputs/personas/persona_cells.csv`
  - paper's pre-cutoff persona-cell definitions.

Article metadata for all 105,542 H&M articles is fetched read-only from the Microsoft Hugging Face mirror `microsoft/hnm-search-data`, `articles` subset. This mirror duplicates the original Kaggle H&M article metadata. The experiment records the dataset revision/fingerprint when available.

## Frozen time design

All dates are calendar dates in the transaction log.

- **Model cutoff / test launch start:** 2019-09-20.
- **Test launch window:** 2019-09-20 through 2019-10-17 inclusive.
- **Outcome horizon:** 28 days beginning at each article's first observed online trouser sale (launch day through launch+27 days).
- **Validation launch window:** 2019-07-26 through 2019-08-22 inclusive. Every validation outcome therefore ends before the 2019-09-20 cutoff.
- **Training launches:** first observed sale on or before 2019-06-27. Their 28-day outcomes end before the validation launch window.
- Articles whose first observed sale falls in the temporal embargo gaps are not used for supervised training/tuning.

The first observed sale is a **proxy launch date**, not a verified listing date. This limitation is reported explicitly and is not upgraded into an availability claim.

## Population definition

Use the first **50 persona cells** from the already-committed `persona_cells.csv`, matching the paper's queried population.

Customers are mapped back into these cells using the same pre-cutoff rules used by the paper:

- age bin;
- engagement tercile from historical transactions per month;
- historical mean-price tercile;
- top product-type taste bucket.

A buyer is evaluable only if the customer maps to one of these 50 frozen cells. No post-cutoff customer behavior may alter cell membership.

## Target construction

For every article:

1. Determine its first observed online-trouser sale date from the committed transaction log.
2. Define its 28-day buyer window as launch day through launch+27 days.
3. Deduplicate to **one buyer event per `(article_id, customer_id)`** within that window so repeat purchases by one customer cannot dominate composition.
4. Map buyers to the frozen top-50 persona cells.

A launch article is **qualified** for evaluation if it has at least **10 mapped unique buyers** in its 28-day window. No minimum persona-diversity rule is imposed.

The primary target is the persona label of each mapped unique buyer, conditional on the article and on being a buyer of that article. At article level this is equivalent to the 50-way buyer-composition distribution.

## What this benchmark can and cannot establish

It **can** test whether product information plus a customer/population representation predicts *relative buyer composition for genuinely cold products* better than generic population priors and historical style models.

It **cannot** establish:

- that non-buyers were exposed;
- conversion probability;
- absolute market demand;
- inventory-conditioned demand;
- causal price elasticity;
- causal effect of a message/promotion;
- true product-listing time.

Any result is reported as **cold-start buyer-composition prediction under a first-sale launch proxy**.

## Stage A — zero-API benchmark and strong baselines

No LLM/API calls are allowed in Stage A.

### P0 — population prior

For each article, predict the same 50-way distribution proportional to `n_customers` in the frozen top-50 persona cells.

### P1 — historical trouser-buyer prior

Using only pre-cutoff transaction history, compute the persona distribution over unique `(article_id, customer_id)` trouser-buyer pairs. Predict this same distribution for every future article.

This controls for the fact that the top-50 cells do not buy trousers in proportion to their raw population sizes.

### P2 — historical style-profile model

Construct a text representation for each trouser article from pre-existing metadata:

- product name;
- product type;
- graphical appearance;
- colour group;
- perceived colour;
- department;
- index/index group;
- section;
- garment group;
- detail description.

Fit TF-IDF using only article metadata from products first observed before the test cutoff. For each persona, construct a style centroid from its **pre-cutoff unique customer-article purchases**. Score a launch article by cosine similarity to each persona centroid.

Convert similarity scores to a 50-way probability using:

`softmax(log(P1 + eps) + tau * cosine_similarity)`

where `tau` is selected **only on the validation launch set** from the frozen grid `{0, 0.5, 1, 2, 4, 8, 16, 32}` by buyer-level negative log likelihood. After choosing `tau`, the persona centroids may be rebuilt from all pre-cutoff history but test labels remain untouched.

P2 is a strong non-LLM analogue of a rich taste-profile population: it uses the actual historical fashion attributes bought by each persona.

### P3 — supervised cold-start content model

Train a multiclass product-content classifier to predict buyer persona from product text/metadata on historical launch articles whose 28-day outcome windows are fully available before validation.

- Feature representation: the same train-only TF-IDF product representation used above.
- Training unit: `(launch article, persona)` with sample weight equal to mapped unique-buyer count for that persona/article.
- Model family: multinomial/log-loss linear classifier with regularization chosen only on validation buyer NLL from a small frozen grid.
- After hyperparameter selection, refit using every eligible launch article whose 28-day outcome ends before 2019-09-20.

P3 is intentionally strong: an LLM population must beat a learned product-to-buyer-composition model, not a toy baseline.

## Primary metrics

### Primary: buyer-level NLL

For each test article and each mapped unique buyer, score `-log(predicted_probability_of_buyer_persona)`. Average across all buyer events.

This is a strictly proper scoring rule and is the primary model ranking metric.

### Secondary

- buyer-level top-1 accuracy;
- buyer-level top-5 accuracy;
- macro article-level Jensen-Shannon divergence between predicted and observed persona compositions;
- macro article-level total-variation distance;
- calibration of predicted persona mass versus observed persona mass.

## Uncertainty

Primary paired uncertainty is a **product-cluster bootstrap** over qualified test articles, 2,000 deterministic resamples with RNG seed `20260814`. Report the 95% percentile interval for NLL difference between each model and the strongest non-product prior (`min(P0,P1)`).

No individual buyer is bootstrapped independently because buyers within one product are not exchangeable with buyers across products.

## Stage-A gate before any LLM spend

Proceed to an LLM treatment only if all are true:

1. at least **30 qualified test cold articles**;
2. at least **500 mapped unique test buyer events**;
3. median mapped buyers per qualified test article is at least **10**;
4. at least one product-specific model, P2 or P3, improves primary NLL over the best of P0/P1 by **>=1.0% relative**;
5. its product-cluster-bootstrap 95% CI for `NLL_model - NLL_best_prior` is strictly below zero.

If these fail, **do not spend LLM calls on H&M**. The result means H&M cannot support even a product-specific buyer-composition benchmark under this construction.

## Stage B — frozen LLM-population challenge (only if Stage A passes)

If the gate passes, create a label-blind query plan for at most the 50 qualified test products with the most mapped buyers and all 50 frozen personas.

Two arms are prepared:

### LLM-C — coarse paper persona

Use the original persona information only: age bin, engagement range, typical paid-price range, top product type, plus the cold product metadata. Ask for one purchase-affinity score/probability for the product. Do **not** include any post-cutoff demand, test buyers, first-28-day counts, or test-derived popularity.

### LLM-R — rich historical persona

Use the same coarse fields plus **pre-cutoff trouser style evidence aggregated for that persona**, such as top colours, graphical appearances, garment groups, sections, representative historically purchased product names/descriptions, trouser purchase frequency and trouser paid-price summary. No future behavior is allowed.

For each product, convert the 50 LLM affinities into predicted buyer composition via population-weighted normalization. No supervised test-set fitting of persona weights is allowed.

The exact query plan and prompts must be written to disk and SHA-256 hashed before LLM outputs or test labels are joined into the evaluation table.

## LLM success criterion

An LLM-population arm earns promotion only if it:

- improves buyer-level NLL over **P3**, the strongest supervised content baseline, by at least **1.0% relative**;
- has a product-cluster-bootstrap 95% CI for `NLL_LLM - NLL_P3` strictly below zero;
- does not materially degrade macro JS divergence (>5% relative worse than P3);
- shows the same direction of improvement when test products are split into early/late launch halves.

LLM-R must also beat LLM-C under the same paired product bootstrap before richer personas can be claimed as beneficial.

A null result is a valid outcome and must not be tuned away on this test set.

## Frozen interpretation rules

- **P2/P3 fail gate:** H&M remains engineering/reproduction only; no LLM spend.
- **P2/P3 pass, LLM fails:** H&M supports cold-start buyer-composition prediction, but not an LLM-population advantage.
- **LLM-C beats P3:** evidence that the coarse LLM population adds predictive value for cold-start buyer composition.
- **LLM-R beats both P3 and LLM-C:** evidence that richer persona grounding adds incremental value.

Even the strongest possible Experiment 006 result is **not** exposure-conditioned conversion or causal demand validation. Those require a different dataset with impressions/availability/interventions.

`main` must remain untouched unless a result later earns promotion and the user explicitly requests merging.
