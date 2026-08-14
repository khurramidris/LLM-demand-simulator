# Experiment 006 — H&M Cold-Start Population Benchmark — Results

Status: **STAGE A FAILED ON PREREGISTERED SAMPLE-SIZE FLOOR; PRODUCT/CUSTOMER SIGNAL IS POSITIVE**

Date: 2026-08-14  
Branch: `research/06-cold-start-population-benchmark`  
Authoritative workflow run: **31795921370**  
Artifact: `exp006-cold-start-population-results` (ID **9217772046**)  
Artifact SHA-256: `d1a1cfbf2bc0b02bdad8e69de4033ceadd4a0c178ed643c751a6dc79aa63c5b4`

## Executive result

Experiment 006 created a materially stronger H&M benchmark than the paper's original positive-only count task: a cold-start, customer-composition prediction problem in which a product has no pre-launch sales trajectory to exploit.

The benchmark produced **21 qualified genuinely future cold trouser articles**, **598 mapped unique buyers**, and a median of **22 mapped buyers per product**.

The strongest generic prior was the historical trouser-buyer persona distribution (P1), with buyer NLL **3.684555**. Both product-specific treatments improved on that prior:

- P2 historical persona style-profile: NLL **3.627002**;
- P3 supervised cold-start product-content model: NLL **3.620464**.

P3 therefore improved buyer NLL by **1.739% relative** versus P1. The preregistered product-cluster bootstrap difference `P3 - P1` was **-0.064091 NLL**, with 95% CI **[-0.107932, -0.024198]**, entirely below zero.

Thus the failure was **not** absence of product/customer signal. Four of the five preregistered Stage-A gates passed. The sole failed gate was benchmark size: **21 qualified products < the frozen minimum of 30**.

Per preregistration, Stage B is not run and **zero LLM/API calls** are made.

## Benchmark

- Online-trouser transaction rows: **1,011,876**.
- Frozen top-50 mapped customers: **244,171**.
- Article metadata coverage: **100%** for the relevant trouser articles.
- Model cutoff: **2019-09-20**.
- Test launch window: **2019-09-20 through 2019-10-17**.
- Each product outcome: mapped unique buyers in the first **28 days** from its first observed sale.
- Qualified product threshold: at least **10 mapped unique buyers**.

The launch date remains a first-observed-sale proxy, not verified listing/exposure time.

## Stage-A results

| Model | Buyer NLL ↓ | Top-1 ↑ | Top-5 ↑ | Macro JS ↓ | Macro TV ↓ |
|---|---:|---:|---:|---:|---:|
| P0 population prior | 3.828607 | 0.0535 | 0.2559 | 0.312271 | 0.630885 |
| P1 historical trouser-buyer prior | 3.684555 | 0.0702 | 0.2826 | 0.260378 | 0.543302 |
| P2 persona style-profile | **3.627002** | **0.0753** | **0.2910** | **0.247815** | **0.525009** |
| P3 supervised product content | **3.620464** | **0.0753** | 0.2843 | 0.250189 | 0.529492 |

P3 has the best primary buyer NLL. P2 has the best Top-5 accuracy, macro Jensen-Shannon divergence and macro total-variation distance.

## Frozen gate

- >=30 qualified test products: **FAIL (21)**.
- >=500 mapped unique test buyers: **PASS (598)**.
- median >=10 mapped buyers/product: **PASS (22)**.
- >=1% relative NLL improvement from P2/P3 over best prior: **PASS (1.739% for P3)**.
- product-cluster-bootstrap 95% CI strictly below zero: **PASS (P3-P1: [-0.107932, -0.024198])**.
- Proceed to LLM challenge: **NO**.

We do not weaken the sample-size rule after observing the result.

## Additional audit: P2 persona style-profile

The primary gate selected P3 as the best product-specific model, so the preregistered summary reported P3's bootstrap comparison. A post-result audit using the exact same product-cluster bootstrap procedure and seed shows that P2 itself improves over P1 by **0.057553 NLL**, or approximately **1.562% relative**, with 95% CI **[-0.080662, -0.035346]**.

This audit is explicitly exploratory because that P2-specific CI was not the single comparison stored by the preregistered decision function. It strengthens the interpretation that the detected signal is not solely an artifact of P3's supervised classifier.

## Scientific interpretation

Experiment 006 is the first H&M experiment in this sequence to find evidence directly aligned with a customer/population question:

> for future products with no own sales history, product characteristics contain statistically detectable information about **which frozen customer/persona groups become buyers**.

That is useful. It means H&M is not devoid of population-composition signal, and a customer-aware cold-start benchmark is scientifically more meaningful than the paper's original positive-only demand task.

However, this is **not evidence that an LLM population helps**. No LLM was run. P2 is a deterministic historical style-profile population model; P3 is a supervised statistical content model.

The confirmatory cohort is also smaller than the frozen sample-size floor, so we do not promote the result as a finished flagship validation claim.

## Next experiment earned by this result

The appropriate next H&M experiment is a new, separately preregistered **larger future launch cohort** using the same frozen customer population, same conditional buyer-composition target, same P0-P3 models and the same untouched 2019-09-20 model cutoff. The larger cohort must be defined before its outcomes are evaluated.

If that independent larger future cohort confirms the product/customer effect with adequate product count, H&M will have earned a direct LLM-C versus LLM-R versus P3 challenge.

Even then, H&M cannot establish exposure-conditioned conversion, absolute demand, inventory-conditioned demand or causal intervention effects because impressions/availability are absent.

`main` remains untouched. PR #7 remains draft and unmerged.
