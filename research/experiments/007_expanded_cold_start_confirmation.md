# Experiment 007 — Expanded H&M Cold-Start Population Confirmation

Status: **PREREGISTERED — confirmatory cohort outcomes not inspected when frozen**

Date: 2026-08-14  
Branch: `research/07-expanded-cold-start-confirmation`
Parent research state: Experiment 006 result commit `2073447f0ab1e13f94d079f07d288b99cffd25fa`
Production `main` remains `b56a7c0acad7406bff81b7cdf179314894b2fa97`.

## Motivation

Experiment 006 found statistically favorable product/customer-composition signal but failed its frozen sample-size floor: 21 qualified cold products versus the required 30. The effect gate, buyer-count gate, support gate and product-cluster-bootstrap uncertainty gate all passed.

We do **not** lower the 30-product requirement after observing Experiment 006. Instead Experiment 007 uses a new, non-overlapping future launch cohort and a stricter size requirement.

The scientific question is now confirmatory:

> With customer states frozen before 2019-09-20, does a pre-cutoff persona style profile predict the composition of buyers for entirely new H&M trouser products launched in a later, untouched future period better than a generic historical trouser-buyer prior?

This is still a conditional buyer-composition benchmark under a first-observed-sale launch proxy. It is not an exposure, inventory, conversion or causal-demand benchmark.

## Frozen model-information cutoff

All customer states, persona assignments, style histories, prior distributions, product-content training data and supervised model training/tuning remain restricted to information available before **2019-09-20**, exactly as in Experiment 006.

No transactions from the Experiment 006 test cohort or the Experiment 007 confirmatory cohort may be used to update:

- persona membership;
- P0/P1 priors;
- P2 style centroids;
- P3 model fitting;
- TF-IDF fitting;
- hyperparameters.

## Independent confirmatory launch cohort

Experiment 006 test launches ended on **2019-10-17**.

Experiment 007 uses only articles whose first observed online-trouser purchase occurs from:

- **2019-10-18 through 2020-01-31 inclusive**.

This cohort does not overlap the Experiment 006 test launch cohort.

For each article, the observed buyer-composition outcome is the first **28 days** from first observed sale (launch day through +27 days). The latest outcome therefore ends on 2020-02-27.

A product qualifies if it has at least **10 mapped unique buyers** among the frozen top-50 persona population in its 28-day outcome window. Each customer is counted at most once per article.

## Frozen population

Use the same first 50 persona cells and the same **244,171** mapped pre-cutoff customers reconstructed in Experiment 006.

No post-cutoff behavior may change customer membership or persona attributes.

## Frozen model ladder

The model definitions are unchanged from Experiment 006.

### P0 — population prior

Top-50 persona probabilities proportional to frozen persona cell population sizes.

### P1 — historical trouser-buyer prior

Top-50 persona distribution over unique pre-cutoff trouser customer/article purchase pairs.

### P2 — historical persona style-profile model

Same train-only TF-IDF article metadata representation and same pre-cutoff persona style centroids as Experiment 006.

**Hyperparameter is frozen, not retuned:** `tau = 8.0`, selected on the pre-Experiment-006 validation cohort.

P2 probabilities remain:

`softmax(log(P1 + eps) + 8.0 * cosine_similarity(product, persona_style_centroid))`.

### P3 — supervised cold-start product-content model

Same historical launch training design and same model family as Experiment 006.

**Regularization is frozen, not retuned:** `alpha = 0.0001`, selected on the pre-Experiment-006 validation cohort.

P3 is retained as the strongest conventional content hurdle. It does not define the population-evidence gate; P2 does.

## Primary confirmatory comparison

The primary comparison is **P2 persona style-profile versus P1 historical trouser-buyer prior**.

This comparison is chosen before seeing Experiment 007 outcomes because P2 is the Stage-A treatment most directly aligned with the population-representation thesis: it uses frozen historical style evidence aggregated within the 50 customer/persona groups.

Primary metric: buyer-level NLL.

Primary effect:

`relative improvement = (NLL_P1 - NLL_P2) / NLL_P1`.

Primary paired uncertainty: product-cluster bootstrap over qualified Experiment 007 products, **2,000 resamples**, RNG seed `20260814`, using buyer-weighted NLL difference `P2 - P1`.

## Secondary comparisons

Report without changing the primary decision:

- P3 versus P1 buyer NLL and product-cluster bootstrap CI;
- P2 versus P3 buyer NLL;
- Top-1 and Top-5 buyer accuracy;
- macro article-level Jensen-Shannon divergence;
- macro article-level total-variation distance;
- early versus late half directional consistency for P2-P1 NLL difference.

## Confirmatory sample-size gate

Experiment 007 is accepted as an adequately sized H&M cold-start composition benchmark only if all are true:

1. at least **60 qualified confirmatory cold products**;
2. at least **1,500 mapped unique buyer events**;
3. median mapped buyers per qualified product >= **10**.

These thresholds are stricter than Experiment 006 and are frozen before the expanded cohort is evaluated.

## Population-signal success gate

The H&M benchmark earns a direct LLM-population challenge only if, in addition to the sample-size gate:

4. P2 improves buyer NLL over P1 by at least **1.0% relative**;
5. the product-cluster-bootstrap 95% CI for `NLL_P2 - NLL_P1` is strictly below zero;
6. P2-P1 NLL improvement has the same favorable direction in both chronological halves of the confirmatory launch cohort;
7. P2 does not worsen macro JS divergence by more than **5% relative** versus P1.

If any criterion fails, do not spend new LLM calls on H&M.

## What a pass means

A pass would establish a limited but meaningful result:

> Pre-cutoff customer/persona style structure predicts which customer groups subsequently buy unseen H&M products on an independent future cohort.

It would **not** establish that an LLM population works. It would only justify the next direct challenge:

- LLM-C: coarse paper personas;
- LLM-R: richer pre-cutoff persona evidence;
- P3: supervised statistical content hurdle.

The LLM would then have to beat P3 under a separately frozen query/evaluation protocol.

## What a fail means

A fail means the promising Experiment 006 result did not survive an adequately sized independent future cohort, or that H&M does not supply enough qualified products even when the confirmatory window is expanded. In either case, stop trying to use H&M as scientific population-validation evidence.

## Interpretation limits

Even a successful Experiment 007 remains conditional on observed buyers under a first-sale launch proxy. H&M does not provide verified impressions, listing state, inventory or exposure. Therefore no result may be described as:

- conversion prediction;
- absolute demand prediction;
- inventory-conditioned demand;
- causal price elasticity;
- randomized intervention evidence.

No LLM/API calls are permitted in Experiment 007.
