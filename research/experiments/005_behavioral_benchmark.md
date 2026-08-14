# Experiment 005 — H&M Behavioral Benchmark Reconstruction

Status: **PREREGISTERED — NO EXPERIMENT 005 MODEL RESULTS INSPECTED**

Date: 2026-08-14
Branch: `research/05-behavioral-benchmark`

## Motivation

Experiments 001–004 showed that the paper's current H&M benchmark is not a convincing behavioral-simulation benchmark. In particular, the saved demand table contains only already-positive product-day counts, and a one-parameter logarithmic-series observation model reaches about 1.4912 heldout NLL without product, customer, price, embedding, persona, or LLM information.

Experiment 005 changes the question rather than tuning the old benchmark.

The intended scientific question is:

> Can we construct a leakage-resistant H&M evaluation in which a model must predict whether an article receives purchases in a future period, and then how many purchases occur, rather than being evaluated only on rows already known to be positive?

## Critical source-data limitation

The public H&M competition transaction table is a purchase log. It does not contain impression, inventory, stock-on-hand, page-view, offer, or explicit article-availability events. Therefore an article-day absent from `transactions_train.csv` is **not an observed no-purchase exposure**. It may mean no demand, no stock, no listing, no traffic, or some combination.

Accordingly:

1. Experiment 005 will **never call inferred zero rows true observed availability-conditioned zeros**.
2. Any zero-inclusive panel made from these data is a **proxy at-risk panel**.
3. All conclusions must be conditional on the selected at-risk rule and must survive sensitivity checks before being treated as useful engineering evidence.
4. If conclusions are materially proxy-dependent, H&M is rejected as a flagship scientific validation dataset for Allegory.

A related limitation is price: on a zero-purchase article-day there is no observed paid price. A carried-forward prior purchase price is not an observed offer. Therefore the primary incidence benchmark will not make causal price-elasticity claims and will not use future-derived price grids.

## Data source

Use the committed artifact:

`outputs/products/txns_trousers_online.csv.zip`

This file was emitted by the original preprocessing code *before* the paper cutoff was applied, after restricting to online-channel trouser purchases. It therefore contains the event history needed for a future temporal audit without requiring new LLM calls.

The original paper cutoff is fixed at **2019-09-20**. Experiment 005 uses only records strictly before this cutoff for train-universe construction and uses records on/after the cutoff only as future outcomes/audit evidence.

## Stage A — feasibility and data audit

Before fitting behavioral baselines, record:

- transaction schema;
- earliest/latest date;
- purchase rows, unique customers, and unique articles;
- per-day and per-article purchase density;
- number of articles with history before the cutoff;
- number first appearing after the cutoff;
- continuation/survival of train-selected products into the first 28 future days;
- gaps between purchase days;
- multiplicity of paid prices on the same article-day;
- whether a price can be observed on inferred zero days (expected: no).

### Fixed temporal windows

- Historical/train side: all dates `< 2019-09-20`.
- Primary future test window: `2019-09-20` through `2019-10-17` inclusive (28 days).
- Where a validation interval is needed, use the final 28 days immediately before the cutoff (`2019-08-23` through `2019-09-19`), and fit feature engineering only on dates before the validation/test observation being predicted.

These dates are fixed before Experiment 005 outcomes are inspected.

## Train-only seen-product universe

Construct a seen-product universe from **pre-cutoff data only**:

1. aggregate purchases by article/date/paid price;
2. retain articles with at least 5 distinct paid prices before cutoff, matching the paper's eligibility concept;
3. rank eligible articles by total pre-cutoff purchase count;
4. take the top 100.

No post-cutoff outcome may affect eligibility or ranking.

This is the primary *seen-product* universe.

## Cold-product audit

Separately identify articles whose **first observed online trouser purchase is on or after 2019-09-20**. Report how many appear in the 28-day test window and their purchase volume.

This stage is descriptive only unless leakage-safe article metadata for those products is available. Article IDs alone are not enough for a credible cold-start semantic model.

## Proxy at-risk definitions

For each seen article and date, define four retrospective/prospective risk-set variants.

### R7 / R14 / R28 — primary future-safe recency proxies

An article is `at_risk_K(t)=1` if it has at least one observed purchase strictly before day `t` within the previous K calendar days, for K in {7, 14, 28}.

The current day's outcome is not used to decide whether the article is at risk. Future observations are never used.

These proxies mean **recently observed selling**, not known in stock.

### Span — retrospective sensitivity only

An article is considered at risk from its first through its last observed purchase date. This uses future information and therefore **must never be the primary operational benchmark**. It exists only to show how much inferred-zero prevalence changes under an optimistic retrospective assumption.

## Labels

For an article-day in a proxy risk set:

- `y_incidence = 1` if at least one purchase of that article is observed that day; otherwise 0.
- `y_count` = total observed purchases of the article that day, including zero.

Multiple transaction rows for the same article/day are summed. Paid-price statistics are retained only on positive days for diagnostics; zero-day price is missing by construction.

## Proxy-feasibility diagnostics and kill rules

Report for R7/R14/R28 and Span, separately for validation and future test:

- number of article-days;
- positive and inferred-zero counts;
- positive prevalence;
- article coverage;
- zero-run lengths;
- fraction of inferred-zero days followed by another purchase within 1/7/14/28 days;
- overlap/Jaccard of risk-set article-days between R7/R14/R28;
- per-product prevalence distribution.

### Kill / downgrade criteria

H&M is **KILLED as a flagship scientific behavioral benchmark** if any of the following holds:

1. the risk-set choice changes incidence prevalence or baseline ranking enough to reverse primary conclusions;
2. fewer than 50 of the 100 train-selected articles contribute at least 7 at-risk article-days in the 28-day future window under R14;
3. the future test has fewer than 2,000 R14 article-days;
4. a substantial fraction of apparent negatives are structurally indistinguishable from discontinuation/stockout and sensitivity analysis cannot bound the effect;
5. richer models appear to win only under the retrospective Span rule but not the future-safe recency rules.

Even if these kill rules do not trigger, H&M remains a **proxy-availability** benchmark, not a ground-truth exposure benchmark.

## Stage B — no-LLM baseline ladder

Only after constructing the proxy panel, fit models with **zero new LLM/API calls**.

The primary risk-set rule is R14. R7 and R28 are mandatory sensitivity analyses.

### Incidence models

I0. Global train prevalence.

I1. Calendar-only logistic model (day-of-week + smooth/ordinal time trend fitted from history).

I2. Article historical-rate model using only lagged purchase history available before each prediction day, with shrinkage toward the global rate.

I3. Article history + recency model using lagged 1/7/14/28-day purchase counts and days since last observed sale.

No model may use same-day outcome-derived price or any future price support.

### Count models

C0. Global unconditional count distribution / mean baseline.

C1. Two-part hurdle baseline: incidence probability from the corresponding I-model plus a positive-count logarithmic-series model estimated only from historical positive counts.

C2. If numerically justified, compare a hurdle negative-binomial positive-count component, but finite dispersion is not interpreted behaviorally because Experiment 003 showed boundary-seeking behavior.

### Evaluation

Primary incidence metric:

- mean Bernoulli log loss on future R14 article-days.

Secondary incidence metrics:

- Brier score;
- average precision / PR-AUC;
- ROC-AUC (secondary because class balance is proxy-dependent);
- calibration error/reliability summary.

Primary unconditional count metric:

- heldout negative log likelihood under the hurdle model.

Secondary count metrics:

- MAE;
- RMSE;
- mean prediction calibration.

Metrics are reported by risk-set definition as well as pooled.

## Stage C — semantic/customer information gate

Experiment 005 does **not** spend LLM calls merely because a harder panel exists.

Proceed to semantic/product/LLM/customer treatments only if Stage B establishes all of the following:

1. the R14 future panel passes the minimum-size rules;
2. R7/R14/R28 yield qualitatively stable baseline conclusions;
3. trivial global/calendar models leave material headroom;
4. product-history models generalize rather than merely fit training prevalence;
5. there is a scientifically meaningful target that product/customer information could plausibly improve.

If this gate passes, a subsequent preregistered treatment experiment will compare, one variable at a time:

- strong non-LLM product-history baseline;
- product semantics without customer state;
- simple/coarse customer state;
- richer leakage-safe customer state;
- LLM-conditioned variants.

The same risk set, outcomes, splits, and downstream observation model must be held fixed across arms.

## Interpretation rules

A better score on this proxy panel may support an **engineering claim** that a representation predicts future purchase events under a recent-selling risk-set assumption.

It does not by itself prove:

- known article availability;
- customer exposure;
- causal price elasticity;
- counterfactual demand under a price intervention;
- inventory-aware lost sales;
- individual purchase probability;
- real-world population simulation validity.

Those require a dataset with explicit exposures/choice sets/inventory/interventions or prospective validation.

## Research hygiene

- `main` remains untouched.
- This branch stays unmerged unless a result earns promotion.
- Audit and treatment code/results are preserved even if negative.
- No success criterion will be changed after seeing results.
- Exploratory sensitivities are labeled exploratory.
- No new LLM calls are authorized in Stage A/B.
