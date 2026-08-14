# Experiment 004 — Frozen GPT-5.6 Sol Scoring Protocol

Frozen before any Experiment 004 treatment metric is evaluated.

## Why latent scores rather than purchase probabilities

The existing H&M positive-count benchmark does not identify exposure, incidence, or literal purchase probability. Experiment 001 also showed that the paper's exposure parameter is not interpretable. GPT-5.6 Sol therefore supplies a **relative behavioral propensity score**, not a claimed calibrated probability.

## Unit of GPT scoring

One product is scored across its entire offered-price grid at once. To reduce chat-mediated inference volume while preserving product/price reasoning, GPT outputs two latent parameters per arm:

- `appeal_ref` in [0, 100]: expected relative appeal/value at the article's reference price;
- `price_sensitivity` in [0, 30]: nonnegative score-point sensitivity to log price ratio.

The reference price is the geometric mean of the article's offered-price grid, computed mechanically from blinded prices.

For an offered price `price`, the frozen row score is:

`score(price) = clip(appeal_ref - price_sensitivity * ln(price / reference_price), 0, 100)`

This transformation is fixed before target inspection. It intentionally encodes ordinary downward price sensitivity rather than allowing the evaluator to exploit the previously observed noncausal positive price association in the H&M data.

## Arms and populations

### G — Generic

GPT reasons about a generic H&M customer using only the product metadata and offered-price grid. No persona/customer evidence.

### P — Paper-style

GPT reasons about the original top-50 paper persona cells. Cells are represented with their age bin, engagement bin, typical paid-price range, top product category, and `n_customers`; population interpretation uses fixed empirical `n_customers` weighting. GPT is not allowed to choose or learn persona weights from demand targets.

### C — Matched coarse

GPT reasons over the fixed 50-customer matched sample using only each record's coarse state: age bin, engagement bin, price tier, and top product type. The deterministic sample is population-proportional over age × engagement × price-tier strata, so records are equally weighted after sampling.

### R — Matched rich

GPT reasons over the exact same 50 customer records, equally weighted, but receives exact observed age, transaction count / transactions per month, exact mean paid price, and top product type.

No additional psychographic or demographic traits may be inferred as if observed.

## GPT instruction used for every batch

> You are the behavioral scoring component in a preregistered blinded retail-demand experiment. You have no access to sales counts, demand targets, product popularity, heldout outcomes, model residuals, or evaluation metrics. Do not guess historical H&M popularity from external memory. Use only the supplied product metadata, price grid, and the population representation for the requested arm. For each product estimate (1) relative expected appeal/value at the geometric-mean reference price and (2) nonnegative price sensitivity. Appeal 0 means essentially no fit/value, 50 means ordinary/typical fit/value, and 100 means unusually broad/strong fit/value. Price sensitivity is 0–30 score points per unit log-price ratio. Base judgments on plausible category fit, demographic breadth, style/formality, garment utility, color/appearance, product description, and compatibility with the supplied population evidence. Do not output literal purchase probabilities or sales counts. Do not tune to any benchmark result. Preserve article IDs exactly and return only the requested structured records.

## Batch independence / context

The 100 articles may be scored in several batches for practical context size. Every batch uses the same frozen instruction and the same population artifact for its arm. Product order is article-ID order. No batch is rescored after evaluation.

Because this is conversation-mediated GPT-5.6 Sol rather than an API snapshot, exact generative replay is not guaranteed. Reproducibility is instead obtained by committing the frozen latent outputs; all downstream transformations and evaluation from those outputs are deterministic.

## Mechanical validation before target unblinding

The frozen score artifact must satisfy:

- exactly 100 unique article IDs;
- all four arms G/P/C/R present for every article;
- `appeal_ref` ∈ [0,100];
- `price_sensitivity` ∈ [0,30];
- no demand/target columns;
- no duplicate article-arm records;
- article IDs exactly match the blinded product artifact;
- derived row scores are finite for every offered price.

Once these checks pass, the latent artifact is frozen. Only then may the evaluator read `demand`.

## Interpretation caution

A strong result would show that the frozen GPT score carries heldout information about **positive-count magnitude conditional on a sale day**. It would not by itself establish that GPT predicts whether an item sells, customer-level choices, causal price elasticity, inventory-aware demand, or market response under interventions.
