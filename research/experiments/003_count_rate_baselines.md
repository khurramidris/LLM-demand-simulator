# Experiment 003 — Count-rate baseline ladder

## Status

Preregistered before looking at treatment results.

## Motivation

Experiments 001 and 002 changed the interpretation of the H&M simulator:

1. the fitted finite Binomial exposure `N` is not identified as literal customer exposure;
2. freely learning ~50 persona mixture weights is unnecessary on the current held-out product benchmark.

The next scientific question is therefore whether the remaining useful signal is genuinely LLM-derived, or whether a conventional count-rate model using product content and price can match it.

This experiment preserves the published paper baseline and the exact original product splits. It adds a rate-model ladder; it does not alter `main` or overwrite the paper artifacts.

## Primary question

**Does a calibrated aggregate of cached LLM behavioral responses materially outperform strong non-LLM count-rate baselines on held-out products?**

## Secondary questions

1. Is overdispersion supported — i.e. does a zero-truncated Negative Binomial materially improve over zero-truncated Poisson?
2. Do SigLIP product embeddings plus nonlinear price information add useful cold-product signal?
3. Is a global/price-only count model already sufficient for this selected trousers benchmark?
4. Does the LLM signal survive after replacing the nonidentified finite-Binomial `N,q` factorization with a direct count-rate model?

## Fixed evaluation protocol

- Use the committed H&M processed data and cached LLM responses only.
- Use the exact original deterministic 10 product splits in `outputs/demand_prediction/split_000` through `split_009`.
- Fit all new model parameters on training-product rows only.
- Evaluate on the same positive-demand rows used by the paper benchmark.
- Do not use test-product demand to select hyperparameters or transformations.
- PCA, feature scaling, spline transformations, and regularization selection must be learned from training products only.
- Preserve the paper's `llm-mix-cal` metrics as a historical control; do not refit or modify the control.
- New LLM calls: 0.
- API cost: $0.

This experiment intentionally retains the paper's positive-only/zero-truncated evaluation so results remain comparable. It does **not** solve the missing-zero-demand limitation.

## Baseline ladder

### A. `ztp-global`

A one-rate zero-truncated Poisson model. This is the minimum serious count baseline.

### B. `ztnb-global`

A global zero-truncated Negative Binomial model with a learned dispersion parameter. Tests whether overdispersion alone explains part of the apparent model advantage.

### C. `ztp-price-spline`

Zero-truncated Poisson with a nonlinear spline of observed price. Predictive only; no causal price interpretation is allowed.

### D. `ztnb-price-spline`

Same price representation with a learned Negative Binomial dispersion parameter.

### E. `ztp-siglip-price`

Zero-truncated Poisson with:

- train-only PCA of the committed SigLIP product text+image embedding;
- nonlinear price spline;
- regularized log-rate regression.

This is the main stronger non-LLM cold-product baseline.

### F. `ztnb-siglip-price`

Same content/price mean model with Negative Binomial observation noise.

### G. `ztp-llm-pooled`

A deliberately low-capacity LLM treatment:

- average the 50 cached persona-conditioned raw purchase probabilities;
- transform the pooled probability to logit space;
- fit a monotone mapping from that behavioral feature directly to a Poisson count rate.

This tests whether the LLM feature itself carries useful held-out signal after removing free persona weights and the finite-exposure interpretation.

### H. `ztnb-llm-pooled`

Same low-capacity LLM rate model with learned Negative Binomial dispersion.

### Historical control

The saved paper `llm-mix-cal` result is reported unchanged for direct comparison.

## Metrics

Primary:

- zero-truncated average NLL;
- zero-truncated CRPS.

Secondary:

- MAE;
- RMSE under the repository's pair-level aggregation convention;
- randomized PIT KS;
- 90% and 95% interval scores;
- fitted Negative Binomial dispersion;
- train-to-test generalization gap;
- paired split-level differences and 95% intervals.

## Regularization

For the high-dimensional SigLIP models, choose L2 strength using group cross-validation **within training products only**. Candidate strengths are fixed before the run. The test products are not consulted.

All feature preprocessing is fit on training data only.

## Decision rules

### Does LLM earn its role?

- **KEEP LLM behavioral elicitation as core evidence** if the best low-capacity LLM rate model beats the best non-LLM rate baseline by at least ~2% on held-out NLL or CRPS and the paired split uncertainty supports the direction.
- **KEEP as useful but not yet core** if the LLM model wins by 1–2% but uncertainty overlaps zero.
- **DOWNGRADE/KILL as core for this benchmark** if a strong non-LLM model is within ~1% of the LLM rate model on both held-out NLL and CRPS.

### Observation family

- Prefer Negative Binomial if it improves held-out NLL/CRPS by ~1% or more over its matched Poisson mean model with stable paired direction.
- Prefer Poisson if NB adds no reliable held-out gain; do not keep a dispersion parameter merely because it improves training likelihood.

### Product content

- If SigLIP+price beats price-only by >1% with stable held-out improvement, keep product semantic/multimodal representation.
- If price-only/global is within ~1%, the selected H&M trousers benchmark is too easy to justify a rich product encoder.

## Interpretation constraints

Regardless of outcome:

- no causal price-elasticity claims;
- no claim that positive-only H&M rows measure total market demand;
- no claim that the LLM simulates literal individual customers;
- no use of the test set to select model complexity;
- a small point-estimate win without paired uncertainty is not called superiority.

## Next action after this experiment

If the LLM survives: attack it with stricter leakage-safe and temporal holdouts before spending new API budget.

If the LLM does not survive: simplify Allegory's H&M engine and move the LLM/persona hypothesis to datasets where genuine customer heterogeneity or intervention response can be validated.
