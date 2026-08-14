# Experiment 002 — Persona necessity / population weighting

## Status

PLANNED / RUNNING. No paper-baseline code or saved outputs are modified.

## Scientific question

Does the current 50-persona virtual population add held-out predictive information that cannot be recovered by a much simpler pooled or empirically weighted population representation?

The saved `llm-mix-cal` fits are sparse and unstable: many persona weights are effectively zero, and a small number of cells carry most of the learned mixture mass. That makes it necessary to test whether the 50-persona interpretation earns its complexity.

## Hypothesis

The free 50-persona mixture may be functioning as a high-dimensional behavioral basis expansion rather than a literal population. If so, a pooled or fixed-population model with only global calibration may recover most of its held-out performance.

## Control

Saved `llm-mix-cal` model for each of the 10 original product splits:

- 50 persona response features
- free simplex mixture weights plus dummy no-buy mass
- monotone logit calibration
- saved exposure `N`

## Treatments

All treatments use the exact same cached raw 50-persona LLM probabilities and the same saved `N`. Each treatment receives only three trainable scalar parameters: calibration intercept, positive calibration slope, and total real-population mass.

1. **pooled_raw_mean** — average raw persona probabilities first, then globally calibrate the single pooled behavioral feature.
2. **uniform_fixed_population** — globally calibrate each persona response, then average all 50 with equal fixed population weights.
3. **empirical_fixed_population** — globally calibrate each persona response, then aggregate using the real H&M cell frequencies from `persona_cells.csv` for P001–P050.

No treatment gets free persona-specific mixture weights.

## What changes

Only the population aggregation mechanism and its parameter count.

## What remains identical

- cached LLM outputs
- 50 persona definitions
- products
- original train/test product splits
- response parsing
- zero-truncated likelihood
- saved `N` for each split
- monotone logit calibration family
- evaluation rows

## Data

Committed repository artifacts only.

## Holdout design

The original 10 deterministic 60:40 product splits.

## LLM calls

0.

## API cost

$0.

## Primary metric

Held-out zero-truncated average NLL.

## Secondary metrics

- MAE
- RMSE
- fitted population mass
- effective number of personas in the saved control
- count of control personas with >=1% normalized mixture weight

## Success / failure interpretation

If a pooled or fixed-weight 3-parameter treatment remains within roughly 1% of the free 50-persona control on mean held-out NLL, the existing benchmark does **not** establish that freely fitted 50-persona heterogeneity is necessary.

If the free 50-persona model is consistently more than about 2% better on held-out NLL, that is evidence that the persona response basis plus learned mixture weights adds real predictive information.

These are diagnostic thresholds, not publication-level significance tests. Any promising result should be followed by a full paired uncertainty analysis and K-persona refit ladder.

## KEEP / KILL rule

- **KEEP the persona-response basis** if it materially improves held-out performance over the fixed/pooled treatments.
- **KILL the literal-population interpretation of free alpha weights** regardless of prediction if empirical H&M weights perform very differently from the fitted mixture.
- **KILL unnecessary persona complexity** if pooled/fixed representations match the control.
- Do not add richer personas until the current persona heterogeneity first proves that it matters.
