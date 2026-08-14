# Experiment 001 — Exposure-N identifiability

## Status

PLANNED / RUNNING. This experiment does not change the paper baseline.

## Scientific question

Does the latent Binomial exposure parameter `N` have an interior optimum supported by the H&M positive-demand observations, or does model fit continue to improve as the upper bound of the search grid is increased?

The committed paper-style fits selected `N=250` for both `llm-mix` and `llm-mix-cal` in all ten saved product splits, and 250 was the largest candidate value. A parameter that repeatedly lands on the largest allowed value is not evidence that the data identify that value.

## Hypothesis

`N` is weakly identified under the zero-truncated Binomial model because the observed data contain only positive-sales days and the mean is largely governed by the product `N*q`. Calibration and mixture weights can rescale `q`, creating additional trade-offs with `N`.

## Control

The original candidate grid:

`[100, 150, 200, 250]`

## Treatment

Profile the same model over a wider grid while refitting all trainable parameters independently at every value of `N`:

`[100, 150, 200, 250, 350, 500, 750, 1000]`

The first execution uses three of the original deterministic product splits as a cheap diagnostic. A full ten-split run is justified only if the preliminary profile is informative and stable enough to warrant the additional compute.

## What changes

Only the fixed value of `N` supplied to the model fit.

## What must remain identical

- H&M sales rows
- cached LLM responses
- personas and persona order
- product split algorithm
- split seed
- zero-truncated likelihood
- calibration family
- mixture optimization
- training fraction
- held-out products

## Data

Committed repository artifacts:

- `outputs/products/sales_top100_online.csv`
- `outputs/responses/llm_responses_online_top100.csv`

No new LLM calls are made.

## Holdout design

Same product-level split rule as the paper code:

- 60% train products
- 40% test products
- seed `2025 + split_idx`

## LLM call count

0.

## API cost

$0.

## Primary metrics

- train zero-truncated average NLL
- test zero-truncated average NLL
- selected/profile-optimal `N`

## Secondary diagnostics

- test MAE and RMSE using the zero-truncated conditional mean
- dummy no-buy mass
- effective number of fitted personas
- largest normalized persona weight
- calibration intercept and slope

## Success condition for the exposure interpretation

The train likelihood should show a clear interior optimum that is stable across splits, and held-out NLL should not systematically improve by increasing `N` beyond that optimum.

## Failure condition

Any of the following is evidence against interpreting fitted `N` as literal customer exposure:

1. the optimum remains at the new upper boundary;
2. likelihood continues to improve materially as `N` increases;
3. the profile is nearly flat over a wide range of large `N`;
4. profile-optimal `N` varies dramatically across otherwise similar splits while predictive performance is unchanged.

## KEEP / KILL rule

- **KEEP predictive model** if held-out prediction remains useful.
- **KILL literal exposure interpretation** if `N` is boundary-seeking or weakly identified.
- If killed, the next model experiment should compare a formulation that predicts a count-rate parameter directly (for example a zero-truncated Poisson / Negative-Binomial family) rather than pretending `N` is measured exposure.

## Reproducibility

The experiment script writes after every completed fit so interrupted runs are resumable. It records software versions, GitHub SHA/ref, grid, seeds, row counts, and all fitted diagnostics under `outputs/research/exposure_identifiability/`.
