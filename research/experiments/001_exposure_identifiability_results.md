# Experiment 001 — Exposure-N identifiability: first result

## Decision

**KILL the interpretation of fitted `N=250` as literal customer exposure.**

**KEEP `llm-mix-cal` as a predictive benchmark.** This experiment attacks the semantics/identifiability of the latent exposure parameter, not the existence of held-out predictive signal.

## Run provenance

- Branch: `research/01-exposure-identifiability`
- Pull request: #2
- GitHub Actions run: `31784171980`
- Diagnostic job: `poisson-limit-diagnostic`
- Cached LLM calls used: existing committed responses only
- New LLM calls: **0**
- API cost: **$0**
- Saved product splits evaluated: **10/10**

The workflow first verified that the committed paper-style baseline artifact still reports:

- LLM-cal test CRPS: `0.9426390618243496`
- LLM-cal test MAE: `1.38048553572638`

## Diagnostic performed

For every saved `llm-mix-cal` split model, start from its fitted `N=250` purchase probabilities `q_250` and define the fitted count rate:

`lambda = 250 * q_250`.

Then move along a constant-rate path by setting:

`q_N = lambda / N`

for progressively larger values of `N`, without refitting mixture weights or calibration. This isolates how much the zero-truncated distribution itself can distinguish finite Binomial exposure from the Poisson limit when the predicted mean rate is preserved.

This is a **practical-identifiability diagnostic**, not a replacement for the independently-refit likelihood profile.

## Test-set result across 10 splits

| N | Mean zero-truncated NLL | Delta vs saved N=250 | Gap to Poisson limit |
|---:|---:|---:|---:|
| 250 | 1.834602 | 0.000000 | 0.009031 |
| 350 | 1.831935 | -0.002667 | 0.006364 |
| 500 | 1.829983 | -0.004619 | 0.004412 |
| 750 | 1.828491 | -0.006112 | 0.002919 |
| 1,000 | 1.827753 | -0.006849 | 0.002182 |
| 2,000 | 1.826656 | -0.007946 | 0.001085 |
| 10,000 | 1.825788 | -0.008815 | 0.000216 |
| Poisson limit | ~1.825571 | -0.009031 | 0 |

Moving from `N=250` to `N=10,000` improves mean test NLL by about **0.48%** while preserving `lambda=Nq`.

The train result behaves the same way: mean NLL falls from `1.795495` at `N=250` to `1.787567` at `N=10,000`, about **0.44%**.

## Conditional-moment result

At the saved `N=250`, the average absolute difference from the zero-truncated Poisson limit is already tiny:

- conditional mean difference: `0.002777` demand units on test rows
- conditional variance difference: `0.012260` demand-units squared on test rows

At `N=10,000` these shrink to approximately:

- conditional mean difference: `0.000069`
- conditional variance difference: `0.000307`

## Interpretation

The result is stronger than merely observing that the original grid selected its upper boundary.

Even without changing the learned count rate, making `N` larger **improves** both train and held-out likelihood and rapidly converges to a zero-truncated Poisson model. The observations therefore have very little distributional leverage for distinguishing a finite `N=250` Binomial from a rate model at the fitted probabilities.

This does **not** prove that the fully refitted maximum-likelihood estimate is mathematically infinite. The independently-refit profile remains a confirmatory experiment because calibration and mixture weights can change with `N`.

However, the burden of proof has reversed: there is no scientific basis left for describing the saved `N=250` as an empirically learned number of exposed customers.

## What changes in our model interpretation

Before this experiment, one might narrate:

> expose N virtual customers, each buys with probability q, producing Binomial demand.

That interpretation is no longer defensible from this dataset.

The safer description is:

> the LLM/calibration stack produces a latent positive-demand **rate**; the current Binomial `N,q` parameterization is one convenient representation of that rate, but `N` itself is not identified as real exposure.

## Consequence for Allegory

For future customer datasets with actual impression/traffic/exposure logs, an exposure model can be reintroduced and tested directly.

For this H&M benchmark, subsequent model work should compare `llm-mix-cal` against count-rate families that do not pretend unobserved exposure is known or identified, especially:

- zero-truncated Poisson;
- zero-truncated Negative Binomial;
- hierarchical/overdispersed count models.

## Next experiment

Proceed to the **persona-necessity / population-weight diagnostic** using the existing cached 50-persona responses. The immediate question is whether the fitted 50-persona population is doing scientifically meaningful work or whether a much smaller/pooled representation preserves the predictive signal.
