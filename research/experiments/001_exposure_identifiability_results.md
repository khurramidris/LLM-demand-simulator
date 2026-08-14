# Experiment 001 — Exposure-N identifiability: result

## Decision

**KILL the interpretation of fitted `N=250` as literal customer exposure.**

**KEEP `llm-mix-cal` as a predictive benchmark.** This experiment attacks the semantics/identifiability of the latent exposure parameter, not the existence of held-out predictive signal.

The result is now supported by **two independent diagnostics**:

1. a 10-split fixed-rate path toward the Poisson limit;
2. a 3-split profile that independently refits the full `llm-mix-cal` calibration and persona mixture at every candidate `N`.

Both point in the same direction.

## Run provenance

### Fast Poisson-limit diagnostic

- Branch: `research/01-exposure-identifiability`
- Pull request: #2
- GitHub Actions run: `31784171980`
- Saved product splits evaluated: 10/10

### Independently-refit profile

- GitHub Actions run: `31783567962`
- Product splits evaluated: 3
- Candidate grid: `[100, 150, 200, 250, 350, 500, 750, 1000]`
- Full calibration and persona mixture refit independently for every admissible `N`

For both experiments:

- cached LLM calls used: existing committed responses only
- new LLM calls: **0**
- API cost: **$0**

The workflows verified that the committed paper-style baseline still reports:

- LLM-cal test CRPS: `0.9426390618243496`
- LLM-cal test MAE: `1.38048553572638`

# Part A — Fixed-rate path toward the Poisson limit

For every saved `llm-mix-cal` split model, start from its fitted `N=250` purchase probabilities `q_250` and define the fitted count rate:

`lambda = 250 * q_250`.

Then move along a constant-rate path:

`q_N = lambda / N`

for progressively larger values of `N`, without refitting mixture weights or calibration. This isolates how much the zero-truncated distribution itself can distinguish finite Binomial exposure from the Poisson limit when the predicted count rate is preserved.

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

## Conditional moments

At saved `N=250`, the average absolute difference from the zero-truncated Poisson limit is already tiny on test rows:

- conditional mean difference: `0.002777` demand units
- conditional variance difference: `0.012260` demand-units squared

At `N=10,000` these shrink to approximately:

- conditional mean difference: `0.000069`
- conditional variance difference: `0.000307`

So the fitted finite-Binomial distribution is already operating close to a count-rate/Poisson regime.

# Part B — Independently-refit profile

The stronger confirmatory test allows the model to re-estimate its calibration and free persona-mixture weights from scratch at each candidate `N`.

If `N=250` were merely disadvantaged by holding `lambda` fixed, this refit could in principle recover an interior optimum.

It did not.

## Result

| Split | Train-optimal N | Test-optimal N (diagnostic only) | Train NLL change: N=1000 minus N=250 | Test NLL change: N=1000 minus N=250 |
|---:|---:|---:|---:|---:|
| 0 | **1000** | **1000** | -0.007272 | -0.005087 |
| 1 | **1000** | **1000** | -0.006485 | -0.006226 |
| 2 | **1000** | **1000** | -0.008160 | -0.004035 |

In **3/3 splits**, the fully refitted training objective chooses the **new upper boundary `N=1000`**.

The held-out diagnostic also prefers `N=1000` in 3/3 splits.

This is the key confirmatory result: the boundary-seeking behavior survives independent re-estimation of the calibration and mixture weights.

## Interpretation

The evidence is now strong enough that we should stop treating `N` as an empirically learned number of exposed customers in this H&M benchmark.

The original saved fits selected `N=250` in 10/10 splits because `250` was the largest available candidate. When we expand the search to `1000` and independently refit the model, the optimum simply moves to `1000` in every tested split.

Together with the Poisson-limit diagnostic, the scientifically safer interpretation is:

> the LLM/calibration stack is producing a latent positive-demand **count rate**. The current Binomial `(N, q)` factorization is a convenient parameterization of that rate, but this dataset does not identify `N` as real customer exposure.

We still have not proved that the mathematical MLE is literally `N=infinity`; an even wider full-refit grid could keep moving. But that distinction no longer matters for the original exposure claim. A parameter whose optimum repeatedly follows the experimenter's upper bound should not be given a real-world exposure interpretation.

## What changes in our model interpretation

Do not narrate the current H&M model as:

> expose 250 virtual customers and independently sample whether each buys.

Instead:

> aggregate calibrated LLM behavioral features into a latent demand-rate signal, then place an observational count distribution around that signal.

If future customer datasets contain actual impression, traffic, availability, or exposure logs, then explicit exposure can be reintroduced and evaluated against observed exposure.

## Consequence for Allegory

For this H&M benchmark, subsequent model work should compare against count-rate families that do not pretend unobserved exposure is identified:

- zero-truncated Poisson;
- zero-truncated Negative Binomial;
- hierarchical/overdispersed count models.

This also means that future Allegory interfaces must distinguish a **modeled demand rate** from a **measured reachable population/exposure count**.

## Next experiment

Experiment 002 tests whether the freely fitted 50-persona population is scientifically necessary or whether a much lower-capacity fixed/pooled aggregation preserves the held-out signal.
