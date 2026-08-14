# Experiment 003 — Count-rate baseline ladder: result

## Decision

This experiment materially changes what the H&M benchmark can support.

### 1. **DOWNGRADE LLM behavioral elicitation as a core mechanism on this H&M benchmark.**

A pooled LLM count-rate feature does **not** materially outperform a model with no LLM, no product features and no price features.

### 2. **KILL the claim that this benchmark demonstrates a need for a virtual population.**

The strongest result is explained almost entirely by the distribution of the already-positive demand counts. A one-parameter logarithmic-series model is essentially as good as the best LLM model.

### 3. **KILL finite latent exposure and finite NB-dispersion semantics.**

Experiment 001 already showed that Binomial exposure `N` is not identified. Experiment 003 now shows that, under positive-only evaluation, Negative Binomial dispersion also often seeks its zero-dispersion boundary and is practically matched by its logarithmic-series limit. Do not interpret either latent parameter as a measured market quantity.

### 4. **DO NOT promote SigLIP/product/price complexity on this benchmark.**

The global count model beats the richer price and SigLIP+price count models on mean held-out NLL/CRPS. The current benchmark therefore does not justify those features either.

### 5. **KEEP the paper model only as a historical reproduction target.**

The paper result remains useful for reproducing the authors' experiment. It is no longer our preferred scientific baseline.

---

## Provenance

- Branch: `research/03-count-rate-baselines`
- Pull request: #4
- Main 10-split workflow run: `31786378878`
- NB zero-limit workflow run: `31787221144`
- Exact original product splits: 10/10
- New LLM calls: **0**
- API cost: **$0**
- `main`: untouched

CI validated:

- exact committed paper baseline metrics before running treatments;
- syntax;
- finite-difference agreement for the analytic zero-truncated Poisson and Negative Binomial gradients.

The richer product models use train-only PCA/scaling/spline fitting and train-only group cross-validation for regularization.

---

# Part A — Count-rate ladder

## Held-out means across the original 10 product splits

| Model | Test zt NLL ↓ | Test zt CRPS ↓ | MAE ↓ | RMSE ↓ | PIT KS ↓ |
|---|---:|---:|---:|---:|---:|
| **ZT-NB pooled LLM** | **1.490694** | 0.883865 | **1.368688** | 1.777270 | 0.244531 |
| **ZT-NB global** | 1.491357 | **0.883598** | 1.369296 | **1.775903** | **0.244284** |
| ZT-NB price spline | 1.496713 | 0.885846 | 1.371760 | 1.778604 | 0.247757 |
| ZT-NB SigLIP + price | 1.501342 | 0.889136 | 1.379621 | 1.786597 | 0.250524 |
| ZT-Poisson pooled LLM | 1.818818 | 0.933005 | 1.368566 | 1.777196 | 0.300422 |
| ZT-Poisson global | 1.820734 | 0.932064 | 1.369311 | 1.775909 | 0.303108 |
| ZT-Poisson price spline | 1.831875 | 0.936503 | 1.373746 | 1.780411 | 0.305392 |
| Paper `llm-mix-cal` | 1.834602 | 0.942639 | 1.380486 | 1.789605 | 0.302080 |
| ZT-Poisson SigLIP + price | 1.840778 | 0.943536 | 1.381490 | 1.787905 | 0.308393 |

The single most important result is not a small model ranking. It is the enormous jump from Poisson/Binomial-like dispersion to a heavy-tailed positive-count family.

## Negative Binomial versus matched Poisson

Across **all four matched feature sets**, Negative Binomial beats Poisson on held-out NLL and CRPS in **10/10 splits**.

### Global mean model

- NLL delta, NB minus Poisson: **-0.329376**
- 95% paired interval: **[-0.366201, -0.292552]**
- better: 10/10 splits
- CRPS delta: **-0.048466**
- 95% paired interval: **[-0.058607, -0.038325]**
- better: 10/10 splits

### Price-spline model

- NLL delta: **-0.335162**
- 95% paired interval: **[-0.367372, -0.302952]**
- CRPS delta: **-0.050658**
- 95% paired interval: **[-0.063349, -0.037966]**

### SigLIP + price model

- NLL delta: **-0.339436**
- 95% paired interval: **[-0.373552, -0.305321]**
- CRPS delta: **-0.054400**
- 95% paired interval: **[-0.065401, -0.043399]**

### Pooled-LLM model

- NLL delta: **-0.328124**
- 95% paired interval: **[-0.365281, -0.290967]**
- CRPS delta: **-0.049140**
- 95% paired interval: **[-0.058636, -0.039644]**

This is far larger than any LLM/product/persona effect in the experiment.

---

# Part B — Does the LLM earn its role after fixing the observation family?

The best LLM rate model is `ztnb-llm-pooled`.

The best non-LLM rate model is the **global** zero-truncated Negative Binomial: one common count distribution for every row.

## Direct comparison

- pooled LLM NB NLL: `1.490694`
- global NB NLL: `1.491357`
- nominal LLM advantage: only **0.044%**
- paired LLM-minus-global NLL interval: **[-0.001642, +0.000317]**

CRPS:

- pooled LLM NB: `0.883865`
- global NB: `0.883598`
- LLM is nominally **0.030% worse**
- paired interval crosses zero

Therefore the preregistered decision is:

**`DOWNGRADE_CORE_ROLE`**.

The data do not establish a meaningful predictive contribution from the cached LLM behavioral signal once the positive-count distribution is modeled appropriately.

---

# Part C — The NB zero-dispersion / logarithmic-series limit

Forensic inspection of the fitted NB models showed extremely small size/dispersion parameters. Several fits hit the original lower bound `r = exp(-6) ≈ 0.00248`.

This matters because, after conditioning a Negative Binomial on `Y>0`, the limit as its size parameter approaches zero is a **logarithmic-series distribution**. In that limit the model can describe the shape of positive counts while the implied untruncated incidence rate tends toward a quantity that is not identified by the positive-only sample.

We therefore ran a dedicated limit diagnostic.

## Widening the NB lower bound

Candidate lower log-dispersion bounds:

`[-6, -8, -10, -12, -16]`.

The train likelihood continues to improve very slightly as the bound is relaxed:

| lower log(r) | Mean train NLL | Median fitted r | Fraction at lower bound |
|---:|---:|---:|---:|
| -6 | 1.484511 | 0.002479 | 0.6 |
| -8 | 1.484504 | 0.000335 | 0.6 |
| -10 | 1.484503 | 0.0000454 | 0.6 |
| -12 | 1.484503 | 0.0000117 | 0.2 |
| -16 | 1.484503 | 0.00000270 | 0.1 |

The held-out difference between the `-6` and `-16` fits is effectively zero:

- mean NLL change: `+0.0000026`
- 95% paired interval: `[-0.0000078, +0.0000130]`

So there is no practical held-out evidence for a particular finite dispersion value in this regime.

## Explicit logarithmic-series limit

| Model | Test NLL ↓ | Test CRPS ↓ | MAE ↓ | Mean positive-count prediction |
|---|---:|---:|---:|---:|
| **Global logarithmic-series** | 1.491200 | **0.883543** | 1.369311 | 2.20258 |
| Logarithmic-series + pooled LLM | **1.490515** | 0.883802 | **1.368693** | 2.19732 |

The pooled LLM-minus-global differences are again tiny:

### NLL

- mean delta: `-0.000685`
- 95% paired interval: **[-0.001668, +0.000297]**
- interval crosses zero

### CRPS

- mean delta: `+0.000259`
- 95% paired interval: **[-0.001026, +0.001544]**
- interval crosses zero

### MAE

- mean delta: `-0.000617`
- 95% paired interval: **[-0.006519, +0.005285]**
- interval crosses zero

Thus even in the explicit positive-count limit, the LLM feature has not earned its complexity.

---

# Part D — Why the global model is so strong

The design matrix contains **11,691 positive-demand rows and no zero-demand rows**.

Observed positive-count distribution:

- mean: **2.204**
- median: **1**
- demand = 1: **6,214 rows (53.15%)**
- demand ≤ 2: **73.70%**
- demand ≤ 3: **84.64%**
- demand ≤ 4: **90.50%**
- demand ≤ 5: **93.97%**

The benchmark therefore rewards a model that predicts the marginal shape of low positive counts extremely well. It does not require a model to decide whether a product gets **zero demand**, whether it was actually available, how many shoppers saw it, or whether a customer chose it over alternatives.

That explains why a global positive-count distribution can match or beat models containing product images, text, prices, personas and LLM outputs.

---

# Part E — Comparison to the paper control

Paper `llm-mix-cal` held-out NLL: `1.834602`.

Global logarithmic-series NLL: `1.491200`.

That is about an **18.7% reduction in NLL** using one fitted distributional parameter and no LLM/product/persona signal.

Paper saved CRPS is `0.942639`; global logarithmic-series exact CRPS is `0.883543`, about 6.3% lower. The paper CRPS artifact was computed by its existing Monte Carlo procedure whereas the new count models use deterministic support summation, so NLL is the cleaner apples-to-apples headline comparison. The gap is nevertheless much larger than plausible Monte Carlo noise.

The constant/global model also slightly improves MAE over the paper control, despite having no row-specific behavioral prediction.

---

# What this proves

## Supported

1. The observation/count family dominates model performance on the paper's positive-only benchmark.
2. A global positive-count distribution can match the best LLM rate model.
3. The cached LLM behavioral feature provides no statistically established held-out advantage after the observation family is corrected.
4. Product embeddings and nonlinear price do not improve over the global positive-count model in mean held-out performance.
5. The benchmark is mostly measuring the shape of positive counts, not market-demand incidence.
6. Neither latent Binomial exposure nor finite NB dispersion should be given a literal market interpretation from this dataset/protocol.

## Not proved

1. LLM behavioral simulation is useless in general.
2. Customer heterogeneity is useless in real enterprise datasets.
3. Product semantics or price never matter for demand.
4. A logarithmic-series model is a realistic generative retail-demand model.
5. Negative Binomial or log-series models solve the missing-zero, availability, exposure or causal-price problems.

The experiment instead shows that **this benchmark is too weak to adjudicate those questions**.

---

# Allegory consequence

This is an important negative result for Allegory.

We should **not** use the current H&M positive-only benchmark as evidence that Allegory can simulate populations or customer behavior. Doing so would be scientifically indefensible after this experiment.

H&M can remain:

- a reproduction demo;
- a pipeline engineering dataset;
- a historical benchmark showing why stronger validation matters.

But it should not be our flagship proof of virtual-population validity in its current form.

## What to do next

Before spending another LLM API dollar on H&M:

1. investigate whether the raw H&M transaction timeline can support a defensible product-day panel including zero-sale days;
2. distinguish zero sales from likely unavailability/stockout where possible;
3. move to strict temporal and true cold-product holdouts;
4. remove test-derived price-grid information;
5. test whether product/LLM/persona signals reappear when the task actually requires predicting incidence and cross-product variation;
6. if availability/exposure cannot be reconstructed credibly, stop using H&M for scientific validation and move to a dataset with observed impressions, choice sets, interventions, randomized prices/messages, or other real exposure information.

Only after that harder benchmark exists should richer customer states/personas receive new LLM calls.
