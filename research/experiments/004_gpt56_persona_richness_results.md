# Experiment 004 — GPT-5.6 Sol × Persona Richness — Results

Status: **COMPLETE — PRIMARY CLAIMS NOT SUPPORTED**

Date: 2026-08-14  
Branch: `research/04-gpt56-persona-richness`  
Draft PR: #5  
Primary evaluation workflow run: **31790190199**  
Result artifact: `exp004-gpt56-persona-richness-results` (artifact ID **9215158259**)  
Pre-unblinding frozen-score SHA-256: `0d5a5ce41db0a546f8a535e0c21df6519a08df1ad0558c92e29299beaea72a73`

## Executive result

The proposed hypothesis did **not** pass the preregistered test.

A GPT-5.6-Sol-authored target-blind semantic/customer scorer was evaluated on the exact ten original H&M positive-demand product splits using the one-parameter logarithmic-series model from Experiment 003 as the nested baseline.

The global baseline reproduced exactly: mean heldout NLL **1.4912003773**.

Heldout mean NLL:

| Model | Mean NLL | Delta vs global | Relative delta vs global |
|---|---:|---:|---:|
| Generic GPT-authored score (G) | **1.4910213983** | -0.0001789790 | -0.0120% |
| Global one-parameter log-series | **1.4912003773** | 0 | 0 |
| Rich matched customer state (R) | 1.4913325064 | +0.0001321291 | +0.0089% |
| Coarse matched customer state (C) | 1.4914063702 | +0.0002059929 | +0.0138% |
| Paper-style population state (P) | 1.4915158841 | +0.0003155068 | +0.0212% |
| Historical paper `llm-mix-cal` reference | 1.8346024862 | +0.3434021089 | +23.03% |

Thus the best point estimate among the treatment arms was **generic product/price scoring**, not a persona-conditioned arm. Its apparent 0.012% NLL improvement over the one-parameter baseline was not statistically established across the ten splits.

## Preregistered claims

### Claim A — rich customer grounding beats the global one-parameter model

**NOT SUPPORTED.**

Paired split NLL delta `R - Global`:

- mean: **+0.0001321291**
- 95% t interval: **[-0.0007278913, +0.0009921494]**
- split bootstrap 95% interval: **[-0.0004740827, +0.0009179391]**
- negative/better in 4/10 splits; exactly tied in 3/10 at the fitted boundary; worse in 3/10.

The rich arm is slightly worse in mean heldout NLL, and both uncertainty intervals cross zero.

### Claim B — rich grounding beats the matched coarse representation

**NOT SUPPORTED under the preregistered primary rule.**

Paired split NLL delta `R - C`:

- mean: **-0.0000738638**
- relative mean advantage over C: about **0.00495%**
- 95% t interval: **[-0.0001715116, +0.0000237840]**
- split bootstrap 95% interval: **[-0.0001600010, -0.0000022656]**

The point estimate consistently favors richer representation enough for the simple split bootstrap interval to fall just below zero, but the **preregistered decision rule was the paired 95% t interval**, which crosses zero. With only ten product splits and a microscopic effect, this is not promoted to a positive claim. The bootstrap result is retained as a sensitivity signal worth testing on a stronger/new benchmark.

### Claim C — rich grounding beats the paper-style persona population

**NOT SUPPORTED under the preregistered primary rule.**

Paired split NLL delta `R - P`:

- mean: **-0.0001833777**
- 95% t interval: **[-0.0003796589, +0.0000129035]**
- split bootstrap 95% interval: **[-0.0003537249, -0.0000378574]**

Again, the point estimate and bootstrap favor R, but the preregistered paired t interval narrowly crosses zero. We do not convert this into a win after seeing the result.

## Full heldout metric means

| Model | NLL | exact CRPS | MAE | RMSE | mid-P PIT KS |
|---|---:|---:|---:|---:|---:|
| Global | 1.4912003773 | **0.8835431576** | 1.3693105172 | 2.3855399639 | 0.2775286904 |
| G | **1.4910213983** | 0.8838911196 | **1.3680856938** | **2.3854601270** | **0.2629093538** |
| P | 1.4915158841 | 0.8843337979 | 1.3699012831 | 2.3863619966 | 0.2657539321 |
| C | 1.4914063702 | 0.8842649253 | 1.3691817632 | 2.3861818708 | 0.2645269891 |
| R | 1.4913325064 | 0.8842092069 | 1.3688885392 | 2.3860692558 | 0.2640516906 |

Important: Global remains best on exact CRPS. Generic GPT scoring has the best point NLL/MAE/RMSE among treatment models, but its NLL superiority is not statistically established.

## Generic scorer versus global

Paired split NLL delta `G - Global`:

- mean: **-0.0001789790**
- 95% t interval: **[-0.0007477526, +0.0003897947]**
- split bootstrap 95% interval: **[-0.0006223788, +0.0003114064]**
- better in 6/10 splits, tied in one, worse in three.

This is at most a weak hint that target-blind product-semantic scoring may contain a tiny amount of heldout positive-count signal. It is **not a demonstrated win**.

## Training-versus-heldout behavior

Every score family reduced training NLL relative to the nested global model:

- G train delta: **-0.0005963440**
- P train delta: **-0.0005255794**
- C train delta: **-0.0005699496**
- R train delta: **-0.0005721236**

But the heldout deltas were:

- G: **-0.0001789790**
- P: **+0.0003155068**
- C: **+0.0002059929**
- R: **+0.0001321291**

So the main failure is generalization, not inability of the training optimizer to use the score.

One split (`split_003`) contributes the largest treatment loss: heldout deltas versus global are approximately +0.001318 (G), +0.003599 (P), +0.003507 (C), and +0.003145 (R), despite that same split showing the strongest training slopes/improvements. This is exactly the sort of train/test reversal the product-level holdout is intended to reveal.

## Fitted score slopes

Mean / median fitted nonnegative slope on standardized training score:

| Arm | Mean b | Median b | Fraction effectively zero |
|---|---:|---:|---:|
| G | 0.058830 | 0.058632 | 20% |
| P | 0.048698 | 0.034097 | 30% |
| C | 0.052722 | 0.043682 | 30% |
| R | 0.053865 | 0.047171 | 30% |

Thus the treatment did not merely collapse to the global model everywhere. The training data usually assigned a positive coefficient to the scores, but that coefficient did not yield robust heldout improvement.

## Score similarity / richness diagnostic

Before outcomes were unblinded, the frozen product-level appeal scores already showed:

- `corr(R, C) = 0.994698`
- mean absolute R–C appeal difference ≈ **0.59085 points** on the 0–100 scale
- max absolute difference ≈ **1.04456 points**
- `corr(P, C) = 0.987020`
- `corr(P, R) = 0.970569`
- `corr(G, R) = 0.927313`

All four arms were mechanically downward-monotone in offered price for **100% of articles**, as frozen before target inspection.

This gives a plausible scientific interpretation of the null richness result: the actual committed H&M customer-history features are too weak/coarse to make the rich and coarse population representations behave very differently. Exact age, transaction rate and mean paid price changed the score only modestly compared with their binned counterparts.

## A subtle but important population issue

The original paper's top-50 persona population is not representative of the matched customer-history population in favorite product type: about **48%** of the paper persona weight has `Trousers` as the favorite category, versus only **6/50 = 12%** of the deterministic matched customer sample.

Because all 100 evaluated products are trousers, directly rewarding a population for having `Trousers` as its favorite category would give the paper-style arm a large mostly constant level advantage. The downstream intercept would absorb much of that level anyway, and using it as a product-specific signal would be scientifically suspect. The frozen scorer therefore did not manufacture a large trousers-favorite bonus.

## What Experiment 004 establishes

### What we can say

1. A **GPT-5.6-Sol-authored deterministic semantic/customer scorer did not robustly beat** the one-parameter log-series model on the original H&M positive-demand benchmark.
2. Rich matched customer states had a slightly better point estimate than their matched coarse counterparts, but the preregistered primary CI crossed zero.
3. Generic product-semantic scoring had the best heldout NLL point estimate among GPT-authored arms, suggesting that whatever tiny signal exists here may be more product-semantic than persona-driven.
4. The modern scorer plus the stronger log-series observation model trivially beats the historical paper's 1.8346 NLL, but **that is not evidence for GPT/personas**, because the one-parameter no-LLM log-series already achieves 1.4912.
5. Experiment 003's conclusion survives: this benchmark is dominated by the positive-count observation law and is too weak to validate a virtual-population thesis.

### What we cannot say

We cannot claim from this result that:

- richer personas improve demand prediction;
- GPT-5.6 beats the one-parameter statistical model;
- H&M customer digital twins are validated;
- the LLM predicts purchase incidence;
- the model captures causal price elasticity;
- the model predicts inventory-aware demand or customer choice;
- the result is a native GPT-5.6 API benchmark.

## Important model-provenance limitation

The user asked to use the assistant itself rather than buy API calls. The ChatGPT harness cannot recursively create hundreds/thousands of independent GPT-5.6 calls, so this experiment used a **GPT-5.6-Sol-authored deterministic target-blind scoring implementation**. The implementation, protocol, and score hash were frozen before outcome inspection and are reproducible, but this is not identical to a native API experiment with independent model generations.

Therefore a future API run could still answer a different question: whether native free-form GPT-5.6 responses extract signal that the deterministic semantic scorer missed. Experiment 004 does not prejudge that result.

## Scientific decision

**DO NOT PROMOTE richer H&M personas or GPT-5.6 semantic scoring as an Allegory core improvement from this experiment.**

Keep Experiment 004 as a negative/null research result.

The next high-value experiment should not tune these 100 positive-count rows. It should construct or select a benchmark where the system must predict meaningful behavioral variation: zero versus purchase, choice among products, temporally future demand, genuine cold products, observed exposure/availability, or an intervention. Only there can richer customer grounding earn its complexity.

`main` remains untouched. PR #5 should remain draft/unmerged.
