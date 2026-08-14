# Experiment 002 — Persona necessity / population weighting: result

## Decision

**KILL the current free 50-persona mixture as the preferred population aggregation mechanism.**

**KEEP the cached persona-conditioned LLM response representation for now.** This experiment does not show that persona prompting itself is unnecessary; it shows that freely learning ~50 population mixture weights is unnecessary on this benchmark.

**DO NOT claim the simpler models are statistically superior on NLL.** Their mean NLL is better, but the paired 95% intervals narrowly include zero. The defensible result is strong non-inferiority with dramatically lower capacity, plus an exploratory MAE improvement.

## Run provenance

- Branch: `research/02-persona-necessity`
- Pull request: #3
- Final GitHub Actions run: `31784857581`
- Final job: `persona-necessity`
- Original deterministic product splits: 10/10
- New LLM calls: **0**
- API cost: **$0**

The workflow verified the committed control before the experiment:

- LLM-cal test zero-truncated NLL: `1.8346024861697727`
- LLM-cal test MAE: `1.38048553572638`

## Models compared

### Control — `control_free_50_persona`

Saved paper-style `llm-mix-cal` model:

- 50 persona response features
- free persona mixture weights
- dummy no-buy mass
- monotone logit calibration
- saved `N`

Approximate fitted capacity relevant to aggregation/calibration: ~52 parameters.

### Treatment — `pooled_raw_mean`

Average the 50 raw persona probabilities first, then fit only:

1. calibration intercept;
2. positive calibration slope;
3. total real-population mass.

### Treatment — `uniform_fixed_population`

Calibrate each persona response identically, then aggregate all 50 with equal fixed weights. Only the same three global scalar parameters are fitted.

### Treatment — `empirical_fixed_population`

Calibrate each persona response identically, then aggregate using the observed H&M customer-cell frequencies for P001–P050. Only the same three global scalar parameters are fitted.

`N` was held at the saved control value so population aggregation was the major changed scientific variable.

## Main held-out result

| Model | Test zt NLL | Delta vs free-50 | Relative delta | Test MAE | MAE delta |
|---|---:|---:|---:|---:|---:|
| Free 50-persona control | 1.834602 | 0 | 0 | 1.380486 | 0 |
| Pooled raw mean | 1.829209 | -0.005393 | -0.294% | 1.372083 | -0.008403 |
| Uniform fixed population | 1.827927 | -0.006676 | -0.364% | 1.370513 | -0.009973 |
| Empirical fixed population | 1.827812 | -0.006790 | -0.370% | 1.370795 | -0.009691 |

All three lower-capacity treatments have slightly better held-out point estimates than the free 50-persona mixture.

## Paired uncertainty across the 10 splits

### Zero-truncated NLL

| Treatment | Mean delta vs control | 95% paired t interval | Splits better than control |
|---|---:|---:|---:|
| Pooled raw mean | -0.005393 | [-0.012737, +0.001950] | 6/10 |
| Uniform fixed population | -0.006676 | [-0.013506, +0.000155] | 8/10 |
| Empirical fixed population | -0.006790 | [-0.013696, +0.000116] | 7/10 |

The intervals narrowly include zero, so **we do not claim NLL superiority at the 95% level**.

However, the experiment preregistered a practical non-inferiority threshold of roughly 1% of control held-out NLL. One percent of `1.834602` is about `0.01835`; the upper confidence bounds for all three treatments are only `0.0001–0.0020`. Therefore the simple models are comfortably non-inferior under that margin.

### MAE

| Treatment | Mean delta vs control | 95% paired t interval | Splits better than control |
|---|---:|---:|---:|
| Pooled raw mean | -0.008403 | [-0.016492, -0.000313] | 7/10 |
| Uniform fixed population | -0.009973 | [-0.017510, -0.002435] | 8/10 |
| Empirical fixed population | -0.009691 | [-0.017313, -0.002069] | 8/10 |

The MAE intervals exclude zero in favor of the lower-capacity treatments. Because these are related exploratory treatments rather than a preregistered multiple-comparison superiority test, we record this as supporting evidence rather than a definitive new headline result.

## Train/test behavior

The free 50-persona model fits training data better:

- control train NLL: `1.795495`
- empirical fixed: `1.802111`
- uniform fixed: `1.802162`
- pooled: `1.802420`

For every simple treatment, the paired train NLL disadvantage has a 95% interval entirely above zero.

Yet on held-out products the ordering reverses in the mean.

This is exactly the pattern expected when the free mixture has enough flexibility to improve in-sample fit without producing a corresponding generalization gain.

## Mixture-collapse diagnostic

Across the saved control fits:

- nominal personas: 50
- mean effective number of personas (inverse-Simpson ESS): **3.52**
- mean number of personas with >=1% normalized fitted weight: **6.0**

So the control is not behaving like a diffuse 50-cell empirical population. It is behaving like a sparse learned basis-function ensemble.

## Empirical weights versus uniform weights

The held-out NLL difference is tiny:

- empirical fixed: `1.827812`
- uniform fixed: `1.827927`

This experiment did not directly test that difference with a paired superiority test, so we do not claim equivalence. But there is currently no evidence that the observed H&M cell frequencies add meaningful predictive value over equal weighting.

## What this proves

### Supported

1. The current benchmark does **not** establish that freely fitted 50-persona mixture weights are necessary.
2. A 3-parameter fixed/pooled aggregation is practically non-inferior on held-out NLL.
3. The free mixture is substantially more flexible in-sample but does not generalize better in the mean.
4. The saved free mixture collapses to only ~3.5 effective personas on average.

### Not proved

1. Persona prompting itself is unnecessary. Every treatment still derives its feature from the 50 persona-conditioned LLM responses.
2. A literal single neutral customer prompt would match the 50-persona aggregate. That requires a new elicitation experiment or a carefully designed cached single-persona test.
3. Empirical population weighting is truly better than uniform weighting.
4. The simple treatment is a publication-level statistically superior replacement; NLL superiority is not established at 95%.

## Allegory consequence

Do **not** respond by making the H&M personas richer.

The present evidence points in the opposite direction: the useful signal appears to live largely in a calibrated aggregate of LLM behavioral responses, while free population reweighting overfits.

For Allegory, customer heterogeneity should be admitted only when it wins a held-out benchmark. A population simulator should not contain dozens of agent/persona degrees of freedom merely because they make the product conceptually attractive.

## Next experiment

The next priority should be a **count-rate baseline ladder**:

1. zero-truncated Poisson;
2. zero-truncated Negative Binomial;
3. nonlinear product-embedding + price count model;
4. stronger temporal/context baselines where the data permit them;
5. compare all against the simplified calibrated LLM aggregate and the preserved paper control.

This follows directly from Experiment 001 (finite `N` is not supported as literal exposure) and Experiment 002 (free 50-persona aggregation is not supported as necessary complexity).
