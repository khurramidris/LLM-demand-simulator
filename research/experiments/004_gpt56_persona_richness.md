# Experiment 004 — GPT-5.6 Sol × Persona Richness

Status: **PREREGISTERED BEFORE GPT SCORING AND BEFORE ROW-LEVEL HELDOUT OUTCOMES ARE INSPECTED IN THIS EXPERIMENT**

Date: 2026-08-14

## Scientific question

Can a modern LLM behavioral scorer (GPT-5.6 Sol) extract product/customer information that improves held-out prediction of the H&M positive-demand count benchmark beyond the one-parameter global logarithmic-series model established in Experiment 003?

A second question is whether **richer evidence-grounded customer representation** improves prediction relative to a matched coarse representation of the same customers.

This experiment is intentionally narrower than an Allegory validity claim. The existing H&M benchmark contains only positive demand rows and therefore cannot establish incidence, exposure, availability, causal price response, or realistic customer choice. A win here means incremental predictive information on this benchmark, not proof of a virtual population.

## Frozen reference benchmarks

From Experiment 003, evaluated on the original ten product-group splits:

- historical paper `llm-mix-cal`: mean heldout NLL ≈ 1.834602486
- one-parameter global log-series: mean heldout NLL ≈ 1.491200377
- global log-series exact CRPS ≈ 0.883543158
- prior log-series + pooled GPT-5-mini-era LLM feature: mean NLL ≈ 1.490515006; its paired CI versus global crossed zero

The target is therefore **not merely to beat the paper model**. The primary target is the one-parameter global log-series model.

## Blinding rule

Before all GPT-5.6 Sol behavioral scores are frozen:

1. The scorer may see product metadata, offered prices, and customer-state inputs.
2. The scorer must not see row-level `demand`, sales counts, heldout targets, per-product demand totals, product popularity ranks, model residuals, or split-specific evaluation results that reveal target mapping.
3. Product selection is inherited from the paper and is acknowledged to have been constructed using full outcomes; that benchmark-level leakage cannot be repaired inside this experiment.
4. The downstream evaluator may read targets only after the frozen score artifact exists.
5. No prompt, feature, score, sign, or transformation may be changed after inspecting treatment test metrics. Any such change is a new exploratory experiment.

## Model provenance

Behavioral scorer: **GPT-5.6 Sol**, used conversation-mediated on 2026-08-14 rather than through paid recursive API calls.

This is not equivalent to thousands of independent API samples. The experiment therefore treats GPT-5.6 Sol as a deterministic/semideterministic semantic scoring engine whose outputs are frozen and then evaluated. Downstream evaluation is reproducible from the frozen scores; the chat-mediated inference itself has no pinned API temperature, seed, or snapshot identifier beyond the product model identity reported by the assistant.

## Arms

### G — Generic product/customer-free scorer

Input: product metadata + offered price only.

Purpose: asks whether GPT product semantics/price reasoning alone contains useful cross-product information.

### P — Paper-style population scorer

Input: the original top 50 H&M persona cells used by the paper, with fixed empirical population weights (`n_customers`). No learned free persona mixture is allowed, because Experiment 002 found that the free 50-weight mixture did not earn its complexity.

Purpose: modern-model reproduction/upgrade of the paper's representation concept.

### C — Matched coarse customer scorer

Construct a fixed, deterministic sample of 50 real customer-history records from the committed `customer_features.csv.zip`. For each sampled customer expose only a coarse version of the evidence:

- age bin: 16–24 / 25–34 / 35–44 / 45–54 / 55+
- engagement bin derived from transaction frequency
- price tier derived from mean paid price
- top product type

The same 50 customers are used in arm R.

Purpose: matched control for representation granularity.

### R — Matched rich customer scorer

Use the same 50 sampled customer-history records as arm C, but preserve the available continuous evidence rather than binning it:

- exact observed age where present
- exact transaction count over the preprocessing window and derived purchases/month
- exact mean paid price
- exact top product type

No invented income, personality, household structure, motivations, brand attitudes, demographics, psychographics, or latent traits are permitted. The committed preprocessing only supports these behavioral/demographic fields; fabricating more would invalidate the test.

This is therefore a **richer representation of the available evidence**, not a claim that H&M supports a full customer digital twin.

## Matched customer sampling

The 50 customer records must be selected deterministically before targets are inspected. The sampling script will:

- exclude customers with zero transactions;
- require non-missing age and positive mean price;
- stratify across age × engagement × price tier as far as available;
- use a fixed seed (`20260814`) only for tie-breaking/sampling within strata;
- emit customer IDs only as irreversible SHA-256 prefixes in the research artifact, not raw customer IDs;
- create both coarse and rich state text from the identical records.

No target-demand information is used in sampling.

## GPT scoring protocol

The GPT scorer does **not** output literal market purchase probabilities. Experiments 001–003 showed that absolute probability/exposure interpretations are not identified on the current benchmark.

Instead GPT outputs a relative behavioral score for each product/price under each arm. Higher score must mean greater expected positive-demand propensity. The downstream model may calibrate level and scale on training data but may not reverse the score.

### Frozen score scale

Each score is on a fixed 0–100 ordinal/interval-like scale:

- 0: essentially no expected appeal/value for the represented population
- 50: ordinary/typical appeal/value
- 100: exceptionally strong expected appeal/value

For population arms, GPT reasons over the provided fixed population representation and returns the aggregate score. It must not use or infer target sales counts.

For each article, all of its offered prices are scored together so that price ordering is visible and internally consistent. GPT may assign non-monotone scores if product/value reasoning justifies it, but downstream fitting cannot learn a sign reversal.

## Downstream observation model

Primary observation model: logarithmic-series distribution, matching the limiting count family identified in Experiment 003.

For each split and each arm, training rows only are used to fit:

`logit(p_i) = a + b * z_i`

where:

- `z_i` is the arm's frozen GPT score standardized using training rows only;
- `a` is a free intercept;
- `b >= 0` is a nonnegative score coefficient.

The global one-parameter baseline is the nested model `b = 0`.

No article embedding, date, article fixed effect, target-derived popularity, or extra price feature is added downstream. Price can influence predictions only through the blinded GPT score supplied to the arm. This keeps the test focused on whether the scorer contains predictive signal beyond the marginal count law.

No regularization is planned because the treatment adds only one slope parameter. Optimization and metric calculations must be numerically checked against the Experiment 003 implementation.

## Splits

Use the exact ten original product-group heldout splits used by Experiments 001–003. All rows belonging to heldout articles remain held out for the downstream calibration fit.

The same frozen GPT scores are used across splits. No split-specific GPT prompting or refitting of customer states is allowed.

## Primary endpoints and decision rules

Primary endpoint: heldout zero-truncated/log-series **negative log likelihood (NLL)** averaged across the ten original splits.

### Claim A — GPT-rich beats the one-parameter statistical baseline

Supported only if paired per-split:

`NLL(R) - NLL(Global)`

has a two-sided 95% confidence interval entirely below 0.

### Claim B — richer customer grounding beats matched coarse grounding

Supported only if paired per-split:

`NLL(R) - NLL(C)`

has a two-sided 95% confidence interval entirely below 0.

### Claim C — rich grounding beats paper-style personas

Supported only if paired per-split:

`NLL(R) - NLL(P)`

has a two-sided 95% confidence interval entirely below 0.

### Secondary endpoints

Report exact CRPS, MAE, RMSE, and PIT KS using the same aggregation convention where possible. Secondary endpoints are descriptive unless their paired CIs are also reported.

## Mandatory ablations / diagnostics

Report:

- global baseline vs G, P, C, R;
- fitted slope `b` by split for every arm;
- fraction of splits with `b` effectively zero;
- score variance overall and by article;
- pairwise correlations among G/P/C/R scores;
- price monotonicity rate within article;
- whether customer-conditioned arms materially differ from generic scoring;
- per-split paired differences and 95% t-intervals;
- bootstrap-over-splits sensitivity if practical, clearly labeled as low-n sensitivity rather than a magic fix for ten splits.

If an arm wins only because its training calibration coefficient collapses to zero on some splits, that is not evidence for its score signal on those splits.

## Interpretation matrix

- **R beats Global and R beats C:** evidence that richer available customer grounding adds heldout information on the positive-count H&M benchmark.
- **R beats Global but not C:** evidence for GPT/customer-conditioned signal, but not for richer representation specifically.
- **G beats Global while C/R do not beat G:** product-semantic GPT signal, not persona evidence.
- **C/R beat Global but R does not beat C:** customer conditioning may help, richness has not earned itself.
- **No arm beats Global:** modern GPT/customer detail still does not earn a role on this benchmark.
- **Any win against paper but not Global:** paper-upgrade result only; not scientifically interesting enough for Allegory core claims.

## Known benchmark limitations retained

This experiment deliberately reuses the paper benchmark for comparability and therefore inherits severe limitations established earlier:

- no explicit zero-demand rows;
- product universe selected using full-period demand;
- product price grids/support can include information from eventual heldout history;
- availability/stockouts/exposure are unobserved;
- observational prices are confounded and are not causal interventions;
- all products are trousers;
- product-history/customer-history construction is not a strict prospective deployment simulation.

Consequently, even a decisive win is labeled only as **incremental predictive information on the paper's positive-count benchmark**. A stronger zero-inclusive temporal/cold-product benchmark is required before an Allegory customer-simulation claim.

## Stop / contamination rules

- Scores are frozen before evaluation.
- If target mapping is accidentally exposed to the scorer before score freeze, Experiment 004 is contaminated and must be restarted with a new untouched holdout construction.
- No cherry-picked subset may replace the preregistered ten-split primary analysis.
- Failed or null results are retained and reported.
- `main` remains untouched; this experiment lives on `research/04-gpt56-persona-richness` and should remain a draft PR unless results justify further action.
