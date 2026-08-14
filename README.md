# LLM-Powered Virtual Population for Demand Simulation and Pricing

This repository is a fork of the 2026 research code by **Chengpiao Huang and Kaizheng Wang** for *LLM-powered virtual population for demand simulation and pricing*.

The original paper workflow is intentionally preserved. The `research/allegory-v1` branch adds a separate research-hardening layer for leakage-resistant evaluation, mechanism ablations, response auditing, tests, and reproducibility.

> **Attribution / licensing note:** the upstream repository did not include a license file when this fork was created. Do not assume that public source availability grants commercial redistribution rights. The additions in this fork are intended first for research reproduction and independent experimentation.

## What the upstream system does

The baseline workflow uses the public H&M Personalized Fashion Recommendations dataset:

1. Select a product universe (default: 100 high-demand online trousers).
2. Build customer features from historical transactions.
3. Aggregate customers into persona cells based on age, engagement, typical paid price, and product taste.
4. Query an LLM for persona-level purchase probabilities across candidate prices, optionally with the product image.
5. Aggregate and calibrate those probabilities against observed H&M demand.
6. Evaluate held-out product demand distributions.
7. Run a separate synthetic-ground-truth pricing sample-efficiency experiment.

The original code and saved outputs remain the replication control.

## Allegory research additions

### 1. STRICT-GLOBAL-HOLDOUT

`scripts/build_strict_holdout.py` chooses the product train/test split **before persona construction** and removes every held-out-product transaction from the history used to build customer state.

This prevents eventual test products from affecting:

- purchase frequency,
- typical paid price,
- preferred product category,
- persona-cell assignment.

The script writes an auditable protocol manifest with split IDs, hashes, row-removal counts, and paths.

### 2. Fixed-holdout evaluation

`scripts/evaluate_fixed_holdout.py` evaluates one frozen product split rather than generating new random splits during model fitting.

It uses strict response parsing and reports train/test metrics for:

- `llm-mix`
- `llm-mix-cal`
- `population-cal` — **new:** actual H&M persona-cell counts are fixed as population weights
- `uniform-cal` — **new:** all queried personas receive equal weight
- `emb`
- `gaussian`

The two new calibrated baselines isolate whether predictive performance comes from a realistic population composition or from the original model's learned mixture weights.

### 3. LLM response audit

`scripts/audit_llm_responses.py` measures:

- loose JSON parse success,
- strict response validity,
- returned price-grid mismatches,
- duplicate prices,
- invalid probabilities,
- price-monotonicity violations.

The strict parser rejects malformed outputs instead of silently clipping invalid probabilities.

### 4. Research diagnostics

`demand_sim/research.py` provides deterministic split creation, transaction filtering, file hashing, mixture-concentration diagnostics, strict response loading, and price-response audits.

### 5. Tests and CI

The fork adds:

- `tests/`
- `.github/workflows/ci.yml`
- `pyproject.toml`
- `requirements-dev.txt`
- `.gitignore`

CI performs source compilation, correctness-focused Ruff checks, and regression tests for the key scientific invariants.

For the full experimental design and interpretation limits, read:

**`docs/RESEARCH_PROTOCOL.md`**

---

## Repository structure

```text
demand_sim/
  config.py
  data.py
  preprocessing.py
  embeddings.py
  llm.py
  metrics.py
  evaluation.py
  research.py                 # Allegory research utilities
  models/
    llm_mix.py
    llm_mix_cal.py
    population_cal.py         # fixed population / uniform ablations
    emb.py
    gaussian.py

scripts/
  build_products.py
  build_personas.py
  build_queries.py
  build_product_embeddings.py
  build_persona_embeddings.py
  run_llm_simulations.py
  evaluate_demand_prediction.py
  evaluate_pricing.py
  make_illustrative_use_case.py

  build_strict_holdout.py     # leakage-safe experiment builder
  evaluate_fixed_holdout.py   # fixed split evaluator
  audit_llm_responses.py      # LLM response diagnostics

docs/
  RESEARCH_PROTOCOL.md

tests/
```

## Data

Download the H&M Personalized Fashion Recommendations competition files from Kaggle after accepting its terms.

Place them under the repository root:

```text
data/
  articles.csv
  customers.csv
  transactions_train.csv

images/
  <article_id>.jpg
```

The Kaggle image archive is nested by article-id prefix. This code expects a flat `images/` directory for the selected products.

## Original paper reproduction

From the repository root:

```bash
python3 scripts/build_products.py
python3 scripts/build_personas.py
python3 scripts/build_queries.py
python3 scripts/build_product_embeddings.py --allow-download
python3 scripts/build_persona_embeddings.py --allow-download

OPENAI_API_KEY=... python3 scripts/run_llm_simulations.py

python3 scripts/evaluate_demand_prediction.py --n-splits 10
python3 scripts/evaluate_pricing.py
python3 scripts/make_illustrative_use_case.py --split 6 --article-id 554450004
```

Product and persona embeddings use `google/siglip2-base-patch16-224` by default.

## Leakage-safe experiment

### Step 1 — build the fixed split and persona-safe history

```bash
python3 scripts/build_strict_holdout.py
```

This creates:

```text
outputs/strict_global_holdout/
  protocol_manifest.json
  split/
    train_products.csv
    test_products.csv
  data/
    transactions_persona_safe.csv
  personas/
    customer_features.csv
    persona_cells.csv
    persona_prompts.csv
  query_plan/
    plan_strict_global_holdout.pkl
    plan_strict_global_holdout.csv
```

### Step 2 — run the same LLM simulation machinery

```bash
OPENAI_API_KEY=... python3 scripts/run_llm_simulations.py \
  --plan outputs/strict_global_holdout/query_plan/plan_strict_global_holdout.pkl \
  --output outputs/strict_global_holdout/responses/llm_responses.csv
```

### Step 3 — audit the raw LLM responses

```bash
python3 scripts/audit_llm_responses.py \
  --responses outputs/strict_global_holdout/responses/llm_responses.csv \
  --output outputs/strict_global_holdout/response_audit.json
```

### Step 4 — build embeddings

Reuse the existing product embedding script for the same fixed product universe.

Build persona embeddings from the strict persona table and place them under the strict experiment directory.

### Step 5 — evaluate the frozen holdout

```bash
python3 scripts/evaluate_fixed_holdout.py \
  --sales outputs/products/sales_top100_online.csv \
  --responses outputs/strict_global_holdout/responses/llm_responses.csv \
  --product-embeddings outputs/embeddings/siglip2_product_embeddings.csv \
  --persona-embeddings outputs/strict_global_holdout/embeddings/siglip2_persona_embeddings.csv \
  --persona-table outputs/strict_global_holdout/personas/persona_prompts.csv \
  --train-products outputs/strict_global_holdout/split/train_products.csv \
  --test-products outputs/strict_global_holdout/split/test_products.csv \
  --output-dir outputs/fixed_holdout_evaluation
```

## Model interpretation

### `llm-mix`

Learns non-negative mixture weights over LLM persona purchase probabilities plus a dummy no-buy component.

### `llm-mix-cal`

Applies monotone logit calibration to LLM probabilities and then learns mixture weights.

### `population-cal`

Holds persona weights fixed to observed H&M persona-cell population counts and fits only the calibration/exposure parameters.

This is a more literal test of the claim that the system represents a virtual population.

### `uniform-cal`

Uses equal persona weights and fits only calibration/exposure parameters.

The comparison:

```text
llm-mix-cal vs population-cal vs uniform-cal
```

helps separate:

- semantic LLM signal,
- empirical population composition,
- flexible learned reweighting.

### `emb`

Uses SigLIP product/persona embeddings plus price in a logistic persona model.

### `gaussian`

Rounded-Gaussian aggregate-demand baseline with ridge regression.

## Evaluation metrics

For comparability with the paper, the evaluator reports:

- MAE
- RMSE
- zero-truncated NLL
- zero-truncated CRPS
- randomized PIT KS
- 90% interval score
- 95% interval score

## Important interpretation limits

### Positive-demand conditioning

The public H&M transaction data does not provide complete product-impression/exposure logs. The current benchmark therefore evaluates positive-demand observations using zero-truncated distributions.

It should not be described as fully observed conversion-rate prediction.

### Pricing is not causal validation

Historical prices are observational and may be confounded by inventory, seasonality, markdown timing, item popularity, and merchandising decisions.

The upstream pricing experiment fits a simulator to real H&M data and then uses that fitted simulator as synthetic ground truth for policy-learning experiments. It is not a randomized real-world H&M pricing experiment.

### LLM explanations are qualitative

The short `reason` text returned by the LLM is useful for inspection but should not be treated as a causal explanation of the prediction.

## Development

Install the original runtime dependencies:

```bash
pip install -r requirements.txt
```

For tests/linting:

```bash
pip install -r requirements-dev.txt
pytest
ruff check demand_sim scripts tests
```

## Research direction

The next controlled experiment should keep the strict product holdout, product information, price grids, LLM settings, calibration, aggregation, and evaluation fixed while replacing only the simple persona representation with a richer customer-state representation.

Any richer representation should have to **beat the strict simple-persona control on untouched H&M outcomes** before it is considered an improvement.
