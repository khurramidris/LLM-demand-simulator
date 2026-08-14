# Allegory Research Protocol

This fork preserves the original paper workflow and adds a stricter experimental path for testing whether LLM-derived customer representations genuinely improve out-of-sample demand prediction.

## Scientific questions

1. Does the reported LLM advantage survive removal of held-out products from persona construction?
2. Is performance driven by a faithful population representation or by learned mixture weights that re-fit the population to observed demand?
3. How much of the signal is semantic preference information from the LLM versus statistical calibration from real sales?
4. Are raw LLM price-response curves internally coherent, especially monotone with respect to price?
5. Do richer customer-state representations improve held-out prediction when every other component is held fixed?

## Protocol A — original paper baseline

Run the upstream workflow unchanged. This remains the replication control.

The original pipeline builds persona features from the full persona-history window and later performs random product train/test splits during evaluation. This is useful as a reproduction benchmark but allows eventual test products to influence persona engagement, typical paid price, and taste before those products are declared held out.

## Protocol B — STRICT-GLOBAL-HOLDOUT

The strict protocol fixes the product split before customer-state construction.

1. Start from the same fixed 100-product universe used by the baseline.
2. Deterministically split products into train and test sets.
3. Remove every transaction involving a held-out test product from the transaction history used to construct personas.
4. Rebuild customer features and persona cells from the filtered history.
5. Query the same products, images, candidate prices, LLM model, and response schema.
6. Fit all statistical/calibration components only on train products.
7. Evaluate once on the untouched test products.
8. Record parse validity, price-monotonicity violations, mixture concentration, calibration parameters, and all predictive metrics.

Build the leakage-safe state and query plan:

```bash
python scripts/build_strict_holdout.py
```

The script writes a protocol manifest, train/test product lists, filtered persona-safe transactions, rebuilt personas, and an LLM query plan under `outputs/strict_global_holdout/`.

Run LLM simulation on the strict plan:

```bash
OPENAI_API_KEY=... python scripts/run_llm_simulations.py \
  --plan outputs/strict_global_holdout/query_plan/plan_strict_global_holdout.pkl \
  --output outputs/strict_global_holdout/responses/llm_responses.csv
```

Build product and strict-persona embeddings using the existing embedding scripts, pointing the persona embedding script at the strict persona table.

Evaluate the fixed holdout:

```bash
python scripts/evaluate_fixed_holdout.py \
  --sales outputs/products/sales_top100_online.csv \
  --responses outputs/strict_global_holdout/responses/llm_responses.csv \
  --product-embeddings outputs/embeddings/siglip2_product_embeddings.csv \
  --persona-embeddings outputs/strict_global_holdout/embeddings/siglip2_persona_embeddings.csv \
  --persona-table outputs/strict_global_holdout/personas/persona_prompts.csv \
  --train-products outputs/strict_global_holdout/split/train_products.csv \
  --test-products outputs/strict_global_holdout/split/test_products.csv \
  --output-dir outputs/fixed_holdout_evaluation
```

## Population-mechanism ablations

The fixed evaluator includes six models by default:

- `llm-mix`: original learned mixture over raw LLM probabilities.
- `llm-mix-cal`: original learned mixture with monotone logit calibration.
- `population-cal`: **new**. Uses observed H&M persona cell counts as fixed population weights; only calibration and exposure are fitted.
- `uniform-cal`: **new**. Gives each queried persona equal weight; only calibration and exposure are fitted.
- `emb`: original product/persona embedding model.
- `gaussian`: original rounded-Gaussian baseline.

These models isolate different hypotheses:

- If `llm-mix-cal >> population-cal`, much of the gain may come from learning flexible mixture weights rather than faithfully representing the observed population.
- If `population-cal >> uniform-cal`, empirical population composition carries predictive information.
- If `population-cal` approaches or beats `llm-mix-cal`, a more defensible population interpretation becomes possible.
- If all LLM models lose to simple baselines under strict holdout, the original advantage was not robust enough for a strong predictive claim.

## LLM response audit

Run:

```bash
python scripts/audit_llm_responses.py \
  --responses outputs/strict_global_holdout/responses/llm_responses.csv \
  --output outputs/strict_global_holdout/response_audit.json
```

The strict parser rejects rather than silently clips:

- malformed JSON,
- returned price grids that differ from the requested grid,
- duplicate prices,
- non-finite probabilities,
- probabilities outside `[0, 1]`.

The audit also measures how often purchase probability increases as price increases for the same persona/product/draw. Such violations are not automatically incorrect in every behavioral setting, but a high violation rate is a strong warning sign for a pricing simulator.

## Metrics

Keep the upstream metrics for comparability:

- MAE,
- RMSE,
- zero-truncated NLL,
- zero-truncated CRPS,
- randomized PIT KS,
- 90% and 95% interval scores.

Report every metric on both train and fixed test products. Do not select a preferred model using test-set performance.

## Pre-registration rules for new customer representations

Before generating LLM responses for a new representation such as ACS-50:

1. Freeze the exact train/test product lists.
2. Freeze the permitted customer-history window.
3. Prohibit held-out product transactions from every customer-state feature, summary, embedding, retrieved memory, or prompt.
4. Freeze product images, descriptions, candidate price grids, LLM model/version, number of draws, and inference settings.
5. Keep aggregation, calibration, evaluation, and exposure search identical to the simple-persona control unless the experiment explicitly studies one of those components.
6. Record every intentional difference in the protocol manifest.

The primary comparison should change **one scientific variable at a time**.

## Interpretation limits

### Zero-truncated demand

The public H&M transaction file does not provide complete impression/exposure logs for every product-day. The current evaluation therefore conditions on positive observed demand. Results should not be described as fully observed conversion-rate prediction.

### Pricing is not causal validation

Historical price is observational and confounded by product popularity, seasonality, markdown strategy, inventory, and other factors. The upstream pricing sample-efficiency experiment uses a fitted simulator as synthetic ground truth. It is useful for studying policy learning inside the learned world, but it is not a real randomized H&M pricing experiment.

### LLM explanations are not mechanisms

The `reason` string returned by the LLM is useful for qualitative inspection but is not evidence that the model used the stated reason causally.

## Next experiments

After the strict simple-persona benchmark is frozen:

1. ACS-50 richer customer state on exactly the same split.
2. Multiple LLM/behavioral-model backends with identical prompts and evaluation.
3. Price-order perturbation and repeated-draw robustness audits.
4. Distribution-first persona construction.
5. Temporal holdout in addition to product holdout.
6. Customer-level or geography-level holdouts if richer data becomes available.
7. Evaluation on datasets with explicit impressions, inventory, and zero-demand exposure periods.

The goal is not to make the simulator look impressive. The goal is to identify which components improve prediction under increasingly difficult, leakage-resistant tests.
