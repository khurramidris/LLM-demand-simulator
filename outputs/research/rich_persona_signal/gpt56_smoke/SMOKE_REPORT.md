# GPT-5.6 Sol Matched-Cell Mechanism Smoke Test

**Audit:** VALID  
**Provider mode:** single-conversation batched, non-independent  
**Scientific status:** mechanism diagnostic only — **not** a demand-prediction benchmark.

## Frozen design

- 10 original H&M persona cells × 5 fixed products = 50 matched pairs.
- Two arms per pair: original paper-style prompt and the same cell enriched with pre-cutoff trouser-history summaries.
- 100 logical structured outputs total.

## Mechanical validity

- Records: **100/100**
- Complete paired arms: **50/50**
- Exact frozen price grids: **100%**
- Finite probabilities in [0,1]: **100%**
- Non-increasing price curves: **100.0%**
- Flat curves: **0.0%**

## Representation effect inside this smoke

- Mean rich − control probability shift across pairs: **+0.0739**
- Median rich − control shift: **+0.0832**
- Median mean absolute pointwise change: **0.1116**
- Pairs with higher rich mean probability: **60.0%**
- Pairs with lower rich mean probability: **40.0%**

Across the 10 persona cells, historical trouser-participation share and the average rich-minus-control shift have Spearman rho **0.927** (descriptive p=0.000112). This is a mechanism diagnostic, not inferential evidence, because the outputs are not independent/blinded.

## Interpretation

The richer state is not being ignored: it changes elicited purchase probabilities materially and in a direction related to the segment's observed pre-cutoff trouser history, while preserving sensible downward price response. That is sufficient for a mechanism smoke test.

It is **not evidence that Allegory beats the H&M paper**. The decisive experiment remains independent, blinded calls followed by the frozen aggregate held-out demand evaluation.
