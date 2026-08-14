# Rich Persona Representation Signal Gate

**Decision:** `KEEP_FOR_LLM_SMOKE_TEST`

This experiment is deliberately narrower than the eventual Allegory test. It asks whether richer behavioral state contains additional real H&M future-purchase signal before spending money on new LLM simulations.

## Metrics

| arm | ROC AUC | Average precision | Log loss | Brier | Top-1% lift |
|---|---:|---:|---:|---:|---:|
| paper_bins | 0.710045 | 0.029975 | 0.067085 | 0.013030 | 3.485 |
| paper_raw | 0.645256 | 0.022099 | 0.070820 | 0.013103 | 2.290 |
| rich_behavior | 0.692517 | 0.037960 | 0.118602 | 0.013087 | 5.078 |

## Rich vs stronger paper_raw control

- Relative average-precision gain: **71.78%**
- Absolute ROC-AUC gain: **0.047261**
- Paired-bootstrap AP delta 95% CI: [0.012616, 0.021269]

## Scope

A pass means the richer state deserves a controlled LLM smoke test. It does not yet mean that rich personas improve the H&M aggregate demand simulator. A failure means we should redesign the representation before paying for LLM calls.
