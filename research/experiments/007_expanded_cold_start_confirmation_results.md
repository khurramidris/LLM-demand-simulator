# Experiment 007 — Expanded H&M Cold-Start Population Confirmation — Results

Status: **NOT CONFIRMED UNDER FROZEN 1% EFFECT GATE — REPRODUCIBLE BUT SMALL CUSTOMER-STYLE SIGNAL**

Date: 2026-08-14  
Authoritative workflow run: **31797369771**  
Artifact ID: **9217907713**  
Artifact SHA-256: `8e3bb1314aa6d6052006977ea3f1653808d6f76169ce287810e5712bb225429c`

## Confirmatory cohort

The Experiment 007 cohort was completely non-overlapping with Experiment 006 and used only later products first observed from **2019-10-18 through 2020-01-31**, while all customer states, persona memberships, style profiles, priors, TF-IDF fitting and model training remained frozen using pre-2019-09-20 information.

- Qualified future cold products: **111**.
- Mapped unique buyers: **3,364**.
- Median mapped buyers/product: **26**.
- Frozen top-50 mapped customers: **244,171**.
- Outcome horizon: first 28 days from first observed sale.

## Frozen model metrics

| Model | Buyer NLL ↓ | Top-1 ↑ | Top-5 ↑ | Macro JS ↓ | Macro TV ↓ |
|---|---:|---:|---:|---:|---:|
| P0 population prior | 3.832770 | 0.0431 | 0.2551 | 0.309756 | 0.623729 |
| P1 historical trouser-buyer prior | 3.673910 | 0.0654 | 0.2857 | 0.254492 | 0.534043 |
| **P2 persona style-profile** | **3.647411** | 0.0728 | **0.3047** | **0.247180** | **0.521686** |
| P3 supervised product content | 3.660195 | **0.0788** | 0.2913 | 0.254306 | 0.533711 |

P2 was the best model on the preregistered primary buyer NLL and also on Top-5 accuracy, macro Jensen-Shannon divergence and macro total-variation distance. P3 had the best Top-1 accuracy.

## Primary confirmatory comparison: P2 versus P1

- P1 NLL: **3.673910**.
- P2 NLL: **3.647411**.
- Buyer-weighted ΔNLL P2−P1: **-0.026499**.
- Relative NLL improvement: **0.7213%**.
- Product-cluster-bootstrap 95% CI: **[-0.039775, -0.013323]**.
- Early chronological half ΔNLL: **-0.012062**.
- Late chronological half ΔNLL: **-0.041765**.
- P2/P1 macro-JS ratio: **0.97127** (P2 improves divergence rather than degrading it).

Thus the direction is favorable, product-cluster uncertainty excludes zero, and the improvement replicates in both chronological halves. However, the frozen minimum practical-effect threshold was **1.0% relative NLL**, and the observed confirmatory effect is **0.721%**.

## Frozen gate

- >=60 qualified products: **PASS (111)**.
- >=1,500 mapped buyers: **PASS (3,364)**.
- median >=10 mapped buyers/product: **PASS (26)**.
- P2 NLL improvement >=1.0% relative: **FAIL (0.721%)**.
- P2−P1 bootstrap CI strictly below zero: **PASS**.
- favorable direction in both chronological halves: **PASS**.
- P2 macro-JS not >5% worse than P1: **PASS; P2 is better**.
- Earn direct LLM-population challenge under preregistration: **NO**.

We do not lower the 1% threshold after observing the result.

## What was learned

The stronger H&M benchmark did uncover a **real, repeatable customer/population-structure signal** that the paper's original benchmark did not isolate. A frozen persona-level representation of historical style preferences predicted the future buyer composition of unseen products better than a generic trouser-buyer prior on 111 independent cold products and 3,364 real buyer events.

The effect is nevertheless **small** on the primary proper scoring rule: 0.721% relative NLL. It is statistically stable but below the predeclared minimum practical effect required to justify spending LLM calls as a scientific H&M validation experiment.

This is therefore more informative than saying “H&M contains no useful customer signal,” but it does **not** support the claim that an LLM virtual population is beneficial. No LLM was run in Experiment 007.

## Scientific decision

- **KEEP** H&M for reproduction, engineering, cold-start segmentation demonstrations and exploratory research.
- **KEEP, but characterize as modest:** evidence that frozen customer/persona style structure helps predict which groups buy unseen H&M products.
- **DO NOT CLAIM:** that LLM populations outperform statistics or predict H&M demand better.
- **DO NOT PROMOTE** H&M as Allegory's flagship population-validation benchmark.
- **DO NOT SPEND** LLM/API calls on H&M as a confirmatory scientific experiment under the frozen protocol.

A future H&M LLM-C/LLM-R run would have to be explicitly labeled **exploratory**, not confirmatory.

The stronger flagship validation should still use data with explicit exposure/choice/intervention ground truth.

`main` remains untouched. This branch/PR remains research-only and unmerged.
