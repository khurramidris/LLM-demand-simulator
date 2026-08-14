from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2025
N_PERSONAS = 10
N_PRODUCTS = 5


def main() -> None:
    outdir = Path('outputs/research/rich_persona_signal/gpt56_smoke')
    outdir.mkdir(parents=True, exist_ok=True)

    plan = pd.read_csv('outputs/query_plan/plan_top100_online.csv')
    rich = pd.read_csv('outputs/research/rich_persona_signal/matched_rich_persona_prompts.csv')
    rich = rich[rich['persona_id'].isin([f'P{i:03d}' for i in range(1, 51)])].copy()

    rng = np.random.default_rng(SEED)
    personas = np.array(sorted(rich['persona_id'].unique().tolist()), dtype=object)
    products = np.array(sorted(plan['article_id'].astype(int).unique().tolist()), dtype=int)
    chosen_personas = sorted(rng.choice(personas, size=N_PERSONAS, replace=False).tolist())
    chosen_products = sorted(int(x) for x in rng.choice(products, size=N_PRODUCTS, replace=False).tolist())

    subset = plan[
        plan['persona_id'].isin(chosen_personas) & plan['article_id'].astype(int).isin(chosen_products)
    ].copy()
    expected = N_PERSONAS * N_PRODUCTS
    if len(subset) != expected:
        raise RuntimeError(f'Expected {expected} cross-product rows, found {len(subset)}')

    subset = subset.merge(
        rich[['persona_id','base_persona_prompt','rich_persona_prompt','share_with_trouser_history']],
        on='persona_id', how='left', validate='many_to_one'
    )
    records = []
    for row in subset.sort_values(['article_id','persona_id']).itertuples(index=False):
        control = str(row.prompt)
        base = str(row.base_persona_prompt).strip()
        treatment = str(row.rich_persona_prompt).strip()
        if base not in control:
            raise RuntimeError(f'Base prompt not found inside original prompt for {row.persona_id}/{row.article_id}')
        rich_prompt = control.replace(base, treatment, 1)
        pair_id = f'{int(row.article_id)}__{row.persona_id}'
        records.append({
            'pair_id': pair_id,
            'persona_id': row.persona_id,
            'article_id': int(row.article_id),
            'price_sig': row.price_sig,
            'prices_json': row.prices_json,
            'share_with_trouser_history': float(row.share_with_trouser_history),
            'control_prompt': control,
            'rich_prompt': rich_prompt,
        })

    with (outdir / 'prompt_pairs.jsonl').open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    manifest = {
        'protocol': 'GPT56-MATCHED-CELL-SMOKE-v1',
        'model_provider': 'ChatGPT conversation model',
        'model': 'GPT-5.6 Sol',
        'seed': SEED,
        'n_personas': N_PERSONAS,
        'n_products': N_PRODUCTS,
        'logical_calls_per_arm': expected,
        'total_logical_calls': expected * 2,
        'chosen_personas': chosen_personas,
        'chosen_products': chosen_products,
        'only_intentional_prompt_difference': 'pre-cutoff behavioral-history enrichment for the same original H&M population cell',
        'evaluation_scope': [
            'strict JSON/schema validity',
            'price-response monotonicity',
            'paired probability shift magnitude',
            'whether rich-minus-control shifts covary with pre-cutoff trouser-history share',
        ],
        'not_claimed': [
            'held-out aggregate demand improvement',
            'causal price elasticity',
            'commercial validation',
        ],
    }
    (outdir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
