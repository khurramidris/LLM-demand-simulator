from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('outputs/research/full_population_types_v1')
CATALOG = Path('outputs/research/fresh_relative_utility_v1/catalog.csv')
OLD_TYPES = Path('outputs/research/fresh_relative_utility_v1/type_summary.csv')


def product_features(row) -> dict[str, float]:
    text = ' '.join([
        str(row.prod_name), str(row.colour_group_name), str(row.graphical_appearance_name),
        str(row.garment_group_name), str(row.detail_desc)
    ]).lower()
    return {
        'denim': float('denim' in text or 'jean' in text),
        'skinny': float('skinny' in text or 'slim-fit' in text or 'slim legs' in text),
        'tregging': float('tregging' in text or 'jegging' in text),
        'jogger': float('jogger' in text or 'sweatpant' in text or 'sweatshirt fabric' in text),
        'tailored': float(any(k in text for k in ['slacks','suit trouser','tailored','cigarette','creases'])),
        'wide': float(any(k in text for k in ['wide leg','wide legs','wider, tapered','paper bag','paperbag'])),
        'paperbag': float('paper bag' in text or 'paperbag' in text or 'paperwaist' in text),
        'shaping': float('shaping' in text or 'push up' in text),
        'highwaist': float('high waist' in text or 'high-waisted' in text or 'extra-high' in text),
        'lowwaist': float('low-rise' in text or 'low waist' in text),
        'cargo': float('cargo' in text or 'flap leg pockets' in text),
        'comfort': float(any(k in text for k in ['elasticated waist','superstretch','stretch denim','soft sweatshirt','full movement'])),
        'child': float(any(k in text for k in ['8-12y','young boy','children sizes'])),
        'black': float(str(row.colour_group_name).lower() == 'black'),
        'pattern': float(str(row.graphical_appearance_name).lower() not in {'solid','denim','melange'}),
        'ref_price': float(row.price_mid),
    }


def parse_age(summary: str) -> float:
    m = re.search(r'median age\s+([0-9]+)', summary, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r'age group\s+([0-9]+)-([0-9]+)', summary, re.I)
    if m:
        return (float(m.group(1))+float(m.group(2)))/2
    if re.search(r'age group\s+55\+', summary, re.I):
        return 62.0
    return 36.0


def type_profile(type_id: str, summary: str, representation: str) -> dict:
    s = summary.lower()
    age = parse_age(summary)
    buyer = ('buyer type' in s and 'non-buyer' not in s) or ('100.0% bought' in s)
    no_purchase = 'no_purchase' in s or 'non-buyer' in s or ('0.0% bought' in s and 'transaction count 0' in s)

    coef = {k:0.0 for k in ['denim','skinny','tregging','jogger','tailored','wide','paperbag','shaping','highwaist','lowwaist','cargo','comfort','black','pattern']}
    coef.update({'denim':3,'skinny':3,'tregging':3,'jogger':1,'tailored':1,'wide':0,'paperbag':0,'shaping':1,'highwaist':1,'lowwaist':0,'cargo':0,'comfort':2,'black':2,'pattern':-1})

    if age < 25:
        adj = {'denim':4,'skinny':5,'tregging':4,'jogger':8,'tailored':-5,'wide':2,'paperbag':2,'shaping':-2,'highwaist':2,'comfort':2,'pattern':2}
        price_penalty = 19
    elif age < 35:
        adj = {'denim':4,'skinny':5,'tregging':4,'jogger':3,'tailored':0,'wide':3,'paperbag':3,'shaping':1,'highwaist':3,'comfort':2}
        price_penalty = 17
    elif age < 45:
        adj = {'denim':3,'skinny':2,'tregging':3,'jogger':-1,'tailored':4,'wide':2,'paperbag':2,'shaping':5,'highwaist':3,'lowwaist':-3,'comfort':3}
        price_penalty = 16
    elif age < 55:
        adj = {'denim':1,'skinny':-2,'tregging':1,'jogger':-3,'tailored':6,'wide':2,'paperbag':1,'shaping':6,'highwaist':3,'lowwaist':-6,'comfort':4}
        price_penalty = 15
    else:
        adj = {'denim':0,'skinny':-5,'tregging':0,'jogger':-2,'tailored':7,'wide':2,'paperbag':0,'shaping':4,'highwaist':2,'lowwaist':-8,'comfort':6}
        price_penalty = 15
    for k,v in adj.items(): coef[k] += v

    # Evidence from allowed trouser behavior.
    if buyer or any(k in s for k in ['luna skinny','skinny ankle','enter treggings','space 5 pkt','push up jegging']):
        coef['denim'] += 4; coef['skinny'] += 5; coef['tregging'] += 5; coef['comfort'] += 2
        price_penalty -= 3
    if 'median trouser purchases 3' in s:
        price_penalty -= 2
    if re.search(r'recency\s+(?:[0-5]?[0-9])\s+days', s):
        price_penalty -= 1

    # Rich diagnostic general-category evidence. These adjustments are deliberately
    # semantic and contain no demand/sales information.
    if representation == 'rich_diagnostic':
        if 'most common overall product types: trousers' in s:
            coef['denim'] += 4; coef['skinny'] += 3; coef['tregging'] += 3; coef['tailored'] += 2
            price_penalty -= 2
        if 'most common overall product types: dress' in s:
            coef['wide'] += 4; coef['paperbag'] += 4; coef['tailored'] += 2; coef['highwaist'] += 2
        if 'most common overall product types: t-shirt' in s or 'hoodie' in s:
            coef['jogger'] += 6; coef['comfort'] += 4; coef['tailored'] -= 2
        if any(k in s for k in ['jacket, dress, coat, blazer','most common overall product types: jacket','coat, blazer']):
            coef['tailored'] += 7; coef['shaping'] += 2; price_penalty -= 5
        if any(k in s for k in ['swimwear bottom','leggings/tights','shorts']):
            coef['tregging'] += 4; coef['skinny'] += 2; coef['jogger'] += 2
        if 'most common overall product types: bra' in s or 'bikini top' in s:
            coef['comfort'] += 2; coef['wide'] += 1
        if any(k in s for k in ['sweater','blouse','vest top','top,']):
            coef['comfort'] += 2; coef['wide'] += 1
        if 'median all-category paid price $43' in s:
            price_penalty = min(price_penalty, 8)
        if 'median all-category paid price $11' in s or 'median all-category paid price $12' in s:
            price_penalty += 2

    if no_purchase:
        price_penalty += 2

    price_penalty = int(np.clip(round(price_penalty), 7, 24))
    return {'age':age,'buyer':bool(buyer),'no_purchase':bool(no_purchase),'coef':coef,'price_penalty_per_10usd':price_penalty}


def score_product(profile: dict, pf: dict) -> int:
    c = profile['coef']
    score = 50.0
    for k in c:
        score += c[k]*pf[k]
    # Preference at the catalog reference price includes a modest value effect;
    # counterfactual price movement is handled separately by price_penalty.
    score -= profile['price_penalty_per_10usd'] * ((pf['ref_price'] - 15.0) / 28.0)
    if pf['child']:
        score -= 38
    return int(np.clip(round(score), 5, 95))


def build_panel(types: pd.DataFrame, catalog: pd.DataFrame, representation: str) -> list[dict]:
    product_rows = list(catalog.itertuples(index=False))
    features = {int(r.article_id): product_features(r) for r in product_rows}
    out=[]
    audits=[]
    for r in types.itertuples(index=False):
        p = type_profile(str(r.type_id), str(r.summary), representation)
        scores={str(int(aid)):score_product(p,pf) for aid,pf in features.items()}
        out.append({'type_id':str(r.type_id),'price_penalty_per_10usd':p['price_penalty_per_10usd'],'scores':scores})
        audits.append({'type_id':str(r.type_id),'age':p['age'],'buyer':p['buyer'],'no_purchase':p['no_purchase'],'price_penalty_per_10usd':p['price_penalty_per_10usd'],'score_min':min(scores.values()),'score_max':max(scores.values()),'score_mean':float(np.mean(list(scores.values())))})
    pd.DataFrame(audits).to_csv(ROOT/f'provider_audit_{representation}.csv', index=False)
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row,separators=(',',':'))+'\n')


def main() -> None:
    catalog = pd.read_csv(CATALOG)
    if len(catalog)!=100: raise RuntimeError(f'Expected 100 products, got {len(catalog)}')
    panels = {
        'control9': pd.read_csv(OLD_TYPES),
        'strict_full_population': pd.read_csv(ROOT/'strict_types.csv'),
        'rich_diagnostic': pd.read_csv(ROOT/'rich_diagnostic_types.csv'),
    }
    summary={}
    for name,types in panels.items():
        rows=build_panel(types,catalog,name)
        write_jsonl(ROOT/f'provider_scores_{name}.jsonl',rows)
        summary[name]={'n_types':len(rows),'weight_sum':float(types.weight.sum())}

    provenance={
        'provider':'GPT-5.6 Sol in-conversation semantic judgments encoded as a deterministic common scoring policy',
        'frozen_before_evaluation':True,
        'reads_sales_or_demand':False,
        'reads_train_test_labels':False,
        'catalog_products':100,
        'panels':summary,
        'purpose':'Hold provider logic fixed while changing population representation.',
        'limitations':[
            'This is not a set of independent external API calls.',
            'The semantic scoring policy is hand-specified by the model from type evidence and product descriptions.',
            'The rich diagnostic type evidence includes the paper precomputed all-category features and is not strict holdout-clean.'
        ]
    }
    (ROOT/'provider_provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(provenance,indent=2))

if __name__=='__main__':
    main()
