from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

from demand_sim.data import load_probability_rows

ROOT = Path("outputs/research/fresh_relative_utility_v1")
ROOT.mkdir(parents=True, exist_ok=True)
HISTORY_START = pd.Timestamp("2018-09-01")
HISTORY_END = pd.Timestamp("2019-09-19")
PRICE_SCALE = 590.0
SEED = 2025
K_BUYER_TYPES = 8


def read_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.endswith('.csv') and not n.startswith('__MACOSX/') and '/._' not in n]
        if len(names) != 1:
            raise RuntimeError(names)
        with zf.open(names[0]) as f:
            return pd.read_csv(f)


def qcut(values: pd.Series, labels: list[str]) -> pd.Series:
    try:
        out = pd.qcut(values, q=len(labels), labels=labels, duplicates='drop')
        if out.nunique(dropna=True) >= 2:
            return out.astype(object)
    except ValueError:
        pass
    ranked = values.rank(method='average')
    n_bins = min(len(labels), max(int(ranked.nunique()), 1))
    if n_bins <= 1:
        return pd.Series([labels[0]] * len(values), index=values.index, dtype=object)
    return pd.qcut(ranked, q=n_bins, labels=labels[:n_bins], duplicates='drop').astype(object)


def assign_members(features: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    d = features.copy()
    d['customer_id'] = d['customer_id'].astype(str)
    d['age'] = pd.to_numeric(d['age'], errors='coerce')
    d['txn_count'] = pd.to_numeric(d['txn_count'], errors='coerce').fillna(0)
    d['mean_price'] = pd.to_numeric(d['mean_price'], errors='coerce').fillna(0)
    d['top_product_type'] = d['top_product_type'].fillna('UNKNOWN').astype(str)
    d = d[d.txn_count > 0].copy()
    months = ((HISTORY_END - HISTORY_START).days + 1) / 30.4375
    d['txn_per_month'] = d.txn_count / months
    d['age_bin'] = pd.cut(d.age, [16,25,35,45,55,200], right=False, labels=['16-24','25-34','35-44','45-54','55+']).astype(object).fillna('UNKNOWN')
    d['engagement_bin'] = qcut(d.txn_per_month, ['low','mid','high'])
    d['price_tier'] = qcut(d.mean_price, ['low','mid','high'])
    top = d.top_product_type.value_counts().head(100).index
    d['taste_bucket'] = np.where(d.top_product_type.isin(top), d.top_product_type, 'OTHER')
    key = ['age_bin','engagement_bin','price_tier','taste_bucket']
    return d.merge(cells[key + ['persona_id','n_customers']], on=key, how='inner')[['customer_id','persona_id']].drop_duplicates('customer_id')


def prepare_tx(raw: pd.DataFrame) -> pd.DataFrame:
    tx = raw.copy()
    tx['t_dat'] = pd.to_datetime(tx.t_dat, errors='coerce')
    tx['customer_id'] = tx.customer_id.astype(str)
    tx['article_id'] = pd.to_numeric(tx.article_id, errors='coerce')
    tx['price'] = pd.to_numeric(tx.price, errors='coerce') * PRICE_SCALE
    tx = tx.dropna(subset=['t_dat','customer_id','article_id','price'])
    tx['article_id'] = tx.article_id.astype(int)
    return tx[(tx.t_dat >= HISTORY_START) & (tx.t_dat <= HISTORY_END)].copy()


def behavior_features(tx: pd.DataFrame) -> pd.DataFrame:
    d = tx.sort_values(['customer_id','t_dat']).copy()
    g = d.groupby('customer_id', sort=False)
    f = g.agg(
        txn_count=('article_id','size'), unique_articles=('article_id','nunique'), mean_price=('price','mean'),
        price_std=('price','std'), active_days=('t_dat','nunique'), first_date=('t_dat','min'), last_date=('t_dat','max')
    ).reset_index()
    f['price_std'] = f.price_std.fillna(0)
    f['recency_days'] = (HISTORY_END - f.last_date).dt.days.astype(float)
    f['span_days'] = (f.last_date - f.first_date).dt.days.astype(float)
    f['repeat_ratio'] = 1 - f.unique_articles / f.txn_count.clip(lower=1)
    for days in [30,90,180]:
        c = d[d.t_dat >= HISTORY_END - pd.Timedelta(days=days-1)].groupby('customer_id').size().rename(f'last_{days}d')
        f = f.merge(c, on='customer_id', how='left')
    return f.drop(columns=['first_date','last_date']).fillna(0)


def top_values(series: pd.Series, n: int = 5) -> str:
    vals = series.dropna().astype(str).value_counts().head(n).index.tolist()
    return ', '.join(vals) if vals else 'none'


def main() -> None:
    split_dir = Path('outputs/demand_prediction/split_000')
    train_ids = set(pd.read_csv(split_dir / 'train_products.csv').article_id.astype(int))
    test_ids = set(pd.read_csv(split_dir / 'test_products.csv').article_id.astype(int))
    if train_ids & test_ids or len(train_ids) != 60 or len(test_ids) != 40:
        raise RuntimeError('Unexpected split_000 product partition')

    cells = pd.read_csv('outputs/personas/persona_cells.csv').head(50).copy()
    members = assign_members(read_zip(Path('outputs/personas/customer_features.csv.zip')), cells)
    tx_all = prepare_tx(read_zip(Path('outputs/products/txns_trousers_online.csv.zip')))
    tx_allowed = tx_all[tx_all.article_id.isin(train_ids)].copy()
    f = behavior_features(tx_allowed)
    pop = members.merge(f, on='customer_id', how='left')
    feat = ['txn_count','unique_articles','mean_price','price_std','active_days','recency_days','span_days','repeat_ratio','last_30d','last_90d','last_180d']
    buyers = pop[pop.txn_count.notna()].copy()
    scaler = RobustScaler()
    x = scaler.fit_transform(buyers[feat])
    km = KMeans(n_clusters=K_BUYER_TYPES, random_state=SEED, n_init=20)
    buyers['type_id'] = ['T'+str(int(v)+1) for v in km.fit_predict(x)]
    pop = pop.merge(buyers[['customer_id','type_id']], on='customer_id', how='left')
    pop['type_id'] = pop.type_id.fillna('T0_NONBUYER')

    # Add type labels back to allowed transactions for evidence-grounded style summaries.
    tx_typed = tx_allowed.merge(pop[['customer_id','type_id']], on='customer_id', how='inner')
    info = pd.read_csv('outputs/products/product_info_top100_online.csv')
    info['article_id'] = info.article_id.astype(int)
    typed_info = tx_typed.merge(info, on='article_id', how='left')

    total = len(pop)
    summaries = []
    for type_id, g in pop.groupby('type_id', sort=True):
        weight = len(g) / total
        if type_id == 'T0_NONBUYER':
            text = (
                f'Behavioral buyer type {type_id}; population share {weight*100:.1f}%. '
                'No purchase of any of the 60 allowed training trouser products during the history window. '
                'Treat this as a low-category-affinity segment; do not assume zero interest in unseen trousers.'
            )
        else:
            q = g[feat].quantile([.25,.5,.75])
            tg = typed_info[typed_info.type_id == type_id]
            text = (
                f'Behavioral buyer type {type_id}; population share {weight*100:.1f}%. '
                f'Using only allowed training products: median {q.loc[.5,"txn_count"]:.0f} trouser purchases across '
                f'{q.loc[.5,"unique_articles"]:.0f} unique articles, median paid price ${q.loc[.5,"mean_price"]:.2f}, '
                f'recency {q.loc[.5,"recency_days"]:.0f} days, active days {q.loc[.5,"active_days"]:.0f}; '
                f'middle-50% purchase count {q.loc[.25,"txn_count"]:.0f}-{q.loc[.75,"txn_count"]:.0f} and mean price '
                f'${q.loc[.25,"mean_price"]:.2f}-${q.loc[.75,"mean_price"]:.2f}. '
                f'Most common colors: {top_values(tg.colour_group_name)}. '
                f'Most common garment groups: {top_values(tg.garment_group_name)}. '
                f'Most common appearances: {top_values(tg.graphical_appearance_name)}. '
                f'Historically common product names: {top_values(tg.prod_name)}.'
            )
        summaries.append({'type_id': type_id, 'n_customers': len(g), 'weight': weight, 'summary': text})
    types = pd.DataFrame(summaries).sort_values('type_id')
    types.to_csv(ROOT / 'type_summary.csv', index=False)

    probs = load_probability_rows(Path('outputs/responses/llm_responses_online_top100.csv'))
    prices = probs.groupby('article_id').offer_price.apply(lambda s: sorted(set(round(float(v),2) for v in s))).to_dict()
    catalog = info.copy()
    catalog = catalog[catalog.article_id.isin(prices)].copy().sort_values('article_id')
    catalog['price_low'] = catalog.article_id.map(lambda a: prices[int(a)][0])
    catalog['price_mid'] = catalog.article_id.map(lambda a: prices[int(a)][len(prices[int(a)])//2])
    catalog['price_high'] = catalog.article_id.map(lambda a: prices[int(a)][-1])
    keep = ['article_id','prod_name','colour_group_name','graphical_appearance_name','garment_group_name','detail_desc','price_low','price_mid','price_high']
    catalog = catalog[keep].copy()
    if len(catalog) != 100:
        raise RuntimeError(f'Expected 100 catalog products, got {len(catalog)}')
    catalog.to_csv(ROOT / 'catalog.csv', index=False)

    catalog_lines = []
    for r in catalog.itertuples(index=False):
        desc = str(r.detail_desc).replace('\n',' ').strip()
        catalog_lines.append(
            f'{int(r.article_id)} | {r.prod_name} | color={r.colour_group_name} | appearance={r.graphical_appearance_name} | '
            f'group={r.garment_group_name} | reference_price=${float(r.price_mid):.2f} | low=${float(r.price_low):.2f} | high=${float(r.price_high):.2f} | {desc}'
        )
    catalog_text = '\n'.join(catalog_lines)

    prompt_rows = []
    for r in types.itertuples(index=False):
        prompt = f'''You are estimating RELATIVE H&M trouser preferences for this evidence-grounded customer type.\n\nTYPE EVIDENCE:\n{r.summary}\n\nCATALOG:\n{catalog_text}\n\nTASK:\nEvaluate every catalog product at its reference_price. Do not estimate purchase probability and do not use or infer sales/demand. Produce relative preferences only, as if summarizing exhaustive pairwise comparisons within this buyer type.\n\nFor every article_id, assign an integer relative_preference_score from 0 to 100. Use differences meaningfully; 50 is neutral within this catalog, not a probability. Also provide price_penalty_per_10usd from 0 to 40: the number of preference-score points this type would lose if the same product cost $10 more than its reference price (and gain if $10 cheaper).\n\nReturn JSON only with exactly this schema:\n{{"type_id":"{r.type_id}","price_penalty_per_10usd":NUMBER,"scores":{{"ARTICLE_ID":INTEGER,...}}}}\n\nEvery one of the 100 article_ids must appear exactly once.''' 
        prompt_rows.append({'type_id': r.type_id, 'weight': float(r.weight), 'prompt': prompt})
    with (ROOT / 'provider_prompts.jsonl').open('w') as fobj:
        for row in prompt_rows:
            fobj.write(json.dumps(row) + '\n')

    manifest = {
        'protocol': 'FRESH-GPT56-RELATIVE-UTILITY-SPLIT0-v1',
        'split': 0,
        'n_train_products': 60,
        'n_test_products': 40,
        'n_catalog_products': 100,
        'n_types': int(len(types)),
        'provider_blinding': 'Provider prompt contains no train/test labels and no demand/sales outcomes.',
        'history_rule': 'Held-out product transactions are excluded before learned-type clustering and type evidence summaries.',
        'pre_registered_primary_gate': 'Fresh learned-type relative utility must beat paper llm-mix-cal on held-out NLL and CRPS, with MAE/RMSE no worse than +1%.',
        'secondary_question': 'Does fresh GPT-5.6 relative utility beat the zero-new-call empirical-persona relative-utility surrogate on held-out NLL or CRPS?',
        'limitations': [
            'Original top-50 persona membership uses the repository precomputed full-period customer_features artifact.',
            'Candidate price grids are historically observed and shared with the paper.',
            'GPT-5.6 judgments are produced inside this conversation rather than independent API calls.',
        ],
    }
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(types[['type_id','n_customers','weight']].to_string(index=False))


if __name__ == '__main__':
    main()
