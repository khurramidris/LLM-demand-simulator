from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler

ROOT = Path('outputs/research/full_population_types_v1')
ROOT.mkdir(parents=True, exist_ok=True)
HISTORY_START = pd.Timestamp('2018-09-01')
HISTORY_END = pd.Timestamp('2019-09-19')
PRICE_SCALE = 590.0
SEED = 2025
K_BUYER_TYPES = 8
K_RICH_TYPES = 16


def read_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.endswith('.csv') and not n.startswith('__MACOSX/') and '/._' not in n]
        if len(names) != 1:
            raise RuntimeError(names)
        with zf.open(names[0]) as f:
            return pd.read_csv(f)


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
        trouser_txn_count=('article_id','size'),
        trouser_unique_articles=('article_id','nunique'),
        trouser_mean_price=('price','mean'),
        trouser_price_std=('price','std'),
        trouser_active_days=('t_dat','nunique'),
        trouser_first_date=('t_dat','min'),
        trouser_last_date=('t_dat','max'),
    ).reset_index()
    f['trouser_price_std'] = f.trouser_price_std.fillna(0)
    f['trouser_recency_days'] = (HISTORY_END - f.trouser_last_date).dt.days.astype(float)
    f['trouser_span_days'] = (f.trouser_last_date - f.trouser_first_date).dt.days.astype(float)
    f['trouser_repeat_ratio'] = 1 - f.trouser_unique_articles / f.trouser_txn_count.clip(lower=1)
    for days in [30,90,180]:
        c = d[d.t_dat >= HISTORY_END - pd.Timedelta(days=days-1)].groupby('customer_id').size().rename(f'trouser_last_{days}d')
        f = f.merge(c, on='customer_id', how='left')
    return f.drop(columns=['trouser_first_date','trouser_last_date']).fillna(0)


def age_bucket(age: pd.Series) -> pd.Series:
    a = pd.to_numeric(age, errors='coerce')
    out = pd.cut(a, [16,25,35,45,55,200], right=False, labels=['16-24','25-34','35-44','45-54','55+']).astype(object)
    return out.fillna('UNKNOWN')


def top_values(series: pd.Series, n: int = 5) -> str:
    vals = series.dropna().astype(str).value_counts().head(n).index.tolist()
    return ', '.join(vals) if vals else 'none'


def strict_types(pop: pd.DataFrame, typed_info: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feat = [
        'trouser_txn_count','trouser_unique_articles','trouser_mean_price','trouser_price_std',
        'trouser_active_days','trouser_recency_days','trouser_span_days','trouser_repeat_ratio',
        'trouser_last_30d','trouser_last_90d','trouser_last_180d'
    ]
    d = pop.copy()
    buyers = d[d.trouser_txn_count > 0].copy()
    x = RobustScaler().fit_transform(buyers[feat])
    km = MiniBatchKMeans(n_clusters=K_BUYER_TYPES, random_state=SEED, n_init=10, batch_size=4096)
    buyers['strict_type_id'] = ['SB'+str(int(v)+1) for v in km.fit_predict(x)]
    d = d.merge(buyers[['customer_id','strict_type_id']], on='customer_id', how='left')
    nb = d.strict_type_id.isna()
    d.loc[nb, 'strict_type_id'] = 'SNB_' + d.loc[nb, 'age_bin'].astype(str).str.replace('+','P', regex=False)

    total = len(d)
    rows = []
    for tid, g in d.groupby('strict_type_id', sort=True):
        weight = len(g)/total
        is_buyer = tid.startswith('SB')
        if is_buyer:
            q = g[feat].quantile([.25,.5,.75])
            tg = typed_info[typed_info.customer_id.isin(set(g.customer_id))]
            text = (
                f'Strict full-population buyer type {tid}; population share {weight*100:.2f}%. '
                f'Only the 60 allowed training trouser products are used for behavioral features. '
                f'Median age {pd.to_numeric(g.age,errors="coerce").median():.0f}; median trouser purchases {q.loc[.5,"trouser_txn_count"]:.0f} '
                f'across {q.loc[.5,"trouser_unique_articles"]:.0f} articles; median paid price ${q.loc[.5,"trouser_mean_price"]:.2f}; '
                f'recency {q.loc[.5,"trouser_recency_days"]:.0f} days; middle-50% purchase count '
                f'{q.loc[.25,"trouser_txn_count"]:.0f}-{q.loc[.75,"trouser_txn_count"]:.0f}. '
                f'Common colors: {top_values(tg.colour_group_name)}. Common garment groups: {top_values(tg.garment_group_name)}. '
                f'Common appearances: {top_values(tg.graphical_appearance_name)}. Common product names: {top_values(tg.prod_name)}.'
            )
        else:
            ab = str(g.age_bin.iloc[0])
            text = (
                f'Strict full-population non-buyer type {tid}; population share {weight*100:.2f}%; age group {ab}. '
                'No purchase of any of the 60 allowed training trouser products during the history window. '
                'No transaction-derived general-category features are used, so held-out trouser products cannot contaminate this type. '
                'Treat this as low observed category affinity, not zero future interest.'
            )
        rows.append({'type_id':tid,'n_customers':len(g),'weight':weight,'summary':text})
    return d, pd.DataFrame(rows).sort_values('type_id')


def rich_diagnostic_types(pop: pd.DataFrame, typed_info: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pop.copy()
    a = pd.to_numeric(d.age, errors='coerce')
    d['_age'] = a.fillna(a.median())
    d['_age_missing'] = a.isna().astype(float)
    d['_log_txn'] = np.log1p(pd.to_numeric(d.txn_count, errors='coerce').fillna(0))
    d['_mean_price'] = pd.to_numeric(d.mean_price, errors='coerce').fillna(0)
    d['_has_allowed_trouser'] = (d.trouser_txn_count > 0).astype(float)
    d['_log_trouser_txn'] = np.log1p(d.trouser_txn_count.clip(lower=0))
    d['_trouser_recency_scaled'] = np.where(d.trouser_txn_count>0, np.log1p(d.trouser_recency_days), 0.0)

    top_types = d.top_product_type.fillna('NO_PURCHASE').astype(str).value_counts().head(12).index
    taste = d.top_product_type.fillna('NO_PURCHASE').astype(str).where(lambda s: s.isin(top_types), 'OTHER')
    taste_dummies = pd.get_dummies(taste, prefix='taste', dtype=np.float32)
    numeric = d[['_age','_age_missing','_log_txn','_mean_price','_has_allowed_trouser','_log_trouser_txn','_trouser_recency_scaled']].astype(np.float32)
    num_scaled = RobustScaler().fit_transform(numeric).astype(np.float32)
    x = np.hstack([num_scaled, taste_dummies.to_numpy(dtype=np.float32)])
    km = MiniBatchKMeans(n_clusters=K_RICH_TYPES, random_state=SEED, n_init=10, batch_size=8192)
    d['rich_type_id'] = ['RD'+str(int(v)+1).zfill(2) for v in km.fit_predict(x)]

    total = len(d)
    rows = []
    for tid, g in d.groupby('rich_type_id', sort=True):
        weight = len(g)/total
        trouser_buyers = g[g.trouser_txn_count > 0]
        tg = typed_info[typed_info.customer_id.isin(set(trouser_buyers.customer_id))] if len(trouser_buyers) else typed_info.iloc[0:0]
        text = (
            f'Richer full-population diagnostic type {tid}; population share {weight*100:.2f}%. '
            f'Median age {pd.to_numeric(g.age,errors="coerce").median():.0f}; median all-category transaction count '
            f'{pd.to_numeric(g.txn_count,errors="coerce").median():.0f}; median all-category paid price '
            f'${pd.to_numeric(g.mean_price,errors="coerce").median():.2f}; most common overall product types: {top_values(g.top_product_type)}. '
            f'{(g.trouser_txn_count>0).mean()*100:.1f}% bought at least one allowed training trouser product. '
        )
        if len(trouser_buyers):
            text += (
                f'Among those buyers, median allowed-trouser purchases {trouser_buyers.trouser_txn_count.median():.0f}, '
                f'median allowed-trouser paid price ${trouser_buyers.trouser_mean_price.median():.2f}, '
                f'common colors {top_values(tg.colour_group_name)}, appearances {top_values(tg.graphical_appearance_name)}, '
                f'and product names {top_values(tg.prod_name)}.'
            )
        text += ' NOTE: all-category txn_count/mean_price/top_product_type come from the paper precomputed feature artifact and are not strict product-holdout-clean.'
        rows.append({'type_id':tid,'n_customers':len(g),'weight':weight,'summary':text})
    return d, pd.DataFrame(rows).sort_values('type_id')


def make_prompts(types: pd.DataFrame, catalog: pd.DataFrame, representation: str) -> None:
    lines = []
    for r in catalog.itertuples(index=False):
        desc = str(r.detail_desc).replace('\n',' ').strip()
        lines.append(
            f'{int(r.article_id)} | {r.prod_name} | color={r.colour_group_name} | appearance={r.graphical_appearance_name} | '
            f'group={r.garment_group_name} | reference_price=${float(r.price_mid):.2f} | low=${float(r.price_low):.2f} | '
            f'high=${float(r.price_high):.2f} | {desc}'
        )
    catalog_text = '\n'.join(lines)
    path = ROOT / f'provider_prompts_{representation}.jsonl'
    with path.open('w') as f:
        for r in types.itertuples(index=False):
            prompt = f'''You are estimating RELATIVE H&M trouser preferences for an evidence-grounded population type.\n\nTYPE EVIDENCE:\n{r.summary}\n\nCATALOG:\n{catalog_text}\n\nTASK:\nEvaluate every catalog product at its reference_price. Do not estimate purchase probabilities and do not use or infer sales/demand. Produce relative preferences only.\n\nFor every article_id, assign an integer relative_preference_score from 0 to 100. Use differences meaningfully; 50 is neutral within this catalog, not a probability. Also provide price_penalty_per_10usd from 0 to 40: preference-score points lost if the same product cost $10 more than reference price.\n\nReturn JSON only with exactly this schema:\n{{"type_id":"{r.type_id}","price_penalty_per_10usd":NUMBER,"scores":{{"ARTICLE_ID":INTEGER,...}}}}\n\nEvery one of the 100 article_ids must appear exactly once.'''
            f.write(json.dumps({'representation':representation,'type_id':r.type_id,'weight':float(r.weight),'prompt':prompt})+'\n')


def main() -> None:
    split = Path('outputs/demand_prediction/split_000')
    train_ids = set(pd.read_csv(split/'train_products.csv').article_id.astype(int))
    test_ids = set(pd.read_csv(split/'test_products.csv').article_id.astype(int))
    if len(train_ids)!=60 or len(test_ids)!=40 or train_ids & test_ids:
        raise RuntimeError('Unexpected split')

    cf = read_zip(Path('outputs/personas/customer_features.csv.zip'))
    cf['customer_id'] = cf.customer_id.astype(str)
    cf['age_bin'] = age_bucket(cf.age)
    tx = prepare_tx(read_zip(Path('outputs/products/txns_trousers_online.csv.zip')))
    allowed = tx[tx.article_id.isin(train_ids)].copy()
    bf = behavior_features(allowed)
    pop = cf.merge(bf, on='customer_id', how='left')
    trouser_cols = [c for c in pop.columns if c.startswith('trouser_')]
    pop[trouser_cols] = pop[trouser_cols].fillna(0)

    info = pd.read_csv('outputs/products/product_info_top100_online.csv')
    info['article_id'] = info.article_id.astype(int)
    typed_base = allowed.merge(info, on='article_id', how='left')

    strict_pop, strict = strict_types(pop, typed_base)
    typed_strict = typed_base.merge(strict_pop[['customer_id','strict_type_id']], on='customer_id', how='left')
    # regenerate strict summaries with type-aware product evidence cheaply
    strict_pop, strict = strict_types(pop, typed_strict.rename(columns={'strict_type_id':'_ignored'}))

    rich_pop, rich = rich_diagnostic_types(pop, typed_base)

    strict.to_csv(ROOT/'strict_types.csv', index=False)
    rich.to_csv(ROOT/'rich_diagnostic_types.csv', index=False)

    catalog = pd.read_csv('outputs/research/fresh_relative_utility_v1/catalog.csv')
    catalog['article_id'] = catalog.article_id.astype(int)
    make_prompts(strict, catalog, 'strict_full_population')
    make_prompts(rich, catalog, 'rich_diagnostic')

    manifest = {
        'protocol':'FULL-POPULATION-TYPES-v1',
        'split':0,
        'fixed_train_products':60,
        'fixed_test_products':40,
        'strict_representation':{
            'population':'all customers in customer_features artifact, including customers outside the paper top-50 persona cells',
            'buyer_features':'only allowed 60-product trouser transactions',
            'nonbuyer_features':'age only',
            'n_types':int(len(strict)),
            'clean_claim':True,
        },
        'rich_diagnostic_representation':{
            'population':'same full customer population',
            'features':'age + paper precomputed all-category txn_count/mean_price/top_product_type + allowed-trouser behavior',
            'n_types':int(len(rich)),
            'clean_claim':False,
            'reason':'precomputed all-category transaction features may contain held-out product contribution',
        },
        'primary_gate':'strict full-population fresh utility must beat previous fresh GPT-5.6 learned-type arm on held-out NLL and CRPS, with MAE/RMSE no worse than +1%',
        'secondary_gate':'rich diagnostic beats strict on NLL or CRPS; if so, acquire raw Kaggle transactions and rebuild all-category features leakage-safely before any scientific claim',
        'provider_blinding':'provider prompts contain no train/test labels and no sales/demand outcomes',
    }
    (ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print('\nSTRICT TYPES\n', strict[['type_id','n_customers','weight']].to_string(index=False))
    print('\nRICH DIAGNOSTIC TYPES\n', rich[['type_id','n_customers','weight']].to_string(index=False))


if __name__ == '__main__':
    main()
