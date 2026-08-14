from __future__ import annotations

import json, re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

ROOT = Path('outputs/research/learned_utility_v1')
ROOT.mkdir(parents=True, exist_ok=True)
CUTOFF = pd.Timestamp('2019-05-31')
HISTORY_START = pd.Timestamp('2018-09-01')
PRICE_SCALE = 590.0
SEED = 2025
K_BUYER_TYPES = 8
SELECTED = ['P017','P019','P031','P038','P039','P041','P042','P043','P045','P049']


def read_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if n.endswith('.csv') and not n.startswith('__MACOSX/') and '/._' not in n]
        if len(names)!=1: raise RuntimeError(names)
        with z.open(names[0]) as f: return pd.read_csv(f)


def qcut(s, labels):
    try: return pd.qcut(s, q=len(labels), labels=labels, duplicates='drop').astype(object)
    except Exception: return pd.qcut(s.rank(method='average'), q=len(labels), labels=labels, duplicates='drop').astype(object)


def assign_cells(features: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    d=features.copy(); d['customer_id']=d['customer_id'].astype(str)
    d['age']=pd.to_numeric(d['age'],errors='coerce'); d['txn_count']=pd.to_numeric(d['txn_count'],errors='coerce').fillna(0)
    d['mean_price']=pd.to_numeric(d['mean_price'],errors='coerce').fillna(0); d['top_product_type']=d['top_product_type'].fillna('UNKNOWN').astype(str)
    d=d[d.txn_count>0].copy(); months=((pd.Timestamp('2019-09-19')-HISTORY_START).days+1)/30.4375
    d['txn_per_month']=d.txn_count/months
    d['age_bin']=pd.cut(d.age,[16,25,35,45,55,200],right=False,labels=['16-24','25-34','35-44','45-54','55+']).astype(object).fillna('UNKNOWN')
    d['engagement_bin']=qcut(d.txn_per_month,['low','mid','high']); d['price_tier']=qcut(d.mean_price,['low','mid','high'])
    top=d.top_product_type.value_counts().head(100).index; d['taste_bucket']=np.where(d.top_product_type.isin(top),d.top_product_type,'OTHER')
    key=['age_bin','engagement_bin','price_tier','taste_bucket']
    return d.merge(cells[key+['persona_id','n_customers','persona_prompt']],on=key,how='inner')


def trouser_features(tx: pd.DataFrame) -> pd.DataFrame:
    tx=tx.copy(); tx['t_dat']=pd.to_datetime(tx.t_dat,errors='coerce'); tx=tx[(tx.t_dat>=HISTORY_START)&(tx.t_dat<=CUTOFF)].copy()
    tx['customer_id']=tx.customer_id.astype(str); tx['price']=pd.to_numeric(tx.price,errors='coerce')*PRICE_SCALE; tx['article_id']=pd.to_numeric(tx.article_id,errors='coerce')
    tx=tx.dropna(subset=['customer_id','price','article_id','t_dat']).sort_values(['customer_id','t_dat'])
    g=tx.groupby('customer_id',sort=False)
    f=g.agg(txn_count=('article_id','size'),unique_articles=('article_id','nunique'),mean_price=('price','mean'),price_std=('price','std'),active_days=('t_dat','nunique'),first_date=('t_dat','min'),last_date=('t_dat','max')).reset_index()
    f['price_std']=f.price_std.fillna(0); f['recency_days']=(CUTOFF-f.last_date).dt.days; f['span_days']=(f.last_date-f.first_date).dt.days
    f['repeat_ratio']=1-f.unique_articles/f.txn_count.clip(lower=1)
    for days in [30,90,180]:
        c=tx[tx.t_dat>=CUTOFF-pd.Timedelta(days=days-1)].groupby('customer_id').size().rename(f'last_{days}d')
        f=f.merge(c,on='customer_id',how='left')
    return f.drop(columns=['first_date','last_date']).fillna(0)


def extract_products():
    src=Path('outputs/research/rich_persona_signal/gpt56_smoke/prompt_pairs.jsonl')
    rows=[json.loads(x) for x in src.read_text().splitlines() if x.strip()]
    seen={}; offers=[]
    for r in rows:
        aid=int(r['article_id'])
        if aid in seen: continue
        p=r['control_prompt']; prices=json.loads(r['prices_json']); idx=[0,len(prices)//2,len(prices)-1]
        def grab(label):
            m=re.search(rf'{label}: (.*)',p); return m.group(1).strip() if m else ''
        seen[aid]={'article_id':aid,'name':grab('Product name'),'type':grab('Product type'),'color':grab('Product color'),'description':grab('Detailed description')}
        for j in idx:
            offers.append({**seen[aid],'offer_id':f'{aid}_{j}','price':float(prices[j]),'level':['low','mid','high'][idx.index(j)]})
    return pd.DataFrame(offers)


def pair_design(offers: pd.DataFrame):
    pairs=[]; n=0
    # within-product price tradeoffs
    for aid,g in offers.groupby('article_id'):
        g=g.set_index('level');
        for a,b in [('low','mid'),('mid','high')]:
            n+=1; pairs.append({'pair_id':f'Q{n:02d}','a':g.loc[a,'offer_id'],'b':g.loc[b,'offer_id']})
    # all mid-price cross-product comparisons
    mids=offers[offers.level=='mid'].sort_values('article_id')
    ids=mids.offer_id.tolist()
    for i in range(len(ids)):
        for j in range(i+1,len(ids)):
            n+=1; pairs.append({'pair_id':f'Q{n:02d}','a':ids[i],'b':ids[j]})
    # 10 deterministic mixed-price cross-product comparisons
    rng=np.random.default_rng(SEED); candidates=[]
    rr=offers.to_dict('records')
    for i in range(len(rr)):
        for j in range(i+1,len(rr)):
            if rr[i]['article_id']!=rr[j]['article_id'] and not (rr[i]['level']=='mid' and rr[j]['level']=='mid'):
                candidates.append((rr[i]['offer_id'],rr[j]['offer_id']))
    for k in rng.choice(len(candidates),size=10,replace=False):
        a,b=candidates[int(k)]; n+=1; pairs.append({'pair_id':f'Q{n:02d}','a':a,'b':b})
    return pairs


def make_prompt(subject: str, offers: pd.DataFrame, pairs):
    lookup=offers.set_index('offer_id').to_dict('index')
    lines=[subject,'','You will compare two H&M offers at a time. Choose which offer this buyer is MORE likely to prefer.','Do NOT estimate purchase probabilities. Treat this as relative preference evidence only. Return JSON exactly: {"choices":[{"pair_id":"Q01","choice":"A"}, ...]}.','', 'Offers:']
    for r in offers.itertuples(): lines.append(f'{r.offer_id}: {r.name}; {r.color}; {r.description}; price ${r.price:.2f}')
    lines += ['', 'Pairs:']
    for q in pairs: lines.append(f"{q['pair_id']}: A={q['a']} vs B={q['b']}")
    return '\n'.join(lines)


def main():
    cells=pd.read_csv('outputs/personas/persona_cells.csv'); cells=cells[cells.persona_id.isin(SELECTED)].copy()
    features=read_zip(Path('outputs/personas/customer_features.csv.zip'))
    members=assign_cells(features,cells); members=members[members.persona_id.isin(SELECTED)].copy()
    tf=trouser_features(read_zip(Path('outputs/products/txns_trousers_online.csv.zip')))
    pop=members[['customer_id','persona_id']].drop_duplicates().merge(tf,on='customer_id',how='left')
    buyer=pop[pop.txn_count.notna()].copy(); feat=['txn_count','unique_articles','mean_price','price_std','active_days','recency_days','span_days','repeat_ratio','last_30d','last_90d','last_180d']
    scaler=RobustScaler(); X=scaler.fit_transform(buyer[feat]); km=KMeans(n_clusters=K_BUYER_TYPES,random_state=SEED,n_init=20).fit(X); buyer['learned_type']=['T'+str(x+1) for x in km.labels_]
    pop=pop.merge(buyer[['customer_id','learned_type']],on='customer_id',how='left'); pop['learned_type']=pop.learned_type.fillna('T0_NONBUYER')
    total=len(pop); summaries=[]
    for t,g in pop.groupby('learned_type',sort=True):
        share=len(g)/total
        if t=='T0_NONBUYER': text=f'Learned buyer type {t}. Population share {share*100:.1f}%. No trouser purchase before {CUTOFF.date()}.'
        else:
            q=g[feat].quantile([.25,.5,.75]); text=(f'Learned buyer type {t}. Population share {share*100:.1f}%. Before {CUTOFF.date()}, median trouser purchases {q.loc[.5,"txn_count"]:.0f}, unique articles {q.loc[.5,"unique_articles"]:.0f}, median paid price ${q.loc[.5,"mean_price"]:.2f}, recency {q.loc[.5,"recency_days"]:.0f} days, active days {q.loc[.5,"active_days"]:.0f}, repeat ratio {q.loc[.5,"repeat_ratio"]:.2f}; middle-50% purchase count {q.loc[.25,"txn_count"]:.0f}-{q.loc[.75,"txn_count"]:.0f} and price ${q.loc[.25,"mean_price"]:.2f}-${q.loc[.75,"mean_price"]:.2f}.')
        summaries.append({'type_id':t,'n_customers':len(g),'weight':share,'summary':text})
    types=pd.DataFrame(summaries); types.to_csv(ROOT/'learned_types.csv',index=False)
    offers=extract_products(); offers.to_csv(ROOT/'offers.csv',index=False); pairs=pair_design(offers); (ROOT/'pair_design.json').write_text(json.dumps(pairs,indent=2))
    simple=[]
    for r in cells.sort_values('persona_id').itertuples():
        base=r.persona_prompt.split('\n\nTask:')[0].strip(); simple.append({'id':r.persona_id,'weight':r.n_customers/cells.n_customers.sum(),'prompt':make_prompt(base,offers,pairs)})
    learned=[{'id':r.type_id,'weight':r.weight,'prompt':make_prompt(r.summary,offers,pairs)} for r in types.itertuples()]
    with (ROOT/'simple_utility_prompts.jsonl').open('w') as f:
        for r in simple: f.write(json.dumps(r)+'\n')
    with (ROOT/'learned_utility_prompts.jsonl').open('w') as f:
        for r in learned: f.write(json.dumps(r)+'\n')
    manifest={'protocol':'SIMPERSONA-PAIRWISE-UTILITY-v1','cutoff':str(CUTOFF.date()),'arms':['paper_probability_baseline','simple_pairwise_utility','learned_pairwise_utility'],'n_simple':len(simple),'n_learned':len(learned),'n_offers':len(offers),'n_pairs_per_subject':len(pairs),'selected_personas':SELECTED,'advance_gate':'learned utility must beat paper baseline on NLL and CRPS, with MAE/RMSE no worse than +5%; simple-utility arm diagnoses elicitation vs representation effect','limitations':['original selected-cell membership comes from repository precomputed full-period features','candidate price grids are shared transductive historical grids','ChatGPT conversation judgments are not independent blinded API calls']}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest,indent=2)); print(types[['type_id','n_customers','weight']].to_string(index=False))

if __name__=='__main__': main()
