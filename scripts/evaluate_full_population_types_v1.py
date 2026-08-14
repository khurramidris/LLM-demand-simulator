from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, gammaln

from demand_sim.data import load_probability_rows, load_sales
from demand_sim.metrics import pair_level_zero_truncated_binomial_scores, summarize_pair_scores

ROOT = Path('outputs/research/full_population_types_v1')
SPLIT = Path('outputs/demand_prediction/split_000')
EXPOSURE_VALUES = [100,150,200,250]
SEED = 2025


def row_zt_nll(y,n,p):
    y=np.asarray(y,int); p=np.clip(np.asarray(p,float),1e-12,1-1e-12)
    out=np.full(len(y),np.inf,float); ok=(y>0)&(y<=n); yy=y[ok]; pp=p[ok]
    lc=gammaln(n+1)-gammaln(yy+1)-gammaln(n-yy+1)
    lp=lc+yy*np.log(pp)+(n-yy)*np.log1p(-pp); z=np.exp(n*np.log1p(-pp))
    out[ok]=-(lp-np.log1p(-z)); return out


def fit_calibration(rows):
    x=rows.signal.to_numpy(float); y=rows.demand.to_numpy(int); best=None
    for n in EXPOSURE_VALUES:
        if n<int(y.max()): continue
        target=float(np.clip(y.mean()/n,1e-6,1-1e-6)); init=math.log(target/(1-target))
        def obj(par):
            a=float(par[0]); s=float(np.exp(par[1])); return float(row_zt_nll(y,n,expit(a+s*x)).mean())
        res=minimize(obj,[init,0.0],method='L-BFGS-B',bounds=[(-20,20),(-5,5)],options={'maxiter':300})
        cand={'exposure_n':int(n),'intercept':float(res.x[0]),'slope':float(np.exp(res.x[1])),'train_zt_nll':float(res.fun),'success':bool(res.success)}
        if best is None or cand['train_zt_nll']<best['train_zt_nll']: best=cand
    if best is None: raise RuntimeError('No calibration fit')
    return best


@dataclass
class SignalModel:
    model_name:str; exposure_n:int; intercept:float; slope:float
    @property
    def name(self): return self.model_name
    def purchase_probability(self,rows): return np.clip(expit(self.intercept+self.slope*rows.signal.to_numpy(float)),1e-9,1-1e-9)
    def mean_demand(self,rows): return self.exposure_n*self.purchase_probability(rows)


def load_scores(path: Path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def build_signal(types: pd.DataFrame, score_records: list[dict], catalog: pd.DataFrame, probability_rows: pd.DataFrame, train_ids:set[int]):
    expected=set(types.type_id.astype(str)); observed={str(r['type_id']) for r in score_records}
    if expected!=observed: raise RuntimeError(f'type mismatch missing={expected-observed} extra={observed-expected}')
    product_ids=set(catalog.article_id.astype(int)); grids=probability_rows.groupby('article_id').offer_price.apply(lambda s:sorted(set(round(float(v),2) for v in s))).to_dict()
    cat=catalog.set_index('article_id'); type_ids=types.type_id.astype(str).tolist(); weights=types.set_index('type_id').loc[type_ids,'weight'].to_numpy(float); weights=weights/weights.sum()
    scoremap={}; penalties={}
    for r in score_records:
        t=str(r['type_id']); scores={int(k):float(v) for k,v in r['scores'].items()}
        if set(scores)!=product_ids: raise RuntimeError(f'{t}: product mismatch')
        scoremap[t]=scores; penalties[t]=float(r['price_penalty_per_10usd'])
    keys=[]; raw=[]
    for aid in sorted(product_ids):
        ref=float(cat.loc[aid,'price_mid'])
        for price in grids[aid]:
            keys.append((aid,round(float(price),2)))
            raw.append([scoremap[t][aid]-penalties[t]*((float(price)-ref)/10.0) for t in type_ids])
    raw=np.asarray(raw,float); keydf=pd.DataFrame(keys,columns=['article_id','offer_price'])
    mask=keydf.article_id.isin(train_ids).to_numpy(); mu=raw[mask].mean(axis=0); sd=raw[mask].std(axis=0); sd=np.where(sd<1e-8,1.0,sd)
    z=np.clip((raw-mu)/sd,-8,8); keydf['signal']=z@weights
    audit={'n_types':len(type_ids),'weights':weights.tolist(),'train_utility_mean':mu.tolist(),'train_utility_sd':sd.tolist()}
    return keydf,audit


def score_model(name,test,fit):
    model=SignalModel(name,int(fit['exposure_n']),float(fit['intercept']),float(fit['slope']))
    p=model.purchase_probability(test); zero=np.exp(model.exposure_n*np.log1p(-p)); mean=model.mean_demand(test)/np.clip(1-zero,1e-12,None)
    pair=pair_level_zero_truncated_binomial_scores(test,model.name,model.exposure_n,p,mean_prediction=mean,split_label='split_000_test',seed=SEED)
    summary={str(r.metric):float(r.value) for r in summarize_pair_scores(pair).itertuples(index=False)}
    return pair,summary


def vector_product_bootstrap(a:pd.DataFrame,b:pd.DataFrame,n_boot=20000):
    keys=['article_id','offer_price']; aa=a.sort_values(keys).reset_index(drop=True); bb=b.sort_values(keys).reset_index(drop=True)
    if not aa[keys+['n_observations']].equals(bb[keys+['n_observations']]): raise RuntimeError('unpaired pair scores')
    products=np.asarray(sorted(aa.article_id.unique()),int); idx={aid:np.flatnonzero(aa.article_id.to_numpy()==aid) for aid in products}; rng=np.random.default_rng(SEED)
    result={}
    for metric in ['zt_avg_nll','zt_avg_crps','mae']:
        delta=bb[metric].to_numpy(float)-aa[metric].to_numpy(float); w=aa.n_observations.to_numpy(float); prod_num=np.array([np.sum(delta[idx[a]]*w[idx[a]]) for a in products]); prod_den=np.array([np.sum(w[idx[a]]) for a in products])
        draws=rng.integers(0,len(products),size=(n_boot,len(products))); vals=prod_num[draws].sum(axis=1)/prod_den[draws].sum(axis=1)
        result[metric]={'b_minus_a_boot_mean':float(vals.mean()),'ci95_low':float(np.quantile(vals,.025)),'ci95_high':float(np.quantile(vals,.975)),'p_b_better':float(np.mean(vals<0))}
    return result


def main():
    manifest=json.loads((ROOT/'manifest.json').read_text()); provenance=json.loads((ROOT/'provider_provenance.json').read_text())
    if not provenance.get('frozen_before_evaluation') or provenance.get('reads_sales_or_demand') or provenance.get('reads_train_test_labels'):
        raise RuntimeError('Provider provenance violates freeze/blinding rules')
    train_ids=set(pd.read_csv(SPLIT/'train_products.csv').article_id.astype(int)); test_ids=set(pd.read_csv(SPLIT/'test_products.csv').article_id.astype(int))
    if len(train_ids)!=60 or len(test_ids)!=40 or train_ids&test_ids: raise RuntimeError('Unexpected split')
    catalog=pd.read_csv('outputs/research/fresh_relative_utility_v1/catalog.csv'); catalog.article_id=catalog.article_id.astype(int)
    probs=load_probability_rows(Path('outputs/responses/llm_responses_online_top100.csv'))
    sales=load_sales(Path('outputs/products/sales_top100_online.csv')); sales=sales[sales.demand>0].copy()

    panels={
        'control9_common_provider':(pd.read_csv('outputs/research/fresh_relative_utility_v1/type_summary.csv'),ROOT/'provider_scores_control9.jsonl'),
        'strict_full_population_common_provider':(pd.read_csv(ROOT/'strict_types.csv'),ROOT/'provider_scores_strict_full_population.jsonl'),
        'rich_diagnostic_common_provider':(pd.read_csv(ROOT/'rich_diagnostic_types.csv'),ROOT/'provider_scores_rich_diagnostic.jsonl'),
    }
    summaries={}; fits={}; pairs={}; audits={}
    for name,(types,path) in panels.items():
        signal,audit=build_signal(types,load_scores(path),catalog,probs,train_ids); merged=sales.merge(signal,on=['article_id','offer_price'],how='inner')
        train=merged[merged.article_id.isin(train_ids)].copy(); test=merged[merged.article_id.isin(test_ids)].copy(); fit=fit_calibration(train); pair,summ=score_model(name,test,fit)
        fits[name]=fit; pairs[name]=pair; summaries[name]=summ; audits[name]=audit

    paper={str(r.metric):float(r.value) for r in pd.read_csv(SPLIT/'evaluation/test/llm-mix-cal_summary.csv').itertuples(index=False)}
    paper_pair=pd.read_csv(SPLIT/'evaluation/test/llm-mix-cal_pair_scores.csv')
    previous=json.loads(Path('outputs/research/fresh_relative_utility_v1/result.json').read_text())['summary']['fresh_gpt56_learned_type_relative_utility']
    summaries['paper_llm_mix_cal']=paper; summaries['previous_fresh_gpt56_9type']=previous

    metrics=['zt_avg_nll','zt_avg_crps','mae','rmse']
    strict=summaries['strict_full_population_common_provider']; control=summaries['control9_common_provider']; rich=summaries['rich_diagnostic_common_provider']
    def effects(a,b): return {m:{'absolute':float(a[m]-b[m]),'relative':float((a[m]-b[m])/b[m])} for m in metrics}
    eff={
        'strict_vs_control9':effects(strict,control),
        'strict_vs_previous_fresh':effects(strict,previous),
        'strict_vs_paper':effects(strict,paper),
        'rich_vs_strict':effects(rich,strict),
    }
    prereg=(strict['zt_avg_nll']<previous['zt_avg_nll'] and strict['zt_avg_crps']<previous['zt_avg_crps'] and strict['mae']<=1.01*previous['mae'] and strict['rmse']<=1.01*previous['rmse'])
    representation_gate=(strict['zt_avg_nll']<control['zt_avg_nll'] and strict['zt_avg_crps']<control['zt_avg_crps'] and strict['mae']<=1.01*control['mae'] and strict['rmse']<=1.01*control['rmse'])
    rich_signal=(rich['zt_avg_nll']<strict['zt_avg_nll'] or rich['zt_avg_crps']<strict['zt_avg_crps'])
    decision='KEEP_STRICT_FULL_POPULATION' if prereg and representation_gate else ('PROMISING_BUT_NOT_PROVEN' if prereg or representation_gate else 'KILL_STRICT_FULL_POPULATION')

    result={
        'protocol':manifest['protocol'],'decision':decision,'preregistered_gate_passed':bool(prereg),'representation_isolation_gate_passed':bool(representation_gate),'rich_diagnostic_signal':bool(rich_signal),
        'summary':summaries,'fits':fits,'effects':eff,
        'bootstrap':{
            'strict_vs_control9':vector_product_bootstrap(pairs['control9_common_provider'],pairs['strict_full_population_common_provider']),
            'strict_vs_paper':vector_product_bootstrap(paper_pair,pairs['strict_full_population_common_provider']),
            'rich_vs_strict':vector_product_bootstrap(pairs['strict_full_population_common_provider'],pairs['rich_diagnostic_common_provider']),
        },
        'signal_audit':audits,'provider_provenance':provenance,'manifest':manifest,
        'interpretation_rule':'The strict arm can support a clean representation claim. The rich diagnostic cannot support a scientific holdout claim until all-category features are rebuilt from raw transactions with held-out products removed.'
    }
    (ROOT/'result.json').write_text(json.dumps(result,indent=2))
    for name,pair in pairs.items(): pair.to_csv(ROOT/f'pair_scores_{name}.csv',index=False)

    order=['paper_llm_mix_cal','previous_fresh_gpt56_9type','control9_common_provider','strict_full_population_common_provider','rich_diagnostic_common_provider']
    lines=['# Full-Population Representation Experiment','',f'**Decision:** {decision}','',f'Pre-registered strict-vs-previous gate: **{prereg}**  ','Same-provider representation gate: **{representation_gate}**  ','Rich diagnostic signal (non-claimable): **'+str(rich_signal)+'**','','| model | NLL | CRPS | MAE | RMSE |','|---|---:|---:|---:|---:|']
    for name in order:
        s=summaries[name]; lines.append(f"| {name} | {float(s['zt_avg_nll']):.6f} | {float(s['zt_avg_crps']):.6f} | {float(s['mae']):.6f} | {float(s['rmse']):.6f} |")
    lines+=['','## Strict full-population effects','']
    for comp in ['strict_vs_control9','strict_vs_previous_fresh','strict_vs_paper']:
        lines.append(f'### {comp}')
        for m in metrics: lines.append(f"- {m}: {eff[comp][m]['relative']*100:+.3f}%")
    lines+=['','## Scientific scope','','Strict non-buyer segmentation uses age only; observed trouser-buyer clustering uses only the 60 allowed training products. Rich diagnostic all-category features are explicitly not claimable because the raw full transaction history is absent from the repository.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
