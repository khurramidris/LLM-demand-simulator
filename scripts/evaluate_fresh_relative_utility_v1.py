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

ROOT = Path('outputs/research/fresh_relative_utility_v1')
SPLIT_DIR = Path('outputs/demand_prediction/split_000')
EXPOSURE_VALUES = [100,150,200,250]
SEED = 2025


def row_zt_nll(y, n, p):
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
    exposure_n:int; intercept:float; slope:float
    @property
    def name(self): return 'fresh_gpt56_learned_type_relative_utility'
    def purchase_probability(self, rows): return np.clip(expit(self.intercept+self.slope*rows.signal.to_numpy(float)),1e-9,1-1e-9)
    def mean_demand(self, rows): return self.exposure_n*self.purchase_probability(rows)


def build_signal(train_ids, type_df, catalog, score_records, probability_rows):
    expected_types=set(type_df.type_id.astype(str)); observed={str(r['type_id']) for r in score_records}
    if observed!=expected_types: raise RuntimeError(f'Type mismatch {expected_types-observed} {observed-expected_types}')
    product_ids=set(catalog.article_id.astype(int)); scoremap={}; penalties={}
    for r in score_records:
        t=str(r['type_id']); scores={int(k):float(v) for k,v in r['scores'].items()}
        if set(scores)!=product_ids: raise RuntimeError(f'{t}: score product mismatch')
        vals=np.asarray(list(scores.values()),float)
        if np.any(~np.isfinite(vals)) or np.any(vals<0) or np.any(vals>100): raise RuntimeError(f'{t}: invalid scores')
        pen=float(r['price_penalty_per_10usd'])
        if not np.isfinite(pen) or not 0<=pen<=40: raise RuntimeError(f'{t}: bad penalty')
        scoremap[t]=scores; penalties[t]=pen
    grids=probability_rows.groupby('article_id').offer_price.apply(lambda s: sorted(set(round(float(v),2) for v in s))).to_dict()
    catalog=catalog.set_index('article_id')
    type_ids=type_df.type_id.astype(str).tolist(); weights=type_df.set_index('type_id').loc[type_ids,'weight'].to_numpy(float); weights=weights/weights.sum()
    keys=[]; raw=[]
    for aid in sorted(product_ids):
        ref=float(catalog.loc[aid,'price_mid'])
        for price in grids[aid]:
            keys.append((aid,round(float(price),2)))
            raw.append([scoremap[t][aid] - penalties[t]*((float(price)-ref)/10.0) for t in type_ids])
    raw=np.asarray(raw,float); keydf=pd.DataFrame(keys,columns=['article_id','offer_price'])
    train_mask=keydf.article_id.isin(train_ids).to_numpy(); mu=raw[train_mask].mean(axis=0); sd=raw[train_mask].std(axis=0); sd=np.where(sd<1e-8,1.0,sd)
    standardized=np.clip((raw-mu)/sd,-8,8); keydf['signal']=standardized@weights
    return keydf, {'type_ids':type_ids,'weights':weights.tolist(),'train_utility_mean':mu.tolist(),'train_utility_sd':sd.tolist()}


def score(test, fit):
    model=SignalModel(int(fit['exposure_n']),float(fit['intercept']),float(fit['slope']))
    p=model.purchase_probability(test); z=np.exp(model.exposure_n*np.log1p(-p)); mean=model.mean_demand(test)/np.clip(1-z,1e-12,None)
    return pair_level_zero_truncated_binomial_scores(test,model.name,model.exposure_n,p,mean_prediction=mean,split_label='split_000_test',seed=SEED)


def summary_dict(d): return {str(r.metric):float(r.value) for r in d.itertuples(index=False)}


def weighted_product_bootstrap(base_pair, fresh_pair, n_boot=20000):
    keys=['article_id','offer_price']; b=base_pair.sort_values(keys).reset_index(drop=True); f=fresh_pair.sort_values(keys).reset_index(drop=True)
    if not b[keys+['n_observations']].equals(f[keys+['n_observations']]): raise RuntimeError('Fresh/baseline pair coverage mismatch')
    products=np.asarray(sorted(b.article_id.unique()),int); rng=np.random.default_rng(SEED); result={}
    sampled_idx=rng.integers(0,len(products),size=(n_boot,len(products)))
    for metric in ['zt_avg_nll','zt_avg_crps','mae']:
        numer=[]; denom=[]
        for aid in products:
            mask=(b.article_id.to_numpy()==aid); w=b.loc[mask,'n_observations'].to_numpy(float)
            delta=f.loc[mask,metric].to_numpy(float)-b.loc[mask,metric].to_numpy(float)
            numer.append(float(np.sum(w*delta))); denom.append(float(np.sum(w)))
        numer=np.asarray(numer); denom=np.asarray(denom)
        vals=numer[sampled_idx].sum(axis=1)/denom[sampled_idx].sum(axis=1)
        result[metric]={'fresh_minus_baseline_boot_mean':float(vals.mean()),'ci95_low':float(np.quantile(vals,.025)),'ci95_high':float(np.quantile(vals,.975)),'p_fresh_better':float(np.mean(vals<0))}
    return result


def main():
    manifest=json.loads((ROOT/'manifest.json').read_text()); provenance=json.loads((ROOT/'provider_provenance.json').read_text())
    if not provenance.get('frozen_before_evaluation'): raise RuntimeError('Provider panel not declared frozen')
    train_ids=set(pd.read_csv(SPLIT_DIR/'train_products.csv').article_id.astype(int)); test_ids=set(pd.read_csv(SPLIT_DIR/'test_products.csv').article_id.astype(int))
    if len(train_ids)!=60 or len(test_ids)!=40 or train_ids&test_ids: raise RuntimeError('Unexpected split')
    type_df=pd.read_csv(ROOT/'type_summary.csv'); catalog=pd.read_csv(ROOT/'catalog.csv'); catalog.article_id=catalog.article_id.astype(int)
    score_records=[json.loads(x) for x in (ROOT/'provider_scores.jsonl').read_text().splitlines() if x.strip()]
    probs=load_probability_rows(Path('outputs/responses/llm_responses_online_top100.csv')); signal, signal_audit=build_signal(train_ids,type_df,catalog,score_records,probs)
    sales=load_sales(Path('outputs/products/sales_top100_online.csv')); sales=sales[sales.demand>0].copy(); merged=sales.merge(signal,on=['article_id','offer_price'],how='inner')
    train=merged[merged.article_id.isin(train_ids)].copy(); test=merged[merged.article_id.isin(test_ids)].copy(); fit=fit_calibration(train); fresh_pair=score(test,fit); fresh_summary=summary_dict(summarize_pair_scores(fresh_pair))
    baseline=summary_dict(pd.read_csv(SPLIT_DIR/'evaluation/test/llm-mix-cal_summary.csv')); base_pair=pd.read_csv(SPLIT_DIR/'evaluation/test/llm-mix-cal_pair_scores.csv')
    if not base_pair[['article_id','offer_price','n_observations']].sort_values(['article_id','offer_price']).reset_index(drop=True).equals(fresh_pair[['article_id','offer_price','n_observations']].sort_values(['article_id','offer_price']).reset_index(drop=True)): raise RuntimeError('Evaluation coverage differs from paper baseline')
    prior=json.loads(Path('outputs/research/full_product_holdout_v1/result.json').read_text()); split0=next(x for x in prior['splits'] if int(x['split'])==0); surrogate=split0['summary']['empirical_persona_relative_utility']
    metrics=['zt_avg_nll','zt_avg_crps','mae','rmse']; vs_baseline={m:{'absolute':fresh_summary[m]-baseline[m],'relative':(fresh_summary[m]-baseline[m])/baseline[m]} for m in metrics}; vs_surrogate={m:{'absolute':fresh_summary[m]-float(surrogate[m]),'relative':(fresh_summary[m]-float(surrogate[m]))/float(surrogate[m])} for m in metrics}
    gate=(fresh_summary['zt_avg_nll']<baseline['zt_avg_nll'] and fresh_summary['zt_avg_crps']<baseline['zt_avg_crps'] and fresh_summary['mae']<=1.01*baseline['mae'] and fresh_summary['rmse']<=1.01*baseline['rmse']); secondary=(fresh_summary['zt_avg_nll']<surrogate['zt_avg_nll'] or fresh_summary['zt_avg_crps']<surrogate['zt_avg_crps']); decision='ADVANCE_FRESH_LEARNED_UTILITY' if gate else 'DO_NOT_ADVANCE_FRESH_LEARNED_UTILITY'
    boot=weighted_product_bootstrap(base_pair,fresh_pair)
    result={'protocol':manifest['protocol'],'decision':decision,'primary_gate_passed':gate,'beats_surrogate_on_nll_or_crps':secondary,'fit':fit,'summary':{'paper_llm_mix_cal':baseline,'empirical_persona_relative_utility_surrogate':surrogate,'fresh_gpt56_learned_type_relative_utility':fresh_summary},'effects_vs_paper':vs_baseline,'effects_vs_surrogate':vs_surrogate,'product_bootstrap_vs_paper':boot,'coverage':{'train_rows':int(len(train)),'test_rows':int(len(test)),'train_products':len(train_ids),'test_products':len(test_ids)},'signal_audit':signal_audit,'provider_provenance':provenance,'limitations':manifest['limitations'],'interpretation_rule':'If fresh beats paper but not the empirical relative-utility surrogate, relative utility earns continuation but learned-type GPT elicitation has not earned its added complexity.'}
    (ROOT/'result.json').write_text(json.dumps(result,indent=2)); fresh_pair.to_csv(ROOT/'fresh_pair_scores.csv',index=False)
    lines=['# Fresh GPT-5.6 Learned-Type Relative-Utility Product Holdout','',f'**Decision:** {decision}','','Split 0: 60 training products, 40 held-out products. Lower is better.','','| model | NLL | CRPS | MAE | RMSE |','|---|---:|---:|---:|---:|']
    for name,s in [('paper_llm_mix_cal',baseline),('empirical_persona_relative_utility_surrogate',surrogate),('fresh_gpt56_learned_type_relative_utility',fresh_summary)]: lines.append(f"| {name} | {float(s['zt_avg_nll']):.6f} | {float(s['zt_avg_crps']):.6f} | {float(s['mae']):.6f} | {float(s['rmse']):.6f} |")
    lines+=['','## Effects','']; [lines.append(f"- vs paper {m}: {vs_baseline[m]['relative']*100:+.3f}%") for m in metrics]; lines+=['']; [lines.append(f"- vs empirical-utility surrogate {m}: {vs_surrogate[m]['relative']*100:+.3f}%") for m in metrics]
    lines+=['','## Product bootstrap vs paper','']; [lines.append(f"- {m}: 95% CI [{r['ci95_low']:.6f}, {r['ci95_high']:.6f}], P(fresh better)={r['p_fresh_better']:.3f}") for m,r in boot.items()]
    lines+=['','## Scope','','Provider scores were frozen before the split labels and demand were reopened for evaluation. This remains a single-conversation GPT-5.6 provider experiment, not independent API calls.']; (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
