from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, gammaln
from scipy.stats import binom

ROOT = Path('outputs/research/learned_utility_v1')
BASE = Path('outputs/research/rich_persona_signal')
CUTOFF = pd.Timestamp('2019-05-31')
EXPOSURE_VALUES = [100, 150, 200, 250]
SEED = 2025
L2 = 0.2


def logit(x):
    x=np.clip(np.asarray(x,float),1e-6,1-1e-6); return np.log(x/(1-x))


def row_zt_nll(y,n,p):
    y=np.asarray(y,int); p=np.clip(np.asarray(p,float),1e-12,1-1e-12)
    out=np.full(len(y),np.inf); ok=(y>0)&(y<=n); yy=y[ok]; pp=p[ok]
    lc=gammaln(n+1)-gammaln(yy+1)-gammaln(n-yy+1)
    lp=lc+yy*np.log(pp)+(n-yy)*np.log1p(-pp); z=np.exp(n*np.log1p(-pp))
    out[ok]=-(lp-np.log1p(-z)); return out


def exact_crps(y,n,p):
    support=np.arange(1,n+1); vals=[]
    for obs,q in zip(np.asarray(y,int),np.asarray(p,float)):
        z=binom.pmf(0,n,q); pmf=binom.pmf(support,n,q)/max(1-z,1e-12); c=np.cumsum(pmf)
        vals.append(float(np.sum((c-(support>=obs).astype(float))**2)))
    return np.asarray(vals)


def load_pair_design():
    pairs=json.loads((ROOT/'pair_design.json').read_text()); offers=pd.read_csv(ROOT/'offers.csv')
    choices=pd.read_csv(ROOT/'provider_choices.csv')
    return pairs,offers,choices


def price_grids():
    rows=[json.loads(x) for x in (BASE/'gpt56_smoke/prompt_pairs.jsonl').read_text().splitlines() if x.strip()]
    out={}
    for r in rows: out.setdefault(int(r['article_id']),[float(v) for v in json.loads(r['prices_json'])])
    return out


def fit_subject_utility(subject_id,seq,pairs,offers,price_mean,price_sd):
    if len(seq)!=len(pairs): raise RuntimeError(f'{subject_id}: {len(seq)} choices != {len(pairs)} pairs')
    offer=offers.set_index('offer_id'); products=sorted(offers.article_id.astype(int).unique()); ref=products[0]; cols=products[1:]
    def fv(oid):
        r=offer.loc[oid]; a=int(r.article_id); x=[1.0 if a==p else 0.0 for p in cols]; x.append((float(r.price)-price_mean)/price_sd); return np.asarray(x)
    X=[]; y=[]
    for ch,q in zip(seq,pairs):
        if ch not in 'AB': raise RuntimeError(f'{subject_id}: bad choice {ch}')
        X.append(fv(q['a'])-fv(q['b'])); y.append(1.0 if ch=='A' else 0.0)
    X=np.vstack(X); y=np.asarray(y)
    def obj(b):
        z=X@b; nll=np.logaddexp(0,z)-y*z; return float(nll.mean()+0.5*L2*np.sum(b*b))
    bounds=[(None,None)]*(len(cols))+[(-10.0,0.0)]
    res=minimize(obj,np.zeros(len(cols)+1),method='L-BFGS-B',bounds=bounds)
    b=res.x; pred=(X@b>=0).astype(float); acc=float((pred==y).mean())
    product_effect={ref:0.0}; product_effect.update({p:float(b[i]) for i,p in enumerate(cols)})
    return {'id':subject_id,'beta':b.tolist(),'price_beta':float(b[-1]),'product_effect':product_effect,'fit_accuracy':acc,'success':bool(res.success)}


def utility_grid(arm,subjects,weights,pairs,offers,grids):
    all_prices=np.asarray([p for vals in grids.values() for p in vals],float); pm=float(all_prices.mean()); ps=float(all_prices.std() or 1)
    fits=[]; rows=[]
    for sid,seq in subjects.items(): fits.append(fit_subject_utility(sid,seq,pairs,offers,pm,ps))
    fitmap={x['id']:x for x in fits}
    for aid,prices in grids.items():
        for price in prices:
            score=0.0
            for sid,w in weights.items():
                f=fitmap[sid]; pe={int(k):v for k,v in f['product_effect'].items()}
                u=pe[int(aid)]+f['price_beta']*((price-pm)/ps); score += float(w)*float(u)
            rows.append({'arm':arm,'article_id':int(aid),'offer_price':round(float(price),2),'signal':score})
    return pd.DataFrame(rows),fits


def baseline_grid(weights):
    recs=[]
    for path in sorted((BASE/'temporal_holdout').glob('responses_*.jsonl')):
        for line in path.read_text().splitlines():
            if line.strip():
                r=json.loads(line)
                if r.get('arm')!='control': continue
                for price,p in zip(r['response']['prices'],r['response']['p_buy']):
                    recs.append({'persona_id':r['persona_id'],'article_id':int(r['article_id']),'offer_price':round(float(price),2),'p':float(p)})
    d=pd.DataFrame(recs); d['w']=d.persona_id.map(weights); d['wp']=d.p*d.w
    q=d.groupby(['article_id','offer_price'],as_index=False).wp.sum().rename(columns={'wp':'q'})
    q['signal']=logit(q.q.to_numpy(float)); q['arm']='paper_probability_baseline'
    return q[['arm','article_id','offer_price','signal']]


def fit_calibration(train):
    x=train.signal.to_numpy(float); y=train.demand.to_numpy(int); best=None
    for n in EXPOSURE_VALUES:
        if n<int(y.max()): continue
        target=float(np.clip(y.mean()/n,1e-6,1-1e-6)); init=float(logit(target)-x.mean())
        def obj(par):
            a=float(par[0]); s=float(np.exp(par[1])); p=expit(a+s*x); return float(row_zt_nll(y,n,p).mean())
        res=minimize(obj,[init,0.0],method='L-BFGS-B',bounds=[(-20,20),(-5,5)],options={'maxiter':300})
        cand={'exposure_n':int(n),'intercept':float(res.x[0]),'slope':float(np.exp(res.x[1])),'train_zt_nll':float(res.fun),'success':bool(res.success)}
        if best is None or cand['train_zt_nll']<best['train_zt_nll']: best=cand
    return best


def score(test,fit):
    out=test.copy(); n=fit['exposure_n']; p=expit(fit['intercept']+fit['slope']*out.signal.to_numpy(float)); z=np.exp(n*np.log1p(-p)); m=n*p/np.clip(1-z,1e-12,None); y=out.demand.to_numpy(int)
    out['p']=p; out['mean_prediction']=m; out['nll']=row_zt_nll(y,n,p); out['crps']=exact_crps(y,n,p); out['abs_error']=np.abs(y-m); out['sq_error']=(y-m)**2; return out


def summary(d):
    return {'n_test_rows':int(len(d)),'zt_avg_nll':float(d.nll.mean()),'zt_avg_crps':float(d.crps.mean()),'mae':float(d.abs_error.mean()),'rmse':float(np.sqrt(d.sq_error.mean()))}


def bootstrap(a,b,n_boot=5000):
    keys=['date','article_id','offer_price','demand']; aa=a.sort_values(keys).reset_index(drop=True); bb=b.sort_values(keys).reset_index(drop=True)
    if not aa[keys].equals(bb[keys]): raise RuntimeError('unpaired test rows')
    rng=np.random.default_rng(SEED); n=len(aa); result={}
    for metric,col in [('nll','nll'),('crps','crps'),('mae','abs_error')]:
        vals=[]
        for _ in range(n_boot):
            idx=rng.integers(0,n,n); vals.append(float(bb.loc[idx,col].mean()-aa.loc[idx,col].mean()))
        x=np.asarray(vals); result[metric]={'b_minus_a':float(x.mean()),'ci95_low':float(np.quantile(x,.025)),'ci95_high':float(np.quantile(x,.975)),'p_b_better':float(np.mean(x<0))}
    vals=[]
    for _ in range(n_boot):
        idx=rng.integers(0,n,n); vals.append(float(np.sqrt(bb.loc[idx,'sq_error'].mean())-np.sqrt(aa.loc[idx,'sq_error'].mean())))
    x=np.asarray(vals); result['rmse']={'b_minus_a':float(x.mean()),'ci95_low':float(np.quantile(x,.025)),'ci95_high':float(np.quantile(x,.975)),'p_b_better':float(np.mean(x<0))}
    return result


def main():
    manifest=json.loads((ROOT/'manifest.json').read_text()); pairs,offers,choice_df=load_pair_design(); grids=price_grids()
    simple_prompts=[json.loads(x) for x in (ROOT/'simple_utility_prompts.jsonl').read_text().splitlines() if x.strip()]
    simple_w={r['id']:float(r['weight']) for r in simple_prompts}; learned_df=pd.read_csv(ROOT/'learned_types.csv'); learned_w=dict(zip(learned_df.type_id,learned_df.weight.astype(float)))
    simple_sub=dict(zip(choice_df[choice_df.arm=='simple_utility'].id,choice_df[choice_df.arm=='simple_utility'].choices))
    learned_sub=dict(zip(choice_df[choice_df.arm=='learned_utility'].id,choice_df[choice_df.arm=='learned_utility'].choices))
    simple_grid,simple_fits=utility_grid('simple_pairwise_utility',simple_sub,simple_w,pairs,offers,grids)
    learned_grid,learned_fits=utility_grid('learned_pairwise_utility',learned_sub,learned_w,pairs,offers,grids)
    baseline=baseline_grid(simple_w); signals=pd.concat([baseline,simple_grid,learned_grid],ignore_index=True)
    sales=pd.read_csv('outputs/products/sales_top100_online.csv'); sales['date']=pd.to_datetime(sales.date); sales=sales.rename(columns={'price':'offer_price'}) if 'price' in sales.columns and 'offer_price' not in sales.columns else sales; sales['offer_price']=sales.offer_price.astype(float).round(2); sales['article_id']=sales.article_id.astype(int); sales=sales[sales.demand>0].copy()
    products=sorted(signals.article_id.unique()); sales=sales[sales.article_id.isin(products)].copy()
    scored={}; fits={}; summaries={}; coverage={}
    for arm,g in signals.groupby('arm'):
        m=sales.merge(g.drop(columns='arm'),on=['article_id','offer_price'],how='inner'); train=m[m.date<=CUTOFF].copy(); test=m[m.date>CUTOFF].copy(); fits[arm]=fit_calibration(train); scored[arm]=score(test,fits[arm]); summaries[arm]=summary(scored[arm]); coverage[arm]={'train':len(train),'test':len(test)}
    base='paper_probability_baseline'; simp='simple_pairwise_utility'; learn='learned_pairwise_utility'
    effects={}
    for arm in [simp,learn]:
        effects[arm]={k:{'absolute':summaries[arm][k]-summaries[base][k],'relative':(summaries[arm][k]-summaries[base][k])/summaries[base][k]} for k in ['zt_avg_nll','zt_avg_crps','mae','rmse']}
    learned_vs_simple={k:{'absolute':summaries[learn][k]-summaries[simp][k],'relative':(summaries[learn][k]-summaries[simp][k])/summaries[simp][k]} for k in ['zt_avg_nll','zt_avg_crps','mae','rmse']}
    gate=(summaries[learn]['zt_avg_nll']<summaries[base]['zt_avg_nll'] and summaries[learn]['zt_avg_crps']<summaries[base]['zt_avg_crps'] and summaries[learn]['mae']<=1.05*summaries[base]['mae'] and summaries[learn]['rmse']<=1.05*summaries[base]['rmse'])
    result={'protocol':manifest['protocol'],'decision':'ADVANCE' if gate else 'DO_NOT_ADVANCE','summary':summaries,'fits':fits,'effects_vs_baseline':effects,'learned_vs_simple_utility':learned_vs_simple,'bootstrap':{'simple_vs_baseline':bootstrap(scored[base],scored[simp]),'learned_vs_baseline':bootstrap(scored[base],scored[learn]),'learned_vs_simple':bootstrap(scored[simp],scored[learn])},'utility_fit_accuracy':{'simple':{x['id']:x['fit_accuracy'] for x in simple_fits},'learned':{x['id']:x['fit_accuracy'] for x in learned_fits}},'coverage':coverage,'limitations':manifest['limitations']+['provider was non-blinded because earlier H&M results exist in conversation context'],'claim_scope':'5-product temporal pilot testing pairwise relative-utility elicitation and SimPersona-inspired learned buyer types.'}
    (ROOT/'result.json').write_text(json.dumps(result,indent=2)); pd.DataFrame(simple_fits+learned_fits).to_json(ROOT/'utility_fits.jsonl',orient='records',lines=True)
    lines=['# Learned Buyer Types + Pairwise Utility H&M Pilot','',f"**Decision:** {result['decision']}",'','## Future-demand metrics (lower is better)','','| arm | NLL | CRPS | MAE | RMSE |','|---|---:|---:|---:|---:|']
    for arm in [base,simp,learn]:
        s=summaries[arm]; lines.append(f"| {arm} | {s['zt_avg_nll']:.6f} | {s['zt_avg_crps']:.6f} | {s['mae']:.6f} | {s['rmse']:.6f} |")
    lines+=['','## Interpretation','',f"Simple utility vs baseline NLL: {effects[simp]['zt_avg_nll']['relative']*100:+.2f}%.",f"Learned utility vs baseline NLL: {effects[learn]['zt_avg_nll']['relative']*100:+.2f}%.",f"Learned vs simple utility NLL: {learned_vs_simple['zt_avg_nll']['relative']*100:+.2f}%.",'','This is a non-blinded five-product temporal pilot, not a publication-grade product-holdout result.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
