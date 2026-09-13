"""Reproducible paired analysis and preregistered leave-client-cluster-out calibration.

Consumes sanitized numeric score vectors only; does not call an LLM. Bootstrap
intervals condition on the recorded/out-of-fold predictions (no fit in resamples).
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, warnings
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import binomtest, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from metrics_local import agreement

SEED=20260913
PRIMARY=('kappa_nominal','alpha_ordinal','spearman')
PROFILES=('train_mode','monotone_map_B','logistic_unweighted','logistic_balanced','isotonic_B')

def checked_scores(values):
    original=np.asarray(values)
    if original.ndim!=1 or original.dtype.kind not in 'iuf' or not np.isfinite(original).all() or not np.isin(original,[0,1,2]).all():
        raise ValueError('Expected finite non-boolean categories 0/1/2')
    return original.astype(int)

def mode(y):
    c=Counter(map(int,y));return min(c,key=lambda k:(-c[k],k))

def fast_metrics(y,p):
    """Exact two-rater coefficients from a fixed-domain 3x3 confusion table."""
    y=checked_scores(y);p=checked_scores(p)
    if y.shape!=p.shape or y.ndim!=1:raise ValueError('Shape mismatch')
    if not np.isin(y,[0,1,2]).all() or not np.isin(p,[0,1,2]).all():raise ValueError('Invalid category')
    c=np.bincount(3*y+p,minlength=9).reshape(3,3).astype(float);n=c.sum()
    if n<2:return {k:None for k in PRIMARY}
    a,b=c.sum(1),c.sum(0);pe=a@b/n**2
    k=(np.trace(c)/n-pe)/(1-pe) if 1-pe>1e-15 else None
    o=c+c.T;m=o.sum(0);e=(np.outer(m,m)-np.diag(m))/(2*n-1)
    mid=np.cumsum(m)-m/2;d=(mid[:,None]-mid[None,:])**2;den=(e*d).sum()
    alpha=1-(o*d).sum()/den if den>0 else None
    ra=np.cumsum(a)-a/2-n/2;rb=np.cumsum(b)-b/2-n/2
    denom=np.sqrt((a@ra**2)*(b@rb**2))
    rho=(c*ra[:,None]*rb[None,:]).sum()/denom if denom>0 else None
    return dict(zip(PRIMARY,[None if v is None else float(v) for v in (k,alpha,rho)]))

def measure(y,p,continuous=None):
    y=checked_scores(y);p=checked_scores(p)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);out=agreement(y,p)
    d=y<2;alert=p<2;tp=int((d&alert).sum());fp=int((~d&alert).sum())
    out.update(total_units=len(y),coverage=1.0,defect_count=int(d.sum()),normal_count=int((~d).sum()),
               tp=tp,fp=fp,fn=int((d&~alert).sum()),tn=int((~d&~alert).sum()),
               defect_recall=tp/int(d.sum()) if d.any() else None,
               fpr=fp/int((~d).sum()) if (~d).any() else None,
               precision=tp/int(alert.sum()) if alert.any() else None,
               severe_count=int((y==0).sum()),severe_recall=float(np.mean(p[y==0]==0)) if (y==0).any() else None)
    if continuous is not None:
        s=np.asarray(continuous,float)
        out['spearman_continuous']=float(spearmanr(y,s).statistic) if len(set(s))>1 and len(set(y))>1 else None
    return out

def crossfit(y,a,b,groups):
    y,a,b=map(checked_scores,(y,a,b));groups=np.asarray(groups)
    if not (y.shape==a.shape==b.shape==groups.shape) or y.ndim!=1:raise ValueError('Shape mismatch')
    if len(set(groups))<2:raise ValueError('Cross-fitting requires at least two clusters')
    if not all(np.isin(v,[0,1,2]).all() for v in (y,a,b)):raise ValueError('Invalid category')
    pred={name:np.full(len(y),-1,int) for name in PROFILES}
    soft={name:np.full(len(y),np.nan) for name in PROFILES};fits=[]
    x=np.concatenate([np.eye(3)[a.astype(int)],np.eye(3)[b.astype(int)]],axis=1)
    mappings=list(itertools.combinations_with_replacement(range(3),3))
    for cluster in sorted(set(groups.tolist())):
        te=groups==cluster;tr=~te;target=y[tr].astype(int);m=mode(target)
        pred['train_mode'][te]=m;soft['train_mode'][te]=m
        candidates=[]
        for mp in mappings:
            k=fast_metrics(target,np.asarray(mp)[b[tr].astype(int)])['kappa_nominal']
            if k is not None:candidates.append((-k,sum(abs(i-v) for i,v in enumerate(mp)),mp))
        mapping=min(candidates)[2] if len(set(target))>1 and candidates else (m,m,m)
        pred['monotone_map_B'][te]=np.asarray(mapping)[b[te].astype(int)]
        soft['monotone_map_B'][te]=pred['monotone_map_B'][te]
        iso=IsotonicRegression(increasing=True,y_min=0,y_max=2,out_of_bounds='clip').fit(b[tr],target)
        q=iso.predict(b[te]);soft['isotonic_B'][te]=q
        pred['isotonic_B'][te]=np.argmin(abs(q[:,None]-np.arange(3)[None,:]),axis=1)
        for name,weight in [('logistic_unweighted',None),('logistic_balanced','balanced')]:
            if len(set(target))<2:
                pred[name][te]=m;soft[name][te]=m
            else:
                model=LogisticRegression(C=1,class_weight=weight,random_state=SEED,max_iter=1000).fit(x[tr],target)
                pred[name][te]=model.predict(x[te]);soft[name][te]=model.predict_proba(x[te])@model.classes_
        fits.append({'held_out_cluster':int(cluster),'train_units':int(tr.sum()),'eval_units':int(te.sum()),
                     'train_mode':m,'monotone_mapping':list(mapping),'monotone_fallback':len(set(target))<2 or not candidates,
                     'logistic_fallback':len(set(target))<2})
    if any(np.any(v<0) for v in pred.values()) or any(not np.isfinite(v).all() for v in soft.values()):raise RuntimeError('Missing cross-fit output')
    return pred,soft,fits

def paired_bootstrap(y,base,other,groups,n_boot=2000,seed=SEED):
    y,base,other,groups=map(np.asarray,(y,base,other,groups));unique=np.unique(groups)
    rng=np.random.default_rng(seed);loc=[np.flatnonzero(groups==g) for g in unique];values={k:[] for k in PRIMARY}
    for _ in range(n_boot):
        ix=np.concatenate([loc[i] for i in rng.integers(0,len(loc),len(loc))])
        m0=fast_metrics(y[ix],base[ix]);m1=fast_metrics(y[ix],other[ix])
        for k in PRIMARY:
            if m0[k] is not None and m1[k] is not None:values[k].append(m1[k]-m0[k])
    m0=fast_metrics(y,base);m1=fast_metrics(y,other)
    out={}
    for k,v in values.items():
        out[k]={'delta':m1[k]-m0[k] if m1[k] is not None and m0[k] is not None else None,
                'ci95':np.quantile(v,[.025,.975]).tolist() if v else None,'valid_replicates':len(v),'undefined_replicates':n_boot-len(v)}
    d0=np.asarray(base)==y;d1=np.asarray(other)==y;wins=int((~d0&d1).sum());losses=int((d0&~d1).sum())
    return {'method':'paired cluster percentile bootstrap of fixed predictions; no model refit','replicates':n_boot,
            'seed':seed,'clusters':len(unique),'metrics':out,'only_candidate_correct':wins,'only_baseline_correct':losses,
            'changed_categories':int(np.sum(base!=other)),
            'mcnemar_exact_unclustered_p':float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.0,
            'mcnemar_caveat':'Descriptive exact unit-level calculation; ignores within-client dependence.'}

def analyze(rows,output:Path,n_boot:int=2000):
    ids=[r['unit'] for r in rows]
    if len(ids)!=len(set(ids)) or not rows:raise ValueError('Empty/duplicate units')
    y=checked_scores([r['human'] for r in rows]);a=checked_scores([r['A'] for r in rows]);b=checked_scores([r['B'] for r in rows])
    groups=np.array([r['cluster'] for r in rows],int);output.mkdir(parents=True,exist_ok=True)
    base_metrics=measure(y,a);cand_metrics=measure(y,b)
    primary={'scope':'96 selected previously studied objects; conversational judge, NOT GigaChat; sequential same-session passes',
             'n':len(y),'clusters':len(set(groups)),'baseline':base_metrics,'candidate':cand_metrics,
             'comparison':paired_bootstrap(y,a,b,groups,n_boot)}
    delta=primary['comparison']['metrics'];d_fpr=cand_metrics['fpr']-base_metrics['fpr']
    primary['preregistered_practical_target']={'delta_kappa_at_least_0_10':delta['kappa_nominal']['delta']>=.1,
        'positive_alpha_delta':delta['alpha_ordinal']['delta']>0,'positive_spearman_delta':delta['spearman']['delta']>0,
        'fpr_increase_at_most_0_02':d_fpr<=.02,'coverage_loss_at_most_0_01':True,
        'all_primary_delta_ci_lower_above_zero':all(v['ci95'] is not None and v['ci95'][0]>0 for v in delta.values())}
    primary['delta_fpr']=d_fpr
    pred,soft,fits=crossfit(y,a,b,groups);cal=[]
    for name in PROFILES:
        cal.append({'profile':name,**measure(y,pred[name],soft[name]),'paired_vs_B':paired_bootstrap(y,b,pred[name],groups,n_boot)})
    numeric=[dict(r,**{name:int(pred[name][i]) for name in PROFILES},
                  continuous={name:float(soft[name][i]) for name in PROFILES}) for i,r in enumerate(rows)]
    objects={'primary_results.json':primary,'calibration_results.json':{'scope':'Exploratory secondary fixed profiles, no multiplicity correction, no held-out-cluster labels in fits','profiles':cal,'fits':fits},
             'calibrated_vectors.json':numeric}
    for filename,obj in objects.items():(output/filename).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('PRIMARY',json.dumps(primary,ensure_ascii=False,indent=2))
    print('CALIBRATION SUMMARY')
    for r in cal:print(r['profile'],{k:r[k] for k in ['kappa_nominal','alpha_ordinal','spearman','spearman_continuous','defect_recall','fpr']})
    return objects

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--vectors',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=2000)
    a=p.parse_args();analyze(json.loads(a.vectors.read_text()),a.output,a.bootstrap)
