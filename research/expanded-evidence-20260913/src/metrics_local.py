"""Agreement on explicit fixed score levels; no NaN-as-zero or implicit casts.
Krippendorff alpha via the standard coincidence matrix (nominal/ordinal/interval).
"""
import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import cohen_kappa_score, confusion_matrix

def alpha_counts(counts, kind='ordinal', levels=(0,1,2)):
    counts=np.asarray(counts,dtype=float); counts=counts[counts.sum(axis=1)>=2]
    if not len(counts):return None
    o=np.zeros((len(levels),len(levels)))
    for c in counts:o+=(np.outer(c,c)-np.diag(c))/(c.sum()-1)
    marg=o.sum(axis=0);n=marg.sum()
    if n<=1:return None
    e=(np.outer(marg,marg)-np.diag(marg))/(n-1)
    if kind=='nominal':dist=1-np.eye(len(levels))
    elif kind=='interval':dist=(np.asarray(levels)[:,None]-np.asarray(levels)[None,:])**2
    elif kind=='ordinal':
        mid=np.cumsum(marg)-marg/2;dist=(mid[:,None]-mid[None,:])**2
    else:raise ValueError(kind)
    den=(e*dist).sum();return float(1-(o*dist).sum()/den) if den else None

def panel_counts(matrix,levels=(0,1,2)):
    matrix=np.asarray(matrix,dtype=float)
    return np.stack([(matrix==v).sum(axis=1) for v in levels],axis=1)

def agreement(y,p,levels=(0,1,2),continuous=None):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    mask=np.isfinite(y)&np.isfinite(p);y=y[mask];p=p[mask]
    score=np.asarray(continuous,dtype=float)[mask] if continuous is not None else p
    if not len(y):return {'n':0}
    if not set(y).issubset(levels) or not set(p).issubset(levels):raise ValueError('out-of-scale label')
    # Encode categories by declared level order for sklearn.
    a=np.searchsorted(levels,y);b=np.searchsorted(levels,p)
    out={'n':len(y),'accuracy':float((y==p).mean()),'bias':float((p-y).mean()),'mae':float(np.abs(p-y).mean()),'confusion':confusion_matrix(a,b,labels=range(len(levels))).tolist(),
         'gold_counts':[int((y==v).sum()) for v in levels],'pred_counts':[int((p==v).sum()) for v in levels]}
    for w in (None,'linear','quadratic'):
        v=cohen_kappa_score(a,b,labels=range(len(levels)),weights=w);out['kappa_'+(w or 'nominal')]=float(v) if np.isfinite(v) else None
    counts=panel_counts(np.stack([y,p],axis=1),levels)
    for kind in ('nominal','ordinal','interval'):out['alpha_'+kind]=alpha_counts(counts,kind,levels)
    good=np.isfinite(score)
    out['spearman_continuous' if continuous is not None else 'spearman']=float(spearmanr(y[good],score[good]).statistic) if len(set(y[good]))>1 and len(set(score[good]))>1 else None
    out['kendall_tau_b']=float(kendalltau(y[good],score[good]).statistic) if len(set(y[good]))>1 and len(set(score[good]))>1 else None
    return out
