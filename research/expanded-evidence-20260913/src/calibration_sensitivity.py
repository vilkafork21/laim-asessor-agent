"""Post-hoc uncertainty sensitivity, not a new candidate or threshold search.

Resamples client clusters and REFITS the preregistered monotone calibrator in each
replicate. All bootstrap copies of an evaluated cluster are excluded from its fit.
"""
import argparse,itertools,json
from pathlib import Path
import numpy as np
from analyze_expansion import fast_metrics,PRIMARY,SEED,mode,paired_bootstrap

MAPPINGS=list(itertools.combinations_with_replacement(range(3),3))
def monotone(target,base):
    if len(set(target))<2:return (mode(target),)*3
    options=[]
    for mp in MAPPINGS:
        k=fast_metrics(target,np.array(mp)[base])['kappa_nominal']
        if k is not None:options.append((-k,sum(abs(i-v) for i,v in enumerate(mp)),mp))
    return min(options)[2] if options else (mode(target),)*3

def run(rows,n_boot=2000):
    y=np.array([r['human'] for r in rows],int);a=np.array([r['A'] for r in rows],int);b=np.array([r['B'] for r in rows],int)
    c=np.array([r['cluster'] for r in rows],int);rng=np.random.default_rng(SEED)
    clusters=np.unique(c);indices={g:np.flatnonzero(c==g) for g in clusters};delta={z:{k:[] for k in PRIMARY} for z in ['A','B']};invalid=0
    for _ in range(n_boot):
        sampled=rng.choice(clusters,size=len(clusters),replace=True)
        if len(set(sampled))<2:invalid+=1;continue
        ix=np.concatenate([indices[g] for g in sampled]);pred=np.empty(len(ix),int)
        for g in set(sampled):
            te=c[ix]==g;tr=~te
            mapping=monotone(y[ix][tr],b[ix][tr]);pred[te]=np.array(mapping)[b[ix][te]]
        m=fast_metrics(y[ix],pred)
        for name,base in [('A',a),('B',b)]:
            baseline=fast_metrics(y[ix],base[ix])
            for k in PRIMARY:
                if m[k] is not None and baseline[k] is not None:delta[name][k].append(m[k]-baseline[k])
    out={'analysis_type':'post-hoc sensitivity, refitted leave-cluster-out monotone model within each bootstrap',
         'replicates':n_boot,'seed':SEED,'invalid_cluster_samples':invalid,'delta_intervals':{}}
    for name,values in delta.items():
        out['delta_intervals'][name]={k:{'ci95':np.quantile(v,[.025,.975]).tolist() if v else None,'valid_replicates':len(v),'undefined_replicates':n_boot-len(v)} for k,v in values.items()}
    out['fixed_prediction_comparison_to_A']=paired_bootstrap(y,a,np.array([r['monotone_map_B'] for r in rows]),c,n_boot)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--vectors',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=2000);args=p.parse_args()
    out=run(json.loads(args.vectors.read_text()),args.bootstrap);args.output.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
