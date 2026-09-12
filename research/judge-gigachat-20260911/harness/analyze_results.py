"""Метрики по реальным единицам, экспертам и группам; без усреднения разных шкал."""
from __future__ import annotations

import json
import math
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import krippendorff
import numpy as np
from sklearn.metrics import cohen_kappa_score

OUT=Path(__file__).parent


def finite(value: object) -> object:
    if isinstance(value,dict):
        return {k:finite(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):
        return [finite(v) for v in value]
    if isinstance(value,np.generic):
        return finite(value.item())
    if isinstance(value,float) and not math.isfinite(value):
        return None
    return value


def mode(values: list) -> float | None:
    counts=Counter(values)
    best=[v for v,n in counts.items() if n==max(counts.values())]
    return best[0] if len(best)==1 else None


def alpha(ratings: np.ndarray, level: str) -> float | None:
    try:
        return float(krippendorff.alpha(ratings,level_of_measurement=level))
    except ValueError:
        return None


def metrics(gold: np.ndarray, mean_gold: np.ndarray, pred: np.ndarray, baseline: float, maximum: float) -> dict:
    valid=np.isfinite(pred)
    paired=valid & np.isfinite(gold)
    h,j=gold[paired],pred[paired]
    labels=sorted(set(gold[np.isfinite(gold)])|set(j))
    confusion=np.array([[np.sum((h==a)&(j==b)) for b in labels] for a in labels])
    n=len(h)
    p_o=float(np.mean(h==j)) if n else np.nan
    p_e=sum(np.mean(h==c)*np.mean(j==c) for c in labels) if n else np.nan
    kappa=(p_o-p_e)/(1-p_e) if n and p_e<1 else np.nan
    known=np.isfinite(gold)
    defects=known & (gold<maximum)
    normals=known & (gold==maximum)
    tp=int(np.sum(defects & valid & (pred<maximum)))
    fp=int(np.sum(normals & valid & (pred<maximum)))
    fn=int(np.sum(defects & valid & (pred==maximum)))
    tn=int(np.sum(normals & valid & (pred==maximum)))
    score_error=pred[valid]-mean_gold[valid]
    return {'units':len(gold),'valid_scores':int(valid.sum()),'coverage':float(valid.mean()),
            'consensus_ties':int((~known).sum()),'paired_consensus':n,'accuracy':p_o,
            'correct_label_yield':int(np.sum(h==j))/np.isfinite(gold).sum() if np.isfinite(gold).any() else np.nan,
            'baseline_score':baseline,'baseline_accuracy_paired':float(np.mean(h==baseline)) if n else np.nan,
            'baseline_accuracy_all':float(np.mean(gold[known]==baseline)) if known.any() else np.nan,
            'delta_accuracy_vs_mode':float(np.mean((h==j).astype(float)-(h==baseline))) if n else np.nan,
            'cohen_kappa':kappa,'observed_agreement':p_o,'chance_agreement':p_e,
            'balanced_accuracy':float(np.mean([np.mean(j[h==c]==c) for c in set(h)])) if n else np.nan,
            'confusion':{'labels':labels,'human_rows_judge_columns':confusion},
            'defects':int(defects.sum()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'score_zero_support':int(np.sum(gold==0)),
            'score_zero_recall_all':float(np.sum((gold==0)&(pred==0))/np.sum(gold==0)) if np.any(gold==0) else np.nan,
            'score_zero_precision':float(np.sum((gold==0)&(pred==0))/np.sum(known&(pred==0))) if np.any(known&(pred==0)) else np.nan,
            'abstained_defects':int(np.sum(defects & ~valid)),
            'abstained_normals':int(np.sum(normals & ~valid)),
            'recall_all_defects':tp/defects.sum() if defects.any() else np.nan,
            'precision':tp/(tp+fp) if tp+fp else np.nan,
            'fpr_paired':fp/(fp+tn) if fp+tn else np.nan,
            'mae_to_panel_mean':float(np.mean(np.abs(score_error))) if len(score_error) else np.nan,
            'rmse_to_panel_mean':float(np.mean(score_error**2)**.5) if len(score_error) else np.nan,
            'bias_to_panel_mean':float(np.mean(score_error)) if len(score_error) else np.nan,
            'human_mean_all':float(mean_gold.mean()),
            'human_mean_paired':float(mean_gold[valid].mean()) if valid.any() else np.nan,
            'judge_mean_paired':float(pred[valid].mean()) if valid.any() else np.nan}


def bootstrap(gold: np.ndarray, means: np.ndarray, pred: np.ndarray, groups: list, baseline: float, maximum: float, repetitions: int=2000) -> dict:
    group_ids=sorted(set(groups))
    if len(group_ids)<2:
        return {'groups':len(group_ids),'intervals':None}
    indices=[np.flatnonzero(np.array(groups)==g) for g in group_ids]
    rng=np.random.default_rng(20260911)
    names=['accuracy','delta_accuracy_vs_mode','cohen_kappa','recall_all_defects','precision','fpr_paired','mae_to_panel_mean','bias_to_panel_mean']
    values=defaultdict(list)
    for _ in range(repetitions):
        take=np.concatenate([indices[i] for i in rng.integers(0,len(indices),len(indices))])
        m=metrics(gold[take],means[take],pred[take],baseline,maximum)
        for name in names:
            if math.isfinite(m[name]):
                values[name].append(m[name])
    return {'groups':len(group_ids),'repetitions':repetitions,'method':'paired_cluster_percentile_95',
            'intervals':{name:{'lower':float(np.quantile(values[name],.025)) if values[name] else None,
                              'upper':float(np.quantile(values[name],.975)) if values[name] else None,
                              'defined_replicates':len(values[name])} for name in names}}


def summarize(case: dict, runs: list[dict], criterion: str, with_bootstrap: bool) -> dict:
    lookup={u['unit_id']:u for u in case['units']}
    units=[lookup[r['unit_id']] for r in runs]
    train=[u for u in case['units'] if u['partition']=='train']
    train_votes=[mode([r['scores'][criterion] for r in u['ratings']]) for u in train]
    counts=Counter(v for v in train_votes if v is not None)
    baseline=min(v for v,n in counts.items() if n==max(counts.values()))
    h=np.array([mode([r['scores'][criterion] for r in u['ratings']]) for u in units],dtype=float)
    means=np.array([np.mean([r['scores'][criterion] for r in u['ratings']]) for u in units])
    pred=np.array([r['scores'][criterion] if r['status']=='ok' else np.nan for r in runs],dtype=float)
    groups=[u['group_id'] for u in units]
    maximum=max(case['scores'][criterion])
    m=metrics(h,means,pred,baseline,maximum)
    m['criterion']=criterion
    m['api_error_units']=sum(r['status']!='ok' for r in runs)
    m['judge_abstentions']=sum(r['status']=='ok' and r['scores'][criterion] is None for r in runs)
    m['groups']=len(set(groups))
    m['train_mode_counts']=counts
    m['train_mean']=float(np.mean([np.mean([r['scores'][criterion] for r in u['ratings']]) for u in train]))
    m['train_mean_bias_all']=m['train_mean']-float(means.mean())
    m['mode_bias_all']=baseline-float(means.mean())
    good=np.isfinite(pred)&np.isfinite(h)
    m['krippendorff_alpha_nominal']=alpha(np.array([h[good],pred[good]]),'nominal') if good.sum()>1 else None
    if len(case['scores'][criterion])>2 and good.sum()>1:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m['kappa_linear']=float(cohen_kappa_score(h[good],pred[good],labels=case['scores'][criterion],weights='linear'))
            m['kappa_quadratic']=float(cohen_kappa_score(h[good],pred[good],labels=case['scores'][criterion],weights='quadratic'))
    raters=sorted({r['rater_id'] for u in units for r in u['ratings']})
    matrix=np.full((len(raters),len(units)),np.nan)
    votes_h,votes_j,weights=[],[],[]
    for i,u in enumerate(units):
        assert len({r['rater_id'] for r in u['ratings']})==len(u['ratings'])
        for r in u['ratings']:
            matrix[raters.index(r['rater_id']),i]=r['scores'][criterion]
            if np.isfinite(pred[i]):
                votes_h.append(r['scores'][criterion])
                votes_j.append(pred[i])
                weights.append(1/len(u['ratings']))
    if len(raters)>1:
        m['human_panel_alpha_nominal']=alpha(matrix,'nominal')
        m['human_panel_alpha_ordinal']=alpha(matrix,'ordinal')
        m['human_panel_alpha_interval']=alpha(matrix,'interval')
        m['human_panel_alpha_paired_nominal']=alpha(matrix[:,np.isfinite(pred)],'nominal') if np.isfinite(pred).sum()>1 else None
        if len(votes_h)>1:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m['judge_to_individual_votes_kappa_unit_weighted']=float(cohen_kappa_score(votes_h,votes_j,sample_weight=weights))
            m['judge_to_individual_votes_accuracy_unit_weighted']=float(np.average(np.array(votes_h)==votes_j,weights=weights))
        m['human_raters']=len(raters)
        m['expert_annotations']=sum(len(u['ratings']) for u in units)
    if with_bootstrap:
        m['uncertainty']=bootstrap(h,means,pred,groups,baseline,maximum)
    return m


def main() -> None:
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--bootstrap',action='store_true')
    args=parser.parse_args()
    cases={p.stem:json.loads(p.read_text()) for p in (OUT/'cases').glob('*.json')}
    grouped=defaultdict(list)
    for path in (OUT/'runs').glob('*.json'):
        r=json.loads(path.read_text())
        grouped[(r['agent'],r['partition'],r['arm'],r['repeat'])].append(r)
    result=[]
    for (agent,partition,arm,repeat),runs in sorted(grouped.items()):
        runs.sort(key=lambda r:r['unit_id'])
        assert len({r['unit_id'] for r in runs})==len(runs), (agent,partition,arm,'Разные версии промпта под одним именем')
        row={'agent':agent,'partition':partition,'arm':arm,'repeat':repeat,
             'criteria':[summarize(cases[agent],runs,name,args.bootstrap) for name in cases[agent]['scores']],
             'responses':len(runs),'models':dict(Counter(r['response']['model'] for r in runs if 'response' in r)),
             'total_tokens':sum(r.get('response',{}).get('usage',{}).get('total_tokens',0) for r in runs)}
        audits=[a for r in runs for a in r.get('audit',[])]
        row['citation_audit']={'items':len(audits),'valid_rubric_quotes':sum(a['rubric_quote_valid'] for a in audits),
            'valid_evidence_quotes':sum(a['evidence_quote_valid'] for a in audits),'declared_missing':sum(a['declared_missing'] for a in audits),
            'id_reference_items':sum(a.get('citation_kind')=='ids' for a in audits),
            'valid_rubric_references':sum(a.get('rubric_reference_valid',False) for a in audits),
            'valid_evidence_references':sum(a.get('evidence_reference_valid',False) for a in audits)}
        result.append(row)
    (OUT/'metrics.json').write_text(json.dumps(finite(result),ensure_ascii=False,indent=2))
    for row in result:
        for m in row['criteria']:
            print(row['agent'],row['partition'],row['arm'],m['criterion'],json.dumps(finite({k:m[k] for k in ['units','coverage','accuracy','baseline_accuracy_paired','cohen_kappa','tp','fn','fp','tn','mae_to_panel_mean','bias_to_panel_mean']})))


if __name__=='__main__':
    main()
