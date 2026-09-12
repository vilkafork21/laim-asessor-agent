"""Парные сравнения, чувствительность весов и наблюдаемая устойчивость judge."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from analyze_results import finite, metrics, mode
from run_judge import build_request, digest

OUT=Path(__file__).parent


def read(path: Path) -> object:
    return json.loads(path.read_text())


def runs_for(case: dict, units: list, arm: str) -> list:
    return [read(OUT/'runs'/f"{digest([build_request(case,u,arm),u['unit_id'],0])}.json") for u in units]


def scores(runs: list, criterion: str) -> np.ndarray:
    return np.array([r['scores'][criterion] if r['status']=='ok' else np.nan for r in runs],dtype=float)


def compare(gold: np.ndarray, means: np.ndarray, candidate: np.ndarray, direct: np.ndarray, maximum: float) -> dict:
    common=np.isfinite(candidate)&np.isfinite(direct)
    known=common&np.isfinite(gold)
    one=metrics(gold[common],means[common],candidate[common],maximum,maximum) if common.any() else {}
    two=metrics(gold[common],means[common],direct[common],maximum,maximum) if common.any() else {}
    return {'common_scores':int(common.sum()),'common_consensus':int(known.sum()),
            'coverage_delta':float(np.mean(np.isfinite(candidate))-np.mean(np.isfinite(direct))),
            'accuracy_delta_common':float(np.mean((candidate[known]==gold[known]).astype(float)-(direct[known]==gold[known]))) if known.any() else np.nan,
            'kappa_delta_common':one.get('cohen_kappa',np.nan)-two.get('cohen_kappa',np.nan),
            'mae_delta_common':float(np.mean(np.abs(candidate[common]-means[common])-np.abs(direct[common]-means[common]))) if common.any() else np.nan,
            'candidate_wins':int(np.sum(known&(candidate==gold)&(direct!=gold))),
            'direct_wins':int(np.sum(known&(direct==gold)&(candidate!=gold)))}


def uncertainty(gold: np.ndarray, means: np.ndarray, candidate: np.ndarray, direct: np.ndarray, groups: list, maximum: float) -> dict:
    unique=sorted(set(groups))
    indices=[np.flatnonzero(np.array(groups)==g) for g in unique]
    values=defaultdict(list)
    rng=np.random.default_rng(20260911)
    for _ in range(2000):
        take=np.concatenate([indices[i] for i in rng.integers(0,len(indices),len(indices))])
        result=compare(gold[take],means[take],candidate[take],direct[take],maximum)
        for name in ['coverage_delta','accuracy_delta_common','kappa_delta_common','mae_delta_common']:
            if np.isfinite(result[name]):
                values[name].append(result[name])
    return {'method':'paired_cluster_percentile_95','groups':len(unique),
            'intervals':{k:{'lower':np.quantile(v,.025),'upper':np.quantile(v,.975),'defined_replicates':len(v)} for k,v in values.items()}}


def aggregate(case: dict, units: list, runs: list) -> dict:
    train=[u for u in case['units'] if u['partition']=='train']
    result={}
    for name,scale in case['scores'].items():
        pred=scores(runs,name)
        means=np.array([np.mean([r['scores'][name] for r in u['ratings']]) for u in units])
        training=np.array([np.mean([r['scores'][name] for r in u['ratings']]) for u in train])
        valid=np.isfinite(pred)
        weights=np.array([1/sum(v['group_id']==u['group_id'] for v in units) for u in units])
        result[name]={'train_mean':training.mean(),'train_median':np.median(training),
            'train_mean_rmse_paired':np.mean((training.mean()-means[valid])**2)**.5 if valid.any() else np.nan,
            'train_median_mae_paired':np.mean(np.abs(np.median(training)-means[valid])) if valid.any() else np.nan,
            'unit_weighted_human_mean':means.mean(),'equal_group_human_mean':np.average(means,weights=weights),
            'paired_unit_human_mean':means[valid].mean() if valid.any() else np.nan,
            'paired_group_human_mean':np.average(means[valid],weights=weights[valid]) if valid.any() else np.nan,
            'paired_unit_judge_mean':pred[valid].mean() if valid.any() else np.nan,
            'paired_group_judge_mean':np.average(pred[valid],weights=weights[valid]) if valid.any() else np.nan,
            'scale':scale}
    if set(case['scores'])=={'factuality','completeness','structure'}:
        f,c=[scores(runs,n)/2 for n in ['factuality','completeness']]
        valid=np.isfinite(f)&np.isfinite(c)
        hf,hc=[np.array([np.mean([r['scores'][n] for r in u['ratings']])/2 for u in units]) for n in ['factuality','completeness']]
        def harmonic(a: np.ndarray,b: np.ndarray) -> float:
            return 2*a.mean()*b.mean()/(a.mean()+b.mean()) if len(a) and a.mean()+b.mean()>0 else np.nan
        result['harmonic_factuality_completeness']={'joint_coverage':valid.mean(),
            'human_all':harmonic(hf,hc),'human_paired':harmonic(hf[valid],hc[valid]),
            'judge_paired':harmonic(f[valid],c[valid]),'weighting':'equal response units, panel mean first; not verified original production KM'}
    return result


def main() -> None:
    selected=read(OUT/'test-selection.json')['selection']
    cases={p.stem:read(p) for p in (OUT/'cases').glob('*.json')}
    comparison=[]
    for ci,arm in selected.items():
        case=cases[ci]
        units=sorted([u for u in case['units'] if u['partition']=='test'],key=lambda u:u['unit_id'])
        candidate=runs_for(case,units,arm)
        direct=runs_for(case,units,'direct')
        for name,scale in case['scores'].items():
            gold=np.array([mode([r['scores'][name] for r in u['ratings']]) for u in units],dtype=float)
            means=np.array([np.mean([r['scores'][name] for r in u['ratings']]) for u in units])
            a,b=scores(candidate,name),scores(direct,name)
            comparison.append({'agent':ci,'selected':arm,'criterion':name,
                **compare(gold,means,a,b,max(scale)),
                'uncertainty':uncertainty(gold,means,a,b,[u['group_id'] for u in units],max(scale)),
                'aggregates':{arm:aggregate(case,units,candidate),'direct':aggregate(case,units,direct)}})
    (OUT/'paired-comparison.json').write_text(json.dumps(finite(comparison),ensure_ascii=False,indent=2))

    book=load_workbook('/Users/antonzyukov/laim/b2c_agents_artifacts/CI09840650/data_for_val.xlsx',read_only=True,data_only=True)
    frequency={i:r[6] for i,r in enumerate(book.active.values,1) if 2<=i<=198}
    book.close()
    weighted=[]
    case=cases['CI09840650']
    for partition in ['dev','test']:
        units=sorted([u for u in case['units'] if u['partition']==partition],key=lambda u:u['unit_id'])
        for arm in dict.fromkeys(['direct',selected['CI09840650']]):
            pred=scores(runs_for(case,units,arm),'assessment_score')
            gold=np.array([u['ratings'][0]['scores']['assessment_score'] for u in units])
            weights=np.array([sum(frequency[row] for row in u['source_rows']) for u in units])
            valid=np.isfinite(pred)
            weighted.append({'partition':partition,'arm':arm,'units':len(units),'sum_count':int(weights.sum()),
                'largest_weight_share':float(weights.max()/weights.sum()),
                'coverage_weighted':np.average(valid,weights=weights),
                'accuracy_weighted_paired':np.average(pred[valid]==gold[valid],weights=weights[valid]) if valid.any() else np.nan,
                'mode_accuracy_weighted_paired':np.average(gold[valid]==1,weights=weights[valid]) if valid.any() else np.nan,
                'weighted_bias_paired':np.average(pred[valid]-gold[valid],weights=weights[valid]) if valid.any() else np.nan})
    (OUT/'frequency-sensitivity.json').write_text(json.dumps(finite(weighted),ensure_ascii=False,indent=2))

    stress=[]
    for item in read(OUT/'stress-manifest.json'):
        case=cases[item['agent']]
        unit=next(u for u in case['units'] if u['unit_id']==item['unit_id'])
        base=runs_for(case,[unit],item['base_arm'])[0]
        altered=read(OUT/'stress-runs'/f"{item['run_id']}.json")
        stress.append({**item,'base_scores':base.get('scores'),'altered_scores':altered.get('scores'),
            'same_scores':base.get('scores')==altered.get('scores') if base['status']=='ok' and altered['status']=='ok' else None})
    stress_counts=[]
    for ci in cases:
        for variant in ['repeat','pretty','temperature','injection']:
            items=[r for r in stress if r['agent']==ci and r['variant']==variant]
            stress_counts.append({'agent':ci,'variant':variant,'units':len(items),
                'changed':sum(r['same_scores'] is False for r in items),'errors':sum(r['same_scores'] is None for r in items)})
    (OUT/'stress-results.json').write_text(json.dumps({'details':stress,'summary':stress_counts},ensure_ascii=False,indent=2))
    provenance=Counter()
    basis_counts=Counter()
    for folder in ['runs','trace-runs','stress-runs']:
        for p in (OUT/folder).glob('*.json'):
            r=read(p)
            provenance[(folder,r['status'])]+=1
            if r['status']=='ok' and r['arm'].startswith('grounded'):
                basis=json.loads(r['response']['choices'][0]['message']['content']).get('basis',[])
                names=[b['criterion'] for b in basis]
                basis_counts['grounded_responses']+=1
                basis_counts['exactly_one_basis_per_criterion']+=set(names)==set(cases[r['agent']]['scores']) and len(names)==len(set(names))
    (OUT/'experiment-accounting.json').write_text(json.dumps({'requests':{str(k):v for k,v in provenance.items()},'basis_counts':basis_counts},indent=2))
    print('Сохранены парные сравнения, чувствительность весов и стресс-проверки.')


if __name__=='__main__':
    main()
