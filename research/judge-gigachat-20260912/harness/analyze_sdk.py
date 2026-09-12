"""Метрики production-цепочки: все единицы, отказы, повторы SDK и согласие."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

OLD=Path('/Users/antonzyukov/laim-artifacts/judge-deep-research-20260911')
sys.path.insert(0,str(OLD))
from analyze_results import alpha,finite,mode,summarize  # noqa: E402
from compare_round3 import compare  # noqa: E402

OUT=Path(__file__).parent


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--bootstrap',action='store_true')
    args=parser.parse_args()
    cases={p.stem:json.loads(p.read_text()) for p in (OLD/'cases').glob('*.json')}
    cases['CI10071255']=json.loads((OUT/'CI10071255-structure.json').read_text())
    cases['CI09877398']=json.loads((OUT/'CI09877398.json').read_text())
    clusters=json.loads((OUT/'client-overlap-audit.json').read_text())['evaluation_cluster_by_unit']
    for unit in cases['CI10071259']['units']:
        unit['group_id']=clusters[unit['unit_id']]
    online_clusters=json.loads((OUT/'online-entity-overlap.json').read_text())['component_by_unit']
    for unit in cases['CI10071255']['units']:
        unit['group_id']=online_clusters[unit['unit_id']]
    records=[json.loads(p.read_text()) for p in (OUT/'sdk-runs').glob('*.json')]
    records=[r for r in records if r['partition'] in ['dev','test']]
    rows=[]
    predictions=[]
    for agent,arm,partition in sorted({(r['agent'],r['arm'],r['partition']) for r in records}):
        case=cases[agent]
        units=sorted([u for u in case['units'] if u['partition']==partition],key=lambda u:u['unit_id'])
        selected=[r for r in records if (r['agent'],r['arm'],r['partition'])==(agent,arm,partition)]
        by_id={r['unit_id']:r for r in selected}
        assert len(by_id)==len(selected)
        if any(u['unit_id'] not in by_id for u in units):
            print(agent,arm,partition,'pending',len(units)-len(selected))
            continue
        ordered=[by_id[u['unit_id']] for u in units]
        models=Counter()
        reasons=Counter()
        tokens=0
        reasoning_responses=0
        empty_final_responses=0
        call_errors=Counter()
        for r in ordered:
            call_errors.update(e['type'] for e in r.get('call_errors',[]))
            for response in r['responses']:
                for group in response['generations']:
                    for generation in group:
                        metadata=generation['response_metadata']
                        reasoning_responses+=bool(generation.get('additional_kwargs',{}).get('reasoning_content'))
                        empty_final_responses+=not bool(generation.get('content'))
                        models[metadata.get('model_name','unknown')]+=1
                        reasons[metadata.get('finish_reason','unknown')]+=1
                        tokens+=metadata.get('token_usage',{}).get('total_tokens',0) or (generation.get('usage_metadata') or {}).get('total_tokens',0)
        requested=set(ordered[0]['request']['schema']['properties'])
        assert all(set(r['request']['schema']['properties'])==requested for r in ordered)
        for criterion in [c for c in case['scores'] if c in requested]:
            row=summarize(case,ordered,criterion,args.bootstrap)
            h=np.array([mode([r['scores'][criterion] for r in u['ratings']]) for u in units],dtype=float)
            p=np.array([r['scores'][criterion] if r['status']=='ok' else np.nan for r in ordered],dtype=float)
            valid=np.isfinite(h)&np.isfinite(p)
            row['balanced_accuracy_all']=float(np.mean([np.mean(p[h==value]==value) for value in set(h[np.isfinite(h)])]))
            row.update(agent=agent,arm=arm,partition=partition,
                alpha_ordinal=alpha(np.array([h[valid],p[valid]]),'ordinal') if valid.sum()>1 else None,
                models=dict(models),finish_reasons=dict(reasons),total_tokens=tokens,
                response_attempts=sum(len(r['responses']) for r in ordered),
                reasoning_responses=reasoning_responses,empty_final_responses=empty_final_responses,
                call_errors=dict(call_errors),records_without_error_log=sum('call_errors' not in r for r in ordered),
                latency_seconds_p50=float(np.percentile([r['seconds'] for r in ordered],50)),
                latency_seconds_p95=float(np.percentile([r['seconds'] for r in ordered],95)))
            rows.append(row)
            predictions.append({'agent':agent,'partition':partition,'arm':arm,'samples':1,'method':'sdk_raw',
                'criterion':criterion,'unit_ids':[u['unit_id'] for u in units],
                'groups':[u['group_id'] for u in units],'gold':h,
                'panel_mean':[float(np.mean([r['scores'][criterion] for r in u['ratings']])) for u in units],
                'prediction':p})
            print(agent,arm,partition,criterion,json.dumps(finite({k:row[k] for k in ['valid_scores','accuracy','baseline_accuracy_paired','cohen_kappa','alpha_ordinal','correct_label_yield','recall_all_defects','fpr_paired']})))
    (OUT/'sdk-metrics.json').write_text(json.dumps(finite(rows),ensure_ascii=False,indent=2))
    saved=finite(predictions)
    (OUT/'sdk-predictions.json').write_text(json.dumps(saved,ensure_ascii=False,indent=2))
    if args.bootstrap:
        references=[r for r in json.loads((OUT/'predictions.json').read_text()) if r['arm']=='grounded_ids_bm25_anchors_t8']
        pairs=[]
        for row in saved:
            if row['agent']!='CI10071259' or row['partition']!='dev':
                continue
            reference=next(r for r in references if r['criterion']==row['criterion'])
            pairs.append(compare(row,reference))
        (OUT/'sdk-paired-comparisons.json').write_text(json.dumps(finite(pairs),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
