"""Полные метрики зафиксированного каскада и парная проверка исходного judge."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

OUT=Path(__file__).parent
OLD=Path('/Users/antonzyukov/laim-artifacts/judge-deep-research-20260911')
sys.path.insert(0,str(OLD))
from analyze_results import alpha,finite,mode,summarize  # noqa: E402
from compare_round3 import compare  # noqa: E402


def vector(case: dict,records: list[dict],criterion: str,arm: str) -> dict:
    lookup={u['unit_id']:u for u in case['units']}
    units=[lookup[r['unit_id']] for r in records]
    return {'arm':arm,'samples':1,'method':'frozen_raw',
            'criterion':criterion,'unit_ids':[u['unit_id'] for u in units],
            'groups':[u['group_id'] for u in units],
            'gold':[mode([r['scores'][criterion] for r in u['ratings']]) for u in units],
            'panel_mean':[float(np.mean([r['scores'][criterion] for r in u['ratings']])) for u in units],
            'prediction':[r.get('scores',{}).get(criterion) if r['status']=='ok' else None for r in records]}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--partition',choices=['dev','test'],default='test')
    parser.add_argument('--bootstrap',action='store_true')
    parser.add_argument('--reference',choices=['legacy_function','grounded'],default='legacy_function')
    args=parser.parse_args()
    case=json.loads((OLD/'cases/CI10071259.json').read_text())
    original_groups={u['unit_id']:u['group_id'] for u in case['units']}
    clients=json.loads((OUT/'client-overlap-audit.json').read_text())
    for u in case['units']:
        u['group_id']=clients['evaluation_cluster_by_unit'][u['unit_id']]
    units=sorted([u for u in case['units'] if u['partition']==args.partition],key=lambda u:u['unit_id'])
    rows=[json.loads(p.read_text()) for p in (OUT/'cascade-runs').glob(f'{args.partition}-*.json')]
    records={r['unit_id']:r for r in rows}
    assert len(records)==len(rows)
    if set(records)!={u['unit_id'] for u in units}:
        print('Ещё не завершено:',len(records),'/',len(units))
        return
    reference_arm=args.reference
    references=[json.loads(p.read_text()) for p in ((OLD/'runs') if reference_arm=='grounded' else (OUT/'sdk-runs')).glob('*.json')]
    base={r['unit_id']:r for r in references if r['agent']=='CI10071259' and r['partition']==args.partition and r['arm']==reference_arm}
    assert set(records)<=set(base)
    exposed=set(json.loads((OUT/'rubric-example-overlap.json').read_text())['affected_groups'])
    subsets={'full':units}
    if args.partition=='test':
        subsets['without_rubric_groups']=[u for u in units if original_groups[u['unit_id']] not in exposed]
    if args.partition=='test':
        client_groups=set(json.loads((OUT/'client-overlap-audit.json').read_text())['shared_client_test_groups'])
        subsets['without_rubric_and_seen_clients']=[u for u in units if original_groups[u['unit_id']] not in exposed|client_groups]
    summaries=[]
    pairs=[]
    for name,selected in subsets.items():
        ordered=[records[u['unit_id']] for u in selected]
        previous=[base[u['unit_id']] for u in selected]
        for c in case['scores']:
            for arm,runs in [('criterion_cascade',ordered),(reference_arm,previous)]:
                row=summarize(case,runs,c,args.bootstrap)
                v=vector(case,runs,c,arm)
                gold=np.array(v['gold'],dtype=float)
                pred=np.array(v['prediction'],dtype=float)
                valid=np.isfinite(gold)&np.isfinite(pred)
                row.update(arm=arm,subset=name,partition=args.partition,
                    alpha_ordinal=alpha(np.array([gold[valid],pred[valid]]),'ordinal'),
                    balanced_accuracy_all=float(np.mean([np.mean(pred[gold==x]==x) for x in set(gold[np.isfinite(gold)])])))
                severe=gold==min(case['scores'][c])
                normal_score=max(case['scores'][c])
                row.update(critical_detection_recall_all=float(np.mean(pred[severe]<normal_score)) if severe.any() else None,
                    critical_undergraded_defects=int(np.sum(severe&(pred>min(case['scores'][c]))&(pred<normal_score))),
                    critical_missed_or_unassessed=int(np.sum(severe&~(pred<normal_score))))
                counts=Counter(v['groups'])
                weights=np.array([1/counts[g] for g in v['groups']])
                assert np.isclose(weights.sum(),len(counts))
                known=np.isfinite(gold)
                row.update(group_sizes=dict(counts),
                    group_balanced_coverage=float(np.average(np.isfinite(pred),weights=weights)),
                    group_balanced_correct_label_yield=float(np.average((gold==pred)[known],weights=weights[known])) if known.any() else None,
                    group_balanced_kappa=float(cohen_kappa_score(gold[valid],pred[valid],sample_weight=weights[valid])) if valid.sum()>1 else None)
                summaries.append(row)
                print(name,arm,c,json.dumps(finite({k:row[k] for k in ['units','valid_scores','cohen_kappa','alpha_ordinal','accuracy','baseline_accuracy_all','correct_label_yield','recall_all_defects','score_zero_recall_all','fpr_paired']})))
            if args.bootstrap:
                pair=compare(vector(case,ordered,c,'criterion_cascade'),vector(case,previous,c,reference_arm))
                pair.update(subset=name,interpretation='Парный групповой bootstrap фиксированных прогнозов. Старый test ранее просмотрен; поиск кандидата не включён в неопределённость.')
                pairs.append(pair)
    suffix='' if reference_arm=='legacy_function' else '-grounded'
    (OUT/f'cascade-{args.partition}-validation{suffix}.json').write_text(json.dumps(finite(summaries),ensure_ascii=False,indent=2))
    if pairs:
        (OUT/f'cascade-{args.partition}-validation{suffix}-paired.json').write_text(json.dumps(finite(pairs),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
