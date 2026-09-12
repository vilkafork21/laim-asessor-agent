"""Отложенная проверка S-кандидата и исходного judge на одинаковых единицах."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_cascade import vector
from analyze_results import alpha, finite, summarize
from compare_round3 import compare

OUT=Path(__file__).parent
CANDIDATE='roles_schema_structure_observed_no_examples_max'
BASELINE='legacy_function'


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--bootstrap',action='store_true')
    args=parser.parse_args()
    case=json.loads((OUT/'CI10071255-structure.json').read_text())
    links=json.loads((OUT/'online-entity-overlap.json').read_text())
    for unit in case['units']:
        unit['group_id']=links['component_by_unit'][unit['unit_id']]
    units=sorted([u for u in case['units'] if u['partition']=='test'],key=lambda u:u['unit_id'])
    records=[json.loads(p.read_text()) for p in (OUT/'sdk-runs').glob('*.json')]
    lookup={}
    for arm in [CANDIDATE,BASELINE]:
        selected=[r for r in records if (r['agent'],r['partition'],r['arm'])==('CI10071255','test',arm)]
        lookup[arm]={r['unit_id']:r for r in selected}
        assert len(lookup[arm])==len(selected)
        if set(lookup[arm])!={u['unit_id'] for u in units}:
            print('Ещё не завершено:',arm,len(selected),'/',len(units))
            return
    affected=set(links['test_units_linked_to_train_or_dev'])
    subsets={'full199':units,'without_detected_links153':[u for u in units if u['unit_id'] not in affected]}
    assert len(subsets['full199'])==199 and len(subsets['without_detected_links153'])==153
    summaries=[]
    pairs=[]
    for name,subset in subsets.items():
        vectors={}
        for arm in [CANDIDATE,BASELINE]:
            ordered=[lookup[arm][u['unit_id']] for u in subset]
            row=summarize(case,ordered,'structure',args.bootstrap)
            value=vector(case,ordered,'structure',arm)
            gold=np.array(value['gold'],dtype=float)
            pred=np.array(value['prediction'],dtype=float)
            valid=np.isfinite(gold)&np.isfinite(pred)
            row.update(arm=arm,subset=name,partition='test',alpha_ordinal=alpha(np.array([gold[valid],pred[valid]]),'ordinal'),
                balanced_accuracy_all=float(np.mean([np.mean(pred[gold==score]==score) for score in set(gold[np.isfinite(gold)])])),
                critical_detection_recall_all=float(np.mean(pred[gold==0]<2)) if np.any(gold==0) else None)
            summaries.append(row)
            vectors[arm]=value
            print(name,arm,json.dumps(finite({k:row[k] for k in ['units','valid_scores','cohen_kappa','alpha_ordinal','accuracy','baseline_accuracy_all','correct_label_yield','recall_all_defects','score_zero_recall_all','fpr_paired','tp','fp']})))
        if args.bootstrap:
            pair=compare(vectors[CANDIDATE],vectors[BASELINE])
            pair.update(subset=name,interpretation='Парный bootstrap консервативных компонент по явным ИНН. Кандидат зафиксирован после dev и до test-вызовов. Неизвестные связи объектов и расхождения версий экспертной разметки остаются ограничениями.')
            pairs.append(pair)
    (OUT/'online-structure-test-validation.json').write_text(json.dumps(finite(summaries),ensure_ascii=False,indent=2))
    if pairs:
        (OUT/'online-structure-test-paired.json').write_text(json.dumps(finite(pairs),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
