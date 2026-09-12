"""Более сильный контроль, чем мода: пять ближайших train-примеров через тот же BM25."""
from __future__ import annotations

import json
from collections import Counter

import sdk_round as sdk
from analyze_results import finite, mode, summarize


def vote(values: list[int],prior: Counter) -> int:
    counts=Counter(values)
    return max(counts,key=lambda value:(counts[value],prior[value],-value))


def main() -> None:
    assert vote([0,1,2,2,1],Counter({2:10,1:5,0:3}))==2
    assert vote([0,0,1,1,2],Counter({2:10,1:5,0:3}))==1
    case=json.loads((sdk.OUT/'CI10071255-structure.json').read_text())
    links=json.loads((sdk.OUT/'online-entity-overlap.json').read_text())
    for unit in case['units']:
        unit['group_id']=links['component_by_unit'][unit['unit_id']]
    train=[]
    for unit in sorted(case['units'],key=lambda u:u['unit_id']):
        if unit['partition']!='train':
            continue
        score=mode([r['scores']['structure'] for r in unit['ratings']])
        if score is not None:
            train.append({'question':sdk._serialize_llm_record({'assessment_context':sdk.context_for(unit)}),
                'score':score,'unit_id':unit['unit_id']})
    assert len(train)==569
    prior=Counter(e['score'] for e in train)
    retriever=sdk.BM25Retriever(None,train)
    runs=[]
    for unit in sorted(case['units'],key=lambda u:u['unit_id']):
        if unit['partition'] not in ['dev','test']:
            continue
        nearest=retriever.hybrid_search(query=sdk._serialize_llm_record({'assessment_context':sdk.context_for(unit)}),k=5)
        assert all(e['unit_id']!=unit['unit_id'] for e in nearest)
        runs.append({'unit_id':unit['unit_id'],'partition':unit['partition'],'arm':'bm25_5nn_train_prior',
            'status':'ok','scores':{'structure':vote([e['score'] for e in nearest],prior)},'neighbors':[e['unit_id'] for e in nearest]})
    summaries=[]
    affected=set(links['test_units_linked_to_train_or_dev'])
    for partition in ['dev','test']:
        for name in ['full','without_detected_links'] if partition=='test' else ['full']:
            selected=[r for r in runs if r['partition']==partition and (name=='full' or r['unit_id'] not in affected)]
            row=summarize(case,selected,'structure',False)
            row.update(partition=partition,subset=name,arm='bm25_5nn_train_prior')
            summaries.append(row)
            print(partition,name,json.dumps(finite({k:row[k] for k in ['units','accuracy','baseline_accuracy_all','cohen_kappa','recall_all_defects','fpr_paired']})))
    (sdk.OUT/'online-lexical-baseline.json').write_text(json.dumps(finite({'method':'k=5 заранее фиксировано; тот же BM25 по полному видимому контексту; мода соседей, ничьи разрешаются частотой train, затем меньшим баллом; не является предлагаемым GigaChat judge.','summaries':summaries,'runs':runs}),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
