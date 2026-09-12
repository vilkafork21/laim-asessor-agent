"""Дополнительные размеченные ответы: только проверенные пакеты и test-группы."""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

from packet_round import SOURCE
from prepare_cases import SCORE_COLUMNS,canonical,digest

OUT=Path(__file__).parent
OLD=Path('/Users/antonzyukov/laim-artifacts/judge-deep-research-20260911')


def main() -> None:
    case=json.loads((OLD/'cases/CI10071259.json').read_text())
    frame=pd.read_parquet(SOURCE)
    oldids={u['unit_id'] for u in case['units']}
    def normalized_qa(context: dict) -> tuple[str,str]:
        return tuple(re.sub(r'\s+',' ',unicodedata.normalize('NFKC',context['current_turn'][k])).strip().casefold() for k in ['input_query','output_answer'])

    oldqa={normalized_qa(u['context']) for u in case['units']}
    packets=set()
    groups={}
    exposed=set(json.loads((OUT/'rubric-example-overlap.json').read_text())['affected_groups'])
    client_groups=set(json.loads((OUT/'client-overlap-audit.json').read_text())['shared_client_test_groups'])
    allowed={u['group_id'] for u in case['units'] if u['partition']=='test'}-exposed-client_groups
    for u in case['units']:
        for _,row in frame.iloc[u['source_rows']].iterrows():
            packets.add((str(row.case_id),str(row.doc_request_id)))
            groups[str(row.case_id)]=u['group_id']
    counts=Counter()
    extra=[]
    for key,rows in frame.groupby(['case_id','doc_request_id','question_id','question','answer'],sort=False,dropna=False):
        if digest(key) in oldids:
            continue
        ctx={'mode':'qa','current_turn':{'input_query':key[3],'output_answer':key[4]}}
        if normalized_qa(ctx) in oldqa:
            counts['normalized_old_qa']+=1
            continue
        if (str(key[0]),str(key[1])) not in packets:
            counts['no_verified_packet']+=1
            continue
        if groups[str(key[0])] not in allowed:
            counts['train_dev_rubric_or_shared_client_group']+=1
            continue
        ratings=[{'rater_id':digest(r['ID Эксперта']),'scores':{c:float(r[col]) for c,col in SCORE_COLUMNS.items()}}
                 for _,r in rows.iterrows() if r[list(SCORE_COLUMNS.values())].notna().all()]
        if not ratings:
            counts['no_complete_rating']+=1
            continue
        if len({r['rater_id'] for r in ratings})!=len(ratings):
            counts['duplicate_rater']+=1
            continue
        assert all(v in [0,1,2] for r in ratings for v in r['scores'].values())
        extra.append({'unit_id':digest(key),'group_id':groups[str(key[0])],'partition':'fresh',
                      'context':ctx,'source_rows':list(map(int,rows.index)),'ratings':ratings,'evidence':[],
                      'binding_note':'Исходный trace отсутствует/не совпадает: его содержание не используется. Доступны только ранее проверенные результаты того же пакета на запасном этапе.'})
    assert len({normalized_qa(u['context']) for u in extra})==len(extra)
    assert not (set(u['unit_id'] for u in extra)&oldids)
    assert not (set(u['group_id'] for u in extra)&{u['group_id'] for u in case['units'] if u['partition'] in ['train','dev']})
    for u in case['units']:
        if u['partition']=='test':
            u['partition']='excluded-fresh'
    case['units']+=extra
    audit={'new_units':len(extra),'groups':len({u['group_id'] for u in extra}),'excluded':dict(counts),
           'selection':'Новые QA после консервативной нормализации NFKC/регистра/пробелов, проверенный тот же пакет, test-группы вне рубрики и без общего клиента с train/dev; значения оценок не использованы для отбора.',
           'interpretation':'Новые ответы из уже известных test-групп; не новые независимые клиенты. Неполный исходный trace — отдельный сценарий переноса.',
           'candidates':[{'unit_id':u['unit_id'],'group_id':u['group_id'],'source_rows':u['source_rows']} for u in extra]}
    (OUT/'CI10071259-unlinked.json').write_text(canonical(case))
    (OUT/'unlinked-candidate-inventory.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k!='candidates'},ensure_ascii=False))


if __name__=='__main__':
    main()
