"""Ретроспективная проверка на свидетельствах того же пакета документов."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

OLD=Path('/Users/antonzyukov/laim-artifacts/judge-deep-research-20260911')
sys.path.insert(0,str(OLD))
from round3 import decoded_unit  # noqa: E402
from run_judge import api,build_request,digest,dumps,validate  # noqa: E402

OUT=Path(__file__).parent
SOURCE=Path('/Users/antonzyukov/laim-data-20260911/artifacts/CI10071259/baskets/agent_post_labeled_all_2025_11_27_v1.parquet')


def enrich(case: dict) -> tuple[dict,dict]:
    frame=pd.read_parquet(SOURCE,columns=['case_id','doc_request_id'])
    pools=defaultdict(dict)
    identities={}
    partitions=defaultdict(set)
    for u in case['units']:
        keys={(str(row.case_id),str(row.doc_request_id)) for _,row in frame.iloc[u['source_rows']].iterrows()}
        assert len(keys)==1
        key=digest(sorted(keys))
        identities[u['unit_id']]=key
        if not u['partition'].startswith('excluded'):
            partitions[key].add(u['partition'])
        for e in u['evidence']:
            content_key=digest([e['tool_name'],e['arguments'],e['content']])
            pools[key][content_key]={'evidence_id':'packet-'+content_key,'kind':'tool_result',
                'tool_name':e['tool_name'],'arguments':e['arguments'],'content':e['content'],
                'binding':'same_case_and_doc_request_retrospective_packet','packet_id':key}
    assert all(len(parts)==1 for parts in partitions.values())
    expanded=json.loads(json.dumps(case))
    for u in expanded['units']:
        key=identities[u['unit_id']]
        u['evidence']=[pools[key][k] for k in sorted(pools[key])]
    return expanded,{'packet_count':len(pools),'unique_tool_results':sum(map(len,pools.values())),
        'assumption':'Все связанные результаты доступны к моменту ретроспективной оценки пакета. Их доступность в момент исходного ответа и полнота пакета не утверждаются.',
        'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'unit_packets':identities}


def request_for(case: dict,unit: dict) -> dict:
    request=build_request(case,decoded_unit(unit),'grounded_ids_bm25_anchors_t8')
    request['messages'][0]['content']=request['messages'][0]['content'].replace('Tool results связаны с этим ответом и предшествуют ему, но не гарантируют полноту пакета документов.', 'Tool results относятся к тому же пакету документов, но могут происходить из других запусков и не гарантируют полноту пакета.')
    request['messages'][0]['content']+='\nОбласть свидетельств: объединены сохранённые результаты инструментов для того же кейса и того же запроса документов. Они собраны из нескольких вопросов по одному пакету. Это факты для ретроспективной оценки, но не гарантированно полный пакет. Не считай отсутствие находки в одном поиске отсутствием документа, если другой результат содержит нужные сведения. Чужие вопросы и ответы здесь не являются свидетельствами. Проверяй соответствие субъекта, типа документа и периода текущему вопросу.'
    return request


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--partition',choices=['dev','test'],default='dev')
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    case,audit=enrich(json.loads((OLD/'cases/CI10071259.json').read_text()))
    (OUT/'packet-evidence-audit.json').write_text(dumps(audit))
    units=sorted([u for u in case['units'] if u['partition']==args.partition],key=lambda u:u['unit_id'])
    if args.limit:
        units=units[:args.limit]
    for i,u in enumerate(units,1):
        request=request_for(case,u)
        identity=digest([request,u['unit_id'],0])
        path=OUT/'runs'/f'{identity}.json'
        if path.exists():
            record=json.loads(path.read_text())
        else:
            record={'agent':case['agent'],'arm':'packet_evidence','unit_id':u['unit_id'],'group_id':u['group_id'],
                    'partition':u['partition'],'repeat':0,'request':request,'requested_at':time.time()}
            started=time.monotonic()
            try:
                response=api('chat/completions',request)
                record['response']=response
                if response['choices'][0]['finish_reason']!='stop':
                    raise ValueError('Генерация не завершилась stop')
                parsed=json.loads(response['choices'][0]['message']['content'])
                scores,basis=validate(case,decoded_unit(u),parsed)
                record.update(status='ok',scores=scores,parsed=parsed,audit=basis)
            except (RuntimeError,ValueError,KeyError,TypeError) as error:
                record.update(status='error',error=str(error))
            record['seconds']=round(time.monotonic()-started,3)
            path.parent.mkdir(exist_ok=True)
            path.write_text(dumps(record))
        print(dumps({'unit':i,'total':len(units),'arm':'packet_evidence','status':record['status'],'scores':record.get('scores'),'error':record.get('error'),'seconds':record['seconds']}),flush=True)
        if 'HTTP 40' in record.get('error',''):
            raise RuntimeError(record['error'])


if __name__=='__main__':
    main()
