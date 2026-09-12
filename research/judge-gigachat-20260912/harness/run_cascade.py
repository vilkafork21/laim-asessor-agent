"""Фиксированный каскад GigaChat: оценка не заменяется, если предыдущий судья её дал."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from pathlib import Path

import sdk_round as sdk
from analyze_round4 import align_parser
from round4 import evaluate

OUT=Path(__file__).parent
OLD=sdk.OLD
COMMENTS='contrastive_comments_aligned_parser'
MAXIMUM='roles_schema_giga2_max'
PACKET='roles_schema_packet_no_examples'
ROUTES={'factuality':[MAXIMUM,COMMENTS,PACKET],
        'completeness':[MAXIMUM,COMMENTS,PACKET],'structure':[COMMENTS,PACKET]}


def merge(records: dict) -> tuple[dict,dict]:
    scores={c:None for c in ROUTES}
    selected={c:None for c in ROUTES}
    for criterion,order in ROUTES.items():
        for arm in order:
            record=records.get(arm,{})
            score=record.get('scores',{}).get(criterion) if record.get('status')=='ok' else None
            if score is not None:
                scores[criterion]=score
                selected[criterion]=arm
                break
    return scores,selected


def freeze() -> dict:
    paths=[OUT/n for n in ['run_cascade.py','sdk_round.py','round4.py','analyze_round4.py','packet_round.py']]
    paths += [OLD/'run_judge.py',OLD/'round3.py',sdk.NODE/'agent/asessor_agent.py',sdk.NODE/'agent/prompts.py']
    config={'agent':'CI10071259','routes':ROUTES,'selection_partition':'dev',
            'validation':'Полный прежний test 201 и отдельный срез без раскрытых рубрикой групп 181; внутренняя валидация.',
            'selection':'Пять заранее перечисленных dev-replay каскадов; Max выбран для фактичности/полноты, reviewer с train-feedback для структуры. Переключение только при отсутствии оценки, без текущих человеческих меток.',
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    path=OUT/'frozen-cascade.json'
    if path.exists():
        assert json.loads(path.read_text())==config,'Код или правило каскада изменились после фиксации'
    else:
        path.write_text(json.dumps(config,ensure_ascii=False,indent=2))
    return config


def sdk_records(partition: str,arm: str) -> dict:
    rows=[json.loads(p.read_text()) for p in (OUT/'sdk-runs').glob('*.json')]
    rows=[r for r in rows if r['agent']=='CI10071259' and r['partition']==partition and r['arm']==arm]
    result={r['unit_id']:r for r in rows}
    assert len(result)==len(rows),'Повторные версии SDK-запроса'
    return result


async def run(partition: str) -> None:
    freeze()
    case=json.loads((OLD/'cases/CI10071259.json').read_text())
    units=sorted([u for u in case['units'] if u['partition']==partition],key=lambda u:u['unit_id'])
    first={}
    for i,u in enumerate(units,1):
        raw=evaluate(case,u,'contrastive_comments')
        first[u['unit_id']]=align_parser(case,u,raw)
        print(json.dumps({'stage':'comments','unit':i,'total':len(units),'status':first[u['unit_id']]['status']},ensure_ascii=False),flush=True)
        if 'HTTP 40' in raw.get('error',''):
            raise RuntimeError('Остановлено: авторизация/квота, ответ сохранён')
    await sdk.run(case,[MAXIMUM],partition,0)
    second=sdk_records(partition,MAXIMUM)
    needed={u['unit_id'] for u in units if any(v is None for v in merge({COMMENTS:first[u['unit_id']],MAXIMUM:second[u['unit_id']]})[0].values())}
    if needed:
        filtered=json.loads(json.dumps(case))
        for u in filtered['units']:
            if u['partition']==partition and u['unit_id'] not in needed:
                u['partition']='excluded-cascade'
        await sdk.run(filtered,[PACKET],partition,0)
    third=sdk_records(partition,PACKET)
    folder=OUT/'cascade-runs'
    folder.mkdir(exist_ok=True)
    for u in units:
        uid=u['unit_id']
        stages={COMMENTS:first[uid],MAXIMUM:second[uid]}
        if uid in needed:
            stages[PACKET]=third[uid]
        scores,selected=merge(stages)
        record={'agent':case['agent'],'unit_id':uid,'group_id':u['group_id'],'partition':partition,
                'arm':'criterion_cascade','status':'ok' if any(v is not None for v in scores.values()) else 'error',
                'scores':scores,'selected_stage':selected,'stage_status':{a:r['status'] for a,r in stages.items()},
                'fallback_requested':uid in needed,'config_sha256':hashlib.sha256((OUT/'frozen-cascade.json').read_bytes()).hexdigest()}
        (folder/f'{partition}-{uid}.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    print(json.dumps({'partition':partition,'completed_units':len(units),'packet_fallback_units':len(needed)}),flush=True)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--partition',choices=['dev','test'],default='test')
    parser.add_argument('--freeze-only',action='store_true')
    args=parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    assert merge({MAXIMUM:{'status':'ok','scores':{'factuality':0}},COMMENTS:{'status':'ok','scores':{'factuality':2,'structure':1}}})[0]=={'factuality':0,'completeness':None,'structure':1}
    freeze()
    if not args.freeze_only:
        asyncio.run(run(args.partition))


if __name__=='__main__':
    main()
