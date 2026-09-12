"""Отдельная проверка структуры 1000 заключений без вымышленных фактов операций."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import docx2txt
import pandas as pd

from packet_round import OLD,OUT
from run_judge import digest,dumps

ROOT=Path('/Users/antonzyukov/Downloads/CI10071255_303777')
SOURCE=ROOT/'agent_online_scoring_1000cases_assesment.xlsx'
INSTRUCTION=ROOT/'judge_promt_docx.docx'


def normalized(text: str) -> str:
    return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',text)).strip().casefold()


def main() -> None:
    frame=pd.read_excel(SOURCE)
    assert len(frame)==1000 and frame.id_oper.nunique()==1000 and frame.case_id_balancer.nunique()==1000
    assert frame.ai_message.notna().all()
    prompt=docx2txt.process(INSTRUCTION)
    rubric=prompt.split('КРИТЕРИЙ 3. СТРУКТУРИРОВАННОСТЬ (0–2)',1)[1].split('==========================================',1)[0].strip()
    assert all(f'{n} —' in rubric for n in range(3))
    old_answers=set()
    for path in (OLD/'cases').glob('*.json'):
        case=json.loads(path.read_text())
        for unit in case['units']:
            context=unit['context']
            for turn in [context.get('current_turn',{}),*context.get('turns',[])]:
                answer=turn.get('output_answer')
                if isinstance(answer,str):
                    old_answers.add(normalized(answer))
    units=[]
    for index,row in frame.iterrows():
        answer=str(row.ai_message)
        group=digest(['CI10071255',normalized(answer)])
        bucket=int(group[:8],16)%100
        partition='train' if bucket<60 else ('dev' if bucket<80 else 'test')
        if normalized(answer) in old_answers:
            partition='excluded-old-answer'
        ratings=[]
        for slot in range(3):
            column='Структурированный формат ответа '+('' if slot==0 else f'.{slot}')
            score=float(row[column])
            assert score in [0,1,2]
            ratings.append({'rater_id':f'anonymous_panel_slot_{slot}','scores':{'structure':score}})
        units.append({'unit_id':digest(['CI10071255',str(row.id_oper)]),'group_id':group,
            'partition':partition,'source_rows':[int(index)],'evidence':[],
            'context':{'mode':'qa','current_turn':{'input_query':None,'output_answer':answer}},'ratings':ratings})
    assert len({u['unit_id'] for u in units})==len(units)
    groups={part:{u['group_id'] for u in units if u['partition']==part} for part in ['train','dev','test']}
    assert not (groups['train']&groups['dev'] or groups['train']&groups['test'] or groups['dev']&groups['test'])
    case={'agent':'CI10071255','rubric':'КРИТЕРИЙ: СТРУКТУРИРОВАННОСТЬ (0–2)\n'+rubric,
        'target':'Оцени только структуру сохранённого комплаенс-заключения: логику изложения, внутренние противоречия, повторы и технические артефакты по приведённой шкале. Это отдельный критерий structure. Исходный текст запроса и факты операции здесь не предоставлены: внешняя фактическая корректность и полнота причин решения не оцениваются.',
        'scores':{'structure':[0,1,2]},'units':units,'gold_kind':'three_anonymous_human_ratings',
        'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE,INSTRUCTION]},
        'limitations':'Проверяется только структура. Инструкция взята из найденного judge prompt; отдельная подробная инструкция экспертов не установлена. Уникальные операции не доказывают уникальность клиентов. Имена экспертов отсутствуют; слоты панели не являются установленными личностями.'}
    (OUT/'CI10071255-structure.json').write_text(dumps(case))
    audit={'source_hashes':case['source_hashes'],'units':len(units),'partitions':dict(Counter(u['partition'] for u in units)),
        'groups_by_partition':{p:len(g) for p,g in groups.items()},'selection':'SHA256 нормализованного ответа, 60/20/20; метки не участвуют в выборе раздела; точные нормализованные повторы объединены; совпадения с ответами старых шести корпусов исключаются.',
        'limitations':case['limitations']}
    (OUT/'CI10071255-structure-inventory.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print('PASS: 1000 исходных ответов, три голоса, шкала и разбиение без пересечений; API не вызывается')
    print(audit['partitions'],audit['groups_by_partition'])


if __name__=='__main__':
    main()
