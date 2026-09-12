"""Контроль отдельной оценки структуры на Max через исходную SDK-цепочку."""
from __future__ import annotations

import argparse
import asyncio
import json
from unittest.mock import patch

from pydantic import ConfigDict, Field, create_model

import sdk_round as sdk

ARM='roles_schema_structure_only_max'
REASON_ARM='roles_schema_structure_reason_max'
FOCUSED_ARM='roles_schema_structure_focused_max'
FOCUSED_REASON_ARM='roles_schema_structure_focused_reason_max'
sdk.MODELS[ARM]='GigaChat-2-Max'
sdk.MODELS[REASON_ARM]='GigaChat-2-Max'
sdk.MODELS[FOCUSED_ARM]='GigaChat-2-Max'
sdk.MODELS[FOCUSED_REASON_ARM]='GigaChat-2-Max'
ORIGINAL_MAKE_JUDGE=sdk.make_judge


def make_reason_judge(case: dict,arm: str) -> tuple[sdk.Asessor,sdk.Recorder]:
    from agent.prompts import SYSTEM_PROMPT
    prompt=SYSTEM_PROMPT.split('<Формат ответа>')[0]+'''<Формат ответа>
Верни плоский JSON: сначала assessment_reason, затем structure.
assessment_reason — краткое проверяемое основание оценки структуры по рубрике.
Если есть нарушение, назови конкретный фрагмент ответа и применимое правило или тэг.
Подробный ход рассуждений не нужен. structure — значение допустимой шкалы либо
not_assessable, если именно структуру невозможно проверить. Не добавляй другие поля.'''
    with patch('agent.asessor_agent.SYSTEM_PROMPT',prompt):
        judge,recorder=ORIGINAL_MAKE_JUDGE(case,arm)
    judge._output_model=create_model('StructureWithReason',__config__=ConfigDict(extra='forbid'),
        assessment_reason=(str,Field(...,description='Краткое проверяемое основание по исходной рубрике')),
        structure=(judge._output_model.model_fields['structure'].rebuild_annotation(),Field(...,description='Оценка структуры по исходной шкале')))
    judge.agent_chain=judge.printing_chain|judge.llm.with_structured_output(judge._output_model,method='json_schema',strict=True)
    return judge,recorder


def prepare(focused: bool=False) -> dict:
    case=json.loads((sdk.OLD/'cases/CI10071259.json').read_text())
    for unit in case['units']:
        if unit['partition']=='train' and any(sdk.mode([r['scores'][c] for r in unit['ratings']]) is None for c in case['scores']):
            unit['partition']='excluded-incomplete-training'
    if focused:
        rubric=case['rubric']
        definition=rubric.split('Структурированный формат ответа',1)[1].split('ТЭГи',1)[0].strip()
        tags=rubric.split('ТЭГи',1)[1]
        tags=tags[tags.index('#Кратко'):].split('КАК ВЫГЛЯДИТ ТАБЛИЦА',1)[0].strip()
        assert definition in rubric and tags in rubric
        assert all(tag in tags for tag in ['#Кратко','#Излишне','#Повтор','#НетЛогики'])
        case['rubric']='Структурированный формат ответа\n'+definition+'\n\n'+tags
        for unit in case['units']:
            unit['evidence']=[]
    case['scores']={'structure':case['scores']['structure']}
    target='Оцени только критерий structure по исходной рубрике. Фактичность и полнота отдельно не оцениваются; не копируй их предполагаемый балл в structure.'
    case['target']=target if focused else case['target']+'\nВ этом запросе оцени только критерий structure по исходной рубрике. Фактичность и полнота отдельно не оцениваются; не копируй их предполагаемый балл в structure.'
    return case


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--partition',choices=['dev','test'],default='dev')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--reason',action='store_true')
    parser.add_argument('--focused',action='store_true')
    args=parser.parse_args()
    case=prepare(args.focused)
    arm=(FOCUSED_REASON_ARM if args.reason else FOCUSED_ARM) if args.focused else (REASON_ARM if args.reason else ARM)
    if args.run:
        with patch.object(sdk,'make_judge',make_reason_judge if args.reason else ORIGINAL_MAKE_JUDGE):
            asyncio.run(sdk.run(case,[arm],args.partition,0))
        return
    if args.focused:
        assert 'три независимые' not in case['target']
    original=json.loads((sdk.OLD/'cases/CI10071259.json').read_text())
    expected={u['unit_id'] for u in original['units'] if u['partition']=='train' and all(sdk.mode([r['scores'][c] for r in u['ratings']]) is not None for c in original['scores'])}
    assert expected=={u['unit_id'] for u in case['units'] if u['partition']=='train'}
    assert [(u['context'],u['ratings']) for u in original['units']]==[(u['context'],u['ratings']) for u in case['units']]
    assert [u['evidence'] for u in case['units']]==([[] for _ in case['units']] if args.focused else [u['evidence'] for u in original['units']])
    judge,_=sdk.make_judge(case,ARM)
    assert set(judge._output_model.model_json_schema()['properties'])=={'structure'}
    assert all(set(json.loads(e['answer']))=={'structure'} for e in judge.examples_retriever.examples)
    reason_judge,_=make_reason_judge(case,REASON_ARM)
    assert list(reason_judge._output_model.model_json_schema()['properties'])==['assessment_reason','structure']
    assert reason_judge._output_model.model_validate({'assessment_reason':'Нарушение по рубрике','structure':'0'}).structure==0
    print('PASS: ответы/голоса/train-пул сохранены; один критерий; основание сохраняет ноль; focused использует только исходные определения структуры и исключает внешние факты; API не вызывается')


if __name__=='__main__':
    main()
