"""Обычный текст вспомогательного проекта: конечные баллы остаются строго типизированы."""
from __future__ import annotations

import hashlib
import json
import re
import time
from unittest.mock import patch

import reference_ids as ids
import reference_round as reference


def request_for(unit: dict) -> dict:
    request=ids.request_for(unit)
    request.pop('response_format')
    request['messages'][0]['content']=request['messages'][0]['content'].split('Верни JSON')[0]+'''Дай обычный текст проекта ответа, затем кратко перечисли ограничения источников.
Для существенных утверждений укажи ссылки вида [E0] или [E1]. Не печатай JSON или поля схемы.
Если данных недостаточно, так и напиши. Не выставляй баллы и не оценивай другой ответ.'''
    return request


def draft_from_text(unit: dict,text: str) -> dict:
    if not isinstance(text,str) or not text.strip():
        raise ValueError('Проект ответа пуст')
    mapping={f'E{i}':e['evidence_id'] for i,e in enumerate(unit['evidence'])}
    cited=sorted(set(re.findall(r'\[(E\d+)\]',text)))
    if not set(cited)<=set(mapping):
        raise ValueError('Текстовый проект ссылается на неизвестный источник')
    return {'draft_answer':text,'source_index':mapping,'cited_sources':cited,
        'source_references_present':bool(cited),'is_gold':False}


def generate(unit: dict) -> dict:
    request=request_for(unit)
    identity=hashlib.sha256(reference.sdk.canonical([request,unit['unit_id']]).encode()).hexdigest()
    path=reference.sdk.OUT/'reference-runs'/f'{identity}.json'
    if path.exists():
        return json.loads(path.read_text())
    result={'unit_id':unit['unit_id'],'partition':unit['partition'],'arm':'reference_text','request':request,'requested_at':time.time()}
    started=time.monotonic()
    try:
        response=reference.api('chat/completions',request)
        result['response']=response
        choice=response['choices'][0]
        if choice['finish_reason']!='stop':
            raise ValueError('Текстовый проект не завершён нормально')
        result.update(status='ok',parsed=draft_from_text(unit,choice['message']['content']))
    except (RuntimeError,ValueError,KeyError,TypeError) as error:
        result.update(status='error',error=str(error))
    result['seconds']=round(time.monotonic()-started,3)
    path.parent.mkdir(exist_ok=True)
    path.write_text(reference.sdk.canonical(result))
    if any(f'HTTP {code}' in result.get('error','') for code in [401,402,403]):
        raise RuntimeError('Недоступна авторизация или квота генерации проекта')
    return result


if __name__=='__main__':
    fixture={'evidence':[{'evidence_id':'packet-original'}]}
    assert draft_from_text(fixture,'Факт [E0]')['cited_sources']==['E0']
    for text in ['', 'Факт [E999]']:
        try:
            draft_from_text(fixture,text)
        except ValueError:
            pass
        else:
            raise AssertionError('Пустой проект или неизвестная ссылка приняты')
    reference.ARM='roles_schema_reference_text_no_examples_ultra'
    with patch.object(reference,'request_for',request_for),patch.object(reference,'generate',generate):
        reference.main()
