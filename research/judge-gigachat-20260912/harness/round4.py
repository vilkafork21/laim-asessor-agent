"""Семантика свидетельств и контрастные train-примеры; только GigaChat."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PREVIOUS=Path('/Users/antonzyukov/laim-artifacts/judge-deep-research-20260911')
sys.path.insert(0,str(PREVIOUS))
from analyze_results import mode  # noqa: E402
from round3 import decoded_unit  # noqa: E402
from run_judge import api,build_request,digest,dumps,reference_indices,validate  # noqa: E402
from retriever.retriever import EnhancedBM25  # noqa: E402

OUT=Path(__file__).parent
ARMS=['semantic','contrastive','contrastive_comments']
RULES='''Проверяй смысл фактов, а не только совпадение строк или наличие реквизита.
Сначала установи, какую информацию просит вопрос: субъект, роль, вид документа или операции, период. Для основных утверждений ответа проверь, относится ли свидетельство именно к этим условиям.
Совпавшая сумма в другом типе документа не подтверждает запрошенный показатель. Поступление дохода, оплата услуг, перевод собственных средств и зарплата — разные назначения операций; связь должна быть доказана источником. Не делай такую связь только по совпадению имени или способа перевода.
Пустая выдача инструмента означает, что этот поиск ничего не нашёл. Она не доказывает отсутствие документа во всём пакете. Учитывай, что источники могут быть неполными: при невозможности проверки внешнего факта используй not_assessable, а не считай отсутствие доказательства доказанной ложью.
Отличай эту неопределённость от наблюдаемой ошибки рассуждения: подмена требуемого документа другим, ложное отождествление ролей, внутренне противоречивые числа и необоснованный безусловный вывод могут быть видны прямо в имеющихся данных.
Полнота определяется выполнением конкретного запроса. Вежливое обещание, совет поискать документы и перечисление нерелевантных сведений не заменяют ответ по существу. Сообщение о реальном отсутствии требуемого документа может быть корректным ответом, если отсутствие подтверждено достаточными сведениями.
Структура включает логику и непротиворечивость по исходной рубрике, а не только наличие списков. Не переноси автоматически балл одного критерия на остальные.
Ссылки на context.current_turn.output_answer показывают только то, что сказал оцениваемый агент; они не доказывают внешний факт. Метаданные инструмента тоже не являются документом.
В поле claim_checks укажи не более шести решающих наблюдений: идентификаторы фрагментов ответа, результат проверки, ссылки на содержимое инструментов и краткое основание. Это проверяемые основания, не подробный ход рассуждения. Если содержательных утверждений нет, список может быть пустым. Затем верни basis и баллы по исходным критериям. Примеры относятся только к своим клиентам и не являются справочником текущего объекта.'''


def question_tokens(unit: dict) -> list[str]:
    text=unit['context']['current_turn']['input_query']
    text=re.sub(r'^Ты анализируешь кейс по клиенту .*?\(ИНН\s+\d+\)\.\s*','',text,flags=re.S)
    return re.findall(r'\w+',text.lower())


def examples(case: dict,unit: dict,comments: bool) -> list[dict]:
    train=[u for u in case['units'] if u['partition']=='train']
    labels={u['unit_id']:{c:mode([r['scores'][c] for r in u['ratings']]) for c in case['scores']} for u in train}
    train=[u for u in train if all(v is not None for v in labels[u['unit_id']].values())]
    ranking=EnhancedBM25([question_tokens(u) for u in train]).get_scores(question_tokens(unit))
    order=sorted(range(len(train)),key=lambda i:(-ranking[i],train[i]['unit_id']))
    selected=[i for level in [0,1,2] for i in [j for j in order if labels[train[j]['unit_id']]['factuality']==level][:2]]
    feedback=json.loads((PREVIOUS/'train-feedback-CI10071259.json').read_text())['feedback'] if comments else {}
    result=[]
    for i in selected:
        u=train[i]
        item={'unit_id':u['unit_id'],'assessment_context':u['context'],'scores':labels[u['unit_id']]}
        if comments:
            item['human_feedback']=feedback[u['unit_id']]
        result.append(item)
    return result


def request_for(case: dict,unit: dict,arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError('Неизвестное плечо '+arm)
    unit=decoded_unit(unit)
    request=build_request(case,unit,'grounded_ids_bm25_anchors_t8')
    request['messages'][0]['content']+='\nПРОВЕРКА СЕМАНТИКИ СВИДЕТЕЛЬСТВ:\n'+RULES
    payload=json.loads(request['messages'][-1]['content'])
    payload['train_examples']=[] if arm=='semantic' else examples(case,unit,arm=='contrastive_comments')
    fragments={f'A{i}':text for i,text in enumerate(re.split(r'(?<=[.!?])\s+|\n+',unit['context']['current_turn']['output_answer'])) if text.strip()}
    payload['answer_fragments']=fragments
    request['messages'][-1]['content']=dumps(payload)
    schema=request['response_format']['schema']
    audit={'type':'array','maxItems':6,'items':{'type':'object','properties':{
        'answer_fragment_ids':{'type':'array','items':{'type':'string','enum':list(fragments)}},
        'finding':{'type':'string','enum':['supported','contradicted','unsupported','wrong_document_type','internal_conflict']},
        'evidence_ids':{'type':'array','items':{'type':'string','enum':[k for k,v in reference_indices(case,unit)[1].items() if v['path'].startswith('evidence[') and '.content' in v['path']]}},
        'reason':{'type':'string'}},'required':['answer_fragment_ids','finding','evidence_ids','reason'],'additionalProperties':False}}
    schema['properties']={'claim_checks':audit,**schema['properties']}
    schema['required']=['claim_checks',*schema['required']]
    request['max_tokens']=2400
    return request


def evaluate(case: dict,unit: dict,arm: str,repeat: int=0) -> dict:
    request=request_for(case,unit,arm)
    identity=digest([request,unit['unit_id'],repeat])
    path=OUT/'runs'/f'{identity}.json'
    if path.exists():
        return json.loads(path.read_text())
    record={'run_id':identity,'agent':case['agent'],'unit_id':unit['unit_id'],'group_id':unit['group_id'],
        'partition':unit['partition'],'arm':arm,'repeat':repeat,'request':request,'requested_at':time.time()}
    started=time.monotonic()
    try:
        response=api('chat/completions',request)
        record['response']=response
        if response['choices'][0]['finish_reason']!='stop':
            raise ValueError('Генерация не завершилась stop')
        parsed=json.loads(response['choices'][0]['message']['content'])
        scores,audit=validate(case,decoded_unit(unit),parsed)
        fragment_ids=json.loads(request['messages'][-1]['content'])['answer_fragments']
        allowed=request['response_format']['schema']['properties']['claim_checks']['items']['properties']['evidence_ids']['items']['enum']
        for check in parsed['claim_checks']:
            if not check['answer_fragment_ids'] or not set(check['answer_fragment_ids'])<=set(fragment_ids) or not set(check['evidence_ids'])<=set(allowed):
                raise ValueError('Ссылка вне фрагментов ответа или содержимого инструментов')
        record.update(status='ok',scores=scores,parsed=parsed,audit=audit)
    except (RuntimeError,ValueError,KeyError,TypeError) as error:
        record.update(status='error',error=str(error))
    record['seconds']=round(time.monotonic()-started,3)
    path.parent.mkdir(exist_ok=True)
    path.write_text(dumps(record))
    return record


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--arms',nargs='+',choices=ARMS,default=ARMS)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--partition',choices=['dev','test'],default='dev')
    args=parser.parse_args()
    case=json.loads((PREVIOUS/'cases/CI10071259.json').read_text())
    units=sorted([u for u in case['units'] if u['partition']==args.partition],key=lambda u:u['unit_id'])
    if args.limit:
        units=units[:args.limit]
    for i,unit in enumerate(units,1):
        for arm in args.arms:
            result=evaluate(case,unit,arm)
            print(dumps({'unit':i,'total':len(units),'arm':arm,'status':result['status'],
                'scores':result.get('scores'),'seconds':result['seconds'],'error':result.get('error')}),flush=True)
            if 'HTTP 40' in result.get('error',''):
                raise RuntimeError(result['error'])


if __name__=='__main__':
    main()
