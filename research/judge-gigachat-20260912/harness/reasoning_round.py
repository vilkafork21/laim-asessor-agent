"""Рассуждения и контроль с одинаковым decoding через настоящий Asessor.run."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import time
from unittest.mock import patch
from pathlib import Path

import httpx
import pandas as pd
from gigachat.client import _get_kwargs
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.json import parse_json_markdown
from pydantic import BaseModel

import sdk_round as sdk
from packet_round import enrich
from agent.prompts import ASSESSMENT_INPUT_PROMPT

MODELS={'ultra':'roles_schema_packet_no_examples','pro':'roles_schema_packet_no_examples_pro'}
ORIGINAL_LLM=sdk.GigaChat


def parse_final(message: AIMessage,output_model: type[BaseModel]) -> BaseModel:
    if message.response_metadata.get('finish_reason')!='stop':
        raise ValueError('Генерация конечного ответа не завершена нормально')
    return output_model.model_validate(parse_json_markdown(message.content,parser=json.loads))


def make_judge(case: dict,model: str,effort: str,output_format: str,streaming: bool=False,pooled: bool=False) -> tuple[sdk.Asessor,sdk.Recorder]:
    def configured_llm(**kwargs):
        kwargs.update(reasoning_effort=None if effort=='off' else effort,
            temperature=.8,top_p=1.0,max_tokens=8192,timeout=300,streaming=streaming)
        token_path=sdk.OLD/'.gigachat-token.json'
        if token_path.exists():
            token=json.loads(token_path.read_text())
            if token.get('expires_at',0)>(time.time()+60)*1000:
                kwargs['access_token']=token['access_token']
        return ORIGINAL_LLM(**kwargs)

    with patch.object(sdk,'GigaChat',configured_llm):
        judge,recorder=sdk.make_judge(case,MODELS[model])
    recorder.http_events=[]
    if pooled:
        async def on_request(request):
            recorder.http_events.append({'event':'request','host':request.url.host,'path':request.url.path,'at':time.time()})

        async def on_response(response):
            recorder.http_events.append({'event':'response_headers','host':response.request.url.host,'path':response.request.url.path,'status':response.status_code,'at':time.time()})

        client=judge.llm._client
        kwargs=_get_kwargs(client._settings)
        kwargs.update(timeout=httpx.Timeout(300,connect=15),limits=httpx.Limits(max_connections=1,keepalive_expiry=120))
        client._aclient_instance=httpx.AsyncClient(**kwargs,event_hooks={'request':[on_request],'response':[on_response]})
        client._auth_aclient.event_hooks={'request':[on_request],'response':[on_response]}
    if output_format=='text':
        system=judge.SYSTEM_PROMPT.split('<Формат ответа>')[0]+'''<Формат ответа>
Верни только конечный JSON-объект без markdown, вводного текста и пояснений.
Используй точные ключи и допустимые значения из следующей схемы:
{final_schema}'''
        prompt=ChatPromptTemplate.from_messages([('system',system),('human',ASSESSMENT_INPUT_PROMPT)]).partial(
            final_schema=json.dumps(judge._output_model.model_json_schema(),ensure_ascii=False))
        judge.printing_chain=judge.retrieval_chain|prompt
        judge.agent_chain=judge.printing_chain|judge.llm.bind(functions=None)|RunnableLambda(lambda message:parse_final(message,judge._output_model))
    return judge,recorder


async def run(case: dict,model: str,effort: str,partition: str,limit: int,output_format: str,streaming: bool,pooled: bool=False) -> None:
    judge,recorder=make_judge(case,model,effort,output_format,streaming,pooled)
    arm=f"reasoning_packet_{model}_{effort}_t8_{output_format}_{'stream' if streaming else 'http'}"
    if pooled:
        arm+='_pool120_c15'
    evaluated=enrich(case)[0] if case['agent']=='CI10071259' else case
    if case['agent']!='CI10071259':
        arm=arm.replace('reasoning_packet_','reasoning_context_')
    units=sorted([u for u in evaluated['units'] if u['partition']==partition],key=lambda u:u['unit_id'])
    if limit:
        units=units[:limit]
    for i,unit in enumerate(units,1):
        context=sdk.context_for(unit)
        messages=judge.printing_chain.invoke(sdk._serialize_llm_record({'assessment_context':context})).to_messages()
        request={'messages':[{'role':m.type,'content':m.content} for m in messages],
            'schema':judge._output_model.model_json_schema(),'arm':arm,'model':judge.llm.model,
            'temperature':judge.llm.temperature,'top_p':judge.llm.top_p,'max_tokens':judge.llm.max_tokens,
            'reasoning_effort':judge.llm.reasoning_effort,'streaming':judge.llm.streaming,'timeout':300,
            'retrieval':'no_examples','structured_output':output_format+'; complete JSON; original output model; final content only',
            'functions_omitted':output_format=='text',
            'execution':'Asessor.run; production retry policy; one unit in flight'}
        if pooled:
            request['transport']={'sdk':'async httpx','connect_timeout':15,'read_timeout':300,'max_connections':1,'keepalive_expiry':120}
        identity=hashlib.sha256(sdk.canonical([request,unit['unit_id']]).encode()).hexdigest()
        path=sdk.OUT/'sdk-runs'/f'{identity}.json'
        if path.exists():
            record=json.loads(path.read_text())
        else:
            record={'agent':case['agent'],'unit_id':unit['unit_id'],'group_id':unit['group_id'],
                    'partition':partition,'arm':arm,'repeat':0,'request':request,'requested_at':time.time(),
                    'initialized_from_token_cache':judge.llm.access_token is not None}
            recorder.responses=[]
            recorder.errors=[]
            recorder.http_events=[]
            started=time.monotonic()
            try:
                values=await judge.run(pd.DataFrame([{'assessment_context':context}]))
                if len(values)!=1 or values[0] is None:
                    raise ValueError('Нода не вернула оценку')
                record.update(status='ok',scores=values[0].model_dump())
            except Exception as error:
                record.update(status='error',error=type(error).__name__+': '+str(error))
            record.update(responses=recorder.responses,call_errors=recorder.errors,http_events=recorder.http_events,seconds=round(time.monotonic()-started,3))
            path.write_text(sdk.canonical(record))
        if any(e.get('status_code') in [401,402,403] for e in record.get('call_errors',[])):
            raise RuntimeError('Прогон остановлен: авторизация или доступный пакет токенов')
        print(sdk.canonical({'arm':arm,'unit':i,'total':len(units),'status':record['status'],
            'scores':record.get('scores'),'seconds':record['seconds']}),flush=True)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',type=Path,default=sdk.OLD/'cases/CI10071259.json')
    parser.add_argument('--model',choices=list(MODELS),default='ultra')
    parser.add_argument('--effort',choices=['off','low','medium'],default='medium')
    parser.add_argument('--partition',choices=['dev','test'],default='dev')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--format',choices=['text','json_schema'],default='text')
    parser.add_argument('--streaming',action='store_true')
    parser.add_argument('--pooled',action='store_true')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    case=json.loads(args.case.read_text())
    if args.run:
        asyncio.run(run(case,args.model,args.effort,args.partition,args.limit,args.format,args.streaming,args.pooled))
        return
    judge,_=make_judge(case,args.model,args.effort,args.format,args.streaming,args.pooled)
    evaluated=enrich(case)[0] if case['agent']=='CI10071259' else case
    unit=next(u for u in evaluated['units'] if u['partition']=='dev')
    data=sdk._serialize_llm_record({'assessment_context':sdk.context_for(unit)})
    messages=judge.printing_chain.invoke(data).to_messages()
    payload=judge.llm._build_payload(messages,**({'functions':None} if args.format=='text' else {}))
    assert payload.reasoning_effort==(None if args.effort=='off' else args.effort)
    assert payload.max_tokens==8192 and payload.temperature==.8 and payload.top_p==1
    assert judge.llm.streaming==args.streaming
    if args.pooled:
        client=judge.llm._client._aclient
        assert client.timeout.connect==15 and client.timeout.read==300
        assert client._transport._pool._max_connections==1
        assert client._transport._pool._keepalive_expiry==120
    if args.format=='text':
        assert 'functions' not in payload.model_dump(exclude_none=True)
    baseline,_=sdk.make_judge(case,MODELS[args.model])
    original=baseline.printing_chain.invoke(data).to_messages()
    assert messages[1:]==original[1:]
    assert messages[0].content.split('<Формат ответа>')[0]==original[0].content.split('<Формат ответа>')[0]
    assert judge._output_model.model_json_schema()==baseline._output_model.model_json_schema()
    parser_case=json.loads((sdk.OLD/'cases/CI10071259.json').read_text())
    parser_judge,_=make_judge(parser_case,args.model,args.effort,args.format,args.streaming)
    valid=AIMessage(content='{"factuality":0,"completeness":1,"structure":2}',response_metadata={'finish_reason':'stop'})
    assert parse_final(valid,parser_judge._output_model).factuality==0
    assert parse_final(AIMessage(content='```json\n'+valid.content+'\n```',response_metadata={'finish_reason':'stop'}),parser_judge._output_model).factuality==0
    for content,finish in [('', 'stop'),('{"factuality":2}','stop'),('{"factuality":3,"completeness":1,"structure":2}','stop'),(valid.content+' текст','stop'),(valid.content,'length'),('```json\n'+valid.content[:-1]+'\n```','stop')]:
        try:
            parse_final(AIMessage(content=content,response_metadata={'finish_reason':finish}),parser_judge._output_model)
        except ValueError:
            pass
        else:
            raise AssertionError('Некорректный конечный ответ принят')
    print('PASS: прежние данные/рубрика/шкала; параметры SDK проверены; пустые, неполные и некорректные ответы отвергаются; API не вызывается')


if __name__=='__main__':
    main()
