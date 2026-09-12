"""Живое сравнение ролей и structured output через Asessor.run и GigaChat SDK."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import runpy
import ssl
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from dotenv import dotenv_values
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_gigachat import GigaChat

NODE=Path('/Users/antonzyukov/laim-judge-roles-20260912')
OLD=Path('/Users/antonzyukov/laim-artifacts/judge-deep-research-20260911')
OUT=Path(__file__).parent
sys.path.insert(0,str(NODE))
from agent.asessor_agent import Asessor, _serialize_llm_record  # noqa: E402
from retriever.retriever import EnhancedBM25  # noqa: E402
sys.path.append(str(OLD))
from analyze_results import mode  # noqa: E402

ARMS=['legacy_function','roles_function','roles_schema','roles_schema_giga3_pro','roles_schema_giga2_max','roles_schema_packet_no_examples','roles_schema_packet_no_examples_pro','roles_schema_packet_feedback','roles_schema_no_examples','roles_schema_packet_no_examples_semantic','roles_schema_packet_no_examples_max','roles_schema_packet_feedback_max']
MODELS={'roles_schema_giga3_pro':'GigaChat-3-Pro','roles_schema_giga2_max':'GigaChat-2-Max','roles_schema_packet_no_examples_pro':'GigaChat-3-Pro','roles_schema_packet_no_examples_max':'GigaChat-2-Max','roles_schema_packet_feedback_max':'GigaChat-2-Max'}
LEGACY=runpy.run_path('/Users/antonzyukov/laim-judge-research-20260911/agent/prompts.py')['SYSTEM_PROMPT']


def canonical(value: object) -> str:
    return json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)


class BM25Retriever:
    """Единый доступный retrieval для всех плеч; векторный канал здесь не воспроизводится."""
    def __init__(self,embedding_model: object,examples: list[dict]):
        self.examples=examples
        self.index=EnhancedBM25([re.findall(r'\w+',e['question'].lower()) for e in examples])

    def hybrid_search(self,query: str,k: int=10) -> list[dict]:
        scores=self.index.get_scores(re.findall(r'\w+',query.lower()))
        order=sorted(range(len(self.examples)),key=lambda i:(-scores[i],self.examples[i]['question']))
        return [self.examples[i] for i in order[:k]]


class Recorder(BaseCallbackHandler):
    def __init__(self):
        self.responses=[]
        self.errors=[]

    def on_llm_error(self,error,**kwargs) -> None:
        self.errors.append({"type":type(error).__name__,"status_code":getattr(error,"status_code",None)})

    def on_llm_end(self,response,**kwargs) -> None:
        self.responses.append({'llm_output':response.llm_output,'generations':[[g.message.model_dump(mode='json') for g in group] for group in response.generations]})


def context_for(unit: dict) -> dict:
    context=json.loads(json.dumps(unit['context']))
    if unit['evidence']:
        context['observations']={'evaluation_evidence':unit['evidence']}
    return context


def make_judge(case: dict,arm: str) -> tuple[Asessor,Recorder]:
    if arm in ['roles_schema_packet_feedback','roles_schema_packet_feedback_max','roles_schema_packet_no_examples_semantic'] and case['agent']!='CI10071259':
        raise ValueError('Доменные гипотезы подготовлены только для CI10071259')
    settings=dotenv_values(OLD/'.gigachat.env')
    ssl_context=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_context.check_hostname=False
    ssl_context.verify_mode=ssl.CERT_NONE
    ssl_context.maximum_version=ssl.TLSVersion.TLSv1_2
    recorder=Recorder()
    llm=GigaChat(model=MODELS.get(arm,'GigaChat-3-Ultra'),base_url='https://api.giga.chat/v1',
                credentials=settings['CREDENTIALS'],scope=settings['SCOPE'],ssl_context=ssl_context,
                timeout=150,max_retries=0,temperature=.001,top_p=.001,max_tokens=600,callbacks=[recorder])
    train=[]
    for u in sorted(case['units'],key=lambda u:u['unit_id']):
        if u['partition']!='train':
            continue
        scores={c:mode([r['scores'][c] for r in u['ratings']]) for c in case['scores']}
        if any(v is None for v in scores.values()):
            continue
        train.append({'assessment_context':context_for(u),**scores})
    instruction=case['rubric']+'\n'+case['target']
    if arm=='roles_schema_packet_no_examples_semantic':
        from round4 import RULES
        instruction+='\nПроверка семантики по исходной рубрике:\n'+RULES.split('В поле claim_checks')[0]
    with patch('agent.asessor_agent.QuestionAnswerRetriever',BM25Retriever):
        judge=Asessor(llm=llm,embedding_model=None,dataset=pd.DataFrame(train),
            instruction=instruction,context_columns=['assessment_context'],
            answer_columns=list(case['scores']),score_values=next(iter(case['scores'].values())),
            instruction_summarization=False,instruction_structuring=False,examples_summarization=False)
    if 'no_examples' in arm:
        judge.examples_retriever=SimpleNamespace(hybrid_search=lambda **_:[])
        judge.defect_retriever=None
        judge.defect_examples=[]
    if arm in ['roles_schema_packet_feedback','roles_schema_packet_feedback_max']:
        from round4 import examples

        def feedback_search(query: str,k: int=10) -> list[dict]:
            unit={'context':json.loads(query)['assessment_context']}
            return [{'question':_serialize_llm_record({'assessment_context':e['assessment_context'],'human_feedback':e['human_feedback']}),
                     'answer':_serialize_llm_record(e['scores'])} for e in examples(case,unit,True)]

        judge.examples_retriever=SimpleNamespace(hybrid_search=feedback_search)
        judge.defect_retriever=None
        judge.defect_examples=[]
    if arm=='legacy_function':
        judge.printing_chain=judge.retrieval_chain|ChatPromptTemplate.from_messages([LEGACY])
    method='json_schema' if arm.startswith('roles_schema') else 'function_calling'
    kwargs={'strict':True} if method=='json_schema' else {}
    judge.agent_chain=judge.printing_chain|llm.with_structured_output(judge._output_model,method=method,**kwargs)
    return judge,recorder


async def run(case: dict,arms: list[str],partition: str,limit: int) -> None:
    judges={a:make_judge(case,a) for a in arms}
    expanded={}
    if any('_packet' in a for a in arms):
        from packet_round import enrich
        expanded={u['unit_id']:u for u in enrich(case)[0]['units']}
    units=sorted([u for u in case['units'] if u['partition']==partition],key=lambda u:u['unit_id'])
    if limit:
        units=units[:limit]
    for arm,(judge,recorder) in judges.items():
        for i,u in enumerate(units,1):
            context=context_for(expanded[u['unit_id']] if '_packet' in arm else u)
            messages=judge.printing_chain.invoke(_serialize_llm_record({'assessment_context':context})).to_messages()
            request={'messages':[{'role':m.type,'content':m.content} for m in messages],
                'schema':judge._output_model.model_json_schema(),'arm':arm,'model':judge.llm.model,
                'temperature':.001,'top_p':.001,'max_tokens':600,
                'retrieval':'question_bm25_6_factuality_balanced_train_feedback' if arm in ['roles_schema_packet_feedback','roles_schema_packet_feedback_max'] else ('no_examples' if 'no_examples' in arm else 'bm25_full_context_k10_defect_quota3'),
                'execution':'Asessor.run; production retry policy; one unit in flight'}
            identity=hashlib.sha256(canonical([request,u['unit_id']]).encode()).hexdigest()
            path=OUT/'sdk-runs'/f'{identity}.json'
            if path.exists():
                record=json.loads(path.read_text())
            else:
                record={'agent':case['agent'],'unit_id':u['unit_id'],'group_id':u['group_id'],
                        'partition':partition,'arm':arm,'repeat':0,'request':request,'requested_at':time.time()}
                recorder.responses=[]
                recorder.errors=[]
                started=time.monotonic()
                try:
                    values=await judge.run(pd.DataFrame([{'assessment_context':context}]))
                    if len(values)!=1 or values[0] is None:
                        raise ValueError('Нода не вернула оценку')
                    record.update(status='ok',scores=values[0].model_dump())
                except Exception as error:
                    record.update(status='error',error=type(error).__name__+': '+str(error))
                record.update(responses=recorder.responses,call_errors=recorder.errors,seconds=round(time.monotonic()-started,3))
                path.parent.mkdir(exist_ok=True)
                path.write_text(canonical(record))
            if any(e.get('status_code') in [401,402,403] for e in record.get('call_errors',[])):
                raise RuntimeError('Прогон остановлен: авторизация или доступный пакет токенов')
            print(canonical({'agent':case['agent'],'unit':i,'total':len(units),'arm':arm,'status':record['status'],'scores':record.get('scores'),'seconds':record['seconds']}),flush=True)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',type=Path,default=OLD/'cases/CI10071259.json')
    parser.add_argument('--arms',nargs='+',choices=ARMS,default=ARMS)
    parser.add_argument('--partition',choices=['dev','test'],default='dev')
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(run(json.loads(args.case.read_text()),args.arms,args.partition,args.limit))


if __name__=='__main__':
    main()
