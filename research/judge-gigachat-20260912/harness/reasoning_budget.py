"""Контроль достаточного бюджета рассуждений без изменения замороженного runner."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from unittest.mock import patch

import reasoning_round as rr

ORIGINAL_LLM=rr.ORIGINAL_LLM


def budget_llm(**kwargs):
    kwargs['max_tokens']=16384
    return ORIGINAL_LLM(**kwargs)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--limit',type=int,default=2)
    args=parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    case=json.loads((rr.sdk.OLD/'cases/CI10071259.json').read_text())
    rr.MODELS['ultra_16k']=rr.MODELS['ultra']
    with patch.object(rr,'ORIGINAL_LLM',budget_llm):
        if args.run:
            asyncio.run(rr.run(case,'ultra_16k','low','dev',args.limit,'text',True,True))
            return
        judge,_=rr.make_judge(case,'ultra_16k','low','text',True,True)
        assert judge.llm.model=='GigaChat-3-Ultra'
        assert judge.llm.max_tokens==16384 and judge.llm.reasoning_effort=='low'
        assert judge.llm.temperature==.8 and judge.llm.top_p==1 and judge.llm.streaming
        assert set(judge._output_model.model_fields)=={'factuality','completeness','structure'}
        assert judge._output_model.model_validate({'factuality':0,'completeness':0,'structure':0}).factuality==0
    print('PASS: Ultra low, 16384, исходная шкала, отдельное имя плеча; API не вызывался')


if __name__=='__main__':
    main()
