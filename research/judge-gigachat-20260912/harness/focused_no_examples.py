"""Контроль структуры без противоречивых train-примеров; исходная рубрика."""
from __future__ import annotations

import argparse
import asyncio
import logging

import sdk_round as sdk
from structure_head import prepare

ARMS={
    'roles_schema_structure_focused_no_examples_max':'GigaChat-2-Max',
    'roles_schema_structure_focused_no_examples_ultra':'GigaChat-3-Ultra',
}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    case=prepare(focused=True)
    sdk.MODELS.update(ARMS)
    if args.run:
        asyncio.run(sdk.run(case,list(ARMS),'dev',0))
        return
    for arm,model in ARMS.items():
        judge,_=sdk.make_judge(case,arm)
        assert judge.llm.model==model
        assert judge.examples_retriever.hybrid_search(query='test',k=10)==[]
        assert judge.defect_retriever is None and judge.defect_examples==[]
        assert set(judge._output_model.model_fields)=={'structure'}
        assert judge._output_model.model_validate({'structure':0}).structure==0
    assert all(not u['evidence'] for u in case['units'])
    print('PASS: отдельная S-рубрика, без примеров/внешних фактов, исходные баллы включая 0; API не вызывался')


if __name__=='__main__':
    main()
