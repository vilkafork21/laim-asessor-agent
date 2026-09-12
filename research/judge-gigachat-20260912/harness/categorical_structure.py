"""Выбор вида нарушения с фиксированным отображением в исходную шкалу."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Literal
from unittest.mock import patch

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, ConfigDict, Field

import sdk_round as sdk
from agent.prompts import ASSESSMENT_INPUT_PROMPT, SYSTEM_PROMPT
from structure_head import prepare

ARM='roles_schema_structure_categorical_no_examples_max'
SCORES={'logic_or_contradiction':0,'technical_noise_or_repeat':1,'no_violation':2,'not_assessable':None}
ORIGINAL_MAKE_JUDGE=sdk.make_judge


class StructureFinding(BaseModel):
    model_config=ConfigDict(extra='forbid')
    assessment_reason: str=Field(description='Краткое основание с фрагментом текущего ответа; не подробный ход рассуждений')
    structure: Literal['logic_or_contradiction','technical_noise_or_repeat','no_violation','not_assessable']


def make_judge(case: dict,arm: str) -> tuple[sdk.Asessor,sdk.Recorder]:
    judge,recorder=ORIGINAL_MAKE_JUDGE(case,arm)
    original_model=judge._output_model
    system=SYSTEM_PROMPT.split('<Допустимые значения критериев>')[0]+"""<Категории ответа>
Назови вид нарушения структуры по исходной рубрике:
logic_or_contradiction — соответствует баллу 0: нарушение логики или внутреннее противоречие;
technical_noise_or_repeat — соответствует баллу 1: логика в целом сохранна, но есть технический мусор, артефакты или повторы;
no_violation — соответствует баллу 2: нарушений структуры по рубрике нет;
not_assessable — обязательный критерий структуры проверить невозможно.
Если одновременно присутствуют нарушения разных уровней, выбирай более тяжёлое по исходной шкале.
Проверь связность утверждений и выводов, затем технические вставки и повторы.
Внешняя фактическая корректность не оценивается. Имеющийся markdown сам по себе не доказывает логичность.
Верни JSON: assessment_reason — краткое проверяемое основание, structure — одна из указанных категорий.
Не возвращай числовой балл: отображение категории в исходную шкалу выполняется автоматически."""
    judge.printing_chain=judge.retrieval_chain|ChatPromptTemplate.from_messages([('system',system),('human',ASSESSMENT_INPUT_PROMPT)])
    judge._output_model=StructureFinding
    judge.agent_chain=judge.printing_chain|judge.llm.with_structured_output(StructureFinding,method='json_schema',strict=True)|RunnableLambda(
        lambda result:original_model.model_validate({'structure':SCORES[result.structure]}))
    return judge,recorder


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',type=Path)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--partition',choices=['dev','test'],default='dev')
    args=parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    case=json.loads(args.case.read_text()) if args.case else prepare(focused=True)
    assert list(case['scores'])==['structure']
    sdk.MODELS[ARM]='GigaChat-2-Max'
    if args.run:
        with patch.object(sdk,'make_judge',make_judge):
            asyncio.run(sdk.run(case,[ARM],args.partition,0))
        return
    judge,_=make_judge(case,ARM)
    assert judge.examples_retriever.hybrid_search(query='test',k=10)==[]
    baseline,_=ORIGINAL_MAKE_JUDGE(case,ARM)
    for category,score in SCORES.items():
        result=StructureFinding(assessment_reason='Проверка',structure=category)
        assert baseline._output_model.model_validate({'structure':SCORES[result.structure]}).structure==score
    assert set(StructureFinding.model_json_schema()['properties']['structure']['enum'])==set(SCORES)
    assert SCORES['logic_or_contradiction']==0
    print('PASS: категории отображаются в 0/1/2/None, числовая схема не смешивается с категориальной; API не вызывался')


if __name__=='__main__':
    main()
