"""Передача проверяемых свойств текста GigaChat без автоматической замены балла."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from unittest.mock import patch

import categorical_structure as categorical
import sdk_round as sdk
from structure_head import prepare

ARM='roles_schema_structure_observed_no_examples_max'


def text_checks(answer: str) -> dict:
    words=answer.split()
    # ponytail: только полный повтор от 20 слов; частичные повторы проверяет GigaChat.
    duplicate=len(words)>=20 and len(words)%2==0 and words[:len(words)//2]==words[len(words)//2:]
    technical=[{'token':m.group(),'start':m.start(),'end':m.end()} for m in re.finditer(r'\b(?:null|none|true|false|fhd|risk|client_id|rehabilitation|online)\b',answer,re.I)]
    return {'whole_answer_repeated_twice':duplicate,'raw_technical_tokens':technical,
        'provenance':'Детерминированные наблюдения только над текущим ответом, не внешние факты и не экспертные оценки.'}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--online',action='store_true')
    parser.add_argument('--stress',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--partition',choices=['dev','test'],default='dev')
    args=parser.parse_args()
    assert not (args.online and args.stress)
    logging.basicConfig(level=logging.ERROR)
    case=json.loads((sdk.OUT/'CI10071255-structure.json').read_text()) if args.online else prepare(focused=True)
    partition=args.partition
    if args.stress:
        from stress_cascade import prepare as prepare_stress
        stress,specs=prepare_stress()
        selected={s['unit_id'] for s in specs if s['variant']=='duplicate_answer'}
        case['units'] += [u for u in stress['units'] if u['unit_id'] in selected]
        partition='stress'
    for unit in case['units']:
        answer=unit['context']['current_turn']['output_answer']
        unit['evidence']=[{'kind':'observed_text_properties','content':text_checks(answer)}]
    case['target']+='''
Перед оценкой проверь observed_text_properties. Эти признаки вычислены по буквальному тексту ответа; это не независимая проверка фактов.
whole_answer_repeated_twice=true означает, что в тексте не менее 20 слов и две его половины совпадают с точностью до пробелов. Полный повтор — нарушение по критерию повторов исходной рубрики.
raw_technical_tokens показывает позиции буквальных null/None/True/False и внутренних имён полей. Проверь, не осталось ли в заключении неадаптированное техническое содержимое JSON вместо понятного читателю изложения. ИНН, суммы и даты сами по себе техническим мусором не считаются.
Эти наблюдения не заменяют остальные проверки структуры и не определяют фактичность или полноту. Окончательную категорию выбери по исходной рубрике.'''
    sdk.MODELS[ARM]='GigaChat-2-Max'
    if args.run:
        with patch.object(sdk,'make_judge',categorical.make_judge):
            asyncio.run(sdk.run(case,[ARM],partition,0))
        return
    source=' '.join(f'слово{i}' for i in range(20))
    assert text_checks(source+'\n\n'+source)['whole_answer_repeated_twice']
    assert not text_checks(source)['whole_answer_repeated_twice']
    assert not text_checks('да да')['whole_answer_repeated_twice']
    sample='Поле risk: null. ИНН: 1234567890, дата 01.01.2025.'
    for match in text_checks(sample)['raw_technical_tokens']:
        assert sample[match['start']:match['end']]==match['token']
    assert not text_checks('Истина и ложь; сумма 123 рублей')['raw_technical_tokens']
    judge,_=categorical.make_judge(case,ARM)
    assert judge.examples_retriever.hybrid_search(query='test',k=10)==[]
    print('PASS: точный полный повтор и позиции токенов проверены; баллы выбирает только GigaChat; API не вызывался')


if __name__=='__main__':
    main()
