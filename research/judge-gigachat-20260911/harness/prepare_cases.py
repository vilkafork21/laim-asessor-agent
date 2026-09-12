"""Фиксирует реальные объекты и разбиение по клиентам/сессиям до модельных сравнений."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

import docx2txt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

OUT = Path(__file__).parent
DATA = Path('/Users/antonzyukov/laim-data-20260911')
PACKAGES = Path('/Users/antonzyukov/laim/b2c_agents_artifacts')
SCORE_COLUMNS = {
    'factuality': 'Фактологическая точность ответа',
    'completeness': 'Полнота предоставленной информации',
    'structure': 'Структурированный формат ответа',
}
FACT_TOOLS = {
    'search_case_docs', 'collect_doc_meta', 'in_out_flows', 'payments_by_category',
    'check_all_docs_present', 'collect_counterparties_meta_tool',
    'collect_counterparties_in_docs_meta_tool',
}


def clean(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def canonical(value: object) -> str:
    return json.dumps(clean(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def get_competence_cases() -> tuple[dict, dict]:
    root = DATA/'artifacts/CI10071259'
    source = root/'baskets/agent_post_labeled_all_2025_11_27_v1.parquet'
    instruction = root/'instructions/instruct_303697.docx'
    frame = pd.read_parquet(source).reset_index(drop=True)
    keys = ['case_id', 'doc_request_id', 'question_id', 'question', 'answer']
    units, rejected = [], Counter()
    for key, rows in frame.groupby(keys, sort=False, dropna=False):
        responses = rows.response.dropna().unique()
        if not len(responses):
            rejected['no_response'] += 1
            continue
        assert len(responses) == 1
        try:
            messages = ast.literal_eval(responses[0])['messages']
        except (ValueError, SyntaxError, KeyError, TypeError):
            rejected['unparseable_response'] += 1
            continue
        humans = [m for m in messages if m.get('type') == 'human']
        assistants = [m for m in messages if m.get('type') == 'ai']
        if (not humans or not assistants or str(humans[0].get('content', '')).strip() != str(key[3]).strip()
                or str(assistants[-1].get('content', '')).strip() != str(key[4]).strip()):
            rejected['question_answer_link_mismatch'] += 1
            continue
        ratings = [{'rater_id': digest(r['ID Эксперта']),
                    'scores': {name: float(r[column]) for name, column in SCORE_COLUMNS.items()}}
                   for _, r in rows.iterrows() if r[list(SCORE_COLUMNS.values())].notna().all()]
        if not ratings:
            rejected['missing_gold'] += 1
            continue
        assert all(v in (0., 1., 2.) for r in ratings for v in r['scores'].values())
        calls = {call['id']: (position, call) for position, m in enumerate(messages)
                 for call in m.get('tool_calls', [])}
        evidence = []
        for position, message in enumerate(messages):
            if message.get('type') != 'tool':
                continue
            call_id = message.get('tool_call_id')
            assert call_id in calls and calls[call_id][0] < position
            assert message.get('status') == 'success'
            call = calls[call_id][1]
            name = call.get('name')
            if name in FACT_TOOLS and message.get('content'):
                evidence.append({'evidence_id': f'tool-{position}', 'kind': 'tool_result',
                                 'tool_name': name, 'arguments': call.get('args'),
                                 'content': message['content'], 'message_position': position,
                                 'binding': 'same_response_before_final_answer'})
        units.append({'unit_id': digest(key), 'group_id': digest(key[0]),
                      'source_rows': list(map(int, rows.index)),
                      'context': {'mode': 'qa', 'current_turn': {'input_query': key[3], 'output_answer': key[4]}},
                      'evidence': evidence, 'ratings': ratings})
    assert len(frame) == 3039
    case = {'agent': 'CI10071259', 'rubric': docx2txt.process(instruction),
            'target': 'Верни три независимые оценки фактичности, полноты и структуры по исходной инструкции. '
                      'Каждая оценка на шкале 0/1/2. Не объединяй их в итоговую КМ.',
            'scores': {name: [0, 1, 2] for name in SCORE_COLUMNS}, 'units': units,
            'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,instruction]},
            'gold_kind': 'individual_expert_ratings',
            'limitations': 'Tool results связаны с ответом, но не заменяют полный пакет документов клиента; '
                            'порядок агрегации периодной КМ не подтверждён.'}
    return case, {'source_rows':len(frame), 'distinct_targets':frame.groupby(keys, dropna=False).ngroups,
                  'retained_targets':len(units), 'rejected_targets':dict(rejected),
                  'targets_with_factual_tools':sum(bool(u['evidence']) for u in units)}


def main() -> None:
    spec = importlib.util.spec_from_file_location('existing_matrix', '/Users/antonzyukov/laim/docs/reviews/monitoring-judge-matrix.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = []
    for ci, (frame, sources, rubric, values, target) in module.prepare_sources(PACKAGES).items():
        units = []
        for _, row in frame.iterrows():
            group = row.get('client_group', row['_group_id'])
            units.append({'unit_id':digest([ci,row['source_row'],row['assessment_context']]),
                          'group_id':digest(group), 'source_rows':[int(row['source_row'])],
                          'context':clean(row['assessment_context']), 'evidence':[],
                          'ratings':[{'rater_id':'human_final', 'scores':{'assessment_score':float(row['assessment_score'])}}]})
        cases.append({'agent':ci,'rubric':rubric,'target':target, 'scores':{'assessment_score':values},
                      'units':units,'gold_kind':'documented_final_human_score',
                      'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}})
    competency, diagnostics = get_competence_cases()
    cases.append(competency)
    summary = {'CI10071259_binding':diagnostics, 'cases':{}}
    for case in cases:
        units = case['units']
        parents = {u['group_id']:u['group_id'] for u in units}
        def find(group: str) -> str:
            while parents[group] != group:
                parents[group] = parents[parents[group]]
                group = parents[group]
            return group
        contexts = {}
        for u in units:
            context_key = digest([u['context'],u['evidence']])
            if context_key in contexts:
                parents[find(u['group_id'])] = find(contexts[context_key])
            contexts[context_key] = u['group_id']
        for u in units:
            u['group_id'] = find(u['group_id'])
        groups = [u['group_id'] for u in units]
        # Стратификация по наличию любой ниже-максимальной экспертной оценки не меняет gold.
        defects = [any(v < max(case['scores'][s]) for r in u['ratings'] for s,v in r['scores'].items()) for u in units]
        distinct_groups = sorted(set(groups))
        group_defects = [any(d for g,d in zip(groups,defects) if g==group) for group in distinct_groups]
        splitter = (StratifiedKFold(5,shuffle=True,random_state=20260911)
                    if min(Counter(group_defects).values()) >= 5
                    else KFold(5,shuffle=True,random_state=20260911))
        group_folds = {}
        for fold, (_, indices) in enumerate(splitter.split(distinct_groups,group_defects)):
            group_folds.update({distinct_groups[i]:fold for i in indices})
        folds = [group_folds[group] for group in groups]
        for u, fold in zip(units,folds):
            u['partition'] = 'test' if fold==0 else 'dev' if fold==1 else 'train'
        # Физические копии внутри dev/test не умножают вес точного контекста с одинаковым gold.
        seen = {}
        for u in units:
            key = (u['partition'],digest([u['context'],u['evidence']]))
            gold = canonical([r['scores'] for r in u['ratings']])
            if key in seen:
                if seen[key] == gold:
                    u['partition']='excluded_duplicate'
                else:
                    u['conflicting_context_panel']=True
            else:
                seen[key]=gold
        active = [u for u in units if not u['partition'].startswith('excluded')]
        assert not ({u['group_id'] for u in active if u['partition']=='test'} &
                    {u['group_id'] for u in active if u['partition']!='test'})
        case['split_seed']=20260911
        path=OUT/'cases'/f"{case['agent']}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(canonical(case))
        summary['cases'][case['agent']]={'units':len(units),'groups':len(set(groups)),
            'partitions':dict(Counter(u['partition'] for u in units)),
            'test_defects':sum(d for u,d in zip(units,defects) if u['partition']=='test'),
            'test_groups':len({u['group_id'] for u in units if u['partition']=='test'}),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT/'case_inventory.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
