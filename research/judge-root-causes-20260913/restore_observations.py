"""Восстановление наблюдений по строгой связи с исходным вопросом и ответом."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from audit import OLD, OUT, cases


def bind_picker_turns(turns: list[dict], observations: list[dict]) -> list[dict]:
    if len(turns) != len(observations):
        raise ValueError('Число исходных реплик не совпадает с подготовленным диалогом')
    restored = []
    for turn, source in zip(turns, observations):
        if any(turn.get(k) != source.get(k) for k in ['input_query', 'output_answer', 'suggestions']):
            raise ValueError('Исходный вопрос, ответ или suggestions не совпадает')
        restored.append({**turn, 'observed_scenario': source.get('scenario'), 'observed_subscenario': source.get('subscenario')})
    return restored


def bound_tools(messages: list[dict], answer: str) -> list[dict]:
    assistants = [(i, m) for i, m in enumerate(messages) if m.get('type') == 'ai']
    if not assistants or assistants[-1][1].get('content', '').strip() != answer.strip():
        raise ValueError('Финальный ответ не совпадает с опубликованным')
    final_position = assistants[-1][0]
    calls = {}
    for position, message in assistants:
        for call in message.get('tool_calls', []):
            if call['id'] in calls:
                raise ValueError('Неоднозначная связь tool_call_id')
            calls[call['id']] = (position, call)
    result = []
    for position, message in enumerate(messages):
        if message.get('type') != 'tool':
            continue
        previous = calls.get(message.get('tool_call_id'))
        if previous is None or not previous[0] < position < final_position:
            raise ValueError('Не подтверждена связь инструмента с предшествующим вызовом')
        if message.get('status') != 'success' or not isinstance(message.get('content'), str) or not message['content'].strip():
            raise ValueError('Результат инструмента пуст или неуспешен')
        call = previous[1]
        result.append({'evidence_id': f'tool-{position}', 'kind': 'tool_result', 'tool_name': call['name'],
                       'arguments': call['args'], 'content': message['content'], 'message_position': position,
                       'binding': 'same_exit_payload_before_final_answer'})
    return result


def bind_nbsp_tools(messages: list[dict], question: str, answer: str) -> list[dict]:
    humans = [m for m in messages if m.get('type') == 'human']
    assistants = [m for m in messages if m.get('type') == 'ai']
    if not humans or humans[0]['content'].strip() != question.strip() or not assistants:
        raise ValueError('Не подтверждена исходная версия вопроса')
    final = assistants[-1]['content']
    if final.strip().replace('\u00a0', ' ') != answer.strip().replace('\u00a0', ' '):
        raise ValueError('Ответы отличаются не только NBSP')
    return [{**e, 'binding': 'same_response_nbsp_equivalent_before_final_answer'} for e in bound_tools(messages, final)]


def restore_post_nbsp() -> None:
    case = next(c for _, c in cases() if c['agent'] == 'CI10071259')
    source = next(Path(p) for p in case['source_hashes'] if p.endswith('.parquet'))
    if hashlib.sha256(source.read_bytes()).hexdigest() != case['source_hashes'][str(source)]:
        raise ValueError('Исходная корзина изменилась')
    frame = pd.read_parquet(source).reset_index(drop=True)
    assignments = defaultdict(set)
    for unit in case['units']:
        if unit['partition'].startswith('excluded'):
            continue
        for row in unit['source_rows']:
            assignments[frame.at[row, 'case_id']].add((unit['group_id'], unit['partition']))
    columns = {'factuality': 'Фактологическая точность ответа', 'completeness': 'Полнота предоставленной информации', 'structure': 'Структурированный формат ответа'}
    variants = []
    keys = ['case_id', 'doc_request_id', 'question_id', 'question', 'answer']
    for key, rows in frame.groupby(keys, sort=False, dropna=False):
        responses = rows.response.dropna().unique()
        if not len(responses):
            continue
        if len(responses) != 1:
            raise ValueError('Несколько response для одной версии ответа')
        messages = ast.literal_eval(responses[0])['messages']
        humans = [m for m in messages if m.get('type') == 'human']
        assistants = [m for m in messages if m.get('type') == 'ai']
        if not humans or not assistants or humans[0]['content'].strip() != str(key[3]).strip():
            continue
        final, answer = assistants[-1]['content'].strip(), str(key[4]).strip()
        if final == answer or final.replace('\u00a0', ' ') != answer.replace('\u00a0', ' '):
            continue
        related = frame[(frame[keys[:3]] == list(key[:3])).all(axis=1)]
        if len(related.response.dropna().unique()) != 1 or len(assignments[key[0]]) != 1:
            raise ValueError('Неоднозначный payload или разбиение исходной группы')
        group, partition = next(iter(assignments[key[0]]))
        ratings = [{'rater_id': hashlib.sha256(json.dumps(r['ID Эксперта'], ensure_ascii=False).encode()).hexdigest(),
                    'scores': {c: float(r[name]) for c, name in columns.items()}} for _, r in rows.iterrows() if r[list(columns.values())].notna().all()]
        variants.append({'source_rows': list(map(int, rows.index)), 'group_id': group, 'partition': partition,
                         'context': {'mode': 'qa', 'current_turn': {'input_query': key[3], 'output_answer': key[4]}},
                         'evidence': bind_nbsp_tools(messages, key[3], key[4]), 'ratings': ratings,
                         'benchmark_status': 'display_variant_not_independent_new_unit',
                         'response_sha256': hashlib.sha256(responses[0].encode()).hexdigest()})
    (OUT/'post-nbsp-variants.json').write_text(json.dumps(variants, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    summary = {'variants': len(variants), 'tool_results': sum(len(v['evidence']) for v in variants),
               'partitions': {p: sum(v['partition'] == p for v in variants) for p in ['train', 'dev', 'test']},
               'source_sha256': case['source_hashes'][str(source)], 'gold_policy': 'unchanged_not_merged', 'new_independent_groups': 0}
    (OUT/'post-nbsp-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(summary)


def main() -> None:
    case = json.loads((OLD/'cases/CI09840670.json').read_text())
    source = next(Path(p) for p in case['source_hashes'] if p.endswith('.xlsx'))
    book = load_workbook(source, read_only=True, data_only=True)
    rows = list(book.active.values)
    book.close()
    grouped = defaultdict(list)
    group = None
    for index, row in enumerate(rows[2:], 3):
        if row[0]:
            group = row[0]
        if row[1] and isinstance(row[3], str):
            if group is None:
                raise ValueError('Исходный диалог не имеет группы')
            grouped[group].append((index, {'input_query': row[3], 'output_answer': row[4], 'suggestions': row[5], 'scenario': row[6], 'subscenario': row[7]}))
    lookup = {entries[0][0]: [v for _, v in entries] for entries in grouped.values()}
    for u in case['units']:
        u['context']['turns'] = bind_picker_turns(u['context']['turns'], lookup[u['source_rows'][0]])
    (OUT/'CI09840670-observations.json').write_text(json.dumps(case, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    root = Path('/Users/antonzyukov/laim-data-20260911/traces/CI10071259')
    raw_path = root/'raw/part-00000-e4933b9f-c7a7-4db8-ad71-2725fcaaf764-c000.snappy.parquet'
    turns_path = root/'processed/CI10071259__turns.parquet'
    raw = pd.read_parquet(raw_path, columns=['span_id', 'trace_id', 'output_text']).set_index('span_id', verify_integrity=True)
    turns = pd.read_parquet(turns_path)
    restored = []
    basket_keys = {(u['context']['current_turn']['input_query'].strip(), u['context']['current_turn']['output_answer'].strip()) for u in json.loads((OLD/'cases/CI10071259.json').read_text())['units']}
    matches = 0
    for turn in turns.itertuples():
        span = raw.loc[turn.exit_span_id]
        messages = json.loads(span.output_text)['payload']['messages']
        humans = [m for m in messages if m.get('type') == 'human']
        if span.trace_id != turn.exit_trace_id or len(humans) != 1 or humans[0]['content'].strip() != turn.input_query.strip():
            raise ValueError('Не подтверждена связь trace, вопроса и ответа')
        evidence = bound_tools(messages, turn.agent_response)
        restored.append({'unit_id': turn.turn_id, 'context': {'mode': 'qa', 'current_turn': {'input_query': turn.input_query, 'output_answer': turn.agent_response}},
                         'evidence': evidence, 'exit_span_id': turn.exit_span_id, 'gold_status': 'unlabelled'})
        matches += (turn.input_query.strip(), turn.agent_response.strip()) in basket_keys
    (OUT/'post-restored-traces.json').write_text(json.dumps(restored, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    original_cases = cases()
    report = {'picker_units': len(case['units']), 'picker_turns': sum(len(u['context']['turns']) for u in case['units']),
              'picker_gold_unchanged': [u['ratings'] for u in case['units']] == [u['ratings'] for u in json.loads((OLD/'cases/CI09840670.json').read_text())['units']],
              'post_traces': len(restored), 'post_with_tools': sum(bool(u['evidence']) for u in restored), 'post_tool_results': sum(len(u['evidence']) for u in restored),
              'post_exact_basket_matches': matches,
              'original_case_sha256': {d['agent']: hashlib.sha256(p.read_bytes()).hexdigest() for p, d in original_cases},
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [source, raw_path, turns_path]}}
    (OUT/'restoration-summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print({k: v for k, v in report.items() if not k.endswith('sha256')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--post-nbsp', action='store_true')
    args = parser.parse_args()
    restore_post_nbsp() if args.post_nbsp else main()
