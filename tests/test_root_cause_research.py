"""Исследовательские контроли не меняют gold и не подают его модели."""
from pathlib import Path


def test_root_cause_selection_and_prompt_keep_the_control_blind(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'research/judge-root-causes-20260913'))
    import audit as causes
    import live
    from agent.prompts import SYSTEM_PROMPT

    unit = {'unit_id': 'a', 'partition': 'dev', 'context': {'mode': 'qa', 'current_turn': {'input_query': 'q', 'output_answer': 'a'}},
            'evidence': [], 'ratings': [{'scores': {'score': 0}}]}
    case = {'scores': {'score': [0, 1]}, 'units': [unit, {**unit, 'unit_id': 'b', 'partition': 'test'}]}
    assert [u['unit_id'] for u in causes.selected_units(case)] == ['a']
    context = live.context_for(unit)
    unit['ratings'][0]['scores']['score'] = 1
    assert live.context_for(unit) == context
    assert live.candidate_prompt().replace(live.NEW_BOUNDARY, live.OLD_BOUNDARY) == SYSTEM_PROMPT
    assert '0, даже если другой пункт непроверяем' in live.candidate_prompt()
    assert 'если они различаются, верни "not_assessable"' in live.candidate_prompt()


def test_restoration_rejects_wrong_answer_and_keeps_only_observations(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'research/judge-root-causes-20260913'))
    from restore_observations import bind_picker_turns, bound_tools
    import pytest

    original = [{'input_query': 'q', 'output_answer': 'a', 'suggestions': None, 'turn_index': 1}]
    restored = bind_picker_turns(original, [{'input_query': 'q', 'output_answer': 'a', 'suggestions': None,
                                          'scenario': 'initial', 'subscenario': 'limitation', 'human_score': 0}])
    assert restored[0]['observed_scenario'] == 'initial'
    assert 'human_score' not in restored[0]
    assert 'observed_scenario' not in original[0]
    with pytest.raises(ValueError, match='не совпадает'):
        bind_picker_turns(original, [{'input_query': 'q', 'output_answer': 'другой'}])
    messages = [{'type': 'ai', 'tool_calls': [{'id': 't', 'name': 'search', 'args': {}}]},
                {'type': 'tool', 'tool_call_id': 't', 'name': 'search', 'status': 'success', 'content': 'данные'},
                {'type': 'ai', 'content': 'итог'}]
    assert bound_tools(messages, 'итог')[0]['content'] == 'данные'
    with pytest.raises(ValueError, match='связь'):
        bound_tools(messages[1:], 'итог')
    with pytest.raises(ValueError, match='Финальный'):
        bound_tools(messages, 'другая версия')


def test_live_control_preserves_missing_consensus(monkeypatch):
    import pandas as pd
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'research/judge-root-causes-20260913'))
    import live
    from agent.asessor_agent import _serialize_llm_record
    case = {'scores': {'score': [0, 1]}, 'units': [
        {'partition': 'train', 'context': {}, 'evidence': [], 'ratings': [{'scores': {'score': 0}}, {'scores': {'score': 1}}]},
        {'partition': 'train', 'context': {}, 'evidence': [], 'ratings': [{'scores': {'score': 1}}]},
    ]}
    frame = live.training_frame(case)
    assert isinstance(frame, pd.DataFrame)
    assert frame.iloc[0]['score'] is None
    assert 'null' in _serialize_llm_record(frame.iloc[0].to_dict())


def test_blind_route_hides_target_decision_and_preserves_prior_context(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'research/judge-root-causes-20260913'))
    import live
    unit = {'context': {'current_turn': {'input_query': 'вопрос', 'output_answer': 'ответ'},
                        'history': [{'input_query': 'раньше', 'output_answer': 'контекст'}],
                        'observed_prediction': 'liabilities'}, 'ratings': [{'scores': {'assessment_score': 0}}]}
    before = live.blind_route_context(unit)
    unit['context']['observed_prediction'] = 'issuance'
    unit['context']['current_turn']['output_answer'] = 'другая версия'
    unit['ratings'][0]['scores']['assessment_score'] = 1
    assert live.blind_route_context(unit) == before
    assert before == {'input_query': 'вопрос', 'history': [{'input_query': 'раньше', 'output_answer': 'контекст'}]}
    from langchain_gigachat import GigaChat
    GigaChat(access_token='offline-test').with_structured_output(live.RouteDecision, method='function_calling')


def test_annotation_audit_binds_comments_to_actual_answer(monkeypatch, tmp_path):
    import json
    import pandas as pd
    import pytest
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'research/judge-root-causes-20260913'))
    import audit as causes
    unit = {'unit_id': 'x', 'partition': 'dev', 'source_rows': [0],
            'context': {'current_turn': {'output_answer': 'версия A'}},
            'evidence': [{'content': '[]'}], 'ratings': [{'scores': {'structure': 0}}]}
    case = {'agent': 'CI10071259', 'units': [unit], 'source_hashes': {'local.parquet': ''}, 'scores': {'structure': [0, 1, 2]}, 'rubric': ''}
    frame = pd.DataFrame([{'answer': 'версия B', 'ТЭГ': '#НетЛогики', 'Комментарий': 'эксперт'}])
    monkeypatch.setattr(causes, 'cases', lambda: [(None, case)])
    monkeypatch.setattr(causes, 'OUT', tmp_path)
    monkeypatch.setattr(causes.pd, 'read_parquet', lambda _: frame)
    with pytest.raises(ValueError, match='другой версии'):
        causes.annotation_audit()
    frame.at[0, 'answer'] = 'версия A'
    causes.annotation_audit()
    result = json.loads((tmp_path/'rubric-annotation-audit.json').read_text())['CI10071259']
    assert result['structure_annotation_audit']['dev:0']['unanimous'] == 1
    assert result['only_empty_tool_results'][1]['units'] == 1


def test_route_training_uses_only_train_and_valid_source_category(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'research/judge-root-causes-20260913'))
    import live
    import pytest
    unit = {'unit_id': 'x', 'partition': 'train', 'source_rows': [2],
            'context': {'current_turn': {'input_query': 'q', 'output_answer': 'a'}, 'history': [], 'observed_prediction': 'liabilities'}}
    source = [None, [None, None, 'liabilities', 0, 'report', None, None, 'q']]
    examples = live.route_training({'units': [unit, {**unit, 'partition': 'test', 'unit_id': 'test'}]}, source)
    assert examples == [{'unit_id': 'x', 'input_query': 'q', 'history': [], 'route': 'report'}]
    source[1][4] = 0
    assert live.route_training({'units': [unit]}, source) == []
    source[1][7] = 'чужой вопрос'
    with pytest.raises(ValueError, match='не совпадает'):
        live.route_training({'units': [unit]}, source)


def test_rubric_examples_keep_original_category_and_text(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'research/judge-root-causes-20260913'))
    import live
    rubric = 'Введение\nliabilities: Текущие обязательства\nПРИМЕРЫ:\nпомощь близкому\n\nunknown: Остальное\nстрахование кредита\n'
    examples = live.rubric_examples(rubric)
    assert {'category': 'liabilities', 'text': 'помощь близкому'} in examples
    assert {'category': 'unknown', 'text': 'страхование кредита'} in examples
    assert all(e['text'] in rubric for e in examples)
    assert not any(e['text'] == 'ПРИМЕРЫ:' for e in examples)


def test_paired_analysis_counts_abstention_as_lost_yield(monkeypatch, tmp_path):
    import json
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'research/judge-root-causes-20260913'))
    import analyze
    units = [{'unit_id': str(i), 'group_id': str(i), 'partition': 'dev',
              'ratings': [{'scores': {'assessment_score': score}}]} for i, score in enumerate([0, 1, 1])]
    (tmp_path/'runs').mkdir()
    for arm, scores in [('baseline', [0, 1, 1]), ('restored_observations', [0, None, 1])]:
        for unit, score in zip(units, scores):
            record = {'agent': 'CI09840670', 'unit_id': unit['unit_id'], 'arm': arm,
                      'scores': {'assessment_score': score}, 'request': {}}
            (tmp_path/'runs'/f"{arm}-{unit['unit_id']}.json").write_text(json.dumps(record))
    monkeypatch.setattr(analyze, 'OUT', tmp_path)
    monkeypatch.setattr(analyze, 'cases', lambda: [(None, {'agent': agent, 'units': units}) for agent in ['CI09840670', 'CI09997438']])
    analyze.main()
    result = json.loads((tmp_path/'paired-comparisons.json').read_text())[0]
    assert result['correct_control'] == 3 and result['correct_candidate'] == 2
    assert result['regressed'] == 1 and result['corrected'] == 0
    assert result['delta_correct_label_yield']['ci95'][1] <= 0


def test_expert_reason_examples_are_train_only_and_bound_to_answer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'research/judge-root-causes-20260913'))
    import live
    import pytest
    unit = {'unit_id': 'train', 'partition': 'train', 'source_rows': [2],
            'context': {'current_turn': {'input_query': 'q', 'output_answer': 'a'}},
            'evidence': [], 'ratings': [{'scores': {'structure': 0}}]}
    case = {'units': [unit, {**unit, 'unit_id': 'dev', 'partition': 'dev', 'source_rows': [9]}], 'scores': {'structure': [0, 1, 2]}}
    rows = {2: {'input_query': 'q', 'output_answer': 'a', 'comment': 'объяснение только train'}}
    examples = live.annotated_training(case, rows)
    assert len(examples) == 1 and examples[0]['unit_id'] == 'train'
    assert examples[0]['comments'] == ['объяснение только train']
    rows[2]['output_answer'] = 'чужая версия'
    with pytest.raises(ValueError, match='другой версии'):
        live.annotated_training(case, rows)
