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
