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
