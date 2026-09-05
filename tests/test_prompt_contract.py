"""Контракт контекста и structured output автоассесора."""

import json

from langchain_core.prompts import ChatPromptTemplate

from agent.asessor_agent import _serialize_llm_record
from agent.prompts import SYSTEM_PROMPT
from agent.pydantic_output import create_simple_output_model
from agent.sds_chat_model import (
    _extract_json_object,
    _structured_contract,
    _validate_structured,
)


def test_system_prompt_has_one_flat_output_contract():
    rendered = ChatPromptTemplate.from_messages([SYSTEM_PROMPT]).format_messages(
        instructions="Оцените корректность.",
        examples="[]",
        domain_knowledge="",
        answer_columns_values_set={"correct": [0, 1]},
        user_input=json.dumps({
            "assessment_context": {
                "mode": "dialogue",
                "turns": [{"input_query": "q", "output_answer": "a"}],
            }
        }),
    )[0].content

    assert "оцени весь упорядоченный список `turns` один раз" in rendered
    assert "оцени только `current_turn`" in rendered
    assert "Не добавляй\nобёртку `answer`" in rendered
    assert '"thinking"' not in rendered


def test_dialogue_context_serialization_keeps_order_and_tail():
    context = {
        "assessment_context": {
            "mode": "dialogue",
            "turns": [
                {"turn_index": 1, "input_query": "первый", "output_answer": "ответ"},
                {"turn_index": 2, "input_query": "последний", "output_answer": "хвост"},
            ],
        }
    }

    serialized = _serialize_llm_record(context)
    restored = json.loads(serialized)

    assert [turn["turn_index"] for turn in restored["assessment_context"]["turns"]] == [1, 2]
    assert "последний" in serialized


def test_sds_structured_contract_requests_the_same_flat_object():
    schema = create_simple_output_model(
        ["correct"],
        [0, 1],
    )

    contract = _structured_contract(schema)

    assert '"correct": null' in contract
    assert '"answer":' not in contract
    assert "Не добавляй обёртку answer" in contract


def test_sds_parser_prefers_final_flat_object_but_accepts_legacy_wrapper():
    schema = create_simple_output_model(["correct"], [0, 1])
    payload = _extract_json_object(
        '<think>{"answer":{"correct":0}}</think>\n{"correct":1}'
    )

    assert payload == {"correct": 1}
    assert _validate_structured(payload, schema).correct == 1
    assert _validate_structured({"answer": {"correct": 0}}, schema).correct == 0


def test_unknown_evidence_requires_abstention_instead_of_automatic_pass():
    assert '"not_assessable"' in SYSTEM_PROMPT
    assert "Отсутствие сведений не доказывает ни успех, ни дефект" in SYSTEM_PROMPT
    assert "в бинарной шкале — большее" not in SYSTEM_PROMPT


def test_missing_domain_source_does_not_request_guessed_facts():
    from types import SimpleNamespace
    from agent.asessor_agent import Asessor

    judge = SimpleNamespace(
        _init_examples_rag=lambda: None, domain_rag_path=None, domain_retriever=None,
        examples_retriever=SimpleNamespace(hybrid_search=lambda **_: []),
        defect_retriever=None, defect_examples=[], _lowest_values={},
        instruction='Проверяй утверждения по представленным фактам.',
        answer_columns_values_set={'assessment_score': [0, 1]},
    )
    Asessor._init_rag(judge)
    inputs = judge.retrieval_chain.invoke('{}')
    assert inputs['domain_knowledge'] == ''
