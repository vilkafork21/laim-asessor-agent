"""Контракт all_assessors (единогласие): принимается и считается как min голосов."""

from __future__ import annotations

import pandas as pd

from laim_monitoring import score_units, unitize, validate_monitoring_metric


def _contract() -> dict:
    return {
        "contract_version": "laim-monitoring-metric.v2", "umr_version": "laim-umr.v2",
        "status": "computed", "basket_id": "CI1", "name": "quality", "score_column": "main_metric",
        "assessment_mode": "qa",
        "scoring": {
            "method": "all_assessors",
            "sources": [
                {"source_id": f"source_{index}", "column_name": f"mark{index}_metric",
                 "role": "assessor_vote", "normalization": "numeric", "polarity": "direct"}
                for index in (1, 2)
            ],
            "missing_policy": "fail", "majority_denominator": None,
        },
        "aggregation": {"method": "mean", "weight_column": None},
        "baseline": {
            "value": 0.5, "scale": "ratio", "value_source": "validation_report",
            "reported_value": 0.5, "reported_scale": "ratio", "recomputed_value": 0.5,
            "reconciliation": "match",
        },
        "primary_validation": {
            "threshold": None, "comparator": None, "scale": "ratio", "verdict": None,
            "affects_monitoring": False,
        },
        "evidence": {},
    }


def test_all_assessors_contract_is_accepted_and_scored_as_unanimity():
    contract = validate_monitoring_metric(_contract())
    frame = pd.DataFrame({
        "query_id": ["q1", "q2"],
        "input_query": ["a", "b"],
        "output_answer": ["x", "y"],
        "mark1_metric": [1, 1],
        "mark2_metric": [1, 0],
    })
    units = unitize(frame, contract)
    assert score_units(units, _contract()).tolist() == [1.0, 0.0]


def test_defect_examples_are_forced_into_context():
    """Судья не распознаёт дефект, если в few-shot нет ни одного примера дефекта."""
    from agent.asessor_agent import _ensure_defect_examples

    found = [{"question": "q", "answer": '{"score":1.0}'} for _ in range(10)]
    pool = [{"question": "bad", "answer": '{"score":0.0}'}]

    result = _ensure_defect_examples(found, pool, {"score": 0.0}, quota=3)

    assert sum(1 for item in result if '"score":0.0' in item["answer"]) == 1
    assert len(result) == 11


def test_defect_examples_not_duplicated_when_already_present():
    from agent.asessor_agent import _ensure_defect_examples

    found = [{"question": "bad", "answer": '{"score":0.0}'}] + [
        {"question": "q", "answer": '{"score":1.0}'} for _ in range(9)
    ]
    pool = [{"question": "bad", "answer": '{"score":0.0}'}]

    result = _ensure_defect_examples(found, pool, {"score": 0.0}, quota=1)

    assert result == found


def test_similar_defects_are_searched_not_taken_in_order():
    """Примеры дефектов должны подбираться по похожести на запрос, а не подряд."""
    from agent.asessor_agent import _defect_pool

    class _Retriever:
        def __init__(self):
            self.queries = []

        def hybrid_search(self, query, k=10):
            self.queries.append((query, k))
            return [{"question": "похожий дефект", "answer": '{"score":0.0}'}]

    retriever = _Retriever()
    pool = _defect_pool(retriever, [{"question": "любой", "answer": '{"score":0.0}'}], "запрос")

    assert retriever.queries == [("запрос", 3)]
    assert pool == [{"question": "похожий дефект", "answer": '{"score":0.0}'}]


def test_defect_pool_falls_back_to_plain_list_without_retriever():
    from agent.asessor_agent import _defect_pool

    examples = [{"question": "дефект", "answer": '{"score":0.0}'}]

    assert _defect_pool(None, examples, "запрос") == examples
