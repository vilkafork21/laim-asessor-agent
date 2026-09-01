"""Reference UMR в формате тестового датасета: packed dialogue и flat с session_id."""

from __future__ import annotations

import pandas as pd
import pytest

from laim_monitoring import MonitoringContractError, broadcast_scores, normalize_umr, unitize


def _contract(mode: str) -> dict:
    return {
        "contract_version": "laim-monitoring-metric.v2", "umr_version": "laim-umr.v2",
        "status": "computed", "basket_id": "CI1", "name": "quality", "score_column": "main_metric",
        "assessment_mode": mode,
        "scoring": {
            "method": "identity",
            "sources": [{
                "source_id": "source_1", "column_name": "score_metric", "role": "final_score",
                "normalization": "numeric", "polarity": "direct",
            }],
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


def test_packed_dialogue_reference_is_unitized_per_session():
    frame = pd.DataFrame({
        "session_id": ["s1", "s2"],
        "dialogue": ["[('q1', 'hi', 'hello'), ('q2', 'bye', 'see you')]", "[('q3', 'x', 'y')]"],
        "input_query_count": [1, 1],
        "score_metric": [1.0, 0.0],
        "main_metric": [1.0, 0.0],
    })
    units = unitize(frame, _contract("dialogue"))
    assert len(units) == 2
    assert [turn["input_query"] for turn in units["dialogue"].iloc[0]] == ["hi", "bye"]
    assert units["source_1"].tolist() == [1.0, 0.0]
    assert units["main_metric"].tolist() == [1.0, 0.0]


def test_flat_reference_with_session_id_keeps_turn_history():
    frame = pd.DataFrame({
        "session_id": ["s1", "s1", "s2"],
        "query_id": ["q1", "q2", "q3"],
        "input_query_count": [1, 1, 1],
        "input_query": ["hi", "bye", "x"],
        "output_answer": ["hello", "see you", "y"],
        "score_metric": [1.0, 0.0, 1.0],
        "main_metric": [1.0, 0.0, 1.0],
    })
    units = unitize(frame, _contract("turn_with_history"))
    assert len(units) == 3
    assert [turn["input_query"] for turn in units["assessment_context"].iloc[1]["history"]] == ["hi"]


def test_flat_reference_without_canonical_columns_is_rejected():
    with pytest.raises(MonitoringContractError):
        unitize(pd.DataFrame({"question": ["q"], "answer": ["a"]}), _contract("qa"))


def test_packed_dialogue_scores_broadcast_onto_normalized_rows():
    packed = pd.DataFrame({
        "session_id": ["s1", "s2"],
        "dialogue": ["[('q1', 'hi', 'hello'), ('q2', 'bye', 'see you')]", "[('q3', 'x', 'y')]"],
        "input_query_count": [1, 1],
        "score_metric": [1.0, 0.0],
        "main_metric": [1.0, 0.0],
    })
    contract = _contract("dialogue")
    frame = normalize_umr(packed, contract)
    units = unitize(frame, contract)

    scored = broadcast_scores(frame, units, pd.Series([1.0, 0.0]))

    assert len(scored) == 3
    assert scored["main_metric"].tolist() == [1.0, 1.0, 0.0]


def test_plan_less_refusal_returns_not_computable_instead_of_crashing():
    """Контракт без assessment_mode (plan-less отказ адаптера) — graceful выход."""
    import main as assessor

    refusal = {
        "contract_version": "laim-monitoring-metric.v2",
        "umr_version": "laim-umr.v2",
        "status": "not_computable",
        "basket_id": "CI1",
        "reason": "MeasurementPlan не построен",
        "reason_code": "structured_output_error",
    }
    monitoring = pd.DataFrame({
        "session_id": ["s1"], "query_id": ["q1"], "input_query_count": [1],
        "input_query": ["в"], "output_answer": ["о"],
    })

    result = assessor.main(
        reference_umr=monitoring.copy(),
        monitoring_metric=refusal,
        monitoring_umr=monitoring,
    )

    assert result["assessment_result"]["status"] == "not_computable"
    assert result["acc_auto"] is None
    assert result["scored_data"]["main_metric"].isna().all()
