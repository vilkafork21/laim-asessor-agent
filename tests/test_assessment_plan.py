"""Судья — оракул разметки, формула — контракт. Без LLM."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assessment_plan import (  # noqa: E402
    CONTRACT_FORMULA,
    JUDGE_FINAL_SCORE,
    JUDGE_SCORE_SOURCE_ID,
    build_judge_plan,
    judge_instruction,
    score_judge_predictions,
    source_observed,
)
from laim_monitoring import (  # noqa: E402
    MonitoringContractError,
    aggregate_main_metric,
    broadcast_scores,
    unitize,
)


def _contract(method: str, sources: list[dict], *, mode: str = "qa", policy: str = "exclude_unit") -> dict:
    return {
        "contract_version": "laim-monitoring-metric.v2",
        "umr_version": "laim-umr.v2",
        "status": "computed",
        "basket_id": "CI1",
        "name": "Accuracy",
        "score_column": "main_metric",
        "assessment_mode": mode,
        "scoring": {
            "method": method,
            "sources": sources,
            "missing_policy": policy,
            "majority_denominator": None,
        },
        "aggregation": {"method": "mean", "weight_column": None},
        "baseline": {
            "value": 0.75, "scale": "ratio", "value_source": "validation_report",
            "reported_value": 0.75, "reported_scale": "ratio",
            "recomputed_value": 0.75, "reconciliation": "match",
        },
        "primary_validation": {
            "threshold": None, "comparator": None, "scale": "ratio",
            "verdict": None, "affects_monitoring": False,
        },
        "evidence": {},
    }


def _source(source_id: str, column: str, role: str, normalization="numeric") -> dict:
    return {
        "source_id": source_id, "column_name": column, "role": role,
        "normalization": normalization, "polarity": "direct",
    }


ACCURACY = _contract(
    "accuracy",
    [
        _source("source_1", "класс_output_answer", "prediction", "label"),
        _source("source_2", "класс_reference_answer", "target", "label"),
    ],
)
CRITERIA = _contract(
    "mean_criteria",
    [_source("source_1", "полнота_metric", "criterion"), _source("source_2", "точность_metric", "criterion")],
)


def _umr(**columns) -> pd.DataFrame:
    size = len(next(iter(columns.values())))
    base = {
        "query_id": [f"q{i}" for i in range(size)],
        "input_query": [f"вопрос {i}" for i in range(size)],
        "output_answer": [f"ответ {i}" for i in range(size)],
    }
    base.update(columns)
    return pd.DataFrame(base)


# ---------------------------------------------------------------- build plan

def test_accuracy_with_observed_prediction_judge_predicts_target_only():
    plan = build_judge_plan(ACCURACY, prediction_observed=True)
    assert plan.semantics == CONTRACT_FORMULA
    assert plan.judge_source_ids == ("source_2",)
    assert plan.contract["scoring"]["method"] == "accuracy"
    assert plan.contract["scoring"]["sources"] == ACCURACY["scoring"]["sources"]


def test_accuracy_without_prediction_falls_back_to_judge_score_and_says_so():
    plan = build_judge_plan(ACCURACY, prediction_observed=False)
    assert plan.semantics == JUDGE_FINAL_SCORE
    assert plan.judge_source_ids == (JUDGE_SCORE_SOURCE_ID,)
    assert plan.contract["scoring"]["method"] == "identity"
    assert plan.contract["scoring"]["missing_policy"] == "exclude_unit"
    assert "класс_output_answer" in plan.reason
    assert "информативное" in plan.reason


def test_criteria_methods_keep_contract_and_all_sources():
    plan = build_judge_plan(CRITERIA, prediction_observed=True)
    assert plan.semantics == CONTRACT_FORMULA
    assert plan.judge_source_ids == ("source_1", "source_2")
    assert plan.contract == CRITERIA


def test_dialogue_mode_always_uses_judge_score():
    contract = _contract("mean_criteria", CRITERIA["scoring"]["sources"], mode="dialogue")
    plan = build_judge_plan(contract, prediction_observed=True)
    assert plan.semantics == JUDGE_FINAL_SCORE
    assert plan.judge_source_ids == (JUDGE_SCORE_SOURCE_ID,)


def test_plan_does_not_mutate_input_contract():
    before = {"scoring": dict(ACCURACY["scoring"])}
    build_judge_plan(ACCURACY, prediction_observed=False)
    assert ACCURACY["scoring"] == before["scoring"]


# --------------------------------------------------------------- instruction

def test_instruction_explains_target_is_truth_not_grading():
    text = judge_instruction(build_judge_plan(ACCURACY, prediction_observed=True))
    assert "source_2" in text
    assert "ИСТИННАЯ метка" in text
    assert "НЕ оценка правильности" in text
    assert "source_1" not in text.split("\n")[1]
    assert "будет вычислена как совпадение" in text


def test_instruction_lists_every_judge_field_for_criteria():
    text = judge_instruction(build_judge_plan(CRITERIA))
    assert "- source_1" in text and "- source_2" in text
    assert "полнота_metric" in text


# ----------------------------------------------------------------- observed

def test_source_observed_requires_non_blank_column():
    source = ACCURACY["scoring"]["sources"][0]
    assert not source_observed(None, source)
    assert not source_observed(_umr(x=[1]), source)
    assert not source_observed(_umr(класс_output_answer=[None, " "]), source)
    assert source_observed(_umr(класс_output_answer=[None, "вклад"]), source)


# ------------------------------------------------------------------ scoring

def test_accuracy_score_is_equality_of_judge_target_and_umr_prediction():
    umr = _umr(
        класс_output_answer=["Вклад", "Кредит", "Ипотека", "Вклад"],
        класс_reference_answer=[None] * 4,
    )
    plan = build_judge_plan(ACCURACY, prediction_observed=True)
    units = unitize(umr, plan.contract)
    predictions = pd.DataFrame({"agent_source_2": ["вклад", "Ипотека", "Ипотека", None]})
    scores = score_judge_predictions(units, predictions, plan)
    assert scores.tolist()[:3] == [1.0, 0.0, 1.0]
    assert math.isnan(scores.tolist()[3])  # отказ судьи — не ноль

    frame = broadcast_scores(umr, units, scores)
    aggregate = aggregate_main_metric(frame, plan.contract)
    assert aggregate["value"] == pytest.approx(2 / 3)
    assert aggregate["excluded_units"] == 1


def test_mean_criteria_score_uses_contract_formula_on_judge_labels():
    umr = _umr(полнота_metric=[None, None], точность_metric=[None, None])
    plan = build_judge_plan(CRITERIA)
    units = unitize(umr, plan.contract)
    predictions = pd.DataFrame({"agent_source_1": [1, 0], "agent_source_2": [1, 1]})
    scores = score_judge_predictions(units, predictions, plan)
    assert scores.tolist() == pytest.approx([1.0, 0.5])


def test_judge_score_fallback_passes_score_through_identity():
    umr = _umr(класс_output_answer=[None, None])
    plan = build_judge_plan(ACCURACY, prediction_observed=False)
    units = unitize(umr, plan.contract)
    predictions = pd.DataFrame({f"agent_{JUDGE_SCORE_SOURCE_ID}": [1, 0]})
    scores = score_judge_predictions(units, predictions, plan)
    assert scores.tolist() == [1.0, 0.0]


def test_missing_judge_field_is_contract_error():
    plan = build_judge_plan(CRITERIA)
    units = unitize(_umr(полнота_metric=[None], точность_metric=[None]), plan.contract)
    with pytest.raises(MonitoringContractError):
        score_judge_predictions(units, pd.DataFrame({"agent_source_1": [1]}), plan)


def test_length_mismatch_is_contract_error():
    plan = build_judge_plan(CRITERIA)
    units = unitize(_umr(полнота_metric=[None, None], точность_metric=[None, None]), plan.contract)
    with pytest.raises(MonitoringContractError):
        score_judge_predictions(
            units, pd.DataFrame({"agent_source_1": [1], "agent_source_2": [1]}), plan,
        )
