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
    apply_judge_labels,
    build_judge_plan,
    input_observed,
    judge_instruction,
    score_judge_predictions,
)
from laim_monitoring import (  # noqa: E402
    JUDGE_SCORE_INPUT,
    MonitoringContractError,
    aggregate_main_metric,
    broadcast_scores,
    unitize,
    validate_monitoring_metric,
)


def contract(formula: str, inputs: list[dict], *, mode: str = "qa") -> dict:
    return validate_monitoring_metric({
        "contract_version": "laim-monitoring-metric.v3",
        "status": "computed",
        "basket_id": "CI1",
        "metric_name": "KM",
        "assessment_mode": mode,
        "formula": formula,
        "inputs": inputs,
        "baseline": {"value": 0.75, "recomputed_value": 0.75, "reconciliation": "match"},
    })


PREDICTION = {"name": "prediction", "column": "класс_output_answer", "judged": False}
TARGET = {"name": "target", "column": "класс_metric", "judged": True}
ACCURACY = contract("mean(prediction == target)", [PREDICTION, TARGET])
CRITERIA = contract("mean((полнота + точность) / 2)", [
    {"name": "полнота", "column": "полнота_metric", "judged": True},
    {"name": "точность", "column": "точность_metric", "judged": True},
])
F1 = contract('f1(prediction, target, "macro")', [PREDICTION, TARGET])


def umr(**columns) -> pd.DataFrame:
    size = len(next(iter(columns.values())))
    frame = pd.DataFrame({
        "query_id": [f"q{i}" for i in range(size)],
        "input_query": [f"вопрос {i}" for i in range(size)],
        "output_answer": [f"ответ {i}" for i in range(size)],
    })
    for name, values in columns.items():
        frame[name] = values
    return frame


def test_judge_predicts_judged_inputs_only():
    plan = build_judge_plan(ACCURACY, agent_observed=True)
    assert plan.semantics == CONTRACT_FORMULA
    assert plan.judge_fields == ("target",)
    assert plan.contract == ACCURACY
    assert build_judge_plan(CRITERIA).judge_fields == ("полнота", "точность")


def test_unobserved_agent_answer_falls_back_to_judge_score_and_says_so():
    plan = build_judge_plan(ACCURACY, agent_observed=False)
    assert plan.semantics == JUDGE_FINAL_SCORE
    assert plan.judge_fields == (JUDGE_SCORE_INPUT,)
    assert plan.contract["formula"] == f"mean({JUDGE_SCORE_INPUT})"
    assert "класс_output_answer" in plan.reason and "информативное" in plan.reason


def test_criteria_contract_without_agent_input_ignores_agent_observed_flag():
    assert build_judge_plan(CRITERIA, agent_observed=False).semantics == CONTRACT_FORMULA


def test_dialogue_mode_always_uses_judge_score():
    dialogue = contract("mean(итог)", [{"name": "итог", "column": "итог_metric", "judged": True}], mode="dialogue")
    assert build_judge_plan(dialogue).semantics == JUDGE_FINAL_SCORE


def test_instruction_names_judge_fields_and_formula():
    text = judge_instruction(build_judge_plan(ACCURACY))
    assert "- target:" in text and "класс_metric" in text
    assert "prediction" not in text.split("\n")[1]
    assert "mean(prediction == target)" in text
    assert "разметка запроса, а не оценка ответа агента" in text


def test_input_observed_requires_non_blank_column():
    assert not input_observed(None, PREDICTION)
    assert not input_observed(umr(x=[1]), PREDICTION)
    assert not input_observed(umr(класс_output_answer=[None, " "]), PREDICTION)
    assert input_observed(umr(класс_output_answer=[None, "вклад"]), PREDICTION)


def test_accuracy_pipeline_judge_labels_to_contract_columns_to_km():
    frame = umr(класс_output_answer=["Вклад", "Кредит", "Ипотека", "Вклад"], класс_metric=[None] * 4)
    plan = build_judge_plan(ACCURACY)
    units = unitize(frame, plan.contract)
    predictions = pd.DataFrame({"agent_target": ["вклад", "Ипотека", "Ипотека", None]})
    scores = score_judge_predictions(units, predictions, plan)
    assert scores.tolist()[:3] == [1.0, 0.0, 1.0] and math.isnan(scores.tolist()[3])

    scored = apply_judge_labels(broadcast_scores(frame, units, scores), units, predictions, plan)
    assert scored["класс_metric"].tolist() == ["вклад", "Ипотека", "Ипотека", None]
    result = aggregate_main_metric(scored, ACCURACY)
    assert result["value"] == pytest.approx(2 / 3) and result["excluded_units"] == 1


def test_f1_contract_aggregates_by_formula_on_judge_labels():
    frame = umr(класс_output_answer=["a", "a", "b"], класс_metric=[None] * 3)
    plan = build_judge_plan(F1)
    units = unitize(frame, plan.contract)
    predictions = pd.DataFrame({"agent_target": ["a", "b", "b"]})
    scored = apply_judge_labels(frame, units, predictions, plan)
    # a: tp=1 fp=1 fn=0 → 2/3; b: tp=1 fp=0 fn=1 → 2/3
    assert aggregate_main_metric(scored, F1)["value"] == pytest.approx(2 / 3)


def test_criteria_scores_use_formula_rowwise():
    frame = umr(полнота_metric=[None, None], точность_metric=[None, None])
    plan = build_judge_plan(CRITERIA)
    units = unitize(frame, plan.contract)
    predictions = pd.DataFrame({"agent_полнота": [1, 0], "agent_точность": [1, 1]})
    assert score_judge_predictions(units, predictions, plan).tolist() == pytest.approx([1.0, 0.5])


def test_judge_score_fallback_passes_score_through():
    plan = build_judge_plan(ACCURACY, agent_observed=False)
    units = unitize(umr(класс_output_answer=[None, None]), plan.contract)
    predictions = pd.DataFrame({f"agent_{JUDGE_SCORE_INPUT}": [1, 0]})
    assert score_judge_predictions(units, predictions, plan).tolist() == [1.0, 0.0]


def test_missing_or_misaligned_judge_output_is_contract_error():
    plan = build_judge_plan(CRITERIA)
    units = unitize(umr(полнота_metric=[None, None], точность_metric=[None, None]), plan.contract)
    with pytest.raises(MonitoringContractError):
        score_judge_predictions(units, pd.DataFrame({"agent_полнота": [1, 1]}), plan)
    with pytest.raises(MonitoringContractError):
        score_judge_predictions(units, pd.DataFrame({"agent_полнота": [1], "agent_точность": [1]}), plan)
