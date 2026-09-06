"""Контракт КМ и расчёт по нему: одна форма, одна версия, доказанный baseline."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laim_monitoring import (  # noqa: E402
    JUDGE_SCORE_INPUT,
    MonitoringContractError,
    aggregate_main_metric,
    judge_score_contract,
    unit_scores,
    unitize,
    validate_monitoring_metric,
)


def contract(formula: str, inputs: list[dict], *, mode: str = "qa", baseline: float = 0.75,
             reconciliation: str = "match") -> dict:
    return {
        "contract_version": "laim-monitoring-metric.v3",
        "status": "computed",
        "basket_id": "CI1",
        "metric_name": "Accuracy",
        "assessment_mode": mode,
        "formula": formula,
        "inputs": inputs,
        "baseline": {"value": baseline, "recomputed_value": baseline, "reconciliation": reconciliation},
    }


ACCURACY = contract(
    "mean(prediction == target)",
    [
        {"name": "prediction", "column": "класс_output_answer", "judged": False},
        {"name": "target", "column": "класс_metric", "judged": True},
    ],
)


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


def test_valid_contract_passes_and_is_copied():
    validated = validate_monitoring_metric(ACCURACY)
    assert validated == ACCURACY and validated is not ACCURACY


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda c: c.update(contract_version="laim-monitoring-metric.v2"), "не поддерживается"),
        (lambda c: c.update(formula="mean(score)"), "ссылается на входы"),
        (lambda c: c.update(formula="prediction.mean()"), "Формула КМ"),
        (lambda c: c["inputs"].append({"name": "mean", "column": "x", "judged": True}), "Недопустимое имя"),
        (lambda c: c["inputs"].append({"name": "target", "column": "x", "judged": True}), "Повторяется"),
        (lambda c: c["baseline"].update(reconciliation="mismatch"), "не воспроизведён"),
        (lambda c: c.update(assessment_mode="chat"), "Недопустимое assessment_mode"),
    ],
)
def test_invalid_contract_is_rejected_with_reason(change, message):
    payload = validate_monitoring_metric(ACCURACY)
    change(payload)
    with pytest.raises(MonitoringContractError, match=message):
        validate_monitoring_metric(payload)


def test_not_computable_contract_is_returned_only_when_allowed():
    payload = {"contract_version": "laim-monitoring-metric.v3", "status": "not_computable", "reason": "нет отчёта"}
    assert validate_monitoring_metric(payload, require_computed=False)["reason"] == "нет отчёта"
    with pytest.raises(MonitoringContractError, match="нет отчёта"):
        validate_monitoring_metric(payload)


def test_aggregate_accuracy_over_umr_columns():
    frame = umr(класс_output_answer=["a", "b", "b", None], класс_metric=["a", "a", "b", "b"])
    result = aggregate_main_metric(frame, ACCURACY)
    assert result["value"] == pytest.approx(2 / 3)
    assert result["formula"] == "mean(prediction == target)"
    assert (result["total_units"], result["scored_units"], result["excluded_units"]) == (4, 3, 1)
    assert result["name"] == "Accuracy"


def test_aggregate_requires_inputs_in_data():
    with pytest.raises(MonitoringContractError, match="нет входов формулы"):
        aggregate_main_metric(umr(класс_output_answer=["a"]), ACCURACY)


def test_weighted_formula_uses_input_query_count():
    payload = contract(
        "wmean(оценка >= 4, weight)", [{"name": "оценка", "column": "оценка_metric", "judged": True}],
    )
    frame = umr(оценка_metric=[5, 3])
    frame["input_query_count"] = [3, 1]
    assert aggregate_main_metric(frame, payload)["value"] == pytest.approx(0.75)


def test_dialogue_unit_takes_one_value_per_session():
    payload = contract("mean(итог)", [{"name": "итог", "column": "итог_metric", "judged": True}], mode="dialogue")
    frame = umr(итог_metric=[1, 1, 0, 0])
    frame["session_id"] = ["s1", "s1", "s2", "s2"]
    result = aggregate_main_metric(frame, payload)
    assert result["total_units"] == 2 and result["value"] == pytest.approx(0.5)


def test_unit_scores_for_mean_formula_and_for_f1():
    frame = umr(класс_output_answer=["a", "b", "b"], класс_metric=["a", "a", "b"])
    units = unitize(frame, ACCURACY)
    assert unit_scores(units, ACCURACY).tolist() == [1.0, 0.0, 1.0]
    f1 = dict(ACCURACY, formula='f1(prediction, target, "macro")')
    assert unit_scores(units, f1).tolist() == [1.0, 0.0, 1.0]  # совпадение как построчный след
    nan_only = contract("f1(a, b, \"macro\")", [
        {"name": "a", "column": "a", "judged": True}, {"name": "b", "column": "b", "judged": True},
    ])
    assert all(math.isnan(v) for v in unit_scores(unitize(umr(a=["x"], b=["x"]), nan_only), nan_only))


def test_judge_score_contract_scores_main_metric():
    scored = judge_score_contract(ACCURACY)
    assert scored["inputs"] == [{"name": JUDGE_SCORE_INPUT, "column": "main_metric", "judged": True}]
    assert scored["formula"] == f"mean({JUDGE_SCORE_INPUT})"
    frame = umr(main_metric=[1, 0, 1])
    assert aggregate_main_metric(frame, scored)["value"] == pytest.approx(2 / 3)
