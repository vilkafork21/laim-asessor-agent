"""Вычислитель формул КМ: каждая формула проверена ручным расчётом."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laim_monitoring.formula import FormulaError, parse  # noqa: E402


def cols(**values) -> dict[str, pd.Series]:
    return {name: pd.Series(value, dtype=object if any(isinstance(v, str) for v in value) else "float64")
            for name, value in values.items()}


def test_accuracy_is_share_of_matching_labels_case_insensitive():
    formula = parse("mean(prediction == target)")
    assert formula.inputs == ("prediction", "target")
    value = formula.evaluate(cols(
        prediction=["Вклад", "Кредит", "вклад", "Ипотека"],
        target=["вклад", "Кредит", "Кредит", "Ипотека"],
    ))
    assert value == pytest.approx(0.75)


def test_blank_label_excludes_row_from_mean():
    formula = parse("mean(prediction == target)")
    value = formula.evaluate(cols(prediction=["a", "b", None], target=["a", "a", "a"]))
    assert value == pytest.approx(0.5)


def test_weighted_accuracy():
    formula = parse("wmean(prediction == target, weight)")
    value = formula.evaluate(cols(prediction=["a", "b"], target=["a", "a"], weight=[3.0, 1.0]))
    assert value == pytest.approx(0.75)


def test_mean_of_criteria_and_all_criteria():
    columns = cols(полнота=[1, 0, 1], точность=[1, 1, 0])
    assert parse("mean((полнота + точность) / 2)").evaluate(columns) == pytest.approx(2 / 3)
    assert parse("mean(min(полнота, точность))").evaluate(columns) == pytest.approx(1 / 3)


def test_avg_skips_blank_criterion_but_sum_does_not():
    columns = cols(a=[1, 1, 0], b=[1, np.nan, 0])
    assert parse("mean(avg(a, b))").evaluate(columns) == pytest.approx(2 / 3)
    assert parse("mean((a + b) / 2)").evaluate(columns) == pytest.approx(0.5)
    assert parse("mean((a + fillna(b, 0)) / 2)").evaluate(columns) == pytest.approx(0.5)


def test_majority_present_and_declared_denominators():
    columns = cols(a=[1, 1, 1], b=[1, 0, np.nan], c=[0, 0, np.nan])
    present = parse("mean(majority(a, b, c))").evaluate(columns)
    declared = parse("mean(majority(a, b, c, declared=True))").evaluate(columns)
    # строка 3: 1 голос из 1 присутствующего → 1; из 3 заявленных → 0
    assert present == pytest.approx(2 / 3)
    assert declared == pytest.approx(1 / 3)


def test_majority_tie_is_blank():
    rows = parse("mean(majority(a, b))").unit_expression().evaluate_rows(cols(a=[1, 1], b=[0, 1]))
    assert math.isnan(rows.iloc[0]) and rows.iloc[1] == 1.0


def test_threshold_share():
    assert parse("mean(оценка >= 4)").evaluate(cols(оценка=[5, 4, 3, np.nan])) == pytest.approx(2 / 3)


def test_boolean_operators():
    columns = cols(a=[1, 1, 0], b=[1, 0, 0])
    assert parse("mean(a == 1 and b == 1)").evaluate(columns) == pytest.approx(1 / 3)
    assert parse("mean(a == 1 or b == 1)").evaluate(columns) == pytest.approx(2 / 3)
    assert parse("mean(not (a == 1))").evaluate(columns) == pytest.approx(1 / 3)


def test_class_metrics_match_hand_computation():
    columns = cols(
        prediction=["a", "a", "b", "b", "a"],
        target=["a", "b", "b", "b", "b"],
    )
    # класс a: tp=1 fp=2 fn=0 → P=1/3 R=1 F1=0.5; класс b: tp=2 fp=0 fn=2 → P=1 R=0.5 F1=2/3
    assert parse('precision(prediction, target, "a")').evaluate(columns) == pytest.approx(1 / 3)
    assert parse('recall(prediction, target, "b")').evaluate(columns) == pytest.approx(0.5)
    assert parse('f1(prediction, target, "macro")').evaluate(columns) == pytest.approx((0.5 + 2 / 3) / 2)
    # weighted: support a=1, b=4 → (0.5·1 + 2/3·4) / 5
    assert parse('f1(prediction, target, average="weighted")').evaluate(columns) == pytest.approx((0.5 + 8 / 3) / 5)
    # micro F1 = accuracy = 3/5
    assert parse('f1(prediction, target, "micro")').evaluate(columns) == pytest.approx(0.6)


def test_scalar_arithmetic_on_aggregates():
    columns = cols(prediction=["a", "b"], target=["a", "a"])
    assert parse("mean(prediction == target) * 100").evaluate(columns) == pytest.approx(50.0)


def test_unit_expression_extracts_rowwise_part():
    formula = parse("wmean((a + b) / 2, weight)")
    unit = formula.unit_expression()
    assert unit.text == "(a + b) / 2"
    assert unit.evaluate_rows(cols(a=[1, 0], b=[1, 1])).tolist() == [1.0, 0.5]
    assert parse('f1(prediction, target, "macro")').unit_expression() is None


@pytest.mark.parametrize(
    "text",
    [
        "__import__('os')",
        "prediction.mean()",
        "[x for x in prediction]",
        "mean(prediction == target); 1",
        "lambda: 1",
        "prediction[0]",
        "open('f')",
    ],
)
def test_only_formula_language_is_accepted(text):
    with pytest.raises(FormulaError):
        parse(text)


def test_unknown_input_and_missing_column_are_reported():
    with pytest.raises(FormulaError, match="доступны"):
        parse("mean(score)").evaluate(cols(target=[1]))


def test_rowwise_result_without_aggregate_is_rejected():
    with pytest.raises(FormulaError, match="mean"):
        parse("prediction == target").evaluate(cols(prediction=["a"], target=["a"]))
