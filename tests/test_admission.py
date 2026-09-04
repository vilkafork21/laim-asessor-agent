"""Правило допуска судьи (карточка 6.3.3) и смещение судьи к разметчикам."""

from __future__ import annotations

import pytest

from admission import admit, judge_bias

SETTINGS = dict(
    min_holdout_units=20,
    min_holdout_defect_units=5,
    weak_holdout_defect_units=5,
    min_defect_recall=0.5,
    min_kappa=0.2,
    max_invalid_share=0.2,
)


def _metrics(**overrides):
    metrics = {
        "holdout_units": 40,
        "holdout_defect_units": 8,
        "invalid_share": 0.0,
        "defect_recall": 0.75,
        "cohen_kappa": 0.45,
    }
    metrics.update(overrides)
    return metrics


@pytest.mark.parametrize(
    "overrides, status, reason_code",
    [
        ({}, "green", "admitted"),
        ({"holdout_units": 12}, "not_assessed", "holdout_too_small"),
        ({"holdout_defect_units": 2}, "not_assessed", "critical_class_underrepresented"),
        ({"invalid_share": 0.3}, "red", "judge_refusals"),
        ({"defect_recall": 0.2, "cohen_kappa": 0.05}, "red", "no_better_than_baseline"),
        ({"defect_recall": 0.2, "cohen_kappa": None}, "red", "no_better_than_baseline"),
        ({"defect_recall": 0.2}, "amber", "weak_agreement"),
        ({"cohen_kappa": None}, "amber", "weak_agreement"),
        ({"cohen_kappa": 0.1}, "amber", "weak_agreement"),
    ],
)
def test_admission_rule(overrides, status, reason_code):
    result = admit(_metrics(**overrides), **SETTINGS)
    assert result.status == status
    assert result.reason_code == reason_code
    assert result.reason


def test_bias_is_judge_minus_human_with_interval():
    bias = judge_bias([1.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0, 1.0])
    assert bias["units"] == 5
    assert bias["mean"] == pytest.approx(0.0)
    assert bias["ci_lower"] < 0.0 < bias["ci_upper"]
    stricter = judge_bias([0.0, 0.0, 1.0], [1.0, 1.0, 1.0])
    assert stricter["mean"] == pytest.approx(-2 / 3)
    assert stricter["ci_upper"] < 0.0


def test_bias_needs_two_pairs():
    assert judge_bias([1.0], [1.0]) is None
    assert judge_bias([], []) is None
    with pytest.raises(ValueError, match="одной длины"):
        judge_bias([1.0, 0.0], [1.0])


def test_few_critical_units_is_amber_between_minimum_and_weak_threshold():
    settings = {**SETTINGS, "min_holdout_defect_units": 4, "weak_holdout_defect_units": 10}
    few = admit(_metrics(holdout_defect_units=4), **settings)
    assert few.status == "amber" and few.reason_code == "few_critical_units"
    assert admit(_metrics(holdout_defect_units=12), **settings).status == "green"
    # Более сильные основания не перекрываются порогом «мало дефектов».
    weak = admit(_metrics(holdout_defect_units=4, cohen_kappa=0.1), **settings)
    assert weak.reason_code == "weak_agreement"
    assert admit(_metrics(holdout_defect_units=3), **settings).reason_code == "critical_class_underrepresented"
