from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import main as assessor


def _metric() -> dict[str, object]:
    return {
        "contract_version": "laim-monitoring-metric.v2",
        "umr_version": "laim-umr.v2",
        "status": "computed",
        "basket_id": "CI09997438",
        "name": "взвешенная accuracy",
        "score_column": "main_metric",
        "assessment_mode": "qa",
        "scoring": {
            "method": "accuracy",
            "sources": [
                {
                    "source_id": "source_1",
                    "column_name": "class",
                    "role": "prediction",
                    "normalization": "label",
                    "polarity": "direct",
                },
                {
                    "source_id": "source_2",
                    "column_name": "GT",
                    "role": "target",
                    "normalization": "label",
                    "polarity": "direct",
                },
            ],
            "missing_policy": "fail",
            "majority_denominator": None,
        },
        "aggregation": {
            "method": "frequency_weighted_mean",
            "weight_column": "input_query_count",
        },
        "baseline": {
            "value": 0.94,
            "scale": "ratio",
            "recomputed_value": 0.94,
        },
        "primary_validation": {"affects_monitoring": False},
    }


def _reference() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": ["r1", "r2", "r3", "r4"],
            "input_query": ["q1", "q2", "q3", "q4"],
            "output_answer": ["route-a", "route-b", "route-a", "route-b"],
            "class": ["route-a", "route-b", "route-a", "route-b"],
            "GT": ["route-a", "route-a", "route-a", "route-b"],
            "main_metric": [1.0, 0.0, 1.0, 1.0],
            "input_query_count": [1, 1, 1, 1],
        }
    )


def _monitoring() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": ["m1", "m2"],
            "input_query": ["q5", "q6"],
            "output_answer": ["answer-1", "answer-2"],
            "agent_response": ["answer-1", "answer-2"],
            "class": ["route-a", "route-b"],
            "input_query_count": [1, 1],
        }
    )


def test_accuracy_is_assessed_from_reference_main_metric() -> None:
    contract = assessor._assessment_contract(_metric())

    assert contract["scoring"]["method"] == "identity"
    assert contract["scoring"]["sources"] == [
        {
            "source_id": "assessment_score",
            "column_name": "main_metric",
            "role": "final_score",
            "normalization": "numeric",
            "polarity": "direct",
        }
    ]


def test_monitoring_accuracy_without_prediction_column_is_judged(monkeypatch) -> None:
    """Трейсы не несут колонок размеченной корзины: судья оценивает output_answer."""
    captured: dict[str, object] = {}

    def fake_calibrate(rag_units, source_ids, *args, **kwargs):
        test_units = rag_units.iloc[:2].reset_index(drop=True)
        predictions = pd.DataFrame({"agent_assessment_score": [1.0, 0.0]})
        return {"acc_auto": 0.875}, test_units, predictions

    def fake_predict(_judge, frame, source_ids, _count):
        captured["monitoring_context"] = frame.loc[0, "assessment_context"]
        return pd.DataFrame({"agent_assessment_score": [1.0, 0.0]})

    monkeypatch.setattr(
        assessor,
        "ModelsConfig",
        lambda **_kwargs: SimpleNamespace(contour_configs={}),
    )
    monkeypatch.setattr(assessor, "GigaChatEmbeddings", lambda **_kwargs: object())
    monkeypatch.setattr(
        assessor, "_build_judge_model", lambda *_args: (object(), "judge")
    )
    monkeypatch.setattr(assessor, "_build_assessor", lambda *_args: object())
    monkeypatch.setattr(assessor, "_calibrate", fake_calibrate)
    monkeypatch.setattr(assessor, "_predict", fake_predict)
    monkeypatch.setattr(
        assessor,
        "_load_instruction",
        lambda _value: "Оцените корректность выбранного маршрута.",
    )

    result = assessor.main(
        reference_umr=_reference(),
        monitoring_metric=_metric(),
        assessor_instruction=Path("instruction.txt"),
        monitoring_umr=_monitoring().drop(columns=["class"]),
        stage="combined",
    )

    monitoring_context = captured["monitoring_context"]
    assert isinstance(monitoring_context, dict)
    assert monitoring_context["current_turn"]["output_answer"] == "answer-1"
    assessment_result = result["assessment_result"]
    assert isinstance(assessment_result, dict)
    assert assessment_result["status"] == "computed"
    assert assessment_result["scored_units"] == 2
    assert result["scored_data"]["main_metric"].tolist() == [1.0, 0.0]


def test_monitoring_accuracy_does_not_require_gt(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_calibrate(rag_units, source_ids, *args, **kwargs):
        captured["source_ids"] = source_ids
        test_units = rag_units.iloc[:2].reset_index(drop=True)
        predictions = pd.DataFrame({"agent_assessment_score": [1.0, 0.0]})
        return {"acc_auto": 0.875}, test_units, predictions

    def fake_predict(_judge, frame, source_ids, _count):
        captured["monitoring_columns"] = list(frame.columns)
        captured["monitoring_context"] = frame.loc[0, "assessment_context"]
        assert source_ids == ["assessment_score"]
        return pd.DataFrame({"agent_assessment_score": [1.0, 0.0]})

    monkeypatch.setattr(
        assessor,
        "ModelsConfig",
        lambda **_kwargs: SimpleNamespace(contour_configs={}),
    )
    monkeypatch.setattr(assessor, "GigaChatEmbeddings", lambda **_kwargs: object())
    monkeypatch.setattr(
        assessor, "_build_judge_model", lambda *_args: (object(), "judge")
    )
    monkeypatch.setattr(assessor, "_build_assessor", lambda *_args: object())
    monkeypatch.setattr(assessor, "_calibrate", fake_calibrate)
    monkeypatch.setattr(assessor, "_predict", fake_predict)
    monkeypatch.setattr(
        assessor,
        "_load_instruction",
        lambda _value: "Оцените корректность выбранного маршрута.",
    )

    result = assessor.main(
        reference_umr=_reference(),
        monitoring_metric=_metric(),
        assessor_instruction=Path("instruction.txt"),
        monitoring_umr=_monitoring(),
        stage="combined",
    )

    assert captured["source_ids"] == ["assessment_score"]
    monitoring_context = captured["monitoring_context"]
    assert isinstance(monitoring_context, dict)
    assert monitoring_context["current_turn"]["output_answer"] == "route-a"
    assert "GT" not in _monitoring()
    assert result["acc_auto"] == 0.875
    assessment_result = result["assessment_result"]
    scored_data = result["scored_data"]
    assert isinstance(assessment_result, dict)
    assert isinstance(scored_data, pd.DataFrame)
    assert assessment_result["status"] == "computed"
    assert assessment_result["scored_units"] == 2
    assert scored_data["main_metric"].tolist() == [1.0, 0.0]
    assert scored_data["assessor_id"].tolist() == ["judge", "judge"]
    assert scored_data["output_answer"].tolist() == ["answer-1", "answer-2"]
    assert scored_data["agent_response"].tolist() == ["answer-1", "answer-2"]


def test_calibrate_prints_agreement_metrics(monkeypatch, capsys) -> None:
    """Калибровка печатает каппу, альфу и Спирмана, а не только accuracy."""
    rag_units = pd.DataFrame({"assessment_score": [1.0, 0.0, 1.0, 0.0]})

    def fake_predict(_judge, test, source_ids, _count):
        return pd.DataFrame(
            {"agent_assessment_score": test["assessment_score"].tolist()}
        )

    monkeypatch.setattr(assessor, "_build_assessor", lambda *_args: object())
    monkeypatch.setattr(assessor, "_predict", fake_predict)

    metrics, _test, _predictions = assessor._calibrate(
        rag_units,
        ["assessment_score"],
        "инструкция",
        None,
        (object(), object()),
        0.5,
        1,
        False,
    )

    assert metrics["acc_auto"] == 1.0
    assert metrics["cohen_kappa"] == 1.0
    assert metrics["krippendorff_alpha"] == 1.0
    # Каппа по горстке дефектов недостоверна: потребитель метрик обязан видеть,
    # на скольких единицах она посчитана.
    assert metrics["holdout_units"] == 2
    assert metrics["holdout_defect_units"] == 1
    output = capsys.readouterr().out
    assert "каппа Коэна" in output
    assert "альфа Криппендорфа" in output
    assert "Спирмана" in output


def test_calibration_metrics_reach_assessment_result(monkeypatch) -> None:
    """calibration_metrics доезжают до assessment_result и в combined-стадии."""
    metrics = {
        "acc_auto": 0.875,
        "baseline_mode_accuracy": 0.5,
        "cohen_kappa": 0.75,
        "krippendorff_alpha": 0.7,
        "spearman_correlation": 0.8,
    }

    def fake_calibrate(rag_units, source_ids, *args, **kwargs):
        test_units = rag_units.iloc[:2].reset_index(drop=True)
        return metrics, test_units, pd.DataFrame({"agent_assessment_score": [1.0, 0.0]})

    def fake_predict(_judge, frame, source_ids, _count):
        return pd.DataFrame({"agent_assessment_score": [1.0, 0.0]})

    monkeypatch.setattr(
        assessor,
        "ModelsConfig",
        lambda **_kwargs: SimpleNamespace(contour_configs={}),
    )
    monkeypatch.setattr(assessor, "GigaChatEmbeddings", lambda **_kwargs: object())
    monkeypatch.setattr(
        assessor, "_build_judge_model", lambda *_args: (object(), "judge")
    )
    monkeypatch.setattr(assessor, "_build_assessor", lambda *_args: object())
    monkeypatch.setattr(assessor, "_calibrate", fake_calibrate)
    monkeypatch.setattr(assessor, "_predict", fake_predict)
    monkeypatch.setattr(
        assessor,
        "_load_instruction",
        lambda _value: "Оцените корректность выбранного маршрута.",
    )

    result = assessor.main(
        reference_umr=_reference(),
        monitoring_metric=_metric(),
        assessor_instruction=Path("instruction.txt"),
        monitoring_umr=_monitoring(),
        stage="combined",
    )

    assert result["acc_auto"] == 0.875
    assert result["assessment_result"]["calibration_metrics"] == metrics


def test_split_keeps_defect_units_in_holdout() -> None:
    """Редкие дефекты обязаны попасть в holdout: иначе каппу не на чем считать."""
    units = pd.DataFrame({
        "_group_id": [f"g{index}" for index in range(20)],
        "assessment_score": [0.0, 0.0] + [1.0] * 18,
    })

    train, test = assessor._split_units(units, ["assessment_score"], 0.8)

    assert (test["assessment_score"] == 0.0).sum() >= 1
    assert (train["assessment_score"] == 0.0).sum() >= 1
    assert len(train) + len(test) == len(units)


def test_split_without_groups_is_stratified() -> None:
    units = pd.DataFrame({"assessment_score": [0.0, 0.0, 0.0] + [1.0] * 27})

    train, test = assessor._split_units(units, ["assessment_score"], 0.8)

    assert (test["assessment_score"] == 0.0).sum() >= 1
    assert len(train) + len(test) == len(units)
