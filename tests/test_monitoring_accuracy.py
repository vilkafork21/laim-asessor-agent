from __future__ import annotations

from tests.measurement_fixture import reviewed_metric

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import main as assessor


def _metric() -> dict[str, object]:
    return reviewed_metric({
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
    })


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
    ).assign(definition_id=_metric()["definition_id"], evaluation_ready=True, dataset_role="reference")


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
    ).assign(definition_id=_metric()["definition_id"], evaluation_ready=True, dataset_role="monitoring")


def test_accuracy_is_assessed_from_reference_main_metric() -> None:
    contract = _metric()

    units = assessor._assessor_units(_reference(), contract, require_sources=True)
    assert contract["scoring"]["method"] == "accuracy"
    assert units["assessment_score"].tolist() == _reference()["main_metric"].tolist()


def test_monitoring_accuracy_without_prediction_column_is_not_computable() -> None:
    result = assessor.main(
        reference_umr=_reference(), monitoring_metric=_metric(),
        monitoring_umr=_monitoring().drop(columns=["class"]), stage="combined",
    )

    assert result["assessment_result"]["status"] == "not_computable"
    assert result["assessment_result"]["reason_code"] == "missing_prediction"
    assert result["scored_data"]["main_metric"].isna().all()


def test_monitoring_accuracy_does_not_require_gt(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_calibrate(rag_units, source_ids, *args, **kwargs):
        captured["source_ids"] = source_ids
        test_units = rag_units.iloc[:2].reset_index(drop=True)
        predictions = pd.DataFrame({"agent_assessment_score": [1.0, 0.0]})
        return {"acc_auto": 0.875}, test_units, predictions, object()

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

    result = assessor.main(
        reference_umr=_reference(),
        monitoring_metric=_metric(),

        monitoring_umr=_monitoring(),
        stage="combined",
    )

    assert captured["source_ids"] == ["assessment_score"]
    monitoring_context = captured["monitoring_context"]
    assert isinstance(monitoring_context, dict)
    assert monitoring_context["current_turn"]["output_answer"] == "answer-1"
    assert monitoring_context["observations"][0]["observed_prediction"] == "route-a"
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


_ADMISSION = dict(
    min_holdout_units=2,
    min_holdout_defect_units=1,
    weak_holdout_defect_units=1,
    min_defect_recall=0.5,
    min_kappa=0.2,
    max_invalid_share=0.2,
)


def _calibrate(monkeypatch, rag_units, fake_predict, **overrides):
    monkeypatch.setattr(assessor, "_build_assessor", lambda *_args: object())
    monkeypatch.setattr(assessor, "_predict", fake_predict)
    return assessor._calibrate(
        rag_units,
        ["assessment_score"],
        "инструкция",
        None,
        (object(), object()),
        0.5,
        1,
        False,
        assessment_contract=_metric(),
        admission_settings={**_ADMISSION, **overrides},
    )


def test_calibrate_logs_agreement_metrics(monkeypatch, caplog) -> None:
    """Калибровка публикует каппу, альфу, Спирмана и допуск, а не только accuracy."""
    import logging

    rag_units = pd.DataFrame({"assessment_score": [1.0, 0.0, 1.0, 0.0]})

    def fake_predict(_judge, test, source_ids, _count):
        return pd.DataFrame(
            {"agent_assessment_score": test["assessment_score"].tolist()}
        )

    with caplog.at_level(logging.INFO):
        metrics, _test, _predictions, _judge = _calibrate(monkeypatch, rag_units, fake_predict)

    assert metrics["acc_auto"] == 1.0
    assert metrics["cohen_kappa"] == 1.0
    assert metrics["krippendorff_alpha"] == 1.0
    # Каппа по горстке дефектов недостоверна: потребитель метрик обязан видеть,
    # на скольких единицах она посчитана.
    assert metrics["holdout_units"] == 2
    assert metrics["holdout_defect_units"] == 1
    assert metrics["invalid_share"] == 0.0
    assert metrics["admission_status"] == "green"
    assert metrics["bias_mean"] == 0.0
    assert "каппа Коэна" in caplog.text and "допуск=green" in caplog.text


def test_calibrate_measures_judge_bias_and_admission(monkeypatch) -> None:
    """Судья строже разметчиков: смещение отрицательное, допуск по правилу."""
    rag_units = pd.DataFrame({"assessment_score": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]})

    def stricter_judge(_judge, test, source_ids, _count):
        scores = test["assessment_score"].tolist()
        scores[0] = 0.0  # одну верную единицу судья счёл дефектом
        return pd.DataFrame({"agent_assessment_score": scores})

    metrics, _test, _predictions, _judge = _calibrate(monkeypatch, rag_units, stricter_judge)

    assert metrics["bias_units"] == metrics["holdout_units"]
    assert metrics["bias_mean"] < 0.0
    assert metrics["bias_ci_lower"] <= metrics["bias_mean"] <= metrics["bias_ci_upper"]
    assert metrics["defect_recall"] == 1.0
    assert metrics["admission_status"] in {"green", "amber"}
    assert metrics["admission_reason"]

    strict = _calibrate(monkeypatch, rag_units, stricter_judge, min_holdout_units=50)[0]
    assert strict["admission_status"] == "not_assessed"
    assert strict["admission_reason_code"] == "holdout_too_small"


def test_monitoring_refusals_are_counted(monkeypatch) -> None:
    """Отказ судьи на мониторинге — счётчик и статус, а не падение ноды."""

    def fake_calibrate(rag_units, source_ids, *args, **kwargs):
        test_units = rag_units.iloc[:2].reset_index(drop=True)
        return {"acc_auto": 0.875}, test_units, pd.DataFrame({"agent_assessment_score": [1.0, 0.0]}), object()

    def refusing_predict(_judge, frame, source_ids, _count):
        return pd.DataFrame({"agent_assessment_score": [1.0, None]})

    monkeypatch.setattr(
        assessor, "ModelsConfig", lambda **_kwargs: SimpleNamespace(contour_configs={})
    )
    monkeypatch.setattr(assessor, "GigaChatEmbeddings", lambda **_kwargs: object())
    monkeypatch.setattr(assessor, "_build_judge_model", lambda *_args: (object(), "judge"))
    monkeypatch.setattr(assessor, "_build_assessor", lambda *_args: object())
    monkeypatch.setattr(assessor, "_calibrate", fake_calibrate)
    monkeypatch.setattr(assessor, "_predict", refusing_predict)
    kwargs = dict(
        reference_umr=_reference(),
        monitoring_metric=_metric(),

        monitoring_umr=_monitoring(),
        stage="combined",
    )

    tolerant = assessor.main(**kwargs, max_invalid_share=0.5)["assessment_result"]
    assert tolerant["status"] == "computed"
    assert tolerant["refused_units"] == 1 and tolerant["refused_share"] == 0.5
    assert tolerant["scored_units"] == 1 and tolerant["total_units"] == 2

    capped = assessor.main(**kwargs, max_invalid_share=0.2)["assessment_result"]
    assert capped["status"] == "not_computable"
    assert capped["reason_code"] == "judge_refusals" and capped["refused_units"] == 1


def test_descriptor_declares_admission_settings() -> None:
    import json

    descriptor = json.loads(
        (Path(__file__).resolve().parents[1] / "descriptor.json").read_text("utf-8")
    )
    defaults = {
        item["parameter"]: item["defaultValue"]
        for section in descriptor["ui"]["settings"]
        for component in section["components"]
        for item in component["config"]["components"]
    }
    assert defaults["min_holdout_units"] == 20
    assert defaults["min_holdout_defect_units"] == 4
    assert defaults["weak_holdout_defect_units"] == 10
    assert defaults["min_defect_recall"] == 0.5
    assert defaults["min_kappa"] == 0.2
    assert defaults["max_invalid_share"] == 0.2
    assert "admission.py" in descriptor["script"]["runConfiguration"]["sourceFiles"]


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
        return metrics, test_units, pd.DataFrame({"agent_assessment_score": [1.0, 0.0]}), object()

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

    result = assessor.main(
        reference_umr=_reference(),
        monitoring_metric=_metric(),

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
