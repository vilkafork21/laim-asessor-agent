"""Сцепка контура на диалоговом агенте: формы выходов adapter и TDC как есть."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import main as assessor


def _metric() -> dict[str, object]:
    """Контракт adapter для диалоговой корзины с тремя голосами разметчиков."""
    return {
        "contract_version": "laim-monitoring-metric.v2",
        "umr_version": "laim-umr.v2",
        "status": "computed",
        "basket_id": "CI09997554",
        "name": "Accuracy",
        "score_column": "main_metric",
        "assessment_mode": "dialogue",
        "scoring": {
            "method": "all_assessors",
            "sources": [
                {
                    "source_id": f"source_{index}",
                    "column_name": f"mark_{index}_metric",
                    "role": "assessor_vote",
                    "normalization": "numeric",
                    "polarity": "direct",
                }
                for index in (1, 2, 3)
            ],
            "missing_policy": "fail",
            "majority_denominator": None,
        },
        "aggregation": {"method": "mean", "weight_column": None},
        "baseline": {
            "value": 0.92, "scale": "ratio", "value_source": "validation_report",
            "reported_value": 0.92, "reported_scale": "ratio",
            "recomputed_value": 0.9293, "reconciliation": "match",
        },
        "primary_validation": {
            "threshold": None, "comparator": None, "scale": "ratio",
            "verdict": None, "affects_monitoring": False,
        },
        "evidence": {},
    }


def _reference_packed() -> pd.DataFrame:
    """Packed reference — ровно то, что кладёт в порт reference_umr адаптер."""
    return pd.DataFrame({
        "session_id": ["s1", "s2", "s3"],
        "dialogue": [
            "[('t1', 'почему отказали', 'Причины отказа...'), ('t2', 'что делать', 'Проверьте историю...')]",
            "[('t3', 'кредитный потенциал', 'Ваш потенциал...')]",
            "[('t4', 'погасить займ', 'Для погашения...')]",
        ],
        "input_query_count": [1, 1, 1],
        "mark_1_metric": [1.0, 1.0, 0.0],
        "mark_2_metric": [1.0, 1.0, 1.0],
        "mark_3_metric": [1.0, 0.0, 1.0],
        "main_metric": [1.0, 0.0, 0.0],
    })


def _monitoring_packed() -> pd.DataFrame:
    """Packed monitoring — то, что кладёт в порт monitoring_umr новый TDC."""
    return pd.DataFrame({
        "scenario": ["decline", "potential"],
        "session_id": ["m1", "m2"],
        "dialogue": [
            "[('mt1', 'почему отказали в кредите', 'Отказ может быть...'), ('mt2', 'а подробнее', 'Подробнее: ...')]",
            "[('mt3', 'какой у меня потенциал', 'Ваш кредитный потенциал...')]",
        ],
        "input_query_count": [1, 1],
    })


def test_dialogue_judge_predicts_only_final_main_metric(monkeypatch) -> None:
    contract = assessor._assessment_contract(_metric())
    source = contract["scoring"]["sources"]

    assert contract["scoring"]["method"] == "identity"
    assert source == [{
        "source_id": "assessment_score",
        "column_name": "main_metric",
        "role": "final_score",
        "normalization": "numeric",
        "polarity": "direct",
    }]
    source_instruction = assessor._source_instruction(contract)
    assert "assessment_score" in source_instruction
    assert "mark_1_metric" not in source_instruction

    reference = assessor.normalize_umr(_reference_packed(), _metric())
    units = assessor._assessor_units(reference, contract, require_sources=True)
    captured: dict[str, object] = {}

    monkeypatch.setattr(assessor, "_build_assessor", lambda *_args: object())

    def fake_predict(_judge, frame, source_ids, _count):
        captured["source_ids"] = source_ids
        return pd.DataFrame({
            "agent_assessment_score": frame["assessment_score"].tolist(),
        })

    monkeypatch.setattr(assessor, "_predict", fake_predict)

    metrics, test_units, predictions = assessor._calibrate(
        units,
        ["assessment_score"],
        "Оцените весь диалог.",
        None,
        (object(), object()),
        0.67,
        1,
        False,
    )

    assert captured["source_ids"] == ["assessment_score"]
    assert units["assessment_score"].tolist() == units["main_metric"].tolist()
    assert test_units["assessment_score"].tolist() == test_units["main_metric"].tolist()
    assert predictions.columns.tolist() == ["agent_assessment_score"]
    assert metrics["acc_auto"] == 1.0


def test_dialogue_contour_scores_monitoring_sessions(monkeypatch) -> None:
    """Packed reference + packed monitoring проходят весь main() ассессора."""
    captured: dict[str, object] = {}

    def fake_calibrate(rag_units, source_ids, *args, **kwargs):
        captured["judge_units"] = rag_units
        test_units = rag_units.iloc[:1].reset_index(drop=True)
        predictions = pd.DataFrame({
            f"agent_{source_id}": [1.0] for source_id in source_ids
        })
        return {"acc_auto": 1.0}, test_units, predictions

    def fake_predict(_judge, frame, source_ids, _count):
        captured["monitoring_units"] = frame
        return pd.DataFrame({
            f"agent_{source_id}": [1.0, 0.0] for source_id in source_ids
        })

    monkeypatch.setattr(
        assessor, "ModelsConfig", lambda **_kwargs: SimpleNamespace(contour_configs={})
    )
    monkeypatch.setattr(assessor, "GigaChatEmbeddings", lambda **_kwargs: object())
    monkeypatch.setattr(assessor, "_build_judge_model", lambda *_args: (object(), "judge"))
    monkeypatch.setattr(assessor, "_build_assessor", lambda *_args: object())
    monkeypatch.setattr(assessor, "_calibrate", fake_calibrate)
    monkeypatch.setattr(assessor, "_predict", fake_predict)
    monkeypatch.setattr(
        assessor, "_load_instruction", lambda _value: "Оцените ответы по кредитам."
    )

    result = assessor.main(
        reference_umr=_reference_packed(),
        monitoring_metric=_metric(),
        assessor_instruction=Path("instruction.txt"),
        monitoring_umr=_monitoring_packed(),
        stage="combined",
    )

    # Судья калибруется на одном итоговом score целого диалога, а не на голосах.
    judge_units = captured["judge_units"]
    assert len(judge_units) == 3
    assert "assessment_score" in judge_units
    assert judge_units["assessment_score"].tolist() == [1.0, 0.0, 0.0]
    assert not {"source_1", "source_2", "source_3"} & set(judge_units.columns)
    # Monitoring-единицы — диалоги, контекст несёт все реплики сессии
    monitoring_units = captured["monitoring_units"]
    assert len(monitoring_units) == 2
    first_context = monitoring_units.loc[0, "assessment_context"]
    assert first_context["mode"] == "dialogue"
    assert [turn["input_query"] for turn in first_context["turns"]] == [
        "почему отказали в кредите", "а подробнее",
    ]
    # Оценка диалога размазана на все его turn-строки
    scored = result["scored_data"]
    assert len(scored) == 3  # 2 + 1 turn после нормализации packed monitoring
    assert scored["main_metric"].tolist() == [1.0, 1.0, 0.0]
    assert scored["agent_assessment_score"].tolist() == [1.0, 1.0, 0.0]
    assert not {"agent_source_1", "agent_source_2", "agent_source_3"} & set(scored)
    assert scored["assessor_id"].tolist() == ["judge", "judge", "judge"]
    assert result["assessment_result"]["status"] == "computed"
    assert result["assessment_result"]["assessment_mode"] == "dialogue"
