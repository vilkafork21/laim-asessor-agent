"""Смоук main() без LLM: судья подменён детерминированной заглушкой.

Проверяется склейка узла: контракт → план судьи → разметка → scored_data →
assessment_result. Требует зависимостей ноды (langchain), иначе пропускается.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("langchain_gigachat")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as node  # noqa: E402
from laim_monitoring import aggregate_main_metric  # noqa: E402


class ScriptedJudge:
    """Истинный класс берётся из текста запроса после двоеточия; отказ — None."""

    def __init__(self, llm, embedding_model, dataset, context_columns, answer_columns,
                 instruction, domain_rag_path, instruction_structuring, instruction_summarization):
        self.answer_columns = answer_columns

    async def run(self, frame):
        answers = []
        for _, row in frame.iterrows():
            query = str(row["input_query"])
            if query.endswith("?"):
                answers.append(None)  # отказ судьи
                continue
            label = query.split(":", 1)[1].strip() if ":" in query else "a"
            answers.append({
                column: (1.0 if column == "assessment_score" else label)
                for column in self.answer_columns
            })
        return answers


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setattr(node, "Asessor", ScriptedJudge)
    monkeypatch.setattr(node, "GigaChatEmbeddings", lambda **kw: object())
    monkeypatch.setattr(node, "BoundedGigaChatEmbeddings", lambda e: e)
    monkeypatch.setattr(node, "ModelsConfig", lambda model=None: types.SimpleNamespace(
        contour_llm_configs={}, contour_configs={}, llm_params={}, contour="sigma", verify_ssl_certs=False,
    ))
    monkeypatch.setattr(node, "_build_judge_model", lambda model_id, config, llm_model: (object(), "fake"))


CONTRACT = {
    "contract_version": "laim-monitoring-metric.v3",
    "status": "computed",
    "basket_id": "CI1",
    "metric_name": "Macro F1",
    "assessment_mode": "qa",
    "formula": 'f1(prediction, target, "macro")',
    "inputs": [
        {"name": "prediction", "column": "класс_output_answer", "judged": False},
        {"name": "target", "column": "класс_reference_answer", "judged": True},
    ],
    "baseline": {"value": 0.5833, "recomputed_value": 0.58333, "reconciliation": "match"},
}

REFERENCE = pd.DataFrame({
    "query_id": [f"r{i}" for i in range(6)],
    "input_query": ["q: a", "q: b", "q: b", "q: b", "q: b", "q: a"],
    "output_answer": ["x"] * 6,
    "класс_output_answer": ["a", "a", "b", "b", "a", "a"],
    "класс_reference_answer": ["a", "b", "b", "b", "b", "a"],
    "main_metric": [1, 0, 1, 1, 0, 1],
})


def _monitoring(queries: list[str], predictions: list[str] | None) -> pd.DataFrame:
    frame = pd.DataFrame({
        "query_id": [f"m{i}" for i in range(len(queries))],
        "input_query": queries,
        "output_answer": ["y"] * len(queries),
    })
    if predictions is not None:
        frame["класс_output_answer"] = predictions
    return frame


def _run(monitoring: pd.DataFrame) -> dict:
    return node.main(
        reference_umr=REFERENCE, monitoring_metric=CONTRACT,
        assessor_instruction={"text": "Инструкция"}, monitoring_umr=monitoring, stage="combined",
    )


def test_judge_labels_land_in_contract_columns_and_formula_reproduces_km():
    result = _run(_monitoring(["q: a", "q: b", "q: b"], ["a", "a", "b"]))
    assessment = result["assessment_result"]
    assert assessment["status"] == "computed"
    assert assessment["scoring_semantics"] == "contract_formula"
    assert assessment["judge_fields"] == ["target"]
    assert result["acc_auto"] == 1.0
    scored = result["scored_data"]
    assert scored["класс_reference_answer"].tolist() == ["a", "b", "b"]
    assert scored["main_metric"].tolist() == [1.0, 0.0, 1.0]
    # pred a,a,b vs true a,b,b → F1(a)=2/3, F1(b)=2/3 → macro 2/3
    assert aggregate_main_metric(scored, CONTRACT)["value"] == pytest.approx(2 / 3)


def test_without_agent_prediction_judge_scores_directly_and_says_so():
    result = _run(_monitoring(["q: a", "q: b"], None))
    assessment = result["assessment_result"]
    assert assessment["scoring_semantics"] == "judge_final_score"
    assert assessment["formula"] == "mean(assessment_score)"
    assert result["scored_data"]["main_metric"].tolist() == [1.0, 1.0]


def test_too_many_judge_refusals_make_result_not_computable():
    result = _run(_monitoring(["q: a", "почему?", "зачем?", "как?"], ["a", "a", "b", "b"]))
    assessment = result["assessment_result"]
    assert assessment["status"] == "not_computable"
    assert "75%" in assessment["reason"]
    assert assessment["scored_units"] == 1
