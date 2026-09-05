"""Домен ответа задаёт утверждённая рубрика, включая отсутствующие в RAG классы."""
import pytest
from pydantic import ValidationError

from agent.pydantic_output import create_simple_output_model


@pytest.mark.parametrize("values", [[0, 1], [0.0, 1.0], [0, 1, 2], [-1, 0, 1]])
def test_explicit_scale_accepts_each_declared_value_and_string_label(values):
    model = create_simple_output_model(["assessment_score"], values)
    for value in values:
        assert model(assessment_score=value).model_dump() == {"assessment_score": value}
        assert model(assessment_score=str(value)).assessment_score == value
    assert all(isinstance(v, str) for v in model.model_json_schema()["properties"]["assessment_score"]["enum"])


@pytest.mark.parametrize("answer", [{}, {"assessment_score": 2}, {"assessment_score": True}, {"assessment_score": 1, "comment": "x"}])
def test_malformed_or_undeclared_answer_is_rejected(answer):
    model = create_simple_output_model(["assessment_score"], [0, 1])
    with pytest.raises(ValidationError):
        model(**answer)


@pytest.mark.parametrize("values", [[], [0, 0]])
def test_invalid_scale_is_rejected(values):
    with pytest.raises(ValueError):
        create_simple_output_model(["assessment_score"], values)


def test_judge_can_abstain_without_inventing_a_score():
    model = create_simple_output_model(["assessment_score"], [0, 1])
    assert model(assessment_score="not_assessable").assessment_score is None
    assert model(assessment_score=None).assessment_score is None
