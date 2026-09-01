"""
Тесты для модуля pydantic_output.
"""

import pandas as pd
import pytest
from pydantic import BaseModel

from agent.pydantic_output import (
    infer_pydantic_type,
    infer_column_type,
    create_simple_output_model,
    validate_output,
)


class TestInferPydanticType:
    """Тесты для функции infer_pydantic_type."""

    def test_infer_int(self):
        assert infer_pydantic_type(pd.Series([1, 2, 3]).dtype) is int

    def test_infer_float(self):
        assert infer_pydantic_type(pd.Series([1.0, 2.0]).dtype) is float

    def test_infer_bool(self):
        assert infer_pydantic_type(pd.Series([True, False]).dtype) is bool

    def test_infer_str(self):
        assert infer_pydantic_type(pd.Series(["a", "b"]).dtype) is str

    def test_infer_nullable_int(self):
        dtype = pd.Series([1, 2, None], dtype="Int64").dtype
        assert infer_pydantic_type(dtype) is int


class TestInferColumnType:
    """Тесты для функции infer_column_type."""

    def test_column_with_few_unique_values(self):
        """Тест: столбец с малым количеством уникальных значений."""
        column = pd.Series([1, 2, 3, 1, 2])
        pydantic_type, literal_values = infer_column_type(column, max_unique_values=10)
        assert pydantic_type is int
        assert literal_values is not None
        assert set(literal_values) == {1, 2, 3}

    def test_column_with_many_unique_values(self):
        """Тест: столбец с большим количеством уникальных значений."""
        column = pd.Series(list(range(100)))
        pydantic_type, literal_values = infer_column_type(column, max_unique_values=10)
        assert pydantic_type is int
        assert literal_values is None

    def test_column_with_nulls(self):
        """Тест: столбец с None значениями."""
        column = pd.Series([1, 2, None, 3])
        pydantic_type, literal_values = infer_column_type(column)
        assert pydantic_type is int

    def test_column_with_string_values(self):
        """Тест: столбец со строковыми значениями."""
        column = pd.Series(["yes", "no", "yes", "no"])
        pydantic_type, literal_values = infer_column_type(column)
        assert pydantic_type is str


class TestCreateOutputModel:
    """Тесты для функции create_output_model."""

    def test_create_model_with_int_column(self):
        """Тест: создание модели с целочисленным столбцом."""
        df = pd.DataFrame({
            "score": [1, 2, 3, 4, 5],
        })
        model = create_simple_output_model(["score"], df)

        assert issubclass(model, BaseModel)
        instance = model(score=5)
        assert instance.score == 5

    def test_create_model_with_bool_column(self):
        """Тест: создание модели с булевым столбцом."""
        df = pd.DataFrame({
            "is_correct": [True, False, True],
        })
        model = create_simple_output_model(["is_correct"], df)

        instance = model(is_correct=True)
        assert instance.is_correct is True

    def test_create_model_with_mixed_columns(self):
        """Тест: создание модели со смешанными типами."""
        df = pd.DataFrame({
            "score": [1, 2, 3],
            "is_correct": [True, False, True],
            "comment": ["good", "bad", "ok"]
        })
        model = create_simple_output_model(
            ["score", "is_correct", "comment"], df
        )

        instance = model(score=3, is_correct=True, comment="good")
        assert instance.score == 3
        assert instance.is_correct is True
        assert instance.comment == "good"

        with pytest.raises(ValueError):
            model(score=5, is_correct=True, comment="excellent")

    def test_create_model_with_none_values(self):
        """Тест: создание модели с nullable полями."""
        df = pd.DataFrame({
            "score": [1, 2, None, 4],
        })
        model = create_simple_output_model(["score"], df)

        # Создание с None значением
        instance = model(score=None)
        assert instance.score is None

        # Создание с конкретным значением
        instance = model(score=4)
        assert instance.score == 4

    def test_invalid_column_raises_error(self):
        """Тест: ошибка при несуществующей колонке."""
        df = pd.DataFrame({
            "score": [1, 2, 3],
        })

        with pytest.raises(ValueError, match="Колонка 'invalid' не найдена"):
            create_simple_output_model(["invalid"], df)

    def test_continuous_column_does_not_create_unbounded_literal(self):
        df = pd.DataFrame({"score": range(51)})
        model = create_simple_output_model(["score"], df)

        instance = model(score=100)

        assert instance.score == 100


class TestGigachatSchemaCompatibility:
    """Chat GigaChat принимает enum функции только из строк."""

    def test_numeric_scale_builds_valid_gigachat_function(self):
        from gigachat.models import Chat
        from langchain_gigachat.utils.function_calling import convert_to_gigachat_tool

        df = pd.DataFrame({"assessment_score": [1.0, 0.0, 1.0]})
        model = create_simple_output_model(["assessment_score"], df)

        enum = model.model_json_schema()["properties"]["assessment_score"]["enum"]
        assert enum == ["1.0", "0.0"]

        func = convert_to_gigachat_tool(model)["function"]
        Chat(messages=[], functions=[func])

    def test_scale_values_survive_string_labels(self):
        df = pd.DataFrame({
            "assessment_score": [1.0, 0.0, 1.0],
            "is_correct": [True, False, True],
        })
        model = create_simple_output_model(["assessment_score", "is_correct"], df)

        parsed = model(assessment_score="1.0", is_correct="False")
        assert parsed.assessment_score == 1.0
        assert isinstance(parsed.assessment_score, float)
        assert parsed.is_correct is False

        assert model(assessment_score=0.0, is_correct=True).model_dump() == {
            "assessment_score": 0.0,
            "is_correct": True,
        }


class TestValidateOutput:
    """Тесты для функции validate_output."""

    def test_validate_valid_data(self):
        """Тест: валидация корректных данных."""
        df = pd.DataFrame({
            "score": [1, 2, 3],
        })
        model = create_simple_output_model(["score"], df)

        success, errors = validate_output(model, {"score": 3})
        assert success is True
        assert errors == []

    def test_validate_rejects_value_outside_observed_scale(self):
        df = pd.DataFrame({"score": [1, 2, 3]})
        model = create_simple_output_model(["score"], df)

        success, errors = validate_output(model, {"score": 4})

        assert success is False
        assert errors

    def test_validate_invalid_data_strict(self):
        """Тест: валидация некорректных данных (strict=True)."""
        df = pd.DataFrame({
            "score": [1, 2, 3],
        })
        model = create_simple_output_model(["score"], df)

        # Пытаемся передать строку вместо int
        success, errors = validate_output(model, {"score": "five"}, strict=True)
        assert success is False
        assert len(errors) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
