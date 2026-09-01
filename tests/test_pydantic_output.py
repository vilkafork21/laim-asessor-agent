"""
Тесты для модуля pydantic_output.
"""

import pandas as pd
import pytest
from pydantic import BaseModel

from agent.pydantic_output import (
    infer_pydantic_type,
    infer_column_type,
    create_output_model,
    create_simple_output_model,
    validate_output,
)


class TestInferPydanticType:
    """Тесты для функции infer_pydantic_type."""

    def test_infer_int(self):
        assert infer_pydantic_type(pd.Series([1, 2, 3]).dtype) == int

    def test_infer_float(self):
        assert infer_pydantic_type(pd.Series([1.0, 2.0]).dtype) == float

    def test_infer_bool(self):
        assert infer_pydantic_type(pd.Series([True, False]).dtype) == bool

    def test_infer_str(self):
        assert infer_pydantic_type(pd.Series(["a", "b"]).dtype) == str

    def test_infer_nullable_int_is_float(self):
        # pandas хранит целые с пропусками как float64 — тип честно float:
        # выдавать его за int значило бы обещать судье целочисленную шкалу,
        # которой в данных нет.
        assert infer_pydantic_type(pd.Series([1, 2, None]).dtype) == float


class TestInferColumnType:
    """Тесты для функции infer_column_type."""

    def test_column_with_few_unique_values(self):
        """Тест: столбец с малым количеством уникальных значений."""
        column = pd.Series([1, 2, 3, 1, 2])
        pydantic_type, literal_values = infer_column_type(column, max_unique_values=10)
        assert pydantic_type == int
        # numpy-скаляры приводятся к питоновским: без этого Literal молча
        # отключался для целочисленных шкал (np.int64 не isinstance int).
        assert literal_values is not None
        assert set(literal_values) == {1, 2, 3}
        assert all(type(v) is int for v in literal_values)

    def test_column_with_many_unique_values(self):
        """Тест: столбец с большим количеством уникальных значений."""
        column = pd.Series(list(range(100)))
        pydantic_type, literal_values = infer_column_type(column, max_unique_values=10)
        assert pydantic_type == int
        assert literal_values is None

    def test_column_with_nulls(self):
        """Тест: столбец с None значениями — pandas делает его float64."""
        column = pd.Series([1, 2, None, 3])
        pydantic_type, literal_values = infer_column_type(column)
        assert pydantic_type == float
        assert literal_values is not None
        assert set(literal_values) == {1.0, 2.0, 3.0}

    def test_column_with_string_values(self):
        """Тест: столбец со строковыми значениями."""
        column = pd.Series(["yes", "no", "yes", "no"])
        pydantic_type, literal_values = infer_column_type(column)
        assert pydantic_type == str


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

        instance = model(score=5, is_correct=True, comment="excellent")
        assert instance.score == 5
        assert instance.is_correct is True
        assert instance.comment == "excellent"

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
        instance = model(score=5)
        assert instance.score == 5

    def test_invalid_column_raises_error(self):
        """Тест: ошибка при несуществующей колонке."""
        df = pd.DataFrame({
            "score": [1, 2, 3],
        })

        with pytest.raises(ValueError, match="Колонка 'invalid' не найдена"):
            create_simple_output_model(["invalid"], df)


class TestValidateOutput:
    """Тесты для функции validate_output."""

    def test_validate_valid_data(self):
        """Тест: валидация корректных данных."""
        df = pd.DataFrame({
            "score": [1, 2, 3],
        })
        model = create_simple_output_model(["score"], df)

        success, errors = validate_output(model, {"score": 5})
        assert success is True
        assert errors == []

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