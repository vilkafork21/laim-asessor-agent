"""
Тесты для утилит в utils.py.
"""

import pandas as pd
import pytest

from utils import get_mode_safe, _to_hashable, _get_assessor_columns


class TestToHashable:
    """Тесты для функции _to_hashable."""

    def test_hashable_types_unchanged(self):
        """Тест: хешируемые типы возвращаются как есть."""
        assert _to_hashable(1) == 1
        assert _to_hashable("str") == "str"
        assert _to_hashable(True) is True

    def test_list_converted_to_tuple(self):
        """Тест: list конвертируется в tuple."""
        assert _to_hashable([1, 2, 3]) == (1, 2, 3)

    def test_dict_converted_to_sorted_tuple(self):
        """Тест: dict конвертируется в отсортированный tuple."""
        result = _to_hashable({"b": 2, "a": 1})
        assert result == (("a", 1), ("b", 2))


class TestGetModeSafe:
    """Тесты для функции get_mode_safe."""

    def test_empty_list(self):
        """Тест: пустой список возвращает None."""
        assert get_mode_safe([]) is None

    def test_single_element(self):
        """Тест: список с одним элементом."""
        assert get_mode_safe([1]) == 1

    def test_most_common_value(self):
        """Тест: возврат самого частого значения."""
        assert get_mode_safe([1, 2, 1, 3, 1]) == 1

    def test_with_none_values(self):
        """Тест: игнорирование None значений."""
        result = get_mode_safe([1, 2, None, 1, None])
        assert result == 1

    def test_with_unhashable_values(self):
        """Тест: работа с нехешируемыми типами (list)."""
        result = get_mode_safe([[1, 2], [3, 4], [1, 2]])
        assert result == [1, 2]

    def test_string_values(self):
        """Тест: работа со строковыми значениями."""
        assert get_mode_safe(["a", "b", "a", "c"]) == "a"


class TestGetAssessorColumns:
    """Тесты для функции _get_assessor_columns."""

    def test_find_single_column(self):
        """Тест: поиск одной колонки."""
        df = pd.DataFrame({
            "agent_score": [1, 2, 3],
            "other_col": [4, 5, 6]
        })
        result = _get_assessor_columns(df, "score")
        assert result == ["agent_score"]

    def test_find_multiple_columns(self):
        """Тест: поиск нескольких колонок."""
        df = pd.DataFrame({
            "agent_1_score": [1, 2, 3],
            "agent_2_score": [4, 5, 6],
            "other_col": [7, 8, 9]
        })
        result = _get_assessor_columns(df, "score")
        assert len(result) == 2
        assert "agent_1_score" in result
        assert "agent_2_score" in result

    def test_no_matching_columns(self):
        """Тест: отсутствие matching колонок."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6]
        })
        result = _get_assessor_columns(df, "nonexistent")
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])