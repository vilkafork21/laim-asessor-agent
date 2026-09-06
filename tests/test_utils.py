"""
Тесты для утилит в utils.py.
"""

import asyncio
import time

import pandas as pd
import pytest

import utils
from utils import get_mode_safe, read_docx, _to_hashable, _get_assessor_columns


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


def test_read_utf8_txt_instruction(tmp_path):
    instruction = tmp_path / "instruction.txt"
    instruction.write_text("Оцените диалог целиком.", encoding="utf-8")

    assert read_docx(str(instruction)) == "Оцените диалог целиком."


def test_read_docx_instruction_without_extension(tmp_path):
    from docx import Document

    instruction = tmp_path / "unstructured_data"
    document = Document()
    document.add_paragraph("Оцените диалог целиком.")
    document.save(instruction)

    assert read_docx(str(instruction)) == "Оцените диалог целиком."


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

class TestAssessorColumnsExactMatch:
    """source_1 не должен захватывать колонки source_10 (подстрочный матч)."""

    def test_source_prefix_does_not_capture_longer_ids(self):
        df = pd.DataFrame({
            "agent_0_source_1": [1],
            "agent_1_source_1": [1],
            "agent_0_source_10": [0],
        })
        assert _get_assessor_columns(df, "source_1") == [
            "agent_0_source_1", "agent_1_source_1",
        ]

    def test_plain_agent_column_still_matches(self):
        df = pd.DataFrame({"agent_score": [1], "agent_score_extra": [0]})
        assert _get_assessor_columns(df, "score") == ["agent_score"]


def test_lowest_allowed_values_only_for_numeric_scales():
    """Перепроверять «низкую» оценку осмысленно лишь на числовой шкале."""
    from agent.asessor_agent import _lowest_allowed_values

    assert _lowest_allowed_values({"score": {0.0, 1.0}}) == {"score": 0.0}
    assert _lowest_allowed_values({"route": {"rag", "deposelector"}}) == {}


def test_answer_is_lowest_score_when_any_criterion_is_lowest():
    from agent.asessor_agent import _is_lowest_score

    lowest = {"score": 0.0}
    assert _is_lowest_score({"score": 0.0}, lowest) is True
    assert _is_lowest_score({"score": 1.0}, lowest) is False
    assert _is_lowest_score({}, lowest) is False


def test_process_with_rate_limit_retrieves_once_and_passes_prompt_to_judge():
    class RetrievalJudgeChain:
        def __init__(self):
            self.retrieval_inputs = []
            self.judge_inputs = []

        async def ainvoke(self, value):
            self.retrieval_inputs.append(value)
            prompt = {"user_input": value}
            self.judge_inputs.append(prompt)
            return {"score": 1}

    chain = RetrievalJudgeChain()

    result = asyncio.run(
        utils.process_with_rate_limit(chain, [{"trace_id": "one"}], delay_seconds=0)
    )

    assert result == [{"score": 1}]
    assert chain.retrieval_inputs == [{"trace_id": "one"}]
    assert chain.judge_inputs == [{"user_input": {"trace_id": "one"}}]


def test_queued_request_waits_for_quota_cooldown(monkeypatch):
    class QuotaChain:
        def __init__(self):
            self.first_attempt_failed_at = None
            self.queued_started_at = None

        async def ainvoke(self, value):
            if value == "first" and self.first_attempt_failed_at is None:
                await asyncio.sleep(0)
                self.first_attempt_failed_at = time.monotonic()
                raise RuntimeError("429 retry-after 0.1")
            if value == "queued":
                self.queued_started_at = time.monotonic()
            return value

    monkeypatch.setattr(utils, "MAX_INFLIGHT_REQUESTS", 1)
    chain = QuotaChain()

    result = asyncio.run(
        utils.process_with_rate_limit(
            chain,
            ["first", "queued"],
            delay_seconds=0,
        )
    )

    assert result == ["first", "queued"]
    assert chain.queued_started_at - chain.first_attempt_failed_at >= 0.08
