"""Схема ответа судьи по утверждённой шкале, независимо от RAG-примеров."""
from typing import Annotated, Any, Literal
from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, create_model

def _scale_field_type(literal_values: tuple[Any, ...]) -> Any:
    """Кодирует шкалу строковыми метками, возвращая исходные значения после валидации.

    Chat GigaChat принимает enum параметров функции только из строк: числовая
    шкала (например, 0.0/1.0) в Literal роняла каждый запрос ещё до отправки.
    """
    labels = tuple(str(value) for value in literal_values)
    if len(set(labels)) != len(labels):
        raise ValueError(f"Строковые метки шкалы неоднозначны: {sorted(labels)}")
    originals = dict(zip(labels, literal_values, strict=True))
    labels += ("not_assessable",)
    originals["not_assessable"] = None

    def normalize_label(value: Any) -> str:
        if value is None:
            return "not_assessable"
        if type(value) in (int, float):
            for label, original in originals.items():
                if type(original) in (int, float) and value == original:
                    return label
        return value if isinstance(value, str) else str(value)

    return Annotated[
        Literal[labels],
        BeforeValidator(normalize_label),
        AfterValidator(originals.__getitem__),
    ]


def create_simple_output_model(
    answer_columns: list[str], score_values: list[float],
    model_name: str = "AssessmentOutput",
) -> type[BaseModel]:
    if not score_values or len(set(score_values)) != len(score_values):
        raise ValueError("Нужна явная шкала разных значений")
    field_type = _scale_field_type(tuple(score_values))
    fields = {name: (field_type, Field(..., description="Итоговая оценка по утверждённой рубрике"))
              for name in answer_columns}
    return create_model(model_name, __config__=ConfigDict(
        extra="forbid", json_schema_extra={"description": "Оценка ответа агента"}), **fields)
