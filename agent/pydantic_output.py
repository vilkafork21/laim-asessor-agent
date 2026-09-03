"""
Динамическое создание Pydantic моделей для structured output.

Основные функции:
- infer_pydantic_type: маппинг pandas dtype на Pydantic типы
- create_output_model: динамическое создание Pydantic модели на основе answer_columns
"""

from typing import Annotated, Any, Optional, Literal

import pandas as pd
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    create_model,
)

MAX_LITERAL_VALUES = 50


# =====================================================
# TYPE INFERENCE
# =====================================================


def infer_pydantic_type(dtype: pd.Series.dtype) -> type:
    """
    Маппинг pandas dtype на Pydantic типы.

    Args:
        dtype: pandas dtype столбца

    Returns:
        Соответствующий Python тип
    """
    dtype_str = str(dtype).lower()

    if "int" in dtype_str:
        return int
    elif "float" in dtype_str:
        return float
    elif "bool" in dtype_str:
        return bool
    else:
        return str


def infer_column_type(
    column: pd.Series, max_unique_values: int = 10
) -> tuple[type, Optional[tuple[Any, ...]]]:
    """
    Определяет тип столбца и возможные значения.

    Args:
        column: pandas Series (столбец датафрейма)
        max_unique_values: максимальное количество уникальных значений для использования Literal

    Returns:
        Кортеж: (тип поля, список возможных значений или None)
    """
    pydantic_type = infer_pydantic_type(column.dtype)
    has_nulls = column.isnull().any()
    unique_values = tuple(
        value.item() if hasattr(value, "item") else value
        for value in column.dropna().unique()
    )

    # pandas без nullable dtype повышает целые с пропусками до float64.
    if (
        pydantic_type is float
        and has_nulls
        and unique_values
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and float(value).is_integer()
            for value in unique_values
        )
    ):
        pydantic_type = int
        unique_values = tuple(int(value) for value in unique_values)

    # Проверяем, можно ли использовать Literal
    can_use_literal = (
        len(unique_values) <= max_unique_values
        and len(unique_values) > 0
        and all(isinstance(value, pydantic_type) for value in unique_values)
    )

    if can_use_literal:
        return pydantic_type, unique_values

    return pydantic_type, None


# =====================================================
# MODEL CREATION
# =====================================================


def create_output_model(
    answer_columns: list[str],
    dataset: pd.DataFrame,
    model_name: str = "AssessmentOutput",
) -> type[BaseModel]:
    """
    Динамически создаёт Pydantic модель на основе answer_columns.

    Типы полей определяются по dtype столбцов в dataset.
    Если столбец содержит мало уникальных значений (<=10),
    используется Literal для строгой валидации.

    Args:
        answer_columns: список названий колонок с ответами
        dataset: DataFrame с данными для определения типов
        model_name: название создаваемой модели

    Returns:
        Pydantic модель с динамически определёнными полями
    """
    field_definitions = {}

    for col in answer_columns:
        if col not in dataset.columns:
            raise ValueError(f"Колонка '{col}' не найдена в датасете")

        column = dataset[col]
        pydantic_type, literal_values = infer_column_type(column)

        # Определяем, может ли поле быть None
        has_nulls = column.isnull().any()

        if literal_values is not None:
            literal_type = Literal[literal_values]

            if has_nulls:
                field_definitions[col] = (Optional[literal_type], None)
            else:
                field_definitions[col] = (literal_type, ...)
        else:
            # Используем базовый тип
            if has_nulls:
                field_definitions[col] = (Optional[pydantic_type], None)
            else:
                field_definitions[col] = (pydantic_type, ...)

    return create_model(model_name, **field_definitions)


def _scale_field_type(literal_values: tuple[Any, ...]) -> Any:
    """Кодирует шкалу строковыми метками, возвращая исходные значения после валидации.

    Chat GigaChat принимает enum параметров функции только из строк: числовая
    шкала (например, 0.0/1.0) в Literal роняла каждый запрос ещё до отправки.
    """
    labels = tuple(str(value) for value in literal_values)
    if len(set(labels)) != len(labels):
        raise ValueError(f"Строковые метки шкалы неоднозначны: {sorted(labels)}")
    originals = dict(zip(labels, literal_values, strict=True))

    def normalize_label(value: Any) -> str:
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
    answer_columns: list[str],
    dataset: pd.DataFrame,
    model_name: str = "AssessmentOutput",
    field_descriptions: dict[str, str] | None = None,
) -> type[BaseModel]:
    """
    Создаёт модель со шкалой наблюдавшихся значений, если она содержит до 50 меток.

    Args:
        answer_columns: список названий колонок с ответами
        dataset: DataFrame с данными для определения типов
        model_name: название создаваемой модели
        field_descriptions: опциональный словарь с описаниями полей

    Returns:
        Pydantic модель со строгими допустимыми значениями
    """
    field_definitions = {}

    if field_descriptions is None:
        field_descriptions = {}

    # Add model-level description required by GigaChat
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Модель для оценки асессора",
        }
    )

    for col in answer_columns:
        if col not in dataset.columns:
            raise ValueError(f"Колонка '{col}' не найдена в датасете")

        column = dataset[col]
        pydantic_type, literal_values = infer_column_type(
            column,
            max_unique_values=MAX_LITERAL_VALUES,
        )
        field_type = _scale_field_type(literal_values) if literal_values else pydantic_type

        # Определяем, может ли поле быть None
        has_nulls = column.isnull().any()

        # Получаем описание поля
        description = field_descriptions.get(col, f"Оценка по критерию {col}")

        if has_nulls:
            field_definitions[col] = (
                Optional[field_type],
                Field(default=None, description=description),
            )
        else:
            field_definitions[col] = (
                field_type,
                Field(..., description=description),
            )

    return create_model(model_name, __config__=model_config, **field_definitions)


# =====================================================
# MODEL VALIDATION
# =====================================================


def validate_output(
    output_model: type[BaseModel],
    data: dict[str, Any],
    strict: bool = True,
) -> tuple[bool, list[str]]:
    """
    Валидирует данные против Pydantic модели.

    Args:
        output_model: Pydantic модель для валидации
        data: словарь с данными для валидации
        strict: если True - строгая валидация с выбросом исключений

    Returns:
        Кортеж: (успех валидации, список ошибок)
    """
    errors = []

    try:
        if strict:
            output_model(**data)
            return True, []
        else:
            # Частичная валидация - проверяем только известные поля
            output_model.model_validate(data)
            return True, []
    except Exception as e:
        errors.append(str(e))
        return False, errors


def get_model_schema(output_model: type[BaseModel]) -> dict[str, Any]:
    """
    Возвращает JSON schema для Pydantic модели.

    Args:
        output_model: Pydantic модель

    Returns:
        Словарь с JSON schema
    """
    return output_model.model_json_schema()
