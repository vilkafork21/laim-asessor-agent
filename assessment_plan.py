"""Разделение ролей: что размечает судья и как из разметки считается КМ.

Принцип: судья — оракул разметки, формула КМ — контракт `monitoring_metric`.
Судья воспроизводит то, что в эталонной корзине проставлял человек (истинную
метку, оценки по критериям, голоса), его разметка записывается в те же колонки,
что в корзине, и КМ считается той же формулой, что baseline, общим пакетом
`laim_monitoring`. Ответ агента (роль prediction) судья не предсказывает — он
наблюдается в UMR.

Готовый score судья ставит только там, где формулу применить нельзя:

* формула использует prediction, а monitoring UMR его не несёт (трейс без
  класса агента);
* единица оценки — целый диалог (`assessment_mode=dialogue`).

Семантика фиксируется в `assessment_result.scoring_semantics`, чтобы отчёт не
выдавал «мнение судьи о правильности» за формулу отчёта о валидации.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import pandas as pd

from laim_monitoring import MonitoringContractError, contract_formula, score_units

CONTRACT_FORMULA = "contract_formula"
JUDGE_FINAL_SCORE = "judge_final_score"

JUDGE_SCORE_SOURCE_ID = "assessment_score"

_ROLE_INSTRUCTION = {
    "target": (
        "ИСТИННАЯ метка запроса (эталонный класс/маршрут/ответ), которую поставил "
        "бы разметчик по инструкции. Это НЕ оценка правильности ответа агента: "
        "определи, какой класс верен для запроса, независимо от того, что ответил агент."
    ),
    "criterion": "оценка по критерию инструкции для этой единицы.",
    "assessor_vote": "голос асессора по инструкции (1 — принято, 0 — нет).",
    "final_score": "итоговая оценка единицы по инструкции.",
}


@dataclass(frozen=True)
class JudgePlan:
    """Что предсказывает судья и каким контрактом считается score."""

    contract: dict
    judge_source_ids: tuple[str, ...]
    semantics: str
    reason: str

    @property
    def scoring_method(self) -> str:
        return self.contract["scoring"]["method"]


def source_by_role(contract: dict, role: str) -> dict:
    for source in contract["scoring"]["sources"]:
        if source["role"] == role:
            return source
    raise MonitoringContractError(f"В контракте нет источника с ролью {role!r}")


def source_observed(frame: pd.DataFrame | None, source: dict) -> bool:
    """Колонка источника есть в UMR и несёт хотя бы одно непустое значение."""
    if frame is None:
        return False
    column = source["column_name"]
    if column not in frame:
        return False
    values = frame[column].dropna()
    if values.empty:
        return False
    return bool(values.astype(str).str.strip().ne("").any())


def _judge_score_contract(contract: dict) -> dict:
    result = deepcopy(contract)
    result["scoring"] = {
        "method": "identity",
        "sources": [
            {
                "source_id": JUDGE_SCORE_SOURCE_ID,
                "column_name": "main_metric",
                "role": "final_score",
                "normalization": "numeric",
                "polarity": "direct",
            }
        ],
        "missing_policy": contract["scoring"]["missing_policy"],
        "majority_denominator": None,
    }
    weighted = contract.get("aggregation", {}).get("method") == "frequency_weighted_mean"
    result["formula"] = (
        f"wmean({JUDGE_SCORE_SOURCE_ID}, weight)" if weighted else f"mean({JUDGE_SCORE_SOURCE_ID})"
    )
    return result


def prediction_source(contract: dict) -> dict | None:
    return next(
        (source for source in contract["scoring"]["sources"] if source["role"] == "prediction"),
        None,
    )


def build_judge_plan(contract: dict, *, prediction_observed: bool = True) -> JudgePlan:
    """Выбрать, что судья размечает, чтобы формула контракта была применима.

    Судья предсказывает все входы формулы, кроме prediction (ответ агента
    наблюдается). `prediction_observed` — есть ли prediction в UMR, на котором
    пойдёт оценка: в эталонной корзине он есть всегда, на мониторинге это
    свойство конвертера трейсов.
    """
    mode = contract["assessment_mode"]
    prediction = prediction_source(contract)
    if mode == "dialogue":
        return JudgePlan(
            contract=_judge_score_contract(contract),
            judge_source_ids=(JUDGE_SCORE_SOURCE_ID,),
            semantics=JUDGE_FINAL_SCORE,
            reason=(
                "assessment_mode=dialogue: судья оценивает диалог целиком, "
                "формула контракта к диалогу не разложима по репликам"
            ),
        )
    if prediction is not None and not prediction_observed:
        return JudgePlan(
            contract=_judge_score_contract(contract),
            judge_source_ids=(JUDGE_SCORE_SOURCE_ID,),
            semantics=JUDGE_FINAL_SCORE,
            reason=(
                f"prediction {prediction['column_name']!r} не наблюдается в UMR: "
                "судья оценивает output_answer напрямую, формула отчёта не "
                "воспроизводится, сравнение с baseline информативное"
            ),
        )
    judge_ids = tuple(
        source["source_id"]
        for source in contract["scoring"]["sources"]
        if source["role"] != "prediction"
    )
    return JudgePlan(
        contract=deepcopy(contract),
        judge_source_ids=judge_ids,
        semantics=CONTRACT_FORMULA,
        reason=(
            "судья воспроизводит разметку корзины, КМ считает формула контракта: "
            + contract_formula(contract)
        ),
    )


def judge_instruction(plan: JudgePlan) -> str:
    """Блок, дописываемый к инструкции ассесора: поля ответа и их смысл."""
    sources = {source["source_id"]: source for source in plan.contract["scoring"]["sources"]}
    lines = ["Поля ответа JSON и что в них указать:"]
    for source_id in plan.judge_source_ids:
        source = sources[source_id]
        role = source["role"]
        lines.append(
            f"- {source_id}: {_ROLE_INSTRUCTION.get(role, role)} "
            f"Исходная колонка корзины: {source['column_name']!r}."
        )
    if plan.semantics == CONTRACT_FORMULA and prediction_source(plan.contract) is not None:
        lines.append(
            "Итоговая оценка НЕ запрашивается: КМ считается формулой "
            f"{contract_formula(plan.contract)!r} по твоей разметке и ответу агента."
        )
    return "\n".join(lines)


def score_judge_predictions(
    units: pd.DataFrame, predictions: pd.DataFrame, plan: JudgePlan
) -> pd.Series:
    """Подставить разметку судьи в единицы и посчитать score формулой контракта.

    Единица, на которую судья не ответил ни одним полем, — отказ судьи, не
    пропуск в данных: missing_policy к ней не применяется, score остаётся NaN.
    """
    judge_columns = [f"agent_{source_id}" for source_id in plan.judge_source_ids]
    missing = [column for column in judge_columns if column not in predictions]
    if missing:
        raise MonitoringContractError(f"Assessor не вернул обязательные поля: {missing}")
    if len(units) != len(predictions):
        raise MonitoringContractError(
            f"Число единиц ({len(units)}) не совпадает с числом ответов судьи ({len(predictions)})"
        )
    values = units.copy()
    for source_id, column in zip(plan.judge_source_ids, judge_columns):
        values[source_id] = predictions[column].tolist()
    failed = predictions[judge_columns].isna().all(axis=1).to_numpy()
    scores = pd.Series(float("nan"), index=values.index, dtype="float64")
    if (~failed).any():
        scores.iloc[~failed] = score_units(values.iloc[~failed], plan.contract).to_numpy()
    return scores


def apply_judge_labels(
    frame: pd.DataFrame, units: pd.DataFrame, predictions: pd.DataFrame, plan: JudgePlan
) -> pd.DataFrame:
    """Записать разметку судьи в колонки контракта на строки UMR.

    После этого scored_data содержит те же колонки, что эталонная корзина, но
    с разметкой судьи, и km-dynamic считает КМ той же формулой, что baseline.
    Сырые ответы судьи остаются в agent_<source_id>.
    """
    sources = {source["source_id"]: source for source in plan.contract["scoring"]["sources"]}
    result = frame.copy()
    for source_id in plan.judge_source_ids:
        column = sources[source_id]["column_name"]
        values = predictions[f"agent_{source_id}"].tolist()
        result[column] = None
        target = result.columns.get_loc(column)
        for positions, value in zip(units["_row_positions"], values):
            for position in positions:
                result.iat[position, target] = value
    return result
