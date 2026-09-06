"""Разделение ролей: что размечает судья и как из разметки считается score.

Принцип: судья — оракул разметки, формула КМ — контракт `monitoring_metric`.
Судья воспроизводит то, что в эталонной корзине проставлял человек (истинную
метку, оценки по критериям, голос), а score единицы считает та же формула
контракта, что у адаптера и km-dynamic (`laim_monitoring.score_units`).

Для `scoring.method=accuracy` это означает: судья предсказывает `target`
(истинный класс запроса), `prediction` берётся из UMR (класс, который выдал
агент), score = prediction == target. Готовый score судья ставит только там,
где формулу применить нельзя:

* prediction не наблюдается в monitoring UMR (трейс не несёт класс агента);
* единица оценки — целый диалог (`assessment_mode=dialogue`), где источники
  заданы один раз на диалог и судья оценивает разговор целиком.

Семантика фиксируется в `assessment_result.scoring_semantics`, чтобы отчёт не
выдавал «мнение судьи о правильности» за формулу отчёта о валидации.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import pandas as pd

from laim_monitoring import MonitoringContractError, score_units

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
    return result


def build_judge_plan(contract: dict, *, prediction_observed: bool = True) -> JudgePlan:
    """Выбрать, что судья размечает, чтобы формула контракта была применима.

    `prediction_observed` — наблюдается ли prediction (класс агента) в UMR,
    на котором пойдёт оценка. Для эталонной корзины он есть всегда; для
    monitoring это свойство конвертера трейсов.
    """
    scoring = contract["scoring"]
    method = scoring["method"]
    mode = contract["assessment_mode"]

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
    if method == "accuracy":
        if not prediction_observed:
            prediction = source_by_role(contract, "prediction")
            return JudgePlan(
                contract=_judge_score_contract(contract),
                judge_source_ids=(JUDGE_SCORE_SOURCE_ID,),
                semantics=JUDGE_FINAL_SCORE,
                reason=(
                    f"prediction {prediction['column_name']!r} не наблюдается в UMR: "
                    "судья оценивает output_answer напрямую, accuracy отчёта не "
                    "воспроизводится, сравнение с baseline информативное"
                ),
            )
        target = source_by_role(contract, "target")
        return JudgePlan(
            contract=deepcopy(contract),
            judge_source_ids=(target["source_id"],),
            semantics=CONTRACT_FORMULA,
            reason=(
                "accuracy: судья предсказывает истинную метку (target), "
                "prediction берётся из UMR, score = prediction == target"
            ),
        )
    return JudgePlan(
        contract=deepcopy(contract),
        judge_source_ids=tuple(source["source_id"] for source in scoring["sources"]),
        semantics=CONTRACT_FORMULA,
        reason=f"{method}: судья размечает все источники контракта, score считает формула",
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
    if plan.semantics == CONTRACT_FORMULA and plan.scoring_method == "accuracy":
        lines.append(
            "Итоговая оценка НЕ запрашивается: она будет вычислена как совпадение "
            "твоей истинной метки с классом, который выдал агент."
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
