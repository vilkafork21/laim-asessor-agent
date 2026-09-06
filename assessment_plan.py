"""Что размечает судья и как из его разметки считается КМ.

Судья — оракул разметки, формула — контракт `monitoring_metric`. Судья
воспроизводит входы формулы с `judged=true` (то, что в эталонной корзине
проставлял человек), его разметка записывается в те же колонки UMR, и КМ
считается той же формулой, что baseline. Входы с `judged=false` — ответ
агента, он наблюдается в UMR и судьёй не предсказывается.

Готовый score судья ставит только там, где формулу применить нельзя:
ответ агента не наблюдается в monitoring UMR либо единица — целый диалог.
Семантика фиксируется в `assessment_result.scoring_semantics`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from laim_monitoring import (
    JUDGE_SCORE_INPUT,
    MonitoringContractError,
    agent_inputs,
    judge_score_contract,
    judged_inputs,
    unit_scores,
)

CONTRACT_FORMULA = "contract_formula"
JUDGE_FINAL_SCORE = "judge_final_score"


@dataclass(frozen=True)
class JudgePlan:
    contract: dict                 # контракт, по которому считается score
    judge_fields: tuple[str, ...]  # имена входов, которые предсказывает судья
    semantics: str                 # CONTRACT_FORMULA | JUDGE_FINAL_SCORE
    reason: str


def input_observed(frame: pd.DataFrame | None, item: dict) -> bool:
    """Колонка входа есть в UMR и несёт хотя бы одно непустое значение."""
    if frame is None or item["column"] not in frame:
        return False
    values = frame[item["column"]].dropna()
    return bool(values.astype(str).str.strip().ne("").any())


def build_judge_plan(contract: dict, *, agent_observed: bool = True) -> JudgePlan:
    """`agent_observed` — наблюдаются ли входы агента в UMR, на котором пойдёт
    оценка: в эталонной корзине всегда, на мониторинге зависит от конвертера."""
    if contract["assessment_mode"] == "dialogue":
        return JudgePlan(
            judge_score_contract(contract), (JUDGE_SCORE_INPUT,), JUDGE_FINAL_SCORE,
            "assessment_mode=dialogue: судья оценивает диалог целиком, формула не разложима по репликам",
        )
    agent = agent_inputs(contract)
    if agent and not agent_observed:
        columns = [item["column"] for item in agent]
        return JudgePlan(
            judge_score_contract(contract), (JUDGE_SCORE_INPUT,), JUDGE_FINAL_SCORE,
            f"ответ агента {columns} не наблюдается в UMR: судья оценивает output_answer, "
            "формула отчёта не воспроизводится, сравнение с baseline информативное",
        )
    return JudgePlan(
        contract, tuple(item["name"] for item in judged_inputs(contract)), CONTRACT_FORMULA,
        "судья воспроизводит разметку корзины, КМ считает формула: " + contract["formula"],
    )


def judge_instruction(plan: JudgePlan) -> str:
    """Блок к инструкции ассесора: какие поля вернуть и что в них."""
    columns = {item["name"]: item["column"] for item in plan.contract["inputs"]}
    lines = ["Поля ответа JSON:"]
    for name in plan.judge_fields:
        lines.append(
            f"- {name}: значение, которое поставил бы разметчик по инструкции "
            f"(колонка корзины {columns[name]!r}). Это разметка запроса, а не оценка ответа агента."
        )
    if plan.semantics == CONTRACT_FORMULA and agent_inputs(plan.contract):
        lines.append(
            "Итоговую оценку не запрашиваем: КМ считается формулой "
            f"{plan.contract['formula']!r} по твоей разметке и ответу агента."
        )
    return "\n".join(lines)


def score_judge_predictions(units: pd.DataFrame, predictions: pd.DataFrame, plan: JudgePlan) -> pd.Series:
    """Подставить разметку судьи в единицы и посчитать построчный score формулой.

    Единица без единого ответа судьи — отказ судьи, не пропуск в данных: её
    score остаётся NaN и она исключается из расчёта.
    """
    columns = [f"agent_{name}" for name in plan.judge_fields]
    missing = [column for column in columns if column not in predictions]
    if missing:
        raise MonitoringContractError(f"Судья не вернул обязательные поля: {missing}")
    if len(units) != len(predictions):
        raise MonitoringContractError(
            f"Единиц {len(units)}, ответов судьи {len(predictions)}: не совпадают"
        )
    values = units.copy()
    for name, column in zip(plan.judge_fields, columns):
        values[name] = predictions[column].tolist()
    failed = predictions[columns].isna().all(axis=1).to_numpy()
    scores = pd.Series(float("nan"), index=values.index, dtype="float64")
    if (~failed).any():
        scores.iloc[~failed] = unit_scores(values.iloc[~failed], plan.contract).to_numpy()
    return scores


def apply_judge_labels(
    frame: pd.DataFrame, units: pd.DataFrame, predictions: pd.DataFrame, plan: JudgePlan
) -> pd.DataFrame:
    """Записать разметку судьи в колонки контракта на строки UMR.

    После этого scored_data содержит те же колонки, что эталонная корзина, но
    с разметкой судьи, и km-dynamic считает КМ той же формулой, что baseline.
    Сырые ответы судьи остаются в agent_<имя>.
    """
    columns = {item["name"]: item["column"] for item in plan.contract["inputs"]}
    result = frame.copy()
    for name in plan.judge_fields:
        column = columns[name]
        values = predictions[f"agent_{name}"].tolist()
        result[column] = None
        target = result.columns.get_loc(column)
        for positions, value in zip(units["_row_positions"], values):
            for position in positions:
                result.iat[position, target] = value
    return result
