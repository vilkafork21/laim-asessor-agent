"""Допуск автоассессора (карточка 6.3.3) и его смещение относительно разметчиков.

Допуск выводится из метрик калибровки на holdout: объём и представленность
критичного класса, доля невалидных ответов, полнота на дефектах и согласие
сверх тривиального базового способа (каппа). Смещение — средняя парная
разность «оценка судьи − оценка человека» на шкале ключевой метрики.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_Z = {0.95: 1.959964}


@dataclass(frozen=True)
class Admission:
    status: str
    reason_code: str
    reason: str


def admit(
    metrics: dict,
    *,
    min_holdout_units: int,
    min_holdout_defect_units: int,
    min_defect_recall: float,
    min_kappa: float,
    max_invalid_share: float,
    weak_holdout_defect_units: int = 10,
) -> Admission:
    """Статус допуска судьи по метрикам калибровки.

    Ниже `min_holdout_defect_units` дефектов в holdout профиль ошибок не измерим
    (не оценено); от минимума до `weak_holdout_defect_units` — измерим грубо,
    допуск жёлтый с усиленным контролем."""
    holdout = int(metrics["holdout_units"])
    defects = int(metrics["holdout_defect_units"])
    invalid_share = float(metrics["invalid_share"])
    recall = float(metrics["defect_recall"])
    kappa = metrics.get("cohen_kappa")
    if holdout < min_holdout_units:
        return Admission(
            "not_assessed", "holdout_too_small",
            f"holdout {holdout} единиц меньше минимума {min_holdout_units}",
        )
    if defects < min_holdout_defect_units:
        return Admission(
            "not_assessed", "critical_class_underrepresented",
            f"единиц критичного класса в holdout {defects} меньше минимума "
            f"{min_holdout_defect_units}",
        )
    if invalid_share > max_invalid_share:
        return Admission(
            "red", "judge_refusals",
            f"доля невалидных ответов судьи {invalid_share:.2f} выше допустимой "
            f"{max_invalid_share:.2f}",
        )
    weak_recall = recall < min_defect_recall
    weak_kappa = kappa is None or float(kappa) < min_kappa
    kappa_text = "не вычислима" if kappa is None else f"{float(kappa):.2f}"
    if weak_recall and weak_kappa:
        return Admission(
            "red", "no_better_than_baseline",
            f"полнота на дефектах {recall:.2f} ниже {min_defect_recall:.2f} и каппа "
            f"{kappa_text} ниже {min_kappa:.2f}: судья не лучше тривиальной оценки",
        )
    if weak_recall or weak_kappa:
        return Admission(
            "amber", "weak_agreement",
            f"полнота на дефектах {recall:.2f} (минимум {min_defect_recall:.2f}), каппа "
            f"{kappa_text} (минимум {min_kappa:.2f}): допуск с усиленным контролем",
        )
    if defects < weak_holdout_defect_units:
        return Admission(
            "amber", "few_critical_units",
            f"единиц критичного класса в holdout {defects} меньше {weak_holdout_defect_units}: "
            "профиль ошибок измерен грубо, допуск с усиленным контролем",
        )
    return Admission("green", "admitted", "судья допущен по профилю ошибок на holdout")


def judge_bias(
    judge_scores: list[float], human_scores: list[float], level: float = 0.95
) -> dict[str, float | int] | None:
    """Смещение судьи: средняя разность «судья − человек» с нормальным интервалом."""
    if len(judge_scores) != len(human_scores):
        raise ValueError("judge_bias: списки оценок должны быть одной длины")
    differences = [judge - human for judge, human in zip(judge_scores, human_scores)]
    n = len(differences)
    if n < 2:
        return None
    mean = sum(differences) / n
    variance = sum((d - mean) ** 2 for d in differences) / (n - 1)
    half = _Z[level] * math.sqrt(variance / n)
    return {"mean": mean, "ci_lower": mean - half, "ci_upper": mean + half, "units": n}
