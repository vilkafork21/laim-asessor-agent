"""Согласие судьи с итоговой человеческой оценкой и охват размеченного holdout."""
from __future__ import annotations

import krippendorff
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


def score_results(
    frame: pd.DataFrame, score_column: str, *, defect_threshold: float,
    higher_is_better: bool,
) -> dict[str, object]:
    """Одна строка — единица КМ; agreement/MAE условны на ответе, recall включает отказы.

    Альфа nominal сравнивает точные метки судьи и итоговой человеческой оценки.
    Это не согласованность исходной панели разметчиков. Все единицы равновесны.
    """
    human = pd.to_numeric(frame[score_column], errors="raise")
    judge = pd.to_numeric(frame[f"agent_{score_column}"], errors="raise")
    if frame.empty or not np.isfinite(human).all():
        raise ValueError(f"{score_column}: нужны конечные человеческие оценки всех единиц")
    if not (judge.isna() | np.isfinite(judge)).all():
        raise ValueError(f"agent_{score_column}: допустимы конечные оценки или пропуски")
    paired = judge.notna()
    observed_human, observed_judge = human[paired], judge[paired]
    n = int(paired.sum())
    kappa = alpha = correlation = None
    # Вырожденная шкала не даёт определённого agreement сверх случайности.
    if n >= 2:
        codes, labels = pd.factorize(pd.concat([observed_judge, observed_human]))
        if len(labels) >= 2:
            kappa = float(cohen_kappa_score(codes[:n], codes[n:]))
            alpha = float(krippendorff.alpha(
                codes.reshape(2, n).astype(float), level_of_measurement="nominal",
            ))
        if observed_human.nunique() > 1 and observed_judge.nunique() > 1:
            correlation = float(spearmanr(observed_human, observed_judge).statistic)
    human_defect = human.lt(defect_threshold) if higher_is_better else human.gt(defect_threshold)
    judge_defect = judge.lt(defect_threshold) if higher_is_better else judge.gt(defect_threshold)
    tp = int((paired & human_defect & judge_defect).sum())
    fn = int((paired & human_defect & ~judge_defect).sum())
    fp = int((paired & ~human_defect & judge_defect).sum())
    tn = int((paired & ~human_defect & ~judge_defect).sum())
    defects = int(human_defect.sum())
    abstained_defects = int((~paired & human_defect).sum())
    abstained_nondefects = int((~paired & ~human_defect).sum())
    return {
        "holdout_units": len(frame), "paired_units": n,
        "holdout_defect_units": defects,
        "coverage": n / len(frame), "invalid_share": 1 - n / len(frame),
        "acc_auto": float(observed_human.eq(observed_judge).mean()) if n else None,
        "baseline_mode_accuracy": float(observed_human.value_counts().max() / n) if n else None,
        "cohen_kappa": kappa, "krippendorff_alpha": alpha,
        "agreement_scale": "nominal", "agreement_scope": "paired_units",
        "spearman_correlation": correlation,
        "mean_absolute_error": float((observed_judge - observed_human).abs().mean()) if n else None,
        "defect_recall": tp / defects if defects else None,
        "defect_precision": tp / (tp + fp) if tp + fp else None,
        "defect_coverage": (defects - abstained_defects) / defects if defects else None,
        "nondefect_coverage": ((len(frame) - defects - abstained_nondefects) / (len(frame) - defects)
                               if len(frame) > defects else None),
        "defect_confusion": {
            "true_positive": tp, "false_negative": fn, "false_positive": fp,
            "true_negative": tn, "abstained_defects": abstained_defects,
            "abstained_nondefects": abstained_nondefects,
        },
    }
