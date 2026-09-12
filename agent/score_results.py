"""Согласие судьи с итоговой человеческой оценкой и охват размеченного holdout."""
from __future__ import annotations

import krippendorff
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


def score_results(
    frame: pd.DataFrame, score_column: str, *, defect_threshold: float,
    higher_is_better: bool, train_mode_score: float | None = None,
) -> dict[str, object]:
    """Одна строка — единица КМ; agreement/MAE условны на ответе, recall включает отказы.

    Альфа nominal сравнивает точные метки, ordinal учитывает порядок баллов.
    Это не согласованность исходной панели разметчиков. Все единицы равновесны.
    """
    human = pd.to_numeric(frame[score_column], errors="raise")
    judge = pd.to_numeric(frame[f"agent_{score_column}"], errors="raise")
    if frame.empty or not np.isfinite(human).all():
        raise ValueError(f"{score_column}: нужны конечные человеческие оценки всех единиц")
    if not (judge.isna() | np.isfinite(judge)).all():
        raise ValueError(f"agent_{score_column}: допустимы конечные оценки или пропуски")
    if train_mode_score is not None and not np.isfinite(train_mode_score):
        raise ValueError(f"train_mode_score: нужна конечная оценка, получено {train_mode_score}")
    paired = judge.notna()
    observed_human, observed_judge = human[paired], judge[paired]
    n = int(paired.sum())
    kappa = alpha = ordinal_alpha = correlation = None
    correlation_reason = 'insufficient_pairs'
    chance = (float((observed_human.value_counts(normalize=True)
                     * observed_judge.value_counts(normalize=True)).sum()) if n else None)
    # Вырожденная шкала не даёт определённого agreement сверх случайности.
    if n >= 2:
        codes, labels = pd.factorize(pd.concat([observed_judge, observed_human]), sort=True)
        if len(labels) >= 2:
            kappa = float(cohen_kappa_score(codes[:n], codes[n:]))
            alpha = float(krippendorff.alpha(
                codes.reshape(2, n).astype(float), level_of_measurement="nominal",
            ))
            ordinal_alpha = float(krippendorff.alpha(
                codes.reshape(2, n).astype(float), level_of_measurement="ordinal",
            ))
        correlation_reason = ('constant_human' if observed_human.nunique() < 2
                              else 'constant_judge' if observed_judge.nunique() < 2 else None)
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
    labels = sorted(set(human) | set(observed_judge))
    confusion = pd.crosstab(observed_human.rename('human'), observed_judge.rename('judge')).reindex(
        index=labels, columns=labels, fill_value=0,
    )
    correct = observed_human.eq(observed_judge)
    return {
        "holdout_units": len(frame), "paired_units": n,
        "holdout_defect_units": defects,
        "coverage": n / len(frame), "invalid_share": 1 - n / len(frame),
        "acc_auto": float(observed_human.eq(observed_judge).mean()) if n else None,
        "correct_label_yield": int(correct.sum()) / len(frame),
        "chance_agreement": chance,
        "baseline_mode_accuracy": (float(observed_human.eq(train_mode_score).mean())
                                   if n and train_mode_score is not None else None),
        "baseline_mode_accuracy_all_units": (float(human.eq(train_mode_score).mean())
                                             if train_mode_score is not None else None),
        "baseline_mode_score": train_mode_score,
        "baseline_mode_source": "train" if train_mode_score is not None else None,
        "holdout_mode_accuracy": float(observed_human.value_counts().max() / n) if n else None,
        "cohen_kappa": kappa, "krippendorff_alpha": alpha,
        "krippendorff_alpha_ordinal": ordinal_alpha,
        "agreement_scale": "nominal", "agreement_scope": "paired_units",
        "spearman_correlation": correlation,
        "spearman_undefined_reason": correlation_reason,
        "mean_absolute_error": float((observed_judge - observed_human).abs().mean()) if n else None,
        "root_mean_squared_error": float(((observed_judge - observed_human) ** 2).mean() ** .5) if n else None,
        "balanced_accuracy": float(correct.groupby(observed_human).mean().mean()) if n else None,
        "label_confusion": {"labels": labels, "human_rows_judge_columns": confusion.to_numpy().tolist()},
        "defect_false_positive_rate": fp / (fp + tn) if fp + tn else None,
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
