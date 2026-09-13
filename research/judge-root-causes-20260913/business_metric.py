"""Отдельная диагностика бизнес-КМ ПОСТ: средние голосов и harmonic(F, C)."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from audit import OUT, cases
from analyze import load_records

CRITERIA = ['completeness', 'factuality', 'structure']


def harmonic(completeness: float, factuality: float) -> float:
    return 2 * completeness * factuality / (completeness + factuality) if completeness + factuality else 0.0


def aggregate(values: np.ndarray) -> dict:
    means = np.mean(values, axis=0)
    return {**dict(zip(CRITERIA, means.tolist(), strict=True)), 'business_f1': harmonic(means[0], means[1])}


def bounded_aggregate(values: np.ndarray) -> dict:
    return {'lower': aggregate(np.nan_to_num(values, nan=0.0)), 'upper': aggregate(np.nan_to_num(values, nan=1.0))}


def main() -> None:
    case = next(c for _, c in cases() if c['agent'] == 'CI10071259')
    units = sorted([u for u in case['units'] if u['partition'] == 'dev'], key=lambda u: u['unit_id'])
    train = [u for u in case['units'] if u['partition'] == 'train']

    def human_mean(subset: list[dict]) -> np.ndarray:
        return np.array([[np.mean([r['scores'][c] for r in u['ratings']])/2 for c in CRITERIA] for u in subset])

    human = human_mean(units)
    training_mean = np.mean(human_mean(train), axis=0)
    records = load_records()
    groups = defaultdict(list)
    for i, u in enumerate(units):
        groups[u['group_id']].append(i)
    indices = list(groups.values())
    rows, predictions = [], {}
    for arm in ['examples_scores_only', 'examples_human_reasons', 'examples_provider_compatible', 'examples_provider_compatible_ultra']:
        if any((case['agent'], u['unit_id'], arm) not in records for u in units):
            continue
        prediction = np.array([[records[case['agent'], u['unit_id'], arm].get('scores', {}).get(c) for c in CRITERIA] for u in units], dtype=float)/2
        predictions[arm] = prediction
        paired = np.isfinite(prediction).all(axis=1)
        observed, target = aggregate(prediction[paired]), aggregate(human[paired])
        rng = np.random.default_rng(20260913)
        biases = []
        for _ in range(2000):
            sampled = np.concatenate([indices[i] for i in rng.integers(0, len(indices), len(indices))])
            sampled = sampled[paired[sampled]]
            if len(sampled):
                pred, gold = aggregate(prediction[sampled]), aggregate(human[sampled])
                biases.append([pred[k]-gold[k] for k in [*CRITERIA, 'business_f1']])
        rows.append({'arm': arm, 'all_units': len(units), 'groups': len(groups), 'paired_units_all_criteria': int(paired.sum()),
                     'human_all_units': aggregate(human), 'human_paired': target, 'judge_paired': observed,
                     'bias_on_paired': {k: observed[k]-target[k] for k in observed},
                     'mean_absolute_unit_error_on_paired': dict(zip(CRITERIA, np.abs(prediction[paired]-human[paired]).mean(axis=0).tolist(), strict=True)),
                     'paired_bias_ci95_group_bootstrap': dict(zip([*CRITERIA, 'business_f1'], np.quantile(biases, [.025, .975], axis=0).T.tolist(), strict=True)),
                     'missing_score_bounds_all_units': bounded_aggregate(prediction),
                     'constant_train_mean_baseline_all_units': aggregate(np.tile(training_mean, (len(units), 1)))})
    common = np.logical_and.reduce([np.isfinite(p).all(axis=1) for p in predictions.values()])
    common_cohort = {'units': int(common.sum()), 'human': aggregate(human[common]), 'judges': {arm: aggregate(p[common]) for arm, p in predictions.items()}}
    result = {'common_scored_cohort_all_arms': common_cohort, 'scope': 'ПОСТ dev, исходный средний человеческий балл каждого объекта /2, затем равные веса объектов. Ties моды сохранены. F1=harmonic(mean completeness, mean factuality), не mean(unit harmonic) и не classification F1.',
              'limitations': 'Это диагностическая равновесная КМ на 63 dev-объектах/7 группах, не восстановление исходного популяционного отчёта и не production допуск. Показатель требует отдельного определения/агрегации. Bounds без допущений об оценках на пропусках; CI условны на отвеченных объектах.',
              'rows': rows}
    (OUT/'business-metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
