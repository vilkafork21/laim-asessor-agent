"""Что решает сравнение судьи с модой и почему каппа на этих наборах неизмерима.

Скрипт не обращается к модели: только пересчёт сохранённого metrics.json и биномиальная модель.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analyze_results import finite

OUT = Path(__file__).parent


def cells(confusion: dict) -> dict[str, int]:
    """TP/FP/FN/TN и неверная тяжесть: строки человек, столбцы судья, дефект = ниже максимума."""
    labels = confusion['labels']
    table = np.array(confusion['human_rows_judge_columns'])
    top = labels.index(max(labels))
    low = [i for i, v in enumerate(labels) if v < max(labels)]
    return {'tp': int(table[np.ix_(low, low)].sum()),
            'misgraded': int(table[np.ix_(low, low)].sum() - sum(table[i, i] for i in low)),
            'fn': int(table[low, top].sum()), 'fp': int(table[top, low].sum()),
            'tn': int(table[top, top])}


def identity_check() -> list[dict]:
    """Δaccuracy против моды = (TP - FP - неверная тяжесть)/n, когда мода = верх шкалы."""
    rows = []
    for row in json.loads((OUT / 'metrics.json').read_text()):
        if row['partition'] != 'test':
            continue
        for m in row['criteria']:
            c, labels = cells(m['confusion']), m['confusion']['labels']
            n, table = m['paired_consensus'], np.array(m['confusion']['human_rows_judge_columns'])
            mode_is_top = m['baseline_score'] == max(labels)
            general = (np.trace(table) - table[labels.index(m['baseline_score']), :].sum()) / n
            decomposed = (c['tp'] - c['fp'] - c['misgraded']) / n if mode_is_top else None
            rows.append({
                'agent': row['agent'], 'arm': row['arm'], 'criterion': m['criterion'],
                'scale': labels, 'mode_score': m['baseline_score'], 'mode_is_top_of_scale': mode_is_top,
                'paired': n, 'defects': c['tp'] + c['fn'], **c,
                'observed_delta': m['delta_accuracy_vs_mode'],
                'general_identity_delta': float(general),
                'defect_decomposition_delta': decomposed,
                'general_matches': abs(general - m['delta_accuracy_vs_mode']) < 1e-9,
                'decomposition_matches': (abs(decomposed - m['delta_accuracy_vs_mode']) < 1e-9
                                          if decomposed is not None else None),
                'best_possible_delta': (c['tp'] + c['fn']) / n if mode_is_top else None,
                'one_false_alarm_costs': 1 / n})
    return rows


def kappa(tp: int, fn: int, fp: int, tn: int) -> float | None:
    n = tp + fn + fp + tn
    human_defect, judge_defect = (tp + fn) / n, (tp + fp) / n
    chance = human_defect * judge_defect + (1 - human_defect) * (1 - judge_defect)
    return ((tp + tn) / n - chance) / (1 - chance) if chance < 1 else None


def sampling_profile(defects: int, normals: int, tpr: float, fpr: float, trials: int = 40000) -> dict:
    """Что покажут метрики для заведомо хорошего судьи при таком числе дефектов."""
    rng = np.random.default_rng(20260912)
    tp = rng.binomial(defects, tpr, trials)
    fp = rng.binomial(normals, fpr, trials)
    values = np.array([kappa(int(a), defects - int(a), int(b), normals - int(b)) or np.nan
                       for a, b in zip(tp, fp)])
    finite = values[np.isfinite(values)]
    return {'defects': defects, 'normals': normals, 'units': defects + normals,
            'true_tpr': tpr, 'true_fpr': fpr,
            'probability_judge_loses_to_mode': float(np.mean(tp < fp)),
            'probability_judge_ties_mode': float(np.mean(tp == fp)),
            'probability_judge_beats_mode': float(np.mean(tp > fp)),
            'kappa_median': float(np.median(finite)),
            'kappa_p2_5': float(np.quantile(finite, .025)),
            'kappa_p97_5': float(np.quantile(finite, .975)),
            'kappa_interval_width': float(np.quantile(finite, .975) - np.quantile(finite, .025)),
            'share_kappa_below_0_2': float(np.mean(finite < .2))}


def false_alarm_budget(tpr: float = .8) -> list[dict]:
    """Предельная доля ложных тревог, при которой судья ещё обгоняет моду."""
    return [{'defect_prevalence': p, 'assumed_tpr': tpr, 'maximum_fpr_to_beat_mode': p / (1 - p) * tpr,
             'normal_units_per_defect': (1 - p) / p} for p in [.02, .04, .07, .10, .20, .35, .50]]


def observed_operating_points() -> list[dict]:
    rows = []
    for row in json.loads((OUT / 'metrics.json').read_text()):
        if row['partition'] != 'test':
            continue
        for m in row['criteria']:
            c = cells(m['confusion'])
            defects, normals = c['tp'] + c['fn'], c['fp'] + c['tn']
            if not defects or not normals:
                continue
            p = defects / (defects + normals)
            rows.append({'agent': row['agent'], 'arm': row['arm'], 'criterion': m['criterion'],
                         'defect_prevalence': p, 'observed_tpr': c['tp'] / defects,
                         'observed_fpr': c['fp'] / normals,
                         'maximum_fpr_allowed': p / (1 - p) * (c['tp'] / defects),
                         'within_budget': bool(c['fp'] / normals < p / (1 - p) * (c['tp'] / defects))})
    return rows


def main() -> None:
    identity, points = identity_check(), observed_operating_points()
    prevalence = 2 / 46
    result = {'baseline_identity': identity,
              'general_identity_holds_everywhere': all(r['general_matches'] for r in identity),
              'decomposition_holds_where_mode_is_top': all(
                  r['decomposition_matches'] for r in identity if r['decomposition_matches'] is not None),
              'sampling_profile_of_a_genuinely_good_judge': [
                  sampling_profile(d, round(d * (1 - prevalence) / prevalence), .8, .05)
                  for d in [2, 4, 7, 15, 30, 60, 120]],
              'false_alarm_budget': false_alarm_budget(),
              'observed_operating_points': points}
    (OUT / 'metric-theory.json').write_text(json.dumps(finite(result), ensure_ascii=False, indent=2))

    print('=== Тождество Δaccuracy против моды ===')
    for r in identity:
        d = f"{r['defect_decomposition_delta']:+.4f}" if r['defect_decomposition_delta'] is not None else '  мода не верх'
        print(f"{r['agent']:12} {r['arm']:24} {r['criterion'][:12]:12} n={r['paired']:3} D={r['defects']:3} "
              f"TP={r['tp']:3} FP={r['fp']:3} M={r['misgraded']:2} Δ={r['observed_delta']:+.4f} "
              f"общее={r['general_identity_delta']:+.4f} TP-FP-M={d} "
              f"{'OK' if r['general_matches'] else 'РАСХОЖДЕНИЕ'}")
    print(f"\nОбщее тождество верно во всех {len(identity)} строках: {result['general_identity_holds_everywhere']}")
    print(f"Разложение TP-FP-M верно всюду, где мода = верх шкалы: {result['decomposition_holds_where_mode_is_top']}")

    print('\n=== Заведомо хороший судья TPR=0.8 FPR=0.05 при редком дефекте (4.3%) ===')
    print('дефектов  единиц  P(проигрыш)  P(ничья)  P(выигрыш)  медиана κ    95% интервал κ      ширина')
    for r in result['sampling_profile_of_a_genuinely_good_judge']:
        print(f"{r['defects']:8} {r['units']:7} {r['probability_judge_loses_to_mode']:11.1%} "
              f"{r['probability_judge_ties_mode']:9.1%} {r['probability_judge_beats_mode']:11.1%} "
              f"{r['kappa_median']:10.3f}  [{r['kappa_p2_5']:+.3f}; {r['kappa_p97_5']:+.3f}] {r['kappa_interval_width']:9.3f}")

    print('\n=== Бюджет ложных тревог: максимальный FPR, при котором судья обгоняет моду (TPR=0.8) ===')
    for r in result['false_alarm_budget']:
        print(f"доля дефектов {r['defect_prevalence']:5.0%}: нормальных на один дефект "
              f"{r['normal_units_per_defect']:6.1f}, допустимый FPR < {r['maximum_fpr_to_beat_mode']:.2%}")

    print('\n=== Наблюдённые рабочие точки на test ===')
    for r in points:
        print(f"{r['agent']:12} {r['arm']:24} {r['criterion'][:12]:12} дефектов {r['defect_prevalence']:6.1%} "
              f"TPR={r['observed_tpr']:.3f} FPR={r['observed_fpr']:.3f} допустимо<{r['maximum_fpr_allowed']:.3f} "
              f"{'в бюджете' if r['within_budget'] else 'ПЕРЕРАСХОД'}")


if __name__ == '__main__':
    main()
