"""Согласие и пределы при фиксированных частотах; только сохранённые прогнозы."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agent.score_results import score_results  # noqa: E402


def audit(gold: list, prediction: list) -> dict:
    human, judge = np.asarray(gold, dtype=float), np.asarray(prediction, dtype=float)
    if human.shape != judge.shape or human.ndim != 1:
        raise ValueError('Нужны сопоставленные одномерные оценки')
    known = np.isfinite(human)
    if not known.any():
        return {'units': len(human), 'consensus_units': 0}
    frame = pd.DataFrame({'score': human[known], 'agent_score': judge[known]})
    result = score_results(frame, 'score', defect_threshold=float(human[known].max()), higher_is_better=True)
    paired = known & np.isfinite(judge)
    h, p = human[paired], judge[paired]
    keys = ['cohen_kappa', 'krippendorff_alpha', 'krippendorff_alpha_ordinal',
            'spearman_correlation', 'spearman_undefined_reason', 'chance_agreement',
            'acc_auto', 'correct_label_yield', 'label_confusion']
    row = {k: result[k] for k in keys}
    row.update(units=len(human), consensus_units=int(known.sum()), paired_units=len(h),
               all_units_coverage=float(np.isfinite(judge).mean()))
    if len(h):
        labels = sorted(set(h) | set(p))
        hc, pc = [int((h == x).sum()) for x in labels], [int((p == x).sum()) for x in labels]
        ceiling = sum(min(a, b) for a, b in zip(hc, pc)) / len(h)
        chance = result['chance_agreement']
        row.update(human_counts=hc, judge_counts=pc,
                   max_accuracy_given_marginals=ceiling,
                   max_kappa_given_marginals=(ceiling-chance)/(1-chance) if chance < 1 else None,
                   judge_top_share=float(np.mean(p == max(labels))),
                   max_spearman_given_marginals=float(spearmanr(np.sort(h), np.sort(p)).statistic)
                   if len(set(h)) > 1 and len(set(p)) > 1 else None)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.predictions is None:
        constant = audit([0, 1, 2, 2], [2, 2, 2, 2])
        assert constant['cohen_kappa'] == constant['max_kappa_given_marginals'] == 0
        assert constant['spearman_correlation'] is None
        shifted = audit([0, 0, 1, 1], [1, 1, 2, 2])
        assert shifted['spearman_correlation'] == 1 and shifted['acc_auto'] == 0
        assert audit([0, 2], [None, None])['all_units_coverage'] == 0
        print('PASS: константа, полный сдвиг шкалы и отказы различаются; API не вызывался')
        return
    rows = []
    for record in json.loads(args.predictions.read_text()):
        row = {k: record.get(k) for k in ['agent', 'partition', 'arm', 'criterion', 'method']}
        row.update(audit(record['gold'], record['prediction']))
        rows.append(row)
    output = {'source_sha256': hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
              'interpretation': 'Пределы при фиксированных частотах — диагностика с известным gold, не достижимая автоматически оценка и не новый judge.',
              'rows': rows}
    if args.output is None:
        raise ValueError('Укажите внешний файл --output для результатов')
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(f'Пересчитано сравнений: {len(rows)}')


if __name__ == '__main__':
    main()
