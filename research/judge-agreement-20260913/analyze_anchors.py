"""Полное парное сравнение ориентиров; пропуски и зависимости не скрываются."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd

from anchor_experiment import ROOT, consensus
from audit_agreement import audit
from agent.score_results import score_results


KEYS = ['cohen_kappa', 'krippendorff_alpha_ordinal', 'spearman_correlation']


def coefficients(human: np.ndarray, judge: np.ndarray) -> dict:
    return score_results(pd.DataFrame({'score': human, 'agent_score': judge}), 'score',
                         defect_threshold=2, higher_is_better=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--clusters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', action='store_true')
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(ROOT):
        raise ValueError('Подробные результаты должны оставаться вне Git')
    case = json.loads(args.case.read_text())
    units = sorted([u for u in case['units'] if u['partition'] == 'dev'], key=lambda u: u['unit_id'])
    mapping = json.loads(args.clusters.read_text())
    mapping = mapping.get('evaluation_cluster_by_unit', mapping.get('component_by_unit'))
    groups = np.array([mapping[u['unit_id']] for u in units])
    human = np.array([consensus(u, 'structure') for u in units], dtype=float)
    records = [json.loads(p.read_text()) for p in args.runs.glob('*.json')]
    records = [r for r in records if r['agent'] == case['agent'] and r['criterion'] == 'structure']
    rows, predictions = [], {}
    for arm in ['baseline', 'anchored']:
        selected = [r for r in records if r['arm'] == arm]
        by_id = {r['unit_id']: r for r in selected}
        if len(by_id) != len(selected) or set(by_id) != {u['unit_id'] for u in units}:
            raise ValueError(f'{arm}: нужны ровно все {len(units)} единиц, получено {len(selected)}')
        values = [by_id[u['unit_id']].get('result', {}).get('assessment_score') for u in units]
        prediction = np.array([v if isinstance(v, (int, float)) else None for v in values], dtype=float)
        predictions[arm] = prediction
        known = np.isfinite(human)
        result = coefficients(human[known], prediction[known])
        row = dict(audit(human.tolist(), prediction.tolist()), arm=arm, agent=case['agent'])
        for key in ['defect_recall', 'defect_precision', 'defect_false_positive_rate', 'defect_confusion']:
            row[key] = result[key]
        row.update(critical_support=int((human == 0).sum()),
                   critical_exact_recall=float(np.mean(prediction[human == 0] == 0)) if np.any(human == 0) else None,
                   critical_detection_recall=float(np.mean(prediction[human == 0] < 2)) if np.any(human == 0) else None,
                   statuses=dict(Counter(r['status'] for r in selected)),
                   call_errors=dict(Counter(e['type'] for r in selected for e in r['call_errors'])),
                   total_tokens=sum(g['response_metadata'].get('token_usage', {}).get('total_tokens', 0)
                       for r in selected for response in r['responses'] for group in response['generations'] for g in group))
        rows.append(row)
        print(arm, {key: row[key] for key in [*KEYS, 'all_units_coverage', 'defect_recall', 'defect_false_positive_rate']})
    paired = np.isfinite(human) & np.isfinite(predictions['baseline']) & np.isfinite(predictions['anchored'])
    comparison = {'common_pairs': int(paired.sum()), 'groups': len(set(groups)), 'bootstrap_repetitions': 0}
    if paired.sum() > 1:
        before, after = [coefficients(human[paired], predictions[arm][paired]) for arm in ['baseline', 'anchored']]
        comparison['common_differences'] = {k: after[k]-before[k] if after[k] is not None and before[k] is not None else None for k in KEYS}
    if args.bootstrap:
        rng = np.random.default_rng(20260913)
        clusters = [np.flatnonzero((groups == group) & paired) for group in sorted(set(groups))]
        differences = {k: [] for k in KEYS}
        for _ in range(2000):
            indices = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])
            if len(indices) < 2:
                continue
            before, after = [coefficients(human[indices], predictions[arm][indices]) for arm in ['baseline', 'anchored']]
            for key in KEYS:
                if before[key] is not None and after[key] is not None:
                    differences[key].append(after[key]-before[key])
        comparison.update(bootstrap_repetitions=2000, intervals_95={key: {
            'defined_replicates': len(values), 'lower': float(np.quantile(values, .025)) if values else None,
            'upper': float(np.quantile(values, .975)) if values else None} for key, values in differences.items()})
    panel = np.full((max(len(u['ratings']) for u in units), len(units)), np.nan)
    for column, unit in enumerate(units):
        panel[:len(unit['ratings']), column] = [r['scores']['structure'] for r in unit['ratings']]
    output = {'rows': rows, 'comparison': comparison,
              'human_panel_alpha_ordinal': float(krippendorff.alpha(panel, level_of_measurement='ordinal')),
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.case, args.clusters]},
              'scope': 'Полный dev; интервалы на общих валидных парах, кластерный percentile bootstrap. Покрытие и дефекты показаны отдельно по всему потоку. Не независимая проверка переноса.'}
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
