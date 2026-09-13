"""Парные изменения и bootstrap по целым группам, включая отказы в yield."""
from __future__ import annotations

import json
import warnings
from collections import Counter, defaultdict

import numpy as np
import krippendorff
from sklearn.metrics import cohen_kappa_score

from audit import OUT, cases, consensus


def main() -> None:
    records = {}
    for path in sorted((OUT/'runs').glob('*.json')):
        record = json.loads(path.read_text())
        if record['arm'].startswith('blind_route') and record['request'].get('transport_profile') != 'default_tls':
            continue
        key = record['agent'], record['unit_id'], record['arm']
        if key in records:
            raise ValueError('Неоднозначная версия запроса; нельзя выбирать удачный ответ')
        records[key] = record
    source = {c['agent']: c for _, c in cases()}
    panel_rows = []
    for agent, case in source.items():
        units = [u for u in case['units'] if not u['partition'].startswith('excluded')]
        if max(len(u['ratings']) for u in units) < 2:
            continue
        raters = sorted({r['rater_id'] for u in units for r in u['ratings']})
        for criterion in case['scores']:
            matrix = np.full((len(raters), len(units)), np.nan)
            for j, unit in enumerate(units):
                for rating in unit['ratings']:
                    matrix[raters.index(rating['rater_id']), j] = rating['scores'][criterion]
            panel_rows.append({'agent': agent, 'criterion': criterion, 'units': len(units), 'rater_slots': len(raters),
                               'ratings_per_unit': dict(Counter(len(u['ratings']) for u in units)),
                               'alpha_nominal': float(krippendorff.alpha(matrix, level_of_measurement='nominal')),
                               'alpha_ordinal': float(krippendorff.alpha(matrix, level_of_measurement='ordinal'))})
    (OUT/'human-panel-audit.json').write_text(json.dumps(panel_rows, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    comparisons = [('CI09840670', 'baseline', 'restored_observations'),
                   ('CI09997438', 'baseline', 'blind_route'),
                   ('CI09997438', 'blind_route', 'blind_route_examples'),
                   ('CI09997438', 'baseline', 'blind_route_examples'),
                   ('CI09997438', 'blind_route_train_retry', 'blind_route_rubric_retry'),
                   ('CI09997438', 'baseline', 'blind_route_rubric_retry')]
    result = []
    for agent, control, candidate in comparisons:
        case = source[agent]
        units = sorted([u for u in case['units'] if u['partition'] == 'dev'], key=lambda u: u['unit_id'])
        if any((agent, u['unit_id'], a) not in records for u in units for a in [control, candidate]):
            continue
        human = np.array([consensus(u, 'assessment_score') for u in units], dtype=float)
        prediction = [np.array([records[agent, u['unit_id'], a].get('scores', {}).get('assessment_score') for u in units], dtype=float) for a in [control, candidate]]
        groups = defaultdict(list)
        for i, u in enumerate(units):
            groups[u['group_id']].append(i)
        indices = list(groups.values())
        rng = np.random.default_rng(20260913)
        deltas = []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            warnings.simplefilter('ignore', UserWarning)
            for _ in range(2000):
                sampled = np.concatenate([indices[i] for i in rng.integers(0, len(indices), len(indices))])
                kappas, yields = [], []
                for p in prediction:
                    known = sampled[np.isfinite(human[sampled])]
                    paired = known[np.isfinite(p[known])]
                    kappas.append(cohen_kappa_score(human[paired], p[paired]) if len(paired) else np.nan)
                    yields.append(float(np.mean(human[known] == p[known])))
                deltas.append([kappas[1]-kappas[0], yields[1]-yields[0]])
        correct = [p == human for p in prediction]
        row = {'agent': agent, 'control': control, 'candidate': candidate, 'units': len(units), 'groups': len(groups),
               'correct_control': int(correct[0].sum()), 'correct_candidate': int(correct[1].sum()),
               'corrected': int((~correct[0] & correct[1]).sum()), 'regressed': int((correct[0] & ~correct[1]).sum()),
               'bootstrap': '2000 paired group replicates; seed 20260913; exploratory dev, not fresh test'}
        for i, name in enumerate(['delta_kappa', 'delta_correct_label_yield']):
            values = np.asarray(deltas)[:, i]
            finite = values[np.isfinite(values)]
            row[name] = {'ci95': np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
                         'defined_replicates': len(finite), 'undefined_replicates': len(values)-len(finite)}
        result.append(row)
    (OUT/'paired-comparisons.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
