"""Проверка сигнала train-меток и клиентской утечки без вызова модели."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from anchor_experiment import ROOT, EnhancedBM25, consensus
from audit_agreement import audit


def vote(train: list[dict], unit: dict, scores: list[float], k: int,
         groups: dict[str, str], fallback: float) -> tuple[float, list[str]]:
    selected = [train[i] for i in sorted(range(len(train)), key=lambda i: (-scores[i], train[i]['unit_id']))
                if train[i]['group_id'] != unit['group_id']
                and groups[train[i]['unit_id']] != groups[unit['unit_id']]][:k]
    counts = Counter(consensus(u, 'structure') for u in selected)
    result = min(counts, key=lambda score: (-counts[score], score != fallback, score)) if counts else fallback
    return result, [u['unit_id'] for u in selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--clusters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(ROOT):
        raise ValueError('Подробная диагностика сохраняется вне Git')
    case = json.loads(args.case.read_text())
    train = sorted([u for u in case['units'] if u['partition'] == 'train' and consensus(u, 'structure') is not None], key=lambda u: u['unit_id'])
    dev = sorted([u for u in case['units'] if u['partition'] == 'dev'], key=lambda u: u['unit_id'])
    mapping = json.loads(args.clusters.read_text())
    strict = mapping.get('evaluation_cluster_by_unit', mapping.get('component_by_unit'))
    separate = {u['unit_id']: u['group_id'] for u in case['units']}
    prior = Counter(consensus(u, 'structure') for u in train)
    fallback = min(prior, key=lambda score: (-prior[score], score))
    human = [consensus(u, 'structure') for u in dev]
    rows = [dict(audit(human, [fallback]*len(dev)), profile='train_mode', predicted_mode=fallback)]
    provenance = {}
    for field in ['input_query', 'output_answer', 'both']:
        def tokenize(unit: dict) -> list[str]:
            turn = unit['context']['current_turn']
            text = ' '.join(str(turn.get(key) or '') for key in (['input_query', 'output_answer'] if field == 'both' else [field]))
            return re.findall(r'\w+', text.lower()) or ['__empty__']
        index = EnhancedBM25([tokenize(u) for u in train])
        scores = [index.get_scores(tokenize(u)) for u in dev]
        for policy, groups in [('group', separate), ('client_component', strict)]:
            for k in [1, 3, 5]:
                results = [vote(train, u, s, k, groups, fallback) for u, s in zip(dev, scores)]
                profile = f'{field}/{policy}/k{k}'
                row = dict(audit(human, [p for p, _ in results]), profile=profile,
                           fallback_without_neighbors=sum(not ids for _, ids in results))
                rows.append(row)
                provenance[profile] = {u['unit_id']: ids for u, (_, ids) in zip(dev, results)}
                print(profile, {key: row[key] for key in ['cohen_kappa', 'krippendorff_alpha_ordinal', 'spearman_correlation', 'acc_auto']})
    args.output.write_text(json.dumps({'agent': case['agent'], 'rows': rows, 'train_neighbors': provenance,
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.case, args.clusters]},
        'scope': 'Разведочная диагностика на dev: 18 заранее заданных профилей; не GigaChat judge и не подтверждение переноса.'},
        ensure_ascii=False, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
