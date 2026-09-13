"""Аудит восьми корпусов и фиксированная диагностическая выборка без смены gold."""
from __future__ import annotations

import hashlib
import json
import re

import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = Path('/Users/antonzyukov/laim-artifacts/judge-deep-research-20260911')
ROUND4 = Path('/Users/antonzyukov/laim-artifacts/judge-deep-research-20260912-round4')
OUT = Path('/Users/antonzyukov/laim-artifacts/judge-root-causes-20260913')


def cases() -> list[tuple[Path, dict]]:
    paths = sorted((OLD/'cases').glob('*.json')) + [ROUND4/'CI09877398.json', ROUND4/'CI10071255-structure.json']
    return [(p, json.loads(p.read_text())) for p in paths]


def consensus(unit: dict, criterion: str) -> float | None:
    votes = Counter(r['scores'][criterion] for r in unit['ratings'])
    winners = [v for v, n in votes.items() if n == max(votes.values())]
    return winners[0] if len(winners) == 1 else None


def selected_units(case: dict) -> list[dict]:
    dev = sorted([u for u in case['units'] if u['partition'] == 'dev'], key=lambda u: u['unit_id'])
    buckets = defaultdict(list)
    for u in dev:
        labels = {c: consensus(u, c) for c in case['scores']}
        bucket = 'tie' if any(v is None for v in labels.values()) else ('defect' if any(labels[c] < max(case['scores'][c]) for c in labels) else 'normal')
        buckets[bucket].append(u)
    return [u for name in ['defect', 'normal', 'tie'] for u in buckets[name][:2 if name == 'tie' else 12]]


def annotation_audit() -> None:
    summary = {}
    for _, case in cases():
        units = [u for u in case['units'] if not u['partition'].startswith('excluded')]
        overlap = Counter()
        for u in units:
            turns = u['context'].get('turns') or [u['context'].get('current_turn', {})]
            if any(len(t.get('output_answer') or '') > 100 and t['output_answer'].strip() in case['rubric'] for t in turns):
                overlap[u['partition']] += 1
        summary[case['agent']] = {'rubric_literal_answer_overlap': dict(overlap)}
        if case['agent'] != 'CI10071259':
            continue
        frame = pd.read_parquet(next(p for p in case['source_hashes'] if p.endswith('parquet'))).reset_index(drop=True)
        stats, ledger = {}, []
        summary[case['agent']]['only_empty_tool_results'] = []
        for partition in ['train', 'dev', 'test']:
            subset = [u for u in units if u['partition'] == partition and all(str(e['content']).strip() in ['[]', '{}', 'null', 'None', ''] for e in u['evidence'])]
            summary[case['agent']]['only_empty_tool_results'].append({'partition': partition, 'units': len(subset), 'labels': {c: dict(Counter(str(consensus(u, c)) for u in subset)) for c in case['scores']}})
        for partition in ['train', 'dev', 'test']:
            for label in [0, 1, 2]:
                selected = [u for u in units if u['partition'] == partition and consensus(u, 'structure') == label]
                tags, comments, unanimous = Counter(), 0, 0
                for u in selected:
                    rows = frame.iloc[u['source_rows']]
                    if not all(str(a).strip() == u['context']['current_turn']['output_answer'].strip() for a in rows.answer):
                        raise ValueError('Комментарий относится к другой версии ответа')
                    unanimous += len({r['scores']['structure'] for r in u['ratings']}) == 1
                    tags.update(set(re.findall(r'#[А-Яа-яЁё]+', ' '.join(str(v) for v in rows['ТЭГ'].dropna()))))
                    comments += any(str(v).strip() not in ['', '-', '.', 'nan'] for v in rows['Комментарий'].dropna())
                    ledger.append({'unit_id': u['unit_id'], 'partition': partition, 'structure': label, 'source_rows': u['source_rows'],
                                   'comments': rows['Комментарий'].fillna('').tolist(), 'tags': rows['ТЭГ'].fillna('').tolist()})
                stats[f'{partition}:{label}'] = {'units': len(selected), 'unanimous': unanimous, 'comments': comments, 'shared_unit_tags': dict(tags)}
        summary[case['agent']]['structure_annotation_audit'] = stats
        (OUT/'post-comment-ledger.json').write_text(json.dumps(ledger, ensure_ascii=False, indent=2)+'\n')
    (OUT/'rubric-annotation-audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')


def main() -> None:
    summaries, selection = [], []
    for path, case in cases():
        active = [u for u in case['units'] if not u['partition'].startswith('excluded')]
        checks = {}
        for source, expected in case['source_hashes'].items():
            p = Path(source)
            checks[p.name] = 'match' if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == expected else 'missing_or_changed'
        criteria = {}
        for c, scale in case['scores'].items():
            labels = [consensus(u, c) for u in active]
            grouped = defaultdict(list)
            for u, label in zip(active, labels):
                grouped[json.dumps([u['context'], u['evidence']], sort_keys=True)].append(label)
            conflicts = [v for v in grouped.values() if len({x for x in v if x is not None}) > 1]
            criteria[c] = {'consensus': dict(Counter(str(v) for v in labels)),
                           'panel_disagreement_units': sum(len({r['scores'][c] for r in u['ratings']}) > 1 for u in active),
                           'identical_input_conflicts': len(conflicts),
                           'invalid_annotations': sum(r['scores'][c] not in scale for u in active for r in u['ratings']),
                           'dev_defects': sum(consensus(u, c) is not None and consensus(u, c) < max(scale) for u in active if u['partition'] == 'dev')}
        groups = Counter(u['group_id'] for u in active)
        row = {'agent': case['agent'], 'units': len(active), 'partitions': dict(Counter(u['partition'] for u in active)),
               'groups': len(groups), 'largest_group': max(groups.values()),
               'modes': dict(Counter(u['context']['mode'] for u in active)),
               'external_evidence_units': sum(bool(u['evidence']) for u in active),
               'observed_routes': sum(bool(u['context'].get('observed_prediction')) for u in active),
               'observed_tools': sum(bool(u['context'].get('observed_tools')) for u in active),
               'missing_current_query': sum(u['context'].get('mode') != 'dialogue' and not u['context'].get('current_turn', {}).get('input_query') for u in active),
               'repeated_rater_units': sum(len({r['rater_id'] for r in u['ratings']}) != len(u['ratings']) for u in active),
               'source_checks': checks, 'criteria': criteria}
        summaries.append(row)
        selected = selected_units(case)
        selection.append({'agent': case['agent'], 'case_path': str(path), 'case_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'units': [u['unit_id'] for u in selected]})
    OUT.mkdir(exist_ok=True)
    (OUT/'audit.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2)+'\n')
    (OUT/'selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2)+'\n')
    annotation_audit()
    for row, subset in zip(summaries, selection):
        print(row['agent'], 'units', row['units'], 'evidence', row['external_evidence_units'], 'selected', len(subset['units']), row['criteria'])


if __name__ == '__main__':
    main()
