"""Полный знаменатель и парное сравнение сохранённых ответов нового раунда."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

OLD = Path('/Users/antonzyukov/laim-artifacts/judge-deep-research-20260911')
sys.path.insert(0, str(OLD))
from analyze_results import alpha, finite, mode, summarize  # noqa: E402
from compare_round3 import compare  # noqa: E402
from round3 import decoded_unit  # noqa: E402
from run_judge import validate  # noqa: E402

OUT = Path(__file__).parent


def align_parser(case: dict, unit: dict, record: dict) -> dict:
    rescored={**record,'arm':record['arm']+'_aligned_parser'}
    if record.get('response',{}).get('choices',[{}])[0].get('finish_reason')!='stop':
        return rescored
    try:
        parsed=json.loads(record['response']['choices'][0]['message']['content'])
        properties=record['request']['response_format']['schema']['properties']['claim_checks']['items']['properties']
        for check in parsed['claim_checks']:
            for field in ['answer_fragment_ids','evidence_ids']:
                if len(check[field])<properties[field].get('minItems',0) or not set(check[field])<=set(properties[field]['items']['enum']):
                    raise ValueError('Ссылка вне отправленной схемы')
        scores,audit=validate(case,decoded_unit(unit),parsed)
        rescored.update(status='ok',scores=scores,audit=audit,original_status=record['status'],
            parser_note='Пустые answer_fragment_ids разрешены отправленной схемой; качество объяснений оценивается отдельно. Ссылки вне enum отклоняются.')
    except (ValueError,KeyError,TypeError):
        pass
    return rescored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap', action='store_true')
    args = parser.parse_args()
    case = json.loads((OLD/'cases/CI10071259.json').read_text())
    units = sorted([u for u in case['units'] if u['partition']=='dev'], key=lambda u:u['unit_id'])
    records = [json.loads(p.read_text()) for p in (OUT/'runs').glob('*.json')]
    lookup={u['unit_id']:u for u in units}
    score_records=[]
    for r in records:
        if r['partition']!='dev' or r['arm'] not in ['semantic','contrastive','contrastive_comments']:
            continue
        rescored=align_parser(case,lookup[r['unit_id']],r)
        score_records.append(rescored)
    records+=score_records
    baselines = [json.loads(p.read_text()) for p in (OLD/'agreement-runs').glob('*.json')]
    records += [r for r in baselines if r['agent']==case['agent'] and r['arm']=='grounded_ids_bm25_anchors_t8' and r['repeat']==0 and r['partition']=='dev']
    rows, predictions = [], []
    for arm in sorted({r['arm'] for r in records}):
        selected = [r for r in records if r['arm']==arm and r['partition']=='dev']
        by_id = {r['unit_id']:r for r in selected}
        assert len(by_id)==len(selected), 'Повторные ответы под одним именем'
        pending = [u['unit_id'] for u in units if u['unit_id'] not in by_id]
        if pending:
            print(arm, 'pending', len(pending))
            continue
        ordered = [by_id[u['unit_id']] for u in units]
        for criterion in case['scores']:
            row = summarize(case, ordered, criterion, args.bootstrap)
            gold = np.array([mode([r['scores'][criterion] for r in u['ratings']]) for u in units], dtype=float)
            pred = np.array([r['scores'][criterion] if r['status']=='ok' else np.nan for r in ordered], dtype=float)
            paired = np.isfinite(gold)&np.isfinite(pred)
            row['balanced_accuracy_all']=float(np.mean([np.mean(pred[gold==value]==value) for value in set(gold[np.isfinite(gold)])]))
            row.update(arm=arm, alpha_ordinal=alpha(np.array([gold[paired],pred[paired]]),'ordinal'),
                       outcomes=dict(Counter(r['response']['choices'][0]['finish_reason'] if 'response' in r else 'transport' for r in ordered)),
                       errors=dict(Counter(r.get('error') for r in ordered if r['status']!='ok')))
            rows.append(row)
            predictions.append({'arm':arm, 'samples':1, 'method':'raw_mode_or_argmax', 'criterion':criterion,
                'unit_ids':[u['unit_id'] for u in units], 'groups':[u['group_id'] for u in units],
                'gold':gold, 'panel_mean':[np.mean([r['scores'][criterion] for r in u['ratings']]) for u in units], 'prediction':pred})
            print(arm,criterion,json.dumps(finite({k:row[k] for k in ['valid_scores','accuracy','cohen_kappa','alpha_ordinal','correct_label_yield','recall_all_defects','score_zero_recall_all','fpr_paired']})))
    pairs = []
    if args.bootstrap:
        for r in predictions:
            if r['arm']=='grounded_ids_bm25_anchors_t8':
                continue
            base = next(b for b in predictions if b['arm']=='grounded_ids_bm25_anchors_t8' and b['criterion']==r['criterion'])
            left, right = finite(r), finite(base)
            pairs.append(compare(left,right))
    (OUT/'metrics.json').write_text(json.dumps(finite(rows),ensure_ascii=False,indent=2))
    (OUT/'predictions.json').write_text(json.dumps(finite(predictions),ensure_ascii=False,indent=2))
    if pairs:
        (OUT/'paired-comparisons.json').write_text(json.dumps(finite(pairs),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
