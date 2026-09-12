"""Замороженные dev-пороги; отдельный живой тест вероятностного GigaChat judge."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

OLD=Path('/Users/antonzyukov/laim-artifacts/judge-deep-research-20260911')
OUT=Path(__file__).parent
sys.path.insert(0,str(OLD))
from analyze_round3 import thresholds  # noqa: E402
from round3 import evaluate  # noqa: E402


MODEL=OUT/'frozen-probability-thresholds.json'


def freeze() -> dict:
    if MODEL.exists():
        return json.loads(MODEL.read_text())
    source=OLD/'round3-predictions-dev.json'
    rows=json.loads(source.read_text())
    result={'fit_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'fit_partition':'dev','samples':1,'arm':'probability','primary_criterion':'structure',
            'secondary_criteria':['factuality','completeness'],
            'selection':'Пороги по всем доступным dev-прогнозам; исходное обоснование — предыдущий leave-one-group-out. Test не используется.',
            'thresholds':{}}
    for c in ['factuality','completeness','structure']:
        r=next(r for r in rows if r['criterion']==c and r['arm']=='probability' and r['samples']==1 and r['method']=='raw_mode_or_argmax')
        score=np.array(r['soft'],dtype=float)
        gold=np.array(r['gold'],dtype=float)
        valid=np.isfinite(score)&np.isfinite(gold)
        cuts=thresholds(score[valid],gold[valid])
        result['thresholds'][c]=[float(v) if np.isfinite(v) else ('-inf' if v<0 else 'inf') for v in cuts]
    MODEL.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return result


def calibrated_scores(record: dict,model: dict) -> dict:
    if record['status']!='ok':
        return {}
    return {c:None if value is None else int(np.searchsorted([float(t) for t in model['thresholds'][c]],value,side='right'))
            for c,value in record['soft'].items()}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    model=freeze()
    assert calibrated_scores({'status':'ok','soft':{'structure':None}},model)=={'structure':None}
    assert calibrated_scores({'status':'error'},model)=={}
    print(json.dumps(model,ensure_ascii=False),flush=True)
    if not args.run:
        return
    case=json.loads((OLD/'cases/CI10071259.json').read_text())
    units=sorted([u for u in case['units'] if u['partition']=='test'],key=lambda u:u['unit_id'])
    for i,unit in enumerate(units,1):
        r=evaluate(case,unit,'probability',0,run_folder=str(OUT/'probability-test-runs'))
        print(json.dumps({'unit':i,'total':len(units),'status':r['status'],'raw_scores':r.get('scores'),
                         'calibrated_scores':calibrated_scores(r,model),'seconds':r['seconds']},ensure_ascii=False),flush=True)
        if 'HTTP 40' in r.get('error',''):
            raise RuntimeError('Остановлено: авторизация/квота, детали в сохранённом record')


if __name__=='__main__':
    main()
