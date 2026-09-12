"""Разрешённый пустой список не ошибка; ссылки вне enum остаются ошибкой."""
import json
from copy import deepcopy
from pathlib import Path

from analyze_round4 import OLD, align_parser

case=json.loads((OLD/'cases/CI10071259.json').read_text())
lookup={u['unit_id']:u for u in case['units']}
restored=[]
for path in Path('runs').glob('*.json'):
    r=json.loads(path.read_text())
    if r['arm'] not in ['semantic','contrastive','contrastive_comments']:
        continue
    result=align_parser(case,lookup[r['unit_id']],r)
    if result['status']=='ok' and r['status']=='error':
        restored.append(r)
assert restored
original=restored[0]
broken=deepcopy(original)
parsed=json.loads(broken['response']['choices'][0]['message']['content'])
parsed['claim_checks'][0]['evidence_ids']=['E_DOES_NOT_EXIST']
broken['response']['choices'][0]['message']['content']=json.dumps(parsed)
assert align_parser(case,lookup[broken['unit_id']],broken)['status']=='error'
assert original['status']=='error'
print('PASS: восстановлено',len(restored),'разрешённых схемой ответов; неизвестная ссылка отклонена; исходные результаты неизменны')
