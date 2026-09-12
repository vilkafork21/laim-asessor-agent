"""Проверка происхождения расширенных свидетельств без обращения к API."""
import json
from copy import deepcopy

from packet_round import OLD, OUT, digest, enrich, request_for

case=json.loads((OLD/'cases/CI10071259.json').read_text())
expanded,audit=enrich(case)
lookup={u['unit_id']:u for u in case['units']}
allowed={}
for u in case['units']:
    key=audit['unit_packets'][u['unit_id']]
    allowed.setdefault(key,set()).update(digest([e['tool_name'],e['arguments'],e['content']]) for e in u['evidence'])
for u in expanded['units']:
    original=lookup[u['unit_id']]
    assert u['context']==original['context'] and u['ratings']==original['ratings']
    key=audit['unit_packets'][u['unit_id']]
    for e in u['evidence']:
        assert e['packet_id']==key
        assert digest([e['tool_name'],e['arguments'],e['content']]) in allowed[key]
        assert 'before_final_answer' not in e['binding']
unit=next(u for u in expanded['units'] if u['partition']=='dev')
changed=deepcopy(unit)
changed['ratings']=[]
changed['source_rows']=['НЕ ПЕРЕДАВАТЬ']
assert request_for(expanded,unit)==request_for(expanded,changed)
assert 'связаны с этим ответом и предшествуют ему' not in request_for(expanded,unit)['messages'][0]['content']
(OUT/'packet-evidence-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
print('PASS:',audit['packet_count'],'пакета;',audit['unique_tool_results'],'реальных результатов; gold и чужие пакеты исключены')
