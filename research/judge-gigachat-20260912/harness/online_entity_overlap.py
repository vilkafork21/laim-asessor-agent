"""Консервативные связи по явно написанным ИНН, без использования меток."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

OUT=Path(__file__).parent


def identifiers(text: str) -> set[str]:
    return set(re.findall(r'\bИНН\W{0,8}(\d{12}|\d{10})(?!\d)',text,re.I))


def main() -> None:
    assert identifiers('ИНН: **123456789012**; ИНН 1234567890')=={'123456789012','1234567890'}
    assert not identifiers('номер 123456789012; ИНН 1234567890123')
    case=json.loads((OUT/'CI10071255-structure.json').read_text())
    units=case['units']
    parent={u['unit_id']:u['unit_id'] for u in units}

    def root(uid: str) -> str:
        while parent[uid]!=uid:
            parent[uid]=parent[parent[uid]]
            uid=parent[uid]
        return uid

    owners={}
    by_unit={}
    for unit in units:
        uid=unit['unit_id']
        values=identifiers(unit['context']['current_turn']['output_answer'])
        by_unit[uid]=sorted(values)
        for key in [f'group:{unit["group_id"]}',*[f'inn:{v}' for v in values]]:
            if key in owners:
                parent[root(uid)]=root(owners[key])
            else:
                owners[key]=uid
    clusters=defaultdict(list)
    for unit in units:
        clusters[root(unit['unit_id'])].append(unit)
    linked=[]
    for members in clusters.values():
        parts={u['partition'] for u in members}
        if len(parts)>1:
            linked.append({'unit_ids':[u['unit_id'] for u in members],'partitions':dict(Counter(u['partition'] for u in members))})
    seen={root(u['unit_id']) for u in units if u['partition'] in ['train','dev']}
    affected=[u['unit_id'] for u in units if u['partition']=='test' and root(u['unit_id']) in seen]
    result={'method':'Все явно названные ИНН в ответе, включая контрагентов, образуют консервативные компоненты; метки не используются. Отсутствие извлечённого ИНН не доказывает независимость клиента. Разбиение и gold не меняются.',
        'units_with_explicit_inn':sum(bool(v) for v in by_unit.values()),'unique_inn':len({v for values in by_unit.values() for v in values}),
        'connected_components':len(clusters),'cross_partition_components':len(linked),'test_units_linked_to_train_or_dev':affected,
        'test_remaining_without_detected_link':sum(u['partition']=='test' for u in units)-len(affected),'component_by_unit':{uid:root(uid) for uid in parent},'explicit_inn_by_unit':by_unit,'cross_partition_links':linked}
    (OUT/'online-entity-overlap.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print({k:v for k,v in result.items() if k in ['units_with_explicit_inn','unique_inn','connected_components','cross_partition_components','test_remaining_without_detected_link']})


if __name__=='__main__':
    main()
