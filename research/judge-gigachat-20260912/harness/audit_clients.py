"""Зависимость единиц по ИНН и прежним группам точного контекста; без раскрытия ИНН."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

OUT=Path(__file__).parent
OLD=Path('/Users/antonzyukov/laim-artifacts/judge-deep-research-20260911')


def main() -> None:
    case=json.loads((OLD/'cases/CI10071259.json').read_text())
    frame=pd.read_parquet('/Users/antonzyukov/laim-data-20260911/artifacts/CI10071259/baskets/agent_post_labeled_all_2025_11_27_v1.parquet',columns=['inn'])
    parts=defaultdict(set)
    byunit={}
    parents={}

    def find(x: str) -> str:
        parents.setdefault(x,x)
        if parents[x]!=x:
            parents[x]=find(parents[x])
        return parents[x]

    for u in case['units']:
        ids={str(x).removesuffix('.0').strip() for x in frame.iloc[u['source_rows']].inn.dropna() if str(x).strip() not in ['','0','0.0']}
        byunit[u['unit_id']]=ids
        for x in ids:
            parts[x].add(u['partition'])
            first=find('group:'+u['group_id'])
            second=find('client:'+hashlib.sha256(x.encode()).hexdigest())
            parents[max(first,second)]=min(first,second)
    shared={x for x,p in parts.items() if 'train' in p and 'test' in p}
    devtest={x for x,p in parts.items() if 'dev' in p and 'test' in p}
    test_train=[u['unit_id'] for u in case['units'] if u['partition']=='test' and byunit[u['unit_id']]&shared]
    test_dev=[u['unit_id'] for u in case['units'] if u['partition']=='test' and byunit[u['unit_id']]&devtest]
    affected=set(test_train)|set(test_dev)
    clusters={u['unit_id']:find('group:'+u['group_id']) for u in case['units']}
    result={'identifiable_clients':len(parts),'train_test_shared_clients':len(shared),'dev_test_shared_clients':len(devtest),
            'test_units_with_train_client':test_train,'test_units_with_dev_client':test_dev,
            'units_missing_client':sum(not v for v in byunit.values()),
            'shared_client_test_groups':sorted({u['group_id'] for u in case['units'] if u['unit_id'] in affected}),
            'evaluation_cluster_by_unit':clusters,
            'evaluation_cluster_by_case_group':{u['group_id']:clusters[u['unit_id']] for u in case['units']},
            'distinct_clients_by_partition':{p:len({x for u in case['units'] if u['partition']==p for x in byunit[u['unit_id']]}) for p in ['train','dev','test']},
            'evaluation_cluster_counts':{p:len({clusters[u['unit_id']] for u in case['units'] if u['partition']==p}) for p in ['train','dev','test']},
            'method':'Связные компоненты ИНН и прежних групп case/exact context; применяется только в анализе неопределённости, входы judge не меняются.'}
    (OUT/'client-overlap-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('Кластеры для неопределённости:',result['evaluation_cluster_counts'])


if __name__=='__main__':
    main()
