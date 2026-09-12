"""Проверка границ данных, контрастных train-примеров и неизменности cache."""
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import round4


def main() -> None:
    case=json.loads((round4.PREVIOUS/'cases/CI10071259.json').read_text())
    unit=next(u for u in case['units'] if u['partition']=='dev')
    changed=copy.deepcopy(unit)
    changed['ratings']=[{'rater_id':'SECRET','scores':{'SECRET':999}}]
    changed['source_rows']=[-999]
    train={u['unit_id'] for u in case['units'] if u['partition']=='train'}
    for arm in round4.ARMS:
        request=round4.request_for(case,unit,arm)
        assert request==round4.request_for(case,changed,arm)
        assert [m['role'] for m in request['messages']]==['system','user']
        data=json.loads(request['messages'][-1]['content'])
        assert all(e['unit_id'] in train for e in data['train_examples'])
        assert all(text in unit['context']['current_turn']['output_answer'] for text in data['answer_fragments'].values())
        if arm!='semantic':
            assert sorted(e['scores']['factuality'] for e in data['train_examples'])==[0,0,1,1,2,2]
        index=data['evidence_index']
        allowed=request['response_format']['schema']['properties']['claim_checks']['items']['properties']['evidence_ids']['items']['enum']
        assert all(index[i]['path'].startswith('evidence[') and '.content' in index[i]['path'] for i in allowed)
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        (root/'runs').mkdir()
        request=round4.request_for(case,unit,'semantic')
        identity=round4.digest([request,unit['unit_id'],0])
        saved={'run_id':identity,'status':'error','error':'Сохранённый отказ'}
        (root/'runs'/f'{identity}.json').write_text(json.dumps(saved))
        with patch.object(round4,'OUT',root),patch.object(round4,'api',side_effect=AssertionError('Повторный запрос')):
            assert round4.evaluate(case,unit,'semantic')==saved
    print('PASS: train-only retrieval, gold isolation, external evidence roles, message roles, immutable cache')


if __name__=='__main__':
    main()
