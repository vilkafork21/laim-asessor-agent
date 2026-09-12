"""Проверка фактических test-запросов: исходный контекст и только train-примеры."""
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).parent
OLD=Path('/Users/antonzyukov/laim-artifacts/judge-deep-research-20260911')


def main() -> None:
    case_path=OLD/'cases/CI10071259.json'
    case=json.loads(case_path.read_text())
    lookup={u['unit_id']:u for u in case['units']}
    train={u['unit_id'] for u in case['units'] if u['partition']=='train'}
    config=json.loads((OUT/'frozen-cascade.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in config['source_sha256'].items())
    checked=[]
    for path in (OUT/'runs').glob('*.json'):
        r=json.loads(path.read_text())
        if r['partition']!='test' or r['arm']!='contrastive_comments':
            continue
        payload=json.loads(r['request']['messages'][-1]['content'])
        u=lookup[r['unit_id']]
        assert u['partition']=='test'
        assert payload['assessment_context']==u['context']
        assert all(e['unit_id'] in train for e in payload['train_examples'])
        assert all(e['assessment_context']==lookup[e['unit_id']]['context'] for e in payload['train_examples'])
        assert r['unit_id'] not in {e['unit_id'] for e in payload['train_examples']}
        assert 'ratings' not in payload and 'human_feedback' not in payload
        checked.append(r['unit_id'])
    assert len(checked)==len(set(checked))
    result={'checked_test_requests':len(checked),'all_contexts_match':True,'only_train_examples':True,
            'frozen_sources_match':True,'case_sha256':hashlib.sha256(case_path.read_bytes()).hexdigest(),
            'scope':'Контроль явного payload; два исходных примера test в самой рубрике учтены отдельным аудитом и срезом.'}
    (OUT/'cascade-input-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
