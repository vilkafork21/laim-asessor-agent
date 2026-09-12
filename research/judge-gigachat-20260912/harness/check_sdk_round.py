"""Проверка ролей, train-only retrieval и входа production-цепочки без API."""
import json
from copy import deepcopy

from sdk_round import ARMS, OLD, OUT, context_for, make_judge, _serialize_llm_record


def main() -> None:
    for path in [OLD/'cases/CI10071259.json',OUT/'CI09877398.json']:
        case=json.loads(path.read_text())
        unit=next(u for u in case['units'] if u['partition']=='dev')
        original=_serialize_llm_record({'assessment_context':context_for(unit)})
        changed=deepcopy(unit)
        changed['ratings']=[]
        changed['source_rows']=['НЕ ПЕРЕДАВАТЬ']
        assert original==_serialize_llm_record({'assessment_context':context_for(changed)})
        payloads=[]
        for arm in [a for a in ARMS if a not in ['roles_schema_packet_feedback','roles_schema_packet_feedback_max','roles_schema_packet_no_examples_semantic']]:
            judge,recorder=make_judge(case,arm)
            assert not recorder.responses
            messages=judge.printing_chain.invoke(original).to_messages()
            assert [m.type for m in messages]==(['human'] if arm=='legacy_function' else ['system','human'])
            assert 'НЕ ПЕРЕДАВАТЬ' not in str(messages)
            payload=judge.retrieval_chain.invoke(original)
            if 'no_examples' in arm:
                assert not payload['examples']
            else:
                payloads.append(payload)
            assert set(judge.dataset.columns)=={'assessment_context',*case['scores']}
            known=[context_for(u) for u in case['units'] if u['partition']=='train']
            assert all(ctx in known for ctx in judge.dataset.assessment_context)
            if arm!='legacy_function':
                assert original not in messages[0].content
                assert original in messages[1].content
        assert all(payload==payloads[0] for payload in payloads)
    case=json.loads((OLD/'cases/CI10071259.json').read_text())
    unit=next(u for u in case['units'] if u['partition']=='dev')
    judge,_=make_judge(case,'roles_schema_packet_feedback')
    context=context_for(unit)
    query=_serialize_llm_record({'assessment_context':context})
    selected=judge.examples_retriever.hybrid_search(query=query)
    assert len(selected)==6
    train_contexts=[u['context'] for u in case['units'] if u['partition']=='train']
    assert all(json.loads(e['question'])['assessment_context'] in train_contexts for e in selected)
    assert all(json.loads(e['question'])['human_feedback'] for e in selected)
    context['current_turn']['output_answer']='ИЗМЕНЁННЫЙ ТЕКУЩИЙ ОТВЕТ'
    assert selected==judge.examples_retriever.hybrid_search(query=_serialize_llm_record({'assessment_context':context}))
    assert len(judge.retrieval_chain.invoke(query)['examples'])==6
    semantic,_=make_judge(case,'roles_schema_packet_no_examples_semantic')
    payload=semantic.retrieval_chain.invoke(query)
    assert not payload['examples']
    assert 'В поле claim_checks' not in payload['instructions']
    assert 'разные назначения операций' in payload['instructions']
    print('PASS: одинаковый retrieval, только train, отделённый контекст, gold не входит во вход ноды')


if __name__=='__main__':
    main()
