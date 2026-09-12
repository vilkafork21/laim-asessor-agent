"""Выбирает реальные наблюдения из traces; метаданные доступа и human gold не передаются."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from run_judge import digest, dumps

OUT=Path(__file__).parent
TRACES=Path('/Users/antonzyukov/laim-data-20260911/traces')


def unit(context: dict, group: str, provenance: dict) -> dict:
    return {'unit_id':digest([context,provenance]),'group_id':digest(group),'context':context,
            'evidence':[],'ratings':[],'partition':'trace','provenance':provenance}


def main() -> None:
    cases={p.stem:json.loads(p.read_text()) for p in (OUT/'cases').glob('*.json')}
    inventory={}
    for ci,case in cases.items():
        units=[]
        if ci=='CI09997438':
            source=TRACES/ci/'raw'/f'{ci}__RAW__70946_spans__3776_traces.parquet'
            frame=pd.read_parquet(source,columns=['span_name','span_id','trace_id','input_text','output_text','start_time_ns','end_time_ns']).reset_index(drop=True)
            for position,row in frame[frame.span_name=='_parse_class'].iterrows():
                try:
                    context=json.loads(row.input_text)
                    prediction=str(row.output_text).strip()
                    if prediction.startswith(chr(34)):
                        prediction=json.loads(prediction)
                except (TypeError,ValueError):
                    continue
                categories={'issuance','available_credits','forced_cp_scenario','forced_credits_scenario',
                            'decline','liabilities','applications','refinance','education_credits','pledge',
                            'report','arrest','credit_card_faq','unknown'}
                if not isinstance(context,dict) or prediction not in categories or not context.get('question'):
                    continue
                messages=context.get('history',{}).get('messages',[])
                history=[{'role':'user' if m['type']=='human' else 'assistant','content':m['content']}
                         for m in messages if m.get('type') in ['human','ai'] and isinstance(m.get('content'),str)]
                units.append(unit({'mode':'turn_with_history','history':history,
                    'current_turn':{'input_query':context['question'],'output_answer':''},'observed_prediction':prediction},str(row.trace_id),
                    {'source_file':str(source),'source_row':int(position),'span_id':row.span_id,
                     'query_source':'input_text.question','history_source':'input_text.history.messages',
                     'prediction_source':'output_text of _parse_class','binding':'input_and_output_of_same_span'}))
        elif ci=='CI09840650':
            source=TRACES/ci/'raw'/'part-00000-9bc5aaf8-2061-4808-8886-0033eca6df48-c000.snappy.parquet'
            frame=pd.read_parquet(source,columns=['span_id','trace_id','input_text','output_text']).reset_index(drop=True)
            for position,row in frame.iterrows():
                try:
                    incoming=json.loads(row.input_text).get('message',{})
                    outgoing=json.loads(row.output_text).get('outgoing',{})
                except (TypeError,ValueError,AttributeError):
                    continue
                prediction=outgoing.get('receiver')
                if prediction not in ['deposelector','depoaftersale']:
                    continue
                text='\n'.join(m['value'] for m in incoming.get('content',{}).get('message',[])
                               if m.get('type')=='text' and isinstance(m.get('value'),str))
                if not text:
                    continue
                units.append(unit({'mode':'qa','current_turn':{'input_query':text,'output_answer':''},
                    'observed_prediction':prediction},str(row.trace_id),
                    {'source_file':str(source),'source_row':int(position),'span_id':row.span_id,
                     'query_source':'input_text.message.content.message[type=text].value',
                     'prediction_source':'output_text.outgoing.receiver',
                     'binding':'input_and_output_of_same_invocation','scope':'routing_to_product_agent_no_terminal_user_answer'}))
        else:
            source=TRACES/ci/'processed'/f'{ci}__turns.parquet'
            frame=pd.read_parquet(source).reset_index(drop=True)
            if ci in ['CI09997554','CI09840670']:
                for session,rows in frame.groupby('session_id',sort=False):
                    rows=rows.sort_values('entry_time_ns',kind='stable')
                    turns=[{'input_query':r.input_query,'output_answer':r.agent_response} for _,r in rows.iterrows()]
                    units.append(unit({'mode':'dialogue','turns':turns,'trace_coverage':'Только извлечённые реплики; полнота исходной сессии не подтверждена'},str(session),
                        {'source_file':str(source),'source_rows':list(map(int,rows.index)),
                         'entry_spans':rows.entry_span_id.tolist(),'binding':'converter_turn_provenance',
                         'raw_evidence_included':False}))
            else:
                for position,row in frame.iterrows():
                    units.append(unit({'mode':'qa','current_turn':{'input_query':row.input_query,'output_answer':row.agent_response}},str(row.session_id),
                        {'source_file':str(source),'source_row':int(position),'entry_span_id':row.entry_span_id,
                         'exit_span_id':row.exit_span_id,'entry_trace_id':row.entry_trace_id,
                         'binding':'converter_turn_provenance','raw_evidence_included':False}))
        unique={digest(u['context']):u for u in units}
        ordered=sorted(unique.values(),key=lambda u:u['unit_id'])
        # Выбор не использует ответы judge: разные наблюдаемые категории, затем hash-порядок.
        selected=[]
        seen=set()
        for u in ordered:
            label=u['context'].get('observed_prediction')
            if label is not None and label not in seen:
                selected.append(u)
                seen.add(label)
                if len(selected)==4:
                    break
        selected += [u for u in ordered if u not in selected][:4-len(selected)]
        if not selected:
            raise ValueError(f'{ci}: нет подтверждённых объектов trace')
        case['units']=selected
        case['gold_kind']='unlabelled_trace_no_accuracy_claim'
        case['source_hashes']={str(source):hashlib.sha256(source.read_bytes()).hexdigest()}
        destination=OUT/'trace-cases'/f'{ci}.json'
        destination.parent.mkdir(exist_ok=True)
        destination.write_text(dumps(case))
        inventory[ci]={'extracted_units':len(units),'unique_contexts':len(unique),'selected':len(selected),
                       'observed_prediction_counts':Counter(u['context'].get('observed_prediction','none') for u in units),
                       'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}
    (OUT/'trace-inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2))
    print(json.dumps(inventory,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
