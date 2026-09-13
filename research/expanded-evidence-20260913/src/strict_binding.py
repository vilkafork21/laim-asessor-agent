"""Candidate v2 trace integrity adapter, separate from frozen experimental inputs.

Only assistant-declared calls can authorize tool results. Error results are not
factual evidence. This checks trace structure, NOT semantic entailment/security.
"""
from __future__ import annotations
import ast,json
from typing import Any

def decode_trace(value:Any):
    if isinstance(value,str):
        if len(value)>10_000_000:raise ValueError('Trace exceeds safety budget')
        try:value=json.loads(value)
        except json.JSONDecodeError:
            try:value=ast.literal_eval(value)
            except (ValueError,SyntaxError,MemoryError,RecursionError) as exc:raise ValueError('Invalid trace') from exc
    if not isinstance(value,dict) or not isinstance(value.get('messages'),list):raise ValueError('No messages')
    if not all(isinstance(m,dict) for m in value['messages']):raise ValueError('Invalid message')
    return value

def role(m):return m.get('type',m.get('role'))

def bind(response:Any,question:str,answer:str):
    obj=decode_trace(response);messages=obj['messages']
    candidates=[i for i,m in enumerate(messages) if role(m) in ('ai','assistant')]
    if not candidates:raise ValueError('No final assistant')
    end=candidates[-1];final=messages[end]
    if final.get('content')!=answer or final.get('tool_calls') or final.get('function_call'):raise ValueError('Final answer mismatch')
    users=[i for i,m in enumerate(messages[:end]) if role(m) in ('human','user')]
    if not users or messages[users[-1]].get('content')!=question:raise ValueError('Question mismatch')
    start=users[-1];known={};seen=set();result=[]
    for i,m in enumerate(messages[start+1:end],start+1):
        calls=m.get('tool_calls') or []
        if calls and role(m) not in ('assistant','ai'):raise ValueError('Call declared by non-assistant')
        if not isinstance(calls,list):raise ValueError('Invalid call list')
        for call in calls:
            if not isinstance(call,dict):raise ValueError('Invalid call')
            cid=call.get('id');nested=call.get('function') or {}
            if not isinstance(nested,dict):raise ValueError('Invalid nested function')
            name=call.get('name') or nested.get('name')
            if not isinstance(cid,str) or not cid or not isinstance(name,str) or not name:raise ValueError('Missing call identity')
            if cid in known:raise ValueError('Ambiguous duplicate call ID')
            known[cid]=name
        if role(m)!='tool':continue
        cid=m.get('tool_call_id');name=m.get('name')
        if cid not in known or known[cid]!=name:continue
        if cid in seen:raise ValueError('Duplicate tool result')
        seen.add(cid)
        if m.get('status') not in (None,'success','ok'):continue
        if name.startswith('delegate_to_') or name=='transfer_back_to_supervisor':continue
        result.append({'evidence_id':f'tool-{i}','tool_name':name,'tool_call_id':cid,
                       'content':m.get('content',''),'status':m.get('status')})
    return result
