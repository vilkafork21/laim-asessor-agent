"""Exact, pre-answer tool binding reused from the prior audit; no provider calls."""
from __future__ import annotations
import ast,json
from typing import Any

def validated_evidence(response:Any,question:str,answer:str):
    """Return pre-answer, current-request tool results linked to actual call IDs.
    All identity checks are exact. Does not infer that the source proves all facts.
    Does not accept alternate answer versions, future tools, or tool-only final calls.
    """
    if isinstance(response,str):
        try:obj=json.loads(response)
        except json.JSONDecodeError:obj=ast.literal_eval(response)
    else:obj=response
    messages=obj.get('messages') if isinstance(obj,dict) else None
    if not isinstance(messages,list):raise ValueError('No messages')
    role=lambda m:m.get('type',m.get('role'))
    candidates=[i for i,m in enumerate(messages) if role(m) in ('ai','assistant')]
    if not candidates:raise ValueError('No final assistant')
    end=candidates[-1];final=messages[end]
    if final.get('content')!=answer or final.get('tool_calls'):raise ValueError('Final-answer mismatch')
    human=[i for i,m in enumerate(messages[:end]) if role(m) in ('human','user')]
    if not human or messages[human[-1]].get('content')!=question:raise ValueError('Current-request mismatch')
    start=human[-1];known={};evidence=[];seen=set()
    for i,m in enumerate(messages[start+1:end],start+1):
        for call in m.get('tool_calls',[]):
            if call.get('id'):known[call['id']]=call.get('name')
        if role(m)!='tool':continue
        cid=m.get('tool_call_id');name=m.get('name')
        if not cid or known.get(cid)!=name:continue
        if str(name).startswith('delegate_to_') or name=='transfer_back_to_supervisor':continue
        if cid in seen:raise ValueError('Duplicate tool result')
        seen.add(cid)
        evidence.append({'evidence_id':f'tool-{i}','tool_name':name,'tool_call_id':cid,'content':m.get('content',''),'status':m.get('status')})
    return evidence
