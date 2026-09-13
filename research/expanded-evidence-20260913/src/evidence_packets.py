"""Lossless-at-text-level evidence packets; provenance retained, no claims of truth.
Exact duplicate texts share one entry. Originals are never rewritten.
Do not infer that a search result set is an exhaustive document inventory.
"""
from __future__ import annotations
import ast,hashlib,json
from collections import OrderedDict
from typing import Any

def decode_content(raw: Any) -> Any:
    if not isinstance(raw,str):return raw
    if len(raw)>2_000_000:raise ValueError('Evidence exceeds safe parsing size')
    try:return json.loads(raw)
    except (json.JSONDecodeError,TypeError):
        try:return ast.literal_eval(raw)
        except (ValueError,SyntaxError,MemoryError,RecursionError):return raw

def packet(records:list[dict]) -> dict:
    unique=OrderedDict();raw_chars=0;occurrences=0
    for record in records:
        raw=record.get('content','');obj=decode_content(raw)
        raw_chars+=len(raw) if isinstance(raw,str) else len(json.dumps(raw,ensure_ascii=False))
        if isinstance(obj,list) and all(isinstance(v,dict) and 'content' in v for v in obj):
            blocks=[(x.get('content'),{k:x[k] for k in ['file','page','doc_type','counterparty','score'] if k in x},'retrieved_excerpt') for x in obj]
        elif isinstance(obj,dict):
            blocks=[(json.dumps(obj,ensure_ascii=False,sort_keys=True),{},'document_metadata')]
        else:blocks=[(str(obj),{},'untyped_tool_result')]
        for text,meta,kind in blocks:
            if not isinstance(text,str):text=json.dumps(text,ensure_ascii=False)
            occurrences+=1;fingerprint=hashlib.sha256(text.encode()).hexdigest()
            key=(kind,fingerprint)
            if key not in unique:
                unique[key]={'passage_id':f'P{len(unique)+1:03d}','kind':kind,'text':text,'sha256':fingerprint,'provenance':[]}
            unique[key]['provenance'].append({'tool_evidence_id':record.get('evidence_id'),'tool_name':record.get('tool_name'),'tool_call_id':record.get('tool_call_id'),'status':record.get('status'),**meta})
    return {'passages':list(unique.values()),'input_serialized_chars':raw_chars,'passage_occurrences':occurrences,'unique_passages':len(unique),'unique_text_chars':sum(len(x['text']) for x in unique.values()),'exhaustive_inventory':False,'coverage_rule':'No negative/existence conclusion from retrieval omission alone; metadata is not source text.'}

def verify_quote(pack:dict,passage_id:str,quote:str)->bool:
    if not isinstance(quote,str) or not quote.strip():return False
    passages={p['passage_id']:p for p in pack['passages']}
    return passage_id in passages and quote in passages[passage_id]['text']
