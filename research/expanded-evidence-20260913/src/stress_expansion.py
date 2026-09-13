"""Deterministic metamorphic validation on real trace shapes; NOT LLM generations."""
from __future__ import annotations
import argparse,collections,copy,hashlib,json
from pathlib import Path
import numpy as np
from strict_binding import bind,decode_trace,role
from evidence_packets import packet,verify_quote,decode_content
from trace_binding import validated_evidence

def must_reject(fun):
    try:fun()
    except ValueError:return True
    return False

def run(private:Path,results:Path):
    units=json.loads((private/'all_valid.json').read_text());checks=collections.Counter();failures=collections.Counter();patterns=collections.Counter();stats=collections.Counter();lengths=[];kinds=collections.Counter();status=collections.Counter();legacy_differences=0
    def check(name,ok):
        checks[name]+=1
        if not ok:failures[name]+=1
    for u in units:
        obj=decode_trace(u['response']);q=u['q'];a=u['a']
        try:e=bind(obj,q,a)
        except ValueError as exc:
            patterns['strict_rejection:'+str(exc)]+=1;continue
        check('original_trace_accepted',True)
        old=validated_evidence(obj,q,a);legacy_differences+=int(e!=old)
        check('wrong_question_rejected',must_reject(lambda:bind(obj,q+' [mismatch]',a)))
        check('wrong_answer_rejected',must_reject(lambda:bind(obj,q,a+' [mismatch]')))
        messages=obj['messages'];end=max(i for i,m in enumerate(messages) if role(m) in ('ai','assistant'))
        start=max(i for i,m in enumerate(messages[:end]) if role(m) in ('human','user'))
        ci=next((i for i in range(start+1,end) if role(messages[i]) in ('ai','assistant') and messages[i].get('tool_calls')),None)
        ti=next((i for i in range(start+1,end) if role(messages[i])=='tool' and any(messages[i].get('tool_call_id')==r['tool_call_id'] for r in e)),None)
        if ci is not None:
            bad=copy.deepcopy(obj);bad['messages'][ci]['tool_calls'].append(copy.deepcopy(bad['messages'][ci]['tool_calls'][0]))
            check('duplicate_call_id_rejected',must_reject(lambda:bind(bad,q,a)))
            patterns['legacy_accepts_duplicate_call_id']+=int(not must_reject(lambda:validated_evidence(bad,q,a)))
            bad=copy.deepcopy(obj);bad['messages'][ci]['type']='tool';bad['messages'][ci].pop('role',None)
            check('nonassistant_call_rejected',must_reject(lambda:bind(bad,q,a)))
            patterns['legacy_accepts_nonassistant_call_declaration']+=int(not must_reject(lambda:validated_evidence(bad,q,a)))
        if ti is not None:
            cid=messages[ti]['tool_call_id']
            bad=copy.deepcopy(obj);bad['messages'][ti]['status']='error'
            check('error_tool_result_excluded',all(r['tool_call_id']!=cid for r in bind(bad,q,a)))
            patterns['legacy_retains_error_tool_result']+=int(any(r['tool_call_id']==cid for r in validated_evidence(bad,q,a)))
            bad=copy.deepcopy(obj);bad['messages'][ti]['name']='wrong_tool_name'
            check('wrong_tool_name_excluded',all(r['tool_call_id']!=cid for r in bind(bad,q,a)))
            bad=copy.deepcopy(obj);bad['messages'][ti]['tool_call_id']='unknown_tool_call'
            check('unknown_tool_id_excluded',all(r['tool_call_id']!='unknown_tool_call' for r in bind(bad,q,a)))
            bad=copy.deepcopy(obj);bad['messages'].append(copy.deepcopy(messages[ti]))
            check('future_tool_result_ignored',bind(bad,q,a)==e)
            bad=copy.deepcopy(obj);bad['messages'].insert(ti+1,copy.deepcopy(messages[ti]))
            check('duplicate_tool_result_rejected',must_reject(lambda:bind(bad,q,a)))
        pack=packet(e);twice=packet(e+e)
        check('duplication_preserves_unique_text',[(p['kind'],p['text']) for p in pack['passages']]==[(p['kind'],p['text']) for p in twice['passages']])
        check('duplication_retains_all_provenance',all(len(y['provenance'])==2*len(x['provenance']) for x,y in zip(pack['passages'],twice['passages'])))
        rev=packet(list(reversed(e)))
        check('reordering_preserves_content_set',{(p['kind'],p['sha256']) for p in rev['passages']}=={(p['kind'],p['sha256']) for p in pack['passages']})
        check('inventory_not_assumed_exhaustive',pack['exhaustive_inventory'] is False)
        for p in pack['passages']:
            check('content_hash_valid',hashlib.sha256(p['text'].encode()).hexdigest()==p['sha256'])
            if p['text'].strip():check('exact_quote_recognized',verify_quote(pack,p['passage_id'],p['text']))
            check('unknown_quote_rejected',not verify_quote(pack,p['passage_id'],'__NONEXISTENT_QUOTE__'+p['sha256']))
            check('unknown_source_rejected',not verify_quote(pack,'__UNKNOWN_SOURCE__',p['text']))
            kinds[p['kind']]+=1
            if p['kind']=='retrieved_excerpt':lengths.append(len(p['text']))
        stats['units']+=1;stats['tool_results']+=len(e);stats['passage_occurrences']+=pack['passage_occurrences'];stats['unique_passages']+=pack['unique_passages']
        stats['raw_passage_text_chars']+=sum(len(p['text'])*len(p['provenance']) for p in pack['passages']);stats['unique_passage_text_chars']+=pack['unique_text_chars']
        stats['units_empty_packet']+=int(not pack['passages']);stats['units_metadata_only']+=int(bool(pack['passages']) and all(p['kind']=='document_metadata' for p in pack['passages']))
        stats['units_with_duplicates']+=int(pack['passage_occurrences']>pack['unique_passages'])
        for r in e:status[str(r.get('status'))]+=1
    out={'scope':'deterministic adapter stress on 380 real traces; no additional model judgments, no claim of prompt-injection robustness',
         'units_input':len(units),'check_counts':dict(checks),'total_assertions':sum(checks.values()),'failures':dict(failures),
         'legacy_adapter_patterns':dict(patterns),'strict_vs_legacy_original_differences':legacy_differences,
         'packet_statistics':dict(stats),'passage_kinds':dict(kinds),'tool_statuses':dict(status),
         'retrieved_excerpt_length':{'n':len(lengths),'exactly_500_chars':sum(v==500 for v in lengths),'at_most_500_chars':sum(v<=500 for v in lengths),
                                     'quantiles':dict(zip(['min','p50','p90','max'],np.quantile(lengths,[0,.5,.9,1]).tolist())) if lengths else {}}}
    (results/'stress_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(out,ensure_ascii=False,indent=2))
    if failures:raise RuntimeError('Stress failures')
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);p.add_argument('--results',type=Path,required=True);a=p.parse_args();run(a.private,a.results)
