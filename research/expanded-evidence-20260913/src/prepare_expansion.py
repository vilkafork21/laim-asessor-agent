"""Prepare a score-blind, deterministic paired study. Never print private records.

The parquet input is read with a narrow read-only fallback when PyArrow is absent.
Human labels are stored separately and not included in judge packets. This is an
exploratory previously studied corpus, not a newly sampled production test set.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from parquet_local import read_columns
from trace_binding import validated_evidence
from evidence_packets import packet

COL='Структурированный формат ответа'
def canonical(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x): return hashlib.sha256(canonical(x).encode()).hexdigest()
def mode_unambiguous(xs):
    c=collections.Counter(float(x) for x in xs if pd.notna(x))
    top=c.most_common()
    return top[0][0] if top and (len(top)==1 or top[0][1]>top[1][1]) else None

class UnionFind:
    def __init__(self):self.parent={}
    def find(self,x):
        self.parent.setdefault(x,x)
        if self.parent[x]!=x:self.parent[x]=self.find(self.parent[x])
        return self.parent[x]
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)

def prepare(source:Path,private:Path,public:Path,n:int=96):
    private.mkdir(parents=True,exist_ok=True,mode=0o700);public.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(read_columns(source)); counts=collections.Counter(); groups=UnionFind();tmp=[]
    for key,g in df.groupby(['case_id','doc_request_id','question_id'],dropna=False,sort=False):
        counts['source_unit_keys']+=1
        case='case:'+str(key[0]);groups.find(case)
        for v in g['inn'].dropna().unique():groups.union(case,'client:'+str(v))
        if len(g[['question','answer']].drop_duplicates())!=1:
            counts['rater_text_identity_conflicts']+=1;continue
        q,a=str(g.question.iloc[0]),str(g.answer.iloc[0]);fp=digest([q,a]);groups.union(case,'qa:'+fp)
        rs=g.response.dropna().unique(); evidence=None
        if len(rs)==1:
            try:evidence=validated_evidence(rs[0],q,a)
            except (ValueError,TypeError,KeyError,SyntaxError):counts['invalid_trace']+=1
        elif len(rs)>1:counts['ambiguous_trace']+=1
        else:counts['no_trace']+=1
        tmp.append({'q':q,'a':a,'fp':fp,'case':case,'gold':mode_unambiguous(g[COL]),
                    'ratings':[float(v) for v in g[COL] if pd.notna(v)],
                    'evidence':evidence,'response':rs[0] if len(rs)==1 else None})
    byfp=collections.defaultdict(list)
    for u in tmp:byfp[u['fp']].append(u)
    allvalid=[];eligible=[]
    for fp,same in byfp.items():
        counts['exact_qa_fingerprints']+=1
        labels={v['gold'] for v in same}
        if len(labels)!=1 or next(iter(labels)) is None:
            counts['ambiguous_consensus_or_duplicate_conflict']+=1;continue
        u=next((v for v in same if v['evidence']),same[0]);u=dict(u)
        u['cluster']=digest(groups.find(u['case']));u['packet']=packet(u['evidence']) if u['evidence'] else None
        if not u['evidence']:counts['no_valid_evidence']+=1;continue
        allvalid.append(u)
        if len(u['q'])+len(u['a'])>5000:counts['qa_over_budget']+=1;continue
        if u['packet']['unique_text_chars']>12000:counts['evidence_over_budget']+=1;continue
        eligible.append(u)
    buckets=collections.defaultdict(list)
    for u in sorted(eligible,key=lambda v:v['fp']):buckets[u['cluster']].append(u)
    chosen=[];keys=sorted(buckets)
    while len(chosen)<min(n,len(eligible)):
        for k in keys:
            if buckets[k] and len(chosen)<n:chosen.append(buckets[k].pop(0))
        if not any(buckets.values()):break
    blinded=[];gold=[]
    for i,u in enumerate(chosen,1):
        uid=f'X{i:03d}'
        blinded.append({'unit':uid,'question':u['q'],'answer':u['a'],'evidence_packet':u['packet']})
        gold.append({'unit':uid,'gold':u['gold'],'ratings':u['ratings'],'cluster':u['cluster'],'fp':u['fp']})
    (private/'blind_inputs.json').write_text(canonical(blinded));(private/'gold.json').write_text(canonical(gold))
    (private/'all_valid.json').write_text(canonical(allvalid))
    sizes=[len(u['q'])+len(u['a'])+u['packet']['unique_text_chars'] for u in chosen]
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'n':len(chosen),'target_n':n,
              'selection_counts':dict(counts),'all_valid_evidence_units':len(allvalid),'eligible_after_budgets':len(eligible),
              'eligible_clusters':len({u['cluster'] for u in eligible}),'selected_clusters':len({u['cluster'] for u in chosen}),
              'raw_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'blind_inputs_sha256':hashlib.sha256((private/'blind_inputs.json').read_bytes()).hexdigest(),
              'gold_sha256':hashlib.sha256((private/'gold.json').read_bytes()).hexdigest(),
              'input_characters':sum(sizes),'max_input_characters':max(sizes,default=0),
              'evaluator':'current_chat_assistant','new_provider_generations':0,'fresh_test':False,
              'prior_pilots_excluded':False,'gold_values_shown_to_evaluator':False}
    (public/'selection_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print(json.dumps(manifest,indent=2,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--private',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--n',type=int,default=96)
    a=p.parse_args();prepare(a.source,a.private,a.results,a.n)
