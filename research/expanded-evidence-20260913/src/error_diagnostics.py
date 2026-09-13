"""Post-hoc error diagnosis; never rewrites predictions or defines a new winner."""
import argparse,collections,itertools,json
from pathlib import Path
import numpy as np
import pandas as pd
from parquet_local import read_columns
from prepare_expansion import digest,mode_unambiguous
from analyze_expansion import fast_metrics,PRIMARY

def run(source:Path,private:Path,results:Path):
    df=pd.DataFrame(read_columns(source));labels={};columns={'F':'Фактологическая точность ответа','C':'Полнота предоставленной информации','S':'Структурированный формат ответа'}
    for _,g in df.groupby(['case_id','doc_request_id','question_id'],sort=False,dropna=False):
        if len(g[['question','answer']].drop_duplicates())!=1:continue
        fp=digest([str(g.question.iloc[0]),str(g.answer.iloc[0])]);row={k:mode_unambiguous(g[col]) for k,col in columns.items()}
        if fp not in labels:labels[fp]=row
        else:
            for key in row:
                if row[key]!=labels[fp][key]:labels[fp][key]=None
    keys={r['unit']:r['fp'] for r in json.loads((private/'gold.json').read_text())}
    rows=json.loads((results/'evaluation_vectors.json').read_text());changed=[]
    for r in rows:
        if r['A']!=r['B']:
            changed.append(dict(r,other_human_consensus=labels[keys[r['unit']]],
                                effect='improved_exact' if r['B']==r['human'] else 'lost_exact' if r['A']==r['human'] else 'both_wrong'))
    states=sorted({(r['A'],r['B']) for r in rows});state_counts=[];y=np.array([r['human'] for r in rows]);maxima={k:-np.inf for k in PRIMARY};maxmaps={}
    for st in states:
        cc=collections.Counter(r['human'] for r in rows if (r['A'],r['B'])==st)
        state_counts.append({'A':st[0],'B':st[1],'human_counts':[cc[i] for i in range(3)]})
    for mapping in itertools.product(range(3),repeat=len(states)):
        lookup=dict(zip(states,mapping));p=[lookup[r['A'],r['B']] for r in rows];m=fast_metrics(y,p)
        for k,v in m.items():
            if v is not None and v>maxima[k]:maxima[k]=v;maxmaps[k]=list(mapping)
    out={'scope':'post-hoc descriptive error analysis; source rubric/gold unchanged',
         'changed_units':changed,'joint_judge_states':state_counts,
         'empirical_oracle':{'scope':'all mappings of already observed A/B categories using evaluation gold; NOT a deployable method or unbiased metric; separate maxima are not a single model',
         'states':len(states),'mappings_evaluated':3**len(states),'maxima':maxima,'argmax_mappings':maxmaps}}
    (results/'error_diagnostics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False));print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--private',type=Path,required=True);p.add_argument('--results',type=Path,required=True);a=p.parse_args();run(a.source,a.private,a.results)
