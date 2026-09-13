"""Write-once conversational judgments, full-pass sealing, no gold reads."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime,timezone

def canon(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def record(private:Path,results:Path,arm:str,rows:list[dict]):
    if arm not in ('A','B'):raise ValueError('Unknown arm')
    inputs=json.loads((private/'blind_inputs.json').read_text());ids=[x['unit'] for x in inputs]
    if arm=='B' and not (results/'A_seal.json').exists():raise ValueError('Seal A before B')
    if (results/f'{arm}_seal.json').exists():raise ValueError('Arm is already sealed')
    dest=results/f'{arm}_predictions.json';old=json.loads(dest.read_text()) if dest.exists() else []
    done={r['unit'] for r in old}
    for row in rows:
        if set(row)!= {'unit','score','reason'}:raise ValueError('Explicit schema required')
        if row['unit'] not in ids or row['unit'] in done:raise ValueError('Unknown/duplicate unit')
        if type(row['score']) is not int or row['score'] not in (0,1,2):raise ValueError('Score outside scale')
        if not isinstance(row['reason'],str) or not row['reason']:raise ValueError('Reason required')
        done.add(row['unit']);old.append(row)
    old.sort(key=lambda r:r['unit']);dest.write_text(canon(old))
    print('Recorded',arm,len(old),'of',len(ids))
    if len(old)==len(ids):
        seal={'arm':arm,'n':len(ids),'sealed_utc':datetime.now(timezone.utc).isoformat(),
              'predictions_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
              'input_sha256':hashlib.sha256((private/'blind_inputs.json').read_bytes()).hexdigest(),
              'gold_opened':False,'evaluator':'current_chat_assistant_not_GigaChat'}
        (results/f'{arm}_seal.json').write_text(json.dumps(seal,indent=2));print(json.dumps(seal,indent=2))

def show(private:Path,start:int,end:int,evidence=False):
    if start<1 or end<start:raise ValueError('Bad range')
    rows=json.loads((private/'blind_inputs.json').read_text())
    for r in rows[start-1:end]:
        print('\n===',r['unit'],'===\nQUESTION:',r['question'],'\nANSWER:\n'+r['answer'])
        if evidence:
            for p in r['evidence_packet']['passages']:
                print('\nSOURCE',p['passage_id'],'type',p['kind'],'occurrences',len(p['provenance']),'\n'+p['text'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);p.add_argument('--start',type=int,default=1);p.add_argument('--end',type=int,default=12);p.add_argument('--evidence',action='store_true');a=p.parse_args();show(a.private,a.start,a.end,a.evidence)
