"""Bounded conversational display; never truncates original passages or reads gold."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--max-chars',type=int,default=8500);a=p.parse_args()
inputs=json.loads((a.private/'blind_inputs.json').read_text());done={r['unit'] for r in json.loads((a.results/'B_predictions.json').read_text())}
cache=a.private/'shown_sources.json';seen=set(json.loads(cache.read_text())) if cache.exists() else set()
indices=[i for i,r in enumerate(inputs) if r['unit'] not in done]
if not indices:raise SystemExit('B completed')
start=indices[0];end=start;used=0
for i in range(start,min(start+4,len(inputs))):
 r=inputs[i];cost=len(r['question'])+len(r['answer'])+200
 for q in r['evidence_packet']['passages']:
  fp=hashlib.sha256(q['text'].encode()).hexdigest();cost+=100+(len(q['text']) if fp not in seen else 80)
 if i>start and used+cost>a.max_chars:break
 used+=cost;end=i
 seen.update(hashlib.sha256(q['text'].encode()).hexdigest() for q in r['evidence_packet']['passages'])
print('Display planned:',start+1,'..',end+1,'estimated characters',used,flush=True)
subprocess.run([sys.executable,str(Path(__file__).with_name('show_sources.py')),'--private',str(a.private),'--start',str(start+1),'--end',str(end+1)],check=True)
