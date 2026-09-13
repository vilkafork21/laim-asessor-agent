"""Render unchanged per-unit packets; identical text already shown is referenced.
This optimizes a single conversational view only, not the actual API payload.
"""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--end',type=int,required=True);a=p.parse_args()
seenpath=a.private/'shown_sources.json';seen=json.loads(seenpath.read_text()) if seenpath.exists() else {}
rows=json.loads((a.private/'blind_inputs.json').read_text())
for r in rows[a.start-1:a.end]:
 print('\n===',r['unit'],'===\nQUESTION:',r['question'],'\nANSWER:\n'+r['answer'])
 ps=r['evidence_packet']['passages']
 if not ps:print('EVIDENCE: EMPTY TOOL RESULTS (no passages).')
 for s in ps:
  print('\nSOURCE',s['passage_id'],'type',s['kind'],'occurrences',len(s['provenance']))
  if s['sha256'] in seen:print('EXACT TEXT PREVIOUSLY SHOWN:',seen[s['sha256']])
  else:
   print(s['text']);seen[s['sha256']]=r['unit']+':'+s['passage_id']
seenpath.write_text(json.dumps(seen,sort_keys=True))
