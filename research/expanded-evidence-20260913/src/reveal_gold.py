"""Open gold only after verifying both full prediction seals and original data hashes."""
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime,timezone

def reveal(private:Path,results:Path):
    manifest=json.loads((results/'selection_manifest.json').read_text())
    for file,key in [('blind_inputs.json','blind_inputs_sha256'),('gold.json','gold_sha256')]:
        if hashlib.sha256((private/file).read_bytes()).hexdigest()!=manifest[key]:raise ValueError('Input changed')
    predictions={}
    for arm in ['A','B']:
        p=results/f'{arm}_predictions.json';seal=json.loads((results/f'{arm}_seal.json').read_text())
        if hashlib.sha256(p.read_bytes()).hexdigest()!=seal['predictions_sha256']:raise ValueError('Prediction changed')
        if seal['input_sha256']!=manifest['blind_inputs_sha256'] or seal['n']!=manifest['n']:raise ValueError('Invalid seal')
        v=json.loads(p.read_text());predictions[arm]={r['unit']:r['score'] for r in v}
        if len(v)!=manifest['n'] or len(predictions[arm])!=len(v):raise ValueError('Incomplete predictions')
    gold=json.loads((private/'gold.json').read_text());clusters={g:i for i,g in enumerate(sorted({r['cluster'] for r in gold}))}
    ids={r['unit'] for r in gold}
    if any(set(v)!=ids for v in predictions.values()) or len(ids)!=len(gold):raise ValueError('Unit mismatch')
    # Arbitrary cluster numbers only; no private IDs, content hashes or customer text.
    rows=[{'unit':r['unit'],'cluster':clusters[r['cluster']],'human':int(r['gold']),
           'A':predictions['A'][r['unit']],'B':predictions['B'][r['unit']]} for r in gold]
    (results/'evaluation_vectors.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    (results/'gold_reveal.json').write_text(json.dumps({'revealed_utc':datetime.now(timezone.utc).isoformat(),
        'both_full_seals_verified':True,'n':len(rows),'note':'Analyst may now inspect labels; no score edits allowed.'},indent=2))
    return rows
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);p.add_argument('--results',type=Path,required=True);a=p.parse_args();reveal(a.private,a.results)
