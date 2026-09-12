"""Дополнительные ответы без исходного trace: неизменный каскад против legacy SDK."""
import argparse
import asyncio
import json

from analyze_cascade import vector
from analyze_results import alpha,finite,summarize
import numpy as np
import sdk_round as sdk
from stress_cascade import run_fixture

OUT=sdk.OUT


async def run(case: dict) -> None:
    await sdk.run(case,['legacy_function'],'fresh',0)
    await run_fixture(case,'fresh')


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    case=json.loads((OUT/'CI10071259-unlinked.json').read_text())
    units=sorted([u for u in case['units'] if u['partition']=='fresh'],key=lambda u:u['unit_id'])
    assert len(units)==8
    if args.run:
        asyncio.run(run(case))
    records=[json.loads(p.read_text()) for p in (OUT/'sdk-runs').glob('*.json')]
    baseline={r['unit_id']:r for r in records if r['partition']=='fresh' and r['arm']=='legacy_function'}
    if len(baseline)!=len(units) or any(not (OUT/'cascade-runs'/f"fresh-{u['unit_id']}.json").exists() for u in units):
        print('Подготовлены восемь новых ответов; live-проверка ещё не завершена')
        return
    clusters=json.loads((OUT/'client-overlap-audit.json').read_text())['evaluation_cluster_by_case_group']
    for u in case['units']:
        u['group_id']=clusters[u['group_id']]
    results=[]
    for arm in ['legacy_function','criterion_cascade']:
        ordered=[baseline[u['unit_id']] if arm=='legacy_function' else json.loads((OUT/'cascade-runs'/f"fresh-{u['unit_id']}.json").read_text()) for u in units]
        for c in case['scores']:
            row=summarize(case,ordered,c,False)
            v=vector(case,ordered,c,arm)
            h=np.array(v['gold'],dtype=float)
            p=np.array(v['prediction'],dtype=float)
            valid=np.isfinite(h)&np.isfinite(p)
            row.update(arm=arm,alpha_ordinal=alpha(np.array([h[valid],p[valid]]),'ordinal'))
            results.append(row)
            print(arm,c,json.dumps(finite({k:row[k] for k in ['units','cohen_kappa','accuracy','coverage','recall_all_defects','score_zero_recall_all']})))
    (OUT/'cascade-fresh-validation.json').write_text(json.dumps(finite(results),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
