"""Dev-replay каскадов: первый доступный балл; выбор не использует human labels."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OLD=Path('/Users/antonzyukov/laim-artifacts/judge-deep-research-20260911')
OUT=Path(__file__).parent
sys.path.insert(0,str(OLD))
from analyze_results import alpha,finite,mode,summarize  # noqa: E402
from compare_round3 import compare  # noqa: E402


def choose(arms: list[str],values: dict[str,object]) -> tuple[float | None,str | None]:
    for arm in arms:
        value=values[arm]
        if value is not None:
            return value,arm
    return None,None


def main() -> None:
    assert choose(['a','b'],{'a':0,'b':2})==(0,'a')
    assert choose(['a','b'],{'a':None,'b':1})==(1,'b')
    case=json.loads((OLD/'cases/CI10071259.json').read_text())
    data=json.loads((OUT/'predictions.json').read_text())+json.loads((OUT/'sdk-predictions.json').read_text())
    data=[r for r in data if r.get('agent','CI10071259')=='CI10071259' and r.get('partition','dev')=='dev']
    lookup={(r['arm'],r['criterion']):r for r in data}
    assert len(lookup)==len(data), 'Повторяющиеся arm/criterion в dev'
    comments='contrastive_comments_aligned_parser'
    maximum='roles_schema_giga2_max'
    packet='roles_schema_packet_no_examples'
    candidates=[[comments,maximum,packet],[maximum,comments,packet],[comments,packet],[maximum,packet]]
    candidates.append({'factuality':[maximum,comments,packet],'completeness':[maximum,comments,packet],'structure':[comments,packet]})
    predictions=[]
    units=sorted([u for u in case['units'] if u['partition']=='dev'],key=lambda u:u['unit_id'])
    ids=[u['unit_id'] for u in units]
    result=[]
    for stages in candidates:
        records=[{'unit_id':u['unit_id'],'status':'ok','scores':{},'selected_stage':{}} for u in units]
        for c in case['scores']:
            order=stages[c] if isinstance(stages,dict) else stages
            arrays={a:lookup[a,c]['prediction'] for a in order}
            assert all(lookup[a,c]['unit_ids']==ids for a in order)
            for i,record in enumerate(records):
                score,stage=choose(order,{a:arrays[a][i] for a in order})
                record['scores'][c]=score
                record['selected_stage'][c]=stage
        for c in case['scores']:
            row=summarize(case,records,c,False)
            gold=np.array([mode([r['scores'][c] for r in u['ratings']]) for u in units],dtype=float)
            pred=np.array([r['scores'][c] for r in records],dtype=float)
            valid=np.isfinite(gold)&np.isfinite(pred)
            row.update(stages=stages,alpha_ordinal=alpha(np.array([gold[valid],pred[valid]]),'ordinal'))
            result.append(row)
            if isinstance(stages,dict):
                predictions.append({'arm':'criterion_cascade','samples':1,'method':'first_available_by_fixed_criterion_order',
                    'criterion':c,'unit_ids':ids,'groups':[u['group_id'] for u in units],'gold':gold,
                    'panel_mean':[float(np.mean([r['scores'][c] for r in u['ratings']])) for u in units],
                    'prediction':pred})
            print(stages,c,json.dumps(finite({k:row[k] for k in ['cohen_kappa','alpha_ordinal','coverage','correct_label_yield','recall_all_defects','score_zero_recall_all','fpr_paired']})))
    (OUT/'cascade-dev-predictions.json').write_text(json.dumps(finite(predictions),ensure_ascii=False,indent=2))
    if '--bootstrap' in sys.argv:
        pairs=[compare(finite(r),lookup['grounded_ids_bm25_anchors_t8',r['criterion']]) for r in predictions]
        (OUT/'cascade-dev-paired.json').write_text(json.dumps(finite(pairs),ensure_ascii=False,indent=2))
    (OUT/'cascade-dev-metrics.json').write_text(json.dumps(finite(result),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
