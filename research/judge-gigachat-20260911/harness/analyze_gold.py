"""Проверяет человеческую панель, баланс классов и эффективную поддержку групп."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_results import alpha, finite, mode

OUT=Path(__file__).parent


def describe(units: list[dict], criterion: str) -> dict:
    raters=sorted({r['rater_id'] for u in units for r in u['ratings']})
    matrix=np.full((len(raters),len(units)),np.nan)
    for i,u in enumerate(units):
        for r in u['ratings']:
            matrix[raters.index(r['rater_id']),i]=r['scores'][criterion]
    means=np.nanmean(matrix,axis=0)
    modes=[mode([r['scores'][criterion] for r in u['ratings']]) for u in units]
    groups=Counter(u['group_id'] for u in units)
    return {'units':len(units),'groups':len(groups),'raters':len(raters),
        'annotations':int(np.isfinite(matrix).sum()),
        'human_alpha_units_with_two_or_more_raters':sum(len(u['ratings'])>=2 for u in units),
        'units_by_panel_size':Counter(len(u['ratings']) for u in units),
        'mean_equal_units':float(means.mean()),'mean_equal_annotations':float(np.nanmean(matrix)),
        'mean_equal_groups':float(np.mean([np.mean([means[i] for i,u in enumerate(units) if u['group_id']==g]) for g in groups])),
        'modal_label_counts':Counter(str(v) for v in modes),
        'unanimous_units':sum(len({r['scores'][criterion] for r in u['ratings']})==1 for u in units),
        'max_group_units':max(groups.values()),
        'effective_groups_by_size':float(len(units)**2/sum(n*n for n in groups.values())),
        'human_alpha':{level:alpha(matrix,level) for level in ['nominal','ordinal','interval']} if len(raters)>1 else None}


def main() -> None:
    results={}
    for path in sorted((OUT/'cases').glob('*.json')):
        case=json.loads(path.read_text())
        partitions={part:[u for u in case['units'] if u['partition']==part] for part in ['train','dev','test']}
        partitions['active_all']=[u for u in case['units'] if u['partition']!='excluded_duplicate']
        results[case['agent']]={part:{name:describe(units,name) for name in case['scores']} for part,units in partitions.items()}
    source=Path('/Users/antonzyukov/laim-data-20260911/artifacts/CI10071259/baskets/agent_post_labeled_all_2025_11_27_v1.parquet')
    frame=pd.read_parquet(source).reset_index(drop=True)
    columns={'factuality':'Фактологическая точность ответа','completeness':'Полнота предоставленной информации','structure':'Структурированный формат ответа'}
    units=[]
    for key,rows in frame.groupby(['case_id','doc_request_id','question_id','question','answer'],dropna=False):
        ratings=[{'rater_id':str(r['ID Эксперта']),'scores':{k:float(r[c]) for k,c in columns.items()}}
                 for _,r in rows.iterrows() if r[list(columns.values())].notna().all()]
        if ratings:
            units.append({'group_id':str(key[0]),'ratings':ratings})
    results['CI10071259_original_all_targets']={name:describe(units,name) for name in columns}
    # Известный контрпример: высокая accuracy постоянной моды при нулевой κ.
    from analyze_results import metrics
    h=np.array([1.]*95+[0.]*5)
    j=np.ones(100)
    results['constant_mode_demo']={**metrics(h,h,j,1,1),'alpha':alpha(np.array([h,j]),'nominal')}
    assert abs(results['constant_mode_demo']['cohen_kappa'])<1e-12
    assert results['constant_mode_demo']['accuracy']==.95
    (OUT/'gold-diagnostics.json').write_text(json.dumps(finite(results),ensure_ascii=False,indent=2))
    for part in ['active_all','test']:
        print(part,json.dumps(finite(results['CI10071259'][part]),ensure_ascii=False))
    print('original',json.dumps(finite(results['CI10071259_original_all_targets']),ensure_ascii=False))


if __name__=='__main__':
    main()
