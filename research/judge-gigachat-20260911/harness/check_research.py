"""Независимые проверки формул, утечки gold и ссылок на исходные данные."""
from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score

from analyze_results import metrics, mode
from run_judge import build_request, reference_indices, retrieve_examples, validate

OUT=Path(__file__).parent


def main() -> None:
    rng=np.random.default_rng(20260911)
    for _ in range(1000):
        human=rng.integers(0,3,40).astype(float)
        judge=rng.integers(0,3,40).astype(float)
        judge[rng.random(40)<.2]=np.nan
        paired=np.isfinite(judge)
        result=metrics(human,human,judge,2,2)
        assert abs(result['accuracy']-accuracy_score(human[paired],judge[paired]))<1e-12
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            expected=cohen_kappa_score(human[paired],judge[paired])
        assert abs(result['cohen_kappa']-expected)<1e-12
        assert result['tp']+result['fn']+result['abstained_defects']==result['defects']
        assert sum(map(sum,result['confusion']['human_rows_judge_columns']))==result['paired_consensus']
    assert mode([0,1]) is None
    assert mode([0,1,1])==1
    checks=0
    for path in (OUT/'cases').glob('*.json'):
        case=json.loads(path.read_text())
        sample=next(u for u in case['units'] if u['partition']=='dev')
        changed=copy.deepcopy(sample)
        changed['ratings']=[{'rater_id':'SECRET_GOLD','scores':{'SECRET_LABEL':42}}]
        for arm in ['direct','grounded','direct_bm25','grounded_bm25','grounded_ids_bm25']:
            assert build_request(case,sample,arm)==build_request(case,changed,arm)
            checks+=1
        train={u['unit_id'] for u in case['units'] if u['partition']=='train'}
        assert all(e['unit_id'] in train for e in retrieve_examples(case,sample))
        rubric,evidence=reference_indices(case,sample)
        parsed={name:'not_assessable' for name in case['scores']}
        parsed['basis']=[{'criterion':next(iter(case['scores'])),'rubric_ids':[next(iter(rubric))],
                         'evidence_ids':[next(iter(evidence))],'missing_information':'Недостаточно данных','reason':'Проверка'}]
        _,audit=validate(case,sample,parsed)
        assert audit[0]['rubric_reference_valid'] and audit[0]['evidence_reference_valid']
        parsed['basis'][0]['rubric_ids']=['DOES_NOT_EXIST']
        _,audit=validate(case,sample,parsed)
        assert not audit[0]['rubric_reference_valid']
    print(f'1000 сверок accuracy/κ с sklearn; {checks} проверок отсутствия query gold; train-only retrieval; проверка существования ссылок: PASS')


if __name__=='__main__':
    main()
