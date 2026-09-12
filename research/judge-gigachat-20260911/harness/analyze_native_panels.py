"""Панели исходных Excel: пропуски и компоненты без изменения замороженного gold."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from analyze_results import alpha, finite

OUT=Path(__file__).parent
ROOT=Path('/Users/antonzyukov/laim/b2c_agents_artifacts')


def rows(ci: str) -> list:
    book=load_workbook(next((ROOT/ci).glob('*.xlsx')),read_only=True,data_only=True)
    result=list(book.active.values)
    book.close()
    return result


def panel(values: list) -> dict:
    matrix=np.array(values,dtype=float).T
    paired=np.isfinite(matrix).sum(axis=0)>=2
    return {'objects':matrix.shape[1],'paired_objects':int(paired.sum()),
            'annotations':int(np.isfinite(matrix).sum()),
            'rater_annotations':np.isfinite(matrix).sum(axis=1),
            'discordant_paired_objects':sum(len(set(column[np.isfinite(column)]))>1 for column in matrix[:,paired].T),
            'alpha_nominal':alpha(matrix,'nominal'),
            'alpha_first_two_columns':alpha(matrix[:2],'nominal')}


def main() -> None:
    result={}
    data=rows('CI09840670')
    groups=defaultdict(list)
    group=None
    for row in data[2:]:
        if row[0]:
            group=row[0]
        if row[1] and isinstance(row[3],str):
            groups[group].append(row)
    turns=[r for group in groups.values() for r in group]
    assert all(sum(r[11] in (0,1) for r in entries)==1 for entries in groups.values())
    components={name:panel([[next((r[j] for r in entries if r[j] in (0,1)),np.nan) for j in cols] for entries in groups.values()])
                for name,cols in {'completeness':[11,12,13],'clarity':[14,15,16],'accuracy':[17,18,19]}.items()}
    final_votes=[]
    mismatch=[]
    consensus_mismatch=[]
    for group,entries in groups.items():
        votes=[]
        for marker in range(3):
            values=[r[c+marker] for r in entries for c in [11,14,17] if r[c+marker] in (0,1)]
            votes.append(min(values) if values else np.nan)
        final_votes.append(votes)
        minimum=min(v for v in votes if np.isfinite(v))
        gold={r[20] for r in entries if r[20] in (0,1)}
        consensus=min(int(sum(r[c+j] for j in range(3) if r[c+j] in (0,1)) / sum(r[c+j] in (0,1) for j in range(3)) > .5)
                      for r in entries for c in [11,14,17] if r[c] in (0,1))
        if gold!={consensus}:
            consensus_mismatch.append({'observed_gold':list(gold),'derived_consensus':consensus})
        if gold!={minimum}:
            mismatch.append({'observed_gold':list(gold),'derived_minimum':minimum})
    result['CI09840670']={'turns':len(turns),'dialogues':len(groups),'components_by_dialogue':components,
        'derived_dialogue_min_by_marker':panel(final_votes),'final_vs_observed_component_min_mismatches':mismatch,
        'final_vs_component_majority_min_mismatches':consensus_mismatch,
        'caveat':'Имена marker обозначают колонки. Критерии заполнены на одной строке каждого диалога; marker3 наблюдается только на 3 диалогах, поэтому не считать его полной независимой панелью. Итог совпадает с MIN большинства каждого критерия; это не MIN всех голосов. Отдельного столбца безопасности нет.'}
    data=rows('CI09840650')[1:198]
    result['CI09840650']={'units':len(data),'count_total':sum(r[6] for r in data),
        'largest_count':max(r[6] for r in data),'count_by_gold':{str(g):sum(r[6] for r in data if r[24]==g) for g in (0,1)},
        'rag_factuality':panel([[r[j] if r[j] in (0,1) else np.nan for j in [16,17,18]] for r in data]),
        'rag_completeness':panel([[r[j] if r[j] in (0,1) else np.nan for j in [20,21,22]] for r in data]),
        'route_expected_labels':panel([[{'deposelector':0,'depoaftersale':1,'rag':2}.get(r[j],np.nan) for j in [8,10,12]] for r in data]),
        'caveat':'Ожидаемый маршрут с допустимыми альтернативами не сводится к совпадению одной категории; alpha основных категорий только диагностическая. Пустые RAG-компоненты могут означать неприменимость.'}
    (OUT/'native-panels.json').write_text(json.dumps(finite(result),ensure_ascii=False,indent=2))
    print(json.dumps(finite(result),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
