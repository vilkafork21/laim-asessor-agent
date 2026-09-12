"""Новая корзина пролонгации: одна экспертная метка на целый диалог."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

OLD=Path('/Users/antonzyukov/laim-artifacts/judge-deep-research-20260911')
sys.path.insert(0,str(OLD))
from prepare_cases import canonical,digest  # noqa: E402

OUT=Path(__file__).parent
SOURCE=Path('/Users/antonzyukov/laim-data-20260911/artifacts/CI09877398/processed/ci09877398_basket_adapter_out.parquet')


def main() -> None:
    frame=pd.read_parquet(SOURCE)
    units=[]
    rejected=Counter()
    for group,rows in frame.groupby('reference_group_id',sort=True):
        if rows.main_metric.isna().any():
            rejected['missing_label_dialogue']+=1
            continue
        labels=rows.main_metric.unique()
        assert len(labels)==1
        rows=rows.assign(_order=rows.turn_index.astype(int)).sort_values('_order')
        assert not rows._order.duplicated().any()
        context={'mode':'dialogue','turns':[{'turn_index':int(r._order),'input_query':r.input_query,'output_answer':r.output_answer} for _,r in rows.iterrows()]}
        units.append({'unit_id':digest(context),'group_id':digest(group),'context':context,'evidence':[],
                      'ratings':[{'rater_id':'human_final','scores':{'assessment_score':float(labels[0])}}],
                      'source_rows':list(map(int,rows.index))})
    by_id={}
    for unit in units:
        if unit['unit_id'] in by_id:
            assert unit['ratings']==by_id[unit['unit_id']]['ratings'], 'Противоречивая метка полного диалога'
            rejected['exact_duplicate_dialogue']+=1
        else:
            by_id[unit['unit_id']]=unit
    units=sorted(by_id.values(),key=lambda u:u['unit_id'])
    parents={u['group_id']:u['group_id'] for u in units}
    def root(group: str) -> str:
        while parents[group]!=group:
            parents[group]=parents[parents[group]]
            group=parents[group]
        return group
    seen={}
    for u in units:
        for t in u['context']['turns']:
            key=digest([t['input_query'].strip(),t['output_answer'].strip()])
            if key in seen:
                parents[root(u['group_id'])]=root(seen[key])
            seen[key]=u['group_id']
    for u in units:
        u['group_id']=root(u['group_id'])
    groups=sorted({u['group_id'] for u in units})
    defects=[any(u['ratings'][0]['scores']['assessment_score']==0 for u in units if u['group_id']==g) for g in groups]
    splits=StratifiedKFold(5,shuffle=True,random_state=20260912)
    for fold,(_,indices) in enumerate(splits.split(groups,defects)):
        held={groups[i] for i in indices}
        for u in units:
            if u['group_id'] in held:
                u['partition']='test' if fold==0 else 'dev' if fold==1 else 'train'
    existing_turns=set()
    for path in (OLD/'cases').glob('*.json'):
        for u in json.loads(path.read_text())['units']:
            ctx=u['context']
            for t in ctx.get('turns',[ctx.get('current_turn',{})]):
                existing_turns.add(digest([str(t.get('input_query','')).strip(),str(t.get('output_answer','')).strip()]))
    overlap=sum(any(digest([t['input_query'].strip(),t['output_answer'].strip()]) in existing_turns for t in u['context']['turns']) for u in units)
    assert overlap==0
    rubric=(OUT/'CI09877398-rubric.txt').read_text()
    case={'agent':'CI09877398','rubric':rubric,'target':'Оцени весь диалог один раз: Итог=1 только когда классификация и релевантность равны 1, иначе 0. Последовательность реплик сохранена; не оценивай их как независимые запросы.',
          'scores':{'assessment_score':[0,1]},'units':units,'gold_kind':'documented_dialogue_final_human_score',
          'split_seed':20260912,'source_hashes':{str(SOURCE):hashlib.sha256(SOURCE.read_bytes()).hexdigest()},
          'limitations':'Персональные банковские сведения и RAG источники этой корзины отдельно не восстановлены. Нет точных пересечений с шестью прежними корзинами; перефразировки и общий клиент между сессиями не исключены.'}
    (OUT/'CI09877398.json').write_text(canonical(case))
    report={'source_rows':len(frame),'source_dialogues':frame.reference_group_id.nunique(),'retained':len(units),'connected_groups':len(groups),'rejected':dict(rejected),'old_exact_turn_overlap':overlap,
            'partitions':{p:dict(Counter(u['ratings'][0]['scores']['assessment_score'] for u in units if u['partition']==p)) for p in ['train','dev','test']}}
    (OUT/'extra-data-audit.json').write_text(canonical(report))
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
