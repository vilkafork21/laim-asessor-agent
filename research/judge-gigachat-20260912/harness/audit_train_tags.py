"""Сопоставление тэгов и баллов train с проверкой исходных строк parquet."""
from __future__ import annotations

import json
from collections import Counter

import pandas as pd

from packet_round import OLD,OUT,SOURCE


def main() -> None:
    case=json.loads((OLD/'cases/CI10071259.json').read_text())
    feedback=json.loads((OLD/'train-feedback-CI10071259.json').read_text())['feedback']
    train={u['unit_id']:u for u in case['units'] if u['partition']=='train'}
    assert set(feedback)==set(train)
    frame=pd.read_parquet(SOURCE).reset_index(drop=True)
    columns={'factuality':'Фактологическая точность ответа','completeness':'Полнота предоставленной информации','structure':'Структурированный формат ответа'}
    rows=[]
    counts={tag:Counter() for tag in ['#НетЛогики','#Повтор','#ТехОшибка']}
    verified=0
    missing_rating_rows=0
    for uid,items in feedback.items():
        source=frame.iloc[train[uid]['source_rows']]
        for i,item in enumerate(items):
            matches=[]
            for index,row in source.iterrows():
                if any(not (pd.isna(row[column]) and pd.isna(item['scores'][c])) and row[column]!=item['scores'][c] for c,column in columns.items()):
                    continue
                if str(row['Комментарий']).strip()!=str(item.get('comment','')).strip():
                    continue
                if str(row['ТЭГ']).strip()!=str(item.get('tags','')).strip():
                    continue
                matches.append(int(index))
            assert matches,(uid,i,'feedback не совпал с исходной строкой')
            verified+=1
            if any(pd.isna(v) for v in item['scores'].values()):
                missing_rating_rows+=1
                continue
    source_annotations=0
    for uid,unit in train.items():
        for index,item in frame.iloc[unit['source_rows']].iterrows():
            if any(pd.isna(item[column]) for column in columns.values()):
                continue
            source_annotations+=1
            for tag,c,max_score in [('#НетЛогики','structure',0),('#Повтор','structure',1),('#ТехОшибка','factuality',0)]:
                if tag not in str(item['ТЭГ']):
                    continue
                score=float(item[columns[c]])
                counts[tag][str(score)]+=1
                if score>max_score:
                    rows.append({'unit_id':uid,'tag':tag,'criterion':c,'score':score,
                        'rubric_max_score':max_score,'verified_source_row':int(index),
                        'comment':str(item['Комментарий']),'tags':str(item['ТЭГ'])})
    assert source_annotations==sum(len(u['ratings']) for u in train.values())
    result={'partition':'train','feedback_rows_verified':verified,'feedback_rows_with_missing_scores':missing_rating_rows,'source_annotations':source_annotations,
        'counts_by_tag_and_score':{k:dict(v) for k,v in counts.items()},'potential_conflicts':rows,
        'note':'Метки не меняются. Тэги общие для строки; нужны согласование рубрики и арбитраж конкретных случаев.'}
    (OUT/'train-tag-score-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('PASS: исходных feedback-строк сверено',verified,'; потенциальных конфликтов',len(rows),'в',len({r['unit_id'] for r in rows}),'train-единицах')
    print(result['counts_by_tag_and_score'])


if __name__=='__main__':
    main()
